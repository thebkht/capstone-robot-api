from __future__ import annotations

import asyncio

import pytest

from app import camera as camera_module
from app.camera import (
    CameraError,
    CameraService,
    CameraSource,
    DepthAICameraSource,
    PlaceholderCameraSource,
)


def test_camera_service_uses_placeholder_frame():
    placeholder = b"test-frame"
    service = CameraService(primary=None, fallback=PlaceholderCameraSource(placeholder), boundary="frame", frame_rate=0)

    frame = asyncio.run(service.get_frame())
    assert frame == placeholder

    async def _consume_stream() -> list[bytes]:
        return [chunk async for chunk in service.stream(max_frames=1)]

    chunks = asyncio.run(_consume_stream())
    assert len(chunks) == 1
    assert placeholder in chunks[0]


def test_camera_service_requires_source():
    with pytest.raises(ValueError):
        CameraService(primary=None, fallback=None)


def test_camera_service_raises_when_no_frame_available():
    service = CameraService(primary=None, fallback=PlaceholderCameraSource(b""), boundary="frame", frame_rate=0)

    with pytest.raises(CameraError):
        asyncio.run(service.get_frame())


class _FailingCameraSource(CameraSource):
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def get_jpeg_frame(self) -> bytes:
        self.calls += 1
        raise CameraError("boom")

    async def close(self) -> None:  # pragma: no cover - best effort cleanup
        self.closed = True


def test_camera_service_disables_failed_primary():
    failing = _FailingCameraSource()
    fallback = PlaceholderCameraSource(b"fallback")
    service = CameraService(primary=failing, fallback=fallback, boundary="frame", frame_rate=0)

    frame = asyncio.run(service.get_frame())
    assert frame == b"fallback"
    assert failing.calls == 1
    assert failing.closed is True

    # Subsequent frames should not try the failed primary again.
    frame = asyncio.run(service.get_frame())
    assert frame == b"fallback"
    assert failing.calls == 1


def test_depthai_camera_source_requires_depthai():
    if camera_module.dai is not None:  # pragma: no cover - depends on optional hardware
        pytest.skip("DepthAI library available; cannot test missing dependency")

    with pytest.raises(CameraError):
        DepthAICameraSource()


def test_depthai_camera_source_uses_legacy_creators(monkeypatch):
    class _FakeStream:
        def __init__(self) -> None:
            self.linked_to = None

        def link(self, target: object) -> None:
            self.linked_to = target

    class _FakeColorCamera:
        def __init__(self) -> None:
            self.preview = _FakeStream()

        def setPreviewSize(self, *_: object) -> None:  # noqa: N802 - mimic DepthAI API
            pass

        def setResolution(self, *_: object) -> None:  # noqa: N802 - mimic DepthAI API
            pass

        def setInterleaved(self, *_: object) -> None:  # noqa: N802 - mimic DepthAI API
            pass

        def setColorOrder(self, *_: object) -> None:  # noqa: N802 - mimic DepthAI API
            pass

        def setFps(self, *_: object) -> None:  # noqa: N802 - mimic DepthAI API
            pass

    class _FakeVideoEncoder:
        def __init__(self) -> None:
            self.input = _FakeStream()
            self.bitstream = _FakeStream()

        def setDefaultProfilePreset(self, *_: object) -> None:  # noqa: N802 - mimic DepthAI API
            pass

    class _FakeXLinkOut:
        def __init__(self) -> None:
            self.input = _FakeStream()
            self.stream_name = None

        def setStreamName(self, name: str) -> None:  # noqa: N802 - mimic DepthAI API
            self.stream_name = name

    class _FakePipeline:
        def __init__(self) -> None:
            self.legacy_xout_used = False

        def create(self, node_cls: type):  # pragma: no cover - defensive
            return node_cls()

        def createColorCamera(self) -> _FakeColorCamera:
            return _FakeColorCamera()

        def createVideoEncoder(self) -> _FakeVideoEncoder:
            return _FakeVideoEncoder()

        def createXLinkOut(self) -> _FakeXLinkOut:
            self.legacy_xout_used = True
            return _FakeXLinkOut()

    class _FakeQueue:
        pass

    class _FakeDevice:
        def __init__(self, pipeline: _FakePipeline) -> None:
            self.queue = _FakeQueue()
            self.pipeline = pipeline

        def getOutputQueue(self, **_: object) -> _FakeQueue:  # pragma: no cover - defensive
            return self.queue

        def getConnectedCameras(self) -> list[object]:  # pragma: no cover - defensive
            return []

        def getUsbSpeed(self):  # pragma: no cover - defensive
            return type("_Speed", (), {"name": "super"})()

        def getDeviceInfo(self):  # pragma: no cover - defensive
            return type("_Info", (), {"getMxId": lambda self: "mxid"})()

    class _FakeColorOrder:
        BGR = object()

    class _FakeSensorResolution:
        THE_1080_P = object()

    fake_node_module = type("_NodeModule", (), {})()

    fake_depthai = type(
        "_FakeDepthAI",
        (),
        {
            "Pipeline": _FakePipeline,
            "Device": _FakeDevice,
            "node": fake_node_module,
            "ColorCameraProperties": type(
                "_ColorProps",
                (),
                {
                    "ColorOrder": type("_ColorOrder", (), {"BGR": _FakeColorOrder.BGR}),
                    "SensorResolution": type(
                        "_SensorResolution",
                        (),
                        {"THE_1080_P": _FakeSensorResolution.THE_1080_P},
                    ),
                },
            ),
            "VideoEncoderProperties": type(
                "_VideoProps",
                (),
                {"Profile": type("_Profile", (), {"MJPEG": object()})},
            ),
        },
    )

    original_dai = camera_module.dai
    monkeypatch.setattr(camera_module, "dai", fake_depthai)

    try:
        source = DepthAICameraSource()
        source._start_pipeline()  # type: ignore[attr-defined]
        assert isinstance(source._queue, _FakeQueue)
        assert isinstance(source._device, _FakeDevice)
        assert source._device.queue is source._queue
        # The fake pipeline toggles legacy_xout_used when createXLinkOut is called.
        assert source._device.pipeline.legacy_xout_used is True
    finally:
        monkeypatch.setattr(camera_module, "dai", original_dai)
