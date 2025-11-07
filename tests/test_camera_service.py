from __future__ import annotations

import asyncio

import pytest

from app.camera import (
    CameraError,
    CameraService,
    CameraSource,
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
