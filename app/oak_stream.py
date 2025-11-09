"""MJPEG streaming helpers modelled after Luxonis' DepthAI example."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Generator, Optional

try:  # pragma: no cover - optional dependency
    import depthai as dai
except Exception:  # pragma: no cover - optional dependency
    dai = None  # type: ignore[assignment]

from fastapi import HTTPException
from fastapi.responses import StreamingResponse


BOUNDARY = "frame"
_FPS = 30
_PREVIEW_WIDTH = 640
_PREVIEW_HEIGHT = 360
_QUEUE_MAX_SIZE = 4
_STREAM_NAME = "mjpeg"

LOGGER = logging.getLogger("uvicorn.error").getChild(__name__)

_lock = threading.Lock()
_device: Optional["dai.Device"] = None
_queue: Optional["dai.DataOutputQueue"] = None


def _require_depthai() -> None:
    if dai is None:  # pragma: no cover - environment dependent
        raise HTTPException(status_code=503, detail="DepthAI streaming dependencies are unavailable")


def _create_color_camera(pipeline: "dai.Pipeline") -> "dai.node.ColorCamera":
    if hasattr(dai, "node") and hasattr(dai.node, "ColorCamera"):
        camera = pipeline.create(dai.node.ColorCamera)
    else:  # pragma: no cover - legacy SDK fallback
        camera = pipeline.createColorCamera()

    camera.setBoardSocket(dai.CameraBoardSocket.RGB)
    camera.setPreviewSize(_PREVIEW_WIDTH, _PREVIEW_HEIGHT)
    if hasattr(camera, "setVideoSize"):
        camera.setVideoSize(1920, 1080)
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camera.setInterleaved(False)
    camera.setFps(_FPS)
    return camera


def _create_video_encoder(pipeline: "dai.Pipeline") -> "dai.node.VideoEncoder":
    if hasattr(dai, "node") and hasattr(dai.node, "VideoEncoder"):
        encoder = pipeline.create(dai.node.VideoEncoder)
    else:  # pragma: no cover - legacy SDK fallback
        encoder = pipeline.createVideoEncoder()

    encoder.setDefaultProfilePreset(_FPS, dai.VideoEncoderProperties.Profile.MJPEG)
    encoder.setKeyframeFrequency(_FPS)
    if hasattr(encoder, "setBitrate"):
        encoder.setBitrate(8_000_000)
    rate_control = getattr(dai.VideoEncoderProperties.RateControlMode, "CBR", None)
    if rate_control is not None:
        encoder.setRateControlMode(rate_control)
    return encoder


def _create_xlink_out(pipeline: "dai.Pipeline") -> "dai.node.XLinkOut":
    if hasattr(dai, "node") and hasattr(dai.node, "XLinkOut"):
        xout = pipeline.create(dai.node.XLinkOut)
    elif hasattr(pipeline, "createXLinkOut"):  # pragma: no cover - legacy SDK fallback
        xout = pipeline.createXLinkOut()
    else:  # pragma: no cover - SDK missing XLinkOut
        raise HTTPException(status_code=503, detail="DepthAI SDK does not expose XLinkOut nodes")

    xout.setStreamName(_STREAM_NAME)
    return xout


def _force_release_devices() -> None:
    """Release lingering DepthAI handles before starting a new stream."""

    if dai is None:
        return

    try:  # pragma: no cover - hardware dependent cleanup
        import gc

        for dev_info in dai.Device.getAllAvailableDevices():
            try:
                temp = dai.Device(dev_info)
            except Exception:
                continue
            try:
                temp.close()
            finally:
                del temp

        gc.collect()
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        LOGGER.debug("DepthAI force-release skipped: %s", exc)


def _create_pipeline() -> tuple["dai.Pipeline", Callable[["dai.Device"], "dai.DataOutputQueue"]]:
    _require_depthai()

    pipeline = dai.Pipeline()

    camera = _create_color_camera(pipeline)
    encoder = _create_video_encoder(pipeline)
    queue_factory: Callable[["dai.Device"], "dai.DataOutputQueue"]

    camera.video.link(encoder.input)

    try:
        xout = _create_xlink_out(pipeline)
    except HTTPException as exc:
        LOGGER.info(
            "DepthAI XLinkOut unavailable; using encoder bitstream queue (reason: %s)",
            exc.detail,
        )

        def _direct_queue(device: "dai.Device") -> "dai.DataOutputQueue":
            bitstream = getattr(encoder, "bitstream", None)
            if bitstream is None or not hasattr(bitstream, "createOutputQueue"):
                raise HTTPException(status_code=503, detail="DepthAI encoder bitstream queue unavailable")
            try:
                return bitstream.createOutputQueue(maxSize=_QUEUE_MAX_SIZE, blocking=False)
            except Exception as queue_exc:  # pragma: no cover - hardware dependent
                raise HTTPException(status_code=503, detail=f"Unable to create encoder output queue: {queue_exc}") from queue_exc

        queue_factory = _direct_queue
    else:
        encoder.bitstream.link(xout.input)

        def _xlink_queue(device: "dai.Device") -> "dai.DataOutputQueue":
            if hasattr(device, "getOutputQueue"):
                return device.getOutputQueue(name=_STREAM_NAME, maxSize=_QUEUE_MAX_SIZE, blocking=False)
            out = getattr(xout, "out", None)
            if out is not None and hasattr(out, "createOutputQueue"):
                return out.createOutputQueue(maxSize=_QUEUE_MAX_SIZE, blocking=False)
            raise HTTPException(status_code=503, detail="DepthAI XLinkOut queue APIs unavailable")

        queue_factory = _xlink_queue

    return pipeline, queue_factory


def _ensure_queue() -> "dai.DataOutputQueue":
    global _device, _queue

    if _queue is not None:
        return _queue

    with _lock:
        if _queue is None:
            pipeline, queue_factory = _create_pipeline()
            _force_release_devices()
            time.sleep(0.5)  # allow USB stack to settle

            device: Optional["dai.Device"] = None
            queue: Optional["dai.DataOutputQueue"] = None

            try:
                device = dai.Device()
                device.startPipeline(pipeline)
                queue = queue_factory(device)
            except Exception as exc:  # pragma: no cover - hardware dependent
                if device is not None:
                    try:
                        device.close()
                    except Exception:  # pragma: no cover - defensive cleanup
                        LOGGER.debug("Error closing DepthAI device after failure", exc_info=True)
                raise HTTPException(status_code=503, detail=f"Unable to start DepthAI stream: {exc}") from exc

            _device = device
            _queue = queue

    return _queue


def _mjpeg_generator() -> Generator[bytes, None, None]:
    _require_depthai()
    queue = _ensure_queue()

    while True:
        try:
            packet = queue.get()
        except Exception:  # pragma: no cover - hardware dependent
            continue

        payload = packet.getData()
        if not payload:
            continue

        frame = bytes(payload)
        yield (
            b"--" + BOUNDARY.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(frame)}\r\n\r\n".encode()
            + frame
            + b"\r\n"
        )


def get_video_response() -> StreamingResponse:
    """Return a streaming response that yields MJPEG frames from the OAK-D."""

    _require_depthai()
    return StreamingResponse(
        _mjpeg_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )


def get_snapshot() -> bytes:
    """Capture a single JPEG frame from the DepthAI stream."""

    _require_depthai()
    queue = _ensure_queue()

    try:
        packet = queue.get()
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise HTTPException(status_code=503, detail=f"Snapshot unavailable: {exc}") from exc

    payload = packet.getData()
    if not payload:
        raise HTTPException(status_code=503, detail="Snapshot unavailable: empty frame")

    return bytes(payload)


def shutdown() -> None:
    """Release cached DepthAI resources when the application exits."""

    global _device, _queue

    with _lock:
        if _queue is not None:
            try:
                _queue.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        if _device is not None:
            try:
                _device.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

        _queue = None
        _device = None
