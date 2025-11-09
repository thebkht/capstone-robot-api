"""DepthAI-powered streaming helpers modelled after the Luxonis RTSP example."""
from __future__ import annotations

import threading
from typing import Generator, Optional

try:  # pragma: no cover - optional dependency
    import depthai as dai
except Exception:  # pragma: no cover - optional dependency
    dai = None  # type: ignore[assignment]

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

BOUNDARY = "frame"
_QUEUE_MAX_SIZE = 4
_FPS = 30
_PREVIEW_WIDTH = 640
_PREVIEW_HEIGHT = 360
_STREAM_NAME = "video"

_lock = threading.Lock()
_device: Optional["dai.Device"] = None
_queue: Optional["dai.DataOutputQueue"] = None


def _require_dependencies() -> None:
    if dai is None:  # pragma: no cover - environment dependent
        raise HTTPException(status_code=503, detail="DepthAI streaming dependencies are unavailable")


def _create_color_camera(pipeline: "dai.Pipeline") -> "dai.node.ColorCamera":
    if hasattr(dai, "node") and hasattr(dai.node, "ColorCamera"):
        camera = pipeline.create(dai.node.ColorCamera)
    else:  # pragma: no cover - legacy SDK fallback
        camera = pipeline.createColorCamera()
    camera.setBoardSocket(dai.CameraBoardSocket.RGB)
    camera.setPreviewSize(_PREVIEW_WIDTH, _PREVIEW_HEIGHT)
    camera.setInterleaved(False)
    camera.setFps(_FPS)
    return camera


def _create_video_encoder(pipeline: "dai.Pipeline") -> "dai.node.VideoEncoder":
    if hasattr(dai, "node") and hasattr(dai.node, "VideoEncoder"):
        encoder = pipeline.create(dai.node.VideoEncoder)
    else:  # pragma: no cover - legacy SDK fallback
        encoder = pipeline.createVideoEncoder()
    encoder.setDefaultProfilePreset(_FPS, dai.VideoEncoderProperties.Profile.MJPEG)
    return encoder


def _create_xlink_out(pipeline: "dai.Pipeline") -> "dai.node.XLinkOut":
    if hasattr(dai, "node") and hasattr(dai.node, "XLinkOut"):
        xout = pipeline.create(dai.node.XLinkOut)
    elif hasattr(pipeline, "createXLinkOut"):  # pragma: no cover - legacy SDK fallback
        xout = pipeline.createXLinkOut()
    else:  # pragma: no cover - missing SDK support
        raise HTTPException(status_code=503, detail="DepthAI SDK does not provide XLinkOut nodes")
    xout.setStreamName(_STREAM_NAME)
    return xout


def _create_pipeline() -> "dai.Pipeline":
    _require_dependencies()

    pipeline = dai.Pipeline()

    camera = _create_color_camera(pipeline)
    encoder = _create_video_encoder(pipeline)
    xout = _create_xlink_out(pipeline)

    camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camera.setPreviewKeepAspectRatio(False)

    camera.preview.link(encoder.input)
    encoder.bitstream.link(xout.input)

    return pipeline


def _ensure_device() -> "dai.DataOutputQueue":
    global _device, _queue

    if _queue is not None:
        return _queue

    with _lock:
        if _queue is None:
            pipeline = _create_pipeline()
            try:
                device = dai.Device()
                device.startPipeline(pipeline)
                queue = device.getOutputQueue(
                    name=_STREAM_NAME,
                    maxSize=_QUEUE_MAX_SIZE,
                    blocking=False,
                )
            except Exception as exc:  # pragma: no cover - hardware dependent
                raise HTTPException(status_code=503, detail=f"Unable to start DepthAI stream: {exc}") from exc

            _device = device
            _queue = queue

    return _queue


def _mjpeg_generator() -> Generator[bytes, None, None]:
    _require_dependencies()
    queue = _ensure_device()

    while True:
        try:
            packet = queue.get()
        except Exception:  # pragma: no cover - hardware dependent
            continue

        data = packet.getData()
        if not data:
            continue

        frame = bytes(data)
        yield (
            b"--" + BOUNDARY.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(frame)}\r\n\r\n".encode()
            + frame
            + b"\r\n"
        )


def shutdown() -> None:
    """Release the cached DepthAI device."""

    global _device, _queue

    with _lock:
        if _queue is not None:
            try:
                _queue.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        if _device is not None:
            try:
                _device.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        _queue = None
        _device = None


def get_video_response() -> StreamingResponse:
    """Return a streaming response that yields MJPEG frames from the OAK-D."""

    _require_dependencies()
    return StreamingResponse(
        _mjpeg_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )


def get_snapshot() -> bytes:
    """Capture a single JPEG frame from the stream."""

    _require_dependencies()
    queue = _ensure_device()
    try:
        packet = queue.get()
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise HTTPException(status_code=503, detail=f"Snapshot unavailable: {exc}") from exc

    data = packet.getData()
    if not data:
        raise HTTPException(status_code=503, detail="Snapshot unavailable: empty frame")

    return bytes(data)
