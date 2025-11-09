"""DepthAI-powered MJPEG streaming helpers for the OAK-D Lite."""
from __future__ import annotations

import threading
from typing import Generator, Optional

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]

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

_lock = threading.Lock()
_device: Optional["dai.Device"] = None
_queue: Optional["dai.DataOutputQueue"] = None


def _require_dependencies() -> None:
    if dai is None or cv2 is None:  # pragma: no cover - environment dependent
        raise HTTPException(status_code=503, detail="DepthAI streaming dependencies are unavailable")


def _create_pipeline() -> "dai.Pipeline":
    _require_dependencies()

    pipeline = dai.Pipeline()

    cam = pipeline.createColorCamera()
    cam.setPreviewSize(_PREVIEW_WIDTH, _PREVIEW_HEIGHT)
    cam.setInterleaved(False)
    cam.setFps(_FPS)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("video")
    cam.preview.link(xout.input)

    return pipeline


def _ensure_device() -> "dai.DataOutputQueue":
    global _device, _queue

    if _queue is not None:
        return _queue

    with _lock:
        if _queue is None:
            pipeline = _create_pipeline()
            try:
                device = dai.Device(pipeline)
                queue = device.getOutputQueue(name="video", maxSize=_QUEUE_MAX_SIZE, blocking=False)
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
            img = packet.getCvFrame()
            ok, jpg = cv2.imencode(".jpg", img)
        except Exception:  # pragma: no cover - hardware dependent
            continue

        if not ok:
            continue

        frame = jpg.tobytes()
        yield (
            b"--" + BOUNDARY.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


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
        img = packet.getCvFrame()
        ok, jpg = cv2.imencode(".jpg", img)
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise HTTPException(status_code=503, detail=f"Snapshot unavailable: {exc}") from exc

    if not ok:
        raise HTTPException(status_code=503, detail="Failed to encode snapshot")

    return jpg.tobytes()
