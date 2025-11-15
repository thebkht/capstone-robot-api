"""Regular camera MJPEG streaming helper using OpenCV."""
from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .camera import CameraError


class CameraStreamError(RuntimeError):
    """Raised when the camera MJPEG runtime cannot be prepared."""

BOUNDARY = "frame"
_DEFAULT_PREVIEW_SIZE = (640, 480)
_DEFAULT_FPS = 30.0
_JPEG_QUALITY = 80

LOGGER = logging.getLogger("uvicorn.error").getChild(__name__)


@dataclass
class _StreamArgs:
    device: int | str | None
    fps_limit: float | None


@dataclass
class _RuntimeState:
    capture: "cv2.VideoCapture"
    device: int | str


_lock = threading.Lock()
_runtime_state: Optional[_RuntimeState] = None


def initialize_argparser() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Match the argument parser using environment defaults."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default=os.getenv("CAMERA_DEVICE", "0"))
    parser.add_argument("--fps-limit", default=os.getenv("CAMERA_FPS_LIMIT"))

    # We do not expose command line parsing in the API runtime. Instead we
    # parse an empty list so that the defaults above are used verbatim.
    args = parser.parse_args([])
    return parser, args


_, _parsed_args = initialize_argparser()
_ARGS = _StreamArgs(
    device=int(_parsed_args.device) if _parsed_args.device.isdigit() else _parsed_args.device,
    fps_limit=float(_parsed_args.fps_limit) if _parsed_args.fps_limit else None,
)


def _require_opencv() -> None:
    if cv2 is None:  # pragma: no cover - environment dependent
        raise HTTPException(status_code=503, detail="OpenCV camera streaming dependencies are unavailable")


def _create_capture(device: int | str) -> "cv2.VideoCapture":
    """Create and configure a VideoCapture instance."""
    _require_opencv()
    
    capture = cv2.VideoCapture(device)
    if not capture.isOpened():
        raise CameraStreamError(f"Unable to open camera device {device}")
    
    # Set resolution
    preview_width, preview_height = _DEFAULT_PREVIEW_SIZE
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(preview_width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(preview_height))
    
    # Set FPS (use limit if specified, otherwise use default)
    fps = _ARGS.fps_limit if _ARGS.fps_limit else _DEFAULT_FPS
    capture.set(cv2.CAP_PROP_FPS, float(fps))
    
    return capture


def _force_release_capture() -> None:
    """Release any existing camera captures."""
    global _runtime_state
    
    if _runtime_state is not None:
        try:
            _runtime_state.capture.release()
        except Exception:  # pragma: no cover - best-effort cleanup
            LOGGER.debug("Error releasing camera capture", exc_info=True)
        _runtime_state = None


def _ensure_runtime() -> _RuntimeState:
    global _runtime_state

    if _runtime_state is not None:
        return _runtime_state

    with _lock:
        if _runtime_state is not None:
            return _runtime_state

        _force_release_capture()
        time.sleep(0.5)

        device = _ARGS.device if _ARGS.device is not None else 0
        capture: Optional["cv2.VideoCapture"] = None

        attempts = 3
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                capture = _create_capture(device)
                # Test that we can read a frame
                ret, frame = capture.read()
                if not ret or frame is None:
                    raise CameraStreamError("Failed to read initial frame from camera")
            except Exception as exc:  # pragma: no cover - hardware dependent
                last_error = exc
                if capture is not None:
                    try:
                        capture.release()
                    except Exception:
                        LOGGER.debug("Error closing camera after failure", exc_info=True)
                capture = None
                if attempt < attempts:
                    LOGGER.info(
                        "Retrying camera startup after failure",
                        extra={"attempt": attempt, "error": str(exc)},
                    )
                    time.sleep(0.5)
                    continue
                break
            else:
                last_error = None
                break

        if capture is None or last_error is not None:
            if last_error is not None:
                detail = f"Unable to start camera stream: {last_error}"
            else:
                detail = "Camera capture unavailable"
            raise CameraStreamError(detail)

        _runtime_state = _RuntimeState(capture=capture, device=device)
        return _runtime_state


def _frame_to_jpeg(frame) -> bytes | None:
    """Convert an OpenCV frame to JPEG bytes."""
    if cv2 is None:  # pragma: no cover - optional dependency
        LOGGER.debug("Discarding frame; OpenCV not available to encode")
        return None

    params = [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
    ok, encoded = cv2.imencode(".jpg", frame, params)
    if not ok:
        return None
    return encoded.tobytes()


def _mjpeg_chunks(state: _RuntimeState) -> Generator[bytes, None, None]:
    """Generate MJPEG stream chunks from camera frames."""
    capture = state.capture
    frame_delay = 1.0 / (_ARGS.fps_limit or _DEFAULT_FPS) if (_ARGS.fps_limit or _DEFAULT_FPS) > 0 else 0.0

    while True:
        try:
            ret, frame = capture.read()
            if not ret or frame is None:
                LOGGER.debug("Failed to read frame from camera; ending stream")
                break
        except Exception:  # pragma: no cover - hardware dependent
            LOGGER.debug("Error reading from camera; ending stream", exc_info=True)
            break

        payload = _frame_to_jpeg(frame)
        if not payload:
            continue

        header = (
            f"--{BOUNDARY}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(payload)}\r\n\r\n"
        ).encode()
        yield header + payload + b"\r\n"
        
        if frame_delay > 0:
            time.sleep(frame_delay)


def get_video_response() -> StreamingResponse:
    """Get a streaming MJPEG response from the camera."""
    try:
        state = _ensure_runtime()
    except CameraStreamError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail=f"Camera stream unavailable: {exc}") from exc

    generator = _mjpeg_chunks(state)
    return StreamingResponse(generator, media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")


def get_snapshot() -> bytes:
    """Get a single JPEG snapshot from the camera."""
    try:
        state = _ensure_runtime()
    except CameraStreamError as exc:
        raise CameraError(str(exc)) from exc

    capture = state.capture

    try:
        ret, frame = capture.read()
        if not ret or frame is None:
            raise CameraError("Failed to capture frame from camera")
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise CameraError(f"Unable to read camera snapshot: {exc}") from exc

    payload = _frame_to_jpeg(frame)
    if not payload:
        raise CameraError("Empty camera snapshot payload")
    return payload


def shutdown() -> None:
    """Shutdown and release camera resources."""
    global _runtime_state

    with _lock:
        _force_release_capture()
