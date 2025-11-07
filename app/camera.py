from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Optional

import anyio

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore


LOGGER = logging.getLogger("uvicorn.error").getChild(__name__)
LOGGER.setLevel(logging.INFO)


class CameraError(RuntimeError):
    """Raised when a camera frame cannot be retrieved."""


class CameraSource:
    """Abstract base class for MJPEG-compatible camera frame sources."""

    async def get_jpeg_frame(self) -> bytes:
        raise NotImplementedError

    async def close(self) -> None:
        return None


@dataclass
class PlaceholderCameraSource(CameraSource):
    """A camera source that always returns the same pre-encoded JPEG payload."""

    frame: bytes

    async def get_jpeg_frame(self) -> bytes:  # pragma: no cover - trivial
        if not self.frame:
            raise CameraError("No placeholder frame available")
        return self.frame


class OpenCVCameraSource(CameraSource):
    """Camera source backed by OpenCV video capture."""

    def __init__(
        self,
        device: int | str = 0,
        *,
        jpeg_quality: int = 80,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        preferred_fps: Optional[float] = None,
    ) -> None:
        if cv2 is None:  # pragma: no cover - depends on optional import
            raise CameraError("OpenCV is not installed")

        self._device = device
        self._jpeg_quality = int(jpeg_quality)
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._preferred_fps = preferred_fps
        self._capture: Optional["cv2.VideoCapture"] = None
        self._lock = asyncio.Lock()

    async def _ensure_capture(self) -> "cv2.VideoCapture":
        async with self._lock:
            if self._capture is None:
                capture = cv2.VideoCapture(self._device)
                if not capture.isOpened():
                    raise CameraError(f"Unable to open camera device {self._device}")

                if self._frame_width:
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._frame_width))
                if self._frame_height:
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._frame_height))
                if self._preferred_fps:
                    capture.set(cv2.CAP_PROP_FPS, float(self._preferred_fps))

                self._capture = capture

        assert self._capture is not None
        return self._capture

    def _read_and_encode(self) -> bytes:
        assert cv2 is not None  # For type checkers

        capture = self._capture or cv2.VideoCapture(self._device)
        if not capture.isOpened():
            raise CameraError(f"Unable to open camera device {self._device}")

        ok, frame = capture.read()
        if not ok or frame is None:
            raise CameraError("Failed to capture frame from camera")

        params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        ok, encoded = cv2.imencode(".jpg", frame, params)
        if not ok:
            raise CameraError("Failed to encode frame as JPEG")

        # Cache the capture instance for future calls after a successful frame.
        if self._capture is None:
            self._capture = capture

        return encoded.tobytes()

    async def get_jpeg_frame(self) -> bytes:
        await self._ensure_capture()
        return await anyio.to_thread.run_sync(self._read_and_encode)

    async def close(self) -> None:
        if self._capture is not None:
            capture = self._capture
            self._capture = None
            await anyio.to_thread.run_sync(capture.release)


class CameraService:
    """Provide access to MJPEG frames with automatic fallback handling."""

    def __init__(
        self,
        primary: Optional[CameraSource],
        *,
        fallback: Optional[CameraSource] = None,
        boundary: str = "frame",
        frame_rate: float = 10.0,
    ) -> None:
        if primary is None and fallback is None:
            raise ValueError("At least one camera source must be provided")

        self._primary = primary
        self._fallback = fallback
        self._boundary = boundary
        self._frame_delay = 1.0 / frame_rate if frame_rate > 0 else 0.0

    @property
    def boundary(self) -> str:
        return self._boundary

    @property
    def frame_delay(self) -> float:
        return self._frame_delay

    async def _fetch_frame(self) -> bytes:
        errors: list[str] = []
        for source in self._sources:
            if source is None:
                continue
            try:
                frame = await source.get_jpeg_frame()
                LOGGER.info(
                    "Fetched frame from %s (%d bytes)",
                    source.__class__.__name__,
                    len(frame),
                )
                return frame
            except CameraError as exc:
                LOGGER.warning("Camera source %s failed: %s", source.__class__.__name__, exc)
                errors.append(str(exc))

                if source is self._primary and self._fallback is not None:
                    LOGGER.info(
                        "Disabling primary camera source after failure; switching to fallback"
                    )
                    try:
                        await source.close()
                    except Exception:  # pragma: no cover - defensive cleanup
                        LOGGER.debug(
                            "Ignored error while closing failed primary camera source",
                            exc_info=True,
                        )
                    self._primary = None

                continue

        if errors:
            raise CameraError("; ".join(errors))
        raise CameraError("No camera source available")

    @property
    def _sources(self) -> Iterable[Optional[CameraSource]]:
        return (self._primary, self._fallback)

    async def get_frame(self) -> bytes:
        return await self._fetch_frame()

    async def stream(self, *, max_frames: Optional[int] = None) -> AsyncIterator[bytes]:
        emitted = 0
        while max_frames is None or emitted < max_frames:
            payload = await self._fetch_frame()
            header = (
                f"--{self._boundary}\r\n"
                "Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(payload)}\r\n\r\n"
            ).encode()
            yield header + payload + b"\r\n"
            emitted += 1
            if self._frame_delay:
                await anyio.sleep(self._frame_delay)

    async def close(self) -> None:
        for source in self._sources:
            if source is not None:
                with contextlib.suppress(Exception):  # pragma: no cover - defensive
                    await source.close()


__all__ = [
    "CameraError",
    "CameraService",
    "CameraSource",
    "OpenCVCameraSource",
    "PlaceholderCameraSource",
]
