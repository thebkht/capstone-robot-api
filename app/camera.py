from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Optional

import anyio

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

try:
    import depthai as dai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    dai = None  # type: ignore


LOGGER = logging.getLogger("uvicorn.error").getChild(__name__)


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

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the OpenCV dependency is installed."""

        return cv2 is not None

    def __init__(
        self,
        device: int | str = 0,
        *,
        jpeg_quality: int = 80,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        preferred_fps: Optional[float] = None,
    ) -> None:
        if not self.is_available():  # pragma: no cover - depends on optional import
            raise CameraError(
                "OpenCV is not installed. Install the 'opencv-python' package to use the USB camera."
            )

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


class DepthAICameraSource(CameraSource):
    """Camera source that captures MJPEG frames from an OAK-D device."""

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the DepthAI dependency is installed."""

        return dai is not None

    def __init__(
        self,
        *,
        preview_width: int = 640,
        preview_height: int = 480,
        fps: float = 30.0,
    ) -> None:
        if not self.is_available():  # pragma: no cover - depends on optional import
            raise CameraError(
                "DepthAI is not installed. Install the 'depthai' package to use the OAK-D camera."
            )

        if preview_width <= 0 or preview_height <= 0:
            raise ValueError("Preview dimensions must be positive")
        if fps <= 0:
            raise ValueError("FPS must be positive")

        self._preview_width = int(preview_width)
        self._preview_height = int(preview_height)
        self._fps = float(fps)
        self._device: Optional["dai.Device"] = None
        self._queue: Optional[Any] = None
        self._stream_name = "mjpeg"
        self._lock = asyncio.Lock()

    def _start_pipeline(self) -> None:
        assert dai is not None  # For type checkers

        pipeline = dai.Pipeline()

        # DepthAI 3.x API uses pipeline.create() with node types. Earlier versions
        # expose helpers such as createColorCamera instead. Support both styles so
        # the service works across DepthAI releases.
        camera = self._create_node(pipeline, "ColorCamera")
        camera.setPreviewSize(self._preview_width, self._preview_height)
        camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camera.setInterleaved(False)
        camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camera.setFps(self._fps)

        encoder = self._create_node(pipeline, "VideoEncoder")
        # DepthAI 3.x API: setDefaultProfilePreset only takes fps and profile
        # Width and height are automatically determined from the input stream
        encoder.setDefaultProfilePreset(
            self._fps,
            dai.VideoEncoderProperties.Profile.MJPEG,
        )
        camera.preview.link(encoder.input)

        queue_factory = self._configure_output_queue(pipeline, encoder)

        device: Optional["dai.Device"] = None
        queue: Any = None
        try:
            device = dai.Device()
            device.startPipeline(pipeline)
            self._log_device_connection(device)
            queue = queue_factory(device)
            if queue is None:
                raise CameraError("DepthAI output queue unavailable")
        except Exception as exc:
            if device is not None:
                try:
                    device.close()
                except Exception as close_error:  # pragma: no cover - depends on SDK internals
                    LOGGER.debug(
                        "Error closing DepthAI device during cleanup: %s",
                        close_error,
                    )
            raise CameraError(f"Failed to initialize DepthAI camera: {exc}") from exc

        self._device = device
        self._queue = queue

    def _configure_output_queue(self, pipeline: "dai.Pipeline", encoder) -> Callable[[Any], Any]:
        """Set up host queue creation for the encoded MJPEG stream."""

        assert dai is not None  # For type checkers

        try:
            xout = self._create_node(pipeline, "XLinkOut")
        except CameraError as exc:
            LOGGER.info(
                "DepthAI XLinkOut node unavailable; using direct output queue (reason: %s)",
                exc,
            )

            def create_direct_queue(_device: Any) -> Any:
                bitstream = getattr(encoder, "bitstream", None)
                if bitstream is None or not hasattr(bitstream, "createOutputQueue"):
                    raise CameraError("DepthAI VideoEncoder output cannot create a host queue")
                LOGGER.debug("Creating DepthAI direct output queue from encoder bitstream")
                try:
                    queue = bitstream.createOutputQueue(maxSize=1, blocking=False)
                except Exception as exc:  # pragma: no cover - depends on SDK internals
                    raise CameraError(f"Failed to create direct output queue: {exc}") from exc
                LOGGER.info("Successfully created DepthAI direct output queue")
                return queue

            return create_direct_queue

        xout.setStreamName(self._stream_name)
        encoder.bitstream.link(xout.input)

        def create_xlink_queue(device: Any) -> Any:
            get_output_queue = getattr(device, "getOutputQueue", None)
            if callable(get_output_queue):
                return get_output_queue(name=self._stream_name, maxSize=1, blocking=False)

            out_socket = getattr(xout, "out", None)
            if out_socket is not None and hasattr(out_socket, "createOutputQueue"):
                return out_socket.createOutputQueue(maxSize=1, blocking=False)

            raise CameraError("DepthAI SDK does not expose XLinkOut queue creation APIs")

        return create_xlink_queue

    def _create_node(self, pipeline: "dai.Pipeline", node_name: str):
        """Create a DepthAI node, supporting both legacy and modern APIs."""

        assert dai is not None  # For type checkers

        errors: list[str] = []
        diagnostics: list[str] = []

        def _attempt(description: str, factory):
            try:
                return factory()
            except Exception as exc:  # pragma: no cover - depends on SDK internals
                errors.append(f"{description} failed: {exc}")
                return None

        def _snake_case(name: str) -> str:
            result: list[str] = []
            for index, char in enumerate(name):
                if char.isupper() and index > 0 and not name[index - 1].isupper():
                    result.append("_")
                result.append(char.lower())
            return "".join(result)

        def _find_node_class(module) -> Optional[type]:
            visited: set[int] = set()

            def _walk(candidate_module) -> Optional[type]:
                module_id = id(candidate_module)
                if module_id in visited:
                    return None
                visited.add(module_id)

                try:
                    direct = getattr(candidate_module, node_name)
                except AttributeError:
                    direct = None

                if inspect.isclass(direct):
                    return direct

                exported = getattr(candidate_module, "__all__", None)
                names = exported if isinstance(exported, Iterable) else dir(candidate_module)
                for attr_name in names:
                    if attr_name.startswith("_"):
                        continue
                    try:
                        attr = getattr(candidate_module, attr_name)
                    except AttributeError:
                        continue
                    if inspect.isclass(attr) and attr.__name__ == node_name:
                        return attr
                    if inspect.ismodule(attr):
                        found = _walk(attr)
                        if found is not None:
                            return found
                return None

            return _walk(module)

        node_module = getattr(dai, "node", None)
        imported_node_module = None

        if node_module is None:
            try:
                imported_node_module = importlib.import_module("depthai.node")
            except ModuleNotFoundError:
                diagnostics.append("depthai.node module: missing")
            except Exception as exc:  # pragma: no cover - depends on SDK internals
                diagnostics.append(f"depthai.node module import failed: {exc}")
            else:
                node_module = imported_node_module

        if node_module is not None:
            direct_class = getattr(node_module, node_name, None)
            if direct_class is not None:
                result = _attempt(f"pipeline.create(node.{node_name})", lambda: pipeline.create(direct_class))
                if result is not None:
                    return result
            else:
                diagnostics.append(f"dai.node.{node_name}: missing")

            discovered_class = _find_node_class(node_module)
            if discovered_class is not None and discovered_class is not direct_class:
                result = _attempt(
                    f"pipeline.create(node.*.{node_name})",
                    lambda: pipeline.create(discovered_class),
                )
                if result is not None:
                    return result
        elif imported_node_module is None:
            diagnostics.append("dai.node: missing")

        top_level_node = getattr(dai, node_name, None)
        if top_level_node is not None:
            result = _attempt(f"pipeline.create({node_name})", lambda: pipeline.create(top_level_node))
            if result is not None:
                return result
        else:
            diagnostics.append(f"dai.{node_name}: missing")

        legacy_creator = getattr(pipeline, f"create{node_name}", None)
        if callable(legacy_creator):
            result = _attempt(f"pipeline.create{node_name}()", legacy_creator)
            if result is not None:
                return result
        else:
            diagnostics.append(f"pipeline.create{node_name}(): missing")

        snake_name = _snake_case(node_name)
        if snake_name:
            module_path = f"depthai.node.{snake_name}"
            try:
                snake_module = importlib.import_module(module_path)
            except ModuleNotFoundError:
                diagnostics.append(f"{module_path}: missing")
            except Exception as exc:  # pragma: no cover - depends on SDK internals
                diagnostics.append(f"{module_path} import failed: {exc}")
            else:
                node_class = getattr(snake_module, node_name, None)
                if inspect.isclass(node_class):
                    result = _attempt(
                        f"pipeline.create({module_path}.{node_name})",
                        lambda: pipeline.create(node_class),
                    )
                    if result is not None:
                        return result
                else:
                    diagnostics.append(f"{module_path}.{node_name}: missing")

        diagnostics_message = ", ".join(sorted({*diagnostics})) or None

        error_hint = f"DepthAI node '{node_name}' is unavailable in this SDK"
        if errors:
            error_hint = f"{error_hint} ({'; '.join(errors)})"

        depthai_version = getattr(dai, "__version__", "unknown")
        diagnostic_parts = [f"DepthAI version: {depthai_version}"]
        if diagnostics_message:
            diagnostic_parts.append(diagnostics_message)

        LOGGER.warning(
            "DepthAI diagnostics for missing node %s: %s",
            node_name,
            "; ".join(diagnostic_parts),
        )

        raise CameraError(error_hint)

    def _log_device_connection(self, device: "dai.Device") -> None:
        """Emit diagnostic details about the connected DepthAI device."""

        connected_cameras: list[str] = []
        try:
            connected_cameras = [socket.name for socket in device.getConnectedCameras()]
        except Exception:  # pragma: no cover - best effort logging
            LOGGER.debug("Unable to query connected cameras from DepthAI device", exc_info=True)

        try:
            usb_speed = device.getUsbSpeed().name
        except Exception:  # pragma: no cover - best effort logging
            usb_speed = "unknown"

        try:
            device_info = device.getDeviceInfo()
            mxid = device_info.getMxId() if hasattr(device_info, "getMxId") else getattr(device_info, "mxid", "unknown")
        except Exception:  # pragma: no cover - best effort logging
            mxid = "unknown"

        LOGGER.info(
            "DepthAI device connected (mxid=%s, usb_speed=%s, cameras=%s)",
            mxid,
            usb_speed,
            connected_cameras or ["none"],
            extra={
                "mxid": mxid,
                "usb_speed": usb_speed,
                "connected_cameras": connected_cameras or ["none"],
            },
        )

        if not connected_cameras:
            LOGGER.warning(
                "DepthAI device reports no connected cameras; verify the OAK-D is seated and powered",
                extra={"mxid": mxid},
            )

    async def _ensure_pipeline(self) -> None:
        async with self._lock:
            if self._queue is not None:
                return

            try:
                await anyio.to_thread.run_sync(self._start_pipeline)
            except Exception as exc:  # pragma: no cover - hardware dependent
                raise CameraError(f"Unable to start DepthAI camera: {exc}") from exc

            if self._queue is None:  # pragma: no cover - defensive
                raise CameraError("DepthAI camera queue unavailable")

    def _read_mjpeg_frame(self) -> bytes:
        if self._queue is None:
            raise CameraError("DepthAI pipeline has not been started")

        packet = self._queue.get()  # Blocking read in worker thread
        if packet is None:
            raise CameraError("No frame received from DepthAI camera")

        data = packet.getData()
        if not data:
            raise CameraError("Received empty frame from DepthAI camera")

        return bytes(data)

    async def get_jpeg_frame(self) -> bytes:
        await self._ensure_pipeline()
        return await anyio.to_thread.run_sync(self._read_mjpeg_frame)

    def _release_resources(self) -> None:
        if self._queue is not None:
            with contextlib.suppress(AttributeError):
                self._queue.close()
            self._queue = None

        if self._device is not None:
            with contextlib.suppress(Exception):
                self._device.close()
            self._device = None

    async def close(self) -> None:
        if self._queue is None and self._device is None:
            return

        await anyio.to_thread.run_sync(self._release_resources)


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
                LOGGER.debug(
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
    "DepthAICameraSource",
    "OpenCVCameraSource",
    "PlaceholderCameraSource",
]
