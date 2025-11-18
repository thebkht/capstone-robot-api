from __future__ import annotations

import json
import asyncio
import base64
import hashlib
import hmac
import importlib
import logging
import os
import secrets
import subprocess
import socket
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any, AsyncIterator, Optional
from starlette.middleware.trustedhost import TrustedHostMiddleware

LOGGER = logging.getLogger("uvicorn.error").getChild(__name__)

# Ensure project root is in Python path for imports when running as service
# This MUST happen before any app.* imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import anyio
from fastapi import FastAPI, HTTPException, Header, Query, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Import Rover controller - try multiple import styles so deployment layouts work
Rover = None
serial = None
_rover_import_error = None


def _attempt_import(module_name: str, package: str | None = None) -> tuple[bool, str | None]:
    """Attempt to import the rover_controller module variant."""

    global Rover, serial, _rover_import_error

    try:
        if package is None:
            module = importlib.import_module(module_name)
        else:
            module = importlib.import_module(module_name, package)
    except ImportError as exc:
        LOGGER.debug("Failed to import %s (package=%s): %s", module_name, package, exc, exc_info=True)
        return False, f"{module_name} ({package or '-'}) import error: {exc}"

    rover_cls = getattr(module, "Rover", None)

    if rover_cls is None:
        message = f"{module_name} missing Rover class (Rover={rover_cls})"
        LOGGER.debug(message)
        return False, message

    serial_module = getattr(module, "serial", None)
    if serial_module is None:
        try:
            serial_module = importlib.import_module("serial")
        except ImportError as exc:
            message = (
                f"{module_name} missing serial module and pyserial import failed: {exc}"
            )
            LOGGER.debug(message, exc_info=True)
            return False, message

    Rover = rover_cls
    serial = serial_module
    _rover_import_error = None
    LOGGER.info("rover_controller module imported via %s", module_name)
    return True, None


_import_failures: list[str] = []
_candidates: list[tuple[str, str | None]] = [("app.rover_controller", None)]
if __package__:
    _candidates.append((".rover_controller", __package__))
_candidates.append(("rover_controller", None))

for candidate, package in _candidates:
    success, error_msg = _attempt_import(candidate, package)
    if success:
        break
    if error_msg:
        _import_failures.append(error_msg)
else:
    _rover_import_error = " ; ".join(_import_failures)

from .camera import (
    CameraError,
    CameraService,
    DepthAICameraSource,
    OpenCVCameraSource,
    PlaceholderCameraSource,
)
from .oak_stream import get_snapshot as oak_snapshot
from .oak_stream import get_video_response as oak_video_response
from .oak_stream import shutdown as oak_shutdown
from .oak_stream import ensure_runtime as oak_ensure_runtime
from .oak_stream import frame_to_jpeg as oak_frame_to_jpeg
from .models import (
    CaptureRequest,
    CaptureResponse,
    CaptureType,
    ClaimConfirmRequest,
    ClaimConfirmResponse,
    ClaimControlResponse,
    ClaimRequestResponse,
    HeadCommand,
    HealthResponse,
    LightCommand,
    Mode,
    ModeResponse,
    MoveCommand,
    NodCommand,
    NetworkInfoResponse,
    StatusResponse,
    StopResponse,
    WiFiConnectRequest,
    WiFiConnectResponse,
    WiFiNetwork,
    WiFiScanResponse,
    WiFiStatusResponse,
)

APP_NAME = "capstone-robot-api"
APP_VERSION = "0.1.0"
ROBOT_NAME = "rover-01"
ROBOT_SERIAL = "rovy-01"
BOUNDARY = "frame"

# Log import status for Rover
if Rover is None:
    if _rover_import_error:
        LOGGER.error("IMPORT ERROR DETAILS: rover_controller module not available: %s; OLED display will be disabled", _rover_import_error)
    else:
        LOGGER.warning("rover_controller module not available; OLED display will be disabled")
else:
    LOGGER.info("rover_controller module imported successfully")

# Claim system state
STATE = {
    "claimed": False,
    "control_token_hash": None,
    "pin": None,
    "pin_exp": 0,
    "controller": {"sid": None, "last": 0, "ttl": 30},
}

_PIN_RESET_TASK: asyncio.Task[None] | None = None

_PLACEHOLDER_JPEG = base64.b64decode(
    """
/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDwwMDw8NDhERExUTGBonHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8f/2wBDARESEhgVGBoZGB4dHy8fLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8v/3QAEAA3/2gAIAQEAAD8A/wD/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAsf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/AR//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/AR//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Ar//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IX//2Q==
""".strip()
)
app = FastAPI(title="Capstone Robot API", version=APP_VERSION)

# Add before CORS middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]
)

@app.middleware("http")
async def handle_proxy_headers(request: Request, call_next):
    # Check if this is coming through Tailscale Funnel
    if request.headers.get("tailscale-funnel-request"):
        # Tailscale Funnel sometimes doesn't preserve POST method
        # Get the intended method from a custom header if available
        pass
    
    response = await call_next(request)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def hash_token(t: str) -> str:
    """Hash a token using SHA-256."""
    return hashlib.sha256(t.encode()).hexdigest()


def verify_token(token: str) -> bool:
    """Verify a control token using constant-time comparison."""
    if not (STATE["claimed"] and STATE["control_token_hash"]):
        return False
    return hmac.compare_digest(hash_token(token), STATE["control_token_hash"])


def verify_session(session_id: str) -> bool:
    """Verify a controller session ID and update last access time."""
    if not STATE["controller"]["sid"]:
        return False
    # Use constant-time comparison for session ID
    if not hmac.compare_digest(session_id, STATE["controller"]["sid"]):
        return False
    now = time.time()
    if now - STATE["controller"]["last"] > STATE["controller"]["ttl"]:
        # Session expired
        STATE["controller"]["sid"] = None
        STATE["controller"]["last"] = 0
        return False
    # Update last access time
    STATE["controller"]["last"] = now
    return True


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to verify x-control-token and session_id for protected endpoints."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Check if endpoint requires control token
        protected = (
            path.startswith("/control")
            or path.startswith("/claim/release")
            or path.startswith("/settings")
            or path == "/claim-control"
        )

        if protected:
            token = request.headers.get("x-control-token")
            if not token or not verify_token(token):
                raise HTTPException(status_code=401, detail="unauthorized")

            # Check if endpoint requires controller session (all /control/* except /claim-control)
            if path.startswith("/control/"):
                session_id = request.headers.get("session-id")
                if not session_id or not verify_session(session_id):
                    raise HTTPException(status_code=403, detail="invalid_or_expired_session")

        return await call_next(request)


app.add_middleware(AuthMiddleware)

def _get_base_controller() -> Optional[Any]:
    cached_controller: Optional[Any] = getattr(app.state, "base_controller", None)

    if cached_controller is not None:
        return cached_controller

    if Rover is None:
        LOGGER.debug("Rover class not available (import failed)")
        return None

    LOGGER.debug("Attempting to initialize base_controller for PIN display")
    device, _ = _find_serial_device()
    if not device:
        LOGGER.debug("No serial device available for Rover initialization")
        return None

    try:
        base_controller = Rover(device)
        LOGGER.info("Rover initialized on %s", device)
    except Exception as exc:
        LOGGER.warning("Failed to initialize Rover on %s: %s", device, exc, exc_info=True)
        return None

    app.state.base_controller = base_controller
    return base_controller


def _cancel_pin_reset_task() -> None:
    """Cancel any scheduled OLED reset task."""

    global _PIN_RESET_TASK

    if _PIN_RESET_TASK is not None and not _PIN_RESET_TASK.done():
        _PIN_RESET_TASK.cancel()

    _PIN_RESET_TASK = None


def _schedule_pin_reset(pin_value: str, expiration: float) -> None:
    """Schedule OLED reset once the current PIN expires."""

    if expiration <= time.time():
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        LOGGER.debug("No running event loop available to schedule PIN reset task")
        return

    _cancel_pin_reset_task()

    global _PIN_RESET_TASK
    _PIN_RESET_TASK = loop.create_task(
        _reset_display_after_expiration(pin_value, expiration)
    )


async def _reset_display_after_expiration(pin_value: str, expiration: float) -> None:
    """Reset the OLED display once the active PIN has expired."""

    global _PIN_RESET_TASK
    current_task = asyncio.current_task()
    delay = max(expiration - time.time(), 0)
    cancelled_exc = anyio.get_cancelled_exc_class()

    try:
        await anyio.sleep(delay)
    except cancelled_exc:  # pragma: no cover - cancellation is timing dependent
        LOGGER.debug("PIN expiration reset task cancelled before completion")
        return

    if STATE["pin"] != pin_value or time.time() < expiration:
        if _PIN_RESET_TASK is current_task:
            _PIN_RESET_TASK = None
        return

    base_controller = _get_base_controller()

    if base_controller and hasattr(base_controller, "display_reset"):
        try:
            base_controller.display_reset()
            LOGGER.info("OLED display reset after PIN expiration")
        except Exception as exc:  # pragma: no cover - hardware dependent
            LOGGER.error(
                "Failed to reset OLED display after PIN expiration: %s", exc, exc_info=True
            )
    else:
        LOGGER.debug(
            "Skipping OLED reset after PIN expiration; base controller unavailable or missing display_reset"
        )

    if STATE["pin"] == pin_value:
        STATE["pin"] = None
        STATE["pin_exp"] = 0

    if _PIN_RESET_TASK is current_task:
        _PIN_RESET_TASK = None

def _get_env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False

    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


_FORCE_WEBCAM = _get_env_flag("CAMERA_FORCE_WEBCAM")
_WEBCAM_DEVICE = os.getenv("CAMERA_WEBCAM_DEVICE")


def _iter_webcam_candidates() -> list[int | str]:
    """Return preferred webcam device identifiers.

    The order favours explicit configuration, then any OAK-D UVC interfaces,
    and finally generic `/dev/video*` indices so we still try something when no
    metadata is available.
    """

    candidates: list[int | str] = []

    if _WEBCAM_DEVICE is not None:
        try:
            candidates.append(int(_WEBCAM_DEVICE))
        except ValueError:
            candidates.append(_WEBCAM_DEVICE)

    by_id_dir = Path("/dev/v4l/by-id")
    if by_id_dir.is_dir():
        for entry in sorted(by_id_dir.iterdir()):
            name = entry.name.lower()
            if "oak" not in name and "depthai" not in name and "luxonis" not in name:
                continue
            try:
                resolved = entry.resolve(strict=True)
            except OSError:
                continue
            candidates.append(str(resolved))

    # Fall back to common numeric indices if nothing more specific was found.
    # These entries are appended after any explicit or detected OAK-D devices
    # so that laptops with built-in webcams still prefer the external device
    # when one is present.
    generic_indices = range(0, 4)
    for index in generic_indices:
        if index not in candidates:
            candidates.append(index)

    return candidates


def _create_camera_service() -> CameraService:
    primary_source = None

    if _FORCE_WEBCAM:
        LOGGER.info(
            "DepthAI camera explicitly disabled via CAMERA_FORCE_WEBCAM; attempting USB webcam sources instead",
        )
    elif DepthAICameraSource.is_available():
        try:
            primary_source = DepthAICameraSource()
            LOGGER.info("Using DepthAI camera source for streaming")
        except CameraError as exc:
            LOGGER.warning("DepthAI camera source unavailable: %s", exc)
    else:
        LOGGER.warning(
            "DepthAI package not installed; skipping OAK-D camera stream. Install the 'depthai' package to enable it."
        )

    if primary_source is None:
        if OpenCVCameraSource.is_available():
            for candidate in _iter_webcam_candidates():
                try:
                    LOGGER.info(
                        "Attempting webcam device %s for primary stream source",
                        candidate,
                    )
                    primary_source = OpenCVCameraSource(device=candidate)
                except CameraError as exc:
                    LOGGER.warning(
                        "OpenCV camera source unavailable on %s: %s",
                        candidate,
                        exc,
                    )
                    primary_source = None
                    continue
                else:
                    LOGGER.info("Using OpenCV camera source for streaming")
                    break
            else:
                LOGGER.warning("Unable to open any webcam device for streaming")
        else:
            LOGGER.warning(
                "OpenCV package not installed; skipping USB camera stream. Install the 'opencv-python' package to enable it."
            )

    fallback_source = None
    if _PLACEHOLDER_JPEG:
        fallback_source = PlaceholderCameraSource(_PLACEHOLDER_JPEG)
        LOGGER.info("Using placeholder camera source for fallback frames")

    if primary_source is None and fallback_source is None:
        raise RuntimeError("No camera source available for streaming")

    return CameraService(primary_source, fallback=fallback_source, boundary=BOUNDARY, frame_rate=10.0)


app.state.camera_service = _create_camera_service()


def _find_serial_device() -> tuple[Optional[str], list[str]]:
    """Find available serial device for rover controller."""
    if serial is None:
        return None, []

    # Allow explicit override via environment variable.
    env_device = os.getenv("ROVER_SERIAL_DEVICE")
    attempted: list[str] = []
    if env_device:
        if os.path.exists(env_device):
            try:
                test_ser = serial.Serial(env_device, 115200, timeout=0.2)
                test_ser.close()
            except (serial.SerialException, PermissionError, OSError) as exc:
                LOGGER.warning(
                    "Configured rover serial device %s unavailable: %s", env_device, exc
                )
            else:
                LOGGER.info("Using rover serial device from environment: %s", env_device)
                return env_device, [env_device]
        else:
            LOGGER.warning("Configured rover serial device %s does not exist", env_device)
        attempted.append(env_device)

    candidates: list[str] = []

    # Probe through pyserial's port listing when available for dynamic detection.
    try:
        from serial.tools import list_ports  # type: ignore

        for port in list_ports.comports():
            if port.device:
                candidates.append(port.device)
    except Exception as exc:  # pragma: no cover - defensive; list_ports may be missing
        LOGGER.debug("Failed to enumerate serial ports via pyserial: %s", exc, exc_info=True)

    # Ensure we also try a sensible default set for Jetson-style deployments.
    candidates.extend(
        device
        for device in ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1"]
        if device not in candidates
    )

    for device in candidates:
        if device not in attempted:
            attempted.append(device)
        if os.path.exists(device):
            try:
                # Try to open it to verify it's accessible
                test_ser = serial.Serial(device, 115200, timeout=0.2)
                test_ser.close()
                LOGGER.info("Detected rover serial device: %s", device)
                return device, attempted
            except (serial.SerialException, PermissionError, OSError) as exc:
                LOGGER.debug("Serial device %s unavailable: %s", device, exc)
                continue
    return None, attempted


@app.get("/")
async def root() -> dict[str, object]:
    """Simple index listing the most commonly used endpoints."""

    return {
        "status": "ok",
        "endpoints": [
            "/video",
            "/shot",
            "/camera/stream",
            "/camera/snapshot",
        ],
    }


@app.get("/video")
async def video_stream() -> StreamingResponse:
    """Expose the main MJPEG stream at the top level for convenience."""

    return oak_video_response()


@app.websocket("/camera/ws")
async def camera_websocket(websocket: WebSocket):
    """WebSocket endpoint for streaming camera frames as base64-encoded JPEG."""
    await websocket.accept()
    LOGGER.info("WebSocket client connected")
    
    try:
        state = oak_ensure_runtime()
        capture = state.capture
        
        while True:
            ret, frame = capture.read()
            if not ret or frame is None:
                LOGGER.debug("Failed to read frame from camera; closing WebSocket")
                await websocket.send_text(json.dumps({"error": "Failed to read frame"}))
                break
                
            # Encode frame to JPEG
            payload = oak_frame_to_jpeg(frame)
            if payload:
                # Send as JSON with base64 frame
                b64_frame = base64.b64encode(payload).decode('utf-8')
                await websocket.send_text(json.dumps({"frame": b64_frame}))  # Send as JSON
            
            # Control frame rate (e.g., 10 FPS)
            await asyncio.sleep(0.1)
            
    except Exception as exc:
        LOGGER.error("WebSocket error: %s", exc, exc_info=True)
        try:
            await websocket.send_text(json.dumps({"error": str(exc)}))
            await websocket.close()
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

@app.get("/shot")
async def single_frame() -> Response:
    """Serve a single JPEG frame without the additional camera namespace."""

    frame = oak_snapshot()
    return Response(content=frame, media_type="image/jpeg")


async def _camera_stream(service: CameraService, frames: int | None) -> AsyncIterator[bytes]:
    emitted = 0
    while frames is None or emitted < frames:
        frame = await service.get_frame()
        header = (
            f"--{service.boundary}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(frame)}\r\n\r\n"
        ).encode()
        yield header + frame + b"\r\n"
        emitted += 1
        if service.frame_delay:
            await anyio.sleep(service.frame_delay)


@app.on_event("shutdown")
async def shutdown_camera() -> None:
    await app.state.camera_service.close()
    oak_shutdown()


@app.get("/health", response_model=HealthResponse, tags=["Discovery"])
async def get_health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        name=ROBOT_NAME,
        serial=ROBOT_SERIAL,
        claimed=STATE["claimed"],
        mode=Mode.ACCESS_POINT,
        version=APP_VERSION,
    )


@app.get("/network-info", response_model=NetworkInfoResponse, tags=["Discovery"])
async def get_network_info() -> NetworkInfoResponse:
    return NetworkInfoResponse(ip="192.168.4.1", ssid=None, hostname=ROBOT_NAME)


@app.get("/camera/snapshot", tags=["Camera"])
async def get_camera_snapshot() -> Response:
    try:
        frame = await app.state.camera_service.get_frame()
    except CameraError as exc:
        raise HTTPException(status_code=503, detail="Snapshot unavailable") from exc

    headers = {"Content-Disposition": "inline; filename=snapshot.jpg"}
    return Response(content=frame, media_type="image/jpeg", headers=headers)


@app.get("/camera/stream", tags=["Camera"])
async def get_camera_stream(frames: int | None = Query(default=None, ge=1)) -> StreamingResponse:
    try:
        response = oak_video_response()
    except HTTPException as exc:
        if exc.status_code != 503:
            raise
        LOGGER.info(
            "DepthAI MJPEG stream unavailable; falling back to camera service",
            extra={"reason": exc.detail},
        )
    else:
        if frames is not None:
            LOGGER.info(
                "Ignoring frame limit request; DepthAI MJPEG stream is continuous",
                extra={"frames": frames},
            )
        return response

    async def stream_generator() -> AsyncIterator[bytes]:
        LOGGER.info("Starting camera stream", extra={"frames": frames})
        frame_count = 0
        try:
            async for chunk in _camera_stream(app.state.camera_service, frames):
                frame_count += 1
                LOGGER.debug("Emitting camera frame chunk (%d bytes)", len(chunk))
                yield chunk
        except CameraError as exc:
            LOGGER.error("Camera stream interrupted: %s", exc)
            raise HTTPException(status_code=503, detail="Camera stream unavailable") from exc
        finally:
            LOGGER.info(
                "Camera stream finished",
                extra={"frames": frames, "frames_sent": frame_count},
            )

    return StreamingResponse(stream_generator(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")


@app.post("/camera/capture", response_model=CaptureResponse, tags=["Camera"])
async def capture_photo(request: CaptureRequest) -> CaptureResponse:
    if request.type != CaptureType.PHOTO:
        raise HTTPException(status_code=400, detail="Only photo capture is supported")

    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    path = f"/media/{timestamp}.jpg"
    url = f"http://192.168.4.1:8000{path}"
    return CaptureResponse(saved=True, path=path, url=url)


def _voltage_to_percentage(voltage: float | None) -> int:
    """Convert a battery voltage reading to a percentage."""

    if voltage is None:
        return 0

    # Heuristic mapping for a 3S LiPo pack commonly used on the rover.
    empty_voltage = 9.0
    full_voltage = 12.6

    percent = (voltage - empty_voltage) / (full_voltage - empty_voltage)
    percent = max(0.0, min(1.0, percent))
    return int(round(percent * 100))


def _default_status() -> StatusResponse:
    """Return a fallback status response when rover data is unavailable."""

    return StatusResponse(battery=82, cpu=37, temp=46.3, ai_state="idle")


@app.get("/status", response_model=StatusResponse, tags=["Status"])
async def get_status() -> StatusResponse:
    LOGGER.info("Status endpoint called")
    base_controller = _get_base_controller()

    if not base_controller:
        LOGGER.info("No base controller available; returning default status")
        return _default_status()
    
    if not hasattr(base_controller, "get_status"):
        LOGGER.warning("Base controller missing get_status method; returning default status")
        return _default_status()

    try:
        LOGGER.debug("Calling base_controller.get_status()")
        rover_status = await anyio.to_thread.run_sync(base_controller.get_status)
        LOGGER.info("Rover status received: %s", rover_status)
    except Exception as exc:  # pragma: no cover - hardware dependent
        LOGGER.error("Failed to obtain rover status: %s", exc, exc_info=True)
        return _default_status()

    battery_percent = _voltage_to_percentage(rover_status["voltage"])
    temperature = rover_status.get("temperature", 0.0) or 0.0

    return StatusResponse(
        battery=battery_percent,
        cpu=int(rover_status.get("cpu", 0)),
        temp=float(temperature),
        ai_state=str(rover_status.get("ai_state", "idle")),
    )


@app.post("/control/move", response_model=MoveCommand, tags=["Control"])
async def move_robot(
    command: MoveCommand,
    x_control_token: str = Header(..., alias="x-control-token"),
    session_id: str = Header(..., alias="session-id"),
) -> MoveCommand:
    """Move the robot. Requires both control token and active session."""
    # Token and session verification handled by middleware
    # This endpoint is currently disabled (returns command without executing)
    return command


@app.post("/control/stop", response_model=StopResponse, tags=["Control"])
async def stop_robot(
    x_control_token: str = Header(..., alias="x-control-token"),
    session_id: str = Header(..., alias="session-id"),
) -> StopResponse:
    """Stop the robot. Requires both control token and active session."""
    # Token and session verification handled by middleware
    # This endpoint is currently disabled (returns success without executing)
    return StopResponse()


@app.post("/control/head", response_model=HeadCommand, tags=["Control"])
async def move_head(command: HeadCommand) -> HeadCommand:
    return command


@app.post("/control/lights", response_model=LightCommand, tags=["Control"])
async def control_lights(
    command: LightCommand,
    x_control_token: str = Header(..., alias="x-control-token"),
    session_id: str = Header(..., alias="session-id"),
) -> LightCommand:
    base_controller = _get_base_controller()

    if not base_controller:
        raise HTTPException(status_code=503, detail="controller_unavailable")

    if not hasattr(base_controller, "lights_ctrl"):
        raise HTTPException(status_code=501, detail="lights_control_not_supported")

    try:
        await anyio.to_thread.run_sync(
            base_controller.lights_ctrl, command.pwmA, command.pwmB
        )
    except Exception as exc:  # pragma: no cover - hardware dependent
        LOGGER.error("Failed to control lights: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="lights_control_failed")

    return command


@app.post("/control/nod", response_model=NodCommand, tags=["Control"])
async def nod(
    command: NodCommand,
    x_control_token: str = Header(..., alias="x-control-token"),
    session_id: str = Header(..., alias="session-id"),
) -> NodCommand:
    base_controller = _get_base_controller()

    if not base_controller:
        raise HTTPException(status_code=503, detail="controller_unavailable")

    if not hasattr(base_controller, "nod"):
        raise HTTPException(status_code=501, detail="nod_not_supported")

    try:
        await anyio.to_thread.run_sync(
            base_controller.nod,
            command.times,
            command.center_tilt,
            command.delta,
            command.pan,
            command.delay,
        )
    except Exception as exc:  # pragma: no cover - hardware dependent
        LOGGER.error("Failed to execute nod command: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="nod_failed")

    return command


@app.get("/mode", response_model=ModeResponse, tags=["Connectivity"])
async def get_mode() -> ModeResponse:
    return ModeResponse(mode=Mode.ACCESS_POINT)


def _get_wifi_status() -> tuple[bool, Optional[str], Optional[str]]:
    """Get WiFi connection status, network name, and IP address.
    
    Returns:
        Tuple of (connected: bool, network_name: Optional[str], ip: Optional[str])
    """
    connected = False
    network_name = None
    ip_address = None
    
    # Try nmcli first (NetworkManager)
    try:
        # Check for active WiFi connection
        result = subprocess.run(
            ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(":")
                if len(parts) >= 4:
                    device = parts[0]
                    device_type = parts[1]
                    state = parts[2]
                    connection = parts[3]
                    # Check if it's a WiFi device and connected
                    if device_type == "wifi" and state == "connected" and connection:
                        network_name = connection
                        connected = True
                        # Get IP address for this device
                        ip_result = subprocess.run(
                            ["nmcli", "-t", "-f", "IP4.ADDRESS", "device", "show", device],
                            capture_output=True,
                            text=True,
                            timeout=1,
                        )
                        if ip_result.returncode == 0 and ip_result.stdout.strip():
                            # Parse format like "IP4.ADDRESS[1]:192.168.200.123/24"
                            output = ip_result.stdout.strip()
                            # Split by colon to get the IP part (after "IP4.ADDRESS[1]:")
                            if ":" in output:
                                ip_part = output.split(":", 1)[1]
                                # Split by "/" to get just the IP address
                                ip_address = ip_part.split("/")[0]
                            else:
                                # Fallback: try splitting by "/" directly
                                ip_address = output.split("/")[0]
                        break
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as exc:
        LOGGER.debug("nmcli not available or failed: %s", exc)
    
    # If nmcli didn't work, try iwconfig
    if not connected:
        try:
            result = subprocess.run(
                ["iwconfig"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "ESSID:" in line:
                        try:
                            essid_part = line.split("ESSID:")[1].strip()
                            if essid_part and essid_part != "off/any":
                                network_name = essid_part.strip('"')
                                connected = True
                        except (IndexError, ValueError):
                            pass
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as exc:
            LOGGER.debug("iwconfig not available or failed: %s", exc)
    
    # Get IP address from network interfaces if not already found
    if connected and not ip_address:
        try:
            # Try to get IP from common WiFi interfaces
            for interface in ["wlan0", "wlp2s0", "wlp3s0"]:
                try:
                    result = subprocess.run(
                        ["ip", "-4", "addr", "show", interface],
                        capture_output=True,
                        text=True,
                        timeout=1,
                    )
                    if result.returncode == 0:
                        for line in result.stdout.split("\n"):
                            if "inet " in line:
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    ip_address = parts[1].split("/")[0]
                                    break
                        if ip_address:
                            break
                except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                    continue
        except Exception as exc:
            LOGGER.debug("Failed to get IP from network interfaces: %s", exc)
    
    # Fallback: try socket to get default route IP
    if connected and not ip_address:
        try:
            # Connect to a remote address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
        except Exception as exc:
            LOGGER.debug("Failed to get IP via socket: %s", exc)
    
    return connected, network_name, ip_address


@app.get("/wifi/status", response_model=WiFiStatusResponse, tags=["Connectivity"])
async def get_wifi_status() -> WiFiStatusResponse:
    """Get WiFi connection status including connection state, network name, and IP address."""
    connected, network_name, ip_address = await anyio.to_thread.run_sync(_get_wifi_status)
    return WiFiStatusResponse(
        connected=connected,
        network_name=network_name,
        ip=ip_address,
    )


def _scan_wifi_networks() -> list[WiFiNetwork]:
    """Scan for available WiFi networks.
    
    Returns:
        List of WiFiNetwork objects
    """
    networks: list[WiFiNetwork] = []
    
    # Try nmcli first (NetworkManager)
    try:
        # Find WiFi device first
        device_result = subprocess.run(
            ["nmcli", "-t", "-f", "DEVICE,TYPE", "device", "status"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        wifi_device = None
        if device_result.returncode == 0:
            for line in device_result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(":")
                if len(parts) >= 2 and parts[1] == "wifi":
                    wifi_device = parts[0]
                    break
        
        # Trigger a scan on the WiFi device (or all devices if device not found)
        if wifi_device:
            scan_result = subprocess.run(
                ["nmcli", "device", "wifi", "rescan", wifi_device],
                capture_output=True,
                text=True,
                timeout=5,
            )
        else:
            scan_result = subprocess.run(
                ["nmcli", "device", "wifi", "rescan"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        # Don't fail if rescan fails, just use existing scan results
        # Wait a moment for scan to complete
        time.sleep(2)
        
        # Get list of ALL available networks (not just connected)
        result = subprocess.run(
            ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY,FREQ", "device", "wifi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(":")
                if len(parts) >= 1:
                    ssid = parts[0].strip()
                    # Skip empty SSIDs only
                    if not ssid or ssid == "--":
                        continue
                    
                    # Parse signal strength (0-100)
                    signal_strength = None
                    if len(parts) >= 2 and parts[1].strip() and parts[1].strip() != "--":
                        try:
                            signal_strength = int(parts[1].strip())
                        except (ValueError, IndexError):
                            pass
                    
                    # Parse security type
                    security = None
                    if len(parts) >= 3 and parts[2].strip() and parts[2].strip() != "--":
                        security = parts[2].strip()
                    
                    # Parse frequency (MHz)
                    frequency = None
                    if len(parts) >= 4 and parts[3].strip() and parts[3].strip() != "--":
                        try:
                            # nmcli returns frequency in MHz
                            frequency = float(parts[3].strip())
                        except (ValueError, IndexError):
                            pass
                    
                    networks.append(WiFiNetwork(
                        ssid=ssid,
                        signal_strength=signal_strength,
                        security=security,
                        frequency=frequency,
                    ))
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as exc:
        LOGGER.debug("nmcli scan not available or failed: %s", exc)
    
    # Also try iwlist to ensure we get all networks (iwlist is often more comprehensive)
    # This will add any additional networks not found by nmcli
    try:
        # Find WiFi interface
        wifi_interface = None
        for interface in ["wlan0", "wlp2s0", "wlp3s0"]:
            try:
                check_result = subprocess.run(
                    ["iwconfig", interface],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                if check_result.returncode == 0:
                    wifi_interface = interface
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                continue
        
        if wifi_interface:
            # Scan for networks
            scan_result = subprocess.run(
                ["iwlist", wifi_interface, "scan"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if scan_result.returncode == 0:
                current_ssid = None
                current_signal = None
                current_security = None
                current_freq = None
                
                for line in scan_result.stdout.split("\n"):
                    line = line.strip()
                    if "ESSID:" in line:
                        # Save previous network if exists
                        if current_ssid:
                            networks.append(WiFiNetwork(
                                ssid=current_ssid,
                                signal_strength=current_signal,
                                security=current_security,
                                frequency=current_freq,
                            ))
                        # Parse new SSID
                        try:
                            essid_part = line.split("ESSID:")[1].strip().strip('"')
                            if essid_part:
                                current_ssid = essid_part
                                current_signal = None
                                current_security = None
                                current_freq = None
                        except (IndexError, ValueError):
                            pass
                    elif "Signal level=" in line:
                        try:
                            # Parse signal level (usually in dBm, convert to percentage approximation)
                            signal_part = line.split("Signal level=")[1].split()[0]
                            # iwlist typically gives negative dBm values, convert roughly
                            if signal_part.replace("-", "").replace(".", "").isdigit():
                                dbm = float(signal_part)
                                # Rough conversion: -100dBm = 0%, -50dBm = 100%
                                signal_strength = max(0, min(100, int(2 * (dbm + 100))))
                                current_signal = signal_strength
                        except (ValueError, IndexError):
                            pass
                    elif "Encryption key:" in line:
                        if "on" in line.lower():
                            current_security = "WEP"  # Default, may be updated
                        else:
                            current_security = "Open"
                    elif "IEEE 802.11" in line or "WPA" in line or "WPA2" in line:
                        if "WPA2" in line:
                            current_security = "WPA2"
                        elif "WPA" in line:
                            current_security = "WPA"
                    elif "Frequency:" in line:
                        try:
                            freq_part = line.split("Frequency:")[1].split()[0]
                            # Convert GHz to MHz if needed
                            if "GHz" in line:
                                frequency = float(freq_part) * 1000
                            else:
                                frequency = float(freq_part)
                            current_freq = frequency
                        except (ValueError, IndexError):
                            pass
                
                # Add last network
                if current_ssid:
                    networks.append(WiFiNetwork(
                        ssid=current_ssid,
                        signal_strength=current_signal,
                        security=current_security,
                        frequency=current_freq,
                    ))
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as exc:
        LOGGER.debug("iwlist scan not available or failed: %s", exc)
    
    # Sort by signal strength (strongest first)
    networks.sort(key=lambda n: n.signal_strength if n.signal_strength is not None else -1, reverse=True)
    
    return networks


@app.get("/wifi/scan", response_model=WiFiScanResponse, tags=["Connectivity"])
async def scan_wifi_networks() -> WiFiScanResponse:
    """Scan for available WiFi networks and return a list of discovered networks."""
    networks = await anyio.to_thread.run_sync(_scan_wifi_networks)
    return WiFiScanResponse(networks=networks)


@app.post("/wifi/connect", response_model=WiFiConnectResponse, tags=["Connectivity"])
async def connect_wifi(request: WiFiConnectRequest) -> WiFiConnectResponse:
    if not request.password:
        raise HTTPException(status_code=400, detail="Password must not be empty")

    message = f"Attempting to connect to {request.ssid}"
    return WiFiConnectResponse(connecting=True, message=message)


@app.post("/claim/request", response_model=ClaimRequestResponse, tags=["Claim"])
async def claim_request() -> ClaimRequestResponse:
    """Generate a PIN code for claiming the robot. PIN is valid for ~120 seconds."""
    STATE["pin"] = f"{secrets.randbelow(10**6):06d}"
    STATE["pin_exp"] = time.time() + 120
    
    # Display PIN on OLED screen (try lazy initialization if not already available)
    base_controller = _get_base_controller()
    
    if base_controller:
        LOGGER.debug("base_controller found, attempting to display PIN")
        try:
            # Rover.display_text uses line numbers 0-3 (0=top, 3=bottom)
            # Using lines 2 and 3 (third and fourth lines) to match original request
            LOGGER.debug("Calling display_text(2, 'PIN Code:')")
            base_controller.display_text(2, "PIN Code:")
            LOGGER.debug("Calling display_text(3, '%s')", STATE["pin"])
            base_controller.display_text(3, STATE["pin"])
            LOGGER.info("PIN displayed on OLED: %s", STATE["pin"])
        except AttributeError as exc:
            LOGGER.error("base_controller missing display_text method: %s", exc, exc_info=True)
            app.state.base_controller = None
        except Exception as exc:
            LOGGER.error("Failed to display PIN on OLED: %s", exc, exc_info=True)
            # Mark as failed so we don't keep trying
            app.state.base_controller = None
    else:
        LOGGER.warning("OLED display not available; PIN generated but not displayed. Rover controller is None or not initialized.")

    _schedule_pin_reset(STATE["pin"], STATE["pin_exp"])

    LOGGER.info("Generated claim PIN (expires in 120s)")
    return ClaimRequestResponse(expiresIn=120)


@app.post("/claim/confirm", response_model=ClaimConfirmResponse, tags=["Claim"])
async def claim_confirm(request: ClaimConfirmRequest) -> ClaimConfirmResponse:
    """Confirm PIN and generate control token. Returns control token and robot ID."""
    if request.pin != STATE["pin"] or time.time() > STATE["pin_exp"] or STATE["claimed"]:
        raise HTTPException(status_code=400, detail="invalid_or_expired_pin")

    token = secrets.token_urlsafe(32)
    STATE["control_token_hash"] = hash_token(token)
    STATE["claimed"] = True
    
    # Reset OLED display when PIN is successfully used
    base_controller = _get_base_controller()
    if base_controller and hasattr(base_controller, "display_reset"):
        try:
            base_controller.display_reset()
            LOGGER.info("OLED display reset after successful PIN claim")
        except Exception as exc:
            LOGGER.error("Failed to reset OLED display after claim: %s", exc, exc_info=True)
    
    STATE["pin"] = None
    STATE["pin_exp"] = 0
    _cancel_pin_reset_task()
    
    LOGGER.info("Robot claimed successfully")
    return ClaimConfirmResponse(controlToken=token, robotId=ROBOT_SERIAL)


@app.post("/claim/release", tags=["Claim"])
async def claim_release() -> dict[str, bool]:
    """Release the claim and rotate the control token."""
    if not STATE["claimed"]:
        raise HTTPException(status_code=400, detail="not_claimed")

    # Rotate token
    new_token = secrets.token_urlsafe(32)
    STATE["control_token_hash"] = hash_token(new_token)
    STATE["claimed"] = False
    STATE["controller"]["sid"] = None
    STATE["controller"]["last"] = 0
    LOGGER.info("Robot claim released")
    return {"released": True}


@app.post("/claim-control", response_model=ClaimControlResponse, tags=["Claim"])
async def claim_control() -> ClaimControlResponse:
    """Claim a controller session. Returns session_id that must be used with control endpoints.
    
    Requires x-control-token header (verified by middleware).
    """
    # Generate new session ID
    session_id = secrets.token_urlsafe(16)
    STATE["controller"]["sid"] = session_id
    STATE["controller"]["last"] = time.time()
    LOGGER.info("Controller session claimed")
    return ClaimControlResponse(sessionId=session_id)
