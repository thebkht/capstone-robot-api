from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import AsyncIterator

import anyio
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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
from .models import (
    CaptureRequest,
    CaptureResponse,
    CaptureType,
    HeadCommand,
    HealthResponse,
    Mode,
    ModeResponse,
    MoveCommand,
    NetworkInfoResponse,
    StatusResponse,
    StopResponse,
    WiFiConnectRequest,
    WiFiConnectResponse,
)

APP_NAME = "capstone-robot-api"
APP_VERSION = "0.1.0"
ROBOT_NAME = "rover-01"
BOUNDARY = "frame"

LOGGER = logging.getLogger("uvicorn.error").getChild(__name__)

_PLACEHOLDER_JPEG = base64.b64decode(
    """
/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDwwMDw8NDhERExUTGBonHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8f/2wBDARESEhgVGBoZGB4dHy8fLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8v/3QAEAA3/2gAIAQEAAD8A/wD/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAsf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/AR//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/AR//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Ar//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IX//2Q==
""".strip()
)
app = FastAPI(title="Capstone Robot API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    return HealthResponse(ok=True, name=ROBOT_NAME, mode=Mode.ACCESS_POINT, version=APP_VERSION)


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


@app.get("/status", response_model=StatusResponse, tags=["Status"])
async def get_status() -> StatusResponse:
    return StatusResponse(battery=82, cpu=37, temp=46.3, ai_state="idle")


@app.post("/control/move", response_model=MoveCommand, tags=["Control"])
async def move_robot(command: MoveCommand) -> MoveCommand:
    return command


@app.post("/control/stop", response_model=StopResponse, tags=["Control"])
async def stop_robot() -> StopResponse:
    return StopResponse()


@app.post("/control/head", response_model=HeadCommand, tags=["Control"])
async def move_head(command: HeadCommand) -> HeadCommand:
    return command


@app.get("/mode", response_model=ModeResponse, tags=["Connectivity"])
async def get_mode() -> ModeResponse:
    return ModeResponse(mode=Mode.ACCESS_POINT)


@app.post("/wifi/connect", response_model=WiFiConnectResponse, tags=["Connectivity"])
async def connect_wifi(request: WiFiConnectRequest) -> WiFiConnectResponse:
    if not request.password:
        raise HTTPException(status_code=400, detail="Password must not be empty")

    message = f"Attempting to connect to {request.ssid}"
    return WiFiConnectResponse(connecting=True, message=message)
