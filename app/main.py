from __future__ import annotations

import base64
import logging
from datetime import datetime
from typing import AsyncIterator

import anyio
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .camera import (
    CameraError,
    CameraService,
    OpenCVCameraSource,
    PlaceholderCameraSource,
)
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

LOGGER = logging.getLogger(__name__)

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


def _create_camera_service() -> CameraService:
    primary_source = None
    try:
        primary_source = OpenCVCameraSource()
    except CameraError as exc:
        LOGGER.warning("OpenCV camera source unavailable: %s", exc)

    fallback_source = None
    if _PLACEHOLDER_JPEG:
        fallback_source = PlaceholderCameraSource(_PLACEHOLDER_JPEG)

    if primary_source is None and fallback_source is None:
        raise RuntimeError("No camera source available for streaming")

    return CameraService(primary_source, fallback=fallback_source, boundary=BOUNDARY, frame_rate=10.0)


app.state.camera_service = _create_camera_service()


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
        try:
            async for chunk in _camera_stream(app.state.camera_service, frames):
                yield chunk
        except CameraError as exc:
            LOGGER.error("Camera stream interrupted: %s", exc)
            raise HTTPException(status_code=503, detail="Camera stream unavailable") from exc

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
