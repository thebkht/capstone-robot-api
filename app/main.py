from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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


async def placeholder_stream(delay_seconds: float = 0.5) -> AsyncIterator[bytes]:
    if not _PLACEHOLDER_JPEG:
        raise HTTPException(status_code=500, detail="Camera stream unavailable")

    frame = _PLACEHOLDER_JPEG
    header = (
        f"--{BOUNDARY}\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(frame)}\r\n\r\n"
    ).encode()

    while True:
        yield header + frame + b"\r\n"
        await asyncio.sleep(delay_seconds)


@app.get("/health", response_model=HealthResponse, tags=["Discovery"])
async def get_health() -> HealthResponse:
    return HealthResponse(ok=True, name=ROBOT_NAME, mode=Mode.ACCESS_POINT, version=APP_VERSION)


@app.get("/network-info", response_model=NetworkInfoResponse, tags=["Discovery"])
async def get_network_info() -> NetworkInfoResponse:
    return NetworkInfoResponse(ip="192.168.4.1", ssid=None, hostname=ROBOT_NAME)


@app.get("/camera/snapshot", tags=["Camera"])
async def get_camera_snapshot() -> Response:
    if not _PLACEHOLDER_JPEG:
        raise HTTPException(status_code=404, detail="Snapshot unavailable")

    headers = {"Content-Disposition": "inline; filename=snapshot.jpg"}
    return Response(content=_PLACEHOLDER_JPEG, media_type="image/jpeg", headers=headers)


@app.get("/camera/stream", tags=["Camera"])
async def get_camera_stream() -> StreamingResponse:
    stream = placeholder_stream()
    return StreamingResponse(stream, media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")


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
