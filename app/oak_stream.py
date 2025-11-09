"""Minimal FastAPI app that streams OAK-D frames as MJPEG."""
from __future__ import annotations

import logging
from typing import AsyncIterator

import anyio
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse

from .camera import CameraError, DepthAICameraSource

LOGGER = logging.getLogger("uvicorn.error").getChild(__name__)

BOUNDARY = "frame"
FRAME_RATE = 30.0
FRAME_DELAY = 1.0 / FRAME_RATE

app = FastAPI(title="OAK-D Lite Video Stream")


async def _get_source() -> DepthAICameraSource:
    source: DepthAICameraSource | None = getattr(app.state, "source", None)
    if source is None:
        raise HTTPException(status_code=503, detail="DepthAI camera unavailable")
    return source


@app.on_event("startup")
async def startup() -> None:
    if not DepthAICameraSource.is_available():
        raise RuntimeError(
            "DepthAI package not installed. Install the 'depthai' package to stream from the OAK-D.",
        )

    LOGGER.info("Starting DepthAI camera for MJPEG streaming")
    app.state.source = DepthAICameraSource(preview_width=640, preview_height=360, fps=FRAME_RATE)


@app.on_event("shutdown")
async def shutdown() -> None:
    source: DepthAICameraSource | None = getattr(app.state, "source", None)
    if source is None:
        return

    LOGGER.info("Stopping DepthAI camera stream")
    try:
        await source.close()
    finally:
        app.state.source = None


async def _mjpeg_chunks(source: DepthAICameraSource) -> AsyncIterator[bytes]:
    while True:
        try:
            frame = await source.get_jpeg_frame()
        except CameraError as exc:
            LOGGER.error("DepthAI frame capture failed: %s", exc)
            await anyio.sleep(FRAME_DELAY)
            continue

        headers = (
            f"--{BOUNDARY}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(frame)}\r\n\r\n"
        ).encode()
        yield headers + frame + b"\r\n"
        await anyio.sleep(FRAME_DELAY)


@app.get("/")
async def root() -> dict[str, object]:
    return {"status": "ok", "endpoints": ["/video", "/shot"]}


@app.get("/video")
async def video_stream() -> StreamingResponse:
    source = await _get_source()

    async def generator() -> AsyncIterator[bytes]:
        async for chunk in _mjpeg_chunks(source):
            yield chunk

    return StreamingResponse(
        generator(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )


@app.get("/shot")
async def single_frame() -> Response:
    source = await _get_source()
    try:
        frame = await source.get_jpeg_frame()
    except CameraError as exc:
        raise HTTPException(status_code=503, detail="Snapshot unavailable") from exc

    return Response(content=frame, media_type="image/jpeg")
