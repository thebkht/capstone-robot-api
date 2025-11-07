from __future__ import annotations

import asyncio

import pytest

from app.camera import CameraError, CameraService, PlaceholderCameraSource


def test_camera_service_uses_placeholder_frame():
    placeholder = b"test-frame"
    service = CameraService(primary=None, fallback=PlaceholderCameraSource(placeholder), boundary="frame", frame_rate=0)

    frame = asyncio.run(service.get_frame())
    assert frame == placeholder

    async def _consume_stream() -> list[bytes]:
        return [chunk async for chunk in service.stream(max_frames=1)]

    chunks = asyncio.run(_consume_stream())
    assert len(chunks) == 1
    assert placeholder in chunks[0]


def test_camera_service_requires_source():
    with pytest.raises(ValueError):
        CameraService(primary=None, fallback=None)


def test_camera_service_raises_when_no_frame_available():
    service = CameraService(primary=None, fallback=PlaceholderCameraSource(b""), boundary="frame", frame_rate=0)

    with pytest.raises(CameraError):
        asyncio.run(service.get_frame())
