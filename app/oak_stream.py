"""DepthAI MJPEG streaming helpers aligned with Luxonis' mjpeg-streaming example."""
from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Optional

try:  # pragma: no cover - optional dependency
    import depthai as dai
except Exception:  # pragma: no cover - optional dependency
    dai = None  # type: ignore[assignment]

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .camera import CameraError

BOUNDARY = "frame"
_QUEUE_MAX_SIZE = 4
_STREAM_NAME = "mjpeg"
_DEFAULT_PREVIEW_SIZE = (640, 480)
_DEFAULT_VIDEO_SIZE = (1920, 1080)
_DEFAULT_FPS = 30.0

LOGGER = logging.getLogger("uvicorn.error").getChild(__name__)


@dataclass
class _StreamArgs:
    device: str | None
    media_path: Path | None
    fps_limit: float | None


@dataclass
class _StreamPipeline:
    pipeline: "dai.Pipeline"
    queue_factory: Callable[["dai.Device"], "dai.DataOutputQueue"]


_lock = threading.Lock()
_device: Optional["dai.Device"] = None
_queue: Optional["dai.DataOutputQueue"] = None
_pipeline: Optional[_StreamPipeline] = None


def initialize_argparser() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Mirror the Luxonis example argument parser using environment defaults."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default=os.getenv("DEPTHAI_DEVICE_ID"))
    parser.add_argument("--media-path", default=os.getenv("DEPTHAI_MEDIA_PATH"))
    parser.add_argument(
        "--fps-limit",
        type=float,
        default=os.getenv("DEPTHAI_FPS_LIMIT"),
    )
    args = parser.parse_args([])
    return parser, args


_, _parsed_args = initialize_argparser()
_ARGS = _StreamArgs(
    device=_parsed_args.device,
    media_path=Path(_parsed_args.media_path).expanduser() if _parsed_args.media_path else None,
    fps_limit=float(_parsed_args.fps_limit) if _parsed_args.fps_limit else None,
)


class MJPEGStreamer:
    """Configure an MJPEG encoder output for the supplied input stream."""

    def __init__(self, pipeline: "dai.Pipeline") -> None:
        self._pipeline = pipeline

    def build(
        self, *, preview: "dai.Node.Output"
    ) -> Callable[["dai.Device"], "dai.DataOutputQueue"]:
        encoder = _create_video_encoder(self._pipeline)
        preview.link(encoder.input)

        try:
            xout = _create_xlink_out(self._pipeline)
        except HTTPException as exc:
            LOGGER.info(
                "DepthAI XLinkOut unavailable; falling back to encoder bitstream queue",
                extra={"reason": exc.detail},
            )

            def _direct_queue(device: "dai.Device") -> "dai.DataOutputQueue":
                bitstream = getattr(encoder, "bitstream", None)
                if bitstream is None or not hasattr(bitstream, "createOutputQueue"):
                    raise HTTPException(
                        status_code=503,
                        detail="DepthAI encoder bitstream queue unavailable",
                    )
                try:
                    return bitstream.createOutputQueue(maxSize=_QUEUE_MAX_SIZE, blocking=False)
                except Exception as queue_exc:  # pragma: no cover - hardware dependent
                    raise HTTPException(
                        status_code=503,
                        detail=f"Unable to create encoder output queue: {queue_exc}",
                    ) from queue_exc

            return _direct_queue

        encoder.bitstream.link(xout.input)

        def _xlink_queue(device: "dai.Device") -> "dai.DataOutputQueue":
            if hasattr(device, "getOutputQueue"):
                return device.getOutputQueue(name=_STREAM_NAME, maxSize=_QUEUE_MAX_SIZE, blocking=False)
            out = getattr(xout, "out", None)
            if out is not None and hasattr(out, "createOutputQueue"):
                return out.createOutputQueue(maxSize=_QUEUE_MAX_SIZE, blocking=False)
            raise HTTPException(status_code=503, detail="DepthAI XLinkOut queue APIs unavailable")

        return _xlink_queue


def _require_depthai() -> None:
    if dai is None:  # pragma: no cover - environment dependent
        raise HTTPException(status_code=503, detail="DepthAI streaming dependencies are unavailable")


def _create_pipeline() -> _StreamPipeline:
    _require_depthai()

    pipeline = dai.Pipeline()

    if _ARGS.media_path is not None:
        preview = _create_replay_node(pipeline, _ARGS.media_path)
    else:
        preview = _create_color_camera(pipeline)

    streamer = MJPEGStreamer(pipeline)
    queue_factory = streamer.build(preview=preview)

    return _StreamPipeline(pipeline=pipeline, queue_factory=queue_factory)


def _create_replay_node(pipeline: "dai.Pipeline", media_path: Path) -> "dai.Node.Output":
    if not media_path.exists():
        raise HTTPException(status_code=503, detail=f"Replay file not found: {media_path}")

    replay_node: Optional[object] = None
    if hasattr(dai, "node") and hasattr(dai.node, "ReplayVideo"):
        replay_node = pipeline.create(dai.node.ReplayVideo)
    elif hasattr(dai, "node") and hasattr(dai.node, "Replay"):
        replay_node = pipeline.create(dai.node.Replay)
    elif hasattr(pipeline, "createReplay"):
        replay_node = pipeline.createReplay()

    if replay_node is None:  # pragma: no cover - SDK specific behaviour
        raise HTTPException(status_code=503, detail="DepthAI SDK does not provide ReplayVideo nodes")

    if hasattr(replay_node, "setReplayVideoFile"):
        replay_node.setReplayVideoFile(media_path)
    if hasattr(replay_node, "setOutFrameType"):
        replay_node.setOutFrameType(dai.ImgFrame.Type.NV12)
    if hasattr(replay_node, "setLoop"):
        replay_node.setLoop(True)
    if _ARGS.fps_limit and hasattr(replay_node, "setFps"):
        replay_node.setFps(_ARGS.fps_limit)
    elif _ARGS.fps_limit and hasattr(replay_node, "setFpsLimit"):
        replay_node.setFpsLimit(_ARGS.fps_limit)
    if hasattr(replay_node, "setSize"):
        width, height = _DEFAULT_VIDEO_SIZE
        replay_node.setSize(width, height)

    output = getattr(replay_node, "out", None)
    if output is None:
        raise HTTPException(status_code=503, detail="DepthAI ReplayVideo node has no output port")
    return output


def _create_color_camera(pipeline: "dai.Pipeline") -> "dai.Node.Output":
    if hasattr(dai, "node") and hasattr(dai.node, "ColorCamera"):
        camera = pipeline.create(dai.node.ColorCamera)
    elif hasattr(pipeline, "createColorCamera"):
        camera = pipeline.createColorCamera()
    else:  # pragma: no cover - SDK specific behaviour
        raise HTTPException(status_code=503, detail="DepthAI SDK lacks ColorCamera support")

    camera.setBoardSocket(dai.CameraBoardSocket.RGB)
    preview_width, preview_height = _DEFAULT_PREVIEW_SIZE
    camera.setPreviewSize(preview_width, preview_height)
    if hasattr(camera, "setVideoSize"):
        video_width, video_height = _DEFAULT_VIDEO_SIZE
        camera.setVideoSize(video_width, video_height)
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camera.setInterleaved(False)
    camera.setFps(_DEFAULT_FPS)

    output = getattr(camera, "preview", None)
    if output is None:
        raise HTTPException(status_code=503, detail="DepthAI ColorCamera preview output unavailable")
    return output


def _create_video_encoder(pipeline: "dai.Pipeline") -> "dai.node.VideoEncoder":
    if hasattr(dai, "node") and hasattr(dai.node, "VideoEncoder"):
        encoder = pipeline.create(dai.node.VideoEncoder)
    elif hasattr(pipeline, "createVideoEncoder"):
        encoder = pipeline.createVideoEncoder()
    else:  # pragma: no cover - SDK specific behaviour
        raise HTTPException(status_code=503, detail="DepthAI SDK lacks VideoEncoder support")

    encoder.setDefaultProfilePreset(_DEFAULT_FPS, dai.VideoEncoderProperties.Profile.MJPEG)
    if hasattr(encoder, "setKeyframeFrequency"):
        encoder.setKeyframeFrequency(int(_DEFAULT_FPS))
    if hasattr(encoder, "setBitrate"):
        encoder.setBitrate(8_000_000)
    rate_control = getattr(dai.VideoEncoderProperties.RateControlMode, "CBR", None)
    if rate_control is not None:
        encoder.setRateControlMode(rate_control)
    return encoder


def _create_xlink_out(pipeline: "dai.Pipeline") -> "dai.node.XLinkOut":
    if hasattr(dai, "node") and hasattr(dai.node, "XLinkOut"):
        xout = pipeline.create(dai.node.XLinkOut)
    elif hasattr(pipeline, "createXLinkOut"):
        xout = pipeline.createXLinkOut()
    else:  # pragma: no cover - SDK missing XLinkOut
        raise HTTPException(status_code=503, detail="DepthAI SDK does not expose XLinkOut nodes")

    xout.setStreamName(_STREAM_NAME)
    return xout


def _force_release_devices() -> None:
    if dai is None:
        return

    try:  # pragma: no cover - hardware dependent cleanup
        import gc

        for dev_info in dai.Device.getAllAvailableDevices():
            try:
                temp = dai.Device(dev_info)
            except Exception:
                continue
            try:
                temp.close()
            finally:
                del temp
        gc.collect()
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        LOGGER.debug("DepthAI force-release skipped", exc_info=exc)


def _ensure_queue() -> "dai.DataOutputQueue":
    global _pipeline, _device, _queue

    if _queue is not None:
        return _queue

    with _lock:
        if _queue is None:
            _pipeline = _create_pipeline()
            _force_release_devices()
            time.sleep(0.5)

            device: Optional["dai.Device"] = None
            queue: Optional["dai.DataOutputQueue"] = None
            try:
                if _ARGS.device:
                    info = dai.DeviceInfo(_ARGS.device)
                    device = dai.Device(info)
                else:
                    device = dai.Device()
                device.startPipeline(_pipeline.pipeline)
                queue = _pipeline.queue_factory(device)
            except Exception as exc:  # pragma: no cover - hardware dependent
                if device is not None:
                    try:
                        device.close()
                    except Exception:  # pragma: no cover - defensive cleanup
                        LOGGER.debug("Error closing DepthAI device after failure", exc_info=True)
                raise HTTPException(status_code=503, detail=f"Unable to start DepthAI stream: {exc}") from exc

            _device = device
            _queue = queue

    assert _queue is not None
    return _queue


def _packet_to_bytes(packet: "dai.ImgFrame") -> bytes | None:
    data = packet.getData()
    if not data:
        return None
    return bytes(data)


def _mjpeg_chunks() -> Generator[bytes, None, None]:
    queue = _ensure_queue()

    while True:
        try:
            packet = queue.get()
        except Exception:  # pragma: no cover - hardware dependent
            continue

        payload = _packet_to_bytes(packet)
        if not payload:
            continue

        header = (
            f"--{BOUNDARY}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(payload)}\r\n\r\n"
        ).encode()
        yield header + payload + b"\r\n"


def get_video_response() -> StreamingResponse:
    try:
        generator = _mjpeg_chunks()
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail=f"DepthAI stream unavailable: {exc}") from exc

    return StreamingResponse(generator, media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")


def get_snapshot() -> bytes:
    queue = _ensure_queue()

    try:
        packet = queue.get()
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise CameraError(f"Unable to read DepthAI snapshot: {exc}") from exc

    payload = _packet_to_bytes(packet)
    if not payload:
        raise CameraError("Empty DepthAI snapshot payload")
    return payload


def shutdown() -> None:
    global _device, _queue, _pipeline

    with _lock:
        queue = _queue
        device = _device
        _queue = None
        _device = None
        _pipeline = None

    if queue is not None:
        try:
            queue.close()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - queue cleanup best effort
            pass

    if device is not None:
        try:
            device.close()
        except Exception:  # pragma: no cover - best effort cleanup
            LOGGER.debug("Error closing DepthAI device during shutdown", exc_info=True)

