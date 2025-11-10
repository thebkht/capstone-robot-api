"""DepthAI MJPEG streaming helper inspired by the Luxonis example."""
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

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .camera import CameraError


class DepthAIStreamError(RuntimeError):
    """Raised when the DepthAI MJPEG runtime cannot be prepared."""

BOUNDARY = "frame"
_DEFAULT_PREVIEW_SIZE = (640, 480)
_DEFAULT_VIDEO_SIZE = (1920, 1080)
_DEFAULT_FPS = 30.0
_QUEUE_SIZE = 4
_STREAM_NAME = "mjpeg"

LOGGER = logging.getLogger("uvicorn.error").getChild(__name__)


@dataclass
class _StreamArgs:
    device: str | None
    media_path: Path | None
    fps_limit: float | None


@dataclass
class _PipelineState:
    pipeline: "dai.Pipeline"
    queue_factory: Callable[["dai.Device"], "dai.DataOutputQueue"]


@dataclass
class _RuntimeState:
    device: "dai.Device"
    queue: "dai.DataOutputQueue"


_lock = threading.Lock()
_pipeline_state: Optional[_PipelineState] = None
_runtime_state: Optional[_RuntimeState] = None


def initialize_argparser() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Match the Luxonis example argument parser using environment defaults."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default=os.getenv("DEPTHAI_DEVICE_ID"))
    parser.add_argument("--media-path", default=os.getenv("DEPTHAI_MEDIA_PATH"))
    parser.add_argument("--fps-limit", default=os.getenv("DEPTHAI_FPS_LIMIT"))

    # We do not expose command line parsing in the API runtime. Instead we
    # parse an empty list so that the defaults above are used verbatim.
    args = parser.parse_args([])
    return parser, args


_, _parsed_args = initialize_argparser()
_ARGS = _StreamArgs(
    device=_parsed_args.device,
    media_path=Path(_parsed_args.media_path).expanduser() if _parsed_args.media_path else None,
    fps_limit=float(_parsed_args.fps_limit) if _parsed_args.fps_limit else None,
)


def _require_depthai() -> None:
    if dai is None:  # pragma: no cover - environment dependent
        raise HTTPException(status_code=503, detail="DepthAI streaming dependencies are unavailable")


def _create_camera_node(pipeline: "dai.Pipeline") -> "dai.Node.Output":
    """Create a color camera preview output compatible with DepthAI 2.x/3.x."""

    camera_node: Optional[object] = None

    if hasattr(dai, "node") and hasattr(dai.node, "ColorCamera"):
        camera_node = pipeline.create(dai.node.ColorCamera)
    elif hasattr(dai, "node") and hasattr(dai.node, "Camera"):
        # DepthAI 3.x exposes a builder style API for the generic camera node.
        builder = pipeline.create(dai.node.Camera)
        if hasattr(builder, "build"):
            camera_node = builder.build()
    elif hasattr(pipeline, "createColorCamera"):
        camera_node = pipeline.createColorCamera()

    if camera_node is None:  # pragma: no cover - SDK specific behaviour
        raise HTTPException(status_code=503, detail="DepthAI SDK lacks ColorCamera support")

    if hasattr(camera_node, "setBoardSocket"):
        camera_node.setBoardSocket(getattr(dai.CameraBoardSocket, "RGB", None))
    preview_width, preview_height = _DEFAULT_PREVIEW_SIZE
    if hasattr(camera_node, "setPreviewSize"):
        camera_node.setPreviewSize(preview_width, preview_height)
    if hasattr(camera_node, "setVideoSize"):
        video_width, video_height = _DEFAULT_VIDEO_SIZE
        camera_node.setVideoSize(video_width, video_height)
    if hasattr(camera_node, "setResolution"):
        camera_node.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    if hasattr(camera_node, "setColorOrder"):
        camera_node.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    if hasattr(camera_node, "setInterleaved"):
        camera_node.setInterleaved(False)
    if hasattr(camera_node, "setFps"):
        camera_node.setFps(_ARGS.fps_limit or _DEFAULT_FPS)

    output = getattr(camera_node, "preview", None)
    if output is None:
        raise HTTPException(status_code=503, detail="DepthAI ColorCamera preview output unavailable")
    return output


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
        video_width, video_height = _DEFAULT_VIDEO_SIZE
        replay_node.setSize(video_width, video_height)

    output = getattr(replay_node, "out", None)
    if output is None:
        raise HTTPException(status_code=503, detail="DepthAI ReplayVideo node has no output port")
    return output


def _create_video_encoder(pipeline: "dai.Pipeline") -> "dai.node.VideoEncoder":
    encoder_node: Optional[object] = None

    if hasattr(dai, "node") and hasattr(dai.node, "VideoEncoder"):
        encoder_node = pipeline.create(dai.node.VideoEncoder)
    elif hasattr(pipeline, "createVideoEncoder"):
        encoder_node = pipeline.createVideoEncoder()

    if encoder_node is None:  # pragma: no cover - SDK specific behaviour
        raise HTTPException(status_code=503, detail="DepthAI SDK lacks VideoEncoder support")

    encoder_node.setDefaultProfilePreset(_ARGS.fps_limit or _DEFAULT_FPS, dai.VideoEncoderProperties.Profile.MJPEG)
    if hasattr(encoder_node, "setKeyframeFrequency"):
        encoder_node.setKeyframeFrequency(int(_ARGS.fps_limit or _DEFAULT_FPS))
    if hasattr(encoder_node, "setBitrate"):
        encoder_node.setBitrate(8_000_000)
    rate_control = getattr(dai.VideoEncoderProperties.RateControlMode, "CBR", None)
    if rate_control is not None:
        encoder_node.setRateControlMode(rate_control)
    return encoder_node


def _create_pipeline() -> _PipelineState:
    _require_depthai()

    pipeline = dai.Pipeline()

    if _ARGS.media_path is not None:
        preview_output = _create_replay_node(pipeline, _ARGS.media_path)
    else:
        preview_output = _create_camera_node(pipeline)

    encoder = _create_video_encoder(pipeline)
    preview_output.link(encoder.input)

    bitstream = getattr(encoder, "bitstream", None)
    xout = None

    if bitstream is None or not hasattr(bitstream, "createOutputQueue"):
        if hasattr(dai, "node") and hasattr(dai.node, "XLinkOut"):
            xout = pipeline.create(dai.node.XLinkOut)
        elif hasattr(pipeline, "createXLinkOut"):
            xout = pipeline.createXLinkOut()

        if xout is None:
            raise HTTPException(status_code=503, detail="DepthAI encoder outputs are unavailable")

        if hasattr(xout, "setStreamName"):
            xout.setStreamName(_STREAM_NAME)
        encoder.bitstream.link(xout.input)

    def _queue_factory(device: "dai.Device") -> "dai.DataOutputQueue":
        if bitstream is not None and hasattr(bitstream, "createOutputQueue"):
            return bitstream.createOutputQueue(maxSize=_QUEUE_SIZE, blocking=False)

        assert xout is not None

        if hasattr(device, "getOutputQueue"):
            return device.getOutputQueue(_STREAM_NAME, maxSize=_QUEUE_SIZE, blocking=False)

        out = getattr(xout, "out", None)
        if out is not None and hasattr(out, "createOutputQueue"):
            return out.createOutputQueue(maxSize=_QUEUE_SIZE, blocking=False)

        raise HTTPException(status_code=503, detail="DepthAI stream output queue unavailable")

    return _PipelineState(pipeline=pipeline, queue_factory=_queue_factory)


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


def _ensure_runtime() -> _RuntimeState:
    global _pipeline_state, _runtime_state

    if _runtime_state is not None:
        return _runtime_state

    with _lock:
        if _runtime_state is not None:
            return _runtime_state

        _pipeline_state = _create_pipeline()

        _force_release_devices()
        time.sleep(0.5)

        device: Optional["dai.Device"] = None
        queue: Optional["dai.DataOutputQueue"] = None

        attempts = 3
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                if _ARGS.device:
                    info = dai.DeviceInfo(_ARGS.device)
                    device = dai.Device(info)
                else:
                    device = dai.Device()

                device.startPipeline(_pipeline_state.pipeline)
                queue = _pipeline_state.queue_factory(device)
            except Exception as exc:  # pragma: no cover - hardware dependent
                last_error = exc
                if device is not None:
                    try:
                        device.close()
                    except Exception:
                        LOGGER.debug("Error closing DepthAI device after failure", exc_info=True)
                device = None
                queue = None
                if attempt < attempts:
                    LOGGER.info(
                        "Retrying DepthAI runtime startup after failure",
                        extra={"attempt": attempt, "error": str(exc)},
                    )
                    _force_release_devices()
                    time.sleep(0.5)
                    continue
                break
            else:
                last_error = None
                break

        if device is None or queue is None or last_error is not None:
            if last_error is not None:
                detail = f"Unable to start DepthAI stream: {last_error}"
            else:
                detail = "DepthAI queue unavailable"
            raise DepthAIStreamError(detail)

        _runtime_state = _RuntimeState(device=device, queue=queue)
        return _runtime_state


def _packet_to_jpeg(packet: "dai.ImgFrame") -> bytes | None:
    data = packet.getData()
    if data:
        return bytes(data)

    if not hasattr(packet, "getCvFrame"):
        return None

    if cv2 is None:  # pragma: no cover - optional dependency
        LOGGER.debug("Discarding frame; OpenCV not available to encode raw data")
        return None

    frame = packet.getCvFrame()
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return encoded.tobytes()


def _mjpeg_chunks(state: _RuntimeState) -> Generator[bytes, None, None]:
    queue = state.queue

    while True:
        try:
            packet = queue.get()
        except Exception:  # pragma: no cover - hardware dependent
            LOGGER.debug("Error receiving DepthAI packet; ending stream", exc_info=True)
            break

        payload = _packet_to_jpeg(packet)
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
        state = _ensure_runtime()
    except DepthAIStreamError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail=f"DepthAI stream unavailable: {exc}") from exc

    generator = _mjpeg_chunks(state)
    return StreamingResponse(generator, media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")


def get_snapshot() -> bytes:
    try:
        state = _ensure_runtime()
    except DepthAIStreamError as exc:
        raise CameraError(str(exc)) from exc

    queue = state.queue

    try:
        packet = queue.get()
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise CameraError(f"Unable to read DepthAI snapshot: {exc}") from exc

    payload = _packet_to_jpeg(packet)
    if not payload:
        raise CameraError("Empty DepthAI snapshot payload")
    return payload


def shutdown() -> None:
    global _runtime_state, _pipeline_state

    with _lock:
        state = _runtime_state
        _runtime_state = None
        _pipeline_state = None

    if state is None:
        return

    queue = state.queue
    device = state.device

    try:
        queue.close()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - best effort cleanup
        pass

    try:
        device.close()
    except Exception:  # pragma: no cover - best effort cleanup
        LOGGER.debug("Error closing DepthAI device during shutdown", exc_info=True)
