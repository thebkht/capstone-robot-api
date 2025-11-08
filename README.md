# Capstone Robot API

A FastAPI service that exposes discovery, telemetry, camera, and control endpoints for a Jetson-powered rover.

## Features

- Discovery endpoints (`/health`, `/network-info`) for mobile app connectivity checks.
- Camera endpoints for MJPEG streaming, single snapshots, and capture triggers.
- Status telemetry endpoint with sample sensor readings.
- Control endpoints for motion, stop, and camera head positioning.
- Connectivity endpoints for querying and initiating Wi-Fi connections.

## Getting Started

### Installation

Use the provided setup script to create a virtual environment and install the API along with its development dependencies:

```bash
./scripts/setup.sh
```

### Running the API

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

By default, the service will be available at `http://127.0.0.1:8000` (or `http://192.168.4.1:8000` when running on the robot hotspot).

#### Forcing USB webcam mode

If you only need MJPEG streaming from a USB camera (including the OAK-D operating in UVC mode) set the following environment variables in your shell or systemd unit before starting the service:

```bash
export CAMERA_FORCE_WEBCAM=1          # Skip DepthAI initialisation entirely
export CAMERA_WEBCAM_DEVICE=/dev/video2  # Optional: explicit device path or index
```

When `CAMERA_FORCE_WEBCAM` is enabled the API ignores the DepthAI SDK and opens the camera just like a standard webcam via OpenCV. When no device override is provided the service scans `/dev/v4l/by-id` for entries containing `oak`, `depthai`, or `luxonis` so an attached OAK-D Lite in UVC mode is automatically preferred before falling back to generic `/dev/video*` indices.

To launch the service automatically on your Jetson at boot, install the provided systemd unit from `scripts/capstone-robot-api.service`:

1. Copy the repository to the target location (for example `/home/jetson/capstone-robot-api`).
2. Adjust the `User`, `WorkingDirectory`, and optional `Environment=` lines in `scripts/capstone-robot-api.service` to match your setup.
3. Install the unit and enable it:

   ```bash
   sudo cp scripts/capstone-robot-api.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now capstone-robot-api.service
   ```

You can confirm it started successfully with `systemctl status capstone-robot-api.service` and inspect logs via `journalctl -u capstone-robot-api.service`.

### Updating a running deployment

Once the service is installed and enabled you can deploy new versions with the following workflow:

1. Copy the updated repository (or run `git pull`) on the Jetson.
2. Run `./scripts/setup.sh` if you introduced new dependencies.
3. Restart the service to pick up the changes:

   ```bash
   sudo systemctl restart capstone-robot-api.service
   ```

`scripts/autorun.sh` honours the `LOG_LEVEL` and `LOG_CONFIG` environment variables if you need to adjust logging for troubleshooting. The defaults enable the packaged `scripts/logging.ini`, which surfaces INFO-level camera diagnostics in `journalctl`.

### Testing

Run the automated tests with pytest:

```bash
pytest
```

## Endpoints Overview

- `GET /health` – returns robot connectivity and firmware metadata.
- `GET /network-info` – exposes current network IP/SSID information.
- `GET /camera/snapshot` – serves an inline JPEG snapshot from the live camera when available, otherwise the placeholder image.
- `GET /camera/stream` – provides an MJPEG stream sourced from the DepthAI camera when present, falling back to the placeholder frames.
- `POST /camera/capture` – simulates capturing a photo and returning saved metadata.
- `GET /status` – reports mock telemetry, battery, and AI state data.
- `POST /control/move` – echoes requested linear/angular velocities.
- `POST /control/stop` – confirms a stop command.
- `POST /control/head` – echoes requested pan/tilt angles.
- `GET /mode` – reports the current connectivity mode.
- `POST /wifi/connect` – simulates initiating a Wi-Fi connection.

## Development Notes

- Camera endpoints automatically prefer an attached OAK-D/DepthAI device unless `CAMERA_FORCE_WEBCAM=1` is set. In webcam mode the service captures frames from the specified USB device (default `0`) using OpenCV.
- Update the data returned by telemetry and discovery endpoints to reflect real robot values as sensors and subsystems come online.
