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

To launch the service automatically on your Jetson at boot, install the provided systemd unit from `scripts/capstone-robot-api.service`:

1. Copy the repository to the target location (for example `/opt/capstone-robot-api`).
2. Adjust the `User`, `WorkingDirectory`, and optional `Environment=` lines in `scripts/capstone-robot-api.service` to match your setup.
3. Install the unit and enable it:

   ```bash
   sudo cp scripts/capstone-robot-api.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now capstone-robot-api.service
   ```

You can confirm it started successfully with `systemctl status capstone-robot-api.service` and inspect logs via `journalctl -u capstone-robot-api.service`.

### Testing

Run the automated tests with pytest:

```bash
pytest
```

## Endpoints Overview

- `GET /health` – returns robot connectivity and firmware metadata.
- `GET /network-info` – exposes current network IP/SSID information.
- `GET /camera/snapshot` – serves an inline placeholder JPEG snapshot.
- `GET /camera/stream` – provides an MJPEG stream using placeholder frames.
- `POST /camera/capture` – simulates capturing a photo and returning saved metadata.
- `GET /status` – reports mock telemetry, battery, and AI state data.
- `POST /control/move` – echoes requested linear/angular velocities.
- `POST /control/stop` – confirms a stop command.
- `POST /control/head` – echoes requested pan/tilt angles.
- `GET /mode` – reports the current connectivity mode.
- `POST /wifi/connect` – simulates initiating a Wi-Fi connection.

## Development Notes

- Camera endpoints currently serve an in-memory placeholder image; replace `_PLACEHOLDER_JPEG` in `app/main.py` with live camera integration when hardware is available.
- Update the data returned by telemetry and discovery endpoints to reflect real robot values as sensors and subsystems come online.
