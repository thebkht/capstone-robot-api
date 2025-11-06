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

To launch the service automatically on your Jetson at boot, point a systemd unit (or another init mechanism) at `scripts/autorun.sh`.
A sample systemd unit might look like:

```ini
[Unit]
Description=Capstone Robot API
After=network-online.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/opt/capstone-robot-api
ExecStart=/opt/capstone-robot-api/scripts/autorun.sh
Restart=on-failure
Environment=HOST=0.0.0.0
Environment=PORT=8000

[Install]
WantedBy=multi-user.target
```

Enable the unit with `sudo systemctl enable --now capstone-robot-api.service` after copying the repository to the target location.

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
