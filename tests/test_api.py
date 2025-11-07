from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["name"] == "rover-01"
    assert payload["mode"] == "ap"
    assert payload["version"] == "0.1.0"


def test_network_info_endpoint():
    response = client.get("/network-info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ip"] == "192.168.4.1"
    assert payload["hostname"] == "rover-01"


def test_camera_snapshot():
    response = client.get("/camera/snapshot")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")


def test_camera_capture_photo():
    response = client.post("/camera/capture", json={"type": "photo"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["saved"] is True
    assert payload["path"].endswith(".jpg")
    assert payload["url"].startswith("http://192.168.4.1:8000")


def test_status_endpoint():
    response = client.get("/status")
    assert response.status_code == 200
    payload = response.json()
    assert "battery" in payload
    assert "sensors" in payload


def test_control_move_echoes_payload():
    body = {"linear": 0.5, "angular": -0.1}
    response = client.post("/control/move", json=body)
    assert response.status_code == 200
    assert response.json() == body


def test_control_stop():
    response = client.post("/control/stop")
    assert response.status_code == 200
    assert response.json() == {"stopped": True}


def test_control_head():
    body = {"pan": 15, "tilt": -5}
    response = client.post("/control/head", json=body)
    assert response.status_code == 200
    assert response.json() == body


def test_mode_endpoint():
    response = client.get("/mode")
    assert response.status_code == 200
    assert response.json() == {"mode": "ap"}


def test_wifi_connect():
    response = client.post("/wifi/connect", json={"ssid": "HomeWiFi", "password": "secret"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["connecting"] is True
    assert "Attempting to connect" in payload["message"]


def test_camera_stream_chunk():
    with TestClient(app) as streaming_client:
        response = streaming_client.get("/camera/stream", params={"frames": 1})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert b"--frame" in response.content
