from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, conint, confloat


class Mode(str, Enum):
    ACCESS_POINT = "ap"
    WIFI = "wifi"


class HealthResponse(BaseModel):
    ok: bool = True
    name: str = Field(..., description="Robot hostname or identifier")
    serial: str = Field(..., description="Robot serial number")
    claimed: bool = Field(..., description="Whether the robot is currently claimed")
    mode: Mode = Field(..., description="Current connectivity mode")
    version: str = Field(..., description="Firmware/software version")


class NetworkInfoResponse(BaseModel):
    ip: str = Field(..., description="IPv4 address of the robot")
    ssid: Optional[str] = Field(None, description="Connected Wi-Fi SSID if in Wi-Fi mode")
    hostname: str = Field(..., description="Robot hostname")


class CaptureType(str, Enum):
    PHOTO = "photo"
    VIDEO = "video"


class CaptureRequest(BaseModel):
    type: CaptureType = Field(..., description="Type of capture to perform")


class CaptureResponse(BaseModel):
    saved: bool
    path: str
    url: str


class StatusSensors(BaseModel):
    air_quality: conint(ge=0) = 0
    humidity: conint(ge=0, le=100) = 0
    gas: conint(ge=0) = 0


class StatusResponse(BaseModel):
    battery: conint(ge=0, le=100) = 100
    cpu: conint(ge=0, le=100) = 0
    temp: confloat(ge=-40, le=125) = 25.0
    ai_state: str = "idle"
    sensors: StatusSensors = StatusSensors()


class MoveCommand(BaseModel):
    linear: float = Field(..., description="Linear velocity in meters per second")
    angular: float = Field(..., description="Angular velocity in radians per second")


class StopResponse(BaseModel):
    stopped: bool = True


class HeadCommand(BaseModel):
    pan: float = Field(..., description="Pan angle in degrees")
    tilt: float = Field(..., description="Tilt angle in degrees")


class LightCommand(BaseModel):
    pwmA: conint(ge=0, le=255) = Field(..., description="PWM level for IO4 (0-255)")
    pwmB: conint(ge=0, le=255) = Field(..., description="PWM level for IO5 (0-255)")


class NodCommand(BaseModel):
    times: conint(ge=1, le=20) = Field(3, description="Number of nod cycles to perform")
    center_tilt: conint(ge=0, le=180) = Field(
        15, description="Neutral tilt angle to center the nod motion"
    )
    delta: conint(ge=0, le=90) = Field(15, description="Tilt delta from neutral")
    pan: conint(ge=0, le=180) = Field(90, description="Pan angle to hold during nod")
    delay: confloat(ge=0, le=5) = Field(0.35, description="Delay between nod movements")


class ModeResponse(BaseModel):
    mode: Mode


class WiFiConnectRequest(BaseModel):
    ssid: str
    password: str


class WiFiConnectResponse(BaseModel):
    connecting: bool
    message: str


class WiFiStatusResponse(BaseModel):
    connected: bool = Field(..., description="Whether WiFi is connected")
    network_name: Optional[str] = Field(None, description="Name of the connected WiFi network (SSID)")
    ip: Optional[str] = Field(None, description="IP address of the device")


class WiFiNetwork(BaseModel):
    ssid: str = Field(..., description="Network SSID (name)")
    signal: Optional[int] = Field(None, description="Signal strength in percentage (0-100)")
    security: Optional[str] = Field(None, description="Security type (e.g., WPA2, WPA, WEP, Open)")
    frequency: Optional[float] = Field(None, description="Frequency in MHz")


class WiFiScanResponse(BaseModel):
    networks: list[WiFiNetwork] = Field(..., description="List of scanned WiFi networks")


class ClaimRequestResponse(BaseModel):
    expiresIn: int = Field(..., description="PIN expiration time in seconds")


class ClaimConfirmRequest(BaseModel):
    pin: str = Field(..., description="PIN code from claim request")


class ClaimConfirmResponse(BaseModel):
    controlToken: str = Field(..., description="Control token for authenticated requests")
    robotId: str = Field(..., description="Robot identifier")


class ClaimControlResponse(BaseModel):
    sessionId: str = Field(..., description="Controller session ID")


# Face Recognition Models
class FaceRecognitionResult(BaseModel):
    name: str = Field(..., description="Recognized person name or 'Unknown'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Recognition confidence score (0-1)")
    bbox: Optional[list[int]] = Field(None, description="Face bounding box [x1, y1, x2, y2]")


class FaceRecognitionResponse(BaseModel):
    faces: list[FaceRecognitionResult] = Field(..., description="List of recognized faces")
    frame_count: int = Field(1, description="Number of faces detected")


class AddFaceRequest(BaseModel):
    name: str = Field(..., description="Name of the person to add")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image (alternative to file upload)")


class AddFaceResponse(BaseModel):
    success: bool = Field(..., description="Whether face was successfully added")
    message: str = Field(..., description="Status message")
    name: str = Field(..., description="Name of the added face")


class KnownFacesResponse(BaseModel):
    faces: list[str] = Field(..., description="List of known face names")
    count: int = Field(..., description="Number of known faces")
