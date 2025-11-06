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


class ModeResponse(BaseModel):
    mode: Mode


class WiFiConnectRequest(BaseModel):
    ssid: str
    password: str


class WiFiConnectResponse(BaseModel):
    connecting: bool
    message: str
