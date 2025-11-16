#!/usr/bin/env python3
"""BLE-based Wi-Fi provisioning service for Jetson robots.

This script exposes a custom BLE GATT service that allows a companion
application to provision Wi-Fi credentials via Bluetooth Low Energy.  It is
intended to be launched as a long-running process (e.g., via systemd) and does
not provide any HTTP or FastAPI interfaces.
"""
from __future__ import annotations

import json
import logging
import signal
import subprocess
import sys
import threading
import uuid
from typing import Callable, Dict, List, Optional

import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib

BLUEZ_SERVICE_NAME = "org.bluez"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
LE_ADVERTISING_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
ADAPTER_IFACE = "org.bluez.Adapter1"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"
LE_ADVERTISEMENT_IFACE = "org.bluez.LEAdvertisement1"

WIFI_SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
WIFI_CONFIG_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"
WIFI_STATUS_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef2"

STATUS_IDLE = "idle"
STATUS_CONNECTING = "connecting"
STATUS_CONNECTED = "connected"
STATUS_FAILED = "failed"


class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.freedesktop.DBus.Error.InvalidArgs"


class NotSupportedException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.bluez.Error.NotSupported"


class Application(dbus.service.Object):
    """GATT application container."""

    def __init__(self, bus: dbus.SystemBus):
        self.path = "/org/rovy/wifi"
        self.bus = bus
        self.services: List[Service] = []
        super().__init__(bus, self.path)

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def add_service(self, service: "Service") -> None:
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
    def GetManagedObjects(self) -> Dict[str, Dict[str, Dict[str, object]]]:
        response: Dict[str, Dict[str, Dict[str, object]]] = {}
        for service in self.services:
            response[service.get_path()] = service.get_properties()
            for chrc in service.characteristics:
                response[chrc.get_path()] = chrc.get_properties()
        return response


class Service(dbus.service.Object):
    """Base class for GATT services."""

    PATH_BASE = "/org/rovy/wifi/service"

    def __init__(self, bus: dbus.SystemBus, index: int, uuid: str, primary: bool):
        self.path = f"{self.PATH_BASE}{index}"
        self.bus = bus
        self.uuid = uuid
        self.primary = primary
        self.characteristics: List[Characteristic] = []
        super().__init__(bus, self.path)

    def get_properties(self) -> Dict[str, Dict[str, object]]:
        return {
            GATT_SERVICE_IFACE: {
                "UUID": self.uuid,
                "Primary": self.primary,
                "Characteristics": dbus.Array(
                    [chrc.get_path() for chrc in self.characteristics],
                    signature="o",
                ),
            }
        }

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, characteristic: "Characteristic") -> None:
        self.characteristics.append(characteristic)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface: str) -> Dict[str, object]:
        if interface != GATT_SERVICE_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_SERVICE_IFACE]


class Characteristic(dbus.service.Object):
    """Base class for GATT characteristics."""

    def __init__(
        self,
        bus: dbus.SystemBus,
        index: int,
        uuid: str,
        flags: List[str],
        service: Service,
    ):
        self.path = f"{service.get_path()}/char{index}"
        self.bus = bus
        self.uuid = uuid
        self.service = service
        self.flags = flags
        self.value: List[int] = []
        self.notifying = False
        super().__init__(bus, self.path)

    def get_properties(self) -> Dict[str, Dict[str, object]]:
        return {
            GATT_CHRC_IFACE: {
                "Service": self.service.get_path(),
                "UUID": self.uuid,
                "Flags": self.flags,
            }
        }

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface: str) -> Dict[str, object]:
        if interface != GATT_CHRC_IFACE:
            raise InvalidArgsException()
        props = self.get_properties()[GATT_CHRC_IFACE]
        if "read" in self.flags:
            props["Value"] = dbus.Array(self.value, signature="y")
        return props

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options: Dict[str, object]) -> dbus.Array:
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value: List[int], options: Dict[str, object]) -> None:
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE)
    def StartNotify(self) -> None:
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE)
    def StopNotify(self) -> None:
        raise NotSupportedException()

    @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
    def PropertiesChanged(
        self, interface: str, changed: Dict[str, object], invalidated: List[str]
    ) -> None:
        pass


class ProvisioningManager:
    """Tracks Wi-Fi status and orchestrates nmcli calls."""

    def __init__(self) -> None:
        self._status = STATUS_IDLE
        self._status_lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._status_callbacks: List[Callable[[str], None]] = []

    @property
    def status(self) -> str:
        with self._status_lock:
            return self._status

    @property
    def last_error(self) -> Optional[str]:
        with self._status_lock:
            return self._last_error

    def add_status_callback(self, callback: Callable[[str], None]) -> None:
        self._status_callbacks.append(callback)

    def set_status(self, status: str, error: Optional[str] = None) -> None:
        with self._status_lock:
            if self._status == status and (error == self._last_error):
                return
            self._status = status
            self._last_error = error
        logging.info("Status changed to %s", status)
        if error:
            logging.error("Last error: %s", error.strip())
        for callback in self._status_callbacks:
            try:
                callback(status)
            except Exception as exc:  # pragma: no cover - defensive
                logging.exception("Status callback failed: %s", exc)

    def start_connection(self, ssid: str, password: str) -> None:
        def worker() -> None:
            logging.info("Attempting to connect to Wi-Fi network '%s'", ssid)
            self.set_status(STATUS_CONNECTING)
            try:
                result = subprocess.run(
                    [
                        "nmcli",
                        "dev",
                        "wifi",
                        "connect",
                        ssid,
                        "password",
                        password,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                logging.info("nmcli stdout: %s", result.stdout.strip())
                if result.stderr:
                    logging.warning("nmcli stderr: %s", result.stderr.strip())
                if result.returncode == 0:
                    self.set_status(STATUS_CONNECTED)
                else:
                    error_msg = result.stderr or result.stdout or "nmcli failed"
                    self.set_status(STATUS_FAILED, error=error_msg)
            except Exception as exc:  # pragma: no cover - unexpected failure
                self.set_status(STATUS_FAILED, error=str(exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()


class WiFiStatusCharacteristic(Characteristic):
    def __init__(self, bus: dbus.SystemBus, index: int, service: Service, manager: ProvisioningManager):
        super().__init__(bus, index, WIFI_STATUS_CHAR_UUID, ["read", "notify"], service)
        self.manager = manager
        self.value = self._encode(manager.status)
        manager.add_status_callback(self._handle_status_change)

    def _encode(self, status: str) -> List[int]:
        return [dbus.Byte(c) for c in status.encode("utf-8")]

    def _handle_status_change(self, status: str) -> None:
        self.value = self._encode(status)
        if self.notifying:
            logging.info("Sending BLE status notification: %s", status)
            self.PropertiesChanged(
                GATT_CHRC_IFACE,
                {"Value": dbus.Array(self.value, signature="y")},
                [],
            )

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options: Dict[str, object]) -> dbus.Array:
        logging.info("Status characteristic read: %s", self.manager.status)
        return dbus.Array(self.value, signature="y")

    @dbus.service.method(GATT_CHRC_IFACE)
    def StartNotify(self) -> None:
        if self.notifying:
            return
        self.notifying = True
        logging.info("Status notifications enabled")
        self.PropertiesChanged(
            GATT_CHRC_IFACE,
            {"Value": dbus.Array(self.value, signature="y")},
            [],
        )

    @dbus.service.method(GATT_CHRC_IFACE)
    def StopNotify(self) -> None:
        if not self.notifying:
            return
        self.notifying = False
        logging.info("Status notifications disabled")


class WiFiConfigCharacteristic(Characteristic):
    def __init__(self, bus: dbus.SystemBus, index: int, service: Service, manager: ProvisioningManager):
        super().__init__(bus, index, WIFI_CONFIG_CHAR_UUID, ["write"], service)
        self.manager = manager

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value: List[int], options: Dict[str, object]) -> None:  # type: ignore[override]
        try:
            payload = bytes(value).decode("utf-8")
            logging.info("Received Wi-Fi config payload: %s", payload)
        except UnicodeDecodeError as exc:
            logging.error("Invalid UTF-8 payload: %s", exc)
            self.manager.set_status(STATUS_FAILED, error="invalid utf-8")
            return

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            logging.error("Malformed JSON: %s", exc)
            self.manager.set_status(STATUS_FAILED, error="invalid json")
            return

        ssid = data.get("ssid")
        password = data.get("password")

        if not isinstance(ssid, str) or not ssid.strip():
            logging.error("Missing or empty SSID in payload")
            self.manager.set_status(STATUS_FAILED, error="missing ssid")
            return

        if not isinstance(password, str) or not password.strip():
            logging.error("Missing or empty password in payload")
            self.manager.set_status(STATUS_FAILED, error="missing password")
            return

        self.manager.start_connection(ssid.strip(), password)


class WiFiProvisioningService(Service):
    def __init__(self, bus: dbus.SystemBus, index: int, manager: ProvisioningManager):
        super().__init__(bus, index, WIFI_SERVICE_UUID, True)
        self.manager = manager
        self.status_char = WiFiStatusCharacteristic(bus, 0, self, manager)
        self.config_char = WiFiConfigCharacteristic(bus, 1, self, manager)
        self.add_characteristic(self.status_char)
        self.add_characteristic(self.config_char)


class Advertisement(dbus.service.Object):
    PATH_BASE = "/org/rovy/wifi/advertisement"

    def __init__(self, bus: dbus.SystemBus, index: int, advertising_type: str, name: str):
        self.path = f"{self.PATH_BASE}{index}"
        self.bus = bus
        self.ad_type = advertising_type
        self.name = name
        super().__init__(bus, self.path)

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def get_properties(self) -> Dict[str, Dict[str, object]]:
        return {
            LE_ADVERTISEMENT_IFACE: {
                "Type": self.ad_type,
                "ServiceUUIDs": dbus.Array([WIFI_SERVICE_UUID], signature="s"),
                "LocalName": self.name,
                "Discoverable": True,
            }
        }

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface: str) -> Dict[str, object]:
        if interface != LE_ADVERTISEMENT_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[LE_ADVERTISEMENT_IFACE]

    @dbus.service.method(LE_ADVERTISEMENT_IFACE)
    def Release(self) -> None:
        logging.info("Advertisement released")


def find_adapter(bus: dbus.SystemBus) -> Optional[str]:
    remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, "/"), DBUS_OM_IFACE)
    objects = remote_om.GetManagedObjects()
    for path, interfaces in objects.items():
        if ADAPTER_IFACE in interfaces:
            return path
    return None


def get_default_name() -> str:
    identifier = None
    candidates = [
        "/sys/class/dmi/id/product_serial",
        "/sys/firmware/devicetree/base/serial-number",
    ]
    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                identifier = handle.read().strip()
                break
        except OSError:
            continue
    if not identifier:
        identifier = uuid.getnode().to_bytes(6, "big").hex()
    suffix = identifier.replace("-", "")[-4:]
    suffix = suffix.upper().rjust(4, "0")[-4:]
    return f"ROVY-{suffix}"


def register_app_and_advertisement(
    bus: dbus.SystemBus,
    adapter_path: str,
    app: Application,
    advertisement: Advertisement,
) -> None:
    service_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), GATT_MANAGER_IFACE)
    ad_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), LE_ADVERTISING_MANAGER_IFACE)

    logging.info("Registering GATT application")
    service_manager.RegisterApplication(app.get_path(), {}, reply_handler=lambda: logging.info("GATT application registered"), error_handler=lambda e: logging.error("Failed to register application: %s", e))

    logging.info("Registering advertisement with name %s", advertisement.name)
    ad_manager.RegisterAdvertisement(advertisement.get_path(), {}, reply_handler=lambda: logging.info("Advertisement registered"), error_handler=lambda e: logging.error("Failed to register advertisement: %s", e))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Starting BLE Wi-Fi provisioning service")
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()

    adapter_path = find_adapter(bus)
    if not adapter_path:
        logging.error("Bluetooth adapter not found. Ensure BlueZ is running and Bluetooth hardware is available.")
        sys.exit(1)

    manager = ProvisioningManager()
    app = Application(bus)
    service = WiFiProvisioningService(bus, 0, manager)
    app.add_service(service)

    name = get_default_name()
    advertisement = Advertisement(bus, 0, "peripheral", name)

    register_app_and_advertisement(bus, adapter_path, app, advertisement)

    loop = GLib.MainLoop()

    def shutdown(signum: int, frame) -> None:
        logging.info("Received signal %s, shutting down", signum)
        loop.quit()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    logging.info("Entering main event loop (status: %s)", manager.status)
    try:
        loop.run()
    finally:
        logging.info("Stopping advertisement")
        ad_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), LE_ADVERTISING_MANAGER_IFACE)
        try:
            ad_manager.UnregisterAdvertisement(advertisement.get_path())
        except dbus.exceptions.DBusException as exc:
            logging.warning("Failed to unregister advertisement: %s", exc)
        logging.info("BLE Wi-Fi provisioning service exited")


if __name__ == "__main__":
    main()
