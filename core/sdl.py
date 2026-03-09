"""Self-driving lab connectivity helpers (MQTT / HTTP / Serial / TCP)."""

import json
import os
import sys
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import socket
from pathlib import Path

import requests

try:
    import paho.mqtt.client as mqtt  # type: ignore
    MQTT_AVAILABLE = True
except ImportError:
    mqtt = None
    MQTT_AVAILABLE = False

try:
    import serial  # type: ignore
    import serial.tools.list_ports  # type: ignore
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False


@dataclass
class SDLSettings:
    protocol: str = "http"  # "mqtt", "http", "serial", "tcp", "embedded"
    response_timeout: float = 20.0

    # MQTT
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_publish_topic: str = "bo/commands"
    mqtt_response_topic: str = "bo/results"
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    mqtt_client_id: str = "bo-platform"

    # HTTP
    http_endpoint: str = "http://localhost:8000/bo"
    http_method: str = "POST"
    http_headers: Dict[str, str] = field(default_factory=dict)

    # Serial
    serial_port: str = "COM3"
    serial_baud: int = 115200
    serial_timeout: float = 2.0

    # TCP
    tcp_host: str = "localhost"
    tcp_port: int = 7000

    # SDL control policy (optional passthrough)
    digital_twin_control: bool = False
    require_continue_each_trial: bool = True

    # Embedded RYB SDL (in-process CEID runner)
    embedded_control_mode: str = "sdl"  # "sdl" | "manual"
    embedded_sensor_timeout: float = 30.0
    embedded_manual_timeout: float = 1800.0
    embedded_manual_aspiration_volume_ml: float = 1.0
    embedded_sdl_start_timeout: float = 7200.0
    embedded_unity_enable: bool = True
    embedded_unity_transport: str = "livekit"  # "livekit" | "tcp" | "none"
    embedded_unity_host: str = "0.0.0.0"
    embedded_unity_port: int = 7100
    embedded_unity_dest_identity: str = "unity"
    embedded_livekit_url: str = "wss://digital-twin-e1hn80jk.livekit.cloud"
    embedded_livekit_room: str = "dt"
    embedded_livekit_topic: str = "twin"
    embedded_sdl_livekit_token: str = ""
    embedded_arduino_port: str = "COM7"
    embedded_arduino_baud: int = 9600
    embedded_log_level: str = "INFO"
    embedded_log_file: str = "sdl_agent.log"


class SDLConnector:
    """Lightweight connector used by the BO UI to talk to SDL endpoints."""

    def __init__(self, settings: SDLSettings):
        self.settings = settings
        self._mqtt_client = None
        self._mqtt_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._mqtt_lock = threading.Lock()
        self._serial = None
        self._embedded_runner = None
        self.connected = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> Tuple[bool, str]:
        """Establish a connection for the chosen protocol."""
        proto = self.settings.protocol.lower()
        try:
            if proto == "mqtt":
                if not MQTT_AVAILABLE:
                    return False, "paho-mqtt is not installed"
                self._connect_mqtt()
            elif proto == "http":
                # For HTTP we lazily test on first send; nothing persistent
                pass
            elif proto == "serial":
                if not SERIAL_AVAILABLE:
                    return False, "pyserial is not installed"
                self._connect_serial()
            elif proto == "tcp":
                # Lazy-open on first send; just mark ok
                pass
            elif proto == "embedded":
                self._connect_embedded()
            else:
                return False, f"Unsupported protocol: {proto}"
            self.connected = True
            return True, "Connected"
        except Exception as exc:
            self.connected = False
            return False, str(exc)

    def close(self):
        """Close any open resources."""
        if self._mqtt_client:
            try:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
            except Exception:
                pass
            self._mqtt_client = None
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        if self._embedded_runner is not None:
            try:
                self._embedded_runner.stop()
            except Exception:
                pass
            self._embedded_runner = None
        self.connected = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def test_connection(self) -> Tuple[bool, str]:
        """Quick connectivity check without pushing a real trial."""
        ok, msg = self.connect()
        if not ok:
            return ok, msg

        try:
            if self.settings.protocol.lower() == "mqtt":
                # Publish a ping and wait briefly for any response
                payload = {"type": "ping", "ts": time.time()}
                self._mqtt_client.publish(self.settings.mqtt_publish_topic, json.dumps(payload))
                return True, "MQTT broker reachable"
            if self.settings.protocol.lower() == "http":
                # Use a harmless GET for connectivity so we don't trigger lab actions
                resp = requests.get(self.settings.http_endpoint, timeout=5)
                return True, f"HTTP {resp.status_code} reachable"
            if self.settings.protocol.lower() == "serial":
                # Opening the port is enough; keep it open
                return True, f"Serial port {self.settings.serial_port} opened"
            if self.settings.protocol.lower() == "tcp":
                with socket.create_connection((self.settings.tcp_host, self.settings.tcp_port), timeout=5) as sock:
                    sock.sendall(b'{"type":"ping"}\n')
                return True, f"TCP {self.settings.tcp_host}:{self.settings.tcp_port} reachable"
            if self.settings.protocol.lower() == "embedded":
                self._connect_embedded()
                if self._embedded_runner is None:
                    return False, "Embedded SDL failed to initialize"
                ping_payload = self._embedded_runner.ping()
                return True, f"Embedded SDL ready: {ping_payload}"
        except Exception as exc:
            return False, f"Connectivity test failed: {exc}"
        return True, "Connection OK"

    def send_candidate_detailed(self, parameters: Dict[str, Any], trial_index: Optional[int] = None) -> Dict[str, Any]:
        """Send a candidate to the SDL and return the full response payload."""
        proto = self.settings.protocol.lower()
        if not self.connected and proto != "http":
            ok, msg = self.connect()
            if not ok:
                raise RuntimeError(msg)

        if proto == "mqtt":
            return self._send_mqtt(parameters, trial_index)
        if proto == "http":
            return self._send_http(parameters, trial_index)
        if proto == "serial":
            return self._send_serial(parameters, trial_index)
        if proto == "tcp":
            return self._send_tcp(parameters, trial_index)
        if proto == "embedded":
            return self._send_embedded(parameters, trial_index)
        raise RuntimeError(f"Unsupported protocol: {proto}")

    def send_candidate(self, parameters: Dict[str, Any], trial_index: Optional[int] = None) -> Dict[str, Any]:
        """Send a candidate to the SDL and return objective values only."""
        payload = self.send_candidate_detailed(parameters, trial_index=trial_index)
        return self._extract_objectives(payload)

    # ------------------------------------------------------------------
    # Protocol implementations
    # ------------------------------------------------------------------
    def _connect_mqtt(self):
        if self._mqtt_client:
            return

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                client.subscribe(self.settings.mqtt_response_topic)

        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                self._mqtt_queue.put(payload)
            except Exception:
                pass

        client_id = f"{self.settings.mqtt_client_id}-{int(time.time()*1000)}"
        self._mqtt_client = mqtt.Client(client_id=client_id, clean_session=True)
        if self.settings.mqtt_username:
            self._mqtt_client.username_pw_set(self.settings.mqtt_username, self.settings.mqtt_password)
        self._mqtt_client.on_connect = on_connect
        self._mqtt_client.on_message = on_message
        self._mqtt_client.connect(self.settings.mqtt_host, self.settings.mqtt_port, keepalive=30)
        self._mqtt_client.loop_start()
        time.sleep(0.2)  # allow subscribe

    def _send_mqtt(self, parameters: Dict[str, Any], trial_index: Optional[int]) -> Dict[str, Any]:
        if not self._mqtt_client:
            self._connect_mqtt()

        message = {
            "type": "bo_candidate",
            "trial_index": trial_index,
            "parameters": parameters,
            "control": {
                "digital_twin_control": bool(self.settings.digital_twin_control),
                "require_continue_each_trial": bool(self.settings.require_continue_each_trial),
            },
            "ts": time.time(),
        }
        with self._mqtt_lock:
            self._mqtt_client.publish(self.settings.mqtt_publish_topic, json.dumps(message))

        deadline = time.time() + self.settings.response_timeout
        while time.time() < deadline:
            try:
                payload = self._mqtt_queue.get(timeout=1)
                if trial_index is None or payload.get("trial_index") == trial_index:
                    return payload
            except queue.Empty:
                continue
        raise TimeoutError(f"MQTT response timeout after {self.settings.response_timeout}s")

    def _send_http(self, parameters: Dict[str, Any], trial_index: Optional[int]) -> Dict[str, Any]:
        body = {
            "trial_index": trial_index,
            "parameters": parameters,
            "control": {
                "digital_twin_control": bool(self.settings.digital_twin_control),
                "require_continue_each_trial": bool(self.settings.require_continue_each_trial),
            },
            "ts": time.time(),
        }
        headers = {"Content-Type": "application/json"}
        headers.update(self.settings.http_headers or {})
        resp = requests.request(
            self.settings.http_method.upper(),
            self.settings.http_endpoint,
            headers=headers,
            json=body,
            timeout=max(self.settings.response_timeout, 2.0),
        )
        resp.raise_for_status()
        payload = resp.json() if resp.content else {}
        return payload

    def _connect_serial(self):
        if self._serial and self._serial.is_open:
            return
        self._serial = serial.Serial(
            port=self.settings.serial_port,
            baudrate=self.settings.serial_baud,
            timeout=self.settings.serial_timeout,
        )
        time.sleep(0.2)

    def _send_serial(self, parameters: Dict[str, Any], trial_index: Optional[int]) -> Dict[str, Any]:
        if not self._serial or not self._serial.is_open:
            self._connect_serial()

        message = {
            "trial_index": trial_index,
            "parameters": parameters,
            "control": {
                "digital_twin_control": bool(self.settings.digital_twin_control),
                "require_continue_each_trial": bool(self.settings.require_continue_each_trial),
            },
            "ts": time.time(),
        }
        payload = (json.dumps(message) + "\n").encode("utf-8")
        self._serial.write(payload)
        self._serial.flush()

        deadline = time.time() + max(self.settings.response_timeout, 2.0)
        buffer = b""
        while time.time() < deadline:
            chunk = self._serial.readline()
            if chunk:
                buffer += chunk
                try:
                    parsed = json.loads(buffer.decode("utf-8"))
                    return parsed
                except Exception:
                    buffer = b""
            else:
                time.sleep(0.05)
        raise TimeoutError(f"Serial response timeout after {self.settings.response_timeout}s")

    def _send_tcp(self, parameters: Dict[str, Any], trial_index: Optional[int]) -> Dict[str, Any]:
        """Simple line-delimited JSON over TCP."""
        timeout = max(self.settings.response_timeout, 2.0)
        message = {
            "trial_index": trial_index,
            "parameters": parameters,
            "control": {
                "digital_twin_control": bool(self.settings.digital_twin_control),
                "require_continue_each_trial": bool(self.settings.require_continue_each_trial),
            },
            "ts": time.time(),
        }
        payload = (json.dumps(message) + "\n").encode("utf-8")
        deadline = time.time() + timeout

        with socket.create_connection((self.settings.tcp_host, self.settings.tcp_port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(payload)
            buffer = b""
            while time.time() < deadline:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                if b"\n" in buffer:
                    line, _, _rest = buffer.partition(b"\n")
                    try:
                        parsed = json.loads(line.decode("utf-8"))
                        return parsed
                    except Exception:
                        buffer = _rest
                        continue
        raise TimeoutError(f"TCP response timeout after {timeout}s")

    def _connect_embedded(self):
        if self._embedded_runner is not None and getattr(self._embedded_runner, "started", False):
            return

        # Ensure the workspace root (contains sdl_agent/) is importable even when CEID is launched from bo_vtwo/.
        workspace_root = Path(__file__).resolve().parents[2]
        if str(workspace_root) not in sys.path:
            sys.path.insert(0, str(workspace_root))

        try:
            from sdl_agent import EmbeddedRYBSDL, EmbeddedSDLConfig  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(f"Unable to import embedded SDL runner: {exc}") from exc

        token = self.settings.embedded_sdl_livekit_token or os.getenv("SDL_LIVEKIT_TOKEN", "")
        cfg = EmbeddedSDLConfig(
            control_mode=str(self.settings.embedded_control_mode or "sdl").strip().lower(),
            digital_twin_control=bool(self.settings.digital_twin_control),
            require_continue_each_trial=bool(self.settings.require_continue_each_trial),
            sdl_start_timeout=float(self.settings.embedded_sdl_start_timeout),
            sensor_timeout=float(self.settings.embedded_sensor_timeout),
            manual_timeout=float(self.settings.embedded_manual_timeout),
            manual_aspiration_volume_ml=float(self.settings.embedded_manual_aspiration_volume_ml),
            unity_enable=bool(self.settings.embedded_unity_enable),
            unity_transport=str(self.settings.embedded_unity_transport or "livekit").strip().lower(),
            unity_host=str(self.settings.embedded_unity_host),
            unity_port=int(self.settings.embedded_unity_port),
            unity_dest_identity=str(self.settings.embedded_unity_dest_identity),
            livekit_url=str(self.settings.embedded_livekit_url),
            livekit_room=str(self.settings.embedded_livekit_room),
            livekit_topic=str(self.settings.embedded_livekit_topic),
            sdl_livekit_token=str(token),
            arduino_port=str(self.settings.embedded_arduino_port),
            arduino_baud=int(self.settings.embedded_arduino_baud),
            log_level=str(self.settings.embedded_log_level),
            log_file=str(self.settings.embedded_log_file),
        )
        runner = EmbeddedRYBSDL(cfg)
        runner.start()
        self._embedded_runner = runner

    def _send_embedded(self, parameters: Dict[str, Any], trial_index: Optional[int]) -> Dict[str, Any]:
        if self._embedded_runner is None or not getattr(self._embedded_runner, "started", False):
            self._connect_embedded()
        if self._embedded_runner is None:
            raise RuntimeError("Embedded SDL runner is not available")

        payload = {
            "trial_index": trial_index,
            "parameters": parameters,
            "control": {
                "digital_twin_control": bool(self.settings.digital_twin_control),
                "require_continue_each_trial": bool(self.settings.require_continue_each_trial),
            },
            "ts": time.time(),
        }
        response = self._embedded_runner.handle_candidate(payload)
        if not isinstance(response, dict):
            raise ValueError("Embedded SDL response is not a dictionary")
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_objectives(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Standardise payload to {objective_name: value} dict."""
        if not isinstance(payload, dict):
            raise ValueError("SDL response is not a dictionary")
        if "objectives" in payload and isinstance(payload["objectives"], dict):
            return payload["objectives"]
        # Fallback: treat remaining keys as objectives
        objectives = {k: v for k, v in payload.items() if k not in {"trial_index", "ts", "type", "status"}}
        if not objectives:
            raise ValueError("SDL response did not include objective values")
        return objectives
