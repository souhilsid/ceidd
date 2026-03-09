"""
Headless self-driving lab agent for the BO platform.

Listens over MQTT / HTTP / TCP / Serial, executes the spectral mixer routine
using the gcode/relay logic, reads frequency from the Arduino, and returns it
as the objective value. GUI is removed; fully headless.
"""

import argparse
import atexit
import csv
import json
import threading
import time
import socket
import os
import math
from typing import Dict, Any, List, Optional, Set, Tuple
import logging
import sys
import re

import requests

try:
    from sdl_agent.livekit_twin import LiveKitTwinClient
    LIVEKIT_CLIENT_AVAILABLE = True
except Exception:
    LiveKitTwinClient = None  # type: ignore
    LIVEKIT_CLIENT_AVAILABLE = False

try:
    import paho.mqtt.client as mqtt  # type: ignore
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

try:
    import serial  # type: ignore
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# =========================
# ------ CONFIG VARS ------
# =========================
PRINTER_IP = "192.168.1.101"
MOONRAKER_URL = f"http://{PRINTER_IP}/printer/gcode/script"
LIVEKIT_DEFAULT_URL = "wss://digital-twin-e1hn80jk.livekit.cloud"
LIVEKIT_DEFAULT_ROOM = "dt"
LIVEKIT_DEFAULT_TOPIC = "twin"

# --- 1. SENSOR CONFIG ---
ARDUINO_PORT = "COM5"
BAUD_RATE = 9600

# --- 2. ROBOT COORDINATES ---
home_pos_x = 25.5
home_pos_y = 131.5
safe_height_z = 320

Beaker_x = 202
Beaker_y = 335
Beaker_z = 310

pip_tip_x1 = 27.7
pip_tip_y1 = 120
pip_tip_z = float(os.getenv("SDL_PIP_TIP_Z", "155"))
pip_tip_z_pickup = float(os.getenv("SDL_PIP_TIP_Z_PICKUP", "260"))
pip_tip_z_drop = float(os.getenv("SDL_PIP_TIP_Z_DROP", "180"))
pip_tip_x_diff = 8
pip_tip_y_diff = 9
cols_pipettes = 8

vial_rack_x1 = 171.00
vial_rack_y1 = 129.00
vial_rack_z = 157.00
vial_rack_x_diff = 32
vial_rack_y_diff = 44

VIAL_MAX_CAPACITY = 13
TIP_MAX_CAPACITY_ML = 1.0
SDL_FIXED_ASPIRATION_VOLUME_ML = 0.2
DEFAULT_MANUAL_ASPIRATION_VOLUME_ML = 0.2

ASPIRATE_SEC = 2
DISPENSE_SEC = 2
TIME_STARTUP_OVERHEAD = 94
TIME_PER_ML_CYCLE = 40
ACTUATOR_SECONDS_PER_ML = float(os.getenv("SDL_ACTUATOR_SECONDS_PER_ML", "1.7"))
ACTUATOR_MIN_SECONDS = float(os.getenv("SDL_ACTUATOR_MIN_SECONDS", "0.02"))
TIP_PICK_DWELL_SEC = float(os.getenv("SDL_TIP_PICK_DWELL_SEC", "0.7"))

STOP_REQUESTED = False

# Relay board runtime config (can be overridden by CLI/env at startup)
RELAY_DEVICE_INDEX: Optional[int] = None
RELAY_ACTIVE_HIGH = True
RELAY_AUTO_SCAN = True
RELAY_OPEN_RETRIES = 2
_RELAY_WORKING_INDEX: Optional[int] = None
_RELAY_BOARD = None
_RELAY_IO_LOCK = threading.Lock()

# Logger (set up after arg parsing; create a module-level handle for convenience)
logger = logging.getLogger("SDLAgent")

# Optional motion/action trace CSV (enabled by --trace-csv or SDL_TRACE_CSV)
_TRACE_FIELDS = [
    "seq",
    "ts_unix",
    "kind",
    "trial_index",
    "x",
    "y",
    "z",
    "feedrate",
    "event",
    "detail",
    "gcode",
    "action",
    "volume_ml",
    "duration_sec",
    "tip_index",
    "vial",
    "cycle",
    "total_cycles",
    "row",
    "note",
]
_TRACE_LOCK = threading.Lock()
_TRACE_SEQ = 0
_TRACE_PATH = ""
_TRACE_FILE = None
_TRACE_WRITER: Optional[csv.DictWriter] = None


def _resolve_trace_csv_path(raw_path: str) -> str:
    path = os.path.expandvars(os.path.expanduser(str(raw_path or "").strip()))
    if "{ts}" in path:
        path = path.replace("{ts}", time.strftime("%Y%m%d_%H%M%S"))
    return path


def _close_trace_csv() -> None:
    global _TRACE_FILE, _TRACE_WRITER
    with _TRACE_LOCK:
        if _TRACE_FILE is None:
            return
        try:
            _TRACE_FILE.flush()
        except Exception:
            pass
        try:
            _TRACE_FILE.close()
        except Exception:
            pass
        _TRACE_FILE = None
        _TRACE_WRITER = None


def _open_trace_csv(path: str) -> None:
    global _TRACE_FILE, _TRACE_WRITER, _TRACE_PATH, _TRACE_SEQ
    resolved = _resolve_trace_csv_path(path)
    if not resolved:
        return
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with _TRACE_LOCK:
        _TRACE_SEQ = 0
        _TRACE_PATH = resolved
        _TRACE_FILE = open(resolved, mode="w", encoding="utf-8", newline="")
        _TRACE_WRITER = csv.DictWriter(_TRACE_FILE, fieldnames=_TRACE_FIELDS)
        _TRACE_WRITER.writeheader()
        _TRACE_FILE.flush()
    logger.info("[TRACE] csv enabled path=%s", resolved)


def _trace_log(kind: str, **kwargs: Any) -> None:
    global _TRACE_SEQ
    with _TRACE_LOCK:
        if _TRACE_WRITER is None or _TRACE_FILE is None:
            return
        _TRACE_SEQ += 1
        row = {k: "" for k in _TRACE_FIELDS}
        row["seq"] = str(_TRACE_SEQ)
        row["ts_unix"] = f"{time.time():.6f}"
        row["kind"] = str(kind)
        trial_index = kwargs.pop("trial_index", _get_active_trial_index())
        row["trial_index"] = "" if trial_index is None else str(trial_index)
        for key, value in kwargs.items():
            if key not in row:
                continue
            if isinstance(value, list):
                row[key] = "|".join(str(v) for v in value)
            elif value is None:
                row[key] = ""
            else:
                row[key] = str(value)
        try:
            _TRACE_WRITER.writerow(row)
            _TRACE_FILE.flush()
        except Exception as exc:
            logger.error("[TRACE] write failed: %s", exc)
            _close_trace_csv()

# ============================================================
# ✅ NEW: UNITY DIGITAL TWIN TCP SERVER (ADDED, NOT REMOVED)
# ============================================================
class UnityTwinServer:
    """
    Unity connects to this server (Python is central).
    Messages are NDJSON (one JSON per line).
    """
    def __init__(self):
        self._srv: Optional[socket.socket] = None
        self._clients: List[socket.socket] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.enabled = False

    def start(self, host: str, port: int):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(5)
        self.enabled = True
        logger.info(f"[UNITY] Twin server listening on {host}:{port}")

        def accept_loop():
            assert self._srv is not None
            while True:
                try:
                    conn, addr = self._srv.accept()
                    try:
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except Exception:
                        pass
                    with self._lock:
                        self._clients.append(conn)
                    logger.info(f"[UNITY] client connected from {addr}")
                except Exception as e:
                    logger.error(f"[UNITY] accept error: {e}")
                    time.sleep(0.5)

        self._thread = threading.Thread(target=accept_loop, daemon=True)
        self._thread.start()

    def broadcast(self, obj: Dict[str, Any]):
        if not self.enabled:
            return
        line = (json.dumps(obj) + "\n").encode("utf-8")
        with self._lock:
            alive: List[socket.socket] = []
            for c in self._clients:
                try:
                    c.sendall(line)
                    alive.append(c)
                except Exception:
                    try:
                        c.close()
                    except Exception:
                        pass
            self._clients = alive

UNITY_TWIN = UnityTwinServer()
LIVEKIT_TWIN: Optional["LiveKitTwinClient"] = None
UNITY_TRANSPORT: str = "none"  # livekit|tcp|none
UNITY_DEST_IDENTITY = "unity"
UNITY_TOPIC = LIVEKIT_DEFAULT_TOPIC

_WAYPOINT_SEQ_LOCK = threading.Lock()
_WAYPOINT_SEQ = 0
_ACTIVE_TRIAL_LOCK = threading.Lock()
_ACTIVE_TRIAL_INDEX: Optional[int] = None

# last known pose (for full pose streaming even when move_to changes only x or y or z)
_last_x = home_pos_x
_last_y = home_pos_y
_last_z = safe_height_z

def _parse_env_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _set_active_trial_index(trial_index: Optional[int]) -> None:
    global _ACTIVE_TRIAL_INDEX
    with _ACTIVE_TRIAL_LOCK:
        _ACTIVE_TRIAL_INDEX = trial_index


def _get_active_trial_index() -> Optional[int]:
    with _ACTIVE_TRIAL_LOCK:
        return _ACTIVE_TRIAL_INDEX


def _next_waypoint_seq() -> int:
    global _WAYPOINT_SEQ
    with _WAYPOINT_SEQ_LOCK:
        _WAYPOINT_SEQ += 1
        return _WAYPOINT_SEQ


def _publish_to_unity(obj: Dict[str, Any]) -> None:
    if UNITY_TRANSPORT == "livekit":
        if LIVEKIT_TWIN is None:
            return
        LIVEKIT_TWIN.send_json(
            obj,
            dest_identities=[UNITY_DEST_IDENTITY],
            reliable=True,
            topic=UNITY_TOPIC,
        )
        return
    if UNITY_TRANSPORT == "tcp":
        UNITY_TWIN.broadcast(obj)


def unity_send_waypoint(
    x: float,
    y: float,
    z: float,
    evts: Optional[List[str]] = None,
    yaw: Optional[float] = None,
) -> None:
    payload: Dict[str, Any] = {
        "type": "waypoint",
        "seq": _next_waypoint_seq(),
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "yaw": None if yaw is None else float(yaw),
        "t": time.time(),
    }
    # Preserve legacy event hooks used by existing Unity interpolation path.
    if evts:
        payload["evts"] = evts
    _trace_log(
        "waypoint",
        x=payload["x"],
        y=payload["y"],
        z=payload["z"],
        event=evts,
    )
    _publish_to_unity(payload)


def unity_send_event(name: str, detail: Optional[str] = None) -> None:
    _trace_log("event", event=str(name), detail=detail)
    _publish_to_unity(
        {
            "type": "event",
            "name": str(name),
            "detail": None if detail is None else str(detail),
            "t": time.time(),
        }
    )


def unity_send_frequency(thz: float, color_name: str = "") -> None:
    _publish_to_unity(
        {
            "type": "frequency",
            "trial_index": _get_active_trial_index(),
            "value": float(thz),
            "unit": "THz",
            "color": str(color_name),
            "t": time.time(),
        }
    )


def unity_send_manual_candidate(trial_index: int, parameters: Dict[str, Any]) -> None:
    _publish_to_unity(
        {
            "type": "manual_candidate",
            "trial_index": int(trial_index),
            "parameters": dict(parameters or {}),
            "t": time.time(),
        }
    )


def unity_send_candidate(trial_index: int, parameters: Dict[str, Any], mode: str = "sdl") -> None:
    _publish_to_unity(
        {
            "type": "candidate",
            "trial_index": int(trial_index),
            "mode": str(mode or "sdl"),
            "parameters": dict(parameters or {}),
            "t": time.time(),
        }
    )


def unity_stream_enabled() -> bool:
    return UNITY_TRANSPORT in {"livekit", "tcp"}

# ============================================================
# ✅ NEW: CONTINUOUS SENSOR STREAM (ADDED)
# - reads Arduino always
# - caches latest frequency
# - broadcasts to Unity once per second
# ============================================================
_FREQ_LOCK = threading.Lock()
_LAST_FREQ_THZ: Optional[float] = None
_LAST_FREQ_COLOR: str = ""
_LAST_FREQ_TS: float = 0.0

# Arduino line format:
# "Freq: 430 THz | Color: Red"
# "Freq: -- THz | Color: Clear (Water)"
_FREQ_RE = re.compile(r"Freq:\s*(--|\d+(?:\.\d+)?)\s*THz\s*\|\s*Color:\s*(.*)\s*$", re.IGNORECASE)

def _parse_freq_line(line: str) -> Optional[tuple]:
    m = _FREQ_RE.search(line.strip())
    if not m:
        return None
    v = m.group(1).strip()
    cname = m.group(2).strip()
    if v == "--":
        thz = 0.0
    else:
        try:
            thz = float(v)
        except ValueError:
            thz = 0.0
    return thz, cname

class SensorThread(threading.Thread):
    def __init__(self, port: str, baud: int):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        if not SERIAL_AVAILABLE:
            logger.warning("[SENSOR] pyserial not installed -> sensor disabled")
            return

        last_sent = 0.0

        while not self._stop.is_set():
            try:
                ser = serial.Serial(self.port, self.baud, timeout=1.0)
                logger.info(f"[SENSOR] opened {self.port} @ {self.baud}")
            except Exception as e:
                logger.error(f"[SENSOR] open failed ({self.port}): {e} (close Serial Monitor!)")
                time.sleep(2.0)
                continue

            with ser:
                while not self._stop.is_set():
                    try:
                        line = ser.readline().decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue

                        parsed = _parse_freq_line(line)
                        if not parsed:
                            continue

                        thz, cname = parsed

                        now = time.time()
                        with _FREQ_LOCK:
                            global _LAST_FREQ_THZ, _LAST_FREQ_COLOR, _LAST_FREQ_TS
                            _LAST_FREQ_THZ = float(thz)
                            _LAST_FREQ_COLOR = cname
                            _LAST_FREQ_TS = now

                        # ✅ broadcast to Unity ONCE PER SECOND
                        if unity_stream_enabled() and (now - last_sent) >= 1.0:
                            unity_send_frequency(float(thz), cname)
                            last_sent = now

                    except Exception:
                        # reopen serial if something breaks
                        break

            time.sleep(1.0)

_SENSOR_THREAD: Optional[SensorThread] = None

def _start_sensor_thread():
    global _SENSOR_THREAD
    if _SENSOR_THREAD is not None:
        return
    _SENSOR_THREAD = SensorThread(ARDUINO_PORT, BAUD_RATE)
    _SENSOR_THREAD.start()

def read_frequency_once(timeout: float = 1800.0) -> Optional[float]:
    """
    Read a single frequency value from Arduino.
    ✅ Now uses cached value from SensorThread (recommended).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _FREQ_LOCK:
            if _LAST_FREQ_THZ is not None and (time.time() - _LAST_FREQ_TS) < 3.0:
                return float(_LAST_FREQ_THZ)
        time.sleep(0.05)
    return None


def _read_cached_frequency(max_age_sec: float = 3.0) -> Optional[Tuple[float, str, float]]:
    max_age = max(0.05, float(max_age_sec))
    now = time.time()
    with _FREQ_LOCK:
        if _LAST_FREQ_THZ is None:
            return None
        if (now - _LAST_FREQ_TS) > max_age:
            return None
        return float(_LAST_FREQ_THZ), str(_LAST_FREQ_COLOR), float(_LAST_FREQ_TS)


def read_frequency_stats(
    sample_count: int = 1,
    timeout: float = 30.0,
    sample_interval: float = 0.0,
    require_fresh_samples: bool = True,
    max_age_sec: float = 3.0,
) -> Optional[Dict[str, Any]]:
    requested = max(1, int(sample_count))
    deadline = time.time() + max(0.1, float(timeout))
    interval = max(0.0, float(sample_interval))
    require_fresh = bool(require_fresh_samples)
    max_age = max(0.05, float(max_age_sec))

    samples: List[float] = []
    sample_timestamps: List[float] = []
    latest_color = ""
    last_ts: Optional[float] = None

    while len(samples) < requested and time.time() < deadline:
        cached = _read_cached_frequency(max_age_sec=max_age)
        if cached is None:
            time.sleep(0.02)
            continue

        freq_thz, color_name, freq_ts = cached
        if require_fresh and last_ts is not None and freq_ts <= (last_ts + 1e-9):
            time.sleep(0.01)
            continue

        samples.append(float(freq_thz))
        sample_timestamps.append(float(freq_ts))
        latest_color = str(color_name)
        last_ts = float(freq_ts)

        if len(samples) < requested and interval > 0.0:
            wait_until = min(deadline, time.time() + interval)
            while time.time() < wait_until:
                time.sleep(0.01)

    if not samples:
        return None

    n = len(samples)
    mean_value = float(sum(samples) / float(n))
    if n >= 2:
        variance = sum((x - mean_value) ** 2 for x in samples) / float(n - 1)
        std_value = float(math.sqrt(max(variance, 0.0)))
        sem_value = float(std_value / math.sqrt(float(n)))
    else:
        std_value = 0.0
        sem_value = 0.0

    return {
        "mean": mean_value,
        "std": std_value,
        "sem": sem_value,
        "samples": [float(v) for v in samples],
        "timestamps": [float(v) for v in sample_timestamps],
        "n_samples": n,
        "n_requested": requested,
        "latest_color": latest_color,
        "incomplete": n < requested,
    }

# =========================
# --- HARDWARE DRIVERS ---
# =========================
def send_gcode(cmd: str, wait: float = 0.5):
    global STOP_REQUESTED
    if STOP_REQUESTED:
        return
    _trace_log("gcode", gcode=cmd, detail=f"wait={wait}")
    try:
        r = requests.post(MOONRAKER_URL, params={"script": cmd})
        r.raise_for_status()
        time.sleep(wait)
    except Exception as e:
        logger.error(f"Printer Error ({cmd}): {e}")

def move_to(x=None, y=None, z=None, feedrate=None, unity_evts: Optional[List[str]] = None):
    """
    ✅ same move_to + ADDED unity_evts (optional)
    Unity gets a waypoint immediately (smooth interpolation happens in Unity).
    """
    global STOP_REQUESTED, _last_x, _last_y, _last_z
    if STOP_REQUESTED:
        return

    # resolve full pose for Unity streaming
    tx = _last_x if x is None else float(x)
    ty = _last_y if y is None else float(y)
    tz = _last_z if z is None else float(z)
    _trace_log(
        "move",
        x=tx,
        y=ty,
        z=tz,
        feedrate=feedrate,
        event=unity_evts,
    )

    # ✅ send waypoint to Unity (does NOT affect printer)
    if unity_stream_enabled():
        unity_send_waypoint(tx, ty, tz, unity_evts)

    cmd = "G1"
    if x is not None:
        cmd += f" X{tx:.2f}"
    if y is not None:
        cmd += f" Y{ty:.2f}"
    if z is not None:
        cmd += f" Z{tz:.2f}"
    if feedrate is not None:
        cmd += f" F{feedrate}"
    send_gcode(cmd)
    send_gcode("M400")

    _last_x, _last_y, _last_z = tx, ty, tz

def _coerce_nonnegative_float(value: Any, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return max(0.0, float(fallback))
    if math.isnan(parsed) or math.isinf(parsed):
        return max(0.0, float(fallback))
    return max(0.0, parsed)


def _fmt_ml(value: float) -> str:
    text = f"{float(value):.3f}"
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def _enumerate_ftdi_indices() -> List[int]:
    try:
        import ftd2xx  # type: ignore
    except Exception:
        return []
    try:
        count = int(ftd2xx.createDeviceInfoList())
    except Exception:
        return []
    if count <= 0:
        return []
    return list(range(count))


def _build_relay_candidate_indices() -> List[int]:
    indices: List[int] = []
    if _RELAY_WORKING_INDEX is not None:
        indices.append(int(_RELAY_WORKING_INDEX))
    if RELAY_DEVICE_INDEX is not None and int(RELAY_DEVICE_INDEX) not in indices:
        indices.append(int(RELAY_DEVICE_INDEX))
    if RELAY_AUTO_SCAN:
        for idx in _enumerate_ftdi_indices():
            if idx not in indices:
                indices.append(idx)
    if not indices:
        indices.append(0)
    return indices


def _open_relay_board(RelayBoard):
    global _RELAY_WORKING_INDEX
    candidates = _build_relay_candidate_indices()
    retries = max(1, int(RELAY_OPEN_RETRIES))
    last_exc: Optional[Exception] = None

    for attempt in range(retries):
        for idx in candidates:
            try:
                board = RelayBoard(device_index=idx, active_high=RELAY_ACTIVE_HIGH)
                if _RELAY_WORKING_INDEX != idx:
                    logger.info("[ACTUATOR] relay connected using FTDI index=%s", idx)
                _RELAY_WORKING_INDEX = idx
                return board
            except TypeError:
                # Compatibility fallback for RelayBoard implementations without device_index argument.
                try:
                    board = RelayBoard(active_high=RELAY_ACTIVE_HIGH)
                    if _RELAY_WORKING_INDEX != idx:
                        logger.info("[ACTUATOR] relay connected using default FTDI device")
                    _RELAY_WORKING_INDEX = idx
                    return board
                except Exception as exc:
                    last_exc = exc
            except Exception as exc:
                # Legacy compatibility: some relay_board implementations accept
                # device_index but still fail on explicit index while succeeding
                # with default device selection.
                last_exc = exc
                try:
                    board = RelayBoard(active_high=RELAY_ACTIVE_HIGH)
                    logger.info(
                        "[ACTUATOR] relay connected using default FTDI device "
                        "after index=%s failed: %s",
                        idx,
                        exc,
                    )
                    _RELAY_WORKING_INDEX = idx
                    return board
                except Exception as legacy_exc:
                    last_exc = legacy_exc
        if attempt < retries - 1:
            time.sleep(0.1)

    detected = _enumerate_ftdi_indices()
    detail = (
        f"detected_indices={detected} candidates={candidates} "
        f"configured_index={RELAY_DEVICE_INDEX} auto_scan={RELAY_AUTO_SCAN}"
    )
    raise RuntimeError(f"Relay open failed ({detail}): {last_exc}")


def _close_relay_board() -> None:
    global _RELAY_BOARD
    if _RELAY_BOARD is None:
        return
    try:
        _RELAY_BOARD.all_off()
    except Exception:
        pass
    try:
        _RELAY_BOARD.close()
    except Exception:
        pass
    _RELAY_BOARD = None


def _get_or_open_relay_board(RelayBoard):
    global _RELAY_BOARD
    if _RELAY_BOARD is not None:
        return _RELAY_BOARD
    _RELAY_BOARD = _open_relay_board(RelayBoard)
    return _RELAY_BOARD


atexit.register(_close_relay_board)
atexit.register(_close_trace_csv)


def actuator_action(action_type, volume_ml: float = DEFAULT_MANUAL_ASPIRATION_VOLUME_ML):
    global STOP_REQUESTED
    if STOP_REQUESTED:
        return

    stroke_ml = _coerce_nonnegative_float(volume_ml, fallback=DEFAULT_MANUAL_ASPIRATION_VOLUME_ML)
    duration_sec = max(ACTUATOR_MIN_SECONDS, ACTUATOR_SECONDS_PER_ML * stroke_ml)
    _trace_log(
        "actuator",
        action=str(action_type),
        volume_ml=stroke_ml,
        duration_sec=duration_sec,
    )

    try:
        from relay_board import RelayBoard
    except Exception:
        logger.warning("RelayBoard not available; skipping actuator action")
        return

    # Legacy relay behavior: open board per action call (no retry/index scan flow).
    try:
        with RelayBoard(active_high=RELAY_ACTIVE_HIGH) as board:
            board.all_off()
            if action_type == "dispense":
                board.toggle(n=4, delay=duration_sec)
            elif action_type == "aspirate":
                board.toggle(n=3, delay=duration_sec)
            elif action_type == "dispose":
                board.toggle(n=2, delay=1.8)  # change if needed
                board.toggle(n=1, delay=1.8)  # change if needed
    except Exception as exc:
        logger.error("[ACTUATOR] %s failed: %s", action_type, exc)
        _trace_log("actuator_error", action=str(action_type), detail=str(exc))
        unity_send_event("actuator_error", f"{action_type}:{exc}")

def get_vial_coords(color_index, use_row_2=False):
    vx = vial_rack_x1 + (color_index * vial_rack_x_diff)
    vy = vial_rack_y1 if not use_row_2 else vial_rack_y1 + vial_rack_y_diff
    return vx, vy

def get_tip_coords(index):
    row = index // cols_pipettes
    col = index % cols_pipettes
    tx = pip_tip_x1 + (row * pip_tip_x_diff)
    ty = pip_tip_y1 + (col * pip_tip_y_diff)
    return tx, ty


def run_tip_pick_cycle(tip_index: int) -> None:
    idx = max(0, int(tip_index))
    tx, ty = get_tip_coords(idx)
    _trace_log(
        "tip_pick_cycle",
        tip_index=idx,
        x=tx,
        y=ty,
        z=pip_tip_z_pickup,
        duration_sec=TIP_PICK_DWELL_SEC,
        detail=f"z_approach={pip_tip_z}",
    )
    logger.info(
        "[TIP] pick cycle idx=%s x=%.2f y=%.2f z_approach=%.2f z_pick=%.2f dwell=%.2fs",
        idx,
        tx,
        ty,
        pip_tip_z,
        pip_tip_z_pickup,
        TIP_PICK_DWELL_SEC,
    )
    move_to(z=safe_height_z, unity_evts=["MOVE"])
    move_to(x=tx, y=ty, unity_evts=["MOVE"])
    move_to(z=pip_tip_z, unity_evts=["MOVE"])
    move_to(z=pip_tip_z_pickup, unity_evts=[f"TIP_PICK_{idx}"])
    if TIP_PICK_DWELL_SEC > 0:
        time.sleep(TIP_PICK_DWELL_SEC)
    move_to(z=safe_height_z, unity_evts=["MOVE"])

def run_mixing_process(
    volumes: List[float],
    start_tip_index: int,
    status_callback,
    aspiration_unit_ml: float = DEFAULT_MANUAL_ASPIRATION_VOLUME_ML,
):
    global STOP_REQUESTED
    STOP_REQUESTED = False
    aspiration_unit_ml = max(0.01, _coerce_nonnegative_float(aspiration_unit_ml, DEFAULT_MANUAL_ASPIRATION_VOLUME_ML))
    _trace_log(
        "mix_start",
        detail=f"start_tip={start_tip_index}",
        volume_ml=aspiration_unit_ml,
    )

    status_callback("Initializing...")
    move_to(z=safe_height_z, feedrate=2000, unity_evts=["INIT"])
    send_gcode("G28 X Y", wait=1)
    move_to(x=home_pos_x, y=home_pos_y, unity_evts=["GCODE:G28 X Y"])

    vial_names = ["RED", "YELLOW", "BLUE", "WATER"]
    current_tip = start_tip_index
    Number_of_Vials = len(vial_names)
    for i in range(Number_of_Vials):
        if STOP_REQUESTED:
            break

        requested_units = _coerce_nonnegative_float(volumes[i], fallback=0.0) if (volumes and i < len(volumes)) else 0.0
        total_req_vol_ml = requested_units * aspiration_unit_ml
        name = vial_names[i]

        if total_req_vol_ml <= 1e-9:
            continue

        vol_row_1 = min(total_req_vol_ml, float(VIAL_MAX_CAPACITY))
        vol_row_2 = max(0.0, total_req_vol_ml - float(VIAL_MAX_CAPACITY))

        status_callback(f"Adding {name} ({_fmt_ml(total_req_vol_ml)} mL)...")
        _trace_log(
            "vial_start",
            vial=name,
            tip_index=current_tip,
            volume_ml=total_req_vol_ml,
            detail=f"row1={vol_row_1};row2={vol_row_2}",
        )
        unity_send_event(f"START_ADD_{name}")

        # Pick tip
        tx, ty = get_tip_coords(current_tip)
        run_tip_pick_cycle(current_tip)

        def transfer_cycles(total_volume_ml: float, use_second_row: bool):
            vx, vy = get_vial_coords(i, use_row_2=use_second_row)
            remaining = max(0.0, float(total_volume_ml))
            total_cycles = max(1, int(math.ceil(remaining / TIP_MAX_CAPACITY_ML)))
            cycle = 0
            while remaining > 1e-9:
                if STOP_REQUESTED:
                    return

                cycle += 1
                cycle_ml = min(TIP_MAX_CAPACITY_ML, remaining)
                _trace_log(
                    "cycle_start",
                    vial=name,
                    tip_index=current_tip,
                    cycle=cycle,
                    total_cycles=total_cycles,
                    row=(2 if use_second_row else 1),
                    volume_ml=cycle_ml,
                )
                status_callback(f"{name}: cycle {cycle}/{total_cycles} ({_fmt_ml(cycle_ml)} mL)")

                actuator_action("dispense", cycle_ml)

                move_to(x=vx, y=vy, unity_evts=["MOVE"])
                move_to(z=vial_rack_z, unity_evts=["ASPIRATE"])
                actuator_action("aspirate", cycle_ml)
                time.sleep(ASPIRATE_SEC)

                move_to(z=safe_height_z, unity_evts=["MOVE"])

                move_to(x=Beaker_x, y=Beaker_y, unity_evts=["MOVE"])
                move_to(z=Beaker_z, unity_evts=["DISPENSE"])
                actuator_action("dispense", cycle_ml)
                time.sleep(DISPENSE_SEC)

                actuator_action("aspirate", cycle_ml)
                unity_send_event("ASPIRATE")

                move_to(z=safe_height_z, unity_evts=["MOVE"])
                remaining -= cycle_ml
                _trace_log(
                    "cycle_end",
                    vial=name,
                    tip_index=current_tip,
                    cycle=cycle,
                    total_cycles=total_cycles,
                    row=(2 if use_second_row else 1),
                    volume_ml=cycle_ml,
                )

        if vol_row_1 > 0:
            transfer_cycles(vol_row_1, False)
        if vol_row_2 > 0 and not STOP_REQUESTED:
            transfer_cycles(vol_row_2, True)

        # Dispose tip
        if STOP_REQUESTED:
            break
        move_to(x=tx, y=ty, unity_evts=["MOVE"])
        move_to(z=Beaker_z, unity_evts=["MOVE"])
        move_to(z=pip_tip_z_drop, unity_evts=["DISPOSE", f"TIP_DISPOSE_{current_tip}"])
        actuator_action("dispose")
        move_to(z=safe_height_z, unity_evts=["MOVE"])
        current_tip += 1

        _trace_log("vial_end", vial=name, tip_index=(current_tip - 1))
        unity_send_event(f"END_ADD_{name}")

    if STOP_REQUESTED:
        _trace_log("mix_end", note="stopped")
        status_callback("STOPPED")
    else:
        _trace_log("mix_end", note="complete")
        status_callback("Complete")
        send_gcode("G1 X0 Y0")
        if unity_stream_enabled():
            unity_send_waypoint(0.0, 0.0, _last_z, ["COMPLETE"])

class SDLHardwareAgent:
    def __init__(self, args):
        self.args = args
        self.control_mode = str(args.control_mode).strip().lower()
        if self.control_mode not in {"sdl", "manual"}:
            self.control_mode = "sdl"

        self._manual_lock = threading.Lock()
        self._manual_waiters: Dict[int, threading.Event] = {}
        self._manual_completed: Set[int] = set()
        self._active_manual_trial_index: Optional[int] = None
        self._manual_proposed_params: Dict[int, Dict[str, Any]] = {}
        self._manual_observed_params: Dict[int, Dict[str, Any]] = {}

        self._command_stop = threading.Event()
        self._command_thread: Optional[threading.Thread] = None

        self.digital_twin_control = bool(getattr(args, "digital_twin_control", False))
        self.require_continue_each_trial = bool(getattr(args, "require_continue_each_trial", True))
        self.sdl_start_timeout = float(getattr(args, "sdl_start_timeout", 7200.0))
        self._sdl_control_lock = threading.Lock()
        self._sdl_can_run = not self.digital_twin_control
        self._sdl_stop_requested = False
        self.manual_aspiration_volume_ml = max(
            0.01,
            _coerce_nonnegative_float(
                getattr(args, "manual_aspiration_volume_ml", DEFAULT_MANUAL_ASPIRATION_VOLUME_ML),
                fallback=DEFAULT_MANUAL_ASPIRATION_VOLUME_ML,
            ),
        )

        logger.info(
            "[SDL] control_mode=%s digital_twin_control=%s require_continue_each_trial=%s manual_aspiration_volume_ml=%s sdl_aspiration_volume_ml=%s",
            self.control_mode,
            self.digital_twin_control,
            self.require_continue_each_trial,
            self.manual_aspiration_volume_ml,
            SDL_FIXED_ASPIRATION_VOLUME_ML,
        )
        if self.digital_twin_control and self.control_mode == "sdl":
            unity_send_event("sdl_waiting_start", "ready")

    def start_unity_command_loop(self) -> None:
        if LIVEKIT_TWIN is None:
            logger.info("[UNITY] command loop disabled (LiveKit not active)")
            return
        if self._command_thread and self._command_thread.is_alive():
            return
        self._command_stop.clear()
        self._command_thread = threading.Thread(target=self._unity_command_loop, daemon=True, name="unity-command-loop")
        self._command_thread.start()
        logger.info("[UNITY] command loop started")

    def _unity_command_loop(self) -> None:
        while not self._command_stop.is_set():
            msg = LIVEKIT_TWIN.recv(timeout=0.5) if LIVEKIT_TWIN else None
            if not msg:
                continue
            try:
                self._dispatch_unity_message(msg)
            except Exception as exc:
                logger.error("[UNITY] command handling error: %s", exc)

    def _dispatch_unity_message(self, msg: Dict[str, Any]) -> None:
        msg_type = str(msg.get("type", "")).strip().lower()
        if not msg_type:
            return

        if msg_type == "command":
            cmd = str(msg.get("cmd", "")).strip()
            args = msg.get("args")
            payload = args if isinstance(args, dict) else {}
            if cmd:
                self._run_command(cmd, payload)
            return

        if msg_type == "set_mode":
            mode = str(msg.get("mode", "")).strip().lower()
            if mode in {"sdl", "manual"}:
                self.control_mode = mode
                logger.info("[UNITY] control mode set to %s", mode)
                unity_send_event("mode_changed", mode)
            else:
                logger.warning("[UNITY] invalid mode: %s", mode)
            return

        if msg_type == "manual_complete":
            try:
                trial_index = int(msg.get("trial_index"))
            except Exception:
                logger.warning("[UNITY] manual_complete missing trial_index")
                return
            logger.info("[UNITY] manual completion received for trial %s", trial_index)
            self._mark_manual_complete(trial_index)
            return

    def _get_active_aspiration_volume_ml(self) -> float:
        if self.control_mode == "sdl":
            return SDL_FIXED_ASPIRATION_VOLUME_ML
        return max(0.01, self.manual_aspiration_volume_ml)

    def _resolve_requested_volume_ml(self, args: Dict[str, Any]) -> float:
        if isinstance(args, dict):
            for key in ("volume_ml", "aspiration_volume_ml", "ml", "value", "units"):
                if key not in args:
                    continue
                parsed = self._safe_float(args.get(key))
                if parsed is None:
                    continue
                if parsed > 0:
                    return float(parsed)
        return self._get_active_aspiration_volume_ml()

    def _set_manual_aspiration_volume(self, raw_value: Any) -> Optional[float]:
        parsed = self._safe_float(raw_value)
        if parsed is None or parsed <= 0:
            return None
        self.manual_aspiration_volume_ml = max(0.01, float(parsed))
        return self.manual_aspiration_volume_ml

    def _run_command(self, cmd: str, args: Dict[str, Any]) -> None:
        global STOP_REQUESTED
        cmd_norm = cmd.strip().lower()
        logger.info("[UNITY] command=%s args=%s", cmd_norm, args)

        if cmd_norm == "home":
            move_to(z=safe_height_z, feedrate=2000, unity_evts=["MOVE"])
            send_gcode("G28 X Y", wait=1.0)
            move_to(x=home_pos_x, y=home_pos_y, unity_evts=["MOVE"])
            return

        if cmd_norm == "open_gripper":
            actuator_action("dispense", self._resolve_requested_volume_ml(args))
            return

        if cmd_norm == "close_gripper":
            actuator_action("aspirate", self._resolve_requested_volume_ml(args))
            return

        if cmd_norm in {"actuator_action", "actuate"}:
            raw_action = str(args.get("action", args.get("mode", args.get("state", "")))).strip().lower()
            volume_ml = self._resolve_requested_volume_ml(args)
            if raw_action in {"aspirate", "in", "on", "true", "1"}:
                actuator_action("aspirate", volume_ml)
                return
            if raw_action in {"dispense", "out", "off", "false", "0"}:
                actuator_action("dispense", volume_ml)
                return
            if raw_action in {"dispose", "open"}:
                actuator_action("dispose")
                return
            unity_send_event("actuator_action_error", f"invalid_action:{raw_action}")
            logger.warning("[UNITY] actuator_action invalid action: %s", raw_action)
            return

        if cmd_norm == "move_to":
            move_to(
                x=args.get("x"),
                y=args.get("y"),
                z=args.get("z"),
                feedrate=args.get("feedrate"),
                unity_evts=["MOVE"],
            )
            return

        if cmd_norm in {"pick_tip", "tip_pick"}:
            tip_raw = args.get("tip_index", args.get("index", 0))
            try:
                tip_index = int(float(tip_raw))
            except Exception:
                tip_index = 0
            run_tip_pick_cycle(tip_index)
            unity_send_event("tip_pick_manual", str(max(0, tip_index)))
            return

        if cmd_norm == "start_mix":
            self._start_mix_from_command(args)
            return

        if cmd_norm == "start":
            STOP_REQUESTED = False
            self._set_sdl_run_state(can_run=True, stop_requested=False)
            unity_send_event("start", "requested_by_unity")
            return

        if cmd_norm == "continue":
            STOP_REQUESTED = False
            self._set_sdl_run_state(can_run=True, stop_requested=False)
            unity_send_event("continue", "requested_by_unity")
            return

        if cmd_norm == "stop":
            STOP_REQUESTED = True
            self._set_sdl_run_state(can_run=False, stop_requested=True)
            unity_send_event("stop", "requested_by_unity")
            return

        if cmd_norm == "validate":
            trial_index = args.get("trial_index", self._active_manual_trial_index)
            if trial_index is None:
                unity_send_event("manual_validate_error", "missing_trial_index")
                logger.warning("[UNITY] validate requested without active manual trial")
                return
            try:
                trial_int = int(trial_index)
            except Exception:
                unity_send_event("manual_validate_error", f"invalid_trial_index:{trial_index}")
                logger.warning("[UNITY] validate invalid trial_index=%s", trial_index)
                return
            self._apply_manual_parameter_override(trial_int, args)
            logger.info("[UNITY] validate received for trial %s", trial_int)
            self._mark_manual_complete(trial_int)
            unity_send_event("manual_validated", str(trial_int))
            return

        if cmd_norm in {"manual_adjust", "update_candidate", "set_candidate_parameters"}:
            trial_index = args.get("trial_index", self._active_manual_trial_index)
            try:
                trial_int = int(trial_index)
            except Exception:
                unity_send_event("manual_adjust_error", "missing_or_invalid_trial_index")
                logger.warning("[UNITY] manual_adjust invalid trial_index=%s", trial_index)
                return
            changed = self._apply_manual_parameter_override(trial_int, args)
            unity_send_event("manual_candidate_updated", f"{trial_int}:{'changed' if changed else 'no_change'}")
            logger.info("[UNITY] manual candidate update trial=%s changed=%s args=%s", trial_int, changed, args)
            return

        if cmd_norm == "set_speed":
            value = args.get("feedrate", args.get("value"))
            if value is not None:
                send_gcode(f"G1 F{float(value):.2f}", wait=0.05)
            return

        if cmd_norm in {"set_aspiration_volume", "set_aspiration", "set_volume_step"}:
            value = args.get("volume_ml", args.get("aspiration_volume_ml", args.get("ml", args.get("value"))))
            updated = self._set_manual_aspiration_volume(value)
            if updated is None:
                unity_send_event("aspiration_volume_error", "invalid_or_missing_value")
                logger.warning("[UNITY] invalid aspiration volume payload: %s", args)
                return
            logger.info("[UNITY] manual aspiration volume updated to %.4f ml", updated)
            unity_send_event("aspiration_volume_changed", f"{updated:.4f}")
            return

        if cmd_norm == "set_pump":
            mode = str(args.get("mode", args.get("state", ""))).strip().lower()
            volume_ml = self._resolve_requested_volume_ml(args)
            if mode in {"aspirate", "in", "on", "true", "1"}:
                actuator_action("aspirate", volume_ml)
            elif mode in {"dispense", "out", "off", "false", "0"}:
                actuator_action("dispense", volume_ml)
            return

        if cmd_norm == "set_valve":
            mode = str(args.get("mode", args.get("state", ""))).strip().lower()
            if mode in {"dispose", "open"}:
                actuator_action("dispose")
            return

        logger.warning("[UNITY] unsupported command '%s'", cmd_norm)
        unity_send_event("command_unsupported", cmd_norm)

    def _start_mix_from_command(self, args: Dict[str, Any]) -> None:
        params = args if isinstance(args, dict) else {}
        try:
            volumes = self._extract_volumes(params)
            tip_index = int(params.get("tip_index", 0))
            aspiration_unit_ml = self._resolve_requested_volume_ml(params)
        except Exception as exc:
            logger.error("[UNITY] start_mix invalid args: %s", exc)
            unity_send_event("start_mix_error", str(exc))
            return

        def _runner() -> None:
            try:
                run_mixing_process(
                    volumes,
                    tip_index,
                    status_callback=lambda m: logger.info("[UNITY-MIX] %s", m),
                    aspiration_unit_ml=aspiration_unit_ml,
                )
            except Exception as exc:
                logger.error("[UNITY] start_mix failure: %s", exc)
                unity_send_event("start_mix_error", str(exc))

        threading.Thread(target=_runner, daemon=True, name="unity-start-mix").start()

    def _mark_manual_complete(self, trial_index: int) -> None:
        with self._manual_lock:
            evt = self._manual_waiters.get(trial_index)
            if evt:
                evt.set()
            else:
                self._manual_completed.add(trial_index)

    def _wait_for_manual_complete(self, trial_index: int, timeout: float) -> bool:
        with self._manual_lock:
            if trial_index in self._manual_completed:
                self._manual_completed.discard(trial_index)
                return True
            evt = self._manual_waiters.get(trial_index)
            if evt is None:
                evt = threading.Event()
                self._manual_waiters[trial_index] = evt

        completed = evt.wait(timeout=timeout)

        with self._manual_lock:
            self._manual_waiters.pop(trial_index, None)
            if trial_index in self._manual_completed:
                self._manual_completed.discard(trial_index)
                completed = True
        return completed

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _resolve_color_key(params: Dict[str, Any], color_name: str) -> str:
        color = str(color_name or "").strip().lower()
        aliases = {
            "red": ["vol_r", "red", "r", "ml_red"],
            "yellow": ["vol_y", "yellow", "y", "ml_yellow"],
            "blue": ["vol_b", "blue", "b", "ml_blue"],
            "water": ["vol_w", "water", "w", "ml_water"],
        }
        candidates = aliases.get(color, [])
        for key in candidates:
            if key in params:
                return key
        return candidates[0] if candidates else color

    def _apply_manual_parameter_override(self, trial_index: Optional[int], args: Dict[str, Any]) -> bool:
        if trial_index is None:
            return False

        if not isinstance(args, dict):
            return False

        observed = dict(self._manual_observed_params.get(trial_index, {}))
        changed = False

        direct_obj = args.get("observed_parameters", args.get("parameters"))
        if isinstance(direct_obj, dict):
            for key, val in direct_obj.items():
                if key is None:
                    continue
                observed[str(key)] = val
                changed = True

        deltas = args.get("delta")
        if isinstance(deltas, dict):
            for key, val in deltas.items():
                if key is None:
                    continue
                sval = self._safe_float(val)
                cur = self._safe_float(observed.get(str(key)))
                if sval is None or cur is None:
                    continue
                observed[str(key)] = cur + sval
                changed = True

        color = args.get("color", args.get("component"))
        if color is not None:
            delta_ml = self._safe_float(args.get("delta_ml"))
            value_ml = self._safe_float(args.get("ml", args.get("value_ml")))
            key = self._resolve_color_key(observed, str(color))
            if value_ml is not None:
                observed[key] = value_ml
                changed = True
            elif delta_ml is not None:
                cur = self._safe_float(observed.get(key))
                if cur is not None:
                    observed[key] = cur + delta_ml
                    changed = True

        if changed:
            self._manual_observed_params[trial_index] = observed
        return changed

    def _manual_candidate_was_modified(self, trial_index: int) -> bool:
        proposed = self._manual_proposed_params.get(trial_index, {})
        observed = self._manual_observed_params.get(trial_index, proposed)
        keys = set(proposed.keys()) | set(observed.keys())
        for key in keys:
            p = proposed.get(key)
            o = observed.get(key)
            pf = self._safe_float(p)
            of = self._safe_float(o)
            if pf is not None and of is not None:
                if abs(pf - of) > 1e-9:
                    return True
            else:
                if p != o:
                    return True
        return False

    def _set_sdl_run_state(self, can_run: Optional[bool] = None, stop_requested: Optional[bool] = None) -> None:
        with self._sdl_control_lock:
            if can_run is not None:
                self._sdl_can_run = bool(can_run)
            if stop_requested is not None:
                self._sdl_stop_requested = bool(stop_requested)

    def _wait_for_sdl_permission(
        self,
        trial_index: Optional[Any],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.digital_twin_control:
            return

        candidate_trial = self._coerce_trial_index(trial_index)
        candidate_params = dict(params or {})
        label = str(trial_index) if trial_index is not None else "unknown"
        unity_send_event("sdl_waiting_start", label)
        deadline = time.time() + max(self.sdl_start_timeout, 1.0)
        next_candidate_heartbeat = 0.0

        while True:
            now = time.time()
            # Re-publish pending candidate while waiting so Unity UI always receives parameters.
            if candidate_params and now >= next_candidate_heartbeat:
                unity_send_candidate(candidate_trial, candidate_params, mode=self.control_mode)
                next_candidate_heartbeat = now + 2.0

            with self._sdl_control_lock:
                if self._sdl_stop_requested:
                    raise RuntimeError("stopped_by_unity")
                if self._sdl_can_run:
                    break

            if now >= deadline:
                raise TimeoutError(f"sdl start/continue timeout for trial_index={label}")
            time.sleep(0.2)

    @staticmethod
    def _coerce_trial_index(trial_index: Optional[Any]) -> int:
        if trial_index is None:
            return int(time.time() * 1000)
        try:
            return int(trial_index)
        except Exception:
            return int(time.time() * 1000)

    def _measure_frequency_payload(self, trial_index: Optional[Any]) -> Dict[str, Any]:
        requested = max(1, int(getattr(self.args, "sensor_replicates", 1)))
        stats = read_frequency_stats(
            sample_count=requested,
            timeout=float(self.args.sensor_timeout),
            sample_interval=float(getattr(self.args, "sensor_sample_interval", 0.0)),
            require_fresh_samples=bool(getattr(self.args, "sensor_require_fresh_samples", True)),
            max_age_sec=float(getattr(self.args, "sensor_max_age_sec", 3.0)),
        )
        if stats is None:
            raise RuntimeError("No frequency reading from sensor")

        n_samples = int(stats.get("n_samples", 0))
        if n_samples <= 0:
            raise RuntimeError("No frequency reading from sensor")

        allow_partial = bool(getattr(self.args, "sensor_allow_partial_samples", True))
        if bool(stats.get("incomplete", False)) and not allow_partial:
            raise RuntimeError(
                f"Incomplete frequency replicate set ({n_samples}/{requested}) and partial samples are disabled"
            )

        freq_mean = float(stats.get("mean", 0.0))
        sem_value = abs(float(stats.get("sem", 0.0)))
        std_value = abs(float(stats.get("std", 0.0)))
        fallback_sem = max(0.0, float(getattr(self.args, "sensor_sem_fallback", 0.0)))

        if n_samples < 2:
            sem_value = max(sem_value, fallback_sem)
            std_value = max(std_value, fallback_sem)

        spread_mode = str(getattr(self.args, "sensor_uncertainty_type", "sem")).strip().lower()
        if spread_mode in {"std", "stdev", "sigma"}:
            uncertainty_value = std_value
            uncertainty_type = "std"
        else:
            uncertainty_value = sem_value
            uncertainty_type = "sem"

        if bool(stats.get("incomplete", False)):
            logger.warning(
                "[SDL] frequency replicates incomplete for trial %s: collected=%s requested=%s",
                trial_index,
                n_samples,
                requested,
            )

        return {
            "objectives": {"frequency": freq_mean},
            "objective_uncertainties": {"frequency": float(uncertainty_value)},
            "uncertainty_type": uncertainty_type,
            "measurement_metadata": {
                "frequency_samples_collected": n_samples,
                "frequency_samples_requested": requested,
                "frequency_sem": float(sem_value),
                "frequency_std": float(std_value),
                "frequency_latest_color": str(stats.get("latest_color", "")),
            },
        }

    def _run_automatic_candidate(self, trial_index: Optional[Any], params: Dict[str, Any]) -> Dict[str, Any]:
        global STOP_REQUESTED
        vols = self._extract_volumes(params)
        tip_index = int(params.get("tip_index", 0))
        aspiration_unit_ml = SDL_FIXED_ASPIRATION_VOLUME_ML

        def status(msg: str):
            logger.info(msg)

        self._wait_for_sdl_permission(trial_index, params)
        self._set_sdl_run_state(stop_requested=False)
        STOP_REQUESTED = False
        unity_send_event("automatic_trial_started", str(trial_index))
        run_mixing_process(
            vols,
            tip_index,
            status_callback=status,
            aspiration_unit_ml=aspiration_unit_ml,
        )
        if STOP_REQUESTED:
            self._set_sdl_run_state(can_run=False, stop_requested=True)
            raise RuntimeError("stopped_by_unity")

        measurement_payload = self._measure_frequency_payload(trial_index)

        if self.digital_twin_control and self.require_continue_each_trial:
            self._set_sdl_run_state(can_run=False, stop_requested=False)
            unity_send_event("sdl_waiting_continue", str(trial_index))

        return {
            "trial_index": trial_index,
            "status": "ok",
            **measurement_payload,
        }

    def _run_manual_candidate(self, trial_index: Optional[Any], params: Dict[str, Any]) -> Dict[str, Any]:
        manual_trial_index = self._coerce_trial_index(trial_index)
        self._active_manual_trial_index = manual_trial_index
        self._manual_proposed_params[manual_trial_index] = dict(params or {})
        self._manual_observed_params[manual_trial_index] = dict(params or {})
        unity_send_manual_candidate(manual_trial_index, params)
        unity_send_event("manual_trial_waiting", str(manual_trial_index))

        try:
            completed = self._wait_for_manual_complete(manual_trial_index, timeout=self.args.manual_timeout)
            if not completed:
                raise TimeoutError(f"manual_complete timeout for trial_index={manual_trial_index}")

            measurement_payload = self._measure_frequency_payload(manual_trial_index)
            observed_parameters = dict(self._manual_observed_params.get(manual_trial_index, params or {}))
            proposed_parameters = dict(self._manual_proposed_params.get(manual_trial_index, params or {}))
            candidate_modified = self._manual_candidate_was_modified(manual_trial_index)
            if candidate_modified:
                unity_send_event("manual_candidate_modified", str(manual_trial_index))
            return {
                "trial_index": trial_index,
                "status": "ok",
                **measurement_payload,
                "candidate_modified": candidate_modified,
                "observed_parameters": observed_parameters,
                "proposed_parameters": proposed_parameters,
            }
        finally:
            self._active_manual_trial_index = None
            self._manual_proposed_params.pop(manual_trial_index, None)
            self._manual_observed_params.pop(manual_trial_index, None)

    def handle_candidate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("type") == "ping":
            return {"status": "ok", "message": "pong"}

        trial_index = payload.get("trial_index")
        params = payload.get("parameters", {})
        control_cfg = payload.get("control")
        if isinstance(control_cfg, dict):
            if "digital_twin_control" in control_cfg:
                self.digital_twin_control = bool(control_cfg.get("digital_twin_control"))
                if self.digital_twin_control:
                    self._set_sdl_run_state(can_run=False, stop_requested=False)
                    unity_send_event("sdl_waiting_start", "ready")
                else:
                    self._set_sdl_run_state(can_run=True, stop_requested=False)
                logger.info("[SDL] digital_twin_control updated from payload: %s", self.digital_twin_control)
            if "require_continue_each_trial" in control_cfg:
                self.require_continue_each_trial = bool(control_cfg.get("require_continue_each_trial"))
                logger.info(
                    "[SDL] require_continue_each_trial updated from payload: %s",
                    self.require_continue_each_trial,
                )
        logger.info(f"Received candidate trial_index={trial_index} params={params}")
        if not params:
            return {"trial_index": trial_index, "status": "ok", "message": "no parameters; skipped"}
        candidate_trial = self._coerce_trial_index(trial_index)
        _set_active_trial_index(candidate_trial)

        try:
            unity_send_candidate(candidate_trial, params, mode=self.control_mode)
            logger.info(
                "[UNITY] candidate forwarded trial_index=%s mode=%s params=%s",
                candidate_trial,
                self.control_mode,
                params,
            )
            if self.control_mode == "manual":
                return self._run_manual_candidate(candidate_trial, params)
            return self._run_automatic_candidate(candidate_trial, params)

        except Exception as exc:
            logger.error(f"[SDL] error: {exc}")
            return {"trial_index": trial_index, "status": "error", "message": str(exc)}
        finally:
            _set_active_trial_index(None)

    def _extract_volumes(self, params: Dict[str, Any]) -> List[float]:
        lower_params = {str(k).lower(): v for k, v in params.items()}

        def pick(values: Dict[str, Any], key: str, default: float = 0.0) -> float:
            parsed = self._safe_float(values.get(key))
            if parsed is None:
                return default
            return max(0.0, float(parsed))

        keys = ["vol_r", "vol_y", "vol_b", "vol_w"]
        if all(k in lower_params for k in keys):
            return [pick(lower_params, k, 0.0) for k in keys]

        alt_keys = ["red", "yellow", "blue", "water"]
        if all(k in lower_params for k in alt_keys):
            return [pick(lower_params, k, 0.0) for k in alt_keys]

        total = int(max(0.0, pick(lower_params, "total_vol", 8.0)))
        base = total // 4
        return [float(base), float(base), float(base), float(total - 3 * base)]

    def run_mqtt(self):
        if not MQTT_AVAILABLE:
            raise RuntimeError("paho-mqtt not installed")

        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                if payload.get("type") == "ping":
                    client.publish(self.args.mqtt_response_topic, json.dumps({"status": "ok", "message": "pong"}))
                    return
                reply = self.handle_candidate(payload)
                client.publish(self.args.mqtt_response_topic, json.dumps(reply))
                print(f"[MQTT] handled trial {reply.get('trial_index')}")
            except Exception as exc:
                print(f"[MQTT] error: {exc}")

        client = mqtt.Client()
        client.on_message = on_message
        client.connect(self.args.mqtt_host, self.args.mqtt_port, keepalive=30)
        client.subscribe(self.args.mqtt_command_topic)
        print(f"[MQTT] listening on {self.args.mqtt_host}:{self.args.mqtt_port} topic={self.args.mqtt_command_topic}")
        client.loop_forever()

    def run_http(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        agent = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, data: Dict[str, Any]):
                body = json.dumps(data).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                reply = agent.handle_candidate(payload)
                self._send(200, reply)

            def do_GET(self):  # noqa: N802
                self._send(200, {"status": "ok", "message": "SDL agent alive"})

            def log_message(self, fmt, *args):
                return

        server = HTTPServer((self.args.http_host, self.args.http_port), Handler)
        print(f"[HTTP] listening on http://{self.args.http_host}:{self.args.http_port}")
        server.serve_forever()

    def run_tcp(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.args.tcp_host, self.args.tcp_port))
        srv.listen(5)
        print(f"[TCP] listening on {self.args.tcp_host}:{self.args.tcp_port}")

        def handle_client(conn, addr):
            with conn:
                buffer = b""
                while True:
                    try:
                        chunk = conn.recv(4096)
                    except (ConnectionAbortedError, ConnectionResetError, OSError):
                        break
                    if not chunk:
                        break
                    buffer += chunk
                    if b"\n" in buffer:
                        line, _, buffer = buffer.partition(b"\n")
                        try:
                            payload = json.loads(line.decode("utf-8"))
                            if payload.get("type") == "ping":
                                try:
                                    conn.sendall(b'{"status":"ok","message":"pong"}\n')
                                except (ConnectionResetError, BrokenPipeError, OSError) as send_err:
                                    logger.warning("[TCP] ping reply dropped for %s: %s", addr, send_err)
                                    break
                                continue
                            reply = self.handle_candidate(payload)
                            try:
                                conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                            except (ConnectionResetError, BrokenPipeError, OSError) as send_err:
                                logger.warning("[TCP] client disconnected before reply for %s: %s", addr, send_err)
                                break
                            print(f"[TCP] handled trial {reply.get('trial_index')} from {addr}")
                        except Exception as exc:
                            try:
                                conn.sendall((json.dumps({"status": "error", "message": str(exc)}) + "\n").encode("utf-8"))
                            except (ConnectionResetError, BrokenPipeError, OSError) as send_err:
                                logger.warning("[TCP] could not send error reply to %s: %s", addr, send_err)
                                break

        while True:
            conn, addr = srv.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

    def run_serial(self):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial not installed")
        ser = serial.Serial(self.args.serial_port, self.args.serial_baud, timeout=1)
        print(f"[Serial] listening on {self.args.serial_port} @ {self.args.serial_baud}")
        while True:
            line = ser.readline()
            if not line:
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
                reply = self.handle_candidate(payload)
                ser.write((json.dumps(reply) + "\n").encode("utf-8"))
                ser.flush()
                print(f"[Serial] handled trial {reply.get('trial_index')}")
            except Exception as exc:
                ser.write(json.dumps({"status": "error", "message": str(exc)}).encode("utf-8"))

def parse_args():
    parser = argparse.ArgumentParser(description="Self-driving lab hardware agent (headless).")
    parser.add_argument("--protocol", choices=["mqtt", "http", "tcp", "serial"], default="http")
    # MQTT
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-command-topic", default="bo/commands")
    parser.add_argument("--mqtt-response-topic", default="bo/results")
    # HTTP
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8000)
    # TCP
    parser.add_argument("--tcp-host", default="0.0.0.0")
    parser.add_argument("--tcp-port", type=int, default=7000)
    # Serial (platform)
    parser.add_argument("--serial-port", default="COM3")
    parser.add_argument("--serial-baud", type=int, default=115200)
    # Logging and timing
    parser.add_argument("--log-file", default="sdl_agent.log", help="Path to log file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument(
        "--trace-csv",
        default=os.getenv("SDL_TRACE_CSV", ""),
        help="Optional CSV trace path for executed motion/actions. Supports {ts} placeholder.",
    )
    parser.add_argument("--sensor-timeout", type=float, default=30.0, help="Seconds to wait for sensor frequency reading")
    parser.add_argument(
        "--sensor-replicates",
        type=int,
        default=int(os.getenv("SDL_SENSOR_REPLICATES", "3")),
        help="Number of repeated frequency samples per trial for uncertainty estimation.",
    )
    parser.add_argument(
        "--sensor-sample-interval",
        type=float,
        default=float(os.getenv("SDL_SENSOR_SAMPLE_INTERVAL", "0.0")),
        help="Optional delay (seconds) between replicate samples.",
    )
    parser.add_argument(
        "--sensor-max-age-sec",
        type=float,
        default=float(os.getenv("SDL_SENSOR_MAX_AGE_SEC", "3.0")),
        help="Maximum age (seconds) for using cached sensor values.",
    )
    parser.add_argument(
        "--sensor-require-fresh-samples",
        action=argparse.BooleanOptionalAction,
        default=_parse_env_bool(os.getenv("SDL_SENSOR_REQUIRE_FRESH_SAMPLES"), default=True),
        help="If enabled, each replicate must come from a new sensor timestamp.",
    )
    parser.add_argument(
        "--sensor-allow-partial-samples",
        action=argparse.BooleanOptionalAction,
        default=_parse_env_bool(os.getenv("SDL_SENSOR_ALLOW_PARTIAL_SAMPLES"), default=True),
        help="If enabled, trials can proceed when fewer than requested replicates are collected before timeout.",
    )
    parser.add_argument(
        "--sensor-uncertainty-type",
        choices=["sem", "std"],
        default=str(os.getenv("SDL_SENSOR_UNCERTAINTY_TYPE", "sem")).strip().lower(),
        help="Which uncertainty to return to CEID in objective_uncertainties.",
    )
    parser.add_argument(
        "--sensor-sem-fallback",
        type=float,
        default=float(os.getenv("SDL_SENSOR_SEM_FALLBACK", "0.0")),
        help="Fallback uncertainty used when only one replicate is available.",
    )
    parser.add_argument("--manual-timeout", type=float, default=1800.0, help="Seconds to wait for manual trial completion")
    parser.add_argument("--control-mode", choices=["sdl", "manual"], default=os.getenv("SDL_CONTROL_MODE", "sdl"))
    parser.add_argument(
        "--manual-aspiration-volume-ml",
        type=float,
        default=float(os.getenv("SDL_MANUAL_ASPIRATION_VOLUME_ML", str(DEFAULT_MANUAL_ASPIRATION_VOLUME_ML))),
        help="Manual mode aspiration volume in mL for set_pump/open_gripper/close_gripper and start_mix.",
    )
    parser.add_argument(
        "--digital-twin-control",
        action=argparse.BooleanOptionalAction,
        default=_parse_env_bool(os.getenv("SDL_DIGITAL_TWIN_CONTROL"), default=False),
        help="If enabled, automatic SDL trials wait for Unity start/continue commands.",
    )
    parser.add_argument(
        "--require-continue-each-trial",
        action=argparse.BooleanOptionalAction,
        default=_parse_env_bool(os.getenv("SDL_REQUIRE_CONTINUE_EACH_TRIAL"), default=True),
        help="If digital twin control is enabled, require continue after each completed automatic trial.",
    )
    parser.add_argument(
        "--sdl-start-timeout",
        type=float,
        default=float(os.getenv("SDL_START_TIMEOUT", "7200")),
        help="Max wait time (seconds) for Unity start/continue before timing out an automatic trial.",
    )
    parser.add_argument(
        "--relay-device-index",
        type=int,
        default=None,
        help="FTDI relay device index (default: auto-detect).",
    )
    parser.add_argument(
        "--relay-active-high",
        action=argparse.BooleanOptionalAction,
        default=_parse_env_bool(os.getenv("SDL_RELAY_ACTIVE_HIGH"), default=True),
        help="Relay logic mode. Use --no-relay-active-high for active-low boards.",
    )
    parser.add_argument(
        "--relay-auto-scan",
        action=argparse.BooleanOptionalAction,
        default=_parse_env_bool(os.getenv("SDL_RELAY_AUTO_SCAN"), default=True),
        help="Auto-scan all FTDI indices if explicit relay index fails.",
    )
    parser.add_argument(
        "--relay-open-retries",
        type=int,
        default=int(os.getenv("SDL_RELAY_OPEN_RETRIES", "2")),
        help="Retry attempts for opening the FTDI relay board.",
    )

    # Unity transport options
    parser.add_argument("--unity-enable", action="store_true", help="Enable Unity digital-twin streaming")
    parser.add_argument("--unity-transport", choices=["livekit", "tcp", "none"], default=os.getenv("UNITY_TRANSPORT", "livekit"))
    parser.add_argument("--unity-host", default="0.0.0.0", help="Bind host for Unity twin TCP server fallback")
    parser.add_argument("--unity-port", type=int, default=7100, help="Bind port for Unity twin TCP server fallback")

    # LiveKit options
    parser.add_argument("--livekit-url", default=os.getenv("LIVEKIT_URL", LIVEKIT_DEFAULT_URL))
    parser.add_argument("--livekit-room", default=os.getenv("LIVEKIT_ROOM", LIVEKIT_DEFAULT_ROOM))
    parser.add_argument("--livekit-topic", default=os.getenv("LIVEKIT_TOPIC", LIVEKIT_DEFAULT_TOPIC))
    parser.add_argument("--sdl-livekit-token", default=os.getenv("SDL_LIVEKIT_TOKEN", ""))
    parser.add_argument("--unity-dest-identity", default=os.getenv("UNITY_DEST_IDENTITY", "unity"))

    # Arduino port override
    parser.add_argument("--arduino-port", default=ARDUINO_PORT)
    parser.add_argument("--arduino-baud", type=int, default=BAUD_RATE)
    return parser.parse_args()


def _setup_unity_transport(args) -> None:
    global UNITY_TRANSPORT, LIVEKIT_TWIN, UNITY_DEST_IDENTITY, UNITY_TOPIC

    UNITY_DEST_IDENTITY = str(args.unity_dest_identity).strip() or "unity"
    UNITY_TOPIC = str(args.livekit_topic).strip() or LIVEKIT_DEFAULT_TOPIC
    requested_transport = str(args.unity_transport).strip().lower()

    env_enable = _parse_env_bool(os.getenv("UNITY_ENABLE"), default=False)
    auto_enable_livekit = bool(args.sdl_livekit_token and requested_transport == "livekit")
    auto_enable_tcp = requested_transport == "tcp"
    unity_enabled = bool(args.unity_enable or env_enable or auto_enable_livekit or auto_enable_tcp)
    UNITY_TRANSPORT = "none"

    if not unity_enabled:
        logger.info("[UNITY] twin transport disabled")
        return

    if requested_transport == "livekit":
        if LIVEKIT_CLIENT_AVAILABLE and args.sdl_livekit_token:
            try:
                LIVEKIT_TWIN = LiveKitTwinClient(
                    url=args.livekit_url,
                    token=args.sdl_livekit_token,
                    room_name=args.livekit_room,
                    topic=UNITY_TOPIC,
                    logger=logger,
                )
                LIVEKIT_TWIN.start()
                UNITY_TRANSPORT = "livekit"
                logger.info(
                    "[UNITY] LiveKit transport enabled: url=%s room=%s topic=%s",
                    args.livekit_url,
                    args.livekit_room,
                    UNITY_TOPIC,
                )
            except Exception as exc:
                logger.error("[UNITY] LiveKit init failed: %s", exc)
        else:
            if not LIVEKIT_CLIENT_AVAILABLE:
                logger.warning("[UNITY] livekit python package missing; cannot enable LiveKit transport")
            if not args.sdl_livekit_token:
                logger.warning("[UNITY] SDL_LIVEKIT_TOKEN missing; cannot enable LiveKit transport")

    if UNITY_TRANSPORT == "none" and requested_transport in {"tcp", "livekit"}:
        UNITY_TWIN.start(args.unity_host, args.unity_port)
        UNITY_TRANSPORT = "tcp"
        logger.info("[UNITY] TCP transport enabled on %s:%s", args.unity_host, args.unity_port)


if __name__ == "__main__":
    args = parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
        ],
    )
    logger.setLevel(log_level)
    trace_csv_path = str(getattr(args, "trace_csv", "") or "").strip()
    if trace_csv_path:
        _open_trace_csv(trace_csv_path)
        _trace_log("session_start", note=f"protocol={args.protocol}")

    # apply Arduino overrides
    ARDUINO_PORT = args.arduino_port
    BAUD_RATE = args.arduino_baud
    RELAY_DEVICE_INDEX = args.relay_device_index
    if RELAY_DEVICE_INDEX is None:
        relay_env_idx = str(os.getenv("SDL_RELAY_DEVICE_INDEX", "")).strip()
        if relay_env_idx:
            try:
                RELAY_DEVICE_INDEX = int(relay_env_idx)
            except Exception:
                logger.warning("[ACTUATOR] invalid SDL_RELAY_DEVICE_INDEX='%s' (ignored)", relay_env_idx)
    RELAY_ACTIVE_HIGH = bool(args.relay_active_high)
    RELAY_AUTO_SCAN = bool(args.relay_auto_scan)
    RELAY_OPEN_RETRIES = max(1, int(args.relay_open_retries))

    logger.info(
        "[ACTUATOR] relay config index=%s active_high=%s auto_scan=%s open_retries=%s",
        RELAY_DEVICE_INDEX,
        RELAY_ACTIVE_HIGH,
        RELAY_AUTO_SCAN,
        RELAY_OPEN_RETRIES,
    )
    logger.info(
        "[TIP] z_approach=%s z_pickup=%s z_drop=%s dwell=%ss",
        pip_tip_z,
        pip_tip_z_pickup,
        pip_tip_z_drop,
        TIP_PICK_DWELL_SEC,
    )
    logger.info(
        "[SENSOR] uncertainty config replicates=%s interval=%ss max_age=%ss require_fresh=%s allow_partial=%s type=%s fallback_sem=%s",
        max(1, int(args.sensor_replicates)),
        max(0.0, float(args.sensor_sample_interval)),
        max(0.05, float(args.sensor_max_age_sec)),
        bool(args.sensor_require_fresh_samples),
        bool(args.sensor_allow_partial_samples),
        str(args.sensor_uncertainty_type).lower(),
        max(0.0, float(args.sensor_sem_fallback)),
    )
    if trace_csv_path:
        logger.info("[TRACE] writing motion/action trace to %s", _TRACE_PATH)

    _setup_unity_transport(args)

    # Start sensor streaming always so Unity receives periodic frequency updates.
    _start_sensor_thread()

    agent = SDLHardwareAgent(args)
    agent.start_unity_command_loop()
    if args.protocol == "mqtt":
        agent.run_mqtt()
    elif args.protocol == "tcp":
        agent.run_tcp()
    elif args.protocol == "serial":
        agent.run_serial()
    else:
        agent.run_http()
