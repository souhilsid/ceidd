# utils/state_manager.py - simple checkpoint utilities
import json
import os
import tempfile
from typing import Any, Dict
import numpy as np


def _default(obj: Any):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: str, payload: Dict[str, Any]) -> str:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_default)
    return path


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_ax_to_dict(ax_client) -> Dict[str, Any]:
    """Persist AxClient into a dict (round-trips through a temp file)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_path = tmp.name
    tmp.close()
    try:
        ax_client.save_to_json_file(tmp_path)
        return load_json(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def load_ax_from_dict(ax_client_cls, payload: Dict[str, Any]):
    """Recreate AxClient from a dict produced by dump_ax_to_dict."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_path = tmp.name
    tmp.close()
    try:
        save_json(tmp_path, payload)
        return ax_client_cls.load_from_json_file(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
