import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _sanitize_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value.strip())
    return cleaned or "file"


def _resolve_storage_root() -> Path:
    candidates = []
    env_root = os.getenv("CEID_STORAGE_DIR")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    project_root = _project_root()
    candidates.append(project_root / ".ceid_runtime")
    candidates.append(Path(tempfile.gettempdir()) / "ceid_runtime")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue

    raise OSError("Unable to create a writable CEID storage directory. Set CEID_STORAGE_DIR to a writable path.")


@dataclass(frozen=True)
class RuntimePaths:
    storage_root: Path
    uploads_dir: Path
    datasets_dir: Path
    configs_dir: Path
    models_dir: Path
    temp_dir: Path
    checkpoints_dir: Path
    exports_dir: Path
    database_path: Path


def get_runtime_paths() -> RuntimePaths:
    storage_root = _resolve_storage_root()

    uploads_dir = Path(os.getenv("CEID_UPLOADS_DIR", storage_root / "uploads")).expanduser()
    datasets_dir = Path(os.getenv("CEID_DATASETS_DIR", uploads_dir / "datasets")).expanduser()
    configs_dir = Path(os.getenv("CEID_CONFIGS_DIR", uploads_dir / "configs")).expanduser()
    models_dir = Path(os.getenv("CEID_MODELS_DIR", uploads_dir / "models")).expanduser()
    temp_dir = Path(os.getenv("CEID_TEMP_DIR", storage_root / "tmp")).expanduser()
    checkpoints_dir = Path(os.getenv("CEID_CHECKPOINTS_DIR", storage_root / "checkpoints")).expanduser()
    exports_dir = Path(os.getenv("CEID_EXPORTS_DIR", storage_root / "exports")).expanduser()
    database_path = Path(os.getenv("CEID_DB_PATH", exports_dir / "bo_platform.db")).expanduser()

    for path in [
        storage_root,
        uploads_dir,
        datasets_dir,
        configs_dir,
        models_dir,
        temp_dir,
        checkpoints_dir,
        exports_dir,
        database_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return RuntimePaths(
        storage_root=storage_root,
        uploads_dir=uploads_dir,
        datasets_dir=datasets_dir,
        configs_dir=configs_dir,
        models_dir=models_dir,
        temp_dir=temp_dir,
        checkpoints_dir=checkpoints_dir,
        exports_dir=exports_dir,
        database_path=database_path,
    )


def persist_uploaded_bytes(
    content: bytes,
    original_name: str,
    target_dir: Path,
    prefix: Optional[str] = None,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(original_name).name or "file"
    safe_name = _sanitize_name(source_name)
    stem = _sanitize_name(Path(safe_name).stem)
    suffix = Path(safe_name).suffix
    name_prefix = f"{_sanitize_name(prefix)}_" if prefix else ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    destination = target_dir / f"{name_prefix}{stem}_{timestamp}{suffix}"
    destination.write_bytes(content)
    return destination