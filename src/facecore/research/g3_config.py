"""G3 local config: enrollment folder, models, store/key paths (W6).

Source of truth: docs/specs/2026-09-24-g3-local-test-app.md §3 (local
config lives under ~/Downloads/face_sample/_facecore/ and never enters
Git; the repo holds a template with no real uids or machine paths).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


G3_CONFIG_TEMPLATE_PATH = (
    Path(__file__).parents[3] / "docs" / "g3-local-config.example.json"
)

_REQUIRED_FIELDS = ("enrollment_dir", "models_dir", "store_dir", "key_dir")


@dataclass(frozen=True)
class G3Config:
    """Resolved local paths for one G3 test-app run."""

    enrollment_dir: Path
    models_dir: Path
    store_dir: Path
    key_dir: Path


def load_g3_config(path: Path) -> G3Config:
    """Load and resolve a G3 local config (``~`` and relative expanded)."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"g3 config unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"g3 config must be a JSON object: {path}")
    missing = [key for key in _REQUIRED_FIELDS if not payload.get(key)]
    if missing:
        raise ValueError(
            f"g3 config {path} missing fields: {', '.join(missing)}"
        )
    base = Path(path).parent
    resolved: dict[str, Path] = {}
    for key in _REQUIRED_FIELDS:
        raw = Path(str(payload[key])).expanduser()
        resolved[key] = raw if raw.is_absolute() else (base / raw).resolve()
    return G3Config(
        enrollment_dir=resolved["enrollment_dir"],
        models_dir=resolved["models_dir"],
        store_dir=resolved["store_dir"],
        key_dir=resolved["key_dir"],
    )


def default_config_path() -> Path:
    """Default local config location (spec §3; never in Git)."""
    return (
        Path.home() / "Downloads" / "face_sample" / "_facecore" / "g3-local.json"
    )


def ensure_config_dir(path: Path | None = None) -> Path:
    """Create the local config dir if absent; return its path."""
    target = (path or default_config_path()).parent
    target.mkdir(parents=True, exist_ok=True)
    return target
