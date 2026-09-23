"""RED: no-factory production probe branch (#89 follow-up fix).

Defect: live_checkpoint.py promises (comment) a no-factory OpenCV
open+read+release probe, but implements only the factory branch —
production probe_shape stays None, and shape-mandatory (PR #99) always
refuses. All prior tests injected capture_factory, so the production
branch never executed (same class as #93).

Isolation WITHOUT bypassing the unit under test: capture_factory is
NOT passed (production branch walks); the LEAF OpenCVCapture class is
monkeypatched to return a known-shape frame (no hardware opened).
assert_builtin_camera itself is never stubbed.

Synthetic UIDs only (F1 governance).

RED items (pre-fix outcomes; post-fix assertions):
P1. full agreement via production probe -> camera_identity PASS.
    Pre-fix: exit 2 (probe None). If GREEN pre-fix, STOP (report).
P2. shape mismatch via production probe -> exit 2.
P3. probe unreadable (open ok, read None) -> exit 2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = str(REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_SRC = str(REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from facecore.live.contracts import FramePacket  # noqa: E402

BUILTIN_UID = "FFFF0000-0000-4000-8000-000000000002"
IPHONE_UID = "AAAA0000-0000-4000-8000-000000000001"


def _write_profile(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "nofactory-probe-v1",
        "timeout_ms": 5000,
        "sample_interval_ms": 200,
        "max_frames": 26,
        "queue_limit": 1,
        "required_support": 3,
        "min_support_interval_ms": 200,
        "match_threshold": 0.45,
        "review_threshold": 0.30,
        "margin_threshold": 0.10,
        "detector_version": "yunet-test",
        "quality_policy_version": "q-test-v1",
        "continuity_max_center_delta_ratio": 0.50,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return path


class _LeafCapture:
    """OpenCVCapture double: open/read/release, scripted frame shape."""

    def __init__(self, shape: tuple[int, int] | None) -> None:
        self._shape = shape
        self.opened_with: str | None = None
        self.released = False

    def open(self, device_id: str) -> None:
        self.opened_with = device_id

    def read(self) -> FramePacket | None:
        if self._shape is None:
            return None
        h, w = self._shape
        return FramePacket(
            sequence=1,
            captured_ns=0,
            rgb=np.ascontiguousarray(
                np.full((h, w, 3), 120, dtype=np.uint8)
            ),
        )

    def close(self) -> None:
        self.released = True


def _run_production(tmp_path: Path, tag: str, shape, **kw):
    """Run WITHOUT capture_factory: production probe branch walks."""
    from unittest.mock import patch as _patch

    from live_checkpoint import run_checkpoint
    import facecore.live.capture as _capture_mod

    store = tmp_path / f"store-{tag}"
    keys = tmp_path / f"keys-{tag}"
    leaf = _LeafCapture(shape)
    with _patch.object(_capture_mod, "OpenCVCapture", lambda: leaf):
        return run_checkpoint(
            device="1",
            profile_path=_write_profile(tmp_path),
            record_consent=True,
            image_consent=True,
            store=store,
            key_dir=keys,
            expected_builtin_unique_id=BUILTIN_UID,
            camera_identity_probe=lambda: ([IPHONE_UID, BUILTIN_UID], 2),
            # NO capture_factory: production branch must probe.
            experiment_id="exp-nofactory-probe",
            session_id=f"chk-{tag}",
            **kw,
        )


def test_p1_production_probe_full_agreement_passes(tmp_path: Path) -> None:
    """Production probe, all correct -> camera_identity PASS."""
    code, summary = _run_production(
        tmp_path, "p1", (16, 16), expected_builtin_shape="16x16"
    )
    phase = summary["phases"].get("camera_identity", {})
    assert phase.get("pass") is True, (
        f"P1: production probe must PASS on full agreement: {phase} "
        f"(exit {code} — if this fails pre-fix, the gap is reproduced)"
    )


def test_p2_production_probe_mismatch_refuses(tmp_path: Path) -> None:
    """Production probe shape mismatch -> exit 2."""
    code, _ = _run_production(
        tmp_path, "p2", (16, 16), expected_builtin_shape="720x1280"
    )
    assert code == 2, f"P2: expected exit 2 on mismatch, got {code}"


def test_p3_production_probe_unreadable_refuses(tmp_path: Path) -> None:
    """Production probe unreadable -> exit 2."""
    code, _ = _run_production(
        tmp_path, "p3", None, expected_builtin_shape="16x16"
    )
    assert code == 2, f"P3: expected exit 2 on unreadable probe, got {code}"
