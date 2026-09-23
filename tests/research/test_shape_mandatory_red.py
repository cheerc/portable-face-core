"""RED: true-device shape flag mandatory (#89 A2 landing).

Behavior contract: true-device runs REQUIRE --expected-builtin-shape
(same level as --expected-builtin-unique-id); fake path unaffected.
Shape mismatch / shape-without-probe keep exit 2 (no regression).
Full agreement => camera_identity PASS (positive control; later phases
may be stub-isolated, but assert_builtin_camera itself is never stubbed).
CLI help drops "Optional but recommended".

Synthetic UIDs only (F1 governance): AAAA... (non-builtin, sorts first)
< FFFF...002 (built-in). No real device uid anywhere.

RED items (pre-fix outcomes; post-fix assertions):
S1. true-device with uid but WITHOUT shape -> exit 2 naming the flag.
S2. shape mismatch -> exit 2.
S3. shape supplied but no probe frame -> exit 2.
S4. uid/index/shape all correct -> camera_identity PASS.
S5. fake without shape -> existing success path unchanged.
S6. existing identity RED + F1 guard stay green (suite-level, in GREEN).
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

from facecore.live.capture import FakeCapture, FramePacket  # noqa: E402

BUILTIN_UID = "FFFF0000-0000-4000-8000-000000000002"
IPHONE_UID = "AAAA0000-0000-4000-8000-000000000001"


def _write_profile(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "shape-mandatory-v1",
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


def _frames(h: int = 16, w: int = 16, n: int = 27) -> list[FramePacket]:
    return [
        FramePacket(
            sequence=seq,
            captured_ns=(seq - 1) * 200_000_000,
            rgb=np.ascontiguousarray(
                np.full((h, w, 3), 120 + (seq % 40), dtype=np.uint8)
            ),
        )
        for seq in range(1, n + 1)
    ]


class _DryCapture(FakeCapture):
    """Opens but never delivers a frame (probe-less path)."""

    def read(self) -> FramePacket | None:
        return None


def _run_shape(tmp_path: Path, tag: str, **kw):
    from live_checkpoint import run_checkpoint

    store = tmp_path / f"store-{tag}"
    keys = tmp_path / f"keys-{tag}"
    base = dict(
        device="1",
        profile_path=_write_profile(tmp_path),
        record_consent=True,
        image_consent=True,
        store=store,
        key_dir=keys,
        expected_builtin_unique_id=BUILTIN_UID,
        camera_identity_probe=lambda: (
            [IPHONE_UID, BUILTIN_UID],
            2,
        ),
        capture_factory=lambda _d: FakeCapture(frames=_frames()),
        experiment_id="exp-shape-mandatory",
        session_id=f"chk-{tag}",
    )
    base.update(kw)
    return run_checkpoint(**base)


def test_s1_missing_shape_refuses(tmp_path: Path) -> None:
    """True-device with uid but no shape must exit 2 naming the flag."""
    code, summary = _run_shape(tmp_path, "s1")
    assert code == 2, f"S1: expected exit 2 without shape, got {code}"
    text = json.dumps(summary["phases"].get("camera_identity", {}))
    assert "shape" in text.lower(), (
        f"S1: refusal must name the missing shape flag: {text[:300]}"
    )


def test_s2_shape_mismatch_refuses(tmp_path: Path) -> None:
    """Probe shape != expected shape -> exit 2."""
    code, _ = _run_shape(
        tmp_path, "s2", expected_builtin_shape="720x1280"
    )
    assert code == 2, f"S2: expected exit 2 on mismatch, got {code}"


def test_s3_shape_without_probe_refuses(tmp_path: Path) -> None:
    """Shape supplied but no probe frame readable -> exit 2."""
    code, _ = _run_shape(
        tmp_path,
        "s3",
        expected_builtin_shape="16x16",
        capture_factory=lambda _d: _DryCapture(frames=[]),
    )
    assert code == 2, f"S3: expected exit 2 without probe, got {code}"


def test_s4_full_agreement_passes(tmp_path: Path) -> None:
    """uid/index/shape all correct -> camera_identity PASS."""
    code, summary = _run_shape(
        tmp_path, "s4", expected_builtin_shape="16x16"
    )
    phase = summary["phases"].get("camera_identity", {})
    assert phase.get("pass") is True, (
        f"S4: camera_identity must PASS on full agreement: {phase}"
    )


def test_s5_fake_without_shape_unchanged(tmp_path: Path) -> None:
    """Fake without shape keeps the existing success path."""
    from live_checkpoint import run_checkpoint

    code, summary = run_checkpoint(
        device="fake",
        profile_path=_write_profile(tmp_path),
        record_consent=True,
        image_consent=True,
    )
    assert code == 0, f"S5: fake path must stay green, got {code}"
    assert "camera_identity" not in summary["phases"], (
        "S5: fake runs must skip the identity layer entirely"
    )
