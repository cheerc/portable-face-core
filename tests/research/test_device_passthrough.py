"""Fix task tests: device passthrough + live clock (blind-001 0-frame fix).

Source of truth: task t-20260914170731341828-76424-57;
governing decision: d-20260914170724830904-8.

RED contract (must fail before fix):
- device 透傳斷言：non-fake live must open the requested --device id
  (not fallback device 0).
- 時鐘零點斷言：session start must use the live monotonic clock
  (elapsed must be a real window, not ~4.8 days).
Synthetic/fake only in repo.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from facecore.live.capture import FakeCapture
from facecore.live.contracts import FramePacket


def _profile_dict(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "fix-test-v1",
        "timeout_ms": 5000,
        "sample_interval_ms": 200,
        "max_frames": 25,
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


def _packet(seq: int) -> FramePacket:
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    return FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=rgb)


class _RecordingCapture(FakeCapture):
    """Records the device_id it was opened with."""

    opened_with: str | None = None

    def open(self, device_id: str) -> None:
        type(self).opened_with = device_id
        super().open(device_id)


def test_device_id_reaches_capture_open(tmp_path: Path) -> None:
    from facecore.research import cli as cli_module

    _RecordingCapture.opened_with = None
    profile_path = _profile_dict(tmp_path)
    frames = [_packet(seq) for seq in range(1, 4)]
    probe = _RecordingCapture(frames=frames)
    rc = cli_module.cmd_live(
        profile_path=profile_path,
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        device="7",
        session_id="sess-fix-001",
        record_consent=True,
        image_consent=True,
        models=tmp_path / "models",
        corpus=tmp_path / "corpus.json",
        capture_factory=lambda _device: probe,
        detector_factory=lambda _models: None,
        embedder_factory=lambda _models: None,
    )
    # models/corpus are absent here; what matters is the device reached
    # the capture before any model gate (fail-clear still exit 2, but the
    # open call must have carried "7", never fallback "default"/0).
    assert _RecordingCapture.opened_with == "7"
    assert rc == 2


def test_session_start_uses_live_clock_not_zero(tmp_path: Path) -> None:
    from facecore.research import cli as cli_module

    before = time.monotonic_ns()
    profile_path = _profile_dict(tmp_path)
    rc = cli_module.cmd_live(
        profile_path=profile_path,
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        device="fake",
        session_id="sess-fix-002",
        record_consent=True,
        image_consent=True,
    )
    assert rc == 0
    manifest = json.loads(
        (tmp_path / "store" / "sess-fix-002" / "manifest.json").read_text()
    )
    assert manifest["status"] == "committed"
    # The committed envelope's elapsed must be a real window anchored at
    # the live clock: far below the ~4.8-day zero-origin symptom.
    from facecore.research.recorder import ResearchRecorder
    from datetime import datetime, timezone

    rec = ResearchRecorder(
        store_root=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        clock=lambda: datetime.now(timezone.utc),
    )
    record = rec.read_record("sess-fix-002")
    assert record.result.elapsed_ms < 60_000.0
    assert before > 0


def test_open_probes_first_frame_fail_clear(tmp_path: Path) -> None:
    from facecore.research import cli as cli_module

    class _DryCapture(FakeCapture):
        def __init__(self) -> None:
            super().__init__(frames=[])

    profile_path = _profile_dict(tmp_path)
    rc = cli_module.cmd_live(
        profile_path=profile_path,
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        device="9",
        session_id="sess-fix-003",
        record_consent=True,
        image_consent=True,
        models=tmp_path / "models",
        corpus=tmp_path / "corpus.json",
        capture_factory=lambda _device: _DryCapture(),
        detector_factory=lambda _models: None,
        embedder_factory=lambda _models: None,
    )
    # Dry source fail-clear: no silent 0-frame commit.
    assert rc == 2
    assert not (tmp_path / "store" / "sess-fix-003").exists()
