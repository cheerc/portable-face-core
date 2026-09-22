"""RED: runner trace predicate must assert true invariants, not ceil+1 count.

Defect: live_checkpoint asserts trace_entries == ceil(timeout/interval)+1
(5000/200 -> 26). A correct real-hardware run with 22 sampled+traced frames
(deadline_reached, complete, dual-full) is judged FAIL/exit 4.

Frozen contract: trace_ok must be
  trace_entries == frames_sampled (no drop) AND frames_sampled <= max_frames,
with frames_sampled from cmd_live existing output paths (no new seam).

RED items (run pre-fix, all must show the stated pre-fix outcome):
1. trace=sampled=22, max=26 -> pre-False (defect), post-True.
2. drop (trace < sampled) -> False pre AND post (predicate not deleted).
3. cap-bound (sampled > max) -> False.
4. staged_errors nonempty -> False (4th guardrail untouched).
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = str(REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from facecore.live.contracts import (  # noqa: E402
    FramePacket,
    SessionResult,
    SessionStatus,
)
from facecore.research.diagnostics import FrameDiagnostics, FrameTraceEntry  # noqa: E402
from facecore.research.experiment import AttemptRecord, ExperimentManifest  # noqa: E402
from facecore.research.records import CollectionWindow, ConsentRecord  # noqa: E402
from facecore.research.recorder import ResearchRecorder  # noqa: E402


def _now() -> datetime:
    return datetime(2026, 9, 22, 3, 0, 0, tzinfo=timezone.utc)


def _write_profile(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "trace-pred-red-v1",
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
    p = tmp_path / "profile.json"
    p.write_text(json.dumps(profile))
    return p


def _trace_entry(seq: int) -> FrameTraceEntry:
    return FrameTraceEntry(
        sequence=seq,
        captured_ns=(seq - 1) * 200_000_000,
        processed_ns=(seq - 1) * 200_000_000 + 1_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(2.0, 2.0, 4.0, 4.0),
        identity_score_pairs=(("id-01", 0.8),),
        quality_rank=0.9,
        model_generation="gen-1",
        gallery_digest="gal-1",
        diagnostics=FrameDiagnostics(
            sequence=seq,
            original_shape=(16, 16, 3),
            normalized_shape=(16, 16, 3),
            orientation=0,
            mirrored=False,
            face_count=1,
            detector_confidence=0.9,
            face_box=(2.0, 2.0, 4.0, 4.0),
            landmarks=None,
            sharpness=0.9,
        ),
    )


def _frame(seq: int) -> FramePacket:
    return FramePacket(
        sequence=seq,
        captured_ns=(seq - 1) * 200_000_000,
        rgb=np.full((16, 16, 3), 100, dtype=np.uint8),
    )


def _mock_live(n_trace: int, n_sampled: int, staged_err: str | None = None):
    """Build a mock cmd_live writing a real bundle: attempt + trace + commit."""

    def _run(**kwargs: object) -> int:
        from pathlib import Path as _P

        store = _P(str(kwargs["store"]))
        key_dir = _P(str(kwargs["key_dir"]))
        session_id = str(kwargs["session_id"])
        attempt_id = str(kwargs.get("attempt_id") or f"att-{session_id}")
        rec = ResearchRecorder(store_root=store, key_dir=key_dir, clock=_now)
        consent = ConsentRecord(
            session_id=session_id,
            participant_id="part-01",
            record_consent=True,
            image_consent=True,
            consented_at_utc=_now().isoformat(),
            record_expires_at_utc=(_now() + timedelta(days=30)).isoformat(),
            image_expires_at_utc=(_now() + timedelta(days=7)).isoformat(),
        )
        rec.begin(session_id, consent)
        for seq in range(1, n_sampled + 1):
            rec.append_frame(_frame(seq))
        manifest = ExperimentManifest.from_dict({
            "identity": {"experiment_id": "exp-trace-pred"},
            "software": {}, "gallery": {}, "policy": {}, "capture": {},
            "privacy": {}, "study": {}, "analysis": {},
        })
        attempt = AttemptRecord(
            experiment_id="exp-trace-pred", attempt_id=attempt_id,
            participant_id="part-01", visit_id="visit-001",
            condition_id="cond-001", attempt_index=1, retry_of=None,
            consent_ref=session_id,
            requested_at_utc=_now().isoformat(),
            accepted_at_utc=_now().isoformat(),
            started_at_utc=_now().isoformat(), ended_at_utc=None,
            operational_status="accepted", error_code=None,
            bundle_ref=session_id,
        )
        rec.begin_attempt(manifest, attempt, consent)
        for seq in range(1, n_trace + 1):
            rec.append_trace(attempt_id, _trace_entry(seq))
        terminal = SessionResult(
            session_id=session_id, schema_version="v1",
            status=SessionStatus.timeout, matched_identity=None,
            reason_codes=("deadline_exceeded",), elapsed_ms=5020.0,
            frames_sampled=n_sampled, frames_usable=n_sampled,
            frames_rejected=0, frames_dropped=0, support_sequences=(),
            profile_digest="trace-pred", model_generation="gen-1",
            gallery_digest="gal-1",
        )
        cw = CollectionWindow(
            session_id=session_id, collection_start_ns=0,
            collection_deadline_ns=5_000_000_000, collection_end_ns=None,
            collection_stop_reason="deadline_reached",
            collection_complete=True, frames_sampled=n_sampled,
        )
        rec.commit(terminal, collection_window=cw)
        print(json.dumps({
            "session_id": session_id, "status": "timeout",
            "window": "fixed-window-complete", "elapsed_ms": 5020.0,
            "reason_codes": ["deadline_exceeded"],
            "generation": "gen-1", "gallery_digest": "gal-1",
        }))
        if staged_err is not None:
            sys.stderr.write(f"research live: staging failed: {staged_err}\n")
        return 0

    return _run


def _run_checkpoint(tmp_path: Path, mock_fn: object) -> tuple[int, dict]:
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    with patch("live_checkpoint.cmd_live", side_effect=mock_fn):
        return run_checkpoint(
            device="fake",
            profile_path=profile_path,
            record_consent=True,
            image_consent=True,
        )


def test_red1_real_hardware_shape_22_22_26(tmp_path: Path) -> None:
    """take-2 real shape: trace=sampled=22, max=26 -> pre-False, post-True."""
    code, summary = _run_checkpoint(tmp_path, _mock_live(22, 22))
    live = summary["phases"]["live"]
    # GREEN: true invariants hold (22==22, 22<=26) -> pass True, exit 0.
    # (RED commit bc49638 recorded pre-fix code=4 pass=False here.)
    assert code == 0
    assert live.get("pass") is True
    assert live.get("trace_entries") == 22
    assert live.get("frames_sampled") == 22
    json.dumps(summary)


def test_red2_drop_stays_false(tmp_path: Path) -> None:
    """trace_entries < frames_sampled -> False pre AND post."""
    code, summary = _run_checkpoint(tmp_path, _mock_live(20, 22))
    live = summary["phases"]["live"]
    assert code == 4
    assert live.get("pass") is False
    json.dumps(summary)


def test_red3_cap_bound_stays_false(tmp_path: Path) -> None:
    """frames_sampled > profile.max_frames -> False."""
    code, summary = _run_checkpoint(tmp_path, _mock_live(27, 27))
    live = summary["phases"]["live"]
    assert code == 4
    assert live.get("pass") is False
    json.dumps(summary)


def test_red4_staged_errors_stays_false(tmp_path: Path) -> None:
    """staged_errors nonempty -> False (4th guardrail untouched)."""
    code, summary = _run_checkpoint(
        tmp_path, _mock_live(22, 22, staged_err="1:ValueError"))
    live = summary["phases"]["live"]
    assert code == 4
    assert live.get("pass") is False
    assert "staged_errors" in live
    json.dumps(summary)


def test_red_guard_ceil_is_cap_not_count() -> None:
    """Concept anchor: ceil(5000/200)+1 == 26 is the cap bound, not a count."""
    assert math.ceil(5000 / 200) + 1 == 26
