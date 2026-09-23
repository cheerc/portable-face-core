"""RED: runner paired replay phase verifies dual-arm collection_extent (#82).

Defect: scripts/live_checkpoint.py has no replay phase (grep
arm|extent|paired|replay is zero-hit); dual-arm `full` evidence cannot be
produced by one operator command.

Frozen contract (Lead ruling D, runner-only):
- New summary["phases"]["paired_replay"] after live, before camera.
  Composed of existing product APIs only: read_record (committed window)
  -> read_trace (trace) -> evaluate_arms(trace, profile, window=window).
  No recomputed extent logic, no new seam.
- True-device path: any missing step / raise, any arm non-full, any
  refusal -> paired FAIL with explicit error (never silent skip).
  Both arms full -> PASS.
- Fake path: envelope-only by design (cmd_live never persists a trace
  there) -> paired records {"pass": True, "skipped": True, ...} and does
  not gate the total PASS.
- Total PASS = existing four live checks AND paired (true path);
  paired never masks another failure.

RED items (pre-fix outcome: no "paired_replay" key in summary):
1. synthetic full window + synthetic full trace -> PASS (positive
   control, CI-provable non-vacuous).
2. stubbed evaluate_arms A=full / B=incomplete -> FAIL (phase checks
   BOTH arms, not one).
3. both arms incomplete -> FAIL.
4. true-device path with missing trace -> FAIL closed (no silent skip).
5. fake happy path -> total PASS with paired explicitly skipped.
6. live + paired full but a later phase fails -> total still FAIL
   (paired never masks).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = str(REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_SRC = str(REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from facecore.live.capture import FakeCapture, FramePacket  # noqa: E402
from facecore.live.contracts import SessionResult, SessionStatus  # noqa: E402
from facecore.research.diagnostics import FrameDiagnostics, FrameTraceEntry  # noqa: E402
from facecore.research.experiment import AttemptRecord, ExperimentManifest  # noqa: E402
from facecore.research.records import CollectionWindow, ConsentRecord  # noqa: E402
from facecore.research.recorder import ResearchRecorder  # noqa: E402
from facecore.research.replay import ArmOutcome  # noqa: E402

BUILTIN_UID = "FFFF0000-0000-4000-8000-000000000002"
IPHONE_UID = "AAAA0000-0000-4000-8000-000000000001"


def _now() -> datetime:
    return datetime(2026, 9, 22, 3, 0, 0, tzinfo=timezone.utc)


def _write_profile(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "paired-red-v1",
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


def _frames16(n: int = 27) -> list[FramePacket]:
    return [
        FramePacket(
            sequence=seq,
            captured_ns=(seq - 1) * 200_000_000,
            rgb=np.ascontiguousarray(
                np.full((16, 16, 3), 120 + (seq % 40), dtype=np.uint8)
            ),
        )
        for seq in range(1, n + 1)
    ]


def _trace_entry(seq: int) -> FrameTraceEntry:
    return FrameTraceEntry(
        sequence=seq,
        captured_ns=(seq - 1) * 200_000_000,
        processed_ns=(seq - 1) * 200_000_000 + 1_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(4.0, 4.0, 8.0, 8.0),
        identity_score_pairs=(("person-01", 0.85), ("person-02", 0.20)),
        quality_rank=50.0,
        model_generation="gen-1",
        gallery_digest="gal-1",
        diagnostics=FrameDiagnostics(
            sequence=seq,
            original_shape=(16, 16, 3),
            normalized_shape=(16, 16, 3),
            orientation=0,
            mirrored=False,
            face_count=1,
            detector_confidence=0.99,
            face_box=(4.0, 4.0, 8.0, 8.0),
            landmarks=None,
            quality_status="accepted",
        ),
        staged_index=seq - 1,
    )


def _mock_live(n_trace: int, n_sampled: int, *, complete: bool) -> object:
    """Mock cmd_live writing a real bundle: attempt + trace + commit.

    complete=True  -> deadline_reached window, "fixed-window-complete".
    complete=False -> source_exhausted window, "fixed-window-incomplete".
    n_trace=0 writes the window but no trace (missing-trace path).
    """

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
        manifest = ExperimentManifest.from_dict({
            "identity": {"experiment_id": "exp-paired-red"},
            "software": {}, "gallery": {}, "policy": {}, "capture": {},
            "privacy": {}, "study": {}, "analysis": {},
        })
        attempt = AttemptRecord(
            experiment_id="exp-paired-red", attempt_id=attempt_id,
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
            reason_codes=("deadline_exceeded",) if complete else ("source_exhausted",),
            elapsed_ms=5020.0 if complete else 2010.0,
            frames_sampled=n_sampled, frames_usable=n_sampled,
            frames_rejected=0, frames_dropped=0, support_sequences=(),
            profile_digest="paired-red", model_generation="gen-1",
            gallery_digest="gal-1",
        )
        cw = CollectionWindow(
            session_id=session_id, collection_start_ns=0,
            collection_deadline_ns=5_000_000_000, collection_end_ns=None,
            collection_stop_reason=(
                "deadline_reached" if complete else "source_exhausted"
            ),
            collection_complete=complete, frames_sampled=n_sampled,
        )
        rec.commit(terminal, collection_window=cw)
        print(json.dumps({
            "session_id": session_id, "status": "timeout",
            "window": (
                "fixed-window-complete" if complete
                else "fixed-window-incomplete"
            ),
            "elapsed_ms": 5020.0 if complete else 2010.0,
            "reason_codes": (
                ["deadline_exceeded"] if complete else ["source_exhausted"]
            ),
            "generation": "gen-1", "gallery_digest": "gal-1",
        }))
        return 0

    return _run


def _probe(uids: list[str], openable: int | None = None):
    ids = sorted(uids)
    return lambda: (ids, openable if openable is not None else len(ids))


def _run_true(tmp_path: Path, tag: str, mock_fn: object, **kw):
    from live_checkpoint import run_checkpoint

    base = dict(
        device="1",
        profile_path=_write_profile(tmp_path),
        record_consent=True,
        image_consent=True,
        capture_factory=lambda _d: FakeCapture(frames=_frames16()),
        expected_builtin_unique_id=BUILTIN_UID,
        expected_builtin_shape="16x16",
        camera_identity_probe=_probe([IPHONE_UID, BUILTIN_UID]),
        # Test env has no cv2: stub the reopen probe (production path
        # untouched). RED6 overrides this with its own failing stub.
        reopen_fn=lambda _dev: {"pass": True, "reopen": True},
        experiment_id="exp-paired-red",
        session_id=f"chk-{tag}",
    )
    base.update(kw)
    with patch("live_checkpoint.cmd_live", side_effect=mock_fn):
        return run_checkpoint(**base)


def _paired(summary: dict) -> dict:
    paired = summary["phases"].get("paired_replay")
    assert paired is not None, (
        "RED: runner must emit phases['paired_replay'] "
        f"(phases present: {sorted(summary['phases'])})"
    )
    return paired


def test_red1_synthetic_full_dual_passes(tmp_path: Path) -> None:
    """Positive control: full window + full trace -> paired PASS, total PASS."""
    code, summary = _run_true(tmp_path, "red1", _mock_live(26, 26, complete=True))
    paired = _paired(summary)
    assert paired.get("pass") is True, f"RED1: paired must pass: {paired}"
    assert paired.get("arm_a_extent") == "full", paired
    assert paired.get("arm_b_extent") == "full", paired
    assert code == 0, f"RED1: total must pass, got {code}: {summary}"
    assert summary["verdict"] == "PASS"
    json.dumps(summary)


def test_red2_mixed_arm_extents_fail(tmp_path: Path) -> None:
    """Phase must check BOTH arms: A=full / B=incomplete -> FAIL."""
    arm_a = ArmOutcome(
        attempt_id="att-chk-red2", run_id="run-red2", arm_id="A",
        profile_digest="d", selected_sequences=(), support_sequences=(),
        terminal="timeout", matched_identity=None,
        collection_extent="full", decision_time_ns=None,
        decision_codes=("stub",), frames_read=26, frames_scored=26,
        frames_consumed=26, frames_staged=26, refusal=None,
    )
    arm_b = ArmOutcome(
        attempt_id="att-chk-red2", run_id="run-red2", arm_id="B",
        profile_digest="d", selected_sequences=(), support_sequences=(),
        terminal="timeout", matched_identity=None,
        collection_extent="incomplete", decision_time_ns=None,
        decision_codes=("stub",), frames_read=26, frames_scored=26,
        frames_consumed=26, frames_staged=26, refusal=None,
    )
    with patch(
        "live_checkpoint.evaluate_arms", create=True, return_value=(arm_a, arm_b)
    ):
        code, summary = _run_true(
            tmp_path, "red2", _mock_live(26, 26, complete=True)
        )
    paired = _paired(summary)
    assert paired.get("pass") is False, (
        f"RED2: mixed A=full/B=incomplete must fail: {paired}"
    )
    assert code == 4, f"RED2: total must fail, got {code}"
    assert summary["verdict"] == "FAIL"
    json.dumps(summary)


def test_red3_both_incomplete_fail(tmp_path: Path) -> None:
    """Exhausted window -> both arms incomplete -> FAIL."""
    code, summary = _run_true(
        tmp_path, "red3", _mock_live(10, 10, complete=False)
    )
    paired = _paired(summary)
    assert paired.get("pass") is False, f"RED3: incomplete must fail: {paired}"
    assert code == 4, f"RED3: total must fail, got {code}"
    assert summary["verdict"] == "FAIL"
    json.dumps(summary)


def test_red4_missing_trace_fails_closed(tmp_path: Path) -> None:
    """True path without a trace must FAIL explicitly, never skip."""
    code, summary = _run_true(tmp_path, "red4", _mock_live(0, 26, complete=True))
    paired = _paired(summary)
    assert paired.get("pass") is False, (
        f"RED4: missing trace must fail closed: {paired}"
    )
    assert paired.get("skipped") is not True, (
        f"RED4: true path must never skip: {paired}"
    )
    assert "error" in paired, f"RED4: failure must carry an error: {paired}"
    assert code == 4, f"RED4: total must fail, got {code}"
    assert summary["verdict"] == "FAIL"
    json.dumps(summary)


def test_red5_fake_skipped_keeps_pass(tmp_path: Path) -> None:
    """Fake path stays envelope-only: paired skipped, total PASS."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    code, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
    )
    paired = _paired(summary)
    assert paired.get("skipped") is True, f"RED5: fake must skip: {paired}"
    assert paired.get("pass") is True, f"RED5: skip must not gate: {paired}"
    assert code == 0, f"RED5: fake total must pass, got {code}: {summary}"
    assert summary["verdict"] == "PASS"
    json.dumps(summary)


def test_red6_paired_full_does_not_mask_later_failure(tmp_path: Path) -> None:
    """Paired PASS must not mask a later phase failure (reopen fails)."""
    code, summary = _run_true(
        tmp_path,
        "red6",
        _mock_live(26, 26, complete=True),
        reopen_fn=lambda dev: {
            "pass": False, "reopen": False, "error": "simulated reopen failure",
        },
    )
    paired = _paired(summary)
    assert paired.get("pass") is True, (
        f"RED6: paired itself must pass here: {paired}"
    )
    assert code == 4, f"RED6: total must still fail, got {code}"
    assert summary["verdict"] == "FAIL"
    assert summary["phases"]["camera"]["pass"] is False
    json.dumps(summary)
