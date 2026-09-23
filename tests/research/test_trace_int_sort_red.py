"""RED: trace read-back sorts by integer sequence (#89 follow-up fix).

Defect: recorder.py writes trace files as frame_{seq:04d}.enc but
read_trace iterates sorted(glob) — lexicographic — so any sequence >=
10000 misorders (frame_34458 before frame_3765); replay's strict
monotonicity then raises duplicate_sequence. The parsed seq_num exists
but is unused for ordering. Genuine defect (60fps / long sessions /
unthrottled sources trigger it), caught by the camera-free rehearsal.

RED: append two entries with sequences crossing 9999 (3765 then 34458
in write order... actually written in TRUE order 3765, 34458), read
back via read_trace, feed to _observations_from_entries (the replay
gate). Pre-fix: duplicate_sequence ValueError (read returns 34458
first). If GREEN pre-fix, STOP (report).

Camera-free, synthetic payloads only.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from facecore.research.diagnostics import FrameDiagnostics, FrameTraceEntry
from facecore.research.experiment import AttemptRecord, ExperimentManifest
from facecore.research.records import ConsentRecord
from facecore.research.recorder import ResearchRecorder


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _recorder(tmp_path: Path) -> ResearchRecorder:
    now = _utc("2026-09-23T10:00:00Z")
    return ResearchRecorder(
        store_root=tmp_path / "store",
        key_dir=tmp_path / "keys",
        clock=lambda: now,
    )


def _consent() -> ConsentRecord:
    return ConsentRecord(
        session_id="sess-sort-001",
        participant_id="part-synth-001",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-23T10:00:00Z",
        record_expires_at_utc="2026-10-23T10:00:00Z",
        image_expires_at_utc="2026-09-30T10:00:00Z",
    )


def _manifest() -> ExperimentManifest:
    return ExperimentManifest.from_dict(
        {
            "identity": {"experiment_id": "exp-sort-001"},
            "software": {},
            "gallery": {},
            "policy": {},
            "capture": {},
            "privacy": {},
            "study": {},
            "analysis": {},
        }
    )


def _attempt() -> AttemptRecord:
    return AttemptRecord(
        experiment_id="exp-sort-001",
        attempt_id="att-sort-001",
        participant_id="part-synth-001",
        visit_id="visit-001",
        condition_id="cond-001",
        attempt_index=1,
        retry_of=None,
        consent_ref="consent-synth",
        requested_at_utc="2026-09-23T10:00:00Z",
        accepted_at_utc="2026-09-23T10:00:01Z",
        started_at_utc=None,
        ended_at_utc=None,
        operational_status="accepted",
        error_code=None,
        bundle_ref=None,
    )


def _diag(seq: int) -> FrameDiagnostics:
    return FrameDiagnostics(
        sequence=seq,
        original_shape=(480, 640, 3),
        normalized_shape=(480, 640, 3),
        orientation=0,
        mirrored=False,
        face_count=0,
        detector_confidence=None,
        face_box=None,
        landmarks=None,
        landmark_confidence_is_constant=True,
        shorter_side_px=None,
        sharpness=None,
        mean_luma=None,
        clipped_fraction=None,
        yaw_deg=None,
        pitch_deg=None,
        quality_status="rejected",
        quality_reason_codes=(),
        detection_missing_reason="no_face_detected",
        quality_missing_reason="quality_rejected",
        scoring_missing_reason="no_scores",
    )


def _entry(seq: int) -> FrameTraceEntry:
    return FrameTraceEntry(
        sequence=seq,
        captured_ns=seq * 1_000_000,
        processed_ns=seq * 1_000_000 + 10_000,
        quality_pass=False,
        quality_reasons=(),
        face_count=0,
        face_box=None,
        identity_score_pairs=(),
        quality_rank=0.0,
        model_generation="gen-1",
        gallery_digest="gal-1",
        diagnostics=_diag(seq),
        decision_event=None,
        staged_index=None,
        stage_missing_reason=None,
    )


def test_cross_9999_trace_reads_in_sequence_order(tmp_path: Path) -> None:
    """Entries 3765 then 34458 (write order) must read back ordered."""
    rec = _recorder(tmp_path)
    rec.begin_attempt(_manifest(), _attempt(), _consent())
    rec.append_trace("att-sort-001", _entry(3765))
    rec.append_trace("att-sort-001", _entry(34458))
    trace = rec.read_trace("att-sort-001")
    got = [e.sequence for e in trace.entries]
    assert got == [3765, 34458], (
        f"SORT-RED: read_trace returned {got} (expected [3765, 34458] — "
        "lexicographic filename sort misorders 5-digit sequences)"
    )
    from facecore.research.replay import _observations_from_entries

    obs = _observations_from_entries(trace.entries)  # must not raise
    assert [o.sequence for o in obs] == [3765, 34458]
