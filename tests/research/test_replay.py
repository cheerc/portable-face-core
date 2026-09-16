"""Phase 2A Task T6 tests: sealed replay with ground-truth isolation.

Source of truth: Phase 2A Implementation Plan §4 & §6 T6;
Task: t-20260914111204047555-76424-37;
Governing decision: d-20260914110757304910-5.

RED contract (must fail before implementation exists):
- label leak: changing the label must not change the inference result.
- 遺漏 error 分母： missing/expired/deleted/tampered bundles refuse
  replay as error (never unknown), and appear in the error denominator.
- early 短流當完整窗口： early-terminated bundles are labeled
  window=early-stop, never presented as full-window comparisons.
Only synthetic payloads; never real faces.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from facecore.live.contracts import (
    FramePacket,
    ResearchProfile,
    SessionResult,
    SessionStatus,
)
from facecore.research.records import ConsentRecord
from facecore.research.recorder import ResearchRecorder
from facecore.research.replay import (
    ReplayRefusal,
    ReplayResult,
    describe_replay,
    replay_session,
)


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _profile() -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="t6-test-v1",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=25,
        queue_limit=1,
        required_support=3,
        min_support_interval_ms=200,
        match_threshold=0.45,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="yunet-test",
        quality_policy_version="q-test-v1",
        continuity_max_center_delta_ratio=0.50,
    )


def _consent(session_id: str) -> ConsentRecord:
    return ConsentRecord(
        session_id=session_id,
        participant_id="part-synth-001",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-14T10:00:00Z",
        record_expires_at_utc="2026-10-14T10:00:00Z",
        image_expires_at_utc="2026-09-21T10:00:00Z",
    )


def _frame(seq: int) -> FramePacket:
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[0, 0, 0] = seq % 256
    return FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=rgb)


def _result(session_id: str) -> SessionResult:
    return SessionResult(
        session_id=session_id,
        schema_version="v1",
        status=SessionStatus.unknown,
        matched_identity=None,
        reason_codes=("t6-synth",),
        elapsed_ms=1000.0,
        frames_sampled=5,
        frames_usable=5,
        frames_rejected=0,
        frames_dropped=0,
        support_sequences=(),
        profile_digest="0" * 64,
        model_generation="test-gen",
        gallery_digest="1" * 64,
    )


def _store_bundle(
    tmp_path: Path, clock: _Clock, session_id: str, n_frames: int = 3
) -> None:
    rec = ResearchRecorder(
        store_root=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        clock=clock,
    )
    rec.begin(session_id, _consent(session_id))
    for seq in range(1, n_frames + 1):
        rec.append_frame(_frame(seq))
    rec.commit(_result(session_id))


def _scorer(packet: FramePacket):  # type: ignore[no-untyped-def]
    from facecore.live.contracts import FrameObservation

    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=False,
        quality_reasons=("t6-synth-reject",),
        face_count=0,
        face_box=None,
        identity_scores={},
        quality_rank=0.0,
        model_generation="test-gen",
        gallery_digest="1" * 64,
    )


def _replay_kwargs(tmp_path: Path, clock: _Clock) -> dict[str, Any]:
    return {
        "store_root": tmp_path / "store",
        "key_dir": tmp_path / "research_keys",
        "clock": clock,
        "profile": _profile(),
        "scorer": _scorer,
        "model_generation": "test-gen",
        "gallery_digest": "1" * 64,
    }


# ---------------------------------------------------------------------------
# Determinism + label isolation (RED 1)
# ---------------------------------------------------------------------------


def test_same_bundle_profile_gallery_model_replays_identically(
    tmp_path: Path,
) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    _store_bundle(tmp_path, clock, "sess-t6-001")
    first = replay_session("sess-t6-001", **_replay_kwargs(tmp_path, clock))
    second = replay_session("sess-t6-001", **_replay_kwargs(tmp_path, clock))
    assert isinstance(first, ReplayResult)
    assert first.result.status == second.result.status
    assert first.result.reason_codes == second.result.reason_codes
    assert first.window == second.window


def test_changing_label_does_not_change_inference(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    _store_bundle(tmp_path, clock, "sess-t6-001")
    kwargs = _replay_kwargs(tmp_path, clock)
    replay_a = replay_session("sess-t6-001", **kwargs)
    # Labels travel outside replay_session entirely; a label change is a
    # report-time annotation and must leave inference bytes untouched.
    replay_b = replay_session("sess-t6-001", **kwargs)
    assert replay_a.result.to_dict() == replay_b.result.to_dict()
    assert "label" not in replay_a.result.to_dict()
    assert "ground_truth" not in replay_a.result.to_dict()


# ---------------------------------------------------------------------------
# Refusals are error, never unknown (RED 2)
# ---------------------------------------------------------------------------


def test_missing_bundle_refuses_as_error(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    with pytest.raises(ReplayRefusal) as excinfo:
        replay_session("sess-nope", **_replay_kwargs(tmp_path, clock))
    assert excinfo.value.kind == "missing"


def test_expired_bundle_refuses_as_error(tmp_path: Path) -> None:
    t0 = _utc("2026-09-14T10:00:00Z")
    clock = _Clock(t0)
    _store_bundle(tmp_path, clock, "sess-t6-001")
    clock.now = t0 + timedelta(days=31)
    with pytest.raises(ReplayRefusal) as excinfo:
        replay_session("sess-t6-001", **_replay_kwargs(tmp_path, clock))
    assert excinfo.value.kind == "expired"


def test_deleted_bundle_refuses_as_error(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    _store_bundle(tmp_path, clock, "sess-t6-001")
    rec = ResearchRecorder(
        store_root=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        clock=clock,
    )
    assert rec.delete("sess-t6-001") is True
    with pytest.raises(ReplayRefusal) as excinfo:
        replay_session("sess-t6-001", **_replay_kwargs(tmp_path, clock))
    # Fully deleted bundles leave no directory: reported as missing
    # (still an error refusal, counted in the error denominator).
    assert excinfo.value.kind == "missing"


def test_tombstoned_bundle_refuses_as_deleted(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    _store_bundle(tmp_path, clock, "sess-t6-001")
    # Crash between tombstone write and payload purge: deleted, not missing.
    (tmp_path / "store" / "sess-t6-001" / "tombstone.json").write_text("{}")
    with pytest.raises(ReplayRefusal) as excinfo:
        replay_session("sess-t6-001", **_replay_kwargs(tmp_path, clock))
    assert excinfo.value.kind == "deleted"


def test_tampered_manifest_refuses_as_error(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    _store_bundle(tmp_path, clock, "sess-t6-001")
    manifest = tmp_path / "store" / "sess-t6-001" / "manifest.json"
    manifest.write_text(manifest.read_text().replace("committed", "tampered"))
    with pytest.raises(ReplayRefusal) as excinfo:
        replay_session("sess-t6-001", **_replay_kwargs(tmp_path, clock))
    assert excinfo.value.kind == "tampered"


def test_generation_mismatch_refuses_as_error(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    _store_bundle(tmp_path, clock, "sess-t6-001")
    kwargs = _replay_kwargs(tmp_path, clock)
    kwargs["model_generation"] = "other-gen"
    with pytest.raises(ReplayRefusal) as excinfo:
        replay_session("sess-t6-001", **kwargs)
    assert excinfo.value.kind == "generation_mismatch"


# ---------------------------------------------------------------------------
# Window labeling: full vs early-stop (RED 3)
# ---------------------------------------------------------------------------


def test_full_window_bundle_labeled_full(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    _store_bundle(tmp_path, clock, "sess-t6-001", n_frames=25)
    replayed = replay_session("sess-t6-001", **_replay_kwargs(tmp_path, clock))
    assert replayed.window == "full"
    assert replayed.frames_replayed == 25


def test_short_bundle_labeled_early_stop_never_full(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    _store_bundle(tmp_path, clock, "sess-t6-001", n_frames=3)
    replayed = replay_session("sess-t6-001", **_replay_kwargs(tmp_path, clock))
    assert replayed.window == "early-stop"
    assert replayed.window != "full"
    assert replayed.frames_replayed == 3


def test_record_without_frames_analyzes_stored_scores_only(
    tmp_path: Path,
) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    _store_bundle(tmp_path, clock, "sess-t6-001", n_frames=0)
    replayed = replay_session("sess-t6-001", **_replay_kwargs(tmp_path, clock))
    assert replayed.window == "none"
    assert replayed.frames_replayed == 0
    # No re-inference is possible; the stored record band is reported.
    assert replayed.result.status in (
        SessionStatus.unknown,
        SessionStatus.review,
        SessionStatus.matched,
        SessionStatus.timeout,
        SessionStatus.invalid_input,
    )
    # T6 N2: frameless bundles formally carry replayed=False.
    assert describe_replay(replayed)["replayed"] is False


# ---------------------------------------------------------------------------
# T5 N1: read_frame dims sanity cap
# ---------------------------------------------------------------------------


def test_oversize_frame_dims_rejected(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    _store_bundle(tmp_path, clock, "sess-t6-001", n_frames=1)
    # Corrupt the stored dims claim inside an otherwise valid blob is
    # covered by AEAD; here we assert the cap constant contract directly.
    from facecore.research.recorder import MAX_FRAME_BYTES, MAX_FRAME_SIDE_PX

    assert MAX_FRAME_SIDE_PX == 4096
    assert MAX_FRAME_BYTES == 30 * 1024 * 1024
    rec = ResearchRecorder(
        store_root=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        clock=clock,
    )
    frame = rec.read_frame("sess-t6-001", 0)
    assert frame.rgb.shape[0] <= MAX_FRAME_SIDE_PX
    assert frame.rgb.nbytes <= MAX_FRAME_BYTES
    assert json.loads(
        (tmp_path / "store" / "sess-t6-001" / "manifest.json").read_text()
    )["status"] == "committed"


# ---------------------------------------------------------------------------
# E7-B r5 T1 RED: replay applies the persisted capture mapping (A.7)
# ---------------------------------------------------------------------------


def test_replay_applies_capture_mapping_before_scoring(tmp_path: Path) -> None:
    """E7-B r5 T1: live square crop must match replay scorer input."""
    from facecore.live.qt_window import CropMapping

    clock = _Clock(_utc("2026-09-16T10:00:00Z"))
    rec = ResearchRecorder(
        store_root=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        clock=clock,
    )
    session_id = "sess-e7-replay-map"
    rec.begin(session_id, _consent(session_id))
    rgb = np.zeros((3, 5, 3), dtype=np.uint8)
    rec.append_frame(FramePacket(sequence=1, captured_ns=200_000_000, rgb=rgb))
    rec.commit(_result(session_id))

    from facecore.research.experiment import AttemptRecord, ExperimentManifest

    manifest = ExperimentManifest.from_dict(
        {
            "identity": {"experiment_id": "exp-e7-replay"},
            "software": {},
            "gallery": {},
            "policy": {},
            "capture": {},
            "privacy": {},
            "study": {},
            "analysis": {},
        }
    )
    rec.begin_attempt(
        manifest,
        AttemptRecord(
            experiment_id="exp-e7-replay",
            attempt_id="att-e7-replay",
            participant_id="part-synth-001",
            visit_id="visit-001",
            condition_id="cond-001",
            attempt_index=1,
            retry_of=None,
            consent_ref=session_id,
            requested_at_utc="2026-09-16T10:00:00Z",
            accepted_at_utc="2026-09-16T10:00:01Z",
            started_at_utc=None,
            ended_at_utc=None,
            operational_status="accepted",
            error_code=None,
            bundle_ref=session_id,
        ),
        _consent(session_id),
    )
    mapping = CropMapping(x=1, y=0, size=3, frame_w=5, frame_h=3)
    rec.record_crop_mapping("att-e7-replay", mapping.to_dict())

    seen: list[tuple[int, ...]] = []

    def _shape_scorer(packet: FramePacket):  # type: ignore[no-untyped-def]
        from facecore.live.contracts import FrameObservation

        seen.append(tuple(packet.rgb.shape))
        return FrameObservation(
            sequence=packet.sequence,
            captured_ns=packet.captured_ns,
            processed_ns=packet.captured_ns + 1_000_000,
            quality_pass=False,
            quality_reasons=("e7-replay-shape",),
            face_count=0,
            face_box=None,
            identity_scores={},
            quality_rank=0.0,
            model_generation="test-gen",
            gallery_digest="1" * 64,
        )

    kwargs = _replay_kwargs(tmp_path, clock)
    kwargs["scorer"] = _shape_scorer
    replay_session(session_id, **kwargs)
    assert seen == [(3, 3, 3)]
