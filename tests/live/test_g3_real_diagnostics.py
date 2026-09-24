"""G3 W5 RED: real per-frame diagnostics reach the encrypted trace.

Source of truth: docs/specs/2026-09-24-g3-local-test-app.md §3 (true
per-frame detection and quality values — confidence, face box,
landmarks, sharpness, luma, angles, rejection reasons — must land in
the trace, never placeholders) and §7 item 6 (diagnostic fields are
true values, proven not None placeholders by test).

All tests use synthetic frames, FakeCapture, and a tmp
ResearchRecorder. No camera, real faces, gallery, embeddings, or
photos.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from facecore.live.capture import FakeCapture
from facecore.live.contracts import (
    FrameDiagnostics,
    FrameObservation,
    FramePacket,
    ResearchProfile,
)
from facecore.live.desktop import DesktopSession
from facecore.live.session import SessionEngine
from facecore.research.diagnostics import FrameTraceEntry
from facecore.research.experiment import AttemptRecord, ExperimentManifest
from facecore.research.records import ConsentRecord
from facecore.research.recorder import ResearchRecorder


def _profile() -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="g3w5-test-v1",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=26,
        queue_limit=1,
        required_support=1,
        min_support_interval_ms=1,
        match_threshold=0.363,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="det-g3w5-test",
        quality_policy_version="qual-g3w5-test",
        continuity_max_center_delta_ratio=0.5,
    )


def _packet(seq: int) -> FramePacket:
    frame = np.full((8, 8, 3), 150, dtype=np.uint8)
    return FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=frame)


def _obs(seq: int) -> FrameObservation:
    packet = _packet(seq)
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(1.0, 2.0, 4.0, 5.0),
        identity_scores={"person-synth-01": 0.9, "person-synth-02": 0.1},
        quality_rank=0.9,
        model_generation="gen-g3w5-test",
        gallery_digest="gallery-g3w5-test",
    )


def _real_diag(seq: int) -> FrameDiagnostics:
    """True-valued diagnostics as score_frame would emit them."""
    return FrameDiagnostics(
        sequence=seq,
        original_shape=(8, 8, 3),
        normalized_shape=(8, 8, 3),
        orientation=0,
        mirrored=False,
        face_count=1,
        detector_confidence=0.93,
        face_box=(1.0, 2.0, 4.0, 5.0),
        landmarks=((1.5, 2.5), (3.5, 2.5)),
        landmark_confidence_is_constant=True,
        shorter_side_px=3,
        sharpness=42.5,
        mean_luma=148.0,
        clipped_fraction=0.01,
        yaw_deg=2.0,
        pitch_deg=-1.5,
        quality_status="accepted",
        quality_reason_codes=(),
        detection_missing_reason=None,
        quality_missing_reason=None,
        scoring_missing_reason=None,
        stage_durations_ms={"detection": 1.0},
    )


def _recorder(tmp_path: Path) -> ResearchRecorder:
    def clock() -> datetime:
        return datetime(2026, 9, 24, 4, tzinfo=timezone.utc)

    return ResearchRecorder(tmp_path / "store", tmp_path / "keys", clock=clock)


def _manifest() -> ExperimentManifest:
    return ExperimentManifest.from_dict(
        {
            "identity": {"experiment_id": "exp-g3w5"},
            "software": {},
            "gallery": {},
            "policy": {},
            "capture": {},
            "privacy": {},
            "study": {},
            "analysis": {},
        }
    )


def _attempt(attempt_id: str = "attempt-g3w5") -> AttemptRecord:
    return AttemptRecord(
        experiment_id="exp-g3w5",
        attempt_id=attempt_id,
        participant_id="participant-synth-01",
        visit_id="visit-g3w5-01",
        condition_id="condition-g3w5",
        attempt_index=1,
        retry_of=None,
        consent_ref="g3w5-session",
        requested_at_utc="2026-09-24T04:00:00Z",
        accepted_at_utc="2026-09-24T04:00:01Z",
        started_at_utc=None,
        ended_at_utc=None,
        operational_status="accepted",
        error_code=None,
        bundle_ref=None,
    )


def _consent(session_id: str = "g3w5-session") -> ConsentRecord:
    return ConsentRecord(
        session_id=session_id,
        participant_id="participant-synth-01",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-24T04:00:00Z",
        record_expires_at_utc="2026-10-24T04:00:00Z",
        image_expires_at_utc="2026-10-24T04:00:00Z",
    )


class TestRealDiagnostics:
    def test_from_observation_carries_true_diag(self) -> None:
        """A supplied true diag lands in the entry verbatim."""
        entry = FrameTraceEntry.from_observation(_obs(3), diag=_real_diag(3))
        diag = entry.diagnostics
        assert diag.detector_confidence == pytest.approx(0.93)
        assert diag.face_box == (1.0, 2.0, 4.0, 5.0)
        assert diag.landmarks == ((1.5, 2.5), (3.5, 2.5))
        assert diag.sharpness == pytest.approx(42.5)
        assert diag.mean_luma == pytest.approx(148.0)
        assert diag.yaw_deg == pytest.approx(2.0)
        assert diag.pitch_deg == pytest.approx(-1.5)
        assert diag.quality_status == "accepted"
        assert diag.shorter_side_px == 3

    def test_from_observation_rejects_sequence_mismatch(self) -> None:
        """A diag for another frame must never attach (fail-closed)."""
        with pytest.raises(ValueError):
            FrameTraceEntry.from_observation(_obs(3), diag=_real_diag(4))

    def test_live_trace_persists_true_values(self, tmp_path: Path) -> None:
        """End to end: scorer-emitted diags reach the encrypted trace."""
        recorder = _recorder(tmp_path)
        recorder.begin_attempt(_manifest(), _attempt(), _consent())
        holder: dict[str, Any] = {}

        def scorer(packet: FramePacket) -> FrameObservation:
            holder["desktop"].note_diagnostics(_real_diag(packet.sequence))
            obs = _obs(packet.sequence)
            return obs

        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-g3w5-test", "gen-g3w5-test"),
            source=FakeCapture(frames=[_packet(1), _packet(2), _packet(3)]),
            scorer=scorer,
            session_id="g3w5-session",
            trace_recorder=recorder,
            trace_attempt_id="attempt-g3w5",
        )
        holder["desktop"] = desktop
        desktop.on_start(_consent(), now_ns=0, device_id="default")
        desktop.run_until_terminal(max_steps=50)
        assert desktop.terminal is not None
        trace = recorder.read_trace("attempt-g3w5")
        # required_support=1 locks on the first matching frame, so one
        # scored frame reaches the trace writer; every entry must carry
        # true values.
        assert len(trace.entries) >= 1
        for entry in trace.entries:
            diag = entry.diagnostics
            assert diag.detector_confidence is not None
            assert diag.face_box is not None
            assert diag.landmarks is not None
            assert diag.sharpness is not None
            assert diag.mean_luma is not None
            assert diag.yaw_deg is not None
            assert diag.pitch_deg is not None
            assert diag.quality_status == "accepted"
        desktop.close()
