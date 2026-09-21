"""E2 RED tests: diagnostic trace contracts and engine integration (Phase 2B §12 E2).

Source of truth:
    - docs/specs/2026-09-16-phase2b-mac-recognition-research.md §5, §6;
    - docs/plans/2026-09-16-phase2b-mac-recognition-execution-plan.md §11.2, §12 E2;
    - ADR 0010 (Phase 2B evidence isolation).

RED contract:
    - ResearchRecorder has no append_trace / read_trace yet (AttributeError).
    - score_frame has no diagnostic_sink parameter yet (TypeError).
    - SessionEngine has no event_sink parameter yet (TypeError).
    - No pixels/embeddings in trace; bounded <= 25 entries;
      evaluator can reconstruct truth rank/score/gap.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from facecore.contracts.policy import PolicyProfile
from facecore.live.contracts import (
    DecisionEvent,
    FrameDiagnostics,
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionStatus,
)
from facecore.live.frame_pipeline import ScoringContext, score_frame
from facecore.live.session import SessionEngine
from facecore.pipeline.embed import Embedder
from facecore.research.diagnostics import (
    FrameTraceEntry,
    SessionTrace,
    extract_truth_diagnostics,
)
from facecore.research.experiment import AttemptRecord, ExperimentManifest
from facecore.research.records import ConsentRecord
from facecore.research.recorder import ResearchRecorder


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _profile(
    *,
    timeout_ms: int = 5000,
    sample_interval_ms: int = 200,
    required_support: int = 3,
    min_support_interval_ms: int = 200,
    match_threshold: float = 0.45,
    review_threshold: float = 0.30,
    margin_threshold: float = 0.10,
    continuity_max_center_delta_ratio: float | None = 0.25,
) -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="prof-e2-001",
        timeout_ms=timeout_ms,
        sample_interval_ms=sample_interval_ms,
        max_frames=26,
        queue_limit=1,
        required_support=required_support,
        min_support_interval_ms=min_support_interval_ms,
        match_threshold=match_threshold,
        review_threshold=review_threshold,
        margin_threshold=margin_threshold,
        detector_version="det-test-1",
        quality_policy_version="qual-test-1",
        continuity_max_center_delta_ratio=continuity_max_center_delta_ratio,
    )


def _consent(session_id: str = "sess-e2-001") -> ConsentRecord:
    return ConsentRecord(
        session_id=session_id,
        participant_id="part-synth-001",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-16T10:00:00Z",
        record_expires_at_utc="2026-10-16T10:00:00Z",
        image_expires_at_utc="2026-09-23T10:00:00Z",
    )


def _manifest(experiment_id: str = "exp-e2-001") -> ExperimentManifest:
    return ExperimentManifest.from_dict(
        {
            "identity": {"experiment_id": experiment_id},
            "software": {},
            "gallery": {},
            "policy": {},
            "capture": {},
            "privacy": {},
            "study": {},
            "analysis": {},
        }
    )


def _attempt(
    attempt_id: str = "attempt-001", experiment_id: str = "exp-e2-001"
) -> AttemptRecord:
    return AttemptRecord(
        experiment_id=experiment_id,
        attempt_id=attempt_id,
        participant_id="part-synth-001",
        visit_id="visit-001",
        condition_id="cond-001",
        attempt_index=1,
        retry_of=None,
        consent_ref="consent-synth",
        requested_at_utc="2026-09-16T10:00:00Z",
        accepted_at_utc="2026-09-16T10:00:01Z",
        started_at_utc="2026-09-16T10:00:02Z",
        ended_at_utc=None,
        operational_status="accepted",
        error_code=None,
        bundle_ref=None,
    )


def _recorder(tmp_path: Path, now: datetime) -> ResearchRecorder:
    current = now
    return ResearchRecorder(
        store_root=tmp_path / "store",
        key_dir=tmp_path / "keys",
        clock=lambda: current,
    )


def _synthetic_packet(sequence: int = 1, captured_ns: int = 0) -> FramePacket:
    rng = np.random.default_rng(42)
    img = rng.integers(80, 180, size=(480, 640, 3), dtype=np.uint8)
    return FramePacket(
        sequence=sequence,
        captured_ns=captured_ns,
        rgb=img,
        orientation=0,
        mirrored=False,
    )


class TestDiagnosticsContracts:
    """Validate FrameDiagnostics, DecisionEvent, trace structures."""

    def test_frame_diagnostics_fields_and_truth_free(self) -> None:
        diag = FrameDiagnostics(
            sequence=1,
            original_shape=(480, 640, 3),
            normalized_shape=(480, 640, 3),
            orientation=0,
            mirrored=False,
            face_count=1,
            detector_confidence=0.98,
            face_box=(100.0, 100.0, 120.0, 120.0),
            landmarks=(
                (120.0, 130.0),
                (180.0, 130.0),
                (150.0, 160.0),
                (130.0, 190.0),
                (170.0, 190.0),
            ),
            landmark_confidence_is_constant=True,
            shorter_side_px=120,
            sharpness=45.2,
            mean_luma=128.0,
            clipped_fraction=0.01,
            yaw_deg=2.5,
            pitch_deg=-1.0,
            quality_status="accepted",
            quality_reason_codes=(),
            stage_durations_ms={"detection": 5.2, "quality": 1.1, "embed": 8.4},
        )
        assert diag.sequence == 1
        assert diag.face_count == 1
        assert diag.landmark_confidence_is_constant is True
        # Check no truth or identity fields exist on FrameDiagnostics
        assert not hasattr(diag, "identity_id")
        assert not hasattr(diag, "truth")
        assert not hasattr(diag, "label")

    def test_frame_diagnostics_missing_reason_when_no_face(self) -> None:
        diag = FrameDiagnostics(
            sequence=2,
            original_shape=(480, 640, 3),
            normalized_shape=(480, 640, 3),
            orientation=0,
            mirrored=False,
            face_count=0,
            detector_confidence=None,
            face_box=None,
            landmarks=None,
            detection_missing_reason="no_face_detected",
            quality_missing_reason="no_face_detected",
            scoring_missing_reason="no_face_detected",
        )
        assert diag.face_count == 0
        assert diag.detector_confidence is None
        assert diag.face_box is None
        assert diag.sharpness is None
        assert diag.detection_missing_reason == "no_face_detected"

    def test_decision_event_fields_and_uniqueness(self) -> None:
        event = DecisionEvent(
            sequence=3,
            event_type="continuity",
            accepted=True,
            reset_reason=None,
            support_before=1,
            support_after=2,
            candidate_before="person-01",
            candidate_after="person-01",
            terminal_status=None,
            terminal_identity=None,
            deadline_remaining_ms=4500.0,
        )
        assert event.sequence == 3
        assert event.event_type == "continuity"
        assert event.accepted is True
        assert event.support_after == 2

    def test_frame_trace_entry_preserves_score_order_and_excludes_pixels(self) -> None:
        diag = FrameDiagnostics(
            sequence=1,
            original_shape=(480, 640, 3),
            normalized_shape=(480, 640, 3),
            orientation=0,
            mirrored=False,
            face_count=1,
            detector_confidence=0.95,
            face_box=(50.0, 50.0, 100.0, 100.0),
            landmarks=None,
        )
        # Iteration order: person-02 first, then person-01 (same score: tie)
        score_pairs = (("person-02", 0.65), ("person-01", 0.65), ("person-03", 0.20))
        entry = FrameTraceEntry(
            sequence=1,
            captured_ns=1000,
            processed_ns=2000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(50.0, 50.0, 100.0, 100.0),
            identity_score_pairs=score_pairs,
            quality_rank=50.0,
            model_generation="gen-1",
            gallery_digest="gal-digest-1",
            diagnostics=diag,
            decision_event=None,
            staged_index=0,
            stage_missing_reason=None,
        )
        assert entry.identity_score_pairs[0] == ("person-02", 0.65)
        assert entry.identity_score_pairs[1] == ("person-01", 0.65)
        # Verify no pixel or embedding attributes exist
        assert not hasattr(entry, "rgb")
        assert not hasattr(entry, "pixels")
        assert not hasattr(entry, "crop")
        assert not hasattr(entry, "embedding")

    def test_session_trace_bounded_max_26(self) -> None:
        diag = FrameDiagnostics(
            sequence=1,
            original_shape=(480, 640, 3),
            normalized_shape=(480, 640, 3),
            orientation=0,
            mirrored=False,
            face_count=1,
            detector_confidence=0.9,
            face_box=None,
            landmarks=None,
        )
        entries = [
            FrameTraceEntry(
                sequence=i + 1,
                captured_ns=(i + 1) * 200_000_000,
                processed_ns=(i + 1) * 200_000_000 + 10_000_000,
                quality_pass=True,
                quality_reasons=(),
                face_count=1,
                face_box=None,
                identity_score_pairs=(),
                quality_rank=10.0,
                model_generation="gen-1",
                gallery_digest="gal-1",
                diagnostics=diag,
            )
            for i in range(27)
        ]
        with pytest.raises(ValueError, match="trace exceeded max 26"):
            SessionTrace(
                schema_version="v2",
                attempt_id="att-overflow",
                manifest_digest="man-1",
                session_start_ns=0,
                deadline_ns=5_000_000_000,
                session_end_ns=5_000_000_000,
                collection_stop_reason="max_frames_reached",
                is_complete=True,
                entries=tuple(entries),
                terminal_result=None,
            )


class TestPipelineDiagnosticsWiring:
    """Test score_frame diagnostic_sink and quality gating behavior."""

    def test_score_frame_emits_diagnostics_to_sink(self) -> None:
        # Mock detector and policy
        mock_detector = MagicMock()
        mock_detected_face = MagicMock()
        mock_detected_face.confidence = 0.99
        mock_detected_face.box = (50.0, 50.0, 120.0, 120.0)
        mock_detected_face.landmarks = [
            (70.0, 80.0),
            (130.0, 80.0),
            (100.0, 110.0),
            (80.0, 140.0),
            (120.0, 140.0),
        ]
        mock_detector.detect.return_value = [mock_detected_face]

        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed.return_value = (
            np.ones(128, dtype=np.float32),
            "test-embed-model",
        )

        policy = PolicyProfile.frozen_v1().with_thresholds(
            match_threshold=0.45, review_threshold=0.30, margin_threshold=0.10
        )
        mock_gallery = MagicMock()
        mock_gallery.embeddings = {"person-01": np.ones(128, dtype=np.float32)}
        mock_gallery.generation = "gen-test"
        mock_gallery.digest = "digest-test"
        mock_gallery.model_version = "test-embed-model"

        ctx = ScoringContext(
            detector=mock_detector,
            embedder=mock_embedder,
            policy=policy,
            gallery=mock_gallery,
            model_version="test-embed-model",
        )

        emitted: list[FrameDiagnostics] = []
        packet = _synthetic_packet(sequence=1, captured_ns=100)

        # Calling score_frame with diagnostic_sink
        obs = score_frame(packet, ctx, diagnostic_sink=emitted.append)
        assert isinstance(obs, FrameObservation)
        assert len(emitted) == 1
        diag = emitted[0]
        assert diag.sequence == 1
        assert diag.face_count == 1
        assert diag.detector_confidence == 0.99
        assert diag.quality_status == "accepted"

    def test_quality_fail_leaves_reason_and_zero_embedder_calls(self) -> None:
        mock_detector = MagicMock()
        mock_detected_face = MagicMock()
        mock_detected_face.confidence = 0.99
        mock_detected_face.box = (50.0, 50.0, 20.0, 20.0)  # small box (< 80px min)
        mock_detected_face.landmarks = [
            (55.0, 55.0),
            (65.0, 55.0),
            (60.0, 60.0),
            (58.0, 65.0),
            (62.0, 65.0),
        ]
        mock_detector.detect.return_value = [mock_detected_face]

        mock_embedder = MagicMock(spec=Embedder)
        policy = PolicyProfile.frozen_v1().with_thresholds(
            match_threshold=0.45, review_threshold=0.30, margin_threshold=0.10
        )
        mock_gallery = MagicMock()
        mock_gallery.embeddings = {"person-01": np.ones(128, dtype=np.float32)}
        mock_gallery.generation = "gen-test"
        mock_gallery.digest = "digest-test"
        mock_gallery.model_version = "test-embed-model"

        ctx = ScoringContext(
            detector=mock_detector,
            embedder=mock_embedder,
            policy=policy,
            gallery=mock_gallery,
            model_version="test-embed-model",
        )

        emitted: list[FrameDiagnostics] = []
        packet = _synthetic_packet(sequence=1, captured_ns=100)
        obs = score_frame(packet, ctx, diagnostic_sink=emitted.append)

        assert obs.quality_pass is False
        assert mock_embedder.embed.call_count == 0  # embedder count = 0
        assert len(emitted) == 1
        diag = emitted[0]
        assert diag.face_count == 1
        assert diag.quality_status != "accepted"
        assert diag.scoring_missing_reason == "quality_rejected"


class TestSessionEngineEventWiring:
    """Test SessionEngine event_sink emission across decision branches."""

    def test_distinct_events_for_reset_skip_and_continuity(self) -> None:
        profile = _profile(required_support=2, min_support_interval_ms=200)
        events: list[DecisionEvent] = []
        engine = SessionEngine(
            profile=profile,
            gallery_digest="gal-1",
            model_generation="gen-1",
            event_sink=events.append,
        )
        engine.start("sess-001", 0)

        # 1. First observation: meets match threshold -> candidate person-01
        obs1 = FrameObservation(
            sequence=1,
            captured_ns=0,
            processed_ns=10_000_000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(100.0, 100.0, 100.0, 100.0),
            identity_scores={"person-01": 0.80, "person-02": 0.20},
            quality_rank=50.0,
            model_generation="gen-1",
            gallery_digest="gal-1",
        )
        res1 = engine.observe(obs1)
        assert res1 is None
        assert len(events) == 1
        assert events[0].event_type == "identity_change"
        assert events[0].candidate_after == "person-01"
        assert events[0].support_after == 1

        # 2. Too fast (interval skip, 100ms < 200ms)
        obs2 = FrameObservation(
            sequence=2,
            captured_ns=100_000_000,
            processed_ns=110_000_000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(100.0, 100.0, 100.0, 100.0),
            identity_scores={"person-01": 0.85, "person-02": 0.20},
            quality_rank=50.0,
            model_generation="gen-1",
            gallery_digest="gal-1",
        )
        res2 = engine.observe(obs2)
        assert res2 is None
        assert len(events) == 2
        assert events[1].event_type == "interval_skip"

        # 3. None runner-up (only 1 candidate in score map -> margin cannot be computed)
        obs3 = FrameObservation(
            sequence=3,
            captured_ns=300_000_000,
            processed_ns=310_000_000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(100.0, 100.0, 100.0, 100.0),
            identity_scores={"person-01": 0.85},  # no runner up!
            quality_rank=50.0,
            model_generation="gen-1",
            gallery_digest="gal-1",
        )
        res3 = engine.observe(obs3)
        assert res3 is None
        assert len(events) == 3
        assert events[2].event_type == "none_runner_up"
        assert events[2].support_after == 0  # reset!

        # 4. Low score -> score_reset
        obs4 = FrameObservation(
            sequence=4,
            captured_ns=600_000_000,
            processed_ns=610_000_000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(100.0, 100.0, 100.0, 100.0),
            identity_scores={"person-01": 0.30, "person-02": 0.10},  # 0.30 < 0.45
            quality_rank=50.0,
            model_generation="gen-1",
            gallery_digest="gal-1",
        )
        engine.observe(obs4)
        assert len(events) == 4
        assert events[3].event_type == "score_reset"

        # 5. Late processing (> deadline)
        obs5 = FrameObservation(
            sequence=5,
            captured_ns=6_000_000_000,  # 6s > 5s timeout
            processed_ns=6_010_000_000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(100.0, 100.0, 100.0, 100.0),
            identity_scores={"person-01": 0.90, "person-02": 0.10},
            quality_rank=50.0,
            model_generation="gen-1",
            gallery_digest="gal-1",
        )
        res5 = engine.observe(obs5)
        assert res5 is not None
        assert res5.status == SessionStatus.timeout
        assert events[-1].event_type == "late_processing"

    def test_injected_clocks_terminal_result_identical_with_or_without_events(
        self,
    ) -> None:
        profile = _profile(required_support=2)
        observations = [
            FrameObservation(
                sequence=1,
                captured_ns=0,
                processed_ns=10_000_000,
                quality_pass=True,
                quality_reasons=(),
                face_count=1,
                face_box=(100.0, 100.0, 100.0, 100.0),
                identity_scores={"person-01": 0.80, "person-02": 0.20},
                quality_rank=50.0,
                model_generation="gen-1",
                gallery_digest="gal-1",
            ),
            FrameObservation(
                sequence=2,
                captured_ns=250_000_000,
                processed_ns=260_000_000,
                quality_pass=True,
                quality_reasons=(),
                face_count=1,
                face_box=(100.0, 100.0, 100.0, 100.0),
                identity_scores={"person-01": 0.85, "person-02": 0.20},
                quality_rank=50.0,
                model_generation="gen-1",
                gallery_digest="gal-1",
            ),
        ]

        # Run without event_sink
        engine_clean = SessionEngine(
            profile=profile, gallery_digest="gal-1", model_generation="gen-1"
        )
        engine_clean.start("s1", 0)
        res_clean = None
        for obs in observations:
            res_clean = engine_clean.observe(obs)
            if res_clean:
                break

        # Run with event_sink
        events: list[DecisionEvent] = []
        engine_traced = SessionEngine(
            profile=profile,
            gallery_digest="gal-1",
            model_generation="gen-1",
            event_sink=events.append,
        )
        engine_traced.start("s1", 0)
        res_traced = None
        for obs in observations:
            res_traced = engine_traced.observe(obs)
            if res_traced:
                break

        assert res_clean is not None
        assert res_traced is not None
        assert res_clean.status == res_traced.status
        assert res_clean.matched_identity == res_traced.matched_identity
        assert res_clean.support_sequences == res_traced.support_sequences


class TestEvaluatorTruthReconstruction:
    """Test evaluator-only extraction of truth rank, score, and truth gap."""

    def test_extract_truth_diagnostics_for_wrong_rank(self) -> None:
        diag = FrameDiagnostics(
            sequence=1,
            original_shape=(480, 640, 3),
            normalized_shape=(480, 640, 3),
            orientation=0,
            mirrored=False,
            face_count=1,
            detector_confidence=0.9,
            face_box=(10.0, 10.0, 50.0, 50.0),
            landmarks=None,
        )
        # Truth person-01 is rank 2 (0.60), runner-up to person-02 (0.75)
        entry = FrameTraceEntry(
            sequence=1,
            captured_ns=0,
            processed_ns=1000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(10.0, 10.0, 50.0, 50.0),
            identity_score_pairs=(
                ("person-02", 0.75),
                ("person-01", 0.60),
                ("person-03", 0.10),
            ),
            quality_rank=30.0,
            model_generation="gen-1",
            gallery_digest="gal-1",
            diagnostics=diag,
        )

        truth_diag = extract_truth_diagnostics(entry, truth_identity="person-01")
        assert truth_diag["truth_score"] == 0.60
        assert truth_diag["truth_rank"] == 2
        assert truth_diag["highest_nontruth_score"] == 0.75
        # truth_gap = score(truth) - max(score(other identities)) = 0.60 - 0.75 = -0.15
        assert pytest.approx(truth_diag["truth_gap"]) == -0.15

    def test_extract_truth_diagnostics_for_unenrolled_or_none(self) -> None:
        diag = FrameDiagnostics(
            sequence=1,
            original_shape=(480, 640, 3),
            normalized_shape=(480, 640, 3),
            orientation=0,
            mirrored=False,
            face_count=1,
            detector_confidence=0.9,
            face_box=None,
            landmarks=None,
        )
        entry = FrameTraceEntry(
            sequence=1,
            captured_ns=0,
            processed_ns=1000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=None,
            identity_score_pairs=(("person-01", 0.40),),
            quality_rank=20.0,
            model_generation="gen-1",
            gallery_digest="gal-1",
            diagnostics=diag,
        )
        truth_diag = extract_truth_diagnostics(entry, truth_identity=None)
        assert truth_diag["truth_score"] is None
        assert truth_diag["truth_rank"] is None
        assert truth_diag["truth_gap"] is None


class TestTracePersistence:
    """Test append_trace and read_trace on ResearchRecorder."""

    def test_append_and_read_trace_round_trip(self, tmp_path: Path) -> None:
        now = _utc("2026-09-16T10:00:00Z")
        rec = _recorder(tmp_path, now)
        manifest = _manifest("exp-e2-trace")
        attempt = _attempt("att-trace-001", "exp-e2-trace")
        consent = _consent("sess-trace-001")

        rec.begin_attempt(manifest, attempt, consent)

        diag = FrameDiagnostics(
            sequence=1,
            original_shape=(480, 640, 3),
            normalized_shape=(480, 640, 3),
            orientation=0,
            mirrored=False,
            face_count=1,
            detector_confidence=0.95,
            face_box=(100.0, 100.0, 120.0, 120.0),
            landmarks=None,
            sharpness=40.0,
            quality_status="accepted",
        )
        entry = FrameTraceEntry(
            sequence=1,
            captured_ns=0,
            processed_ns=10_000_000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(100.0, 100.0, 120.0, 120.0),
            identity_score_pairs=(
                ("person-01", 0.82),
                ("person-02", 0.35),
            ),
            quality_rank=40.0,
            model_generation="gen-1",
            gallery_digest="gal-1",
            diagnostics=diag,
        )

        rec.append_trace("att-trace-001", entry)
        trace = rec.read_trace("att-trace-001")

        assert isinstance(trace, SessionTrace)
        assert trace.attempt_id == "att-trace-001"
        assert len(trace.entries) == 1
        loaded_entry = trace.entries[0]
        assert loaded_entry.sequence == 1
        assert loaded_entry.identity_score_pairs == (
            ("person-01", 0.82),
            ("person-02", 0.35),
        )
        assert loaded_entry.diagnostics.sharpness == 40.0
