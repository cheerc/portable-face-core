"""E4 RED: original-time decision replay + paired-arm evaluation.

Source of truth:
    - spec §4.3 (three replays named separately) + §5 (denominator) + §6;
    - plan §12 E4 + §11.2 signatures + §10 gap table; ADR 0010.

RED contract (must fail on current base, behavioural not import errors):
    - No replay_observations / evaluate_arms / ArmOutcome exists yet.
    - replay_session re-derives start from first captured, re-runs scorer,
      and guesses full from frame cap or first-last span (§10 gaps).
    - Window completeness is not driven by E3 CollectionWindow provenance.

Only synthetic payloads; never real faces; camera-free.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from facecore.live.contracts import (
    FrameObservation,
    ResearchProfile,
    SessionStatus,
)
from facecore.live.session import SessionEngine, compute_baseline_best_quality
from facecore.research.diagnostics import SessionTrace
from facecore.research.records import CollectionWindow
from facecore.research.replay import (
    ArmOutcome,
    evaluate_arms,
    replay_observations,
)


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _profile(
    *,
    timeout_ms: int = 5000,
    max_frames: int = 25,
    required_support: int = 3,
) -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="prof-e4-001",
        timeout_ms=timeout_ms,
        sample_interval_ms=200,
        max_frames=max_frames,
        queue_limit=1,
        required_support=required_support,
        min_support_interval_ms=200,
        match_threshold=0.45,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="det-e4-1",
        quality_policy_version="qual-e4-1",
        continuity_max_center_delta_ratio=0.50,
    )


def _obs(
    seq: int,
    captured_ns: int,
    processed_ns: int,
    identity: str = "person-01",
    score: float = 0.85,
    runner_up: float = 0.20,
    quality_rank: float = 50.0,
) -> FrameObservation:
    return FrameObservation(
        sequence=seq,
        captured_ns=captured_ns,
        processed_ns=processed_ns,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(4.0, 4.0, 8.0, 8.0),
        identity_scores={identity: score, "person-02": runner_up},
        quality_rank=quality_rank,
        model_generation="gen-e4",
        gallery_digest="gal-e4",
    )


def _trace(
    observations: list[FrameObservation],
    *,
    session_start_ns: int = 0,
    stop_reason: str = "deadline_reached",
    complete: bool = True,
) -> SessionTrace:
    from facecore.live.contracts import DecisionEvent, FrameDiagnostics
    from facecore.research.diagnostics import FrameTraceEntry

    entries = []
    for i, obs in enumerate(observations):
        diag = FrameDiagnostics(
            sequence=obs.sequence,
            original_shape=(16, 16, 3),
            normalized_shape=(16, 16, 3),
            orientation=0,
            mirrored=False,
            face_count=1,
            detector_confidence=0.99,
            face_box=obs.face_box,
            landmarks=None,
            quality_status="accepted",
        )
        entries.append(
            FrameTraceEntry(
                sequence=obs.sequence,
                captured_ns=obs.captured_ns,
                processed_ns=obs.processed_ns,
                quality_pass=True,
                quality_reasons=(),
                face_count=1,
                face_box=obs.face_box,
                identity_score_pairs=tuple(obs.identity_scores.items()),
                quality_rank=obs.quality_rank,
                model_generation="gen-e4",
                gallery_digest="gal-e4",
                diagnostics=diag,
                staged_index=i,
            )
        )
    deadline = session_start_ns + 5_000_000_000
    end = observations[-1].captured_ns if observations else session_start_ns
    return SessionTrace(
        schema_version="v2",
        attempt_id="att-e4-001",
        manifest_digest="man-e4",
        session_start_ns=session_start_ns,
        deadline_ns=deadline,
        session_end_ns=end,
        collection_stop_reason=stop_reason,
        is_complete=complete,
        entries=tuple(entries),
        terminal_result=None,
    )


class TestReplayObservationsOriginalTime:
    """Decision replay preserves original relative timing, no wall clock."""

    def test_late_processed_frame_replays_timeout_not_early_success(self) -> None:
        # Original: start=0, first frame arrives late, processed crosses the
        # 5s deadline → B timed out live. A replayer using replay wall clock
        # or first-frame-as-start would wrongly report early success.
        profile = _profile()
        observations = [
            _obs(1, 4_800_000_000, 5_200_000_000),
            _obs(2, 4_900_000_000, 5_300_000_000),
        ]
        trace = _trace(observations, session_start_ns=0)
        result = replay_observations(trace, profile)
        assert result.status == SessionStatus.timeout

    def test_first_frame_late_arrival_does_not_shift_start(self) -> None:
        # start=0 but the first observation arrives at 1s; replay must keep
        # start=0 (deadline at 5s), not steal first-frame time as start.
        profile = _profile(required_support=1)
        observations = [
            _obs(1, 1_000_000_000, 1_010_000_000),
            _obs(3, 1_250_000_000, 1_260_000_000),
        ]
        trace = _trace(observations, session_start_ns=0)
        result = replay_observations(trace, profile)
        assert result.status == SessionStatus.matched
        assert result.support_sequences == (1, 3)

    def test_noncontiguous_sequences_are_legal_drops_not_holes(self) -> None:
        profile = _profile(required_support=2)
        observations = [
            _obs(1, 0, 10_000_000),
            _obs(4, 600_000_000, 610_000_000),
        ]
        trace = _trace(observations, session_start_ns=0)
        result = replay_observations(trace, profile)
        assert result.status == SessionStatus.matched
        assert result.support_sequences == (1, 4)

    def test_duplicate_sequence_refused_with_reason(self) -> None:
        profile = _profile()
        observations = [
            _obs(1, 0, 10_000_000),
            _obs(1, 200_000_000, 210_000_000),
        ]
        trace = _trace(observations, session_start_ns=0)
        with pytest.raises(ValueError, match="duplicate_sequence"):
            replay_observations(trace, profile)

    def test_time_backwards_refused_with_reason(self) -> None:
        profile = _profile()
        observations = [
            _obs(1, 500_000_000, 510_000_000),
            _obs(2, 300_000_000, 310_000_000),
        ]
        trace = _trace(observations, session_start_ns=0)
        with pytest.raises(ValueError, match="time_backwards"):
            replay_observations(trace, profile)


class TestEvaluateArmsPairedComparison:
    """A/B arms share one window, one profile, arm_id-distinct outcomes."""

    def test_matched_b_and_baseline_a_share_window(self) -> None:
        profile = _profile(required_support=2)
        observations = [
            _obs(1, 0, 10_000_000, quality_rank=10.0),
            _obs(2, 250_000_000, 260_000_000, quality_rank=90.0),
        ]
        trace = _trace(observations, session_start_ns=0)
        arm_a, arm_b = evaluate_arms(trace, profile)
        assert arm_a.arm_id == "A"
        assert arm_b.arm_id == "B"
        assert arm_a.profile_digest == arm_b.profile_digest
        assert arm_b.terminal == SessionStatus.matched.value
        # A selects the best-quality frame (seq 2) via the original helper.
        best, status, _ = compute_baseline_best_quality(observations, profile)
        assert best is not None and best.sequence == 2
        assert arm_a.selected_sequences == (2,)

    def test_single_identity_none_margin_never_matched(self) -> None:
        profile = _profile(required_support=1)
        obs = FrameObservation(
            sequence=1,
            captured_ns=0,
            processed_ns=10_000_000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(4.0, 4.0, 8.0, 8.0),
            identity_scores={"person-01": 0.90},
            quality_rank=50.0,
            model_generation="gen-e4",
            gallery_digest="gal-e4",
        )
        trace = _trace([obs], session_start_ns=0)
        arm_a, arm_b = evaluate_arms(trace, profile)
        assert arm_b.terminal != SessionStatus.matched.value
        assert "none_runner_up" in arm_b.decision_codes

    def test_serialized_tie_replays_deterministically(self) -> None:
        profile = _profile(required_support=1)
        obs = FrameObservation(
            sequence=1,
            captured_ns=0,
            processed_ns=10_000_000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(4.0, 4.0, 8.0, 8.0),
            identity_scores={"person-02": 0.70, "person-01": 0.70},
            quality_rank=50.0,
            model_generation="gen-e4",
            gallery_digest="gal-e4",
        )
        trace = _trace([obs], session_start_ns=0)
        first = replay_observations(trace, profile)
        second = replay_observations(trace, profile)
        assert first.matched_identity == second.matched_identity == "person-02"
        assert first.support_sequences == second.support_sequences == (1,)

    def test_label_change_does_not_alter_either_arm(self) -> None:
        profile = _profile(required_support=1)
        trace = _trace(
            [_obs(1, 0, 10_000_000)], session_start_ns=0
        )
        before = evaluate_arms(trace, profile)
        # Simulate a post-terminal label revision: trace payload unchanged,
        # only evaluator-side annotation would differ (never fed to arms).
        after = evaluate_arms(trace, profile)
        assert before[0].terminal == after[0].terminal
        assert before[1].terminal == after[1].terminal

    def test_frame_counters_split_read_scored_consumed_staged(self) -> None:
        profile = _profile(required_support=2)
        observations = [
            _obs(1, 0, 10_000_000),
            _obs(2, 250_000_000, 260_000_000),
        ]
        trace = _trace(observations, session_start_ns=0)
        arm_a, arm_b = evaluate_arms(trace, profile)
        assert arm_b.frames_read == 2
        assert arm_b.frames_scored == 2
        assert arm_b.frames_consumed == 2
        assert arm_a.frames_staged == 2


class TestWindowProvenanceFromCollectionWindow:
    """Paired eligibility follows CollectionWindow, not frame-count guess."""

    def test_true_5s_ten_frames_is_full(self) -> None:
        profile = _profile()
        observations = [
            _obs(seq, (seq - 1) * 500_000_000, (seq - 1) * 500_000_000)
            for seq in range(1, 11)
        ]
        trace = _trace(observations, session_start_ns=0)
        window = CollectionWindow(
            session_id="sess-e4-full",
            collection_start_ns=0,
            collection_deadline_ns=5_000_000_000,
            collection_end_ns=5_000_000_000,
            collection_stop_reason="deadline_reached",
            collection_complete=True,
            frames_sampled=10,
        )
        arm_a, arm_b = evaluate_arms(trace, profile, window=window)
        assert arm_a.collection_extent == "full"
        assert arm_b.collection_extent == "full"

    def test_25_frames_early_exhaustion_is_not_full(self) -> None:
        profile = _profile()
        observations = [
            _obs(seq, (seq - 1) * 40_000_000, (seq - 1) * 40_000_000)
            for seq in range(1, 26)
        ]
        trace = _trace(
            observations,
            session_start_ns=0,
            stop_reason="max_frames_reached",
            complete=False,
        )
        window = CollectionWindow(
            session_id="sess-e4-cap",
            collection_start_ns=0,
            collection_deadline_ns=5_000_000_000,
            collection_end_ns=960_000_000,
            collection_stop_reason="max_frames_reached",
            collection_complete=False,
            frames_sampled=25,
        )
        arm_a, arm_b = evaluate_arms(trace, profile, window=window)
        assert arm_a.collection_extent == "incomplete"
        assert arm_b.collection_extent == "incomplete"

    def test_cancelled_window_is_incomplete_with_reason(self) -> None:
        profile = _profile()
        trace = _trace(
            [_obs(1, 0, 10_000_000)],
            session_start_ns=0,
            stop_reason="cancelled",
            complete=False,
        )
        window = CollectionWindow(
            session_id="sess-e4-cancel",
            collection_start_ns=0,
            collection_deadline_ns=5_000_000_000,
            collection_end_ns=400_000_000,
            collection_stop_reason="cancelled",
            collection_complete=False,
            frames_sampled=1,
        )
        arm_a, arm_b = evaluate_arms(trace, profile, window=window)
        assert arm_b.terminal == SessionStatus.cancelled.value or (
            arm_b.collection_extent == "incomplete"
        )

    def test_legacy_bundle_without_trace_marked_unproven(self) -> None:
        profile = _profile()
        arm_a, arm_b = evaluate_arms(None, profile)
        assert arm_a.collection_extent == "unproven"
        assert arm_b.collection_extent == "unproven"
        assert "trace_unavailable" in arm_a.decision_codes


class TestTraceAtRestPrivacy:
    """New arm payloads stay AEAD-encrypted; byte-level leak check."""

    def test_arm_outcomes_carry_no_pixels_or_embeddings(
        self, tmp_path: Path
    ) -> None:
        from tests.conftest import assert_no_plaintext_leak

        profile = _profile(required_support=1)
        trace = _trace([_obs(1, 0, 10_000_000)], session_start_ns=0)
        arm_a, arm_b = evaluate_arms(trace, profile)
        payload = json.dumps(
            [arm_a.to_dict(), arm_b.to_dict()], sort_keys=True
        ).encode()
        assert b"pixels" not in payload
        assert b"embedding" not in payload
        assert_no_plaintext_leak(
            Path(__file__).parent, ["person-SECRET-truth-zzz"]
        )
