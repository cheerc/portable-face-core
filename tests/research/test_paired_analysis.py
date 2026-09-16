"""Phase 2B Task E5 tests: paired analysis, denominators, triggers, and diagnostics.

Source of truth:
    - docs/specs/2026-09-16-phase2b-mac-recognition-research.md §5, §6, §7, §8, §9;
    - docs/plans/2026-09-16-phase2b-mac-recognition-execution-plan.md
      §9.1, §11.2, §12 E5;
    - ADR 0010 (Phase 2B evidence isolation).

RED contract:
    - §9.1 six-row exact reconciliation: attempted=6, enrolled=4, unknown=2,
      paired_complete=5, A correct=1/4, B correct=0/4, A wrong=1/4, B wrong=1/4,
      unknown FA=1/2 for both arms.
    - Multiple runs, replay refusals, and two arms do not expand attempted count.
    - Honest zero denominators: 0/0 is 'not estimable', never 0.0 or '0%'.
    - Label revisions isolate to evaluator and never mutate original outcomes.
    - Diagnostics classify failure layers (capture, quality, ranking, threshold,
      temporal) or output UNRESOLVED_EVIDENCE when evidence is missing.
    - At-rest encryption protects case details with zero plaintext leaks.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from facecore.live.contracts import (
    FrameDiagnostics,
    ResearchProfile,
    SessionResult,
    SessionStatus,
)
from facecore.research.analysis import (
    FAILURE_LAYER_CAPTURE,
    FAILURE_LAYER_QUALITY,
    FAILURE_LAYER_RANKING,
    FAILURE_LAYER_THRESHOLD,
    FAILURE_LAYER_UNRESOLVED,
    TRIGGER_T01,
    TRIGGER_T06,
    CaseSummary,
    analyze_batch,
    format_rate,
)
from facecore.research.diagnostics import FrameTraceEntry, SessionTrace
from facecore.research.experiment import AttemptRecord, EvaluationLabel
from facecore.research.replay import ArmOutcome
from tests.conftest import assert_no_plaintext_leak


def _attempt(
    attempt_id: str,
    *,
    participant_id: str = "p1",
    visit_id: str = "v1",
    operational_status: str = "completed",
    error_code: str | None = None,
    bundle_ref: str | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        experiment_id="exp-e5",
        attempt_id=attempt_id,
        participant_id=participant_id,
        visit_id=visit_id,
        condition_id="cond-01",
        attempt_index=1,
        retry_of=None,
        consent_ref="cs-01",
        requested_at_utc="2026-09-16T08:00:00Z",
        accepted_at_utc="2026-09-16T08:00:01Z",
        started_at_utc="2026-09-16T08:00:02Z",
        ended_at_utc="2026-09-16T08:00:07Z",
        operational_status=operational_status,
        error_code=error_code,
        bundle_ref=bundle_ref,
    )


def _arm_outcome(
    attempt_id: str,
    arm_id: str,
    terminal: str,
    *,
    matched_identity: str | None = None,
    collection_extent: str = "full",
    decision_time_ns: int | None = 1_000_000_000,
    run_id: str = "run-001",
    refusal: str | None = None,
) -> ArmOutcome:
    return ArmOutcome(
        attempt_id=attempt_id,
        run_id=run_id,
        arm_id=arm_id,
        profile_digest="prof-" + "0" * 59,
        selected_sequences=(1,),
        support_sequences=(1, 2) if terminal == SessionStatus.matched.value else (),
        terminal=terminal,
        matched_identity=matched_identity,
        collection_extent=collection_extent,  # type: ignore[arg-type]
        decision_time_ns=decision_time_ns,
        decision_codes=(f"{arm_id}_{terminal}",),
        frames_read=5,
        frames_scored=5,
        frames_consumed=5,
        frames_staged=5,
        refusal=refusal,
    )


def _trace_with_scores(
    attempt_id: str,
    scores_per_frame: list[dict[str, float]],
    *,
    quality_pass: bool = True,
) -> SessionTrace:
    entries = []
    for i, scores in enumerate(scores_per_frame, start=1):
        diag = FrameDiagnostics(
            sequence=i,
            original_shape=(16, 16, 3),
            normalized_shape=(16, 16, 3),
            orientation=0,
            mirrored=False,
            face_count=1 if quality_pass else 0,
            detector_confidence=0.99 if quality_pass else 0.0,
            face_box=(4.0, 4.0, 8.0, 8.0) if quality_pass else None,
            landmarks=None,
            quality_status="accepted" if quality_pass else "rejected",
        )
        entries.append(
            FrameTraceEntry(
                sequence=i,
                captured_ns=(i - 1) * 200_000_000,
                processed_ns=(i - 1) * 200_000_000 + 10_000_000,
                quality_pass=quality_pass,
                quality_reasons=() if quality_pass else ("low_quality",),
                face_count=1 if quality_pass else 0,
                face_box=(4.0, 4.0, 8.0, 8.0) if quality_pass else None,
                identity_score_pairs=tuple(scores.items()),
                quality_rank=50.0,
                model_generation="gen-e5",
                gallery_digest="gal-e5",
                diagnostics=diag,
                staged_index=i,
            )
        )
    return SessionTrace(
        schema_version="v2",
        attempt_id=attempt_id,
        manifest_digest="man-e5",
        session_start_ns=0,
        deadline_ns=5_000_000_000,
        session_end_ns=1_000_000_000,
        collection_stop_reason="deadline_reached",
        is_complete=True,
        entries=tuple(entries),
        terminal_result=None,
    )


class TestSixRowReconciliation:
    """§9.1 Exact six-row accounting test case."""

    def test_six_row_reconciliation_exact(self) -> None:
        attempts = [
            _attempt("s1", participant_id="p1", visit_id="v1", bundle_ref="b_s1"),
            _attempt("s2", participant_id="p1", visit_id="v1", bundle_ref="b_s2"),
            _attempt("s3", participant_id="p1", visit_id="v1", bundle_ref="b_s3"),
            _attempt(
                "s4",
                participant_id="p1",
                visit_id="v1",
                operational_status="open_error",
                error_code="camera_open_failed",
                bundle_ref=None,
            ),
            _attempt("s5", participant_id="p2", visit_id="v1", bundle_ref="b_s5"),
            _attempt("s6", participant_id="p3", visit_id="v1", bundle_ref="b_s6"),
        ]

        labels = [
            EvaluationLabel(
                "s1", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            ),
            EvaluationLabel(
                "s2", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            ),
            EvaluationLabel(
                "s3", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            ),
            EvaluationLabel(
                "s4", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            ),
            EvaluationLabel(
                "s5", 1, "unenrolled", None, "evaluator", "2026-09-16T08:10:00Z"
            ),
            EvaluationLabel(
                "s6", 1, "unenrolled", None, "evaluator", "2026-09-16T08:10:00Z"
            ),
        ]

        outcomes = [
            # s1: A matched p1, B timeout
            _arm_outcome("s1", "A", "matched", matched_identity="p1"),
            _arm_outcome("s1", "B", "timeout"),
            # s2: A matched p2, B matched p2
            _arm_outcome("s2", "A", "matched", matched_identity="p2"),
            _arm_outcome("s2", "B", "matched", matched_identity="p2"),
            # s3: A invalid_input, B invalid_input
            _arm_outcome("s3", "A", "invalid_input"),
            _arm_outcome("s3", "B", "invalid_input"),
            # s4: no inference result (camera open error)
            # s5: A review, B timeout
            _arm_outcome("s5", "A", "review"),
            _arm_outcome("s5", "B", "timeout"),
            # s6: A matched p2, B matched p2
            _arm_outcome("s6", "A", "matched", matched_identity="p2"),
            _arm_outcome("s6", "B", "matched", matched_identity="p2"),
        ]

        traces = {
            "s1": _trace_with_scores("s1", [{"p1": 0.85, "p2": 0.20}]),
            "s2": _trace_with_scores("s2", [{"p2": 0.90, "p1": 0.30}]),
            "s3": _trace_with_scores("s3", [{}], quality_pass=False),
            "s5": _trace_with_scores("s5", [{"p1": 0.35, "p2": 0.31}]),
            "s6": _trace_with_scores("s6", [{"p2": 0.88, "p1": 0.15}]),
        }

        report = analyze_batch(attempts, outcomes, labels, traces=traces)

        # §9.1 Exact counts
        assert report.attempted == 6
        assert report.truth_known_enrolled == 4
        assert report.unknown == 2
        assert report.paired_complete == 5
        assert report.operation_errors == 1

        # E2E correct
        assert report.arm_a.correct == 1
        assert report.arm_a.enrolled_correct_rate == 0.25  # 1/4
        assert report.arm_b.correct == 0
        assert report.arm_b.enrolled_correct_rate == 0.0  # 0/4

        # E2E wrong enrolled
        assert report.arm_a.wrong_enrolled == 1
        assert report.arm_a.enrolled_wrong_rate == 0.25  # 1/4
        assert report.arm_b.wrong_enrolled == 1
        assert report.arm_b.enrolled_wrong_rate == 0.25  # 1/4

        # E2E unknown false accept
        assert report.arm_a.unknown_false_accept == 1
        assert report.arm_a.unknown_fa_rate == 0.50  # 1/2
        assert report.arm_b.unknown_false_accept == 1
        assert report.arm_b.unknown_fa_rate == 0.50  # 1/2

        # Conditional paired-enrolled correct (3 paired enrolled: s1, s2, s3)
        assert report.arm_a.conditional_paired_enrolled_count == 3
        assert report.arm_a.conditional_paired_enrolled_correct == 1
        assert (
            abs((report.arm_a.conditional_paired_enrolled_correct_rate or 0.0) - 1 / 3)
            < 1e-4
        )
        assert report.arm_b.conditional_paired_enrolled_count == 3
        assert report.arm_b.conditional_paired_enrolled_correct == 0
        assert report.arm_b.conditional_paired_enrolled_correct_rate == 0.0

        # Triggers: s1 -> T06, s2 & s6 -> T01
        assert "s1" in report.triggers.get(TRIGGER_T06, ())
        assert "s2" in report.triggers.get(TRIGGER_T01, ())
        assert "s6" in report.triggers.get(TRIGGER_T01, ())
        assert report.hard_triggers_tripped is True

    def test_same_attempt_multiple_runs_and_arms_do_not_expand_attempted(self) -> None:
        attempts = [
            _attempt("s1"),
            _attempt("s2"),
        ]
        labels = [
            EvaluationLabel(
                "s1", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            ),
            EvaluationLabel(
                "s2", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            ),
        ]
        # s2 has 2 runs: run-001 (matched) and run-002 (refused replay)
        outcomes = [
            _arm_outcome("s1", "A", "matched", matched_identity="p1", run_id="run-001"),
            _arm_outcome("s1", "B", "matched", matched_identity="p1", run_id="run-001"),
            _arm_outcome("s2", "A", "matched", matched_identity="p2", run_id="run-001"),
            _arm_outcome("s2", "B", "matched", matched_identity="p2", run_id="run-001"),
            _arm_outcome("s2", "A", "refused", refusal="tampered", run_id="run-002"),
            _arm_outcome("s2", "B", "refused", refusal="tampered", run_id="run-002"),
        ]
        report = analyze_batch(attempts, outcomes, labels)
        # attempted MUST remain 2! Never 6 or 12!
        assert report.attempted == 2

    def test_participant_and_visit_grouping(self) -> None:
        attempts = [
            _attempt("s1", participant_id="p1", visit_id="v1"),
            _attempt("s2", participant_id="p1", visit_id="v1"),
            _attempt("s3", participant_id="p1", visit_id="v2"),
            _attempt("s4", participant_id="p2", visit_id="v1"),
        ]
        labels = [
            EvaluationLabel(
                "s1", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            ),
            EvaluationLabel(
                "s2", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            ),
            EvaluationLabel(
                "s3", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            ),
            EvaluationLabel(
                "s4", 1, "enrolled", "p2", "evaluator", "2026-09-16T08:10:00Z"
            ),
        ]
        outcomes = [
            _arm_outcome("s1", "A", "timeout"),
            _arm_outcome("s1", "B", "timeout"),
            _arm_outcome("s2", "A", "timeout"),
            _arm_outcome("s2", "B", "timeout"),
            _arm_outcome("s3", "A", "timeout"),
            _arm_outcome("s3", "B", "timeout"),
            _arm_outcome("s4", "A", "timeout"),
            _arm_outcome("s4", "B", "timeout"),
        ]
        report = analyze_batch(attempts, outcomes, labels)
        assert report.attempted == 4
        assert report.participants_count == 2
        assert report.visits_count == 3  # (p1, v1), (p1, v2), (p2, v1)


class TestHonestZeroDenominator:
    """Zero denominator returns None / renders as 'not estimable', never 0.0 or 0%."""

    def test_zero_denominator_enrolled_and_unenrolled(self) -> None:
        attempts = [_attempt("s1", participant_id="p1", visit_id="v1")]
        # Only uncertain label
        labels = [
            EvaluationLabel(
                "s1", 1, "uncertain", None, "evaluator", "2026-09-16T08:10:00Z"
            )
        ]
        outcomes = [
            _arm_outcome("s1", "A", "review"),
            _arm_outcome("s1", "B", "timeout"),
        ]
        report = analyze_batch(attempts, outcomes, labels)
        assert report.attempted == 1
        assert report.truth_known_enrolled == 0
        assert report.unknown == 0
        assert report.uncertain == 1

        # Denominator is 0 -> rates MUST be None
        assert report.arm_a.enrolled_correct_rate is None
        assert report.arm_a.enrolled_wrong_rate is None
        assert report.arm_a.unknown_fa_rate is None
        assert report.arm_b.enrolled_correct_rate is None

        # format_rate honesty
        assert format_rate(0, 0) == "not estimable"
        summary_text = report.render_summary()
        assert "not estimable" in summary_text
        assert "0%" not in summary_text


class TestLabelChangeIsolation:
    """Label revision updates evaluator judgment without altering inference outcomes."""

    def test_label_revision_updates_classification_only(self) -> None:
        attempts = [_attempt("s1")]
        # Revision 1: uncertain
        labels_v1 = [
            EvaluationLabel(
                "s1", 1, "uncertain", None, "evaluator", "2026-09-16T08:10:00Z"
            )
        ]
        outcomes = [
            _arm_outcome("s1", "A", "matched", matched_identity="p1"),
            _arm_outcome("s1", "B", "matched", matched_identity="p1"),
        ]
        rep_v1 = analyze_batch(attempts, outcomes, labels_v1)
        assert rep_v1.uncertain == 1
        assert rep_v1.truth_known_enrolled == 0
        assert rep_v1.arm_a.correct == 0

        # Revision 2: corrected to enrolled p1
        labels_v2 = [
            EvaluationLabel(
                "s1", 1, "uncertain", None, "evaluator", "2026-09-16T08:10:00Z"
            ),
            EvaluationLabel(
                "s1", 2, "enrolled", "p1", "evaluator", "2026-09-16T08:20:00Z"
            ),
        ]
        rep_v2 = analyze_batch(attempts, outcomes, labels_v2)
        assert rep_v2.uncertain == 0
        assert rep_v2.truth_known_enrolled == 1
        assert rep_v2.arm_a.correct == 1
        # Outcome records are untouched
        assert outcomes[0].terminal == "matched"
        assert outcomes[0].matched_identity == "p1"


class TestDiagnosticClassification:
    """Failure root-cause classification into failure layers."""

    def test_capture_failure_layer(self) -> None:
        attempts = [
            _attempt(
                "s4", operational_status="open_error", error_code="camera_open_failed"
            )
        ]
        labels = [
            EvaluationLabel(
                "s4", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            )
        ]
        report = analyze_batch(attempts, [], labels)
        case = report.cases[0]
        assert case.earliest_blocking_layer == FAILURE_LAYER_CAPTURE

    def test_quality_failure_layer(self) -> None:
        attempts = [_attempt("s3")]
        labels = [
            EvaluationLabel(
                "s3", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            )
        ]
        outcomes = [
            _arm_outcome("s3", "A", "invalid_input"),
            _arm_outcome("s3", "B", "invalid_input"),
        ]
        traces = {"s3": _trace_with_scores("s3", [{}], quality_pass=False)}
        report = analyze_batch(attempts, outcomes, labels, traces=traces)
        case = report.cases[0]
        assert case.earliest_blocking_layer == FAILURE_LAYER_QUALITY

    def test_ranking_failure_layer(self) -> None:
        attempts = [_attempt("s2")]
        labels = [
            EvaluationLabel(
                "s2", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            )
        ]
        outcomes = [
            _arm_outcome("s2", "A", "matched", matched_identity="p2"),
            _arm_outcome("s2", "B", "matched", matched_identity="p2"),
        ]
        # p2 top1 (0.90), p1 ranked lower (0.30)
        traces = {"s2": _trace_with_scores("s2", [{"p2": 0.90, "p1": 0.30}])}
        report = analyze_batch(attempts, outcomes, labels, traces=traces)
        case = report.cases[0]
        assert case.earliest_blocking_layer == FAILURE_LAYER_RANKING

    def test_threshold_failure_layer(self) -> None:
        attempts = [_attempt("s_thresh")]
        labels = [
            EvaluationLabel(
                "s_thresh", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            )
        ]
        outcomes = [
            _arm_outcome("s_thresh", "A", "review"),
            _arm_outcome("s_thresh", "B", "timeout"),
        ]
        # p1 top1 (0.40 < 0.45 match threshold), p2 runner up (0.20)
        traces = {
            "s_thresh": _trace_with_scores("s_thresh", [{"p1": 0.40, "p2": 0.20}])
        }
        profile = ResearchProfile(
            schema_version="v1",
            profile_version="prof-e5",
            timeout_ms=5000,
            sample_interval_ms=200,
            max_frames=25,
            queue_limit=1,
            required_support=3,
            min_support_interval_ms=200,
            match_threshold=0.45,
            review_threshold=0.30,
            margin_threshold=0.10,
            detector_version="det-1",
            quality_policy_version="qual-1",
            continuity_max_center_delta_ratio=0.50,
        )
        report = analyze_batch(
            attempts, outcomes, labels, traces=traces, profile=profile
        )
        case = report.cases[0]
        assert case.earliest_blocking_layer == FAILURE_LAYER_THRESHOLD
        assert case.threshold_detail in (
            "score_only",
            "margin_only",
            "both",
            "none_runner_up",
        )

    def test_missing_trace_yields_unresolved_evidence(self) -> None:
        attempts = [_attempt("s_missing")]
        labels = [
            EvaluationLabel(
                "s_missing", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            )
        ]
        outcomes = [
            _arm_outcome("s_missing", "A", "timeout"),
            _arm_outcome("s_missing", "B", "timeout"),
        ]
        # No trace provided
        report = analyze_batch(attempts, outcomes, labels, traces={})
        case = report.cases[0]
        assert case.earliest_blocking_layer == FAILURE_LAYER_UNRESOLVED


class TestElapsedMsAndLatencySemantics:
    """Latency measures actual time-to-decision, never deadline as recognition time."""

    def test_latencies_separate_time_to_correct_wrong_nondecision(self) -> None:
        attempts = [
            _attempt("s_corr"),
            _attempt("s_wrng"),
            _attempt("s_tout"),
        ]
        labels = [
            EvaluationLabel(
                "s_corr", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            ),
            EvaluationLabel(
                "s_wrng", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            ),
            EvaluationLabel(
                "s_tout", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            ),
        ]
        outcomes = [
            _arm_outcome(
                "s_corr",
                "B",
                "matched",
                matched_identity="p1",
                decision_time_ns=1_200_000_000,
            ),
            _arm_outcome(
                "s_wrng",
                "B",
                "matched",
                matched_identity="p2",
                decision_time_ns=800_000_000,
            ),
            _arm_outcome("s_tout", "B", "timeout", decision_time_ns=5_100_000_000),
        ]
        report = analyze_batch(attempts, outcomes, labels)
        assert report.arm_b.time_to_correct_ms == (1200.0,)
        assert report.arm_b.time_to_wrong_ms == (800.0,)
        assert report.arm_b.nondecision_ms == (5100.0,)


class TestAtRestBytesNoPlaintextLeak:
    """Detailed case templates stored in AEAD store leak zero plaintext."""

    def test_case_file_aead_encrypted_at_rest(self, tmp_path: Path) -> None:
        # Synthetic secret tokens that must NOT appear in raw bytes
        secret_identity = "confidential-subject-xyz"
        secret_diagnostic = "proprietary-landmark-detail-123"

        case = CaseSummary(
            attempt_id="att-secret-01",
            participant_id=secret_identity,
            visit_id="vis-01",
            truth_kind="enrolled",
            truth_identity=secret_identity,
            operational_status="completed",
            arm_a_terminal="matched",
            arm_b_terminal="matched",
            earliest_blocking_layer="none",
            threshold_detail=secret_diagnostic,
            triggers=(),
            evidence_locator="loc-01",
            recommended_action="none",
        )

        from facecore.research.keys import ResearchKeyProvider
        from facecore.storage.cipher import AeadCipher
        from facecore.research.recorder import build_research_aad, _blob_to_wire

        key_dir = tmp_path / "keys"
        store_dir = tmp_path / "store"
        key_dir.mkdir()
        store_dir.mkdir()

        keys = ResearchKeyProvider(key_dir)
        dek = keys.get_or_create_record_key("att-secret-01")

        case_bytes = json.dumps(case.to_dict()).encode("utf-8")
        aad = build_research_aad("v2", "att-secret-01", "case_summary", "0")
        blob = AeadCipher(dek).encrypt(case_bytes, aad)

        case_dir = store_dir / "_cases" / "exp-e5"
        case_dir.mkdir(parents=True)
        (case_dir / "att-secret-01.enc").write_bytes(_blob_to_wire(blob))

        # Assert no plaintext leak across both key_dir and store_dir!
        assert_no_plaintext_leak(tmp_path, [secret_identity, secret_diagnostic])


class TestCLIAnalyzeIntegration:
    """F5: Test end-to-end CLI analyze invocation with evaluate_arms."""

    def test_cli_analyze_calls_evaluate_arms_and_saves_cases(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from datetime import datetime, timezone
        from facecore.research.cli import cmd_analyze
        from facecore.research.experiment import ExperimentManifest
        from facecore.research.records import ConsentRecord
        from facecore.research.recorder import ResearchRecorder

        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        store_dir.mkdir()
        key_dir.mkdir()

        def clock() -> datetime:
            return datetime(2026, 9, 16, 8, 0, 0, tzinfo=timezone.utc)

        recorder = ResearchRecorder(store_dir, key_dir, clock=clock)

        profile = ResearchProfile(
            schema_version="v1",
            profile_version="prof-e5",
            timeout_ms=5000,
            sample_interval_ms=200,
            max_frames=25,
            queue_limit=1,
            required_support=3,
            min_support_interval_ms=200,
            match_threshold=0.45,
            review_threshold=0.30,
            margin_threshold=0.10,
            detector_version="det-1",
            quality_policy_version="qual-1",
            continuity_max_center_delta_ratio=0.50,
        )
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(json.dumps(profile.to_dict()))

        manifest_data = {
            "identity": {
                "experiment_id": "exp-e5",
                "schema_version": "v2",
                "owner": "lead-test",
                "custodian": "custodian-test",
            },
            "software": {
                "code_sha": "0" * 40,
                "generation": "gen-e5",
            },
            "gallery": {
                "gallery_digest": "gal-e5",
            },
            "policy": {
                "profile_version": "prof-e5",
                "profile_digest": profile.profile_digest(),
            },
            "capture": {"device": "fake"},
            "privacy": {"record_ttl_days": 30},
            "study": {"participants": ["p1"]},
            "analysis": {"arms": ["A", "B"]},
        }
        manifest = ExperimentManifest.from_dict(manifest_data)

        secret_identity = "confidential-token-subject-777"
        consent = ConsentRecord(
            session_id="s1",
            participant_id=secret_identity,
            record_consent=True,
            image_consent=True,
            consented_at_utc="2026-09-16T08:00:00Z",
            record_expires_at_utc="2026-10-16T08:00:00Z",
            image_expires_at_utc="2026-09-23T08:00:00Z",
        )
        att = _attempt(
            "s1", participant_id=secret_identity, visit_id="v1", bundle_ref=None
        )
        recorder.begin_attempt(manifest, att, consent)

        # Write label
        lbl = EvaluationLabel(
            "s1", 1, "enrolled", secret_identity, "evaluator", "2026-09-16T08:10:00Z"
        )
        recorder.write_label(lbl)

        # Append trace
        trace = _trace_with_scores("s1", [{secret_identity: 0.85, "other": 0.20}])
        for entry in trace.entries:
            recorder.append_trace("s1", entry)

        # Run cmd_analyze (production path)
        rc = cmd_analyze(
            store=store_dir,
            key_dir=key_dir,
            experiment_id="exp-e5",
            mode="development",
            profile_path=profile_path,
        )
        assert rc == 0

        captured = capsys.readouterr()
        assert "=== Batch Analysis: exp-e5 ===" in captured.out
        assert "Run code: exp-e5-development" in captured.out
        assert "Attempted: 1" in captured.out

        # Verify case file written and AEAD-encrypted
        case_file = store_dir / "_cases" / "exp-e5" / "s1.enc"
        assert case_file.is_file()

        # Secret values must not leak in plaintext anywhere in store or keys
        assert_no_plaintext_leak(store_dir, [secret_identity])
        assert_no_plaintext_leak(key_dir, [secret_identity])


class TestReworkFindingsRED:
    """Rework r1 RED tests demonstrating dual reviewer findings."""

    def test_tampered_trace_fails_closed_in_cmd_analyze(
        self, tmp_path: Path
    ) -> None:
        from datetime import datetime, timezone
        from facecore.research.cli import cmd_analyze
        from facecore.research.experiment import ExperimentManifest
        from facecore.research.records import ConsentRecord
        from facecore.research.recorder import ResearchRecorder

        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        store_dir.mkdir()
        key_dir.mkdir()

        def clock() -> datetime:
            return datetime(2026, 9, 16, 8, 0, 0, tzinfo=timezone.utc)

        recorder = ResearchRecorder(store_dir, key_dir, clock=clock)

        manifest_data = {
            "identity": {
                "experiment_id": "exp-e5",
                "schema_version": "v2",
                "owner": "lead-test",
                "custodian": "custodian-test",
            },
            "software": {"code_sha": "0" * 40, "generation": "gen-e5"},
            "gallery": {"gallery_digest": "gal-e5"},
            "policy": {
                "profile_version": "prof-e5",
                "profile_digest": "prof-000",
            },
            "capture": {"device": "fake"},
            "privacy": {"record_ttl_days": 30},
            "study": {"participants": ["p1"]},
            "analysis": {"arms": ["A", "B"]},
        }
        manifest = ExperimentManifest.from_dict(manifest_data)
        consent = ConsentRecord(
            session_id="s1",
            participant_id="p1",
            record_consent=True,
            image_consent=True,
            consented_at_utc="2026-09-16T08:00:00Z",
            record_expires_at_utc="2026-10-16T08:00:00Z",
            image_expires_at_utc="2026-09-23T08:00:00Z",
        )
        att = _attempt("s1", participant_id="p1", visit_id="v1")
        recorder.begin_attempt(manifest, att, consent)
        trace = _trace_with_scores("s1", [{"p1": 0.85, "p2": 0.20}])
        for entry in trace.entries:
            recorder.append_trace("s1", entry)

        # Corrupt 1 byte in the encrypted trace file
        trace_file = store_dir / "_traces" / "s1" / "frame_0001.enc"
        raw = bytearray(trace_file.read_bytes())
        raw[-1] ^= 0xFF
        trace_file.write_bytes(bytes(raw))

        # Tampered trace MUST fail closed (rc != 0)
        rc = cmd_analyze(
            store=store_dir,
            key_dir=key_dir,
            experiment_id="exp-e5",
            mode="development",
        )
        assert rc != 0, f"expected fail-closed non-zero exit code, got {rc}"

    def test_live_vs_replay_divergence_trips_t02(self) -> None:
        attempts = [_attempt("s1")]
        labels = [
            EvaluationLabel(
                "s1", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            )
        ]
        live_result = SessionResult(
            session_id="s1",
            schema_version="v1",
            status=SessionStatus.matched,
            matched_identity="p1",
            reason_codes=("live_matched",),
            elapsed_ms=1000.0,
            frames_sampled=5,
            frames_usable=5,
            frames_rejected=0,
            frames_dropped=0,
            support_sequences=(1,),
            profile_digest="prof-" + "0" * 59,
            model_generation="gen-e5",
            gallery_digest="gal-e5",
        )
        trace = _trace_with_scores("s1", [{"p1": 0.85, "p2": 0.20}])
        trace_with_live = SessionTrace(
            schema_version="v2",
            attempt_id="s1",
            manifest_digest="man-e5",
            session_start_ns=0,
            deadline_ns=5_000_000_000,
            session_end_ns=1_000_000_000,
            collection_stop_reason="deadline_reached",
            is_complete=True,
            entries=trace.entries,
            terminal_result=live_result,
        )
        outcomes = [
            _arm_outcome("s1", "A", "matched", matched_identity="p1"),
            _arm_outcome("s1", "B", "timeout"),
        ]
        report = analyze_batch(
            attempts, outcomes, labels, traces={"s1": trace_with_live}
        )
        assert "T02" in report.triggers, f"expected T02, got {report.triggers}"
        assert report.hard_triggers_tripped is True

    def test_per_run_refusals_not_erased_and_paired_requires_same_run_profile(
        self,
    ) -> None:
        attempts = [_attempt("s1")]
        labels = [
            EvaluationLabel(
                "s1", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            )
        ]
        outcomes = [
            _arm_outcome(
                "s1", "A", "matched", matched_identity="p1", run_id="run-001"
            ),
            _arm_outcome(
                "s1", "B", "matched", matched_identity="p1", run_id="run-001"
            ),
            _arm_outcome(
                "s1", "A", "refused", run_id="run-002", refusal="tampered"
            ),
            _arm_outcome(
                "s1", "B", "refused", run_id="run-002", refusal="tampered"
            ),
        ]
        report = analyze_batch(attempts, outcomes, labels)
        assert report.arm_a.refused >= 1
        assert report.arm_b.refused >= 1

    def test_paired_complete_refuses_mismatched_run_or_profile(self) -> None:
        attempts = [_attempt("s1")]
        labels = [
            EvaluationLabel(
                "s1", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            )
        ]
        outcomes = [
            _arm_outcome(
                "s1", "A", "matched", matched_identity="p1", run_id="run-a"
            ),
            _arm_outcome(
                "s1", "B", "matched", matched_identity="p1", run_id="run-b"
            ),
        ]
        report = analyze_batch(attempts, outcomes, labels)
        assert report.paired_complete == 0

    def test_refused_replay_not_classified_as_temporal_or_t10(self) -> None:
        attempts = [_attempt("s1")]
        labels = [
            EvaluationLabel(
                "s1", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            )
        ]
        outcomes = [
            _arm_outcome("s1", "A", "matched", matched_identity="p1"),
            _arm_outcome("s1", "B", "refused", refusal="tampered"),
        ]
        traces = {"s1": _trace_with_scores("s1", [{"p1": 0.85, "p2": 0.20}])}
        report = analyze_batch(attempts, outcomes, labels, traces=traces)
        case = report.cases[0]
        assert case.earliest_blocking_layer in ("refused", "UNRESOLVED_EVIDENCE")
        assert "T10" not in case.triggers

    def test_cli_analyze_rejects_holdout_mode(self, tmp_path: Path) -> None:
        from datetime import datetime, timezone
        from facecore.research.cli import cmd_analyze
        from facecore.research.experiment import ExperimentManifest
        from facecore.research.records import ConsentRecord
        from facecore.research.recorder import ResearchRecorder

        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        store_dir.mkdir()
        key_dir.mkdir()

        def clock() -> datetime:
            return datetime(2026, 9, 16, 8, 0, 0, tzinfo=timezone.utc)

        recorder = ResearchRecorder(store_dir, key_dir, clock=clock)
        manifest_data = {
            "identity": {
                "experiment_id": "exp-e5",
                "schema_version": "v2",
                "owner": "lead-test",
                "custodian": "custodian-test",
            },
            "software": {"code_sha": "0" * 40, "generation": "gen-e5"},
            "gallery": {"gallery_digest": "gal-e5"},
            "policy": {
                "profile_version": "prof-e5",
                "profile_digest": "prof-000",
            },
            "capture": {"device": "fake"},
            "privacy": {"record_ttl_days": 30},
            "study": {"participants": ["p1"]},
            "analysis": {"arms": ["A", "B"]},
        }
        manifest = ExperimentManifest.from_dict(manifest_data)
        consent = ConsentRecord(
            session_id="s1",
            participant_id="p1",
            record_consent=True,
            image_consent=True,
            consented_at_utc="2026-09-16T08:00:00Z",
            record_expires_at_utc="2026-10-16T08:00:00Z",
            image_expires_at_utc="2026-09-23T08:00:00Z",
        )
        att = _attempt("s1", participant_id="p1", visit_id="v1")
        recorder.begin_attempt(manifest, att, consent)

        rc = cmd_analyze(
            store=store_dir,
            key_dir=key_dir,
            experiment_id="exp-e5",
            mode="holdout",
        )
        assert rc == 2, f"expected rc 2 rejecting holdout in E5, got {rc}"

    def test_cli_analyze_verifies_profile_against_provenance(
        self, tmp_path: Path
    ) -> None:
        from datetime import datetime, timezone
        from facecore.research.cli import cmd_analyze
        from facecore.research.experiment import ExperimentManifest
        from facecore.research.records import ConsentRecord
        from facecore.research.recorder import ResearchRecorder

        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        store_dir.mkdir()
        key_dir.mkdir()

        def clock() -> datetime:
            return datetime(2026, 9, 16, 8, 0, 0, tzinfo=timezone.utc)

        recorder = ResearchRecorder(store_dir, key_dir, clock=clock)

        manifest_data = {
            "identity": {
                "experiment_id": "exp-e5",
                "schema_version": "v2",
                "owner": "lead-test",
                "custodian": "custodian-test",
            },
            "software": {"code_sha": "0" * 40, "generation": "gen-e5"},
            "gallery": {"gallery_digest": "gal-e5"},
            "policy": {
                "profile_version": "prof-frozen-001",
                "profile_digest": "expected-frozen-digest",
            },
            "capture": {"device": "fake"},
            "privacy": {"record_ttl_days": 30},
            "study": {"participants": ["p1"]},
            "analysis": {"arms": ["A", "B"]},
        }
        manifest = ExperimentManifest.from_dict(manifest_data)
        consent = ConsentRecord(
            session_id="s1",
            participant_id="p1",
            record_consent=True,
            image_consent=True,
            consented_at_utc="2026-09-16T08:00:00Z",
            record_expires_at_utc="2026-10-16T08:00:00Z",
            image_expires_at_utc="2026-09-23T08:00:00Z",
        )
        att = _attempt("s1", participant_id="p1", visit_id="v1")
        recorder.begin_attempt(manifest, att, consent)

        rc = cmd_analyze(
            store=store_dir,
            key_dir=key_dir,
            experiment_id="exp-e5",
            mode="development",
        )
        assert rc != 0, "expected non-zero exit code when profile unverified"

    def test_stdout_summary_does_not_leak_attempt_ids(self) -> None:
        attempts = [_attempt("att-secret-123")]
        labels = [
            EvaluationLabel(
                "att-secret-123",
                1,
                "enrolled",
                "p1",
                "evaluator",
                "2026-09-16T08:10:00Z",
            )
        ]
        outcomes = [
            _arm_outcome(
                "att-secret-123", "A", "matched", matched_identity="p2"
            ),
            _arm_outcome(
                "att-secret-123", "B", "matched", matched_identity="p2"
            ),
        ]
        report = analyze_batch(attempts, outcomes, labels)
        summary = report.render_summary()
        assert "att-secret-123" not in summary, f"attempt id leaked: {summary}"

    def test_case_summary_deleted_on_withdraw_attempt(
        self, tmp_path: Path
    ) -> None:
        from datetime import datetime, timezone
        from facecore.research.analysis import save_case_summaries
        from facecore.research.recorder import ResearchRecorder

        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        store_dir.mkdir()
        key_dir.mkdir()

        def clock() -> datetime:
            return datetime(2026, 9, 16, 8, 0, 0, tzinfo=timezone.utc)

        recorder = ResearchRecorder(store_dir, key_dir, clock=clock)

        case = CaseSummary(
            attempt_id="s1",
            participant_id="p1",
            visit_id="v1",
            truth_kind="enrolled",
            truth_identity="p1",
            operational_status="completed",
            arm_a_terminal="matched",
            arm_b_terminal="matched",
            earliest_blocking_layer="none",
            threshold_detail=None,
            triggers=(),
            evidence_locator="loc-01",
            recommended_action="none",
        )
        saved = save_case_summaries(recorder, "exp-e5", [case])
        case_file = saved[0]
        assert case_file.is_file()

        recorder.withdraw_attempt("s1")
        assert not case_file.is_file(), f"case file survived: {case_file}"


class TestReworkR2FindingsRED:
    """Rework r2 RED tests for the 4 findings in Lead dispatch."""

    def test_refused_run_quarantines_attempt_out_of_paired_complete(
        self,
    ) -> None:
        attempts = [_attempt("s1")]
        labels = [
            EvaluationLabel(
                "s1", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
            )
        ]
        outcomes = [
            _arm_outcome(
                "s1", "A", "matched", matched_identity="p1", run_id="run-001"
            ),
            _arm_outcome(
                "s1", "B", "matched", matched_identity="p1", run_id="run-001"
            ),
            _arm_outcome(
                "s1", "A", "refused", run_id="run-002", refusal="tampered"
            ),
            _arm_outcome(
                "s1", "B", "refused", run_id="run-002", refusal="tampered"
            ),
        ]
        report = analyze_batch(attempts, outcomes, labels)
        assert report.paired_complete == 0, (
            "expected paired_complete=0 for refused attempt, "
            f"got {report.paired_complete}"
        )
        assert "T03" in report.triggers, (
            f"expected T03 on refused attempt, got {report.triggers}"
        )
        assert report.hard_triggers_tripped is True

    def test_missing_evidence_emits_blocking_t03(self) -> None:
        attempts = [_attempt("s_missing")]
        labels = [
            EvaluationLabel(
                "s_missing",
                1,
                "enrolled",
                "p1",
                "evaluator",
                "2026-09-16T08:10:00Z",
            )
        ]
        outcomes = [
            _arm_outcome("s_missing", "A", "timeout"),
            _arm_outcome("s_missing", "B", "timeout"),
        ]
        report = analyze_batch(attempts, outcomes, labels, traces={})
        case = report.cases[0]
        assert case.earliest_blocking_layer == FAILURE_LAYER_UNRESOLVED
        assert "T03" in report.triggers, (
            f"expected T03 on missing evidence, got {report.triggers}"
        )
        assert report.hard_triggers_tripped is True

    def test_purge_expired_deletes_case_files_directly(
        self, tmp_path: Path
    ) -> None:
        from datetime import datetime, timezone
        from facecore.research.analysis import save_case_summaries
        from facecore.research.recorder import ResearchRecorder

        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        store_dir.mkdir()
        key_dir.mkdir()

        def clock() -> datetime:
            return datetime(2026, 9, 16, 8, 0, 0, tzinfo=timezone.utc)

        recorder = ResearchRecorder(store_dir, key_dir, clock=clock)

        case = CaseSummary(
            attempt_id="s_exp",
            participant_id="p1",
            visit_id="v1",
            truth_kind="enrolled",
            truth_identity="p1",
            operational_status="completed",
            arm_a_terminal="matched",
            arm_b_terminal="matched",
            earliest_blocking_layer="none",
            threshold_detail=None,
            triggers=(),
            evidence_locator="loc-01",
            recommended_action="none",
        )
        saved = save_case_summaries(recorder, "exp-e5", [case])
        case_file = saved[0]
        assert case_file.is_file()

        # purge_expired at future time must guarantee case files are deleted
        exp_time = datetime(2026, 11, 16, 8, 0, 0, tzinfo=timezone.utc)
        recorder.purge_expired(exp_time)
        assert not case_file.is_file(), (
            f"case file {case_file} survived purge_expired!"
        )

    def test_bundle_less_attempt_profile_verification_fails_closed(
        self, tmp_path: Path
    ) -> None:
        from datetime import datetime, timezone
        from facecore.research.cli import cmd_analyze
        from facecore.research.experiment import ExperimentManifest
        from facecore.research.records import ConsentRecord
        from facecore.research.recorder import ResearchRecorder

        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        store_dir.mkdir()
        key_dir.mkdir()

        def clock() -> datetime:
            return datetime(2026, 9, 16, 8, 0, 0, tzinfo=timezone.utc)

        recorder = ResearchRecorder(store_dir, key_dir, clock=clock)

        frozen_profile = ResearchProfile(
            schema_version="v1",
            profile_version="prof-frozen-001",
            timeout_ms=5000,
            sample_interval_ms=200,
            max_frames=25,
            queue_limit=1,
            required_support=3,
            min_support_interval_ms=200,
            match_threshold=0.45,
            review_threshold=0.30,
            margin_threshold=0.10,
            detector_version="det-1",
            quality_policy_version="qual-1",
            continuity_max_center_delta_ratio=0.50,
        )
        frozen_digest = frozen_profile.profile_digest()

        manifest_data = {
            "identity": {
                "experiment_id": "exp-e5",
                "schema_version": "v2",
                "owner": "lead-test",
                "custodian": "custodian-test",
            },
            "software": {"code_sha": "0" * 40, "generation": "gen-e5"},
            "gallery": {"gallery_digest": "gal-e5"},
            "policy": {
                "profile_version": "prof-frozen-001",
                "profile_digest": frozen_digest,
            },
            "capture": {"device": "fake"},
            "privacy": {"record_ttl_days": 30},
            "study": {"participants": ["p1"]},
            "analysis": {"arms": ["A", "B"]},
        }
        manifest = ExperimentManifest.from_dict(manifest_data)
        consent = ConsentRecord(
            session_id="s1",
            participant_id="p1",
            record_consent=True,
            image_consent=True,
            consented_at_utc="2026-09-16T08:00:00Z",
            record_expires_at_utc="2026-10-16T08:00:00Z",
            image_expires_at_utc="2026-09-23T08:00:00Z",
        )
        # BUNDLE-LESS attempt (bundle_ref=None)
        att = _attempt(
            "s1", participant_id="p1", visit_id="v1", bundle_ref=None
        )
        recorder.begin_attempt(manifest, att, consent)

        trace = _trace_with_scores("s1", [{"p1": 0.85, "p2": 0.20}])
        for entry in trace.entries:
            recorder.append_trace("s1", entry)

        lbl = EvaluationLabel(
            "s1", 1, "enrolled", "p1", "evaluator", "2026-09-16T08:10:00Z"
        )
        recorder.write_label(lbl)

        # Invented profile with different thresholds
        invented_profile = ResearchProfile(
            schema_version="v1",
            profile_version="prof-invented-999",
            timeout_ms=5000,
            sample_interval_ms=200,
            max_frames=25,
            queue_limit=1,
            required_support=3,
            min_support_interval_ms=200,
            match_threshold=0.99,
            review_threshold=0.90,
            margin_threshold=0.50,
            detector_version="det-1",
            quality_policy_version="qual-1",
            continuity_max_center_delta_ratio=0.50,
        )
        invented_path = tmp_path / "invented.json"
        invented_path.write_text(json.dumps(invented_profile.to_dict()))

        rc = cmd_analyze(
            store=store_dir,
            key_dir=key_dir,
            experiment_id="exp-e5",
            mode="development",
            profile_path=invented_path,
        )
        assert rc != 0, (
            f"expected fail-closed on unverified bundle-less profile, got {rc}"
        )


class TestReworkR3FindingsRED:
    """Rework r3 RED tests for corrupt attempt ledger under purge_expired."""

    def test_corrupt_attempt_ledger_purges_case_fail_closed(
        self, tmp_path: Path
    ) -> None:
        from datetime import datetime, timezone
        from facecore.research.analysis import save_case_summaries
        from facecore.research.recorder import ResearchRecorder

        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        store_dir.mkdir()
        key_dir.mkdir()

        def clock() -> datetime:
            return datetime(2026, 9, 16, 8, 0, 0, tzinfo=timezone.utc)

        recorder = ResearchRecorder(store_dir, key_dir, clock=clock)

        # Create an attempt file that is corrupt / undecryptable
        att_dir = store_dir / "_attempts" / "exp-e5"
        att_dir.mkdir(parents=True)
        corrupt_att_file = att_dir / "s_corrupt.enc"
        corrupt_att_file.write_bytes(
            b"\x00\x00corrupt-junk-data-cannot-decrypt"
        )

        # Create a case file for this attempt
        case = CaseSummary(
            attempt_id="s_corrupt",
            participant_id="p1",
            visit_id="v1",
            truth_kind="enrolled",
            truth_identity="p1",
            operational_status="completed",
            arm_a_terminal="matched",
            arm_b_terminal="matched",
            earliest_blocking_layer="none",
            threshold_detail=None,
            triggers=(),
            evidence_locator="loc-01",
            recommended_action="none",
        )
        saved = save_case_summaries(recorder, "exp-e5", [case])
        case_file = saved[0]
        assert case_file.is_file()

        # Calling purge_expired must NOT crash with StoreCorruptionError,
        # and MUST delete the unverified case file (fail-closed deletion)!
        exp_time = datetime(2026, 11, 16, 8, 0, 0, tzinfo=timezone.utc)
        recorder.purge_expired(exp_time)

        assert not case_file.is_file(), (
            f"case file {case_file} survived purge_expired on corrupt attempt!"
        )
