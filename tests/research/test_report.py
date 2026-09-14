"""Phase 2A Task T6 tests: research report denominators and grouping.

Source of truth: Phase 2A Implementation Plan §4 & §6 T6;
Task: t-20260914111204047555-76424-37;
Governing decision: d-20260914110757304910-5.

RED contract:
- label leak into inference is refused at the summarize boundary
  (summarize takes labels, never feeds them back into replay).
- error denominator: refused sessions appear under error with reasons,
  never silently dropped and never counted as unknown.
- pairing: same session under different profiles is paired; omitted
  sessions list reasons.
Only synthetic payloads; never real faces.
"""

from __future__ import annotations

import pytest

from facecore.live.contracts import SessionResult, SessionStatus
from facecore.research.replay import ReplayRefusal, ReplayResult
from facecore.research.report import (
    LabeledOutcome,
    ResearchReport,
    summarize,
)


def _result(
    session_id: str,
    status: SessionStatus,
    *,
    elapsed_ms: float = 1000.0,
    matched: str | None = None,
) -> SessionResult:
    return SessionResult(
        session_id=session_id,
        schema_version="v1",
        status=status,
        matched_identity=matched,
        reason_codes=("t6-synth",),
        elapsed_ms=elapsed_ms,
        frames_sampled=5,
        frames_usable=5,
        frames_rejected=0,
        frames_dropped=0,
        support_sequences=(),
        profile_digest="0" * 64,
        model_generation="test-gen",
        gallery_digest="1" * 64,
    )


def _replay(
    session_id: str,
    status: SessionStatus,
    window: str = "full",
    *,
    elapsed_ms: float = 1000.0,
    matched: str | None = None,
) -> ReplayResult:
    return ReplayResult(
        session_id=session_id,
        result=_result(session_id, status, elapsed_ms=elapsed_ms, matched=matched),
        window=window,  # type: ignore[arg-type]
        frames_replayed=5,
        profile_version="t6-test-v1",
    )


def test_denominators_count_every_attempt() -> None:
    outcomes = [
        LabeledOutcome(
            replay=_replay("s1", SessionStatus.matched, matched="person-01"),
            label="person-01",
            split="development",
        ),
        LabeledOutcome(
            replay=_replay("s2", SessionStatus.matched, matched="person-02"),
            label="person-01",
            split="development",
        ),
        LabeledOutcome(
            replay=_replay("s3", SessionStatus.unknown),
            label="not_enrolled",
            split="development",
        ),
        LabeledOutcome(
            replay=_replay("s4", SessionStatus.timeout, elapsed_ms=5000.0),
            label="person-01",
            split="development",
        ),
        LabeledOutcome(
            replay=_replay("s5", SessionStatus.invalid_input),
            label=None,
            split="development",
        ),
        LabeledOutcome(
            replay=_replay("s6", SessionStatus.error),
            label="person-01",
            split="development",
        ),
        LabeledOutcome(
            replay=_replay("s7", SessionStatus.cancelled),
            label="person-01",
            split="development",
        ),
    ]
    report = summarize(outcomes, refusals=[])
    assert report.attempted == 7
    assert report.labeled == 6
    assert report.eligible == 6
    assert report.successful == 1
    assert report.correct == 1
    assert report.wrong_enrolled_identity == 1
    assert report.unknown_false_accept == 0
    assert report.timeouts == 1
    assert report.invalid_inputs == 1
    assert report.errors == 1
    assert report.cancelled == 1


def test_unknown_false_accept_counts_not_me_matched() -> None:
    outcomes = [
        LabeledOutcome(
            replay=_replay("s1", SessionStatus.matched, matched="person-01"),
            label="not_me",
            split="development",
        ),
    ]
    report = summarize(outcomes, refusals=[])
    assert report.unknown_false_accept == 1
    assert report.correct == 0


def test_refusals_land_in_error_denominator_with_reasons() -> None:
    outcomes = [
        LabeledOutcome(
            replay=_replay("s1", SessionStatus.matched, matched="person-01"),
            label="person-01",
            split="development",
        ),
    ]
    refusals = [
        ReplayRefusal(session_id="s2", kind="missing", detail="not found"),
        ReplayRefusal(session_id="s3", kind="tampered", detail="wire mismatch"),
    ]
    report = summarize(outcomes, refusals=refusals)
    assert report.attempted == 3
    assert report.errors == 2
    assert "s2:missing" in report.omitted
    assert "s3:tampered" in report.omitted


def test_profile_pairing_and_omitted_reasons() -> None:
    outcomes = [
        LabeledOutcome(
            replay=_replay("s1", SessionStatus.matched, matched="person-01"),
            label="person-01",
            split="development",
            profile_version="prof-a",
        ),
        LabeledOutcome(
            replay=_replay("s1", SessionStatus.review),
            label="person-01",
            split="development",
            profile_version="prof-b",
        ),
    ]
    report = summarize(outcomes, refusals=[])
    assert report.paired_sessions == ["s1"]
    assert report.omitted == []


def test_latency_separates_matched_from_censored() -> None:
    outcomes = [
        LabeledOutcome(
            replay=_replay(
                "s1", SessionStatus.matched, elapsed_ms=1200.0, matched="p1"
            ),
            label="p1",
            split="development",
        ),
        LabeledOutcome(
            replay=_replay("s2", SessionStatus.timeout, elapsed_ms=5000.0),
            label="p1",
            split="development",
        ),
    ]
    report = summarize(outcomes, refusals=[])
    assert report.matched_latency_ms == [1200.0]
    assert report.censored_latency_ms == [5000.0]
    assert isinstance(report, ResearchReport)


def test_holdout_touched_by_tuning_is_reflagged_development() -> None:
    outcomes = [
        LabeledOutcome(
            replay=_replay("s1", SessionStatus.unknown),
            label="person-01",
            split="holdout",
            used_for_tuning=True,
        ),
    ]
    report = summarize(outcomes, refusals=[])
    assert report.holdout_count == 0
    assert report.development_count == 1
    assert "s1:holdout-reflagged-development" in report.omitted


def test_unlabeled_sessions_excluded_from_eligible() -> None:
    outcomes = [
        LabeledOutcome(
            replay=_replay("s1", SessionStatus.unknown),
            label=None,
            split="development",
        ),
    ]
    report = summarize(outcomes, refusals=[])
    assert report.attempted == 1
    assert report.labeled == 0
    assert report.eligible == 0


def test_empty_inputs_report_honest_zeros() -> None:
    report = summarize([], refusals=[])
    assert report.attempted == 0
    assert report.to_dict()["attempted"] == 0


def test_summarize_rejects_label_feedback_into_inference() -> None:
    # Labels must not alter stored inference bytes: summarize is a pure
    # annotation pass; re-running replay is unnecessary and forbidden here.
    outcomes = [
        LabeledOutcome(
            replay=_replay("s1", SessionStatus.unknown),
            label="person-01",
            split="development",
        ),
    ]
    before = outcomes[0].replay.result.to_dict()
    summarize(outcomes, refusals=[])
    assert outcomes[0].replay.result.to_dict() == before
    with pytest.raises(TypeError):
        summarize(outcomes, refusals=[], replay_fn=object())  # type: ignore[arg-type]
