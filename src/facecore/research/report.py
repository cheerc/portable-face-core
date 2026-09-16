"""Research evaluation report: denominators, pairing, grouping (T6).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T6;
    - Task: t-20260914111204047555-76424-37;
    - Governing decision: d-20260914110757304910-5.

Hard boundaries:
    - Labels are consumed here and only here. summarize is a pure
      annotation pass over sealed ReplayResults: it never feeds labels
      back into inference and never mutates the replayed results.
    - Every attempted session is counted: refusals land in the error
      denominator with reasons (omitted list), never silently dropped
      and never counted as unknown.
    - Same session under different profiles is paired explicitly.
    - Latency separates matched completions from censored attempts
      (timeout/invalid/error/cancel/unknown/review carry censoring).
    - development/holdout grouping is by visit/session. A holdout
      session touched by tuning is reflagged development on the spot.
    - No representative sample is claimed: with small or synthetic
      inputs the report carries no cross-identity or 500-person claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from facecore.live.contracts import SessionStatus
from facecore.research.analysis import format_rate
from facecore.research.replay import ReplayRefusal, ReplayResult

Split = Literal["development", "holdout"]


@dataclass(frozen=True)
class LabeledOutcome:
    """One replayed session annotated with its ground-truth label.

    ``label`` is an opaque operator string (``None`` = unlabeled);
    ``not_me``/``not_enrolled`` mark negative sessions. ``split`` is the
    visit/session grouping; ``used_for_tuning`` reflags a holdout row to
    development.
    """

    replay: ReplayResult
    label: str | None
    split: Split = "development"
    used_for_tuning: bool = False
    profile_version: str = "t6-test-v1"


@dataclass
class ResearchReport:
    """Denominator-explicit evaluation summary over sealed replays."""

    attempted: int = 0
    labeled: int = 0
    eligible: int = 0
    successful: int = 0
    correct: int = 0
    wrong_enrolled_identity: int = 0
    unknown_false_accept: int = 0
    timeouts: int = 0
    invalid_inputs: int = 0
    errors: int = 0
    cancelled: int = 0
    development_count: int = 0
    holdout_count: int = 0
    paired_sessions: list[str] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)
    matched_latency_ms: list[float] = field(default_factory=list)
    censored_latency_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "labeled": self.labeled,
            "eligible": self.eligible,
            "successful": self.successful,
            "correct": self.correct,
            "wrong_enrolled_identity": self.wrong_enrolled_identity,
            "unknown_false_accept": self.unknown_false_accept,
            "timeouts": self.timeouts,
            "invalid_inputs": self.invalid_inputs,
            "errors": self.errors,
            "cancelled": self.cancelled,
            "development_count": self.development_count,
            "holdout_count": self.holdout_count,
            "paired_sessions": list(self.paired_sessions),
            "omitted": list(self.omitted),
            "matched_latency_ms": list(self.matched_latency_ms),
            "censored_latency_ms": list(self.censored_latency_ms),
        }

    @property
    def accuracy_rate(self) -> float | None:
        """Correct match rate over eligible attempts; None when eligible == 0."""
        return (self.correct / self.eligible) if self.eligible > 0 else None

    def format_accuracy(self) -> str:
        """Format accuracy honestly; returns 'not estimable' when eligible == 0."""
        return format_rate(self.correct, self.eligible)


def _is_negative_label(label: str | None) -> bool:
    return label in ("not_me", "not_enrolled", "unknown_person")


def summarize(
    outcomes: list[LabeledOutcome],
    *,
    refusals: list[ReplayRefusal],
) -> ResearchReport:
    """Annotate sealed replays with labels; count every denominator.

    ``refusals`` (decrypt/gate-stage) count as error attempts with their
    kinds recorded in ``omitted``. No inference is (re-)run here.
    """
    report = ResearchReport()
    by_session: dict[str, set[str]] = {}

    for outcome in outcomes:
        replay = outcome.replay
        status = replay.result.status
        report.attempted += 1

        split: Split = outcome.split
        if split == "holdout" and outcome.used_for_tuning:
            split = "development"
            report.omitted.append(
                f"{replay.session_id}:holdout-reflagged-development"
            )
        if split == "holdout":
            report.holdout_count += 1
        else:
            report.development_count += 1

        by_session.setdefault(replay.session_id, set()).add(
            outcome.profile_version
        )

        if outcome.label is not None:
            report.labeled += 1
        else:
            # Unlabeled sessions cannot be scored for correctness.
            if status == SessionStatus.timeout:
                report.timeouts += 1
                report.censored_latency_ms.append(replay.result.elapsed_ms)
            elif status == SessionStatus.invalid_input:
                report.invalid_inputs += 1
                report.censored_latency_ms.append(replay.result.elapsed_ms)
            elif status == SessionStatus.error:
                report.errors += 1
                report.censored_latency_ms.append(replay.result.elapsed_ms)
            elif status == SessionStatus.cancelled:
                report.cancelled += 1
                report.censored_latency_ms.append(replay.result.elapsed_ms)
            else:
                report.censored_latency_ms.append(replay.result.elapsed_ms)
            continue

        report.eligible += 1
        if status == SessionStatus.matched:
            if _is_negative_label(outcome.label):
                report.unknown_false_accept += 1
                report.matched_latency_ms.append(replay.result.elapsed_ms)
            elif outcome.label == replay.result.matched_identity:
                report.successful += 1
                report.correct += 1
                report.matched_latency_ms.append(replay.result.elapsed_ms)
            else:
                report.wrong_enrolled_identity += 1
                report.matched_latency_ms.append(replay.result.elapsed_ms)
        elif status == SessionStatus.timeout:
            report.timeouts += 1
            report.censored_latency_ms.append(replay.result.elapsed_ms)
        elif status == SessionStatus.invalid_input:
            report.invalid_inputs += 1
            report.censored_latency_ms.append(replay.result.elapsed_ms)
        elif status == SessionStatus.error:
            report.errors += 1
            report.censored_latency_ms.append(replay.result.elapsed_ms)
        elif status == SessionStatus.cancelled:
            report.cancelled += 1
            report.censored_latency_ms.append(replay.result.elapsed_ms)
        else:
            # review / unknown: censored, no success claimed.
            report.censored_latency_ms.append(replay.result.elapsed_ms)

    for refusal in refusals:
        report.attempted += 1
        report.errors += 1
        report.omitted.append(f"{refusal.session_id}:{refusal.kind}")

    report.paired_sessions = sorted(
        session_id
        for session_id, versions in by_session.items()
        if len(versions) > 1
    )
    return report
