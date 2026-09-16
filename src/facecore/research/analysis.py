"""Phase 2B study analysis: denominators, paired arms, diagnostics, triggers (E5, G1).

Source of truth:
    - docs/specs/2026-09-16-phase2b-mac-recognition-research.md §5, §6, §7, §8, §9;
    - docs/plans/2026-09-16-phase2b-mac-recognition-execution-plan.md
      §9.1, §11.2, §12 E5;
    - ADR 0010 (Phase 2B evidence isolation).

Hard boundaries:
    - Attempt ledger is authoritative (ADR 0010 item 2): outcomes/runs/arms
      never expand the attempted denominator.
    - Ground truth is evaluator-only: truth is never fed to scorer, quality,
      support rules, or frame selection.
    - Honest zero denominator: 0/0 is 'not estimable', never 0.0 or '0%'.
    - Hard triggers (T01, T02, T03) are never masked by overall averages.
    - Detailed cases are AEAD-encrypted at rest; stdout gets only aggregates
      and run code.
    - No real faces, camera-free, synthetic test doubles only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from facecore.live.contracts import ResearchProfile
from facecore.research.diagnostics import SessionTrace
from facecore.research.experiment import AttemptRecord, EvaluationLabel
from facecore.research.replay import ArmOutcome

# Triggers (§7)
TRIGGER_T01 = "T01"  # wrong enrolled matched / unenrolled false accept (HARD)
TRIGGER_T02 = "T02"  # live/decision replay discrepancy (HARD)
TRIGGER_T03 = "T03"  # accounting/denominator/label/split discrepancy (HARD)
TRIGGER_T04 = "T04"  # high quality wrong rank or truth gap <= 0
TRIGGER_T05 = "T05"  # truth rank=1 but blocked by score or margin
TRIGGER_T06 = "T06"  # A correct, B unsuccessful
TRIGGER_T07 = "T07"  # B wrong with stable support
TRIGGER_T08 = "T08"  # low usable frames / bottleneck
TRIGGER_T09 = "T09"  # condition repeated worse
TRIGGER_T10 = "T10"  # enrolled not correctly matched

HARD_TRIGGERS = frozenset({TRIGGER_T01, TRIGGER_T02, TRIGGER_T03})

# Failure Layers (§6)
FAILURE_LAYER_CAPTURE = "capture"
FAILURE_LAYER_QUALITY = "quality"
FAILURE_LAYER_RANKING = "ranking"
FAILURE_LAYER_THRESHOLD = "threshold"
FAILURE_LAYER_TEMPORAL = "temporal"
FAILURE_LAYER_UNRESOLVED = "UNRESOLVED_EVIDENCE"


def format_rate(num: int, den: int) -> str:
    """Format an accuracy/error fraction honestly.

    Returns 'not estimable' when den == 0, never '0%' or '0.0%'.
    """
    if den <= 0:
        return "not estimable"
    pct = (num / den) * 100.0
    return f"{num}/{den} ({pct:.1f}%)"


@dataclass(frozen=True)
class ArmAnalysis:
    """Per-arm performance and error breakdown across attempted sessions."""

    arm_id: str
    correct: int = 0
    wrong_enrolled: int = 0
    unknown_false_accept: int = 0
    review: int = 0
    unknown: int = 0
    timeout: int = 0
    invalid_input: int = 0
    error: int = 0
    cancelled: int = 0
    refused: int = 0
    no_result: int = 0

    enrolled_correct_rate: float | None = None
    enrolled_wrong_rate: float | None = None
    unknown_fa_rate: float | None = None

    conditional_paired_enrolled_correct: int = 0
    conditional_paired_enrolled_count: int = 0
    conditional_paired_enrolled_correct_rate: float | None = None

    time_to_correct_ms: tuple[float, ...] = ()
    time_to_wrong_ms: tuple[float, ...] = ()
    nondecision_ms: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "correct": self.correct,
            "wrong_enrolled": self.wrong_enrolled,
            "unknown_false_accept": self.unknown_false_accept,
            "review": self.review,
            "unknown": self.unknown,
            "timeout": self.timeout,
            "invalid_input": self.invalid_input,
            "error": self.error,
            "cancelled": self.cancelled,
            "refused": self.refused,
            "no_result": self.no_result,
            "enrolled_correct_rate": self.enrolled_correct_rate,
            "enrolled_wrong_rate": self.enrolled_wrong_rate,
            "unknown_fa_rate": self.unknown_fa_rate,
            "conditional_paired_enrolled_correct": (
                self.conditional_paired_enrolled_correct
            ),
            "conditional_paired_enrolled_count": (
                self.conditional_paired_enrolled_count
            ),
            "conditional_paired_enrolled_correct_rate": (
                self.conditional_paired_enrolled_correct_rate
            ),
            "time_to_correct_ms": list(self.time_to_correct_ms),
            "time_to_wrong_ms": list(self.time_to_wrong_ms),
            "nondecision_ms": list(self.nondecision_ms),
        }


@dataclass(frozen=True)
class CaseSummary:
    """§9 case summary template for one attempted session."""

    attempt_id: str
    participant_id: str
    visit_id: str
    truth_kind: str
    truth_identity: str | None
    operational_status: str
    arm_a_terminal: str | None
    arm_b_terminal: str | None
    earliest_blocking_layer: str
    threshold_detail: str | None
    triggers: tuple[str, ...]
    evidence_locator: str
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "participant_id": self.participant_id,
            "visit_id": self.visit_id,
            "truth_kind": self.truth_kind,
            "truth_identity": self.truth_identity,
            "operational_status": self.operational_status,
            "arm_a_terminal": self.arm_a_terminal,
            "arm_b_terminal": self.arm_b_terminal,
            "earliest_blocking_layer": self.earliest_blocking_layer,
            "threshold_detail": self.threshold_detail,
            "triggers": list(self.triggers),
            "evidence_locator": self.evidence_locator,
            "recommended_action": self.recommended_action,
        }


@dataclass(frozen=True)
class BatchAnalysis:
    """Complete paired batch evaluation report (§5/§6/§7/§8/§9)."""

    experiment_id: str
    attempted: int
    labeled: int
    truth_known_enrolled: int
    unknown: int
    uncertain: int
    unlabeled: int
    paired_complete: int
    operation_errors: int
    participants_count: int
    visits_count: int

    arm_a: ArmAnalysis
    arm_b: ArmAnalysis

    triggers: dict[str, tuple[str, ...]]
    hard_triggers_tripped: bool

    layer_breakdown: dict[str, int]
    cases: tuple[CaseSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "attempted": self.attempted,
            "labeled": self.labeled,
            "truth_known_enrolled": self.truth_known_enrolled,
            "unknown": self.unknown,
            "uncertain": self.uncertain,
            "unlabeled": self.unlabeled,
            "paired_complete": self.paired_complete,
            "operation_errors": self.operation_errors,
            "participants_count": self.participants_count,
            "visits_count": self.visits_count,
            "arm_a": self.arm_a.to_dict(),
            "arm_b": self.arm_b.to_dict(),
            "triggers": {k: list(v) for k, v in self.triggers.items()},
            "hard_triggers_tripped": self.hard_triggers_tripped,
            "layer_breakdown": dict(self.layer_breakdown),
            "cases": [c.to_dict() for c in self.cases],
        }

    def render_summary(self) -> str:
        """Render human-readable aggregate summary for stdout/logging."""
        a_cor = format_rate(self.arm_a.correct, self.truth_known_enrolled)
        a_wrg = format_rate(self.arm_a.wrong_enrolled, self.truth_known_enrolled)
        a_fa = format_rate(self.arm_a.unknown_false_accept, self.unknown)
        b_cor = format_rate(self.arm_b.correct, self.truth_known_enrolled)
        b_wrg = format_rate(self.arm_b.wrong_enrolled, self.truth_known_enrolled)
        b_fa = format_rate(self.arm_b.unknown_false_accept, self.unknown)
        cond_a = format_rate(
            self.arm_a.conditional_paired_enrolled_correct,
            self.arm_a.conditional_paired_enrolled_count,
        )
        cond_b = format_rate(
            self.arm_b.conditional_paired_enrolled_correct,
            self.arm_b.conditional_paired_enrolled_count,
        )

        lines = [
            f"=== Batch Analysis: {self.experiment_id} ===",
            (
                f"Attempted: {self.attempted} "
                f"(enrolled={self.truth_known_enrolled}, "
                f"unknown={self.unknown}, uncertain={self.uncertain}, "
                f"unlabeled={self.unlabeled})"
            ),
            f"Participants: {self.participants_count}, Visits: {self.visits_count}",
            (
                f"Paired-complete: {self.paired_complete}, "
                f"Operation errors: {self.operation_errors}"
            ),
            f"Arm A (Quality Best): correct={a_cor}, wrong={a_wrg}, unknown_fa={a_fa}",
            f"Arm B (Consistency): correct={b_cor}, wrong={b_wrg}, unknown_fa={b_fa}",
            f"Conditional Paired-Complete Enrolled: Arm A={cond_a}, Arm B={cond_b}",
            f"Hard Triggers Tripped: {self.hard_triggers_tripped}",
        ]
        if self.triggers:
            lines.append("Triggers:")
            for trig, att_ids in sorted(self.triggers.items()):
                lines.append(f"  [{trig}]: {', '.join(att_ids)}")
        return "\n".join(lines)


def analyze_batch(
    attempts: list[AttemptRecord],
    outcomes: list[ArmOutcome],
    labels: list[EvaluationLabel],
    *,
    traces: dict[str, SessionTrace] | None = None,
    profile: ResearchProfile | None = None,
    mode: str = "development",
) -> BatchAnalysis:
    """Analyze a batch of attempts, paired outcomes, and labels (§5-§9)."""
    # Minimal stub for RED test phase: returns empty dummy to fail behavioral assertions
    return BatchAnalysis(
        experiment_id="",
        attempted=0,
        labeled=0,
        truth_known_enrolled=0,
        unknown=0,
        uncertain=0,
        unlabeled=0,
        paired_complete=0,
        operation_errors=0,
        participants_count=0,
        visits_count=0,
        arm_a=ArmAnalysis(arm_id="A"),
        arm_b=ArmAnalysis(arm_id="B"),
        triggers={},
        hard_triggers_tripped=False,
        layer_breakdown={},
        cases=(),
    )
