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
import json
from pathlib import Path
from typing import Any

from facecore.live.contracts import ResearchProfile, SessionStatus
from facecore.research.diagnostics import FrameTraceEntry, SessionTrace
from facecore.research.experiment import (
    STUDY_SCHEMA_VERSION,
    AttemptRecord,
    EvaluationLabel,
)
from facecore.research.recorder import (
    ResearchRecorder,
    _blob_to_wire,
    build_research_aad,
)
from facecore.research.replay import ArmOutcome
from facecore.storage.cipher import AeadCipher

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseSummary:
        return cls(
            attempt_id=str(data["attempt_id"]),
            participant_id=str(data["participant_id"]),
            visit_id=str(data["visit_id"]),
            truth_kind=str(data["truth_kind"]),
            truth_identity=data.get("truth_identity"),
            operational_status=str(data["operational_status"]),
            arm_a_terminal=data.get("arm_a_terminal"),
            arm_b_terminal=data.get("arm_b_terminal"),
            earliest_blocking_layer=str(data["earliest_blocking_layer"]),
            threshold_detail=data.get("threshold_detail"),
            triggers=tuple(str(t) for t in data.get("triggers", ())),
            evidence_locator=str(data["evidence_locator"]),
            recommended_action=str(data["recommended_action"]),
        )


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
                lines.append(f"  [{trig}]: {len(att_ids)} attempts")
        return "\n".join(lines)


def save_case_summaries(
    recorder: ResearchRecorder,
    experiment_id: str,
    cases: tuple[CaseSummary, ...] | list[CaseSummary],
) -> list[Path]:
    """Persist case summaries AEAD-encrypted at rest under attempt record keys."""
    case_dir = recorder._store / "_cases" / experiment_id
    case_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for case in cases:
        dek = recorder._keys.get_or_create_record_key(case.attempt_id)
        payload = case.to_dict()
        plaintext = json.dumps(payload, sort_keys=True).encode("utf-8")
        aad = build_research_aad(
            STUDY_SCHEMA_VERSION, case.attempt_id, "case_summary", "0"
        )
        blob = AeadCipher(dek).encrypt(plaintext, aad)
        target_path = case_dir / f"{case.attempt_id}.enc"
        recorder._atomic_write_bytes(target_path, _blob_to_wire(blob))
        saved_paths.append(target_path)
    return saved_paths


def _classify_failure_layer(
    attempt: AttemptRecord,
    lbl: EvaluationLabel | None,
    trace: SessionTrace | None,
    out_b: ArmOutcome | None,
    profile: ResearchProfile | None,
) -> tuple[str, str | None]:
    """Classify the earliest blocking layer for an enrolled failure (§6)."""
    if out_b is not None and out_b.refusal is not None:
        return "refused", out_b.refusal

    if (
        attempt.operational_status in ("open_error", "setup_error", "error")
        or attempt.error_code is not None
    ):
        return FAILURE_LAYER_CAPTURE, None

    if trace is None:
        return FAILURE_LAYER_UNRESOLVED, None

    # Quality check: any usable frames?
    has_usable = any(e.quality_pass for e in trace.entries)
    if not has_usable:
        return FAILURE_LAYER_QUALITY, None

    if lbl is None or lbl.kind != "enrolled" or not lbl.identity_id:
        return FAILURE_LAYER_UNRESOLVED, None

    truth_id = lbl.identity_id
    scored_entries = [e for e in trace.entries if e.identity_score_pairs]
    if not scored_entries:
        return FAILURE_LAYER_UNRESOLVED, None

    ever_scored = any(
        any(k == truth_id for k, _ in e.identity_score_pairs)
        for e in scored_entries
    )
    if not ever_scored:
        return FAILURE_LAYER_UNRESOLVED, None

    # Find highest score for truth identity
    best_truth_score = -1.0
    best_entry: FrameTraceEntry | None = None
    for e in scored_entries:
        for k, v in e.identity_score_pairs:
            if k == truth_id and v > best_truth_score:
                best_truth_score = v
                best_entry = e

    if best_entry is None:
        return FAILURE_LAYER_UNRESOLVED, None

    sorted_pairs = sorted(
        best_entry.identity_score_pairs, key=lambda x: x[1], reverse=True
    )
    top1_id, top1_val = sorted_pairs[0]
    runner_up_val = sorted_pairs[1][1] if len(sorted_pairs) > 1 else None

    if top1_id != truth_id:
        return FAILURE_LAYER_RANKING, None

    # Truth is rank 1 -> check threshold criteria
    match_thresh = profile.match_threshold if profile else 0.45
    margin_thresh = profile.margin_threshold if profile else 0.10

    if runner_up_val is None:
        return FAILURE_LAYER_THRESHOLD, "none_runner_up"

    margin = top1_val - runner_up_val
    score_bad = top1_val < match_thresh
    margin_bad = margin < margin_thresh

    if score_bad and margin_bad:
        return FAILURE_LAYER_THRESHOLD, "both"
    if score_bad:
        return FAILURE_LAYER_THRESHOLD, "score_only"
    if margin_bad:
        return FAILURE_LAYER_THRESHOLD, "margin_only"

    # Truth passed score and margin on individual frame, but failed temporal support
    return FAILURE_LAYER_TEMPORAL, None


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
    _ = mode
    traces = traces or {}
    experiment_id = attempts[0].experiment_id if attempts else "unknown-experiment"

    # Step 1: Attempt ledger is authoritative (ADR 0010 item 2)
    attempt_map: dict[str, AttemptRecord] = {}
    for a in attempts:
        if a.attempt_id not in attempt_map:
            attempt_map[a.attempt_id] = a

    attempted = len(attempt_map)
    participants = {a.participant_id for a in attempt_map.values()}
    visits = {(a.participant_id, a.visit_id) for a in attempt_map.values()}
    participants_count = len(participants)
    visits_count = len(visits)

    operation_errors = sum(
        1
        for a in attempt_map.values()
        if a.operational_status in ("open_error", "setup_error", "error")
        or a.error_code is not None
    )

    # Step 2: Labels (evaluator-only, latest revision wins)
    latest_labels: dict[str, EvaluationLabel] = {}
    for lbl_item in sorted(labels, key=lambda x: x.revision):
        if lbl_item.attempt_id in attempt_map:
            latest_labels[lbl_item.attempt_id] = lbl_item

    labeled = 0
    truth_known_enrolled = 0
    unknown = 0
    uncertain = 0
    unlabeled = 0

    for att_id in attempt_map:
        if att_id not in latest_labels:
            unlabeled += 1
        else:
            labeled += 1
            l_rec = latest_labels[att_id]
            if l_rec.kind == "enrolled":
                truth_known_enrolled += 1
            elif l_rec.kind == "unenrolled":
                unknown += 1
            elif l_rec.kind == "uncertain":
                uncertain += 1

    # Step 3: Outcomes grouped by attempt and run
    outcomes_by_attempt: dict[str, list[ArmOutcome]] = {
        att_id: [] for att_id in attempt_map
    }
    for outcome in outcomes:
        if outcome.attempt_id in attempt_map:
            outcomes_by_attempt[outcome.attempt_id].append(outcome)

    canonical_outcomes: dict[str, dict[str, ArmOutcome]] = {
        att_id: {} for att_id in attempt_map
    }
    for att_id, att_outs in outcomes_by_attempt.items():
        runs: dict[str, dict[str, ArmOutcome]] = {}
        for o in att_outs:
            runs.setdefault(o.run_id, {})[o.arm_id] = o

        # Find clean paired run (both A and B non-refused)
        chosen_run: dict[str, ArmOutcome] | None = None
        for r_id, arm_map in runs.items():
            if "A" in arm_map and "B" in arm_map:
                if arm_map["A"].refusal is None and arm_map["B"].refusal is None:
                    chosen_run = arm_map
                    break
        if chosen_run is None and runs:
            chosen_run = next(iter(runs.values()))
        if chosen_run:
            canonical_outcomes[att_id] = dict(chosen_run)

    # Step 4: Paired completeness
    # Requires identical run_id, identical profile_digest, full extent, neither refused
    paired_complete_set: set[str] = set()
    for att_id, att_outs in outcomes_by_attempt.items():
        runs = {}
        for o in att_outs:
            runs.setdefault(o.run_id, {})[o.arm_id] = o

        for r_id, arm_map in runs.items():
            if "A" in arm_map and "B" in arm_map:
                p_a = arm_map["A"]
                p_b = arm_map["B"]
                if (
                    p_a.run_id == p_b.run_id
                    and p_a.profile_digest == p_b.profile_digest
                    and p_a.collection_extent == "full"
                    and p_b.collection_extent == "full"
                    and p_a.refusal is None
                    and p_b.refusal is None
                ):
                    paired_complete_set.add(att_id)
                    break

    paired_complete = len(paired_complete_set)

    # Step 5: Per-arm analysis
    arm_analyses: dict[str, ArmAnalysis] = {}
    for arm_id in ("A", "B"):
        correct = 0
        wrong_enrolled = 0
        unknown_false_accept = 0
        review = 0
        unknown_term = 0
        timeout = 0
        invalid_input = 0
        error = 0
        cancelled = 0
        no_result = 0
        time_to_correct: list[float] = []
        time_to_wrong: list[float] = []
        nondecision: list[float] = []

        # Count refusals across ALL runs for this arm without collapsing
        refused = sum(
            1
            for o in outcomes
            if o.attempt_id in attempt_map
            and o.arm_id == arm_id
            and o.refusal is not None
        )

        for att_id, att in attempt_map.items():
            out = canonical_outcomes[att_id].get(arm_id)
            lbl = latest_labels.get(att_id)

            if out is None:
                no_result += 1
                continue

            if out.refusal is not None:
                refused += 1
                continue

            start_ns = traces[att_id].session_start_ns if att_id in traces else 0
            dec_time = out.decision_time_ns
            elapsed_ms = (
                round((dec_time - start_ns) / 1_000_000.0, 2)
                if dec_time is not None
                else None
            )

            term = out.terminal
            if term == SessionStatus.matched.value:
                if lbl is not None and lbl.kind == "enrolled":
                    if out.matched_identity == lbl.identity_id:
                        correct += 1
                        if elapsed_ms is not None:
                            time_to_correct.append(elapsed_ms)
                    else:
                        wrong_enrolled += 1
                        if elapsed_ms is not None:
                            time_to_wrong.append(elapsed_ms)
                elif lbl is not None and lbl.kind == "unenrolled":
                    unknown_false_accept += 1
                    if elapsed_ms is not None:
                        time_to_wrong.append(elapsed_ms)
            elif term == SessionStatus.timeout.value:
                timeout += 1
                if elapsed_ms is not None:
                    nondecision.append(elapsed_ms)
            elif term == SessionStatus.invalid_input.value:
                invalid_input += 1
                if elapsed_ms is not None:
                    nondecision.append(elapsed_ms)
            elif term == SessionStatus.review.value:
                review += 1
                if elapsed_ms is not None:
                    nondecision.append(elapsed_ms)
            elif term == SessionStatus.unknown.value:
                unknown_term += 1
                if elapsed_ms is not None:
                    nondecision.append(elapsed_ms)
            elif term == SessionStatus.cancelled.value:
                cancelled += 1
                if elapsed_ms is not None:
                    nondecision.append(elapsed_ms)
            elif term == SessionStatus.error.value:
                error += 1
                if elapsed_ms is not None:
                    nondecision.append(elapsed_ms)
            elif term == "refused":
                refused += 1

        enrolled_correct_rate = (
            (correct / truth_known_enrolled) if truth_known_enrolled > 0 else None
        )
        enrolled_wrong_rate = (
            (wrong_enrolled / truth_known_enrolled)
            if truth_known_enrolled > 0
            else None
        )
        unknown_fa_rate = (
            (unknown_false_accept / unknown) if unknown > 0 else None
        )

        # Conditional paired-complete enrolled counts
        cond_count = 0
        cond_correct = 0
        for att_id in paired_complete_set:
            lbl = latest_labels.get(att_id)
            if lbl is not None and lbl.kind == "enrolled":
                cond_count += 1
                out = canonical_outcomes[att_id].get(arm_id)
                if (
                    out is not None
                    and out.terminal == SessionStatus.matched.value
                    and out.matched_identity == lbl.identity_id
                ):
                    cond_correct += 1

        cond_rate = (cond_correct / cond_count) if cond_count > 0 else None

        arm_analyses[arm_id] = ArmAnalysis(
            arm_id=arm_id,
            correct=correct,
            wrong_enrolled=wrong_enrolled,
            unknown_false_accept=unknown_false_accept,
            review=review,
            unknown=unknown_term,
            timeout=timeout,
            invalid_input=invalid_input,
            error=error,
            cancelled=cancelled,
            refused=refused,
            no_result=no_result,
            enrolled_correct_rate=enrolled_correct_rate,
            enrolled_wrong_rate=enrolled_wrong_rate,
            unknown_fa_rate=unknown_fa_rate,
            conditional_paired_enrolled_correct=cond_correct,
            conditional_paired_enrolled_count=cond_count,
            conditional_paired_enrolled_correct_rate=cond_rate,
            time_to_correct_ms=tuple(time_to_correct),
            time_to_wrong_ms=tuple(time_to_wrong),
            nondecision_ms=tuple(nondecision),
        )

    # Step 6: Triggers detection
    triggers_dict: dict[str, list[str]] = {}
    for att_id, att in attempt_map.items():
        out_a = canonical_outcomes[att_id].get("A")
        out_b = canonical_outcomes[att_id].get("B")
        lbl = latest_labels.get(att_id)

        # T01: wrong enrolled match OR unenrolled false accept
        is_t01 = False
        for out in (out_a, out_b):
            if out is not None and out.terminal == SessionStatus.matched.value:
                if (
                    lbl is not None
                    and lbl.kind == "enrolled"
                    and out.matched_identity != lbl.identity_id
                ):
                    is_t01 = True
                elif lbl is not None and lbl.kind == "unenrolled":
                    is_t01 = True
        if is_t01:
            triggers_dict.setdefault(TRIGGER_T01, []).append(att_id)

        # T02: Live vs decision replay discrepancy
        if att_id in traces:
            live_res = traces[att_id].terminal_result
            if live_res is not None and out_b is not None and out_b.refusal is None:
                if (
                    live_res.status.value != out_b.terminal
                    or live_res.matched_identity != out_b.matched_identity
                ):
                    triggers_dict.setdefault(TRIGGER_T02, []).append(att_id)

        # T06: A correct, B unsuccessful
        if (
            lbl is not None
            and lbl.kind == "enrolled"
            and out_a is not None
            and out_a.terminal == SessionStatus.matched.value
            and out_a.matched_identity == lbl.identity_id
        ):
            b_success = (
                out_b is not None
                and out_b.terminal == SessionStatus.matched.value
                and out_b.matched_identity == lbl.identity_id
            )
            if not b_success:
                triggers_dict.setdefault(TRIGGER_T06, []).append(att_id)

        # T08: Low usable frames / quality rejection
        if (
            out_b is not None
            and out_b.terminal == SessionStatus.invalid_input.value
        ):
            triggers_dict.setdefault(TRIGGER_T08, []).append(att_id)

        # T10: Any enrolled not correctly matched by B (refusal excluded)
        if lbl is not None and lbl.kind == "enrolled":
            b_refused = (out_b is not None and out_b.refusal is not None) or any(
                o.arm_id == "B" and o.refusal is not None
                for o in outcomes_by_attempt.get(att_id, ())
            )
            if not b_refused:
                b_cor = (
                    out_b is not None
                    and out_b.terminal == SessionStatus.matched.value
                    and out_b.matched_identity == lbl.identity_id
                )
                if not b_cor:
                    triggers_dict.setdefault(TRIGGER_T10, []).append(att_id)

    triggers = {k: tuple(v) for k, v in triggers_dict.items()}
    hard_triggers_tripped = any(k in HARD_TRIGGERS for k in triggers)

    # Step 7: Failure layers and Case Summaries
    cases: list[CaseSummary] = []
    layer_breakdown: dict[str, int] = {}

    for att_id, att in attempt_map.items():
        lbl = latest_labels.get(att_id)
        out_a = canonical_outcomes[att_id].get("A")
        out_b = canonical_outcomes[att_id].get("B")
        trace = traces.get(att_id)

        layer = "none"
        detail: str | None = None

        if lbl is not None and lbl.kind == "enrolled":
            b_refused = (out_b is not None and out_b.refusal is not None) or any(
                o.arm_id == "B" and o.refusal is not None
                for o in outcomes_by_attempt.get(att_id, ())
            )
            if b_refused:
                layer = "refused"
                detail = (
                    out_b.refusal
                    if out_b and out_b.refusal
                    else next(
                        (
                            o.refusal
                            for o in outcomes_by_attempt.get(att_id, ())
                            if o.refusal
                        ),
                        "refused",
                    )
                )
                layer_breakdown[layer] = layer_breakdown.get(layer, 0) + 1
            else:
                b_cor = (
                    out_b is not None
                    and out_b.terminal == SessionStatus.matched.value
                    and out_b.matched_identity == lbl.identity_id
                )
                if not b_cor:
                    layer, detail = _classify_failure_layer(
                        att, lbl, trace, out_b, profile
                    )
                    layer_breakdown[layer] = (
                        layer_breakdown.get(layer, 0) + 1
                    )

        att_triggers = tuple(
            trig for trig, att_list in triggers.items() if att_id in att_list
        )

        if TRIGGER_T01 in att_triggers:
            action = (
                "立即隔離候選改善的放行；保留合規證據、建 incident"
            )
        elif TRIGGER_T06 in att_triggers:
            action = "升級有界 RCA 實驗"
        else:
            action = "單次觀察，等待同 signature 再現"

        cases.append(
            CaseSummary(
                attempt_id=att_id,
                participant_id=att.participant_id,
                visit_id=att.visit_id,
                truth_kind=lbl.kind if lbl else "unlabeled",
                truth_identity=lbl.identity_id if lbl else None,
                operational_status=att.operational_status,
                arm_a_terminal=out_a.terminal if out_a else None,
                arm_b_terminal=out_b.terminal if out_b else None,
                earliest_blocking_layer=layer,
                threshold_detail=detail,
                triggers=att_triggers,
                evidence_locator=f"{att_id}:{att.bundle_ref or 'no_bundle'}",
                recommended_action=action,
            )
        )

    return BatchAnalysis(
        experiment_id=experiment_id,
        attempted=attempted,
        labeled=labeled,
        truth_known_enrolled=truth_known_enrolled,
        unknown=unknown,
        uncertain=uncertain,
        unlabeled=unlabeled,
        paired_complete=paired_complete,
        operation_errors=operation_errors,
        participants_count=participants_count,
        visits_count=visits_count,
        arm_a=arm_analyses["A"],
        arm_b=arm_analyses["B"],
        triggers=triggers,
        hard_triggers_tripped=hard_triggers_tripped,
        layer_breakdown=layer_breakdown,
        cases=tuple(cases),
    )
