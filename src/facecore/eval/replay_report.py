"""Phase-1B replay report with conditional closeout (1B plan §11 Task 10).

Side-by-side table: Phase-1A frozen baseline vs Phase-1B adaptive bank
with exact denominators always; rate formatting only at N >= 30.

Conditional closeout: real SFace Pair-1 weights / multi-timestamp probes
are operator-dual-gated. While blocked, the report evaluates the
synthetic adversarial stream only, marks real replay
``blocked-with-reason``, leaves the model selection gate OPEN, and is
labeled ``partial governance validation`` — never Phase-1B completion.
``check_gate_open`` is the executable assertion of that invariant.

Redaction: the report carries counts and digests only — no raw
embeddings, face crops, or local paths (verified by test + the shared
``write_report`` guard at write time).
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReplayComparison:
    baseline_matched: int
    baseline_review: int
    baseline_unknown: int
    baseline_denominator: int
    adaptive_matched: int
    adaptive_review: int
    adaptive_unknown: int
    adaptive_denominator: int
    creations: int
    promotions: int
    evictions: int
    drift_exceeded: int


def _cell(count: int, denominator: int) -> str:
    if denominator >= 30:
        return f"{count}/{denominator} ({count / denominator:.1%})"
    return f"{count}/{denominator}"


def _row(
    outcome: str,
    base_count: int,
    base_denominator: int,
    adaptive_count: int,
    adaptive_denominator: int,
) -> str:
    base = _cell(base_count, base_denominator)
    adaptive = _cell(adaptive_count, adaptive_denominator)
    return f"| {outcome} | {base} | {adaptive} |"


def check_gate_open(report: str) -> None:
    """Fail unless an OPEN selection gate is declared for blocked replay."""
    if "blocked-with-reason" in report:
        if "selection gate: OPEN" not in report:
            raise AssertionError(
                "Model selection gate must remain OPEN when real replay"
                " is blocked"
            )


def build_replay_report(
    comparison: ReplayComparison,
    real_replay_status: str,
    replay_summary: Any | None = None,
    corpus_path: str | None = None,
) -> str:
    """Render the Phase-1B replay report body (redaction-safe text)."""
    lines = [
        "# Phase-1B replay report",
        "",
        "## Side-by-side comparison "
        "(Phase-1A frozen baseline vs Phase-1B adaptive bank)",
        "",
        "| outcome | baseline (1A frozen) | adaptive (1B bank) |",
        "| --- | --- | --- |",
        _row(
            "matched",
            comparison.baseline_matched,
            comparison.baseline_denominator,
            comparison.adaptive_matched,
            comparison.adaptive_denominator,
        ),
        _row(
            "review",
            comparison.baseline_review,
            comparison.baseline_denominator,
            comparison.adaptive_review,
            comparison.adaptive_denominator,
        ),
        _row(
            "unknown",
            comparison.baseline_unknown,
            comparison.baseline_denominator,
            comparison.adaptive_unknown,
            comparison.adaptive_denominator,
        ),
        "",
        "## Governance counters",
        "",
        f"- candidate creations: {comparison.creations}",
        f"- promotions: {comparison.promotions}",
        f"- evictions: {comparison.evictions}",
        f"- drift boundary exceeded: {comparison.drift_exceeded}",
        "",
    ]
    if replay_summary is not None:
        lines += [
            "## Synthetic replay execution",
            "",
            f"- {replay_summary.events_processed} events replayed",
            "- sequence: "
            + ",".join(str(s) for s in replay_summary.processed_sequence),
            "",
        ]
    if real_replay_status == "complete":
        lines += [
            "## Real replay (SFace Pair 1, consented P1 corpus)",
            "",
            "- status: complete",
            "",
        ]
    else:
        lines += [
            "## Real replay (SFace Pair 1, consented P1 corpus)",
            "",
            f"- status: blocked-with-reason: {real_replay_status}",
            "- selection gate: OPEN",
            "- label: partial governance validation (NOT Phase-1B completion)",
            "",
        ]
    _ = corpus_path
    return "\n".join(lines)
