"""Threshold-sweep operating table with exact denominators (Task 10).

Denominator honesty: non-target FA denominator is the probe count, never the
identity count. Below 30 a cell shows counts only (k/N); at or above 30 a
rate may appear alongside, never replacing k/N. Zero is 0/N, never a rate.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SweepRow:
    match_threshold: float
    margin_threshold: float
    matched: int
    review: int
    unknown: int
    false_accepts: int
    target_denom: int
    nontarget_denom: int


def format_cell(k: int, n: int) -> str:
    """Render k/N; a rate is appended only at denominator >= 30."""
    if n >= 30:
        return f"{k}/{n} ({k / n:.1%})"
    return f"{k}/{n}"


def sweep_thresholds(
    *,
    target_scores: list[float],
    target_margins: list[float | None],
    nontarget_scores: list[float],
    match_grid: list[float],
    margin_grid: list[float],
    review_threshold: float,
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for match_t in match_grid:
        for margin_t in margin_grid:
            matched = review = unknown = fa = 0
            for score, margin in zip(target_scores, target_margins, strict=True):
                if score >= match_t and (margin is None or margin >= margin_t):
                    matched += 1
                elif score >= match_t or score >= review_threshold:
                    review += 1
                else:
                    unknown += 1
            for score in nontarget_scores:
                if score >= match_t:
                    fa += 1
            rows.append(
                SweepRow(
                    match_threshold=match_t,
                    margin_threshold=margin_t,
                    matched=matched,
                    review=review,
                    unknown=unknown,
                    false_accepts=fa,
                    target_denom=len(target_scores),
                    nontarget_denom=len(nontarget_scores),
                )
            )
    return rows
