"""Cross-identity FA matrix: leave-one-out + minus-23 + margin-gated sweep.

Implements acceptance SSOT commander real-photo run
(/tmp/face-accept/fa_matrix.py + fa_matrix.log @ detector gate 0.8,
repo-external, read-only) inside the bakeoff evaluation layer:

- ``leave_one_out``: each enrolled identity's own template scored against
  the gallery WITHOUT itself (fa_matrix.py C1). A high top-1 score here is
  an FA risk: the identity would be accepted as somebody else.
- ``minus_23``: person-23 probes scored against the gallery WITHOUT
  person-23 (fa_matrix.py C2). Measures whether an unenrolled person is
  falsely accepted as an enrolled identity.
- ``target_person23_scores``: person-23 probe scores against the FULL
  gallery (fa_matrix.py C3 target arm).
- ``fa_sweep_table``: match-threshold sweep with the margin firewall on
  BOTH arms (``score >= match`` AND ``margin >= 0.1``), reproducing the
  fa_matrix.py C3 counting rule exactly.

All scoring goes through ``bakeoff.run_candidate`` (raw ``ProbeOutcome``:
top identity / top score / runner-up margin) — no parallel scoring path.
Thresholds are swept, never chosen here; frozen operating defaults
(detector gate 0.9, margin/match waterlines) are untouched.
"""

from dataclasses import dataclass

import numpy as np

from facecore.eval.bakeoff import run_candidate
from facecore.policy.identify import cosine_score
from facecore.repository.base import Repository

#: Margin firewall shared with the commander SSOT run: a top-1 score only
#: counts as a (false) accept when the top-1/top-2 margin reaches this.
MARGIN_FIREWALL = 0.1


@dataclass(frozen=True)
class LooRow:
    true_id: str
    top_identity: str | None
    top_score: float | None
    margin: float | None


@dataclass(frozen=True)
class Minus23Row:
    probe_index: int
    top_identity: str | None
    top_score: float | None
    margin: float | None


@dataclass(frozen=True)
class TargetProbeScore:
    probe_index: int
    person23_score: float
    top_is_person23: bool
    margin: float | None


@dataclass(frozen=True)
class FaSweepRow:
    match_threshold: float
    target_hits: int
    target_denom: int
    nontarget_fa: int
    nontarget_denom: int


def _require_outcome(outcome_top: str | None, context: str) -> None:
    if outcome_top is None:
        raise ValueError(f"empty gallery for {context}: no identity scored")


def leave_one_out(
    gallery: dict[str, np.ndarray],
    repository: Repository,
    model_version: str,
) -> list[LooRow]:
    """Score each member's own template against the gallery without itself."""
    rows: list[LooRow] = []
    for true_id, probe in gallery.items():
        others = {ident: vec for ident, vec in gallery.items() if ident != true_id}
        outcome = run_candidate(others, [probe], repository, model_version)[0]
        _require_outcome(outcome.top_identity, f"leave-one-out {true_id}")
        rows.append(
            LooRow(
                true_id=true_id,
                top_identity=outcome.top_identity,
                top_score=outcome.top_score,
                margin=outcome.margin,
            )
        )
    return rows


def minus_23(
    probes: list[np.ndarray],
    gallery_without_23: dict[str, np.ndarray],
    repository: Repository,
    model_version: str,
) -> list[Minus23Row]:
    """Score person-23 probes against the gallery without person-23."""
    if "person-23" in gallery_without_23:
        raise ValueError("minus_23 gallery must exclude person-23")
    outcomes = run_candidate(gallery_without_23, probes, repository, model_version)
    rows: list[Minus23Row] = []
    for outcome in outcomes:
        _require_outcome(outcome.top_identity, f"minus-23 probe {outcome.probe_index}")
        rows.append(
            Minus23Row(
                probe_index=outcome.probe_index,
                top_identity=outcome.top_identity,
                top_score=outcome.top_score,
                margin=outcome.margin,
            )
        )
    return rows


def target_person23_scores(
    probes: list[np.ndarray],
    full_gallery: dict[str, np.ndarray],
    repository: Repository,
    model_version: str,
) -> list[TargetProbeScore]:
    """Score person-23 probes vs the full gallery; report person-23's cut."""
    if "person-23" not in full_gallery:
        raise ValueError("target arm requires person-23 in the gallery")
    outcomes = run_candidate(full_gallery, probes, repository, model_version)
    scores: list[TargetProbeScore] = []
    for probe, outcome in zip(probes, outcomes, strict=True):
        _require_outcome(outcome.top_identity, f"target probe {outcome.probe_index}")
        top_is_23 = outcome.top_identity == "person-23"
        scores.append(
            TargetProbeScore(
                probe_index=outcome.probe_index,
                person23_score=cosine_score(probe, full_gallery["person-23"]),
                top_is_person23=top_is_23,
                margin=outcome.margin if top_is_23 else None,
            )
        )
    return scores


def fa_sweep_table(
    target: list[TargetProbeScore],
    nontarget_scores: list[float],
    nontarget_margins: list[float | None],
    match_grid: list[float],
) -> list[FaSweepRow]:
    """Margin-firewalled sweep reproducing the fa_matrix.py C3 rule.

    Target hit: person-23 top-1 AND person-23 score >= match AND
    margin >= MARGIN_FIREWALL. Non-target FA: top-1 score >= match AND
    margin >= MARGIN_FIREWALL. A ``None`` margin (single-member gallery
    or person-23 not top-1) never counts on either arm.
    """
    if len(nontarget_scores) != len(nontarget_margins):
        raise ValueError("nontarget scores and margins must pair one-to-one")
    rows: list[FaSweepRow] = []
    for match_t in match_grid:
        hits = sum(
            1
            for entry in target
            if entry.top_is_person23
            and entry.person23_score >= match_t
            and entry.margin is not None
            and entry.margin >= MARGIN_FIREWALL
        )
        fa = sum(
            1
            for score, margin in zip(nontarget_scores, nontarget_margins, strict=True)
            if score >= match_t and margin is not None and margin >= MARGIN_FIREWALL
        )
        rows.append(
            FaSweepRow(
                match_threshold=match_t,
                target_hits=hits,
                target_denom=len(target),
                nontarget_fa=fa,
                nontarget_denom=len(nontarget_scores),
            )
        )
    return rows


def format_fa_cell(count: int, denominator: int) -> str:
    """Render k/N; a rate is appended only at denominator >= 30."""
    if denominator >= 30:
        return f"{count}/{denominator} ({count / denominator:.1%})"
    return f"{count}/{denominator}"


def render_fa_section(
    loo_rows: list[LooRow],
    minus23_rows: list[Minus23Row],
    sweep_rows: list[FaSweepRow],
) -> str:
    """Render report section C1/C2/C3 with exact denominators (counts only)."""
    lines = [
        "## C1. leave-one-out: enrollee vs gallery without self (N=23)",
        "",
        "| identity | top-1 | score | margin |",
        "| --- | --- | --- | --- |",
    ]
    for loo in loo_rows:
        score = f"{loo.top_score:.4f}" if loo.top_score is not None else "n/a"
        margin = f"{loo.margin:.4f}" if loo.margin is not None else "n/a"
        lines.append(f"| {loo.true_id} | {loo.top_identity} | {score} | {margin} |")
    lines += [
        "",
        "## C2. person-23 probes vs gallery without person-23 (N=13)",
        "",
        "| probe | top-1 | score | margin |",
        "| --- | --- | --- | --- |",
    ]
    for probe_row in minus23_rows:
        score = (
            f"{probe_row.top_score:.4f}" if probe_row.top_score is not None else "n/a"
        )
        margin = f"{probe_row.margin:.4f}" if probe_row.margin is not None else "n/a"
        identity = probe_row.top_identity
        lines.append(
            f"| probe-{probe_row.probe_index:02d} | {identity} | {score} | {margin} |"
        )
    lines += [
        "",
        "## C3. match-threshold sweep (margin firewall >= 0.10 on both arms)",
        "",
        "| match>= | target hits | non-target FA |",
        "| --- | --- | --- |",
    ]
    for sweep in sweep_rows:
        hits = format_fa_cell(sweep.target_hits, sweep.target_denom)
        fa = format_fa_cell(sweep.nontarget_fa, sweep.nontarget_denom)
        lines.append(f"| {sweep.match_threshold:.2f} | {hits} | {fa} |")
    return "\n".join(lines) + "\n"
