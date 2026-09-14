"""Real non-target 30 FA evaluation + sweep (M1 selection gate).

Implements the acceptance SSOT commander real-photo run
(/tmp/face-accept/fa_real.py + fa_real.log @ detector gate 0.8,
repo-external, read-only) inside the bakeoff evaluation layer:

- ``load_real_nontarget_vectors``: embed the operator-provided corpus
  (default repo-external ``non-target/nontarget-01..30.jpg``) with the same
  decode -> detect(single face @ gate) -> align -> embed protocol as the
  SSOT. A missing corpus dir, missing files, or unusable photos yield a
  fail-clear ``skipped`` outcome — never an exception, never a hardcoded
  in-repo path.
- ``real_fa_rows``: score each usable non-target vector against the gallery
  through ``bakeoff.run_candidate`` (raw ``ProbeOutcome``) and apply the
  SSOT F1 FA rule (score >= FA_MATCH AND margin >= MARGIN_FIREWALL).
- ``real_fa_sweep_table``: match-threshold sweep with the margin firewall
  on BOTH arms, reproducing the fa_real.py F2 counting rule exactly.
- ``render_real_fa_section``: report section with exact N=30 denominators
  written out (counts only, never a bare rate).
- ``real_fa_grid_table``: match x margin 2-D operating table (M5) with the
  margin firewall swept on BOTH arms; the margin=0.10 row reproduces the
  R2 sweep exactly. ``render_real_fa_grid_section`` renders it in the
  selection-skeleton §1.1 line shape.

Thresholds are swept, never chosen here; frozen operating defaults
(detector gate 0.9, margin/match waterlines) are untouched. Real photos,
embeddings, and databases never enter Git (product hard boundary).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from facecore.eval.bakeoff import run_candidate
from facecore.repository.base import Repository

if TYPE_CHECKING:
    from facecore.eval.fa_matrix import TargetProbeScore

#: FA match waterline shared with the commander SSOT run (F1 rule).
FA_MATCH = 0.45

#: Margin firewall shared with the commander SSOT runs.
MARGIN_FIREWALL = 0.1

#: Exact SSOT corpus size; denominators stay pinned to this count.
REAL_NONTARGET_COUNT = 30

#: SSOT F2 sweep grid (fa_real.log verbatim).
REAL_MATCH_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

#: M5 margin dimension: representative檔 covering the 0.10 firewall line.
REAL_MARGIN_GRID = [0.05, 0.10, 0.15, 0.20]

#: Default corpus location: repo-external, never committed to Git.
DEFAULT_NONTARGET_DIR = Path("/Users/cheerc/Downloads/face_sample/non-target")

#: Detector gate used by the commander SSOT run (informational: the
#: evaluation layer scores vectors; gating happens at embed time).
SSOT_DETECTOR_GATE = 0.8


@dataclass(frozen=True)
class RealFaRow:
    probe_name: str
    top_identity: str | None
    top_score: float | None
    margin: float | None
    is_false_accept: bool


@dataclass(frozen=True)
class RealFaSweepRow:
    match_threshold: float
    target_hits: int
    target_denom: int
    real_fa: int
    real_denom: int


@dataclass(frozen=True)
class ProbeTargetScore:
    """Target-arm score of one probe: person-23's cut vs the full gallery."""

    person23_score: float
    top_is_person23: bool
    margin: float | None


def stable_target_scores(
    scored: list[ProbeTargetScore],
) -> list["TargetProbeScore"]:
    """Adapt CLI-scored probes to the sweep's target arm (order-preserving)."""
    from facecore.eval.fa_matrix import TargetProbeScore

    return [
        TargetProbeScore(
            probe_index=index,
            person23_score=entry.person23_score,
            top_is_person23=entry.top_is_person23,
            margin=entry.margin,
        )
        for index, entry in enumerate(scored)
    ]


@dataclass(frozen=True)
class CorpusLoadOutcome:
    files: list[Path]
    skipped: bool
    reason: str


def load_real_nontarget_vectors(
    corpus_dir: Path = DEFAULT_NONTARGET_DIR,
) -> CorpusLoadOutcome:
    """Locate the 30 real non-target photos; fail-clear skip when absent.

    Only locates files — embedding stays with the caller/commander
    protocol so this layer never couples to model artifacts.
    """
    expected = [corpus_dir / f"nontarget-{i:02d}.jpg" for i in range(1, 31)]
    if not corpus_dir.is_dir():
        return CorpusLoadOutcome(
            files=[],
            skipped=True,
            # No absolute path here: the reason is rendered into the report,
            # whose redaction guard rejects absolute local paths. The CLI
            # logs the concrete dir to stderr separately.
            reason="real non-target corpus dir missing "
            "(repo-external SSOT dir; skipped)",
        )
    missing = [path for path in expected if not path.is_file()]
    if missing:
        return CorpusLoadOutcome(
            files=[],
            skipped=True,
            reason=(
                f"real non-target corpus incomplete: "
                f"{len(missing)}/30 missing, e.g. {missing[0].name} (skipped)"
            ),
        )
    return CorpusLoadOutcome(files=expected, skipped=False, reason="")


def real_fa_rows(
    gallery: dict[str, np.ndarray],
    probes: list[np.ndarray],
    repository: Repository,
    model_version: str,
    *,
    names: list[str],
) -> list[RealFaRow]:
    """Score usable non-target probes; apply the SSOT F1 FA rule."""
    if len(probes) != len(names):
        raise ValueError("probes and names must pair one-to-one")
    outcomes = run_candidate(gallery, probes, repository, model_version)
    rows: list[RealFaRow] = []
    for name, outcome in zip(names, outcomes, strict=True):
        if outcome.top_identity is None:
            raise ValueError(f"empty gallery for real non-target {name}")
        score = outcome.top_score if outcome.top_score is not None else -1.0
        margin = outcome.margin
        admitted = (
            score >= FA_MATCH
            and margin is not None
            and margin >= MARGIN_FIREWALL
        )
        rows.append(
            RealFaRow(
                probe_name=name,
                top_identity=outcome.top_identity,
                top_score=outcome.top_score,
                margin=outcome.margin,
                is_false_accept=admitted,
            )
        )
    return rows


def real_fa_sweep_table(
    target: list["TargetProbeScore"],
    real_rows: list[RealFaRow],
    match_grid: list[float],
) -> list[RealFaSweepRow]:
    """Margin-firewalled sweep reproducing the fa_real.py F2 rule.

    Target hit: person-23 top-1 AND person-23 score >= match AND
    margin >= MARGIN_FIREWALL. Real FA: top-1 score >= match AND
    margin >= MARGIN_FIREWALL. A ``None`` margin (person-23 not top-1
    or single-member gallery) never counts on either arm.
    """
    rows: list[RealFaSweepRow] = []
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
            for row in real_rows
            if row.top_score is not None
            and row.top_score >= match_t
            and row.margin is not None
            and row.margin >= MARGIN_FIREWALL
        )
        rows.append(
            RealFaSweepRow(
                match_threshold=match_t,
                target_hits=hits,
                target_denom=len(target),
                real_fa=fa,
                real_denom=len(real_rows),
            )
        )
    return rows


@dataclass(frozen=True)
class RealFaGridRow:
    match_threshold: float
    margin_threshold: float
    target_hits: int
    target_denom: int
    real_fa: int
    real_denom: int


def real_fa_grid_table(
    target: list["TargetProbeScore"],
    real_rows: list[RealFaRow],
    match_grid: list[float],
    margin_grid: list[float],
) -> list[RealFaGridRow]:
    """Match x margin operating table (M5): same rule on both arms.

    Target hit: person-23 top-1 AND person-23 score >= match AND
    margin >= margin-cell. Real FA: top-1 score >= match AND
    margin >= margin-cell. A ``None`` margin never counts on either arm
    (same semantics as ``real_fa_sweep_table``). Thresholds are swept,
    never chosen here.
    """
    rows: list[RealFaGridRow] = []
    for match_t in match_grid:
        for margin_t in margin_grid:
            hits = sum(
                1
                for entry in target
                if entry.top_is_person23
                and entry.person23_score >= match_t
                and entry.margin is not None
                and entry.margin >= margin_t
            )
            fa = sum(
                1
                for row in real_rows
                if row.top_score is not None
                and row.top_score >= match_t
                and row.margin is not None
                and row.margin >= margin_t
            )
            rows.append(
                RealFaGridRow(
                    match_threshold=match_t,
                    margin_threshold=margin_t,
                    target_hits=hits,
                    target_denom=len(target),
                    real_fa=fa,
                    real_denom=len(real_rows),
                )
            )
    return rows


def render_real_fa_grid_section(grid_rows: list[RealFaGridRow]) -> str:
    """Render R3 in the selection-skeleton §1.1 line shape (counts only)."""
    lines = [
        "## R3. real-data match x margin operating table "
        "(target arm vs real non-target FA; N exact, counts only)",
        "",
    ]
    for cell in grid_rows:
        lines.append(
            f"- match>={cell.match_threshold:.2f} "
            f"margin>={cell.margin_threshold:.2f}: "
            f"target hits {cell.target_hits}/{cell.target_denom}, "
            f"real FA {cell.real_fa}/{cell.real_denom}"
        )
    return "\n".join(lines) + "\n"


def render_real_fa_section(
    rows: list[RealFaRow],
    sweep_rows: list[RealFaSweepRow],
    *,
    usable: int,
    total: int,
) -> str:
    """Render report section: per-probe detail (N=30) + sweep table."""
    lines = [
        "## R1. real non-target 30 FA: per-probe top-1 + margin "
        f"(usable {usable}/{total}; N=30 exact)",
        "",
        "| probe | top-1 | score | margin | FA |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        score = f"{row.top_score:.4f}" if row.top_score is not None else "n/a"
        margin = f"{row.margin:.4f}" if row.margin is not None else "n/a"
        flag = "FA" if row.is_false_accept else ""
        lines.append(
            f"| {row.probe_name} | {row.top_identity} | {score} | {margin} | {flag} |"
        )
    lines += [
        "",
        "## R2. match-threshold sweep: target arm vs real non-target FA "
        f"(margin firewall >= {MARGIN_FIREWALL:.2f} on both arms; "
        "N=30 exact, counts only)",
        "",
        "| match>= | target hits | real non-target FA |",
        "| --- | --- | --- |",
    ]
    for sweep in sweep_rows:
        lines.append(
            f"| {sweep.match_threshold:.2f} | "
            f"{sweep.target_hits}/{sweep.target_denom} | "
            f"{sweep.real_fa}/{sweep.real_denom} |"
        )
    return "\n".join(lines) + "\n"


def render_real_fa_skipped(reason: str) -> str:
    """Render the fail-clear section when the real corpus is unavailable."""
    return (
        "## R1. real non-target 30 FA: skipped "
        "(corpus unavailable; N=30 exact)\n"
        "\n"
        f"- {reason}\n"
        "- synthetic C1/C2/C3 arms below are unaffected.\n"
    )
