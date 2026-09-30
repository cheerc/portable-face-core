"""D6 A2: compare one candidate against the frozen corpus, honestly.

The comparison's difficulty is not arithmetic. It is that a number
printed next to the wrong label is indistinguishable from a number
printed next to the right one, so every decision this module makes is
about refusing rather than computing.

What it will not let happen:

- **Two models in one cosine.** Facenet512 and ArcFace are both 512-D,
  so a dimension check waves them through; the guard is on
  `model_version`, on both the gallery side and the probe side.
- **A smaller comparison called a comparison.** A gallery short of its
  declared identity count stops the run. Dropping the missing identity
  also drops the runner-up it was most likely to beat, so every
  false-accept count after it would improve for a reason unrelated to
  the model.
- **The control's thresholds on a candidate.** g3-v1's 0.363/0.3/0.10
  is SFace's calibration. A candidate explores its own (match, margin)
  grid and reports its own ranking; `review` is reported as *not
  configured* rather than fabricated, because no independent review
  source exists for a model the product has never run.
- **A winner.** The 13+30 photos are the evaluation set, so naming a
  best cell from this grid is tuning on the test data. Every cell
  carries `selected=False`, and `frontier_at_budget` reports the
  *description* of an envelope, never a choice.
- **An interpreter that was not declared.** ORT 1.30 on 3.13 is not a
  3.14 result. A DeepFace candidate that claims the product's 3.14, or
  omits the field, is refused before anything is measured.

Counting is D5's, not this module's: candidate rows are built by the
same `score_photo` and counted by the same `summarize`/`summarize_r`,
so the L/R definitions cannot drift between the control and the
candidates. What this module adds is the per-candidate gallery, the
frontier, and the refusals.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from facecore.contracts.policy import PolicyProfile
from facecore.eval.static_baseline import (
    BAND_MATCHED,
    SET_NONTARGET,
    SET_PROBE,
    PhotoResult,
    _band,
    _classify,
    summarize,
    summarize_r,
)
from facecore.live.frame_pipeline import ResearchGallery, ScoringContext

#: The plan's (match, margin) exploration grid, §4.2. Match runs -1..1
#: by 0.01 so the no-accept end of the space is in the grid rather than
#: being implied; margin is the plan's list, unscaled.
MATCH_GRID: tuple[float, ...] = tuple(round(-1.0 + 0.01 * i, 2) for i in range(201))
MARGIN_GRID: tuple[float, ...] = (
    0.0,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    1.0,
    2.0,
)

#: False-accept budgets the frontier is summarised at, §4.2.5. These
#: are the *observed* FA counts D5 measured on this corpus, so they
#: make the control and the candidates comparable at equal observation
#: rather than at equal threshold.
FA_BUDGETS: tuple[int, ...] = (0, 1, 2)

#: The product's own interpreter, from `requires-python`. A DeepFace
#: candidate cannot claim it: DeepFace 0.0.93 needs TensorFlow, which
#: does not support 3.14 in the pinned lock.
PRODUCT_REQUIRES_PYTHON = "3.14"


class CandidateContractError(ValueError):
    """A candidate cannot be measured under the protocol as declared."""


class GalleryIncomplete(CandidateContractError):
    """The candidate's gallery is not the frozen one — a stop, not a subset."""


class PhotoScorer(Protocol):
    """What a candidate must provide to be scored.

    Deliberately narrow: a candidate takes a context and a photo path
    and returns D5's own ``PhotoResult``. It never gets to reimplement
    the band rule, which is the one thing that must not vary between
    the control and the candidates.
    """

    def score(
        self, path: Path, *, set_name: str, truth_identity: str | None, sequence: int
    ) -> PhotoResult: ...


@dataclass(frozen=True)
class CandidateSpec:
    """Everything about a candidate that has to be written down.

    The provenance fields are not decoration. A comparison whose
    weight hash, input size, normalization and interpreter are not all
    on the page cannot be reproduced, and two of them (input size and
    normalization) change the numbers by construction.
    """

    candidate_id: str
    model_name: str
    model_version: str
    embedding_dim: int
    input_size: int
    normalization: str
    interpreter: str
    weight_sha256: str
    backend: str

    @property
    def is_control(self) -> bool:
        """The control is the product path, and is scored at its frozen point."""
        return self.backend == "product"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "embedding_dim": self.embedding_dim,
            "input_size": self.input_size,
            "normalization": self.normalization,
            "interpreter": self.interpreter,
            "weight_sha256": self.weight_sha256,
            "backend": self.backend,
        }


def require_declared_interpreter(
    spec: CandidateSpec, *, product_requires_python: str = PRODUCT_REQUIRES_PYTHON
) -> CandidateSpec:
    """Refuse a candidate whose interpreter is undeclared or misattributed.

    Two refusals, both about the same confusion:

    - an empty field, because a comparison that does not say which
      interpreter produced it cannot be compared to one that does;
    - a DeepFace candidate claiming the product's 3.14. The candidate
      and the control are measured on different interpreters by
      necessity, and filing them under one number is the specific error
      the plan forbids.
    """
    if not spec.interpreter.strip():
        raise CandidateContractError(
            f"candidate {spec.candidate_id!r} declares no interpreter; the "
            "report cannot attribute its numbers to a runtime"
        )
    if spec.is_control:
        return spec
    if spec.interpreter.strip() == product_requires_python.strip():
        raise CandidateContractError(
            f"candidate {spec.candidate_id!r} runs on backend "
            f"{spec.backend!r} but declares interpreter "
            f"{spec.interpreter!r}, the product's own. A backend candidate "
            "cannot produce numbers on the product interpreter; state the "
            "interpreter it actually ran on, and treat its results as "
            "unrelated to the control's runtime."
        )
    return spec


def build_candidate_gallery(
    *,
    identities: dict[str, np.ndarray],
    model_version: str,
    expected_identities: int,
    generation: str = "gen-d6",
) -> ResearchGallery:
    """Build one candidate's own gallery, or refuse.

    The count is exact in both directions. Too few means the enrollment
    is not the frozen one; too many means the directory holds more than
    the freeze, and taking a subset by sort order would be a rule nobody
    could re-derive.
    """
    actual = len(identities)
    if actual != expected_identities:
        raise GalleryIncomplete(
            f"candidate gallery has {actual} identities but the freeze "
            f"declares {expected_identities}; a comparison needs the same "
            "identities as the control, since a missing one is also the "
            "runner-up most likely to be beaten"
        )
    # The digest is computed with `build_gallery_from_folder`'s own
    # recipe (sorted identity name, then vector bytes) rather than by
    # calling `ResearchGallery.compute_digest`, which hashes the
    # float32 copies the constructor makes while that function hashes
    # the float32 arrays it is handed. Same rule, and the two agree —
    # but agreeing by construction is the property under test, so the
    # rule is written out rather than delegated to a method whose
    # receiver this module does not own.
    hasher = hashlib.sha256()
    for ident in sorted(identities):
        hasher.update(ident.encode("utf-8"))
        hasher.update(np.asarray(identities[ident], dtype=np.float32).tobytes())
    return ResearchGallery(
        embeddings={k: np.asarray(v, dtype=np.float32) for k, v in identities.items()},
        model_version=model_version,
        generation=generation,
        digest=hasher.hexdigest(),
    )


def build_candidate_context(
    *,
    gallery: ResearchGallery,
    model_version: str,
    detector: Any,
    embedder: Any,
    profile: PolicyProfile,
) -> ScoringContext:
    """Assemble the candidate's scoring context, refusing a model mismatch.

    ``ScoringContext.__post_init__`` already compares the context's
    version against the gallery's and the embedder's, and raising here
    rather than catching means the refusal is the same one the product
    path uses. The explicit check is here so the *reason* names the
    comparison, which is the case that actually occurs: a new probe
    scored against the control's gallery.
    """
    if gallery.model_version != model_version:
        raise GalleryIncomplete(
            f"gallery model {gallery.model_version!r} does not match "
            f"candidate model {model_version!r}; vectors from two models "
            "must never be compared, and equal dimension is not evidence "
            "that they are the same model"
        )
    return ScoringContext(
        gallery=gallery,
        model_version=model_version,
        policy=profile,
        detector=detector,
        embedder=embedder,
    )


@dataclass(frozen=True)
class FrontierCell:
    """One (match, margin) point under a candidate's own thresholds.

    ``selected`` is always False and the field exists so that a reader
    of the CSV cannot have to guess whether some later step promoted
    one. The promotion is not implemented; the field is the record that
    it was not done.
    """

    match_threshold: float
    margin_threshold: float
    correct_accepts: int
    wrong_identities: int
    false_accepts: int
    no_accept: bool
    review_configured: bool
    total: int
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_threshold": self.match_threshold,
            "margin_threshold": self.margin_threshold,
            "correct_accepts": self.correct_accepts,
            "wrong_identities": self.wrong_identities,
            "false_accepts": self.false_accepts,
            "no_accept": self.no_accept,
            "review_configured": self.review_configured,
            "total": self.total,
            "selected": self.selected,
        }


def _frontier_for(
    rows: Sequence[PhotoResult],
    *,
    match_grid: Sequence[float],
    margin_grid: Sequence[float],
    review_threshold: float | None,
) -> list[FrontierCell]:
    """Sweep the grid over rows that carry a score.

    A row with no score — quality-rejected on L, or unprocessable — is
    skipped, exactly as D5's sweep skips it. It stays in ``total``,
    because the denominator is the photo count, not the scorable count.
    """
    cells: list[FrontierCell] = []
    for match_t in match_grid:
        for margin_t in margin_grid:
            correct = wrong = fa = 0
            for row in rows:
                s1 = row.l_top1_score
                if s1 is None:
                    continue
                # `review` is a policy concept, not a geometric one. When
                # the candidate has no review threshold, a cell splits
                # only into matched and not-matched, so the sweep uses a
                # sentinel that no score can reach.
                review = (
                    review_threshold if review_threshold is not None else float("inf")
                )
                band = _band(
                    s1,
                    row.l_top2_score,
                    match_threshold=match_t,
                    review_threshold=review,
                    margin_threshold=margin_t,
                )
                if band != BAND_MATCHED:
                    continue
                if row.truth_identity is None:
                    fa += 1
                elif row.l_top1 == row.truth_identity:
                    correct += 1
                else:
                    wrong += 1
            cells.append(
                FrontierCell(
                    match_threshold=match_t,
                    margin_threshold=margin_t,
                    correct_accepts=correct,
                    wrong_identities=wrong,
                    false_accepts=fa,
                    no_accept=correct == 0 and wrong == 0 and fa == 0,
                    review_configured=review_threshold is not None,
                    total=len(rows),
                )
            )
    return cells


def frontier_at_budget(
    cells: Sequence[FrontierCell], *, fa_budget: int
) -> FrontierCell | None:
    """The best cell at or under a false-accept budget, or None.

    "Best" is ordered by correct accepts, then by *fewer* wrong
    identities, then by a looser match threshold. Wrong identity is
    ranked above threshold tightness on purpose: a cell that accepts
    more probes by naming the wrong person is worse, not better, and a
    tie broken on score alone would prefer it.

    ``None`` is a real answer. When no cell meets the budget, the
    honest result is that the budget is unreachable at this corpus, and
    returning the least-bad cell would answer a different question
    while hiding it.

    The returned cell is the one that already exists, ``selected`` and
    all — this function describes, it does not choose.
    """
    eligible = [c for c in cells if c.false_accepts <= fa_budget]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda c: (-c.correct_accepts, c.wrong_identities, -c.match_threshold),
    )


@dataclass(frozen=True)
class ComparisonOutcome:
    """One set's counts, reusing D5's definitions unchanged."""

    set_name: str
    total: int
    counts: dict[str, int]
    l_top1_correct: int
    r_top1_correct: int
    score_ranges: dict[str, dict[str, float | None]] = field(default_factory=dict)

    @classmethod
    def from_rows(
        cls, rows: Sequence[PhotoResult], *, set_name: str
    ) -> ComparisonOutcome:
        # Both branches read the same `BaselineSummary` fields. They look
        # duplicated only because D5's two `summarize` bodies spell out
        # each set's zeroed classes; collapsing them here would mean
        # re-deciding which class is meaningless per set, which is D5's
        # call and not this module's.
        if set_name in (SET_PROBE, SET_NONTARGET):
            summary = summarize(list(rows), set_name=set_name)
            counts = {
                "correct_accepts": summary.correct_accepts,
                "wrong_identities": summary.wrong_identities,
                "review": summary.review,
                "unknown_rejects": summary.unknown_rejects,
                "quality_rejected": summary.quality_rejected,
                "unprocessable": summary.unprocessable,
                "false_accepts": summary.false_accepts,
                "correct_rejects": summary.correct_rejects,
            }
        else:
            raise ValueError(f"unknown set_name {set_name!r}")
        r = summarize_r(list(rows), set_name=set_name)
        if r["unprocessable"] == len(rows) and rows:
            raise CandidateContractError(
                f"every one of the {len(rows)} {set_name} photos was "
                "unprocessable; nothing was scorable, so the run measures "
                "a broken setup rather than the candidate"
            )
        # `summarize_r` is typed `dict[str, object]` because it is a
        # report-shaped side table. Narrowing through `isinstance` is
        # what keeps a wrong type a loud error rather than an `int()`
        # that accepts a string and turns it into a plausible count.
        raw_top1 = r["r_top1_correct"]
        if not isinstance(raw_top1, int):
            raise CandidateContractError(
                f"R-branch top1 count came back as {type(raw_top1).__name__}, "
                "not an int; the count would be a guess"
            )
        return cls(
            set_name=set_name,
            total=len(rows),
            counts=counts,
            l_top1_correct=summary.l_top1_correct,
            r_top1_correct=raw_top1,
            score_ranges=summary.score_ranges,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "set_name": self.set_name,
            "total": self.total,
            **self.counts,
            "l_top1_correct": self.l_top1_correct,
            "r_top1_correct": self.r_top1_correct,
            "score_ranges": self.score_ranges,
        }


def classify_branch(result: PhotoResult, *, branch: str) -> str:
    """Bucket one row on one branch, delegating to D5's single rule.

    This is a named re-export, not a second implementation. D5 fixed
    these eight classes and the two branches' subtle difference — a
    quality-rejected row has no L identity but a populated R one — so a
    candidate that classified rows any other way could score a
    correctly-identified probe as a wrong identity.
    """
    return _classify(result, branch=branch)


@dataclass(frozen=True)
class CandidateComparison:
    """One candidate's full result: counts, frontier, and provenance."""

    spec: CandidateSpec
    probe: ComparisonOutcome
    nontarget: ComparisonOutcome
    frontier: list[FrontierCell]
    frontier_summary: list[dict[str, Any]]
    review_threshold: float | None
    operating_point: dict[str, float] | None

    @property
    def review_configured(self) -> bool:
        return self.review_threshold is not None

    @classmethod
    def build(
        cls,
        *,
        spec: CandidateSpec,
        probe_rows: Sequence[PhotoResult],
        nontarget_rows: Sequence[PhotoResult],
        match_grid: Sequence[float],
        margin_grid: Sequence[float],
        review_threshold: float | None,
        product_requires_python: str = PRODUCT_REQUIRES_PYTHON,
    ) -> CandidateComparison:
        require_declared_interpreter(
            spec, product_requires_python=product_requires_python
        )
        probe = ComparisonOutcome.from_rows(probe_rows, set_name=SET_PROBE)
        nontarget = ComparisonOutcome.from_rows(nontarget_rows, set_name=SET_NONTARGET)

        if spec.is_control:
            # The control has a frozen operating point. Exploring a grid
            # for it would invent alternatives to a point the App
            # actually runs, which is the same error as applying SFace's
            # thresholds to somebody else's model.
            return cls(
                spec=spec,
                probe=probe,
                nontarget=nontarget,
                frontier=[],
                frontier_summary=[],
                review_threshold=review_threshold,
                operating_point=None,
            )

        cells = _frontier_for(
            probe_rows,
            match_grid=match_grid,
            margin_grid=margin_grid,
            review_threshold=None,
        ) + _frontier_for(
            nontarget_rows,
            match_grid=match_grid,
            margin_grid=margin_grid,
            review_threshold=None,
        )
        # One sweep per set, reused across the three budgets. The grid is
        # 201 x 10 and the corpus is 43 rows, so recomputing it per
        # budget is 6x the work for a result that cannot differ: a cell
        # does not know which budget is being read.
        summary_rows: list[dict[str, Any]] = []
        probe_cells = cells[: len(cells) // 2]
        nontarget_cells = cells[len(cells) // 2 :]
        for budget in FA_BUDGETS:
            probe_cell = frontier_at_budget(probe_cells, fa_budget=budget)
            fa_cell = frontier_at_budget(nontarget_cells, fa_budget=budget)
            summary_rows.append(
                {
                    "fa_budget": budget,
                    "probe": probe_cell.to_dict() if probe_cell else None,
                    "nontarget": fa_cell.to_dict() if fa_cell else None,
                    "selected": False,
                }
            )
        return cls(
            spec=spec,
            probe=probe,
            nontarget=nontarget,
            frontier=cells,
            frontier_summary=summary_rows,
            review_threshold=review_threshold,
            operating_point=None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "probe": self.probe.to_dict(),
            "nontarget": self.nontarget.to_dict(),
            "review_configured": self.review_configured,
            "review_threshold": self.review_threshold,
            "operating_point": self.operating_point,
            "frontier_summary": self.frontier_summary,
            "frontier_cells": len(self.frontier),
        }


def compare_candidate(
    *,
    spec: CandidateSpec,
    scorer: PhotoScorer,
    probes: Sequence[Path],
    nontarget: Sequence[Path],
    expected_probes: int,
    expected_nontarget: int,
    probe_truth: str | None,
    match_grid: Sequence[float] = MATCH_GRID,
    margin_grid: Sequence[float] = MARGIN_GRID,
    review_threshold: float | None = None,
    product_requires_python: str = PRODUCT_REQUIRES_PYTHON,
) -> CandidateComparison:
    """Score one candidate over the frozen corpus and assemble its result.

    The input counts are checked *before* anything is scored, so a wrong
    corpus is a refusal rather than a comparison whose ratios are all
    quietly computed against the wrong denominator.
    """
    require_declared_interpreter(spec, product_requires_python=product_requires_python)
    if len(probes) != expected_probes:
        raise CandidateContractError(
            f"candidate {spec.candidate_id!r} was given {len(probes)} probe "
            f"photos but the freeze declares {expected_probes}; every count "
            "would be reported against a denominator the reader cannot see"
        )
    if len(nontarget) != expected_nontarget:
        raise CandidateContractError(
            f"candidate {spec.candidate_id!r} was given {len(nontarget)} "
            f"non-target photos but the freeze declares {expected_nontarget}"
        )
    probe_rows = [
        scorer.score(p, set_name=SET_PROBE, truth_identity=probe_truth, sequence=i)
        for i, p in enumerate(probes, start=1)
    ]
    nontarget_rows = [
        scorer.score(p, set_name=SET_NONTARGET, truth_identity=None, sequence=i)
        for i, p in enumerate(nontarget, start=1)
    ]
    return CandidateComparison.build(
        spec=spec,
        probe_rows=probe_rows,
        nontarget_rows=nontarget_rows,
        match_grid=match_grid,
        margin_grid=margin_grid,
        review_threshold=review_threshold,
        product_requires_python=product_requires_python,
    )


__all__ = [
    "FA_BUDGETS",
    "MARGIN_GRID",
    "MATCH_GRID",
    "PRODUCT_REQUIRES_PYTHON",
    "CandidateComparison",
    "CandidateContractError",
    "CandidateSpec",
    "ComparisonOutcome",
    "FrontierCell",
    "GalleryIncomplete",
    "PhotoScorer",
    "build_candidate_context",
    "build_candidate_gallery",
    "classify_branch",
    "compare_candidate",
    "frontier_at_budget",
    "require_declared_interpreter",
]
