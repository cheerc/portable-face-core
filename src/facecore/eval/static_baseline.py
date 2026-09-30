"""D5 static baseline harness — score the current pipeline on still photos.

Plan: ``docs/plans/2026-09-30-d5-static-baseline-plan.md`` (Task 1).

The point of this harness is to measure **the pipeline the App already
runs**, not a second scorer bolted alongside it. So the L branch calls
``facecore.live.frame_pipeline.score_frame`` — the same function
``cmd_live`` calls on every camera frame — and derives the band with the
same rule live uses (``live/session.py:773-784``). There is no second
policy engine here to drift out of sync.

Two branches per photo, because a single number cannot separate "the
model could not rank this face" from "the quality gate threw it away":

- **L** is what the App sees. It goes through the quality gate, and a
  rejected photo carries no scores at all.
- **R** skips the quality gate and reports only the model's ranking.
  It is systematically *more* optimistic than the App and must never be
  read as an App result.

**Private imports, deliberately.** ``_build_true_context`` and
``_load_profile`` are private to ``research/cli.py``. The plan's
alternative was to promote them to public API, which would change a
module the D5 boundary says must not change. Importing them here keeps
``src/facecore/research/cli.py`` byte-identical; the cost is that a
rename upstream would surface as an ImportError in this file, which is
a louder and safer failure than a silent behaviour drift.

Two counting rules are load-bearing and easy to get wrong:

1. **FA means `matched`** — score ≥ match *and* margin ≥ margin.
   ``eval/sweep.py:50-52`` decides FA on score alone; that would
   disagree with the App, so it is not used here.
2. **Rejected photos stay in the denominator.** ``summarize`` divides by
   the photo count. Dropping an unusable photo turns a messy run into a
   clean-looking one.

Per-photo rows are written outside the repo by ``main``; only counts and
score ranges reach a committed report, through
``facecore.eval.report.write_report``'s mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

# Imported for their side of the App path, unchanged:
#   score_frame                 live/frame_pipeline.py:366
#   enforce_single_face         pipeline/detect.py:13
#   align_crop                  pipeline/align.py:90
#   cosine_score                policy/identify.py:26
# All four are the exact functions score_frame itself calls, so the R
# branch ranks with the same arithmetic the L branch does.
from facecore.live.contracts import (
    FrameDiagnostics,
    FrameObservation,
)
from facecore.live.frame_pipeline import ScoringContext, score_frame
from facecore.pipeline.align import align_crop
from facecore.pipeline.decode import DecodedImage
from facecore.pipeline.detect import enforce_single_face
from facecore.policy.identify import cosine_score

# Private by name only; see the module docstring.
from facecore.research.cli import _build_true_context, _load_profile  # noqa: PLC2701

#: The probe set's truth identity. The operator confirmed all 13 probe
#: photos are the enrolled `enroll-23` (decision
#: `d-20260930051029074216-21`); kept as a constant so Task 2 passes it
#: explicitly on the command line rather than having it buried here.
PROBE_TRUTH = "enroll-23"

#: Photo-set labels used in the per-photo CSV and the summary.
SET_PROBE = "probe"
SET_NONTARGET = "nontarget"

BAND_MATCHED = "matched"
BAND_REVIEW = "review"
BAND_UNKNOWN = "unknown"
BAND_REJECTED = "rejected"

CLASS_CORRECT_ACCEPT = "correct_accept"
CLASS_WRONG_IDENTITY = "wrong_identity"
CLASS_REVIEW = "review"
CLASS_UNKNOWN_REJECT = "unknown_reject"
CLASS_QUALITY_REJECT = "quality_reject"
CLASS_FALSE_ACCEPT = "false_accept"
CLASS_CORRECT_REJECT = "correct_reject"
CLASS_UNPROCESSABLE = "unprocessable"


@dataclass(frozen=True)
class PhotoResult:
    """One photo scored on both branches.

    ``l_*`` is empty (``None``) whenever the quality gate rejected the
    photo or no single face was found — that emptiness is the App's
    view, not a missing measurement.
    """

    set_name: str
    photo: str
    truth_identity: str | None
    face_count: int
    unprocessable_reason: str | None
    quality_pass: bool
    quality_reasons: tuple[str, ...]
    l_top1: str | None
    l_top1_score: float | None
    l_top2: str | None
    l_top2_score: float | None
    l_margin: float | None
    l_band: str
    r_top1: str | None
    r_top1_score: float | None
    r_top2: str | None
    r_top2_score: float | None
    r_margin: float | None
    r_band: str | None
    stage_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineSummary:
    """Counts and score ranges for one photo set.

    ``total`` is the denominator and is always the number of photos
    handed in. The class counts partition it.
    """

    set_name: str
    total: int
    correct_accepts: int
    wrong_identities: int
    review: int
    unknown_rejects: int
    quality_rejected: int
    false_accepts: int
    correct_rejects: int
    unprocessable: int
    l_top1_correct: int
    r_top1_correct: int
    score_ranges: dict[str, dict[str, float | None]]


@dataclass(frozen=True)
class SweepCell:
    """One (match, margin) grid point, under live semantics."""

    match_threshold: float
    margin_threshold: float
    correct_accepts: int
    wrong_identities: int
    review: int
    unknown_rejects: int
    false_accepts: int
    total: int
    #: Always False. D5 has no holdout set, so naming a "best" cell
    #: would be tuning thresholds on the data being measured.
    selected: bool = False


def _band(
    top_score: float,
    runner_up_score: float | None,
    *,
    match_threshold: float,
    review_threshold: float,
    margin_threshold: float,
) -> str:
    """Live's single-frame band rule, copied from ``session.py:773-784``.

    The ordering matters and is not cosmetic: ``runner_up is None`` is
    checked *before* any threshold. A lone gallery identity produces no
    margin to test, and live refuses the match on that ground alone — so
    a perfect 1.0 against one identity is ``unknown``, not ``matched``.

    The thresholds are passed in rather than read from a module global so
    the sweep can reuse this exact function at every grid point instead
    of re-deriving the rule.
    """
    if runner_up_score is None:
        return BAND_UNKNOWN
    margin = top_score - runner_up_score
    if top_score >= match_threshold and margin >= margin_threshold:
        return BAND_MATCHED
    if top_score >= review_threshold:
        return BAND_REVIEW
    return BAND_UNKNOWN


def _ranked(
    identity_scores: dict[str, float],
) -> tuple[str | None, float | None, str | None, float | None, float | None]:
    """Sort a score map into (top1, score, top2, score, margin)."""
    if not identity_scores:
        return (None, None, None, None, None)
    ordered = sorted(identity_scores.items(), key=lambda it: it[1], reverse=True)
    top_ident, top_score = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else None
    runner_ident = ordered[1][0] if len(ordered) > 1 else None
    margin = None if runner_up is None else top_score - runner_up
    return (top_ident, top_score, runner_ident, runner_up, margin)


def _band_of(
    observation: FrameObservation, profile: Any
) -> tuple[str, str | None, float | None, float | None, str | None, float | None]:
    """Derive the App's band from a live observation.

    A rejected observation reports ``rejected`` with no scores at all —
    that emptiness is what the App actually has, and it is why the R
    branch has to exist.
    """
    if not observation.quality_pass or not observation.identity_scores:
        return (BAND_REJECTED, None, None, None, None, None)
    top1, s1, top2, s2, margin = _ranked(observation.identity_scores)
    band = _band(
        s1 or 0.0,
        s2,
        match_threshold=profile.match_threshold,
        review_threshold=profile.review_threshold,
        margin_threshold=profile.margin_threshold,
    )
    return (band, top1, s1, margin, top2, s2)


def _r_branch(
    rgb: np.ndarray,
    context: ScoringContext,
    profile: Any,
) -> tuple[str | None, float | None, str | None, float | None, float | None, str]:
    """Rank one photo without the quality gate.

    This deliberately re-uses ``context.detector.detect``,
    ``align_crop`` and ``cosine_score`` — the same three calls
    ``score_frame`` makes at ``frame_pipeline.py:400``/``:457``/``:540``
    — so R and L differ *only* by the skipped gate. It stops at the
    single-face check: a photo with no face, or more than one, has no
    crop to embed, so R reports ``None`` and the photo is counted as
    unprocessable in both branches.
    """
    height, width, _ = rgb.shape
    decoded = DecodedImage(
        width=width,
        height=height,
        color_order="RGB",
        pixels=rgb.tobytes(),
    )
    faces = context.detector.detect(decoded)
    status, _reason, face = enforce_single_face(faces)
    if status != "ok" or face is None:
        return (None, None, None, None, None, BAND_UNKNOWN)

    crop = align_crop(decoded.pixels, decoded.width, decoded.height, face)
    probe_vec, embed_model = context.embedder.embed(crop)
    if embed_model != context.model_version:
        raise ValueError(
            f"cross-model comparison refused: {embed_model!r} "
            f"vs {context.model_version!r}"
        )
    scores = {
        ident: cosine_score(probe_vec, gal_vec)
        for ident, gal_vec in context.gallery.embeddings.items()
    }
    top1, s1, top2, s2, margin = _ranked(scores)
    band = _band(
        s1 or 0.0,
        s2,
        match_threshold=profile.match_threshold,
        review_threshold=profile.review_threshold,
        margin_threshold=profile.margin_threshold,
    )
    return (top1, s1, top2, s2, margin, band)


def _decode_photo(path: Path) -> np.ndarray:
    """Read a photo as the uint8 RGB array the camera path produces.

    The live gallery decode does not apply EXIF rotation
    (``frame_pipeline.py:210-213``); the same is done here so the
    static path sees exactly what the App would see. The D5 corpus has
    no orientation-tagged photos, so this is a no-op in practice, but
    copying it keeps the two paths from diverging later.
    """
    with Image.open(path) as img:
        return np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))


def score_photo(
    path: Path,
    context: ScoringContext,
    profile: Any,
    *,
    set_name: str,
    truth_identity: str | None,
    sequence: int,
) -> PhotoResult:
    """Score one photo file on both branches.

    ``path`` is decoded to the same uint8 RGB array the camera path
    produces, then wrapped in a ``FramePacket`` — no live module is
    modified to make a still photo scorable (plan H2).
    """
    from facecore.live.contracts import FramePacket  # noqa: PLC0415

    rgb = _decode_photo(Path(path))
    packet = FramePacket(
        sequence=sequence, captured_ns=0, rgb=rgb, orientation=0, mirrored=False
    )

    diagnostics: list[FrameDiagnostics] = []
    observation = score_frame(packet, context, diagnostic_sink=diagnostics.append)
    stage_ms = dict(diagnostics[-1].stage_durations_ms) if diagnostics else {}

    face_count = observation.face_count
    unprocessable_reason: str | None = None
    if face_count != 1:
        unprocessable_reason = (
            "no_face_detected" if face_count == 0 else "multiple_faces"
        )

    if face_count == 1:
        r_top1, r_s1, r_top2, r_s2, r_margin, r_band = _r_branch(rgb, context, profile)
    else:
        r_top1 = r_s1 = r_top2 = r_s2 = r_margin = None
        r_band = BAND_UNKNOWN

    l_band, l_top1, l_s1, l_margin, l_top2, l_s2 = _band_of(observation, profile)

    return PhotoResult(
        set_name=set_name,
        photo=Path(path).name,
        truth_identity=truth_identity,
        face_count=face_count,
        unprocessable_reason=unprocessable_reason,
        quality_pass=observation.quality_pass,
        quality_reasons=tuple(observation.quality_reasons),
        l_top1=l_top1,
        l_top1_score=l_s1,
        l_top2=l_top2,
        l_top2_score=l_s2,
        l_margin=l_margin,
        l_band=l_band,
        r_top1=r_top1,
        r_top1_score=r_s1,
        r_top2=r_top2,
        r_top2_score=r_s2,
        r_margin=r_margin,
        r_band=r_band,
        stage_ms=stage_ms,
    )


def classify_nontarget(result: PhotoResult) -> str:
    """Bucket one non-target row using live's definition.

    ``false_accept`` requires the ``matched`` band, which is score *and*
    margin. This is where ``sweep.py``'s score-only rule would differ,
    and the plan requires live's rule.
    """
    if result.unprocessable_reason is not None:
        return CLASS_UNPROCESSABLE
    if result.l_band == BAND_REJECTED:
        return CLASS_QUALITY_REJECT
    if result.l_band == BAND_MATCHED:
        return CLASS_FALSE_ACCEPT
    if result.l_band == BAND_REVIEW:
        return CLASS_REVIEW
    return CLASS_CORRECT_REJECT


def classify_probe(result: PhotoResult) -> str:
    """Bucket one probe row (truth is ``result.truth_identity``)."""
    if result.unprocessable_reason is not None:
        return CLASS_UNPROCESSABLE
    if result.l_band == BAND_REJECTED:
        return CLASS_QUALITY_REJECT
    if result.l_band == BAND_MATCHED:
        return (
            CLASS_CORRECT_ACCEPT
            if result.l_top1 == result.truth_identity
            else CLASS_WRONG_IDENTITY
        )
    if result.l_band == BAND_REVIEW:
        return CLASS_REVIEW
    return CLASS_UNKNOWN_REJECT


def _range(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 4),
        "median": round(statistics.median(ordered), 4),
        "max": round(ordered[-1], 4),
    }


def summarize(results: list[PhotoResult], *, set_name: str) -> BaselineSummary:
    """Count one photo set.

    ``total`` is ``len(results)`` — the denominator never shrinks to
    exclude a photo the pipeline could not use. The class counts are
    asserted to partition it in the tests, because a harness that drops
    a row is a harness that reports a better run than happened.
    """
    if set_name == SET_PROBE:
        counts = [classify_probe(r) for r in results]
        correct_accepts = counts.count(CLASS_CORRECT_ACCEPT)
        wrong = counts.count(CLASS_WRONG_IDENTITY)
        review = counts.count(CLASS_REVIEW)
        unknown = counts.count(CLASS_UNKNOWN_REJECT)
        false_accepts = correct_rejects = 0
    elif set_name == SET_NONTARGET:
        counts = [classify_nontarget(r) for r in results]
        correct_accepts = wrong = 0
        review = counts.count(CLASS_REVIEW)
        unknown = 0
        false_accepts = counts.count(CLASS_FALSE_ACCEPT)
        correct_rejects = counts.count(CLASS_CORRECT_REJECT)
    else:
        raise ValueError(f"unknown set_name {set_name!r}")

    quality_rejected = counts.count(CLASS_QUALITY_REJECT)
    unprocessable = counts.count(CLASS_UNPROCESSABLE)

    def _top1_correct(r: PhotoResult, branch: str) -> bool:
        """Ranking correctness, counted independently of the band.

        A non-target has no truth identity, so "ranked correctly" is not
        a question this set can answer — only the probe set reports it.
        """
        if set_name != SET_PROBE:
            return False
        top1 = r.l_top1 if branch == "l" else r.r_top1
        if top1 is None or r.truth_identity is None:
            return False
        return top1 == r.truth_identity

    probe_scores = [r.l_top1_score for r in results if r.l_top1_score is not None]
    probe_margins = [r.l_margin for r in results if r.l_margin is not None]
    nt_scores = [r.l_top1_score for r in results if r.l_top1_score is not None]
    nt_margins = [r.l_margin for r in results if r.l_margin is not None]

    return BaselineSummary(
        set_name=set_name,
        total=len(results),
        correct_accepts=correct_accepts,
        wrong_identities=wrong,
        review=review,
        unknown_rejects=unknown,
        quality_rejected=quality_rejected,
        false_accepts=false_accepts,
        correct_rejects=correct_rejects,
        unprocessable=unprocessable,
        l_top1_correct=sum(_top1_correct(r, "l") for r in results),
        r_top1_correct=sum(_top1_correct(r, "r") for r in results),
        score_ranges={
            "l_top1_score": _range(probe_scores),
            "l_margin": _range(probe_margins),
            "nt_top1_score": _range(nt_scores),
            "nt_margin": _range(nt_margins),
        },
    )


def sweep_live_semantics(
    results: list[PhotoResult],
    match_grid: list[float],
    margin_grid: list[float],
    *,
    review: float,
) -> list[SweepCell]:
    """Present the frozen grid under live semantics. Never selects a cell.

    D5 has no holdout: the 13+30 photos *are* the evaluation set, so
    naming a "best" threshold from this grid would be tuning on the test
    data. Every cell therefore carries ``selected=False``.
    """
    cells: list[SweepCell] = []
    for match_t in match_grid:
        for margin_t in margin_grid:
            matched = wrong = review_n = unknown = false_accept = 0
            for r in results:
                s1 = r.l_top1_score
                if s1 is None:
                    continue
                band = _band(
                    s1,
                    r.l_top2_score,
                    match_threshold=match_t,
                    review_threshold=review,
                    margin_threshold=margin_t,
                )
                if band == BAND_MATCHED:
                    if r.truth_identity is None:
                        false_accept += 1
                    elif r.l_top1 == r.truth_identity:
                        matched += 1
                    else:
                        wrong += 1
                elif band == BAND_REVIEW:
                    review_n += 1
                else:
                    unknown += 1
            cells.append(
                SweepCell(
                    match_threshold=match_t,
                    margin_threshold=margin_t,
                    correct_accepts=matched,
                    wrong_identities=wrong,
                    review=review_n,
                    unknown_rejects=unknown,
                    false_accepts=false_accept,
                    total=len(results),
                )
            )
    return cells


def render_report_body(summary: BaselineSummary) -> str:
    """Counts and ranges only — no per-photo identity, no paths.

    ``write_report`` rejects a body carrying an absolute path or a
    per-image identity label, so a summary rendered here is safe to
    commit; the per-photo detail lives in the repo-external CSV.
    """
    lines = [
        f"## {summary.set_name}",
        "",
        f"- photos: {summary.total}",
        f"- correct accepts: {summary.correct_accepts}/{summary.total}",
        f"- wrong identity: {summary.wrong_identities}/{summary.total}",
        f"- review: {summary.review}/{summary.total}",
        f"- unknown reject: {summary.unknown_rejects}/{summary.total}",
        f"- quality rejected: {summary.quality_rejected}/{summary.total}",
        f"- false accepts: {summary.false_accepts}/{summary.total}",
        f"- correct rejects: {summary.correct_rejects}/{summary.total}",
        f"- unprocessable: {summary.unprocessable}/{summary.total}",
    ]
    for key, stats in summary.score_ranges.items():
        if stats["min"] is None:
            continue
        lines.append(
            f"- {key}: min {stats['min']}, median {stats['median']}, max {stats['max']}"
        )
    return "\n".join(lines) + "\n"


def _photos_in(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _write_detail(rows: list[PhotoResult], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) | {"stage_ms": json.dumps(r.stage_ms)} for r in rows]
    fieldnames = list(payload[0].keys()) if payload else []
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)


def main(argv: list[str] | None = None) -> int:
    """Score both photo sets and write the repo-external detail CSV.

    The committed report is a separate step (plan Task 3 / PR-B): this
    entry point deliberately does not write into ``docs/``.
    """
    parser = argparse.ArgumentParser(prog="facecore.eval.static_baseline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--probe-truth", default=PROBE_TRUTH)
    parser.add_argument("--nontarget", type=Path, required=True)
    parser.add_argument("--detail-out", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=None)
    parser.add_argument("--profile", type=Path, default=None)
    args = parser.parse_args(argv)

    from facecore.research.g3_config import load_g3_config  # noqa: PLC0415

    config = load_g3_config(args.config)
    models = args.models or config.models_dir
    profile = (
        _load_profile(args.profile)
        if args.profile
        else _load_profile(Path(__file__).parents[3] / "profiles" / "g3-v1.json")
    )
    context = _build_true_context(
        models, None, profile, gallery_dir=config.enrollment_dir
    )

    rows: list[PhotoResult] = []
    for seq, photo in enumerate(_photos_in(args.probes), start=1):
        rows.append(
            score_photo(
                photo,
                context,
                profile,
                set_name=SET_PROBE,
                truth_identity=args.probe_truth,
                sequence=seq,
            )
        )
    for seq, photo in enumerate(_photos_in(args.nontarget), start=1):
        rows.append(
            score_photo(
                photo,
                context,
                profile,
                set_name=SET_NONTARGET,
                truth_identity=None,
                sequence=seq,
            )
        )

    _write_detail(rows, args.detail_out)
    for set_name in (SET_PROBE, SET_NONTARGET):
        subset = [r for r in rows if r.set_name == set_name]
        summary = summarize(subset, set_name=set_name)
        print(render_report_body(summary))
    print(f"detail rows: {len(rows)} -> {args.detail_out.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
