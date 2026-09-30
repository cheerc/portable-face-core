"""D6 C: measure *why* ``quality_exposure`` fires, one predicate at a time.

The product emits a single code for three different conditions, because
``pipeline/quality.py:41-46`` is one ``if`` with an ``or``:

    low, high = policy.exposure_luma_range
    if (
        not (low <= mean_luma <= high)
        or clipped_fraction > policy.exposure_clipped_fraction_max
    ):
        codes.append("quality_exposure")

So a `quality_exposure` rejection means **too dark** or **too bright**
or **clipped**, and the code alone cannot say which. D5 counted 23 of
them across 43 photos and was careful not to conclude they were all too
dark; this module replaces that restraint with a measurement.

Three properties the rest of D6 leans on, and which are cheap to state
and expensive to lose:

1. **The predicates are the product's, not a second copy of them.**
   ``exposure_causes`` is checked against ``evaluate_quality`` on a
   grid in the tests. A split that disagreed with the gate would make
   the audit describe a rule the pipeline does not run.
2. **Causes are a set, not a label.** A photo that is both dark and
   clipped carries both, and the count of such photos is itself part
   of the summary. Collapsing them to one column would re-create the
   ambiguity C exists to remove.
3. **The measured pixels are the aligned crop.** ``evaluate_quality``
   is called at ``live/frame_pipeline.py:456`` with the values
   ``measure.exposure_of(crop)`` returns; the original photo is never
   measured on that path. A full-image figure is computed too, kept in
   its own fields, and never allowed to decide a cause.

**This module selects nothing.** There is no threshold grid and no
"best" value, because the corpus has no holdout — the same reason
``static_baseline.SweepCell.selected`` is always False. It can say how
far a photo sits from a bound; it cannot say the bound should move.

Private imports from ``research/cli.py`` mirror ``static_baseline``'s
deliberate choice, so ``research/cli.py`` stays byte-identical.

Per-photo rows are written outside the repo by ``main``. Only counts
and ranges reach a committed report, through
``facecore.eval.report.write_report``'s mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from facecore.contracts.policy import PolicyProfile
from facecore.live.contracts import FrameDiagnostics, FrameObservation, FramePacket
from facecore.live.frame_pipeline import ScoringContext, score_frame

# Private by name only; see the module docstring.
from facecore.research.cli import _build_true_context, _load_profile  # noqa: PLC2701

SET_GALLERY = "gallery"
SET_PROBE = "probe"
SET_NONTARGET = "nontarget"
KNOWN_SETS = (SET_GALLERY, SET_PROBE, SET_NONTARGET)

CAUSE_TOO_DARK = "too_dark"
CAUSE_TOO_BRIGHT = "too_bright"
CAUSE_CLIPPED = "clipped"
#: Fixed order, so a row's ``causes`` tuple is comparable across runs.
CAUSE_ORDER = (CAUSE_TOO_DARK, CAUSE_TOO_BRIGHT, CAUSE_CLIPPED)

#: BT.601 luma, the same coefficients ``measure._luma`` uses
#: (``pipeline/measure.py:26``). Duplicated rather than imported because
#: that helper takes an ``AlignedCrop`` and this call site has whole
#: decoded pixels; the test that matters here is
#: ``test_aligned_crop_measurement_matches_the_pipeline``, which forces
#: the two to agree on a real frame.
_LUMA_R, _LUMA_G, _LUMA_B = 0.299, 0.587, 0.114


@dataclass(frozen=True)
class ExposureCauses:
    """Which of the three exposure predicates a measurement fires.

    Kept as three independent booleans rather than one enum because the
    combinations are real: a dark frame with blown highlights trips two
    at once, and reporting only the first would be a guess.
    """

    too_dark: bool
    too_bright: bool
    clipped: bool

    @property
    def triggered(self) -> tuple[str, ...]:
        return tuple(
            cause
            for cause in CAUSE_ORDER
            if getattr(self, cause)
        )

    @property
    def any_triggered(self) -> bool:
        return bool(self.triggered)


def exposure_causes(
    *,
    mean_luma: float,
    clipped_fraction: float,
    policy: PolicyProfile,
) -> ExposureCauses:
    """Split the product's single exposure gate into its three predicates.

    The two comparison styles are the product's, copied deliberately:
    the luma range is **inclusive** (``low <= mean_luma <= high``) and
    the clip test is **strict** (``>``). Flipping either would move a
    boundary photo across the line, so both boundaries are asserted on
    each side in the tests.
    """
    low, high = policy.exposure_luma_range
    return ExposureCauses(
        too_dark=mean_luma < low,
        too_bright=mean_luma > high,
        clipped=clipped_fraction > policy.exposure_clipped_fraction_max,
    )


def luma_margin(mean_luma: float, policy: PolicyProfile) -> float:
    """Signed distance from the mean luma to the nearest luma bound.

    Positive means inside the range, negative means outside by that much
    luma. This is a *distance*, never a suggestion: it says how far a
    photo sits from a bound, and nothing here is authorised to say the
    bound should move.
    """
    low, high = policy.exposure_luma_range
    return min(mean_luma - low, high - mean_luma)


def _full_image_exposure(rgb: np.ndarray) -> tuple[float, float]:
    """Mean luma and clipped fraction over the whole decoded photo.

    Counterfactual only. ``evaluate_quality`` never sees this — it is
    handed the aligned crop's numbers — and the two can disagree in
    either direction, which is why they are stored apart and no cause
    is ever derived from these.
    """
    mix = _LUMA_R * rgb[:, :, 0] + _LUMA_G * rgb[:, :, 1] + _LUMA_B * rgb[:, :, 2]
    return float(np.mean(mix)), float(np.mean((mix <= 2) | (mix >= 253)))


def _decode_photo(path: Path) -> np.ndarray:
    """Read a photo as the uint8 RGB array the camera path produces.

    Mirrors ``static_baseline._decode_photo`` — no EXIF rotation, the
    same as the live gallery decode at ``frame_pipeline.py:210-213``.
    """
    with Image.open(path) as img:
        return np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))


def _last_measurement(diagnostics: list[FrameDiagnostics]) -> FrameDiagnostics | None:
    """The one diagnostic carrying a measurement, if the pipeline got one.

    ``score_frame`` emits a diagnostic on every path, but the
    no-single-face path fills every measurement with ``None``. Rather
    than check ``face_count`` here and hope the two stay in step, this
    looks for a diagnostic that actually measured something — a
    photo with one face and no luma would be a bug, and silently
    treating it as unprocessable would hide it.
    """
    for diag in reversed(diagnostics):
        if diag.mean_luma is not None and diag.clipped_fraction is not None:
            return diag
    return None


@dataclass(frozen=True)
class AuditRow:
    """One photo, with the exposure causes measured separately.

    The ``crop_*`` fields are what the gate saw. The ``full_*`` fields
    are the same measurement taken over the original photo, kept only
    for comparison; they never decide a cause. Both are ``None`` for a
    photo with no single face, which has no crop to measure.
    """

    set_name: str
    photo: str
    face_count: int
    unprocessable_reason: str | None
    quality_pass: bool
    quality_reasons: tuple[str, ...]
    crop_mean_luma: float | None
    crop_clipped_fraction: float | None
    crop_sharpness: float | None
    too_dark: bool
    too_bright: bool
    clipped: bool
    luma_margin: float | None
    full_mean_luma: float | None
    full_clipped_fraction: float | None
    full_too_dark: bool
    full_too_bright: bool
    full_clipped: bool
    stage_ms: dict[str, float] = field(default_factory=dict)

    @property
    def causes(self) -> tuple[str, ...]:
        return tuple(c for c in CAUSE_ORDER if getattr(self, c))

    @property
    def measured(self) -> bool:
        """True when a crop was measured, i.e. there is a number to attribute."""
        return self.crop_mean_luma is not None

    @property
    def disagrees_with_full_image(self) -> bool:
        """True when the crop and the original photo would judge differently.

        Informational. The crop is what the gate uses, so a
        disagreement is a fact about the photo's framing, not a defect
        in either measurement.
        """
        if not self.measured or self.full_mean_luma is None:
            return False
        return self.causes != self._full_causes()

    def _full_causes(self) -> tuple[str, ...]:
        flags = {
            CAUSE_TOO_DARK: self.full_too_dark,
            CAUSE_TOO_BRIGHT: self.full_too_bright,
            CAUSE_CLIPPED: self.full_clipped,
        }
        return tuple(c for c in CAUSE_ORDER if flags[c])

    def to_dict(self) -> dict[str, Any]:
        """The repo-external detail row. No paths, no truth identity.

        ``photo`` is the bare filename, matching D5's detail CSV, and
        ``truth_identity`` is deliberately absent: the audit asks why a
        photo was rejected, never who it was.
        """
        return {
            "set_name": self.set_name,
            "photo": self.photo,
            "face_count": self.face_count,
            "unprocessable_reason": self.unprocessable_reason,
            "quality_pass": self.quality_pass,
            "quality_reasons": repr(self.quality_reasons),
            "causes": list(self.causes),
            "crop_mean_luma": self.crop_mean_luma,
            "crop_clipped_fraction": self.crop_clipped_fraction,
            "crop_sharpness": self.crop_sharpness,
            "too_dark": self.too_dark,
            "too_bright": self.too_bright,
            "clipped": self.clipped,
            "luma_margin": self.luma_margin,
            "full_mean_luma": self.full_mean_luma,
            "full_clipped_fraction": self.full_clipped_fraction,
            "full_too_dark": self.full_too_dark,
            "full_too_bright": self.full_too_bright,
            "full_clipped": self.full_clipped,
            "disagrees_with_full_image": self.disagrees_with_full_image,
            "stage_ms": json.dumps(self.stage_ms, sort_keys=True),
        }


def audit_photo(
    path: Path,
    context: ScoringContext,
    policy: PolicyProfile,
    *,
    set_name: str,
    sequence: int = 1,
    measure_full_image: bool = True,
) -> AuditRow:
    """Audit one photo file.

    Runs the real ``score_frame`` and reads the measured luma, clip
    fraction and sharpness off the ``FrameDiagnostics`` the pipeline
    itself emitted — the same values ``evaluate_quality`` was handed at
    ``frame_pipeline.py:456``. Nothing is re-measured from the original
    photo, because a number the gate never saw cannot explain the gate.

    ``measure_full_image=False`` skips the counterfactual. It is
    exposed so a test can prove the comparison does not feed back into
    any cause. ``sequence`` only labels the frame; ``FramePacket``
    requires it to be >= 1, so ``main`` numbers photos per set and the
    default suits single-photo callers.
    """
    if set_name not in KNOWN_SETS:
        raise ValueError(f"unknown set_name {set_name!r}")
    if sequence < 1:
        raise ValueError(f"sequence must be >= 1, got {sequence}")

    rgb = _decode_photo(Path(path))
    packet = FramePacket(
        sequence=sequence, captured_ns=0, rgb=rgb, orientation=0, mirrored=False
    )
    diagnostics: list[FrameDiagnostics] = []
    observation: FrameObservation = score_frame(
        packet, context, diagnostic_sink=diagnostics.append
    )
    diag = _last_measurement(diagnostics)

    unprocessable_reason: str | None = None
    if observation.face_count != 1:
        unprocessable_reason = (
            "no_face_detected" if observation.face_count == 0 else "multiple_faces"
        )
    elif diag is None:
        # One face found but nothing measured: the pipeline would have
        # crashed before reaching the gate. Refusing is the only honest
        # row — labelling it unprocessable would report a pipeline
        # defect as a photo problem.
        raise ValueError(
            f"{Path(path).name}: single face found but no quality measurement; "
            "score_frame did not reach the gate"
        )

    causes = (
        ExposureCauses(False, False, False)
        if diag is None
        else exposure_causes(
            mean_luma=float(diag.mean_luma),  # type: ignore[arg-type]
            clipped_fraction=float(diag.clipped_fraction),  # type: ignore[arg-type]
            policy=policy,
        )
    )

    full_mean = full_clip = None
    full_flags = ExposureCauses(False, False, False)
    if measure_full_image:
        full_mean, full_clip = _full_image_exposure(rgb)
        full_flags = exposure_causes(
            mean_luma=full_mean, clipped_fraction=full_clip, policy=policy
        )

    return AuditRow(
        set_name=set_name,
        photo=Path(path).name,
        face_count=observation.face_count,
        unprocessable_reason=unprocessable_reason,
        quality_pass=observation.quality_pass,
        quality_reasons=tuple(observation.quality_reasons),
        crop_mean_luma=None if diag is None else float(diag.mean_luma),  # type: ignore[arg-type]
        crop_clipped_fraction=(
            None if diag is None else float(diag.clipped_fraction)  # type: ignore[arg-type]
        ),
        crop_sharpness=None if diag is None else float(diag.sharpness),  # type: ignore[arg-type]
        too_dark=causes.too_dark,
        too_bright=causes.too_bright,
        clipped=causes.clipped,
        luma_margin=(
            None if diag is None else luma_margin(float(diag.mean_luma), policy)  # type: ignore[arg-type]
        ),
        full_mean_luma=full_mean,
        full_clipped_fraction=full_clip,
        full_too_dark=full_flags.too_dark,
        full_too_bright=full_flags.too_bright,
        full_clipped=full_flags.clipped,
        stage_ms=dict(diag.stage_durations_ms) if diag is not None else {},
    )


@dataclass(frozen=True)
class AuditSummary:
    """Counts and ranges for one photo set.

    ``total`` is the row count and never shrinks. The three cause
    counts overlap on purpose — they count *predicates that fired*, not
    photos — so they are not required to sum to anything, while
    ``quality_passed + quality_rejected + unprocessable`` does partition
    ``total``.
    """

    set_name: str
    total: int
    measured: int
    quality_passed: int
    quality_rejected: int
    unprocessable: int
    too_dark: int
    too_bright: int
    clipped: int
    two_or_more_causes: int
    exposure_rejected: int
    luma_min: float | None
    luma_max: float | None
    clip_max: float | None
    disagree_with_full_image: int

    def render_body(self) -> str:
        """Counts and ranges only — no per-photo identity, no paths.

        ``write_report`` rejects a body carrying an absolute path or a
        per-image identity label, so a summary rendered here is safe to
        commit; the per-photo detail lives in the repo-external CSV.
        """
        lines = [
            f"## {self.set_name}",
            "",
            f"- photos: {self.total}",
            f"- measured (single face, crop available): {self.measured}",
            f"- quality passed: {self.quality_passed}/{self.total}",
            f"- quality rejected: {self.quality_rejected}/{self.total}",
            f"- unprocessable: {self.unprocessable}/{self.total}",
            "",
            "Predicates fired (these overlap; a photo may fire two):",
            f"- mean luma below lower bound (too dark): {self.too_dark}",
            f"- mean luma above upper bound (too bright): {self.too_bright}",
            f"- clipped fraction over max: {self.clipped}",
            f"- photos firing two or more: {self.two_or_more_causes}",
            f"- quality_exposure rejections: {self.exposure_rejected}",
        ]
        if self.luma_min is not None and self.luma_max is not None:
            lines.append(
                f"- crop mean luma range: {self.luma_min:.2f} .. {self.luma_max:.2f}"
            )
        if self.clip_max is not None:
            lines.append(f"- max crop clipped fraction: {self.clip_max:.4f}")
        lines.append(
            f"- crop/photo exposure judgment differs: {self.disagree_with_full_image}"
        )
        return "\n".join(lines) + "\n"


def _rounded_extreme(
    values: Sequence[float | None], pick: Callable[[Sequence[float]], float]
) -> float | None:
    """``pick`` over the present values, rounded; ``None`` if there are none.

    Written as its own function because the naive ``round(min(values), 4)
    if values else None`` needs a ``type: ignore`` under mypy strict:
    the list is typed ``float | None`` (an unprocessable row has no
    measurement) but ``min`` over it is only reached once it is known
    to be non-empty, and the ignore comment then says nothing.
    """
    present = [v for v in values if v is not None]
    return None if not present else round(pick(present), 4)


def summarize_audit(rows: Sequence[AuditRow], *, set_name: str) -> AuditSummary:
    """Count one photo set.

    The denominator is ``len(rows)``. A rejected photo is the subject
    of this audit, so dropping one would turn "23 photos were rejected"
    into a report about the photos that were not — and an unprocessable
    photo has no measurements, so excluding it from the range while
    keeping it in the count is the honest split, not an inconsistency.
    """
    if set_name not in KNOWN_SETS:
        raise ValueError(f"unknown set_name {set_name!r}")
    matching = [r for r in rows if r.set_name == set_name]
    lumas = [r.crop_mean_luma for r in matching if r.crop_mean_luma is not None]
    clips = [
        r.crop_clipped_fraction
        for r in matching
        if r.crop_clipped_fraction is not None
    ]
    return AuditSummary(
        set_name=set_name,
        total=len(matching),
        measured=len(lumas),
        quality_passed=sum(1 for r in matching if r.quality_pass),
        quality_rejected=sum(1 for r in matching if not r.quality_pass),
        unprocessable=sum(
            1 for r in matching if r.unprocessable_reason is not None
        ),
        too_dark=sum(1 for r in matching if r.too_dark),
        too_bright=sum(1 for r in matching if r.too_bright),
        clipped=sum(1 for r in matching if r.clipped),
        two_or_more_causes=sum(1 for r in matching if len(r.causes) >= 2),
        exposure_rejected=sum(
            1 for r in matching if "quality_exposure" in r.quality_reasons
        ),
        luma_min=_rounded_extreme(lumas, min),
        luma_max=_rounded_extreme(lumas, max),
        clip_max=_rounded_extreme(clips, max),
        disagree_with_full_image=sum(
            1 for r in matching if r.disagrees_with_full_image
        ),
    )


@dataclass(frozen=True)
class MaskResult:
    """The outcome of comparing C's rows against a D5 detail CSV.

    ``missing_in_mask`` is the interesting one: a photo C measured that
    D5 never scored means the two runs covered different corpora, and
    any conclusion carried across between them would be unfounded.
    """

    checked: int
    mismatched: int
    mismatched_photos: tuple[str, ...]
    missing_in_audit: tuple[str, ...]
    missing_in_mask: tuple[str, ...]
    duplicate_photos: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            self.mismatched == 0
            and not self.missing_in_mask
            and not self.missing_in_audit
            and not self.duplicate_photos
        )


def _reasons_repr(reasons: str) -> tuple[str, ...]:
    """Parse a D5 ``quality_reasons`` cell into a tuple of codes.

    D5 writes ``repr(tuple(...))`` (``static_baseline._write_detail``),
    so a matching cell is ``"('quality_exposure',)"``. Parsing is done
    with ``ast.literal_eval`` rather than by stripping punctuation, so a
    cell that is not a tuple raises instead of being half-read.
    """
    import ast  # noqa: PLC0415

    return tuple(ast.literal_eval(reasons))


def verify_d5_mask(rows: Sequence[AuditRow], mask_path: Path) -> MaskResult:
    """Check C's rows against a D5 detail CSV's quality columns.

    Compares ``quality_pass`` and ``quality_reasons`` per photo. The
    file is opened read-only and never written: it is the D5 evidence
    this audit has to be consistent with, and editing it would destroy
    the only record of what D5 actually observed.
    """
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"D5 detail CSV not found: {path.name}")

    seen: dict[str, int] = {}
    for r in rows:
        seen[r.photo] = seen.get(r.photo, 0) + 1
    duplicates = tuple(sorted(p for p, n in seen.items() if n > 1))
    if duplicates:
        # Two sets can legitimately reuse a filename, but then a name
        # alone no longer identifies a row and the comparison below
        # would silently match the wrong one.
        return MaskResult(
            checked=0,
            mismatched=0,
            mismatched_photos=(),
            missing_in_audit=(),
            missing_in_mask=(),
            duplicate_photos=duplicates,
        )

    audit_by_photo = {r.photo: r for r in rows}
    with path.open(encoding="utf-8", newline="") as handle:
        mask_rows = list(csv.DictReader(handle))

    mismatched: list[str] = []
    missing_in_mask: list[str] = []
    checked = 0
    for entry in mask_rows:
        photo = entry["photo"]
        if photo in missing_in_mask or photo in mismatched:
            continue
        if photo not in audit_by_photo:
            missing_in_mask.append(photo)
            continue
        checked += 1
        row = audit_by_photo[photo]
        mask_pass = entry["quality_pass"] == "True"
        mask_reasons = _reasons_repr(entry["quality_reasons"])
        if mask_pass != row.quality_pass or mask_reasons != row.quality_reasons:
            mismatched.append(photo)

    missing_in_audit = tuple(
        sorted(p for p in audit_by_photo if p not in {e["photo"] for e in mask_rows})
    )
    return MaskResult(
        checked=checked,
        mismatched=len(mismatched),
        mismatched_photos=tuple(sorted(mismatched)),
        missing_in_audit=missing_in_audit,
        missing_in_mask=tuple(sorted(missing_in_mask)),
        duplicate_photos=(),
    )


def _photos_in(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _write_detail(rows: list[AuditRow], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.to_dict() for r in rows]
    fieldnames = list(payload[0].keys()) if payload else []
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)


def main(argv: list[str] | None = None) -> int:
    """Audit the gallery, probe and non-target sets, writing detail outside the repo.

    The committed report is a separate step; this entry point writes
    only the repo-external detail CSV and prints counts.
    """
    parser = argparse.ArgumentParser(prog="facecore.eval.quality_audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, default=None)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--nontarget", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--d5-detail", type=Path, default=None)
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
    # The quality bounds live on the policy, not on the research
    # profile. ``_build_true_context`` already built exactly the policy
    # the pipeline scored with, and taking *that* object is the only
    # way the audit is guaranteed to be splitting the same gate the run
    # applied. Re-deriving one from the profile here would be a second
    # copy that could silently drift.
    policy: PolicyProfile = context.policy

    out_dir: Path = args.out_dir
    detail = out_dir / "d6-quality-audit-detail.csv"
    if detail.exists():
        # Fail clear rather than overwrite: a rerun that silently
        # replaced the evidence would leave two different audits with
        # one filename, and only the newer one would be checkable.
        print(
            f"quality_audit: {detail.name} already exists; refusing to overwrite",
            file=sys.stderr,
        )
        return 2

    rows: list[AuditRow] = []
    sets: list[tuple[str, Path]] = []
    if args.gallery is not None:
        sets.append((SET_GALLERY, args.gallery))
    sets.append((SET_PROBE, args.probes))
    sets.append((SET_NONTARGET, args.nontarget))
    for set_name, directory in sets:
        for seq, photo in enumerate(_photos_in(directory), start=1):
            rows.append(
                audit_photo(photo, context, policy, set_name=set_name, sequence=seq)
            )

    _write_detail(rows, detail)

    if args.d5_detail is not None:
        mask = verify_d5_mask(
            [r for r in rows if r.set_name in (SET_PROBE, SET_NONTARGET)],
            args.d5_detail,
        )
        print(
            f"d5 mask: checked {mask.checked}, mismatched {mask.mismatched}, "
            f"missing in mask {len(mask.missing_in_mask)}, "
            f"missing in audit {len(mask.missing_in_audit)}"
        )
        if not mask.ok:
            print("quality_audit: D5 mask did not verify", file=sys.stderr)
            return 3

    for set_name, _directory in sets:
        summary = summarize_audit(rows, set_name=set_name)
        print(summary.render_body())
    print(f"detail rows: {len(rows)} -> {detail.name}")
    print("note: this audit selects no threshold; the corpus has no holdout.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
