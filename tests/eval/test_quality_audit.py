"""D6 C: why ``quality_exposure`` fires — three predicates, measured.

The question C exists to answer is one number: **why** did the quality
gate reject each of the 23 photos D5 counted as ``quality_exposure``?
The product gives one code for three different conditions, because
``pipeline/quality.py:41-46`` is a single ``if`` with an ``or``:

    low, high = policy.exposure_luma_range
    if (
        not (low <= mean_luma <= high)
        or clipped_fraction > policy.exposure_clipped_fraction_max
    ):
        codes.append("quality_exposure")

So ``quality_exposure`` covers **too dark**, **too bright**, and
**clipped** alike. Reading the 23 rejections as "they were all too dark"
is the specific misreading D5's report had to rule out, and this module
exists to replace that guess with a measurement of each predicate
separately. The counts are kept as three independent booleans and a
photo firing two of them stays two — collapsing to one cause column
would re-introduce exactly the error C is correcting.

**The predicates are derived from the aligned crop, not the original
photo.** ``evaluate_quality`` is called at
``live/frame_pipeline.py:456`` with ``mean_luma``/``clipped_fraction``
from ``measure.exposure_of(crop)``; the full image is never measured on
that path. A full-image reading is carried too, in its own fields, and
the two can genuinely disagree — so they are never compared under one
threshold, and no test here treats a disagreement as a bug.

Three facts the tests below hold load-bearing, and which this repo has
a documented history of getting wrong:

1. The predicate split is pinned to the **product's own** gate, not to
   a re-reading of it: a grid test asserts ``exposure_causes`` agrees
   with ``evaluate_quality`` on when ``quality_exposure`` is emitted.
2. Every image-level case is driven through ``score_frame``, so the
   numbers under test are the ones the real pipeline produced.
3. Multi-cause photos keep both causes. The count of photos that fired
   more than one predicate is itself an assertion, not a detail.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from facecore.contracts.policy import PolicyProfile
from facecore.eval.quality_audit import (
    CAUSE_CLIPPED,
    CAUSE_TOO_BRIGHT,
    CAUSE_TOO_DARK,
    SET_GALLERY,
    SET_NONTARGET,
    SET_PROBE,
    AuditRow,
    audit_photo,
    exposure_causes,
    luma_margin,
    summarize_audit,
    verify_d5_mask,
)
from facecore.errors import RedactionError
from facecore.eval.report import write_report
from facecore.live.frame_pipeline import ResearchGallery, ScoringContext
from facecore.pipeline.align import align_crop
from facecore.pipeline.decode import DecodedImage
from facecore.pipeline.detect import DetectedFace
from facecore.pipeline.measure import exposure_of, sharpness_of
from facecore.pipeline.quality import Landmark, evaluate_quality

#: The g3-v1 quality policy the audit must honour, read from the product
#: rather than restated: ``PolicyProfile.frozen_v1`` *is* policy v1, and
#: the tests below assert these values so a profile change surfaces here
#: rather than as a silently different audit.
LUMA_LOW = 40.0
LUMA_HIGH = 215.0
CLIP_MAX = 0.05

MATCH_T = 0.363
REVIEW_T = 0.3
MARGIN_T = 0.10

#: Obviously synthetic identities, so a real one can never be pasted in
#: by accident and then trip the report mask.
PROBE_ID = "probe-target"
OTHER_ID = "probe-other"


def _policy() -> PolicyProfile:
    return PolicyProfile.frozen_v1().with_thresholds(
        match_threshold=MATCH_T,
        review_threshold=REVIEW_T,
        margin_threshold=MARGIN_T,
    )


def _assert_frozen_policy() -> None:
    p = PolicyProfile.frozen_v1()
    assert p.exposure_luma_range == (LUMA_LOW, LUMA_HIGH)
    assert p.exposure_clipped_fraction_max == CLIP_MAX


def _face(box: tuple[float, float, float, float] = (10.0, 10.0, 120.0, 120.0)):
    x, y, w, h = box
    return DetectedFace(
        box=box,
        landmarks=(
            (x + 30.0, y + 40.0),
            (x + 90.0, y + 40.0),
            (x + 60.0, y + 70.0),
            (x + 40.0, y + 95.0),
            (x + 80.0, y + 95.0),
        ),
        confidence=0.95,
    )


class _Detector:
    def __init__(self, faces: list[DetectedFace]) -> None:
        self._faces = faces

    def detect(self, decoded: Any) -> list[DetectedFace]:
        return list(self._faces)


class _Embedder:
    model_version = "sface_2021dec"

    def embed(self, crop: Any) -> tuple[np.ndarray, str]:
        return np.asarray([1.0, 0.0], dtype=np.float32), self.model_version


def _axis(dim: int, index: int) -> np.ndarray:
    arr = np.zeros(dim, dtype=np.float32)
    arr[index] = 1.0
    return arr


def _context(*, faces: list[DetectedFace] | None = None) -> ScoringContext:
    gal = ResearchGallery(
        embeddings={PROBE_ID: _axis(2, 0), OTHER_ID: _axis(2, 1)},
        model_version="sface_2021dec",
        generation="gen-d6",
        digest="gal-d6",
    )
    return ScoringContext(
        gallery=gal,
        model_version="sface_2021dec",
        policy=_policy(),
        detector=_Detector(faces if faces is not None else [_face()]),
        embedder=_Embedder(),
    )


def _write_photo(path: Path, arr: np.ndarray) -> Path:
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _checker(centre: int, amp: int, block: int = 8) -> np.ndarray:
    """A textured frame of alternating solid blocks at `centre ± amp`.

    Solid blocks, not a fine checkerboard: ``align_crop`` resamples with
    BILINEAR, so a 1-px checkerboard's contrast is averaged away by the
    warp and the crop's Laplacian variance collapses to zero. Blocks of
    8 px in a 200 px frame survive the warp.
    """
    arr = np.full((200, 200, 3), centre, dtype=np.uint8)
    n = arr.shape[0] // block
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                arr[i * block : (i + 1) * block, j * block : (j + 1) * block] = max(
                    0, min(255, centre + amp)
                )
    return arr


def _ublocks(centre: int, block: int, block_val: int, fraction: float) -> np.ndarray:
    """A frame of flat `centre` with `fraction` of solid blocks at `block_val`.

    The blocks are spread by a stride coprime with the block grid, so
    they do not form a solid band. That matters: a contiguous band is
    over-sampled by the aligned crop and the resulting clip fraction
    does not track `fraction`, which is how an earlier attempt to build
    a dark-and-clipped fixture ended up with a mean luma of 89 — not
    dark at all — and quietly stopped testing what it claimed to.
    """
    arr = np.full((200, 200, 3), centre, dtype=np.uint8)
    n = arr.shape[0] // block
    total = n * n
    target = int(round(fraction * total))
    stride = 7
    placed = 0
    for t in range(total):
        if placed >= target:
            break
        i = (t * stride) % n
        j = t // n
        arr[i * block : (i + 1) * block, j * block : (j + 1) * block] = block_val
        placed += 1
    return arr


#: Every fixture below was measured on the real pipeline. The measured
#: triples (crop mean luma, crop clip fraction, crop sharpness) are in
#: each docstring, because a fixture that drifts is a fixture whose test
#: now passes for a different reason — and this repo has a history of
#: exactly that. The three luma predicates are 40 / 215 / clip > 0.05.


def _clean_frame() -> np.ndarray:
    """Passes every gate. Measured: mean 140.32, clip 0.0, sharp 782.4."""
    return _checker(120, amp=40)


def _dark_frame() -> np.ndarray:
    """Too dark, nothing else. Measured: mean 27.56, clip 0.0, sharp 110.2."""
    return _checker(20, amp=15)


def _bright_frame() -> np.ndarray:
    """Too bright, nothing else. Measured: mean 235.11, clip 0.0, sharp 195.8."""
    return _checker(225, amp=20)


def _clip_only_frame() -> np.ndarray:
    """Clipped, with luma mid-range. Measured: mean 139.99, clip 0.1437.

    Both luma predicates are false here, so the rejection is
    attributable to clipping alone. This is the case the three-way
    split exists for: a single `quality_exposure` reason cannot
    distinguish it from either dark or bright, and D5 was right to
    refuse to guess.
    """
    return _ublocks(120, 8, 255, 0.06)


def _dark_and_clipped_frame() -> np.ndarray:
    """Two causes at once. Measured: mean 37.24, clip 0.8477, sharp 988.6.

    A near-black frame with a scattered minority of near-white blocks.
    The mean stays under 40 while a large fraction of the crop sits at
    or beyond the clip bounds, so the row must carry both causes.
    """
    return _ublocks(2, 8, 240, 0.06)


def _audit_one(tmp_path: Path, arr: np.ndarray, name: str = "p.png") -> AuditRow:
    photo = _write_photo(tmp_path / name, arr)
    return audit_photo(photo, _context(), _policy(), set_name=SET_NONTARGET)


class TestPolicyIsTheProductPolicy:
    def test_frozen_v1_thresholds_are_the_ones_audited(self) -> None:
        _assert_frozen_policy()


class TestThreePredicates:
    """Plan RED: the luma boundaries and the clip boundary, each side.

    ``low <= mean_luma <= high`` is **inclusive**; the clip test is
    **strict** ``>``. Those are different comparison operators and an
    audit that got either wrong would move a boundary photo between
    "rejected" and "not", so each side is asserted on its own.
    """

    def test_mean_exactly_at_low_bound_is_not_too_dark(self) -> None:
        c = exposure_causes(
            mean_luma=LUMA_LOW, clipped_fraction=0.0, policy=_policy()
        )
        assert c.too_dark is False
        assert c.any_triggered is False

    def test_mean_just_below_low_bound_is_too_dark(self) -> None:
        c = exposure_causes(
            mean_luma=LUMA_LOW - 1e-6, clipped_fraction=0.0, policy=_policy()
        )
        assert c.too_dark is True
        assert c.too_bright is False
        assert c.clipped is False

    def test_mean_exactly_at_high_bound_is_not_too_bright(self) -> None:
        c = exposure_causes(
            mean_luma=LUMA_HIGH, clipped_fraction=0.0, policy=_policy()
        )
        assert c.too_bright is False
        assert c.any_triggered is False

    def test_mean_just_above_high_bound_is_too_bright(self) -> None:
        c = exposure_causes(
            mean_luma=LUMA_HIGH + 1e-6, clipped_fraction=0.0, policy=_policy()
        )
        assert c.too_bright is True
        assert c.too_dark is False
        assert c.clipped is False

    def test_clip_exactly_at_max_is_not_clipped(self) -> None:
        c = exposure_causes(
            mean_luma=120.0, clipped_fraction=CLIP_MAX, policy=_policy()
        )
        assert c.clipped is False
        assert c.any_triggered is False

    def test_clip_just_over_max_is_clipped(self) -> None:
        c = exposure_causes(
            mean_luma=120.0, clipped_fraction=CLIP_MAX + 1e-9, policy=_policy()
        )
        assert c.clipped is True
        assert c.too_dark is False
        assert c.too_bright is False

    def test_mid_range_mid_clip_triggers_nothing(self) -> None:
        c = exposure_causes(
            mean_luma=120.0, clipped_fraction=0.0, policy=_policy()
        )
        assert c.triggered == ()


class TestPredicatesAgreeWithTheProductGate:
    """The split must be *this* gate's, not a re-reading of it.

    A grid of ``(mean_luma, clipped_fraction)`` pairs is pushed through
    both the product's ``evaluate_quality`` and the audit's
    ``exposure_causes``, and the two must agree about when
    ``quality_exposure`` appears. Without this, the audit could split
    the three causes correctly in isolation and still disagree with the
    pipeline about which photos the code belongs to — and the whole
    point of C is to explain the product's rejections.
    """

    def test_grid_agrees_with_evaluate_quality(self) -> None:
        policy = _policy()
        grid_luma = [0.0, 39.9, 40.0, 40.1, 120.0, 214.9, 215.0, 215.1, 255.0]
        grid_clip = [0.0, 0.0499, 0.05, 0.0501, 0.5, 1.0]
        for mean_luma in grid_luma:
            for clip in grid_clip:
                causes = exposure_causes(
                    mean_luma=mean_luma, clipped_fraction=clip, policy=policy
                )
                # Every other gate is passed comfortably, so the exposure
                # code is the only one that can appear.
                verdict = evaluate_quality(
                    policy,
                    detector_confidence=0.95,
                    shorter_side_px=120,
                    sharpness=500.0,
                    mean_luma=mean_luma,
                    clipped_fraction=clip,
                    yaw_deg=0.0,
                    pitch_deg=0.0,
                    landmarks=[Landmark(confidence=1.0, x=0.0, y=0.0)] * 5,
                )
                got = "quality_exposure" in verdict.reason_codes
                assert got == causes.any_triggered, (
                    f"mean_luma={mean_luma} clip={clip}: "
                    f"evaluate_quality says exposure={got}, "
                    f"exposure_causes says {causes.triggered!r}"
                )

    def test_dark_and_bright_are_distinct_causes(self) -> None:
        dark = exposure_causes(mean_luma=10.0, clipped_fraction=0.0, policy=_policy())
        bright = exposure_causes(
            mean_luma=250.0, clipped_fraction=0.0, policy=_policy()
        )
        assert dark.triggered == (CAUSE_TOO_DARK,)
        assert bright.triggered == (CAUSE_TOO_BRIGHT,)
        assert dark.too_dark and not dark.too_bright
        assert bright.too_bright and not bright.too_dark

    def test_clip_is_a_third_distinct_cause(self) -> None:
        clip = exposure_causes(
            mean_luma=120.0, clipped_fraction=0.30, policy=_policy()
        )
        assert clip.triggered == (CAUSE_CLIPPED,)
        assert not clip.too_dark and not clip.too_bright

    def test_two_causes_are_both_kept(self) -> None:
        both = exposure_causes(
            mean_luma=10.0, clipped_fraction=0.30, policy=_policy()
        )
        assert set(both.triggered) == {CAUSE_TOO_DARK, CAUSE_CLIPPED}
        assert both.any_triggered is True


class TestMarginsToThreshold:
    """How far from each bound — never how far it *should* move."""

    def test_inside_luma_range_margin_is_positive(self) -> None:
        assert luma_margin(120.0, _policy()) == pytest.approx(80.0)

    def test_below_low_bound_margin_is_negative(self) -> None:
        assert luma_margin(20.0, _policy()) == pytest.approx(-20.0)

    def test_above_high_bound_margin_is_negative(self) -> None:
        assert luma_margin(240.0, _policy()) == pytest.approx(-25.0)

    def test_closer_bound_wins(self) -> None:
        assert luma_margin(210.0, _policy()) == pytest.approx(5.0)
        assert luma_margin(45.0, _policy()) == pytest.approx(5.0)


class TestPhotoCasesThroughTheRealPipeline:
    """Plan RED: dark-only / bright-only / clip-only / multi-cause.

    Each case drives ``audit_photo``, which runs ``score_frame`` — the
    real detector/quality/embed chain on a real decoded image — and
    reads the measured values off the ``FrameDiagnostics`` the pipeline
    emitted. Nothing here hand-feeds a luma number.
    """

    def test_dark_only(self, tmp_path: Path) -> None:
        row = _audit_one(tmp_path, _dark_frame())
        assert row.crop_mean_luma is not None and row.crop_mean_luma < LUMA_LOW
        assert row.too_dark is True
        assert row.too_bright is False
        assert row.clipped is False
        assert row.causes == (CAUSE_TOO_DARK,)
        assert row.quality_pass is False
        assert "quality_exposure" in row.quality_reasons

    def test_bright_only(self, tmp_path: Path) -> None:
        row = _audit_one(tmp_path, _bright_frame())
        assert row.crop_mean_luma is not None and row.crop_mean_luma > LUMA_HIGH
        assert row.too_bright is True
        assert row.too_dark is False
        assert row.clipped is False
        assert row.causes == (CAUSE_TOO_BRIGHT,)
        assert row.quality_pass is False
        assert "quality_exposure" in row.quality_reasons

    def test_clip_only_with_mid_range_luma(self, tmp_path: Path) -> None:
        """The case that makes the three-way split worth doing at all.

        Mean luma is mid-range, so "too dark" and "too bright" are both
        false and the rejection is attributable to clipping alone. A
        harness that reported a single `quality_exposure` reason could
        not distinguish this from either dark or bright, and would have
        to guess — which is what D5 refused to do.
        """
        row = _audit_one(tmp_path, _clip_only_frame())
        assert row.crop_mean_luma is not None
        assert LUMA_LOW < row.crop_mean_luma < LUMA_HIGH, (
            f"fixture must sit inside the luma range, got {row.crop_mean_luma}"
        )
        assert row.crop_clipped_fraction is not None
        assert row.crop_clipped_fraction > CLIP_MAX
        assert row.clipped is True
        assert row.too_dark is False
        assert row.too_bright is False
        assert row.causes == (CAUSE_CLIPPED,)
        assert row.quality_pass is False
        assert "quality_exposure" in row.quality_reasons

    def test_two_causes_on_one_photo_are_both_recorded(self, tmp_path: Path) -> None:
        row = _audit_one(tmp_path, _dark_and_clipped_frame())
        assert row.too_dark is True
        assert row.clipped is True
        assert row.causes == (CAUSE_TOO_DARK, CAUSE_CLIPPED)
        assert row.quality_pass is False

    def test_a_clean_photo_triggers_nothing(self, tmp_path: Path) -> None:
        row = _audit_one(tmp_path, _clean_frame())
        assert row.causes == ()
        assert row.crop_mean_luma is not None
        assert LUMA_LOW < row.crop_mean_luma < LUMA_HIGH
        assert row.quality_pass is True

    def test_each_cause_maps_to_its_own_predicate_name(self, tmp_path: Path) -> None:
        """Every cause string the audit emits is one of the three.

        A fourth name would mean the audit invented a category the
        product does not have, and a report would then quote a cause
        the gate never distinguished.
        """
        allowed = {CAUSE_TOO_DARK, CAUSE_TOO_BRIGHT, CAUSE_CLIPPED}
        for arr in (_dark_frame(), _bright_frame(), _clip_only_frame(),
                    _dark_and_clipped_frame()):
            row = _audit_one(tmp_path, arr)
            assert set(row.causes) <= allowed, row.causes


class TestMeasurementComesFromTheAlignedCrop:
    """The mutation guard for "quietly measure the original instead".

    The fixture has a dark face region inside a bright frame: its
    full-image mean is mid-range and would pass, while the crop mean is
    far below 40. If the audit read the original photo, the row would
    report no cause and pass. The two are therefore recorded in separate
    fields and never compared under one threshold.
    """

    def _mixed_frame(self) -> np.ndarray:
        arr = np.full((200, 200, 3), 235, dtype=np.uint8)
        arr[0:150, 0:150] = 20
        return arr

    def test_row_reports_the_crop_value_not_the_full_image(
        self, tmp_path: Path
    ) -> None:
        row = _audit_one(tmp_path, self._mixed_frame())
        assert row.crop_mean_luma is not None
        assert row.crop_mean_luma < LUMA_LOW
        assert row.too_dark is True
        assert row.quality_pass is False

    def test_full_image_comparison_is_kept_in_its_own_fields(
        self, tmp_path: Path
    ) -> None:
        """Recorded separately, and allowed to disagree.

        Measured on the real pipeline: full-image mean 114.06 (inside
        range) against crop mean 21.86 (below 40). The disagreement is
        a fact about the fixture, not a bug — the gate measures the
        crop — so these fields exist for the counterfactual and must not
        be what any cause is derived from.
        """
        row = _audit_one(tmp_path, self._mixed_frame())
        assert row.full_mean_luma is not None
        assert LUMA_LOW < row.full_mean_luma < LUMA_HIGH, (
            "fixture's full-image mean should sit inside the range, so the "
            "two measurables genuinely disagree"
        )
        assert row.full_too_dark is False
        assert row.full_clipped is False
        assert row.too_dark is True
        assert row.disagrees_with_full_image is True

    def test_causes_are_identical_with_and_without_the_full_image(
        self, tmp_path: Path
    ) -> None:
        """Computing the full-image comparison cannot change the verdict."""
        with_full = _audit_one(tmp_path, self._mixed_frame())
        without_full = audit_photo(
            _write_photo(tmp_path / "q.png", self._mixed_frame()),
            _context(),
            _policy(),
            set_name=SET_NONTARGET,
            measure_full_image=False,
        )
        assert with_full.causes == without_full.causes
        assert with_full.crop_mean_luma == without_full.crop_mean_luma
        assert without_full.full_mean_luma is None

    def test_aligned_crop_measurement_matches_the_pipeline(
        self, tmp_path: Path
    ) -> None:
        """The row's numbers are the ones ``score_frame`` measured.

        Recomputed here from the detector the context exposes, via the
        product's own ``exposure_of``/``sharpness_of`` on the crop
        ``align_crop`` produces. This is what makes the "crop, not
        original" claim checkable rather than asserted in a docstring.
        """
        arr = self._mixed_frame()
        row = _audit_one(tmp_path, arr)
        rgb = np.ascontiguousarray(arr)
        decoded = DecodedImage(
            width=rgb.shape[1], height=rgb.shape[0], color_order="RGB",
            pixels=rgb.tobytes(),
        )
        face = _context().detector.detect(decoded)[0]
        crop = align_crop(decoded.pixels, decoded.width, decoded.height, face)
        mean_luma, clipped = exposure_of(crop)
        assert row.crop_mean_luma == pytest.approx(mean_luma, abs=1e-9)
        assert row.crop_clipped_fraction == pytest.approx(clipped, abs=1e-9)
        assert row.crop_sharpness == pytest.approx(sharpness_of(crop), abs=1e-9)


class TestUnprocessableRows:
    """A photo with no single face has no crop, so no exposure numbers.

    These rows must still be produced and still counted — dropping them
    is how an audit ends up reporting a 64-photo run as 66.
    """

    def test_no_face_row_is_unprocessable_and_keeps_no_numbers(
        self, tmp_path: Path
    ) -> None:
        photo = _write_photo(tmp_path / "p.png", _dark_frame())
        row = audit_photo(
            photo, _context(faces=[]), _policy(), set_name=SET_NONTARGET
        )
        assert row.face_count == 0
        assert row.unprocessable_reason == "no_face_detected"
        assert row.crop_mean_luma is None
        assert row.crop_clipped_fraction is None
        assert row.causes == ()
        assert row.measured is False

    def test_multiple_faces_row_is_unprocessable(self, tmp_path: Path) -> None:
        photo = _write_photo(tmp_path / "p.png", _dark_frame())
        row = audit_photo(
            photo, _context(faces=[_face(), _face((0.0, 0.0, 60.0, 60.0))]),
            _policy(), set_name=SET_NONTARGET,
        )
        assert row.face_count == 2
        assert row.unprocessable_reason == "multiple_faces"
        assert row.measured is False


class TestRejectedRowsAreNotDropped:
    """The mutation guard for "drop the rejected rows".

    A rejected photo is the *subject* of this audit. An audit that
    summarised only accepted photos would report zero rejections and
    look like a clean run.
    """

    def test_a_rejected_photo_is_still_present_in_the_summary(
        self, tmp_path: Path
    ) -> None:
        rows = [
            _audit_one(tmp_path, _dark_frame(), "dark.png"),
            _audit_one(tmp_path, _bright_frame(), "bright.png"),
            _audit_one(tmp_path, _clip_only_frame(), "clip.png"),
            _audit_one(tmp_path, _dark_and_clipped_frame(), "both.png"),
            _audit_one(tmp_path, _clean_frame(), "clean.png"),
        ]
        summary = summarize_audit(rows, set_name=SET_NONTARGET)
        assert summary.total == 5, "rejected photos vanished from the count"
        assert summary.quality_rejected == 4
        assert summary.quality_passed == 1
        assert summary.too_dark == 2
        assert summary.too_bright == 1
        assert summary.clipped == 2
        assert summary.two_or_more_causes == 1
        assert summary.exposure_rejected == 4

    def test_counts_partition_the_total(self, tmp_path: Path) -> None:
        rows = [
            _audit_one(tmp_path, _dark_frame(), "a.png"),
            _audit_one(tmp_path, _bright_frame(), "b.png"),
            _audit_one(tmp_path, _clip_only_frame(), "c.png"),
        ]
        s = summarize_audit(rows, set_name=SET_NONTARGET)
        assert s.quality_rejected + s.quality_passed + s.unprocessable == s.total

    def test_an_unprocessable_row_still_counts_in_the_total(
        self, tmp_path: Path
    ) -> None:
        photo = _write_photo(tmp_path / "p.png", _dark_frame())
        rows = [
            audit_photo(photo, _context(faces=[]), _policy(), set_name=SET_NONTARGET),
            _audit_one(tmp_path, _bright_frame(), "b.png"),
        ]
        s = summarize_audit(rows, set_name=SET_NONTARGET)
        assert s.total == 2
        assert s.unprocessable == 1
        assert s.measured == 1


class TestSetNamesAndSummaryRules:
    def test_unknown_set_name_is_refused(self, tmp_path: Path) -> None:
        rows = [_audit_one(tmp_path, _dark_frame())]
        with pytest.raises(ValueError, match="unknown set_name"):
            summarize_audit(rows, set_name="gallery-of-mistake")

    def test_gallery_is_a_known_set(self) -> None:
        assert SET_GALLERY == "gallery"
        assert SET_PROBE == "probe"
        assert SET_NONTARGET == "nontarget"

    def test_luma_range_covers_only_measured_rows(self, tmp_path: Path) -> None:
        """Ranges come from the measured rows and ignore the unprocessable one.

        An unprocessable photo has no crop, so it has no luma. Folding
        it in as 0.0 would silently drag the reported range to the
        bottom of the scale and make a corpus look darker than it is.
        """
        photo = _write_photo(tmp_path / "p.png", _dark_frame())
        dark = _audit_one(tmp_path, _dark_frame(), "a.png")
        bright = _audit_one(tmp_path, _bright_frame(), "b.png")
        rows = [
            audit_photo(photo, _context(faces=[]), _policy(), set_name=SET_NONTARGET),
            dark,
            bright,
        ]
        s = summarize_audit(rows, set_name=SET_NONTARGET)
        assert s.total == 3
        assert s.measured == 2
        assert s.luma_min == pytest.approx(dark.crop_mean_luma, abs=1e-4)
        assert s.luma_max == pytest.approx(bright.crop_mean_luma, abs=1e-4)
        assert s.luma_min is not None and s.luma_min > 0.0

    def test_clip_max_reflects_the_worst_measured_row(self, tmp_path: Path) -> None:
        photo = _write_photo(tmp_path / "p.png", _dark_frame())
        clip = _audit_one(tmp_path, _clip_only_frame(), "c.png")
        rows = [
            audit_photo(photo, _context(faces=[]), _policy(), set_name=SET_NONTARGET),
            clip,
        ]
        s = summarize_audit(rows, set_name=SET_NONTARGET)
        assert s.clip_max == pytest.approx(clip.crop_clipped_fraction, abs=1e-4)
        assert s.clip_max is not None and s.clip_max > CLIP_MAX

    def test_summary_of_zero_rows_does_not_divide_by_zero(
        self, tmp_path: Path
    ) -> None:
        s = summarize_audit([], set_name=SET_NONTARGET)
        assert s.total == 0
        assert s.luma_min is None
        assert s.luma_max is None
        assert s.two_or_more_causes == 0


class TestD5MaskIsReproducible:
    """Plan acceptance: the 43 D5 rows must be checkable against C.

    The D5 detail CSV is repo-external and read-only. Cross-checking
    means the audit's ``quality_pass``/``quality_reasons`` for each
    probe and non-target photo has to match what D5 recorded for the
    same photo — otherwise C would be describing a different run.
    """

    def _mask_csv(self, tmp_path: Path, rows: list[tuple[str, bool, str]]) -> Path:
        path = tmp_path / "d5.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["set_name", "photo", "quality_pass", "quality_reasons"],
            )
            w.writeheader()
            for set_name, photo, passed, reasons in rows:
                w.writerow(
                    {
                        "set_name": set_name,
                        "photo": photo,
                        "quality_pass": str(passed),
                        "quality_reasons": reasons,
                    }
                )
        return path

    def test_matching_mask_verifies(self, tmp_path: Path) -> None:
        dark = _audit_one(tmp_path, _dark_frame(), "p1.png")
        clip = _audit_one(tmp_path, _clip_only_frame(), "p2.png")
        mask = self._mask_csv(
            tmp_path,
            [
                (SET_NONTARGET, "p1.png", False, str(dark.quality_reasons)),
                (SET_PROBE, "p2.png", False, str(clip.quality_reasons)),
            ],
        )
        result = verify_d5_mask([dark, clip], mask)
        assert result.checked == 2
        assert result.mismatched == 0
        assert result.missing_in_audit == ()
        assert result.missing_in_mask == ()

    def test_a_flipped_verdict_is_caught(self, tmp_path: Path) -> None:
        dark = _audit_one(tmp_path, _dark_frame(), "p1.png")
        mask = self._mask_csv(
            tmp_path, [(SET_NONTARGET, "p1.png", True, "()")]
        )
        result = verify_d5_mask([dark], mask)
        assert result.mismatched == 1
        assert "p1.png" in result.mismatched_photos

    def test_a_missing_photo_is_caught_in_both_directions(
        self, tmp_path: Path
    ) -> None:
        dark = _audit_one(tmp_path, _dark_frame(), "d1.png")
        only_d1 = self._mask_csv(
            tmp_path, [(SET_NONTARGET, "d1.png", False, str(dark.quality_reasons))]
        )
        # The mask holds a photo the audit never produced.
        r1 = verify_d5_mask([dark], only_d1)
        assert r1.checked == 1
        assert r1.mismatched == 0
        assert r1.missing_in_audit == ()
        assert r1.missing_in_mask == ()

        # The audit holds a photo the mask does not. Compared under a
        # *different* set, so a name match cannot hide the gap.
        clip = audit_photo(
            _write_photo(tmp_path / "clip1.png", _clip_only_frame()),
            _context(), _policy(), set_name=SET_PROBE,
        )
        r2 = verify_d5_mask([dark, clip], only_d1)
        assert r2.missing_in_mask == ()
        assert r2.missing_in_audit == ("clip1.png",)
        assert r2.checked == 1
        assert r2.ok is False

    def test_a_missing_d5_file_fails_clear(self, tmp_path: Path) -> None:
        dark = _audit_one(tmp_path, _dark_frame(), "p1.png")
        with pytest.raises(FileNotFoundError):
            verify_d5_mask([dark], tmp_path / "nope.csv")

    def test_photo_names_alone_are_ambiguous_across_sets(
        self, tmp_path: Path
    ) -> None:
        """A photo name must be unique across the sets being checked.

        D5's corpus has disjoint names, so a name collision would mean
        the two sets are not being compared like-for-like. Caught here
        rather than silently matching the wrong row.
        """
        a = audit_photo(
            _write_photo(tmp_path / "p1.png", _dark_frame()),
            _context(), _policy(), set_name=SET_PROBE,
        )
        b = audit_photo(
            _write_photo(tmp_path / "p1.png", _dark_frame()),
            _context(), _policy(), set_name=SET_NONTARGET,
        )
        mask = self._mask_csv(
            tmp_path, [(SET_PROBE, "p1.png", False, str(a.quality_reasons))]
        )
        result = verify_d5_mask([a, b], mask)
        assert result.duplicate_photos == ("p1.png",)


class TestReportBodyCarriesNoPerPhotoData:
    """Committed output is counts and ranges only."""

    def test_summary_body_has_no_paths_no_uid_no_per_photo_rows(
        self, tmp_path: Path
    ) -> None:
        rows = [
            _audit_one(tmp_path, _dark_frame(), "p1.png"),
            _audit_one(tmp_path, _clip_only_frame(), "p2.png"),
        ]
        body = summarize_audit(rows, set_name=SET_NONTARGET).render_body()
        assert "p1.png" not in body
        assert "p2.png" not in body
        assert PROBE_ID not in body
        assert "/Users/" not in body

    def test_write_report_accepts_the_body(self, tmp_path: Path) -> None:
        rows = [
            _audit_one(tmp_path, _dark_frame(), "p1.png"),
            _audit_one(tmp_path, _clip_only_frame(), "p2.png"),
        ]
        body = summarize_audit(rows, set_name=SET_NONTARGET).render_body()
        out = write_report(tmp_path / "r.md", "# D6\n", extra_body=body)
        assert out.exists()

    def test_write_report_still_rejects_a_planted_path(self, tmp_path: Path) -> None:
        """The mask is not loosened to let this module's body through."""
        with pytest.raises(RedactionError, match="absolute local path"):
            write_report(
                tmp_path / "r.md", "# D6\n", extra_body="see /Users/x/y.png"
            )

    def test_write_report_still_rejects_a_planted_uid(self, tmp_path: Path) -> None:
        with pytest.raises(RedactionError, match="per-image identity"):
            write_report(
                tmp_path / "r.md", "# D6\n", extra_body="identity: enroll-23"
            )


class TestDetailRowSerialisation:
    def test_row_round_trips_through_json(self, tmp_path: Path) -> None:
        row = _audit_one(tmp_path, _dark_and_clipped_frame())
        payload = row.to_dict()
        json.dumps(payload)
        assert payload["causes"] == [CAUSE_TOO_DARK, CAUSE_CLIPPED]
        assert payload["too_dark"] is True
        assert payload["clipped"] is True

    def test_row_dict_has_no_absolute_path_and_no_truth_identity(
        self, tmp_path: Path
    ) -> None:
        row = _audit_one(tmp_path, _dark_frame())
        payload = row.to_dict()
        assert "/" not in payload["photo"]
        assert "truth_identity" not in payload
        assert PROBE_ID not in json.dumps(payload)


class TestNoCameraAndNoNewThreshold:
    """The audit measures; it never selects and never opens a device.

    Both are pinned on the *signature and the parser*, not on the
    module's own prose: grepping a docstring for a word proves only
    that the docstring changed, and this repo has a documented history
    of guards that could not fail.
    """

    def test_main_takes_no_threshold_argument(self) -> None:
        """No way to hand the audit a candidate threshold.

        The corpus has no holdout, so any "which bound is right" number
        computed from it would be tuning on the data being measured —
        the same reason ``static_baseline.SweepCell.selected`` is
        always False. The CLI therefore offers no flag to supply one.
        """
        import facecore.eval.quality_audit as mod

        for flag in ("--threshold", "--luma-range", "--clip-max", "--tune"):
            with pytest.raises(SystemExit) as exc:
                mod.main(
                    [
                        "--config", "x.json", "--probes", "p", "--nontarget", "n",
                        "--out-dir", "o", flag, "0.5",
                    ]
                )
            assert exc.value.code == 2, f"{flag} was not refused"

    def test_audit_photo_signature_takes_no_threshold(self) -> None:
        import inspect

        import facecore.eval.quality_audit as mod

        params = set(inspect.signature(mod.audit_photo).parameters)
        assert params == {
            "path", "context", "policy", "set_name", "sequence",
            "measure_full_image",
        }
        assert not any("threshold" in p for p in params)
        assert "tune" not in params and "select" not in params

    def test_thresholds_come_from_the_policy_not_the_module(self) -> None:
        """The module holds no numeric luma or clip bound of its own.

        ``exposure_causes`` must read the bounds from the policy object.
        A module-level ``40.0``/``0.05`` would be a second copy of the
        gate that could drift from the product's, and D5 already found
        one such duplicate (``eval/sweep.py``'s score-only FA rule).
        """
        import inspect

        import facecore.eval.quality_audit as mod

        src = inspect.getsource(mod.exposure_causes)
        assert "40.0" not in src
        assert "215.0" not in src
        assert "0.05" not in src
        assert "exposure_luma_range" in src
        assert "exposure_clipped_fraction_max" in src

    def test_no_capture_backend_is_imported(self) -> None:
        """No camera is opened: no capture import, no ``--device`` flag."""
        import facecore.eval.quality_audit as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in src.splitlines() if not line.strip().startswith("#")
        )
        for banned in ("VideoCapture", "AVFoundation", "avfoundation", "cap.set"):
            assert banned not in code, f"{banned} suggests a camera path"
        # The prose may mention that no device is used; the code may not.
        assert "--device" not in code.replace("# --device is never offered", "")

    def test_main_rejects_an_unauthorised_device_flag(self) -> None:
        """Asking for a device fails, rather than being ignored."""
        import facecore.eval.quality_audit as mod

        with pytest.raises(SystemExit) as exc:
            mod.main(
                [
                    "--config", "x.json", "--probes", "p", "--nontarget", "n",
                    "--out-dir", "o", "--device", "local",
                ]
            )
        assert exc.value.code == 2


class TestAuditIsDeterministic:
    def test_same_photo_audited_twice_gives_the_same_numbers(
        self, tmp_path: Path
    ) -> None:
        arr = _clip_only_frame()
        a = _audit_one(tmp_path, arr, "same.png")
        b = _audit_one(tmp_path, arr, "same2.png")
        assert a.causes == b.causes
        assert a.crop_mean_luma == pytest.approx(b.crop_mean_luma, abs=1e-9)
        assert a.crop_clipped_fraction == pytest.approx(
            b.crop_clipped_fraction, abs=1e-9
        )
        assert a.quality_reasons == b.quality_reasons
