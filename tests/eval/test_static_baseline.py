"""D5 PR-A: the static-baseline harness's counting semantics, test-first.

The plan (`docs/plans/2026-09-30-d5-static-baseline-plan.md`, Task 1)
names the failure mode this file is built to avoid, as Highest risk #1:

    the band / FA tests only exercise a helper the harness itself
    wrote, and never reach `score_frame`'s real output path.

So **every assertion below drives `score_photo`**, which is the function
Task 2 will actually call, and every one of them goes through the live
`score_frame` with a synthetic detector and embedder. Nothing here calls
a band-classification helper directly with hand-fed scores. That is the
whole point: a test of `_band()` proves only that `_band()` matches its
own author's reading of the rule, not that a photo lands in the right
bucket.

The synthetic context is cheap on purpose: a 2-D "gallery" of unit
vectors, a detector that returns exactly the boxes the test asks for,
and an embedder that returns the probe vector it was told to. Real
weights, real photos and a real camera are all out of scope for PR-A.

The rules under test, copied from live `session.py:773-784`:

    if runner_up is None:                      -> unknown
    elif score >= match and margin >= margin:   -> matched
    elif score >= review:                       -> review
    else:                                       -> unknown

and the plan's counting rules, which are the part D5 actually reports:

- FA (non-target wrongly accepted) means **matched**, i.e. score AND
  margin. `eval/sweep.py:50-52` decides FA on score alone; using it here
  would silently disagree with what the App does.
- A rejected photo stays in the denominator. Dropping it is how a
  harness makes a 43-photo run look better than it was.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from facecore.contracts.policy import PolicyProfile
from facecore.errors import RedactionError
from facecore.eval.nontarget_fa import REAL_MARGIN_GRID, REAL_MATCH_GRID
from facecore.eval.report import write_report
from facecore.eval.static_baseline import (
    PhotoResult,
    classify_nontarget,
    render_report_body,
    score_photo,
    summarize,
    sweep_live_semantics,
)
from facecore.live.contracts import FramePacket, ResearchProfile
from facecore.live.frame_pipeline import (
    ResearchGallery,
    ScoringContext,
    score_frame,
)
from facecore.pipeline.detect import DetectedFace

#: The g3-v1 thresholds the harness must honour (profiles/g3-v1.json).
#: Frozen: D5 may present a sweep but may not pick new thresholds.
MATCH_T = 0.363
REVIEW_T = 0.3
MARGIN_T = 0.10

#: Placeholder identities. Kept obviously synthetic so a real one can
#: never be pasted in by accident and then trip the report mask.
PROBE_ID = "probe-target"
OTHER_ID = "probe-other"


def _profile() -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="d5-test",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=26,
        queue_limit=1,
        required_support=3,
        min_support_interval_ms=200,
        match_threshold=MATCH_T,
        review_threshold=REVIEW_T,
        margin_threshold=MARGIN_T,
        detector_version="det-d5",
        quality_policy_version="qual-d5",
        continuity_max_center_delta_ratio=0.5,
    )


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
    """Returns exactly the faces the test declares, then nothing else."""

    def __init__(self, faces: list[DetectedFace]) -> None:
        self._faces = faces

    def detect(self, decoded: Any) -> list[DetectedFace]:
        return list(self._faces)


class _Embedder:
    """Returns the probe vector the test declared, and the model version."""

    model_version = "sface_2021dec"

    def __init__(self, probe_vec: np.ndarray) -> None:
        self._probe_vec = probe_vec

    def embed(self, crop: Any) -> tuple[np.ndarray, str]:
        return self._probe_vec, self.model_version


def _probe_for(top: float, runner_up: float) -> np.ndarray:
    """A unit probe whose cosine scores are exactly `(top, runner_up)`.

    Held against two orthogonal gallery identities, the cosine of a unit
    probe with a unit basis vector is just that coordinate — so a probe
    built as `(top, runner_up, sqrt(1 - top² - runner_up²))` lands the two
    scores exactly where the test asks. The third coordinate is real only
    when `top² + runner_up² ≤ 1`, which holds for every case here; the
    helper raises rather than returning a probe that would silently score
    something else.

    That identity is what lets these tests state exact expected scores
    instead of "roughly 0.4", and it means a drifted fixture surfaces as a
    wrong score rather than as a mysteriously wrong band.
    """
    remaining = 1.0 - top * top - runner_up * runner_up
    if remaining < 0.0:
        raise ValueError(
            f"no unit vector scores exactly ({top}, {runner_up}) against two "
            "orthogonal identities; that pair is impossible"
        )
    arr = np.asarray([top, runner_up, np.sqrt(remaining)], dtype=np.float32)
    return arr / float(np.linalg.norm(arr))


def _axis(dimension: int, index: int) -> np.ndarray:
    """A unit vector pointing along one axis of a `dimension`-D space."""
    arr = np.zeros(dimension, dtype=np.float32)
    arr[index] = 1.0
    return arr


def _context(
    *,
    gallery: dict[str, np.ndarray],
    probe_vec: np.ndarray,
    faces: list[DetectedFace] | None = None,
) -> ScoringContext:
    """A ScoringContext with a synthetic detector, embedder and gallery.

    Every gallery vector is made the same length as `probe_vec` — cosine
    refuses mismatched shapes, so the fixtures have to agree on the
    dimension rather than leaving that to chance.
    """
    dim = len(probe_vec)
    for ident, vec in gallery.items():
        assert len(vec) == dim, f"gallery vector {ident} has dim {len(vec)}, want {dim}"
    gal = ResearchGallery(
        embeddings={k: np.asarray(v, dtype=np.float32) for k, v in gallery.items()},
        model_version="sface_2021dec",
        generation="gen-d5",
        digest="gal-d5",
    )
    return ScoringContext(
        gallery=gal,
        model_version="sface_2021dec",
        policy=PolicyProfile.frozen_v1().with_thresholds(
            match_threshold=MATCH_T,
            review_threshold=REVIEW_T,
            margin_threshold=MARGIN_T,
        ),
        detector=_Detector(faces if faces is not None else [_face()]),
        embedder=_Embedder(probe_vec),
    )


def _two_identity_context(top: float, runner_up: float) -> ScoringContext:
    """Two orthogonal identities; the probe scores exactly (top, runner_up)."""
    probe = _probe_for(top, runner_up)
    dim = len(probe)
    return _context(
        gallery={PROBE_ID: _axis(dim, 0), OTHER_ID: _axis(dim, 1)},
        probe_vec=probe,
    )


def _single_identity_context() -> ScoringContext:
    """One identity only: the probe scores 1.0 and has no runner-up."""
    probe = np.asarray([1.0, 0.0], dtype=np.float32)
    return _context(gallery={PROBE_ID: _axis(2, 0)}, probe_vec=probe)


def _write_photo(path: Path, arr: np.ndarray) -> Path:
    """Write a real image file, because `score_photo` takes a path.

    Driving the actual file-I/O path matters: it is the one Task 2 runs,
    and it is where the PIL decode and the uint8 conversion live. A
    fixture that passed arrays in would skip exactly that.
    """
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _good_frame() -> np.ndarray:
    """A frame that clears the frozen-v1 quality gates.

    Same trick as `tests/live/test_frame_pipeline.py:112-120`: mid-grey
    base, a checkerboard for the Laplacian variance, and one asymmetric
    patch so the aligned crop is not perfectly uniform.
    """
    arr = np.full((200, 200, 3), 120, dtype=np.uint8)
    arr[::2, ::2] = 160
    arr[1::2, 1::2] = 80
    arr[20:60, 20:60, 0] = 180
    arr[20:60, 20:60, 1] = 60
    arr[20:60, 20:60, 2] = 100
    return arr


def _flat_frame() -> np.ndarray:
    """A frame whose crop cannot clear `sharpness_min=60` (flat grey)."""
    return np.full((200, 200, 3), 120, dtype=np.uint8)


class TestBandRuleReachedThroughScoreFrame:
    """Plan test 1: the band cases, each via the real code path.

    Each case states the top1/runner-up scores it wants and asserts the
    band `score_photo` produced. `score_photo` decodes a real image,
    calls `score_frame` — which runs the detector, the quality gate, the
    embedder and the cosine comparison — so a break anywhere in that
    chain turns the assertion red.
    """

    def _band_for(self, tmp_path: Path, top: float, runner_up: float) -> str:
        photo = _write_photo(tmp_path / "p.jpg", _good_frame())
        got = score_photo(
            photo,
            _two_identity_context(top, runner_up),
            _profile(),
            set_name="nontarget",
            truth_identity=None,
            sequence=1,
        )
        assert got.quality_pass, (
            f"fixture meant to pass quality did not: {got.quality_reasons!r}"
        )
        # Confirm the scores are the ones this case is about, so a
        # failure below can only mean the band rule, not a drifted fixture.
        assert got.l_top1_score == pytest.approx(top, abs=1e-2)
        assert got.l_top2_score == pytest.approx(runner_up, abs=1e-2)
        return got.l_band

    def test_score_over_match_but_margin_under_is_review_not_matched(
        self, tmp_path: Path
    ) -> None:
        """top1=0.50 (>= 0.363) with margin 0.05 (< 0.10) is `review`.

        This is the case that separates live's definition from
        `sweep.py`'s, which would call it matched on score alone.
        """
        assert self._band_for(tmp_path, 0.50, 0.45) == "review"

    def test_score_and_margin_both_over_threshold_is_matched(
        self, tmp_path: Path
    ) -> None:
        assert self._band_for(tmp_path, 0.40, 0.20) == "matched"

    def test_score_just_above_review_is_review(self, tmp_path: Path) -> None:
        assert self._band_for(tmp_path, 0.32, 0.10) == "review"

    def test_score_under_review_is_unknown(self, tmp_path: Path) -> None:
        assert self._band_for(tmp_path, 0.29, 0.10) == "unknown"

    def test_a_single_candidate_is_unknown_even_above_match(
        self, tmp_path: Path
    ) -> None:
        """`runner_up is None` outranks every threshold (session.py:773-775).

        With one gallery identity there is no margin to check, and live
        refuses the match on that ground before it ever looks at the
        score — so a perfect 1.0 against a lone identity is `unknown`,
        not `matched`.
        """
        photo = _write_photo(tmp_path / "p.jpg", _good_frame())
        ctx = _single_identity_context()
        got = score_photo(
            photo,
            ctx,
            _profile(),
            set_name="nontarget",
            truth_identity=None,
            sequence=1,
        )
        assert got.l_top1_score == pytest.approx(1.0, abs=1e-6)
        assert got.l_top2 is None
        assert got.l_band == "unknown", (
            "runner_up=None must short-circuit to unknown regardless of score; "
            f"got {got.l_band!r}"
        )


class TestQualityRejectionKeepsTheRBranch:
    """Plan test 2: a rejected photo scores nothing on L and still ranks on R.

    `score_frame` returns `identity_scores={}` when quality fails
    (frame_pipeline.py:509), so the App has nothing to show. The R
    branch exists precisely to recover the model's ranking for those
    photos, which is what makes "quality filtered it" separable from
    "the model could not rank it".
    """

    def test_l_branch_is_empty_and_r_branch_still_ranks(self, tmp_path: Path) -> None:
        photo = _write_photo(tmp_path / "flat.jpg", _flat_frame())
        ctx = _two_identity_context(0.80, 0.20)

        # Precondition, measured rather than assumed: this frame really
        # is quality-rejected by the live gate. Without it, the rest of
        # the test could pass for the wrong reason.
        obs = score_frame(
            FramePacket(sequence=1, captured_ns=0, rgb=_flat_frame()), ctx
        )
        assert obs.quality_pass is False, (
            "this fixture was meant to be quality-rejected; if it passes, "
            f"reasons were {obs.quality_reasons!r} and the test proves nothing"
        )

        got = score_photo(
            photo,
            ctx,
            _profile(),
            set_name="nontarget",
            truth_identity=None,
            sequence=1,
        )
        assert got.quality_pass is False
        assert got.quality_reasons, "a rejection must name why"
        assert got.l_band == "rejected"
        assert got.l_top1 is None
        assert got.l_top1_score is None
        assert got.l_top2 is None
        assert got.l_margin is None

        assert got.r_top1 is not None, (
            "the R branch exists to rank photos the quality gate removed; "
            "an all-None R branch makes the two branches indistinguishable"
        )
        assert got.r_top1_score == pytest.approx(0.80, abs=2e-2)
        assert got.r_band is not None
        assert got.stage_ms, "stage timings come from the diagnostic sink"


class TestFalseAcceptUsesTheLiveDefinition:
    """Plan test 3: FA means matched (score AND margin), not score alone."""

    def test_score_over_match_with_thin_margin_is_not_a_false_accept(self) -> None:
        """The discriminating row, built directly.

        `sweep.py:50-52` counts FA on `score >= match_t` alone, so it
        calls this one a false accept. Live does not: the 0.10 margin
        firewall rejects it first and the row lands in `review`. The
        harness must report that distinction, not fold it into a reject.

        The expected class is `review`, **not** `correct_reject` — the
        plan lists 正確拒絕 (`unknown`) and review as separate buckets,
        and collapsing them would hide exactly the rows where the model
        was close. What matters for FA is only that it is not
        `false_accept`.
        """
        result = PhotoResult(
            set_name="nontarget",
            photo="nt-01.jpg",
            truth_identity=None,
            face_count=1,
            unprocessable_reason=None,
            quality_pass=True,
            quality_reasons=(),
            l_top1=PROBE_ID,
            l_top1_score=0.40,
            l_top2=OTHER_ID,
            l_top2_score=0.35,
            l_margin=0.05,
            l_band="review",
            r_top1=PROBE_ID,
            r_top1_score=0.40,
            r_top2=OTHER_ID,
            r_top2_score=0.35,
            r_margin=0.05,
            r_band="review",
            stage_ms={},
        )
        assert classify_nontarget(result) != "false_accept", (
            "score 0.40 clears the 0.363 match gate but margin 0.05 does not "
            "clear the 0.10 firewall, so live says review -> not a false accept"
        )
        assert classify_nontarget(result) == "review"

        # Prove the two definitions really differ here, so this test
        # cannot pass merely because both agree by accident.
        sweep_says_fa = result.l_top1_score >= MATCH_T
        assert sweep_says_fa, "precondition: sweep's score-only rule says FA"
        assert result.l_band != "matched", "precondition: live does not"

    def test_thin_margin_reached_through_score_frame_is_not_a_false_accept(
        self, tmp_path: Path
    ) -> None:
        """Same rule, reached through the real pipeline instead of a literal.

        top1 0.40 clears the match gate; runner-up 0.35 leaves margin
        0.05, under the firewall. `score_photo` must land on `review`,
        and `classify_nontarget` must not call that an accept.
        """
        photo = _write_photo(tmp_path / "p.jpg", _good_frame())
        got = score_photo(
            photo,
            _two_identity_context(0.40, 0.35),
            _profile(),
            set_name="nontarget",
            truth_identity=None,
            sequence=1,
        )
        assert got.l_top1_score == pytest.approx(0.40, abs=2e-2)
        assert got.l_margin == pytest.approx(0.05, abs=2e-2)
        assert got.l_band == "review"
        assert got.l_top1_score >= MATCH_T, "precondition: score clears match"
        assert classify_nontarget(got) != "false_accept", (
            "a score that clears match with a margin under the firewall is "
            "not an accept; sweep's score-only rule would miscount this one"
        )
        assert classify_nontarget(got) == "review"

    def test_a_wide_margin_at_the_same_score_is_a_false_accept(
        self, tmp_path: Path
    ) -> None:
        """The contrast case: same 0.40 score, margin 0.20 -> matched -> FA.

        Without this, the test above could pass for a reason unrelated to
        the margin rule.
        """
        photo = _write_photo(tmp_path / "p.jpg", _good_frame())
        got = score_photo(
            photo,
            _two_identity_context(0.40, 0.20),
            _profile(),
            set_name="nontarget",
            truth_identity=None,
            sequence=1,
        )
        assert got.l_top1_score == pytest.approx(0.40, abs=2e-2)
        assert got.l_margin == pytest.approx(0.20, abs=2e-2)
        assert got.l_band == "matched"
        assert classify_nontarget(got) == "false_accept"


class TestRejectedPhotosStayInTheDenominator:
    """Plan test 4: the denominator is the photo count, not the scoreable count."""

    def test_three_photos_one_unprocessable_gives_denominator_three(self) -> None:
        results = [
            _matched_nontarget("nt-01.jpg"),
            _matched_nontarget("nt-02.jpg"),
            _unprocessable("nt-03.jpg"),
        ]
        summary = summarize(results, set_name="nontarget")
        assert summary.total == 3, (
            f"the denominator is {summary.total}; dropping the unprocessable "
            "photo would report 2/2 and read as a clean run"
        )
        assert summary.false_accepts == 2
        assert summary.unprocessable == 1
        assert (
            sum(
                [
                    summary.false_accepts,
                    summary.correct_rejects,
                    summary.review,
                    summary.quality_rejected,
                    summary.unprocessable,
                ]
            )
            == summary.total
        ), (
            "the buckets must partition the set: every photo lands in "
            "exactly one, so none can be silently dropped"
        )


class TestWrongIdentityIsNotACorrectAccept:
    """Plan test 5: a matched probe whose top1 is someone else is `wrong_identity`."""

    def test_matched_but_other_identity_counts_as_wrong_identity(self) -> None:
        good = _matched_probe("probe-01.jpg", truth=PROBE_ID, top1=PROBE_ID)
        wrong = _matched_probe("probe-02.jpg", truth=PROBE_ID, top1=OTHER_ID)
        summary = summarize([good, wrong], set_name="probe")
        assert summary.total == 2
        assert summary.correct_accepts == 1
        assert summary.wrong_identities == 1, (
            "a matched round naming the wrong person must not be counted as "
            "a correct accept; that is the failure mode a baseline exists "
            "to measure"
        )
        assert (summary.correct_accepts + summary.wrong_identities) == 2, (
            "the two matched photos must split between the two classes"
        )
        assert summary.correct_accepts != summary.total, (
            "if a misidentification were folded into a correct accept, the "
            "summary would read as a perfect 2/2 run"
        )

    def test_top1_ranking_is_counted_regardless_of_band(self) -> None:
        """Plan: `top1 排序正確` is band-independent and counted per branch."""
        base = _matched_probe("probe-03.jpg", truth=PROBE_ID, top1=PROBE_ID)
        review_row = replace(base, l_band="review", r_band="review")
        summary = summarize([review_row], set_name="probe")
        assert summary.l_top1_correct == 1, (
            "ranking correctness does not depend on the band; a correct "
            "ranking that only reached `review` still counts"
        )
        assert summary.correct_accepts == 0


class TestReportMaskHolds:
    """Plan test 6: the summary must pass `write_report`; a path must not."""

    def test_summary_body_passes_the_mask(self, tmp_path: Path) -> None:
        summary = summarize(
            [_matched_nontarget("nt-01.jpg"), _unprocessable("nt-02.jpg")],
            set_name="nontarget",
        )
        body = render_report_body(summary)
        out = write_report(tmp_path / "nested" / "report.md", "# D5", extra_body=body)
        assert out.exists()
        assert "- photos: 2" in out.read_text(encoding="utf-8"), (
            "the report must carry the photo count as its denominator"
        )

    def test_an_absolute_path_in_the_body_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(RedactionError):
            write_report(
                tmp_path / "r.md",
                "# D5",
                extra_body="scanned /Users/someone/photos/probe-01.jpg",
            )

    def test_per_image_identity_label_is_refused(self, tmp_path: Path) -> None:
        """A count-and-range summary must not carry per-image identities.

        This is the boundary that keeps the committed report free of the
        biometric detail that lives in the repo-external CSV.
        """
        with pytest.raises(RedactionError):
            write_report(
                tmp_path / "r.md",
                "# D5",
                extra_body="probe-01.jpg: identity: enroll-23",
            )


class TestSweepPresentsButDoesNotChoose:
    def test_grid_comes_from_the_shared_constants(self) -> None:
        cells = sweep_live_semantics(
            [_matched_nontarget("nt-01.jpg")],
            REAL_MATCH_GRID,
            REAL_MARGIN_GRID,
            review=REVIEW_T,
        )
        assert cells, "a sweep over the frozen grids must produce cells"
        pairs = {(c.match_threshold, c.margin_threshold) for c in cells}
        assert pairs == {(m, g) for m in REAL_MATCH_GRID for g in REAL_MARGIN_GRID}, (
            "the sweep must cover the shared grids exactly, not a copy of them"
        )
        for cell in cells:
            assert cell.selected is False, (
                "a sweep cell must never mark itself as the chosen "
                "threshold: D5 has no holdout, so picking one would be "
                "tuning on the test set"
            )

    def test_sweep_denominator_is_the_photo_count(self) -> None:
        """A rejected photo stays in the denominator at every grid point."""
        cells = sweep_live_semantics(
            [_matched_nontarget("nt-01.jpg"), _unprocessable("nt-02.jpg")],
            REAL_MATCH_GRID,
            REAL_MARGIN_GRID,
            review=REVIEW_T,
        )
        assert cells, "the grid must still produce cells with an unusable photo"
        assert {c.total for c in cells} == {2}, (
            "the sweep denominator must be the photo count; a per-cell "
            "denominator that shrinks with the scoreable set would let a "
            "grid point look better by having fewer rows"
        )


# --------------------------------------------------------------------------
# Row builders — these assemble PhotoResult directly. Every assertion that
# matters about band/FA classification is made against `score_photo`'s
# output above; these builders exist only to build multi-row populations
# for the denominator and misidentification cases.
# --------------------------------------------------------------------------


def _matched_nontarget(photo: str) -> PhotoResult:
    return PhotoResult(
        set_name="nontarget",
        photo=photo,
        truth_identity=None,
        face_count=1,
        unprocessable_reason=None,
        quality_pass=True,
        quality_reasons=(),
        l_top1=PROBE_ID,
        l_top1_score=0.80,
        l_top2=OTHER_ID,
        l_top2_score=0.10,
        l_margin=0.70,
        l_band="matched",
        r_top1=PROBE_ID,
        r_top1_score=0.80,
        r_top2=OTHER_ID,
        r_top2_score=0.10,
        r_margin=0.70,
        r_band="matched",
        stage_ms={"detection": 1.0, "quality": 0.5, "embed": 2.0},
    )


def _matched_probe(photo: str, *, truth: str, top1: str) -> PhotoResult:
    return PhotoResult(
        set_name="probe",
        photo=photo,
        truth_identity=truth,
        face_count=1,
        unprocessable_reason=None,
        quality_pass=True,
        quality_reasons=(),
        l_top1=top1,
        l_top1_score=0.75,
        l_top2=truth if top1 != truth else OTHER_ID,
        l_top2_score=0.10,
        l_margin=0.65,
        l_band="matched",
        r_top1=top1,
        r_top1_score=0.75,
        r_top2=truth if top1 != truth else OTHER_ID,
        r_top2_score=0.10,
        r_margin=0.65,
        r_band="matched",
        stage_ms={},
    )


def _unprocessable(photo: str) -> PhotoResult:
    return PhotoResult(
        set_name="nontarget",
        photo=photo,
        truth_identity=None,
        face_count=0,
        unprocessable_reason="no_face_detected",
        quality_pass=False,
        quality_reasons=("input_no_face",),
        l_top1=None,
        l_top1_score=None,
        l_top2=None,
        l_top2_score=None,
        l_margin=None,
        l_band="rejected",
        r_top1=None,
        r_top1_score=None,
        r_top2=None,
        r_top2_score=None,
        r_margin=None,
        r_band=None,
        stage_ms={},
    )
