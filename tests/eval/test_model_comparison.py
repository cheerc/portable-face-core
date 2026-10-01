"""D6 A2: the cross-candidate comparison runner, test-first.

The plan (`docs/plans/2026-09-30-d6-deepface-comparison-plan.md`, A2)
names the failure this file exists to prevent: a comparison that is
really four separate runs wearing one table's clothes. Every rule below
comes from a way that can go wrong *silently* — the run still prints
numbers, they are just about something else.

The load-bearing decisions, each with the failure it forecloses:

1. **A candidate gets its own gallery, and it must be complete.**
   Reusing the SFace gallery for a Facenet512 probe would compare
   vectors from two different models in one cosine. Same dimension is
   not the same model — Facenet512 and ArcFace are both 512-D, so a
   dimension check alone would wave them through. The guard is on
   `model_version`, and a short gallery is a stop, never a smaller
   comparison.

2. **D5's counting is reused, not reimplemented.** Candidate rows go
   through D5's own `score_photo`/`summarize`/`summarize_r` with a
   candidate `ScoringContext`, so the L/R definitions cannot drift
   between the control and the candidates. A second implementation of
   "what counts as a false accept" is a second opinion nobody
   re-derives.

3. **The denominator is the photo count, always.** 13 and 30 are what
   went in. A rejected or unprocessable photo stays in, because a
   harness that drops the rows it could not use reports a better run
   than happened.

4. **New candidates get a frontier, never the control's thresholds.**
   g3-v1's 0.363/0.3/0.10 is calibrated for SFace. Applying it to
   another model answers a question nobody asked, so candidates explore
   their own (match, margin) grid, `review` is reported as *not
   configured* rather than fabricated as 0.3, and **every** cell
   carries `selected=False` — this corpus is the evaluation set, so
   naming a winner from it is tuning on the test data.

5. **Each candidate declares its interpreter.** ORT 1.30 results on
   3.13 are not 3.14 results. A candidate that does not name its
   interpreter, or that claims to have run on the product's 3.14 when
   it is a DeepFace candidate, is refused before anything is measured.

The synthetic context is the same shape as D5's: unit-vector gallery,
detector returning the boxes the test asks for. Real weights and real
photos are A3's job, not this file's.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from facecore.contracts.policy import PolicyProfile
from facecore.eval.candidate_registry import (
    UnknownCandidateModel,
    resolve_candidate_model,
)
from facecore.eval.model_comparison import (
    FA_BUDGETS,
    MARGIN_GRID,
    MATCH_GRID,
    CandidateComparison,
    CandidateSpec,
    CandidateContractError,
    GalleryIncomplete,
    ModelBindingError,
    build_candidate_gallery,
    build_candidate_context,
    compare_candidate,
    frontier_at_budget,
    require_declared_interpreter,
    require_vector_shape,
    sha256_of_file,
)
from facecore.eval.static_baseline import SET_PROBE, score_photo
from facecore.live.frame_pipeline import ResearchGallery, ScoringContext
from facecore.pipeline.decode import DecodedImage
from facecore.pipeline.detect import DetectedFace

MATCH_T, REVIEW_T, MARGIN_T = 0.363, 0.3, 0.10
PROBE_ID = "probe-target"
OTHER_ID = "other-identity"
MODEL = "facenet512_xyz"
PRODUCT_MODEL = "sface_2021dec"
PRODUCT_PY = "3.14"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _axis(dimension: int, index: int) -> np.ndarray:
    arr = np.zeros(dimension, dtype=np.float32)
    arr[index] = 1.0
    return arr


def _face() -> DetectedFace:
    x, y, w, h = (10.0, 10.0, 120.0, 120.0)
    return DetectedFace(
        box=(x, y, w, h),
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

    def detect(self, decoded: DecodedImage) -> list[DetectedFace]:
        return self._faces


class _Embedder:
    """Returns the vector it was told to, after checking the crop's shape.

    The shape check is what makes the 112->160 resize visible: a
    candidate that needs a different input size has to say so, and this
    records what it actually received.
    """

    def __init__(self, vector: np.ndarray, model_version: str) -> None:
        self._vector = vector
        self.model_version = model_version
        self.seen_shapes: list[tuple[int, int]] = []

    def embed(self, crop: Any) -> tuple[np.ndarray, str]:
        self.seen_shapes.append((crop.width, crop.height))
        return self._vector, self.model_version


def _gallery(identities: dict[str, np.ndarray], model_version: str) -> ResearchGallery:
    return ResearchGallery(
        embeddings={k: np.asarray(v, dtype=np.float32) for k, v in identities.items()},
        model_version=model_version,
        generation="gen-d6",
        digest=f"digest-{model_version}",
    )


def _probe_for(top: float, runner_up: float) -> np.ndarray:
    """A unit vector whose cosine against axis 0 / axis 1 is (top, runner_up)."""
    v = np.array([top, runner_up], dtype=np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _context(
    *, probe: np.ndarray, identities: dict[str, np.ndarray], model: str, faces=None
) -> ScoringContext:
    dim = len(probe)
    for ident, vec in identities.items():
        assert len(vec) == dim
    embedder = _Embedder(probe, model)
    return ScoringContext(
        gallery=_gallery(identities, model),
        model_version=model,
        policy=PolicyProfile.frozen_v1().with_thresholds(
            match_threshold=MATCH_T,
            review_threshold=REVIEW_T,
            margin_threshold=MARGIN_T,
        ),
        detector=_Detector(faces if faces is not None else [_face()]),
        embedder=embedder,
    )


def _two_identity_context(top: float, runner_up: float, model: str = MODEL):
    probe = _probe_for(top, runner_up)
    dim = len(probe)
    return _context(
        probe=probe,
        identities={PROBE_ID: _axis(dim, 0), OTHER_ID: _axis(dim, 1)},
        model=model,
    )


class _Stub:
    """The minimal candidate surface `compare_candidate` consumes.

    A comparison must not reach inside the model: it takes a context and
    a photo list. Stubbing the context here is what keeps the runner
    testable without weights, and it is also the seam a real DeepFace
    worker plugs into in A3.
    """

    def __init__(self, context: ScoringContext) -> None:
        self.context = context
        self.photos: list[Any] = []

    def score(self, path, *, set_name, truth_identity, sequence):
        self.photos.append(path)
        return score_photo(
            path,
            self.context,
            self.context.policy,
            set_name=set_name,
            truth_identity=truth_identity,
            sequence=sequence,
        )


def _spec(**over: Any) -> CandidateSpec:
    """A candidate spec built from the repo-fixed record.

    The weight hash and the dimension are not arguments here and cannot
    be: they are read from
    :mod:`facecore.eval.candidate_registry` and a spec that disagrees
    with the record is refused at construction. Tests about the
    *bindings* do not go through this helper -- they pass an explicit
    ``registry_id`` so the file and the record can disagree.
    """
    defaults: dict[str, Any] = {
        "candidate_id": "deepface-facenet512",
        "model_version": MODEL,
        "normalization": "Facenet",
        "interpreter": "3.13",
        "backend": "deepface-worker",
    }
    return CandidateSpec.from_registry_id(
        "deepface_facenet512",
        **{k: v for k, v in (defaults | over).items() if k not in _WEIGHT_FACTS},
    )


#: Where the recorded Facenet512 weights live, if they are present. The
#: registry holds their real checksum, so a test cannot manufacture a
#: file that legitimately matches it; the positive path is therefore
#: conditional on the artifact existing, and says so when it is absent.
_RECORDED_FACENET512_WEIGHT = (
    Path.home() / "facecore-models" / "facenet512_weights.h5"
)


#: The fields a ``CandidateSpec`` takes from the repo-fixed record. A
#: helper that defaults them would be re-introducing the free-form
#: declaration this rework removed, so they are filtered out by name
#: wherever a spec is built through ``from_registry_id``.
_WEIGHT_FACTS = frozenset({"weight_sha256", "embedding_dim", "input_size"})


def _registry_spec(registry_id: str, **over: Any) -> CandidateSpec:
    """A spec built from the repo-fixed record, with only the caller-owned
    fields (name, normalization, interpreter, backend) supplied.

    The weight facts are never arguments here. That is the whole point:
    a test that could pass them would be testing the old, bypassable
    design.
    """
    defaults: dict[str, Any] = {
        "candidate_id": "c",
        "normalization": "Facenet",
        "interpreter": "3.13",
        "backend": "deepface-worker",
    }
    return CandidateSpec.from_registry_id(
        registry_id,
        **{k: v for k, v in (defaults | over).items() if k not in _WEIGHT_FACTS},
    )


def _weight_file(payload: bytes) -> Path:
    """A stand-in weight file, on disk, hashed by the binding.

    Real bytes rather than a stubbed hash function: the point of the
    binding is that it reads the file, so a test that skipped the file
    would be testing a mock of the thing under test.
    """
    name = hashlib.sha256(payload).hexdigest()[:12]
    p = Path(os.environ.get("TMPDIR", "/tmp")) / f"d6a2_weight_{name}"
    p.write_bytes(payload)
    return p


def _vectors(*, seed: int, dim: int) -> dict[str, np.ndarray]:
    """Deterministic unit vectors standing in for one model's embeddings."""
    rs = np.random.RandomState(seed)
    out = {}
    for i in range(4):
        v = rs.randn(dim)
        out[f"id-{i:02d}"] = (v / np.linalg.norm(v)).astype(np.float32)
    return out


def _bound_spec(dim: int = 4) -> tuple:
    """Retained for call sites that want "a spec and a file".

    The spec comes from the registry, so the file cannot be made to
    match it: no byte string a test writes will hash to the recorded
    Facenet512 digest. Callers therefore use this to reach a *refusal*,
    which is what the byte check now means.
    """
    return _spec(), _weight_file(b"facenet512-weights-bytes")


# --------------------------------------------------------------------------
# 1. Gallery isolation and completeness
# --------------------------------------------------------------------------


class TestCandidateGalleryIsItsOwn:
    @pytest.mark.skipif(
        not _RECORDED_FACENET512_WEIGHT.is_file(),
        reason=(
            "the recorded Facenet512 weights are absent, and the count "
            "guard is only reachable after the byte check passes; the "
            "registry holds a real checksum, so no substitute file can "
            "legitimately match it"
        ),
    )
    def test_a_gallery_of_the_right_size_is_accepted(self) -> None:
        """The count guard is only reachable once the bytes are right.

        Ordering matters here: the byte check runs before the count
        check, so proving "the right size is accepted" needs a file that
        genuinely matches the record. That is why this test is
        conditional on the artifact existing -- and why the *refusal*
        tests below do not need it.
        """
        spec = _registry_spec("deepface_facenet512")
        gal = build_candidate_gallery(
            spec=spec,
            weight_path=_RECORDED_FACENET512_WEIGHT,
            identities=_vectors(seed=0, dim=512),
            expected_identities=4,
        )
        assert len(gal.embeddings) == 4
        assert gal.model_version

    def test_a_short_gallery_stops_the_comparison_rather_than_shrinking_it(
        self,
    ) -> None:
        """22 of 23 is not a comparison; it is a different, easier one.

        Dropping the missing identity would also drop the runner-up it
        was most likely to beat, so every false-accept count after it
        would improve for a reason that has nothing to do with the
        model. The plan makes this a stop condition.
        """
        spec, weight = _bound_spec()
        with pytest.raises(
            GalleryIncomplete, match="has 2 identities but the freeze declares 3"
        ):
            build_candidate_gallery(
                spec=spec,
                weight_path=weight,
                identities={"a": _axis(4, 0), "b": _axis(4, 1)},
                expected_identities=3,
            )

    def test_an_overfull_gallery_also_stops(self) -> None:
        """More identities than declared means the corpus is not the one frozen.

        Silently taking the first N would compare against a subset
        chosen by directory order, which is not a reproducible rule.
        """
        spec, weight = _bound_spec()
        with pytest.raises(
            GalleryIncomplete, match="has 3 identities but the freeze declares 2"
        ):
            build_candidate_gallery(
                spec=spec,
                weight_path=weight,
                identities={
                    "a": _axis(4, 0),
                    "b": _axis(4, 1),
                    "c": _axis(4, 2),
                },
                expected_identities=2,
            )

    def test_two_candidates_of_the_same_dimension_are_still_not_interchangeable(
        self,
    ) -> None:
        """The dimension check alone would wave Facenet512 and ArcFace through.

        Both are 512-D, so a runner that only compared vector lengths
        would let a Facenet512 probe score against an ArcFace gallery and
        report the result under the Facenet512 name.
        """
        spec, _ = _bound_spec()
        ctx = _two_identity_context(0.9, 0.1, model=MODEL)
        with pytest.raises(GalleryIncomplete, match="arcface_xyz"):
            build_candidate_context(
                spec=spec,
                gallery=_gallery(
                    {PROBE_ID: _axis(2, 0), OTHER_ID: _axis(2, 1)}, "arcface_xyz"
                ),
                detector=ctx.detector,
                embedder=ctx.embedder,
                profile=ctx.policy,
            )

    def test_a_context_whose_gallery_and_probe_disagree_is_refused(self) -> None:
        """The reverse direction: SFace gallery, candidate probe.

        This is the "new probe scored against the old gallery" case, and
        it is the one that produces plausible-looking numbers.
        """
        probe = _probe_for(0.9, 0.1)
        spec, _ = _bound_spec()
        with pytest.raises(GalleryIncomplete, match="sface_2021dec"):
            build_candidate_context(
                gallery=_gallery(
                    {PROBE_ID: _axis(2, 0), OTHER_ID: _axis(2, 1)}, PRODUCT_MODEL
                ),
                spec=spec,
                detector=_Detector([_face()]),
                embedder=_Embedder(probe, MODEL),
                profile=PolicyProfile.frozen_v1(),
            )


# --------------------------------------------------------------------------
# 1b. The binding must be to bytes, not to a label
# --------------------------------------------------------------------------


class TestTheWeightFactsAreNotTheCallersToChoose:
    """The bypass two reviewers found, and the property that closes it.

    The previous design put ``weight_sha256`` and ``embedding_dim`` on
    the spec as free fields, and every check compared the caller's
    declaration against the caller's file. Those are the same fact read
    twice. A caller who wrote the declaration to match the file it
    supplied -- Facenet512's name over ArcFace's bytes, with ArcFace's
    own hash declared -- satisfied every check, and
    ``assert_binding_recorded()`` passed too, because the observed and
    declared values agreed with each other. The result was complete,
    plausible, and reported under the wrong model.

    The fix is not another comparison; it is that the two facts now
    come from :mod:`facecore.eval.candidate_registry` and a caller may
    only *select* a record, never write one. Every test here is about
    that freedom being gone.

    **What none of these tests claim.** A caller that supplies the
    correct Facenet512 weights together with ArcFace-computed vectors
    still passes. Proving otherwise needs the crops the vectors were
    computed from, which this interface does not carry; that is A3's
    work and it is recorded as UNVERIFIED. A test asserting the stronger
    property would be asserting something false, which is how the
    deleted ``test_a_fully_relabelled_arcface_gallery_is_still_refused``
    came to be wrong in the first place.
    """

    def test_a_spec_cannot_be_written_without_naming_a_recorded_model(self) -> None:
        """The old API is gone: no registry id means no spec.

        Previously every field was supplied by the caller, so a spec
        could exist that the repo had no record of at all. That is the
        freedom these tests exist to remove, and it is refused at
        construction rather than at scoring time.
        """
        with pytest.raises(CandidateContractError, match="no registry_id"):
            CandidateSpec(
                candidate_id="c",
                model_name="Facenet512",
                model_version="facenet512_v",
                embedding_dim=512,
                input_size=160,
                normalization="Facenet",
                interpreter="3.13",
                weight_sha256="a" * 64,
                backend="deepface-worker",
            )

    def test_an_unknown_model_id_is_refused(self) -> None:
        """A registry that accepts new ids is the defect, not the fix.

        So an unknown id stops the run. A typo fails here, loudly, with
        the list of real ids -- and a caller cannot add its own entry on
        the way past.
        """
        with pytest.raises(UnknownCandidateModel, match="no repo-fixed record"):
            resolve_candidate_model("arcface")

    def test_the_recorded_hash_cannot_be_overridden(self) -> None:
        """The specific bypass: supply a real file and declare *its* hash.

        This is the move that worked. The registry id says Facenet512,
        the caller tries to attach the hash of the file it is actually
        holding, and the override is refused because the recorded value
        is not a parameter of the call.
        """
        arcface_weight = _weight_file(b"arcface-weights-bytes")
        with pytest.raises(CandidateContractError, match="may not override"):
            CandidateSpec.from_registry_id(
                "deepface_facenet512",
                candidate_id="c",
                normalization="Facenet",
                interpreter="3.13",
                backend="deepface-worker",
                weight_sha256=sha256_of_file(arcface_weight),
            )

    def test_a_hand_tuned_spec_disagreeing_with_its_record_is_refused(self) -> None:
        """Reaching past the classmethod does not help either.

        ``from_registry_id`` is the convenient door, not the only one:
        a caller can still reach the constructor. Building a spec
        directly and then editing the hash is the same bypass in
        another shape, so the constructor checks the declared facts
        against the record rather than trusting the classmethod.
        """
        record = resolve_candidate_model("deepface_facenet512")
        with pytest.raises(CandidateContractError, match="the repo records"):
            CandidateSpec(
                candidate_id="c",
                model_name="Facenet512",
                model_version="facenet512_v",
                embedding_dim=512,
                input_size=160,
                normalization="Facenet",
                interpreter="3.13",
                weight_sha256="b" * 64,  # not the recorded value
                backend="deepface-worker",
                registry_id="deepface_facenet512",
            )
        assert record.weight_sha256 is not None

    def test_the_dimension_comes_from_the_record_too(self) -> None:
        """Facenet512 and ArcFace are both 512-D, so the hash is the only
        discriminator between them -- but a 128-D model must not borrow a
        512-D record's dimension either."""
        with pytest.raises(CandidateContractError, match="embedding_dim"):
            CandidateSpec(
                candidate_id="c",
                model_name="SFace",
                model_version="sface_2021dec",
                embedding_dim=512,
                input_size=112,
                normalization="base",
                interpreter="3.13",
                weight_sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
                backend="product",
                registry_id="sface_2021dec_fp32",
            )

    def test_a_file_that_is_not_the_recorded_weights_is_refused(self) -> None:
        """The honest case, wrong file: the byte check still has a job.

        The registry makes the *declaration* trustworthy, so this is no
        longer the check that catches a lie -- it is the check that
        catches pointing an honest spec at the wrong artifact on disk.
        The spec here is built from the record, so the only thing wrong
        is the file.
        """
        spec = _registry_spec("deepface_facenet512")
        wrong_file = _weight_file(b"not-the-recorded-facenet512-bytes")
        with pytest.raises(ModelBindingError, match="is recorded as hashing to"):
            build_candidate_gallery(
                spec=spec,
                weight_path=wrong_file,
                identities=_vectors(seed=0, dim=512),
                expected_identities=4,
            )

    def test_a_vector_of_the_wrong_dimension_is_refused(self) -> None:
        """Dimensions are read off the arrays and the record, not the caller.

        A 128-D vector under a 512-D record. The byte check would pass
        here (the weight is the recorded one), so this is the check
        doing the work the dimension is actually needed for.
        """
        spec = _registry_spec("deepface_facenet512")
        mixed = _vectors(seed=2, dim=512)
        mixed["id-00"] = _axis(128, 0)
        with pytest.raises(ModelBindingError, match="not 512-D"):
            require_vector_shape(spec=spec, identities=mixed)

    @pytest.mark.skipif(
        not _RECORDED_FACENET512_WEIGHT.is_file(),
        reason=(
            "the recorded Facenet512 weights are not present; the registry "
            "holds their real checksum, so no substitute file can match it "
            "and the positive path cannot be faked"
        ),
    )
    def test_the_correct_weights_do_bind(self) -> None:
        """The positive case, so the refusals above are refusals and not a wall.

        Skipped when the artifact is absent, which is the normal state
        in CI. The skip says why it cannot be substituted: a registry
        entry records a real checksum, so there is no byte string this
        test could write that would legitimately match it.
        """
        spec = _registry_spec("deepface_facenet512")
        gal = build_candidate_gallery(
            spec=spec,
            weight_path=_RECORDED_FACENET512_WEIGHT,
            identities=_vectors(seed=0, dim=512),
            expected_identities=4,
        )
        assert len(gal.embeddings) == 4

    def test_the_observed_hash_is_recorded_for_the_report(self) -> None:
        """The report must carry what ran, and it now equals the record.

        A result recording only the declared hash is indistinguishable
        from one whose file was verified, and the counts are identical
        either way -- which is why nothing else catches it.

        Asserted against the record rather than against a successful
        bind, so it holds without the artifact present.
        """
        record = resolve_candidate_model("deepface_facenet512")
        comparison = _comparison(
            observed_weight_sha256=record.weight_sha256,
            spec_overrides={"registry_id": "deepface_facenet512"},
        )
        assert comparison.observed_weight_sha256 == comparison.spec.weight_sha256
        assert (
            comparison.to_dict()["observed_weight_sha256"]
            == record.weight_sha256
        )

    def test_a_comparison_without_a_recorded_hash_cannot_be_published(self) -> None:
        """A result with no observed hash is a claim, not a measurement.

        The observed hash is a plain string field, so deleting the
        first clause lets execution fall into the second one, which
        slices ``None`` and raises ``TypeError``. That failure is real
        but it belongs to the line *below* -- and a mutation reporting it
        looks exactly like a working guard, which is the shape of false
        evidence this rework exists to remove.

        So the call is made inside a ``try`` that catches *any*
        exception, and the assertions then demand this guard's own
        refusal. Catching only ``CandidateContractError`` would let the
        ``TypeError`` escape as an error naming the line it came from --
        red, but not attributable to the clause that was deleted.
        """
        comparison = _comparison(
            observed_weight_sha256=None,
            spec_overrides={"registry_id": "deepface_facenet512"},
        )
        refusal: Exception | None = None
        try:
            comparison.assert_binding_recorded()
        except Exception as exc:  # noqa: BLE001 - the point is to name whatever came
            refusal = exc
        assert refusal is not None, "the guard did not refuse an unbound result"
        assert isinstance(refusal, CandidateContractError), (
            f"raised {type(refusal).__name__} rather than this guard's "
            f"refusal; a deleted clause falling through to the next line "
            f"gives {refusal}"
        )
        assert "no observed weight hash" in str(refusal)

    def test_a_hash_disagreeing_with_the_record_cannot_be_published(self) -> None:
        """An observed value that is not the recorded one is a silent failure.

        This is the state a mislabelled run would produce if the
        binding were advisory. The counts are already computed by then,
        so nothing else looks wrong.
        """
        comparison = _comparison(
            observed_weight_sha256="f" * 64,
            spec_overrides={"registry_id": "deepface_facenet512"},
        )
        with pytest.raises(
            CandidateContractError, match="recorded observed weight hash"
        ):
            comparison.assert_binding_recorded()


# --------------------------------------------------------------------------
# 2. Interpreter declaration
# --------------------------------------------------------------------------


class TestInterpreterMustBeDeclared:
    def test_a_candidate_with_no_interpreter_is_refused(self) -> None:
        with pytest.raises(CandidateContractError, match="interpreter"):
            require_declared_interpreter(_spec(interpreter=""))

    def test_a_deepface_candidate_claiming_the_product_interpreter_is_refused(
        self,
    ) -> None:
        """3.13 ORT results are not 3.14 results, and the label is the guard.

        Running a DeepFace candidate on the product's 3.14 and filing it
        under 3.13 (or vice versa) is the specific error the plan calls
        out, and nothing downstream can detect it — the numbers look
        fine either way.
        """
        with pytest.raises(CandidateContractError, match="3.14"):
            require_declared_interpreter(
                _spec(interpreter="3.14"), product_requires_python=PRODUCT_PY
            )

    def test_a_candidate_on_its_own_interpreter_is_accepted(self) -> None:
        require_declared_interpreter(
            _spec(interpreter="3.13"), product_requires_python=PRODUCT_PY
        )

    def test_the_control_may_declare_the_product_interpreter(self) -> None:
        """The control *is* the product path; refusing it would be backwards."""
        require_declared_interpreter(
            _spec(
                candidate_id="ort-sface-control", backend="product", interpreter="3.14"
            ),
            product_requires_python=PRODUCT_PY,
        )

    def test_the_declaration_is_carried_into_the_result(self) -> None:
        result = compare_candidate(
            spec=_spec(interpreter="3.13"),
            scorer=_Stub(_two_identity_context(0.9, 0.1)),
            probes=[],
            nontarget=[],
            expected_probes=0,
            expected_nontarget=0,
            probe_truth=PROBE_ID,
            observed_weight_sha256=None,
            product_requires_python=PRODUCT_PY,
        )
        assert result.spec.interpreter == "3.13"
        assert result.spec.backend == "deepface-worker"


# --------------------------------------------------------------------------
# 3. Denominators never shrink
# --------------------------------------------------------------------------


class TestDenominatorsAreTheInputCount:
    def test_a_rejected_probe_is_still_counted(self) -> None:
        """A quality rejection is a result the App produced, not a missing row.

        13 photos went in. If a rejection leaves the denominator, the
        run reports 12/12 and looks perfect.
        """
        rows = _rows(2, rejected=1)
        summary = ComparisonOutcomeShim(rows)
        assert summary.total == 2
        assert summary.counts["quality_rejected"] == 1
        partition = (
            summary.counts["correct_accepts"]
            + summary.counts["wrong_identities"]
            + summary.counts["review"]
            + summary.counts["unknown_rejects"]
            + summary.counts["quality_rejected"]
            + summary.counts["unprocessable"]
        )
        assert partition == 2

    def test_an_unprocessable_row_is_counted_not_dropped(self) -> None:
        rows = _rows(3, unprocessable=1)
        summary = ComparisonOutcomeShim(rows)
        assert summary.total == 3
        assert summary.counts["unprocessable"] == 1

    def test_a_wrong_count_of_inputs_stops_the_run(self) -> None:
        """The frozen 13/30 is a precondition, not a hope.

        A run that quietly scored 12 probes has no way to say so in its
        own output, and every ratio downstream is then computed against
        a number the reader has to trust.

        The probe list holds real ``Path`` objects, not bare
        ``object()``s. With ``object()`` the count guard still fires, but
        the mutation that removes it fails on a ``TypeError`` from the
        fixture's input type rather than on the refusal — green for an
        accident. A Path keeps the failure attributable to the guard.
        """
        with pytest.raises(CandidateContractError, match="probe"):
            compare_candidate(
                spec=_spec(),
                scorer=_Stub(_two_identity_context(0.9, 0.1)),
                probes=[Path("synthetic-probe.jpg")],
                nontarget=[],
                expected_probes=13,
                expected_nontarget=0,
                probe_truth=PROBE_ID,
                observed_weight_sha256=None,
                product_requires_python=PRODUCT_PY,
            )

    def test_a_wrong_count_of_nontargets_stops_the_run(self, tmp_path: Path) -> None:
        """The 30-photo side of the count guard, which is the larger half.

        The probe guard and the non-target guard are separate clauses,
        and only the probe one had a test. A run that quietly scored 29
        non-targets would report every false-accept count against a
        denominator the reader cannot see — and 29 of 30 is a
        ``1/30``-shaped result that looks like an improvement.

        ``tmp_path`` supplies a real, *decodable* image on purpose. The
        guard fires before anything is read, so the file is never
        opened when the guard works; a path that does not exist, or a
        file that is not an image, would instead make a *deleted* guard
        fail inside the scorer — a different refusal than the one under
        test, and one that a mutation could be scored as a pass.
        """
        nt = tmp_path / "synthetic-nt.jpg"
        Image.new("RGB", (16, 16), (120, 120, 120)).save(nt)
        refusal: Exception | None = None
        try:
            compare_candidate(
                spec=_spec(),
                scorer=_Stub(_two_identity_context(0.9, 0.1)),
                probes=[],
                nontarget=[nt],
                expected_probes=0,
                expected_nontarget=30,
                probe_truth=PROBE_ID,
                observed_weight_sha256=None,
                product_requires_python=PRODUCT_PY,
            )
        except CandidateContractError as exc:
            refusal = exc
        assert refusal is not None, "the count guard did not refuse a short cohort"
        assert "non-target" in str(refusal)

    def test_a_fully_rejected_cohort_stops_the_run(self) -> None:
        """Every row unprocessable is a broken setup, not a result.

        Reporting 0/13 correct accept for a cohort where nothing could be
        embedded is indistinguishable from a model that failed to
        recognise anyone, and the two call for opposite responses.
        """
        rows = _rows(4, unprocessable=4)
        with pytest.raises(CandidateContractError, match="nothing was scorable"):
            ComparisonOutcomeShim(rows)


class TestRBranchCountMustBeAnInt:
    """The narrowing that keeps a report-shaped dict from inventing a count.

    ``summarize_r`` is typed ``dict[str, object]`` because it is a
    report side table. The narrowing is loud on purpose: a count that
    arrived as a string or a float is a bug upstream, and coercing it
    would publish a number nobody computed.

    ``np.int64`` is worth naming. ``isinstance(np.int64(5), int)`` is
    **False** on this numpy, so a legitimate numpy count would be
    refused. That is the intended direction — A3's D5 path uses
    ``sum(1 for ...)`` and yields a Python ``int`` — but it is a real
    constraint on what A3 may pass, not an accident.
    """

    def test_a_real_d5_r_count_is_accepted(self) -> None:
        rows = _rows(3)
        summary = ComparisonOutcomeShim(rows)
        assert summary.r_top1_correct == 3

    @pytest.mark.parametrize("bad", [1.5, "3", None, np.int64(3), [3], True])
    def test_a_count_that_is_not_a_python_int_is_refused(self, bad: Any) -> None:
        """Floats, strings, numpy ints, containers and ``bool`` all refuse.

        ``bool`` is an ``int`` subclass, so it passes ``isinstance`` —
        which is why it is listed here as a case to think about rather
        than one the check catches. It is included to document the
        boundary honestly, not to claim coverage the check lacks.
        """
        from facecore.eval.model_comparison import ComparisonOutcome

        if isinstance(bad, bool):
            pytest.xfail("bool is an int subclass; not caught by isinstance")
        with _fake_r_count(bad):
            with pytest.raises(CandidateContractError, match="not an int"):
                ComparisonOutcome.from_rows(_rows(3), set_name=SET_PROBE)


# --------------------------------------------------------------------------
# 4. Frontiers, not borrowed thresholds
# --------------------------------------------------------------------------


class TestCandidatesGetTheirOwnFrontier:
    def test_the_grid_is_the_plan_s_grid(self) -> None:
        assert MATCH_GRID[0] == -1.0
        assert MATCH_GRID[-1] == 1.0
        assert MATCH_GRID[1] - MATCH_GRID[0] == pytest.approx(0.01)
        assert MARGIN_GRID == (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 2.0)

    def test_no_cell_is_ever_selected(self) -> None:
        """The corpus *is* the evaluation set; a winner would be tuning on it."""
        comparison = _comparison()
        assert comparison.frontier, "frontier must not be empty"
        assert all(cell.selected is False for cell in comparison.frontier)
        assert all(c["selected"] is False for c in comparison.frontier_summary), (
            "the summary rows must not smuggle a selection back in"
        )

    def test_review_is_reported_as_not_configured_not_as_zero_point_three(self) -> None:
        """A candidate has no independent review source, so 0.3 is invented.

        Carrying the control's 0.3 into a candidate's band would make its
        "review" count mean nothing while looking like a measurement.
        """
        comparison = _comparison()
        assert comparison.review_configured is False
        assert comparison.review_threshold is None

    def test_the_control_keeps_g3_v1_and_does_not_explore(self) -> None:
        """The control has a frozen operating point; exploring would invent one."""
        control = _comparison(
            backend="product",
            candidate_id="ort-sface-control",
            review_threshold=REVIEW_T,
        )
        assert control.review_configured is True
        assert control.review_threshold == REVIEW_T
        assert control.frontier == []
        assert control.frontier_summary == []

    def test_a_candidate_is_not_scored_with_the_control_thresholds(self) -> None:
        """0.363 is SFace's calibration; applying it to another model is a claim.

        The runner reports the candidate's own ranking and a frontier
        instead. This test fails if a candidate comparison ever starts
        reporting accept counts at the control's operating point.
        """
        comparison = _comparison()
        assert comparison.operating_point is None

    def test_a_candidate_is_never_given_a_review_threshold(self) -> None:
        """The code refuses 0.3 for a candidate; this pins the refusal.

        A candidate has no independent review source, so any review
        threshold handed to it is fabricated. The refusal is on the
        *flag*, not on the counts: because the frontier sweeps with an
        ``inf`` sentinel, supplying 0.3 changes no cell's accept count
        at all — so a mutation that let the value through would leave
        every number in the report identical while silently publishing
        ``review_configured=True``. That flag is the thing a reader
        would act on, which is why it needs a test of its own.
        """
        comparison = _comparison()
        assert comparison.review_threshold is None
        assert comparison.review_configured is False
        assert all(
            cell["review_configured"] is False
            for row in comparison.frontier_summary
            for cell in (row["probe"], row["nontarget"])
            if cell is not None
        )
        with pytest.raises(CandidateContractError, match="review"):
            CandidateComparison.build(
                spec=_spec(),
                probe_rows=_rows(13),
                nontarget_rows=_rows(30),
                match_grid=[0.0, 0.5],
                margin_grid=[0.0],
                review_threshold=0.3,
                observed_weight_sha256=None,
                product_requires_python=PRODUCT_PY,
            )


class TestFrontierCellCarriesTheFrozenDenominator:
    """The cell's ``total`` is the photo count, and ``to_dict`` publishes it.

    ``ComparisonOutcome.total`` had a test; the ``FrontierCell`` version
    did not, even though the module states the same rule for both and
    the cell's value is written straight into the per-cell CSV. A cell
    whose denominator drifted to "rows that happened to be scorable"
    would make every frontier cell look better than the corpus it was
    measured on, and the sibling guard is a different code path.
    """

    def test_a_cell_reports_the_photo_count_not_the_scorable_count(self) -> None:
        from facecore.eval.model_comparison import _frontier_for

        rows = _rows(10, rejected=4, unprocessable=2)
        cells = _frontier_for(
            rows, match_grid=[0.0], margin_grid=[0.0], review_threshold=None
        )
        assert cells, "the sweep produced no cells"
        for cell in cells:
            assert cell.total == 10
            assert cell.to_dict()["total"] == 10

    def test_the_summarised_cells_keep_the_denominator_too(self) -> None:
        comparison = _comparison()
        published = [
            cell
            for row in comparison.frontier_summary
            for cell in (row["probe"], row["nontarget"])
            if cell is not None
        ]
        assert published
        assert {c["total"] for c in published} == {13, 30}


class TestFrontierAtAFaBudget:
    def test_a_strict_budget_picks_the_cell_with_no_false_accepts(self) -> None:
        cells = [
            _cell(match=0.5, correct=3, wrong=0, fa=0, margin=0.0),
            _cell(match=0.9, correct=9, wrong=0, fa=6, margin=0.0),
        ]
        best = frontier_at_budget(cells, fa_budget=0)
        assert best is not None
        assert best.correct_accepts == 3
        assert best.false_accepts == 0
        assert best.selected is False

    def test_a_looser_budget_admits_more_correct_accepts(self) -> None:
        """The budget is the comparison's axis; it must actually move."""
        cells = [
            _cell(match=0.5, correct=3, wrong=0, fa=0, margin=0.0),
            _cell(match=0.9, correct=9, wrong=0, fa=6, margin=0.0),
        ]
        assert frontier_at_budget(cells, fa_budget=0).correct_accepts == 3
        assert frontier_at_budget(cells, fa_budget=6).correct_accepts == 9

    def test_an_unreachable_budget_reports_nothing_rather_than_the_best_available(
        self,
    ) -> None:
        """Budget 0 with no clean cell is a real finding, not a cell to relax.

        Returning the least-bad cell would answer a question nobody
        asked and hide the fact that the budget could not be met.
        """
        cells = [_cell(match=0.9, correct=9, wrong=1, fa=3, margin=0.0)]
        assert frontier_at_budget(cells, fa_budget=0) is None

    def test_wrong_identity_outranks_a_higher_score(self) -> None:
        """Among equal accept counts, accept the one that identifies nobody wrongly."""
        cells = [
            _cell(match=0.4, correct=5, wrong=2, fa=0, margin=0.0),
            _cell(match=0.6, correct=5, wrong=0, fa=0, margin=0.0),
        ]
        best = frontier_at_budget(cells, fa_budget=0)
        assert best.wrong_identities == 0
        assert best.match_threshold == 0.6

    def test_the_default_budgets_are_the_plans(self) -> None:
        assert FA_BUDGETS == (0, 1, 2)


# --------------------------------------------------------------------------
# 5. L and R are the same rules on different arms
# --------------------------------------------------------------------------


class TestBranchesStayDistinct:
    def test_a_quality_rejection_is_only_visible_on_the_l_branch(self) -> None:
        """The L/R split is the App's view versus the model's ranking.

        Reading the L identity against the R band (or the reverse) is
        how a correctly-identified probe gets counted as a wrong
        identity, so the two must not be cross-read.
        """
        result = _photo(
            quality_pass=False,
            l_band="rejected",
            l_top1=None,
            l_margin=None,
            r_top1=PROBE_ID,
            r_band="matched",
        )
        from facecore.eval.model_comparison import classify_branch

        assert classify_branch(result, branch="l") == "quality_reject"
        assert classify_branch(result, branch="r") == "correct_accept"

    def test_top1_ranking_is_counted_independently_of_the_band(self) -> None:
        """Ranking is the model-agnostic headline; the band is policy.

        A candidate whose scores are all below any usable threshold still
        has a ranking, and that ranking is the one thing comparable
        across models without borrowing SFace's calibration.
        """
        rows = _rows(2)
        for row in rows:
            object.__setattr__(row, "l_band", "unknown")
            object.__setattr__(row, "r_band", "unknown")
        summary = ComparisonOutcomeShim(rows)
        assert summary.l_top1_correct == 2
        assert summary.r_top1_correct == 2


# --------------------------------------------------------------------------
# 6. False accept means matched, not "high score"
# --------------------------------------------------------------------------


class TestFalseAcceptNeedsTheMargin:
    def test_a_high_score_without_the_margin_is_not_a_false_accept(self) -> None:
        """FA is score *and* margin, the App's rule.

        `eval/sweep.py` decides FA on score alone, which would disagree
        with the App. A non-target at 0.90 with a 0.05 margin is a
        review, not an accept.
        """
        result = _photo(
            truth_identity=None, l_band="review", l_top1=OTHER_ID, l_margin=0.05
        )
        from facecore.eval.model_comparison import classify_branch

        assert classify_branch(result, branch="l") == "review"

    def test_the_same_non_target_at_the_margin_is_a_false_accept(self) -> None:
        result = _photo(
            truth_identity=None, l_band="matched", l_top1=OTHER_ID, l_margin=0.10
        )
        from facecore.eval.model_comparison import classify_branch

        assert classify_branch(result, branch="l") == "false_accept"

    def test_no_runner_up_is_never_matched_however_high_the_score(self) -> None:
        """A single-identity gallery has no margin to clear.

        `margin=None` and `score=1.0` must not become a match, or a
        one-identity gallery accepts every photo.
        """
        result = _photo(
            truth_identity=None, l_band="unknown", l_top1=OTHER_ID, l_margin=None
        )
        from facecore.eval.model_comparison import classify_branch

        assert classify_branch(result, branch="l") == "correct_reject"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _photo(**over: Any):
    from facecore.eval.static_baseline import PhotoResult

    base = {
        "set_name": SET_PROBE,
        "photo": "synthetic.jpg",
        "truth_identity": PROBE_ID,
        "face_count": 1,
        "unprocessable_reason": None,
        "quality_pass": True,
        "quality_reasons": (),
        "l_top1": PROBE_ID,
        "l_top1_score": 0.9,
        "l_top2": OTHER_ID,
        "l_top2_score": 0.2,
        "l_margin": 0.7,
        "l_band": "matched",
        "r_top1": PROBE_ID,
        "r_top1_score": 0.9,
        "r_top2": OTHER_ID,
        "r_top2_score": 0.2,
        "r_margin": 0.7,
        "r_band": "matched",
    }
    return PhotoResult(**(base | over))


def _rows(total: int, *, rejected: int = 0, unprocessable: int = 0) -> list:
    rows = []
    for i in range(total):
        if i < unprocessable:
            rows.append(
                _photo(
                    face_count=0,
                    unprocessable_reason="no_face_detected",
                    l_top1=None,
                    l_band="unknown",
                    r_top1=None,
                    r_band="unknown",
                )
            )
        elif i < unprocessable + rejected:
            rows.append(
                _photo(
                    quality_pass=False,
                    l_top1=None,
                    l_band="rejected",
                    l_margin=None,
                    l_top1_score=None,
                    l_top2=None,
                    l_top2_score=None,
                )
            )
        else:
            rows.append(_photo())
    return rows


def _cell(*, match: float, correct: int, wrong: int, fa: int, margin: float):
    from facecore.eval.model_comparison import FrontierCell

    return FrontierCell(
        match_threshold=match,
        margin_threshold=margin,
        correct_accepts=correct,
        wrong_identities=wrong,
        false_accepts=fa,
        no_accept=correct == 0,
        review_configured=False,
        total=13,
        selected=False,
    )


def _comparison(
    *,
    review_threshold: float | None = None,
    observed_weight_sha256: str | None = None,
    spec_overrides: dict[str, Any] | None = None,
    **over: Any,
) -> CandidateComparison:
    # Weight facts come from the registry, so they are not spelled out
    # here; a test that needs a different recorded model passes
    # ``registry_id`` and everything weight-related follows from it.
    # Defaults first, caller overrides last: a test that sets
    # ``backend="product"`` (the control case) must win over the
    # worker's default, and one that sets ``registry_id`` must be able
    # to reach a different recorded model. Explicit keyword arguments
    # cannot express "override me", so the merge happens in the dict.
    defaults: dict[str, Any] = {
        "registry_id": "deepface_facenet512",
        "candidate_id": "deepface-facenet512",
        "model_version": MODEL,
        "normalization": "Facenet",
        "interpreter": "3.13",
        "backend": "deepface-worker",
    }
    merged = defaults | (spec_overrides or {}) | over
    spec = CandidateSpec.from_registry_id(
        merged.pop("registry_id"),
        **{k: v for k, v in merged.items() if k not in _WEIGHT_FACTS},
    )
    return CandidateComparison.build(
        spec=spec,
        probe_rows=_rows(13),
        nontarget_rows=_rows(30),
        match_grid=list(MATCH_GRID),
        margin_grid=list(MARGIN_GRID),
        review_threshold=review_threshold,
        observed_weight_sha256=observed_weight_sha256,
        product_requires_python=PRODUCT_PY,
    )


def ComparisonOutcomeShim(rows):  # noqa: N802 - test helper
    from facecore.eval.model_comparison import ComparisonOutcome

    return ComparisonOutcome.from_rows(rows, set_name=SET_PROBE)


@contextmanager
def _fake_r_count(value: object):
    """Make ``summarize_r`` return a chosen ``r_top1_correct``.

    Patching the module's own reference is deliberate: the point is to
    feed a bad *value* through the real narrowing, not to reimplement
    it. A monkeypatch of ``static_baseline.summarize_r`` would not be
    seen by the already-imported name in ``model_comparison``, so the
    attribute is patched where it is used.
    """
    from facecore.eval import model_comparison as mc

    real = mc.summarize_r

    def _fake(rows, *, set_name):
        out = dict(real(rows, set_name=set_name))
        out["r_top1_correct"] = value
        return out

    mc.summarize_r = _fake  # type: ignore[assignment]
    try:
        yield
    finally:
        mc.summarize_r = real  # type: ignore[assignment]
