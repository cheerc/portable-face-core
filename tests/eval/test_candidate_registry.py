"""The candidate registry itself, tested apart from the comparison runner.

The registry is the thing that makes a spec's weight facts
unforgeable, so it gets tested on its own terms rather than only
through ``model_comparison``. Two properties matter and they are not
the same property:

1. **The recorded facts are real.** Each entry's hash is a checksum of
   an actual artifact and its dimension is that model's, not a
   plausible-looking number. A registry that records a wrong hash
   would make every downstream refusal correct-but-wrong: the
   comparison would refuse the genuine weights and accept nothing.

2. **The registry is closed.** A caller can select an entry and cannot
   add one, edit one, or construct a manifest and slip it in. This is
   the property both reviewers' bypasses defeated, and it is worth
   tests that do not go through the comparison runner, because the
   runner could grow a second door later. It is also worth tests that
   do not go *through the registry either* — a test that looks an id
   up in the mapping is satisfied by a mapping anyone can rewrite, so
   those tests would all be green on the bypass. The expected id set
   is pinned as a literal here for that reason.

The license state is tested too, and it is not decoration. The two
H5 entries exist under a decision that authorized *use* while
explicitly leaving license and provenance unproven. A future edit that
"tidies" ``UNPROVEN`` into a real license string would be the exact
kind of quiet upgrade the decision forbids, so the value is pinned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from facecore.contracts.manifest import ModelManifest, ProvenanceStatus
from facecore.eval import candidate_registry
from facecore.eval.candidate_registry import (
    CANDIDATE_MODELS,
    UnknownCandidateModel,
    known_ids,
    resolve_candidate_model,
)

#: The id set the repo records. Pinned by value rather than read from
#: the module so that a test cannot pass by agreeing with a registry
#: that has been altered underneath it. See the module docstring of
#: ``candidate_registry`` for the two bypasses this blocks.
RECORDED_IDS = (
    "deepface_arcface",
    "deepface_facenet512",
    "sface_2021dec_fp32",
    "sface_2021dec_int8bq",
)


class TestEveryRecordedFactIsWellFormed:
    def test_every_entry_resolves_to_a_manifest_with_a_hash(self) -> None:
        for registry_id in RECORDED_IDS:
            manifest = resolve_candidate_model(registry_id)
            assert isinstance(manifest, ModelManifest), registry_id
            assert manifest.weight_sha256 is not None, registry_id
            assert len(manifest.weight_sha256) == 64, registry_id
            assert all(c in "0123456789abcdef" for c in manifest.weight_sha256)

    def test_the_recorded_hashes_are_the_ones_the_weights_actually_have(self) -> None:
        """A wrong recorded hash would make every refusal correct-but-wrong.

        The two H5 checksums are of files fetched during A1 and hashed
        in a repo-external cache. They are recorded here as *integrity*
        records — which is not the same as a license review, see the
        next class — and a reader comparing against the candidate gate
        should find the same values.
        """
        facenet = resolve_candidate_model("deepface_facenet512")
        arcface = resolve_candidate_model("deepface_arcface")
        assert (
            facenet.weight_sha256
            == "3f76b5117a9ca574d536af8199e6720089eb4ad3dc7e93534496d88265de864f"
        )
        assert (
            arcface.weight_sha256
            == "6336979c0c602cae08d1122a66f4dfb862d059bbcd8ef80306aef2b2249b0c93"
        )

    def test_the_product_models_come_from_the_product_manifest(self) -> None:
        """SFace is recorded once, in the product's own module.

        A second copy of SFace's hash inside this file would be two
        authorities for one artifact, free to drift. The registry
        delegates instead, so the product manifest stays the only place
        that fact lives.
        """
        assert (
            resolve_candidate_model("sface_2021dec_fp32").weight_sha256
            == ModelManifest.sface_2021dec_fp32().weight_sha256
        )
        assert (
            resolve_candidate_model("sface_2021dec_int8bq").weight_sha256
            == ModelManifest.sface_2021dec_int8bq().weight_sha256
        )

    def test_dimensions_match_the_models_they_name(self) -> None:
        """SFace is 128-D; Facenet512 and ArcFace are both 512-D.

        The last two being equal is precisely why a dimension cannot
        tell them apart, which is why the hash is the discriminator.
        """
        assert resolve_candidate_model("sface_2021dec_fp32").embedding_dim == 128
        assert resolve_candidate_model("deepface_facenet512").embedding_dim == 512
        assert resolve_candidate_model("deepface_arcface").embedding_dim == 512

    def test_the_two_same_shaped_models_have_different_weights(self) -> None:
        """The fact the whole binding rests on.

        If these two hashes were equal, no byte-level check could ever
        tell a Facenet512 gallery from an ArcFace one, and the registry
        would provide no discrimination at all.
        """
        assert (
            resolve_candidate_model("deepface_facenet512").weight_sha256
            != resolve_candidate_model("deepface_arcface").weight_sha256
        )


class TestTheRegistryIsClosed:
    """The class name's claim, checked.

    Every other test in this file looks an id up *through* the
    registry, so each one is satisfied by a registry that anyone can
    rewrite. These are the tests that do not: they attack the
    registry itself. A previous revision of this class contained
    fifteen well-formed checks and none of this, while its docstring
    and its name both asserted the property.
    """

    def test_the_recorded_id_set_is_pinned_by_value(self) -> None:
        """The pin does not read the registry, or it proves nothing.

        Iterating ``CANDIDATE_MODELS`` and asserting the results are
        well-formed would pass on an injected entry, because the
        injected entry *is* well-formed by then. The expected set is
        therefore a literal here.
        """
        assert known_ids() == RECORDED_IDS
        assert tuple(sorted(CANDIDATE_MODELS)) == RECORDED_IDS

    def test_assigning_into_the_mapping_is_refused(self) -> None:
        """Route 1: in-place write. ``TypeError``, not a silent no-op."""
        with pytest.raises(TypeError):
            CANDIDATE_MODELS["deepface_evil"] = lambda: None  # type: ignore[index]

    def test_deleting_from_the_mapping_is_refused(self) -> None:
        """Removing a genuine entry is the same class of attack."""
        with pytest.raises((TypeError, AttributeError)):
            del CANDIDATE_MODELS["deepface_arcface"]  # type: ignore[attr-defined]

    def test_the_public_name_is_not_a_plain_dict(self) -> None:
        """Names the type, so a future edit back to ``dict`` is red."""
        assert not isinstance(CANDIDATE_MODELS, dict)

    def test_an_unknown_id_is_refused_and_names_the_real_ones(self) -> None:
        with pytest.raises(UnknownCandidateModel, match="no repo-fixed record"):
            resolve_candidate_model("facenet512")

    def test_the_refusal_lists_the_known_ids(self) -> None:
        """A caller that mistypes should learn the real names, not guess.

        The list is model identifiers, not file paths, so printing it
        discloses nothing about anyone's machine.
        """
        with pytest.raises(UnknownCandidateModel) as exc:
            resolve_candidate_model("sface")
        for known in RECORDED_IDS:
            assert known in str(exc.value)

    def test_an_empty_id_is_refused(self) -> None:
        with pytest.raises(UnknownCandidateModel):
            resolve_candidate_model("")

    def test_entries_are_builders_not_manifests(self) -> None:
        """Resolution happens at use, so a later edit cannot be pre-baked in.

        A registry of already-constructed manifests would still be
        closed to callers, but it would freeze every value at import
        time; keeping builders makes "resolved now" the only behaviour
        available.
        """
        for registry_id in RECORDED_IDS:
            entry = CANDIDATE_MODELS[registry_id]
            assert callable(entry), registry_id
            assert not isinstance(entry, ModelManifest), registry_id

    def test_resolving_twice_gives_equal_but_distinct_manifests(self) -> None:
        """Rebuilt per call, so no caller can mutate a shared instance.

        ``ModelManifest`` is frozen, so this is about the registry not
        handing out a single object whose fields something could later
        be derived from, rather than about a live mutation risk.
        """
        first = resolve_candidate_model("deepface_facenet512")
        second = resolve_candidate_model("deepface_facenet512")
        assert first == second
        assert first is not second

    def test_an_entry_that_returns_a_non_manifest_is_refused(self) -> None:
        """The isinstance guard, which was unpinned and carried a pragma.

        With a plain dict a caller could install a builder returning
        anything at all; the guard is what stops the caller-controlled
        value from reaching the fields every later check reads.
        """
        with pytest.raises(
            UnknownCandidateModel, match="did not produce a ModelManifest"
        ):
            resolve_candidate_model("x", _records={"x": lambda: "not a manifest"})

    def test_an_entry_recording_no_hash_is_refused(self) -> None:
        """The ``weight_sha256 is None`` guard, likewise unpinned.

        ``verify_sha256`` fails closed on a missing hash, so letting one
        through would mean binding a file against nothing.
        """
        manifest = ModelManifest.sface_2021dec_fp32()
        object.__setattr__(manifest, "weight_sha256", None)
        with pytest.raises(UnknownCandidateModel, match="records no weight_sha256"):
            resolve_candidate_model("x", _records={"x": lambda: manifest})


class TestRebindingTheModuleNameChangesNothing:
    """Route 2 and route 3: the proxy alone does not stop these.

    ``MappingProxyType`` refuses assignment *into* the mapping. It says
    nothing about rebinding the module attribute that names it, which
    is why the readers capture the mapping as a default argument
    instead of looking it up. ``monkeypatch`` restores the attribute
    afterwards, so these tests cannot pollute the rest of the suite.
    """

    def test_rebinding_the_module_attribute_does_not_change_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        genuine = resolve_candidate_model("deepface_arcface").weight_sha256
        monkeypatch.setattr(
            candidate_registry,
            "CANDIDATE_MODELS",
            {"deepface_arcface": lambda: "not a manifest"},
        )
        assert resolve_candidate_model("deepface_arcface").weight_sha256 == genuine

    def test_rebinding_the_private_name_does_not_change_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The name is underscored, which is a convention and not access control."""
        genuine = resolve_candidate_model("deepface_arcface").weight_sha256
        monkeypatch.setattr(
            candidate_registry,
            "_CANDIDATE_RECORDS",
            {"deepface_arcface": lambda: "not a manifest"},
        )
        assert resolve_candidate_model("deepface_arcface").weight_sha256 == genuine

    def test_reaching_the_module_through_sys_modules_changes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reviewer1's second route: no import by name required."""
        module = sys.modules["facecore.eval.candidate_registry"]
        genuine = module.resolve_candidate_model("deepface_arcface").weight_sha256
        monkeypatch.setattr(
            module, "CANDIDATE_MODELS", {"deepface_arcface": lambda: "not a manifest"}
        )
        assert (
            module.resolve_candidate_model("deepface_arcface").weight_sha256 == genuine
        )

    def test_known_ids_ignores_a_rebound_module_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same capture, on the accessor — otherwise it is a second door."""
        monkeypatch.setattr(
            candidate_registry,
            "CANDIDATE_MODELS",
            {"deepface_evil": (lambda: None)},
        )
        assert known_ids() == RECORDED_IDS


def test_the_residual_gap_is_stated_in_the_repo():
    """The bypass that is *not* closed must be written where a reader meets it.

    The assertions below were deliberately made narrow. A first version
    checked only that the strings ``__defaults__`` and ``A3`` appeared
    somewhere in the file, and a mutation that deleted the whole
    residual-gap paragraph left both strings present elsewhere — the
    test stayed green on the exact edit it was written to catch. So
    this pins the *sentences*, not the vocabulary.
    """
    source = Path(candidate_registry.__file__).read_text(encoding="utf-8")
    assert "**The residual gap, stated rather than implied.**" in source, (
        "the residual gap is no longer named as a paragraph in the module"
    )
    assert "still substitute a mapping" in source, (
        "the mechanism that survives the fix is no longer described"
    )
    assert "**A3 precondition**" in source, "the residual gap names no follow-up owner"


class TestTheResearchOnlyEntriesCarryTheirUnresolvedLicense:
    """The state decision d-32 left open, pinned where it is hardest to miss.

    d-32 authorized *downloading and using* Facenet512 and ArcFace for
    local research comparison. It did not resolve their weight license
    or training-data provenance, and it said so explicitly. The
    temptation this test exists to resist is the tidy-up: an entry with
    a checksum sitting in the repo reads like a reviewed artifact, and
    changing ``UNPROVEN`` to a real license string would look like
    tidying while being exactly the quiet upgrade the decision forbids.
    """

    @pytest.mark.parametrize("registry_id", ["deepface_facenet512", "deepface_arcface"])
    def test_the_weight_license_is_still_marked_unproven(
        self, registry_id: str
    ) -> None:
        manifest = resolve_candidate_model(registry_id)
        assert manifest.weight_license == "UNPROVEN", (
            f"{registry_id} is no longer marked UNPROVEN; that is a "
            "provenance change and needs a decision, not an edit"
        )
        assert manifest.provenance is ProvenanceStatus.UNRESOLVED

    @pytest.mark.parametrize("registry_id", ["deepface_facenet512", "deepface_arcface"])
    def test_the_provenance_note_says_the_artifact_is_not_committed(
        self, registry_id: str
    ) -> None:
        """The checksum is in the repo; the weights are not, and must not be.

        d-32 permits recording checksums and forbids committing the
        weight files. The note is where a future reader learns both
        halves, rather than having to infer the second from the
        absence of a file.
        """
        note = resolve_candidate_model(registry_id).provenance_note
        assert "not committed" in note
        assert "UNPROVEN" in note

    def test_the_product_models_keep_their_real_licenses(self) -> None:
        """The research entries are the exception, not the rule.

        SFace's Apache-2.0 record is settled in the product manifest;
        if this ever fails, the fix is in the product manifest, not
        here.
        """
        assert resolve_candidate_model("sface_2021dec_fp32").weight_license == (
            "Apache-2.0"
        )

    def test_no_weight_artifact_is_referenced_by_a_repo_relative_path(self) -> None:
        """A relative path into the repo would invite committing the file.

        The artifact URLs are remote and the provenance notes name no
        location on disk, so nothing in this module tells a reader
        where to drop a 95 MB file and call it recorded.
        """
        repo = Path(__file__).resolve().parents[2]
        for registry_id in RECORDED_IDS:
            manifest = resolve_candidate_model(registry_id)
            assert not manifest.artifact_url.startswith("/"), registry_id
            assert ".." not in manifest.artifact_url, registry_id
            assert not (repo / "weights").exists() or not any(
                (repo / "weights").iterdir()
            ), "a weight artifact appears to have been committed"


def test_the_two_candidate_hashes_match_the_candidate_gate() -> None:
    """One more place to catch drift, at the cost of a duplicated fact.

    The gate document and this module both state the two checksums.
    That is a deliberate redundancy: the gate is the record a reader
    consults about licensing, and a silent divergence between the two
    would be found nowhere. The assertion names the gate rather than
    re-deriving the value.
    """
    gate = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "research"
        / ("2026-09-30-deepface-candidate-gate.md")
    )
    text = gate.read_text(encoding="utf-8")
    for registry_id in ("deepface_facenet512", "deepface_arcface"):
        digest = resolve_candidate_model(registry_id).weight_sha256
        assert digest in text, f"{registry_id}'s hash is not in the candidate gate"
