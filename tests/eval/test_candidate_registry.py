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

import inspect
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

from facecore.contracts.manifest import ModelManifest, ProvenanceStatus
from facecore.eval import candidate_registry
from facecore.eval.candidate_registry import (
    CANDIDATE_MODELS,
    UnknownCandidateModel,
    _resolve_from,
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
            _resolve_from("x", {"x": lambda: "not a manifest"})

    def test_an_entry_recording_no_hash_is_refused(self) -> None:
        """The ``weight_sha256 is None`` guard, likewise unpinned.

        ``verify_sha256`` fails closed on a missing hash, so letting one
        through would mean binding a file against nothing.
        """
        manifest = ModelManifest.sface_2021dec_fp32()
        object.__setattr__(manifest, "weight_sha256", None)
        with pytest.raises(UnknownCandidateModel, match="records no weight_sha256"):
            _resolve_from("x", {"x": lambda: manifest})


class TestThePublicResolverTakesNoMapping:
    """The bypass three independent measurements found, and the closure
    that closes it.

    An earlier revision passed the records as a *default argument*. That
    closed the module-rebinding route and left this one wide open:
    ``resolve_candidate_model(id, _records={...})`` was a public,
    documented-looking way for a caller to supply its own records. Three
    sources found it independently — the author, reviewer 1, and the
    lead, each measuring it rather than inferring it.

    **Positional passing is covered separately and deliberately.** A
    ``/`` marker was the lead's first instruction for closing this, and
    it does not work: ``/`` governs whether an argument may be *named*,
    not whether it may be *passed*. Both placements left
    ``resolve_candidate_model("x", {...})`` accepted. A test that only
    tried the keyword form would have passed against that broken fix,
    which is why the positional case is its own test rather than another
    parametrization of the keyword one.
    """

    @staticmethod
    def _injected(registry_id: str, value: str = "injected") -> dict[str, object]:
        return {registry_id: (lambda: value)}

    def test_supplying_records_by_keyword_is_refused(self) -> None:
        """Keyword: the form the default-argument revision was open to."""
        with pytest.raises(TypeError):
            resolve_candidate_model(  # type: ignore[call-arg]
                "deepface_arcface", _records=self._injected("deepface_arcface")
            )

    def test_supplying_records_by_keyword_under_any_name_is_refused(self) -> None:
        """A renamed parameter is not a new door.

        The argument is refused because there is no second parameter at
        all, not because this particular name is known. Asserting on the
        *absence* of a second parameter is what makes that true, so it
        is asserted separately below.
        """
        with pytest.raises(TypeError):
            resolve_candidate_model(  # type: ignore[call-arg]
                "deepface_arcface", records=self._injected("deepface_arcface")
            )

    def test_supplying_records_positionally_is_refused(self) -> None:
        """Positional: the route a ``/`` fix would have left open."""
        with pytest.raises(TypeError):
            resolve_candidate_model(  # type: ignore[call-arg]
                "deepface_arcface", self._injected("deepface_arcface")
            )

    def test_the_public_resolver_declares_exactly_one_parameter(self) -> None:
        """The structural claim the three refusals rest on.

        Asserting the signature rather than the exceptions: a caller
        supplying a second argument fails on arity, and that is only a
        property worth keeping if there is in fact no second parameter.
        """
        parameters = inspect.signature(resolve_candidate_model).parameters
        assert list(parameters) == ["registry_id"], parameters
        assert parameters["registry_id"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    def test_the_public_resolver_has_no_writable_defaults(self) -> None:
        """``__defaults__`` is the attribute the earlier shape exposed.

        With a closure there is nothing there to rewrite, so the whole
        ``__defaults__`` route is gone rather than merely narrowed.
        """
        assert resolve_candidate_model.__defaults__ is None
        assert known_ids.__defaults__ is None

    def test_the_records_live_in_a_closure_rather_than_a_parameter(self) -> None:
        """Where the captured mapping actually is, stated as a fact."""
        cells = resolve_candidate_model.__closure__ or ()
        assert any(
            isinstance(cell.cell_contents, MappingProxyType) for cell in cells
        ), cells

    def test_rebinding_every_module_name_leaves_resolution_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All three names, rebound to a different mapping in turn.

        Each rebinding is a separate assignment rather than one test
        that does all three, so a failure names which name leaked.
        """
        genuine = resolve_candidate_model("deepface_arcface").weight_sha256
        for name in ("CANDIDATE_MODELS", "_CANDIDATE_RECORDS", "_resolve_from"):
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(
                    candidate_registry, name, self._injected("deepface_arcface")
                )
                assert resolve_candidate_model("deepface_arcface").weight_sha256 == (
                    genuine
                ), name

    def test_known_ids_also_takes_no_mapping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The accessor had the same shape and the same fix."""
        with pytest.raises(TypeError):
            known_ids(  # type: ignore[call-arg]
                _records=self._injected("evil")
            )
        with pytest.raises(TypeError):
            known_ids(self._injected("evil"))  # type: ignore[call-arg]
        monkeypatch.setattr(
            candidate_registry, "CANDIDATE_MODELS", self._injected("evil")
        )
        assert known_ids() == RECORDED_IDS


class TestTheInternalHelperIsDocumentedAsAResidualGap:
    """Lead's third condition, turned into something executable.

    ``_resolve_from`` is a module-level symbol, so it *can* be imported
    and handed a caller-supplied mapping. That is not closed, and the
    module docstring says so. This test fails if the docstring ever
    stops saying so — which is the failure mode worth guarding: a reader
    concluding the registry is fully closed.
    """

    def test_the_module_docstring_states_the_helper_is_importable(self) -> None:
        source = Path(candidate_registry.__file__).read_text(encoding="utf-8")
        assert "is a module-level symbol, so a caller that" in source, (
            "the docstring no longer states that _resolve_from is importable"
        )
        assert "is not established against one that imports an underscore" in source, (
            "the docstring no longer bounds what the closure closes"
        )

    def test_the_helper_docstring_calls_itself_internal(self) -> None:
        doc = candidate_registry._resolve_from.__doc__ or ""
        assert "**Internal." in doc
        assert "not access" in doc, (
            "the helper does not say the underscore is a convention"
        )

    def test_the_helper_is_not_exported_in_all(self) -> None:
        """There is no ``__all__`` here, so the check is the public
        surface: what ``from module import *`` would pick up.

        ``candidate_registry`` has no ``__all__`` of its own (the one at
        ``model_comparison.py:897`` is a different module), so an
        unprefixed import is the only export list that exists. This test
        pins that ``_resolve_from`` is reachable only under its
        underscore name.
        """
        exported = {
            name for name in dir(candidate_registry) if not name.startswith("_")
        }
        assert "_resolve_from" not in exported
        assert "resolve_candidate_model" in exported


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
    """Every gap that is *not* closed must be written where a reader meets it.

    The assertions below were deliberately made narrow, and twice had to
    be corrected. A first version checked only that the strings
    ``__defaults__`` and ``A3`` appeared somewhere in the file, and a
    mutation that deleted the whole residual-gap paragraph left both
    strings present elsewhere — the test stayed green on the exact edit
    it was written to catch. A second version pinned the sentences of
    *one* gap, which passed just as silently when the closure rewrite
    replaced that gap with a differently-worded one. Pinning prose
    means pinning it against the prose that is actually there, so each
    assertion names a gap the current module docstring states.
    """
    source = Path(candidate_registry.__file__).read_text(encoding="utf-8")
    flat = " ".join(source.split())
    assert "**Two residual gaps, stated rather than implied.**" in flat, (
        "the residual gaps are no longer named as a section in the module"
    )
    assert "is a module-level symbol, so a caller that" in flat, (
        "the docstring no longer states that _resolve_from is importable"
    )
    assert "a naming practice, not access control" in flat, (
        "the docstring no longer says the underscore is a convention, not enforcement"
    )
    assert "writes into the closure's cells" in flat, (
        "the docstring no longer states the closure-cell route"
    )
    assert "is not established against one that imports an underscore" in flat, (
        "the docstring no longer bounds what the closure closes"
    )


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
