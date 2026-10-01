"""D6 A2: the repo-fixed record of what each candidate's weights are.

Why this module exists, stated as the defect it removes.

``CandidateSpec`` used to carry ``weight_sha256`` and ``embedding_dim``
as free fields. Every check built on them compared what the caller
*said* against what the caller *pointed at*: ``sha256_of_file(
weight_path)`` versus ``spec.weight_sha256`` is the same fact read
twice, and both facts came from the same caller. A caller who wrote the
declaration to match the file it supplied — Facenet512's name over
ArcFace's bytes, with ArcFace's own hash declared — satisfied every
check and got a complete result reported under the wrong model's name.
Two independent reviewers found that path; it was not closed by adding
another comparison.

What closes it is removing the freedom rather than adding a check. The
entries here are module-level functions with their values written into
this file, exactly as ``ModelManifest`` records the product's own
weights through its classmethods. A caller may **select** an entry; a
caller may not **write** one, and an unknown id is a refusal rather than
a new entry. The product's models come from
:mod:`facecore.contracts.manifest` so there is one authority for SFace,
not two.

**Two mechanisms, because one does not suffice.** An earlier revision
exposed the entries as a plain module-level ``dict`` and asserted this
property in prose. Two reviewers independently reproduced the bypass:
one assignment to that dict — no function monkeypatched, and in one
case using the *honest* registry id — let a caller publish one model's
embeddings under another's name with every check passing. Writing the
claim is not the same as enforcing it, so the two routes that reached
the records are now closed separately:

- the mapping is a :class:`~types.MappingProxyType`, so assigning into
  it raises ``TypeError``;
- every read goes through a default argument captured when the
  function was defined, so rebinding the module attribute — by name or
  through ``sys.modules`` — does not change what is read.

**What this does and does not establish.** Given a file, these entries
prove the bytes on disk are the ones the repo records for that id. They
do **not** prove that a set of embeddings was computed from those
bytes: nothing here re-derives a vector and compares it. That requires
the crops the vectors came from, which the candidate interface does not
carry, so it is A3's work and it is **UNVERIFIED until then** — see
``ModelBindingError`` in ``model_comparison`` for the same boundary.

**The residual gap, stated rather than implied.** A caller that rewrites
``resolve_candidate_model.__defaults__`` can still substitute a mapping,
because that attribute is reachable from the function object. The two
mechanisms above do not close that, and no arrangement of module-level
state does — closing it needs the caller to have nothing to substitute.
That is an **A3 precondition**: the wiring code must not accept a
registry reference from its caller. Until A3 does that, treat the
binding as strong against a caller that reads and rewrites module
attributes, and **not** established against one that patches function
objects.
"""

from __future__ import annotations

from typing import Callable, Mapping
from types import MappingProxyType

from facecore.contracts.manifest import ModelManifest, ProvenanceStatus


class UnknownCandidateModel(ValueError):
    """No repo-fixed record exists for the requested model id.

    Deliberately *not* a "register it and carry on" path. A registry
    that accepts new entries from the caller is the defect this module
    was written to remove, so an unknown id stops the run.
    """


def sface_2021dec_fp32() -> ModelManifest:
    """The product's own SFace artifact — the D5/D6 control."""
    return ModelManifest.sface_2021dec_fp32()


def sface_2021dec_int8bq() -> ModelManifest:
    """The product's quantized SFace, recorded in the product manifest."""
    return ModelManifest.sface_2021dec_int8bq()


def facenet512() -> ModelManifest:
    """DeepFace Facenet512, research use only.

    **Research-only entry. Not a product model.** Decision
    ``d-20260930132705762549-32`` authorized downloading and using
    Facenet512/ArcFace for local research comparison under limits that
    are *not* relaxed by this entry existing:

    - local research comparison only; not distributed, not commercial,
      never in product runtime;
    - the weight file itself is **not** committed — only this record;
    - **the exact weight license and provenance are still unproven.**
      The checksum below is an integrity record, not a license review.
      Its presence must not be read as authorization having been
      resolved; that gap is unchanged from A1 and is restated here so
      a future reader meets it at the hash rather than in a report they
      may not open.
    """
    return ModelManifest(
        model_id="deepface_facenet512",
        source_repo="serengil/deepface",
        artifact_url=(
            "https://github.com/serengil/deepface/releases/download/"
            "v0.0.93/facenet512_weights.h5"
        ),
        retrieval_date="2026-09-30",
        code_license="MIT",
        weight_license="UNPROVEN",
        license_locator=(
            "No weight-license text accompanies the artifact; see the "
            "candidate gate's provenance section"
        ),
        weight_sha256=(
            "3f76b5117a9ca574d536af8199e6720089eb4ad3dc7e93534496d88265de864f"
        ),
        provenance=ProvenanceStatus.UNRESOLVED,
        provenance_note=(
            "Research-only entry under d-20260930132705762549-32. "
            "94,955,648 bytes, downloaded to a repo-external cache and "
            "hashed there; the artifact is not committed. Exact weight "
            "license and training-data provenance remain UNPROVEN."
        ),
        embedding_dim=512,
        input_width=160,
        input_height=160,
        mobile_usability=None,
    )


def arcface() -> ModelManifest:
    """DeepFace ArcFace, research use only.

    Same authorization and same unresolved license state as
    :func:`facenet512`; see that docstring rather than repeating it.
    """
    return ModelManifest(
        model_id="deepface_arcface",
        source_repo="serengil/deepface",
        artifact_url=(
            "https://github.com/serengil/deepface/releases/download/"
            "v0.0.93/arcface_weights.h5"
        ),
        retrieval_date="2026-09-30",
        code_license="MIT",
        weight_license="UNPROVEN",
        license_locator=(
            "No weight-license text accompanies the artifact; see the "
            "candidate gate's provenance section"
        ),
        weight_sha256=(
            "6336979c0c602cae08d1122a66f4dfb862d059bbcd8ef80306aef2b2249b0c93"
        ),
        provenance=ProvenanceStatus.UNRESOLVED,
        provenance_note=(
            "Research-only entry under d-20260930132705762549-32. "
            "137,026,640 bytes, downloaded to a repo-external cache and "
            "hashed there; the artifact is not committed. Exact weight "
            "license and training-data provenance remain UNPROVEN."
        ),
        embedding_dim=512,
        input_width=112,
        input_height=112,
        mobile_usability=None,
    )


#: Every model a candidate may name. Keyed by the id a ``CandidateSpec``
#: carries; the value is a zero-argument builder defined in this file,
#: so adding a model means writing a function here, not registering a
#: value from a caller. The type is a callable rather than a
#: ``ModelManifest`` on purpose: a manifest is frozen and cheap to
#: rebuild, and a callable makes "resolved at use, not captured at
#: import" visible in the type itself.
#:
#: The mapping is a proxy because this name is importable: an earlier
#: revision used a plain ``dict`` here and two reviewers restored the
#: whole bypass with one assignment to it. The proxy refuses in-place
#: writes, and the readers below deliberately do not consult this name —
#: see the module docstring for why closing one route was not enough.
_CANDIDATE_RECORDS: Mapping[str, Callable[[], ModelManifest]] = MappingProxyType(
    {
        "sface_2021dec_fp32": sface_2021dec_fp32,
        "sface_2021dec_int8bq": sface_2021dec_int8bq,
        "deepface_facenet512": facenet512,
        "deepface_arcface": arcface,
    }
)

#: Read-only view of the recorded ids, for callers that need to list or
#: iterate them. ``MappingProxyType`` blocks assignment into the
#: mapping; the captured default argument is what keeps a rebinding of
#: the module attribute from changing what this reports. Both are
#: needed, and neither alone closes the other route.
CANDIDATE_MODELS: Mapping[str, Callable[[], ModelManifest]] = _CANDIDATE_RECORDS


def known_ids(
    _records: Mapping[str, Callable[[], ModelManifest]] = _CANDIDATE_RECORDS,
) -> tuple[str, ...]:
    """Return the recorded ids, sorted.

    The record set is pinned rather than read through a module
    attribute, so rebinding a module name cannot change this answer.
    """

    return tuple(sorted(_records))


def resolve_candidate_model(
    registry_id: str,
    _records: Mapping[str, Callable[[], ModelManifest]] = _CANDIDATE_RECORDS,
) -> ModelManifest:
    """Return the repo-fixed record for ``registry_id``, or refuse.

    The refusal names the known ids. A caller that typos an id should
    learn which ids exist rather than re-running to find out, and the
    list here is short enough to be printed without leaking anything —
    they are model names, not file paths.

    ``_records`` is a default argument rather than a module lookup on
    purpose: a caller that rebinds :data:`CANDIDATE_MODELS` on this
    module does not change what this function reads. That closes the
    rebinding route the proxy alone leaves open. It does not close a
    caller that rewrites ``__defaults__`` itself — see the module
    docstring, which records that as an A3 precondition instead of
    leaving it for the reader to discover.
    """
    entry = _records.get(registry_id)
    if entry is None:
        raise UnknownCandidateModel(
            f"no repo-fixed record for candidate model {registry_id!r}; "
            f"known ids are {sorted(_records)}. A candidate may "
            "select a recorded model, not introduce one: a registry that "
            "accepts new entries from the caller is exactly the defect "
            "the registry exists to remove."
        )
    manifest = entry()
    if not isinstance(manifest, ModelManifest):
        raise UnknownCandidateModel(
            f"registry entry {registry_id!r} did not produce a ModelManifest"
        )
    if manifest.weight_sha256 is None:
        # `verify_sha256` fails closed on a missing hash; a candidate
        # with no recorded hash has nothing to bind against, so it is
        # refused here rather than compared against None later.
        raise UnknownCandidateModel(
            f"candidate model {registry_id!r} records no weight_sha256; "
            "there is nothing to bind a file against"
        )
    return manifest
