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
claim is not the same as enforcing it, so the routes that reached the
records are now closed separately:

- the mapping is a :class:`~types.MappingProxyType`, so assigning into
  it raises ``TypeError``;
- the records are captured in a **closure cell**, and the public
  resolver takes a single argument, so there is no parameter through
  which a caller can supply its own mapping — neither by keyword nor
  by position, and not by rebinding the module attribute either.

An earlier revision used a default argument for the second point. It
closed the rebinding route and left a *different* one wide open: because
the parameter was public, ``resolve_candidate_model(id, _records={...})``
let a caller hand in its own records and have them accepted. A `/`
marker does not fix that — it governs whether an argument may be named,
not whether it may be passed positionally. Both were measured before
this shape was chosen, not inferred.

**What this does and does not establish.** Given a file, these entries
prove the bytes on disk are the ones the repo records for that id. They
do **not** prove that a set of embeddings was computed from those
bytes: nothing here re-derives a vector and compares it. That requires
the crops the vectors came from, which the candidate interface does not
carry, so it is A3's work and it is **UNVERIFIED until then** — see
``ModelBindingError`` in ``model_comparison`` for the same boundary.

**Residual gaps, stated rather than implied.** An earlier version of
this section listed exactly two, and both were true — but the list was
not the boundary. Rewriting a function *object* is the route that makes
the other two moot, so it is stated here too and the count is left open
on purpose: a reader should not conclude that an unlisted route is
closed.

1. :func:`_resolve_from` is a module-level symbol, so a caller that
   imports it can supply its own records. It is internal by
   convention, not by enforcement — a leading underscore is a naming
   practice, not access control. Nothing here prevents that import; the
   closure shape only means no bypass is *advertised* by the public
   signature any more.

2. A caller that writes into a function's closure cells (via
   ``resolve_candidate_model.__closure__``) is still patching
   internals. The records are captured, so rebinding
   ``_CANDIDATE_RECORDS`` or ``CANDIDATE_MODELS`` does not reach them,
   and the resolver *function* is captured too, so rebinding
   ``_resolve_from`` does not either — the first version of this
   closure read that name as a global and was caught by measurement.

3. **Replacing ``__code__`` invalidates the premise the other two rest
   on.** A code object with a matching number of free variables can be
   assigned to either public function here, and the replacement need
   not read the cells at all — it can read a global the caller put in
   this module's namespace. Because the records then come from
   somewhere the closure never governed, *rebinding a module attribute
   also stops mattering*: routes that items 1 and 2 describe as closed
   become open in that state. Measured for both functions:
   ``resolve_candidate_model`` (two cells) and ``known_ids`` (one).

   How it was measured, and how to reproduce or refute it:

   - record the function's ``__code__`` object first;
   - assign the substitute, then assert ``f.__code__ is not <that
     object>`` — **that identity check is the only reliable signal.**
     ``co_freevars`` is not: CPython raises ``ValueError`` on an arity
     mismatch without swapping anything, and on success it reports
     whatever names the substitute happened to use, which may match
     the original exactly. In this module's own measurements a
     successful swap read back an identical ``('records', 'resolve')``.
     Checking freevars would have reported "nothing changed" about a
     swap that had worked;
   - then ask for an id that **the repo actually records** — e.g.
     ``deepface_arcface`` — not a newly invented one. An invented id is
     refused by the unknown-id check whether or not the substitution
     took, so it cannot distinguish the two states. Asking for a real
     id is what makes the hijack visible;
   - observe whether the returned manifest is the caller's.

So the accurate statement is: **the registry is closed against a caller
that uses the module as documented or reaches for its attributes by
name, and it is not established against one that imports an underscore
symbol deliberately, writes into a function's closure cells, or
replaces a function's ``__code__`` (or its ``__globals__``).** None of
these is closed here, and a module-level arrangement cannot close them
— a caller able to rewrite a function object can also rewrite the
module object it lives in.
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
#: writes, and the resolvers below capture this mapping in a closure
#: rather than consulting any module attribute — see the module
#: docstring for why closing one route was not enough.
_CANDIDATE_RECORDS: Mapping[str, Callable[[], ModelManifest]] = MappingProxyType(
    {
        "sface_2021dec_fp32": sface_2021dec_fp32,
        "sface_2021dec_int8bq": sface_2021dec_int8bq,
        "deepface_facenet512": facenet512,
        "deepface_arcface": arcface,
    }
)

#: Read-only view of the recorded ids, for callers that need to list or
#: iterate them. It is a name, not the reading path: :func:`known_ids`
#: and :func:`resolve_candidate_model` close over the mapping above, so
#: rebinding this attribute does not change what either of them reports.
CANDIDATE_MODELS: Mapping[str, Callable[[], ModelManifest]] = _CANDIDATE_RECORDS


def _resolve_from(
    registry_id: str,
    records: Mapping[str, Callable[[], ModelManifest]],
) -> ModelManifest:
    """Resolve ``registry_id`` against ``records``, or refuse.

    **Internal. Leading underscore is a convention, not access
    control** — see the module docstring, which records this as a
    residual gap rather than a closed route. A caller that imports this
    can supply its own records; the closure shape only means no bypass
    is advertised by the public signature.

    It exists as a separate module-level function so the two
    self-defences below — an entry that does not produce a manifest, and
    an entry recording no hash — stay *directly testable*. Those two
    checks took five review rounds to establish, and folding them into
    a closure would have made them reachable only by constructing a
    whole substitute registry.
    """
    entry = records.get(registry_id)
    if entry is None:
        raise UnknownCandidateModel(
            f"no repo-fixed record for candidate model {registry_id!r}; "
            f"known ids are {sorted(records)}. A candidate may "
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


def _resolver_using(
    records: Mapping[str, Callable[[], ModelManifest]],
    resolve: Callable[[str, Mapping[str, Callable[[], ModelManifest]]], ModelManifest],
) -> "Callable[[str], ModelManifest]":
    """Build the public resolver over ``records``.

    ``resolve`` is passed in rather than looked up as the module global
    ``_resolve_from`` so that the returned function captures it in a
    closure cell too. That is a second bypass route, and it was found by
    measurement rather than by reasoning: the inner function referenced
    ``_resolve_from`` by global name, so rebinding that name to a
    non-callable made resolution fail with ``TypeError: 'dict' object
    is not callable``. A rebind to a *function* would have been
    accepted silently instead.
    """

    def resolve_candidate_model(registry_id: str) -> ModelManifest:
        """Return the repo-fixed record for ``registry_id``, or refuse.

        The refusal names the known ids. A caller that typos an id
        should learn which ids exist rather than re-running to find out,
        and the list here is short enough to be printed without leaking
        anything — they are model names, not file paths.

        **One parameter, deliberately.** The records live in this
        function's closure rather than in a second argument, so there
        is no signature through which a caller can supply its own
        mapping: not by keyword, not by position, and not by
        rebinding a module attribute. An earlier revision passed them
        as a default argument, which closed the rebinding route and
        left ``resolve_candidate_model(id, _records={...})`` open to
        any caller; a ``/`` marker does not close that either.
        """
        return resolve(registry_id, records)

    return resolve_candidate_model


def _known_ids_using(
    records: Mapping[str, Callable[[], ModelManifest]],
) -> "Callable[[], tuple[str, ...]]":
    """Build the id accessor with ``records`` captured in a closure."""

    def known_ids() -> tuple[str, ...]:
        """Return the recorded ids, sorted.

        Captured rather than read through a module attribute, so
        rebinding a module name cannot change this answer.
        """
        return tuple(sorted(records))

    return known_ids


#: Public entry points. Bound to closures over the records, so neither
#: exposes a mapping parameter. ``_resolve_from`` is deliberately absent
#: from this list: it is internal, and the module docstring records that
#: its importability is a residual gap rather than a closed route.
resolve_candidate_model: "Callable[[str], ModelManifest]" = _resolver_using(
    _CANDIDATE_RECORDS, _resolve_from
)
known_ids: "Callable[[], tuple[str, ...]]" = _known_ids_using(_CANDIDATE_RECORDS)
