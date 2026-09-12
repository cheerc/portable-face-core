"""Encrypted export/import with key re-homing (1B plan §11 Task 6, §8).

File layout (JSON): ``{"metadata": {...}, "envelope": {...}}`` where the
envelope is an AES-256-GCM seal over the canonical inner package, keyed by
``K_envelope``. KDF: pinned Argon2id (64 MiB, 3 iterations, parallelism 1,
version 1); derives ``K_wrap`` + ``K_envelope`` (2×256-bit halves of one
64-byte raw output). No Argon2id at runtime, or an unknown kdf
algorithm/version in the header → ``UnsupportedKdfError`` (exit 4), never
a weaker fallback.

Import order: envelope MAC → KDF-parameter check → 9-field predicate
(exit 3, zero writes) → DEK re-homing → transactional insert.

The 9-field predicate uses the plan's hard subset (1,4,5,6,7,8,9):
any difference → ``ModelIncompatibilityError`` (exit 3) with zero DB
writes. Detector/preprocessing drift (2,3) reports ``MIGRATION_REQUIRED``
and proceeds (Task 7 owns the re-embedder).
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from facecore.contracts.candidate import (
    CandidateStatus,
    CandidateTemplate,
    EvidenceRecord,
)
from facecore.contracts.crypto import (
    EncryptedBlob,
    KeyProviderProtocol,
    StoreCorruptionError,
    UnsupportedKdfError,
    WrappedKey,
)
from facecore.contracts.export import (
    ExportContainer,
    ExportedIdentity,
    ExportMetadata,
    TombstoneRecord,
)
from facecore.contracts.migration import (
    ModelIncompatibilityError,
    ModelMigrationManifest,
)
from facecore.contracts.policy import GovernancePolicy
from facecore.contracts.template import TemplateRevision
from facecore.errors import StoreError
from facecore.storage.cipher import build_canonical_aad

EXPORT_FORMAT_VERSION = 1
_KDF_ALGORITHM = "argon2id"
_KDF_VERSION = 1
_KDF_MEMORY_KIB = 65536
_KDF_TIME_COST = 3
_KDF_PARALLELISM = 1
_SALT_BYTES = 16

# Plan §8 hard subset: fields (1,4,5,6,7,8,9), aligned with the
# contract helper hard list (migration.HARD_INCOMPATIBLE_FIELDS).
# Detector/preprocessing drift (2,3) is Task 7's migration trigger,
# not an incompatibility.
_HARD_FIELDS: tuple[str, ...] = (
    "embedder_artifact_hash",
    "tensor_layout",
    "normalization_contract",
    "embedding_dimension",
    "numerical_precision",
    "quantization_type",
    "execution_runtime",
    "score_metric",
)


@dataclass(frozen=True)
class ExportManifest:
    identities: int
    active_templates: int
    retired_templates: int
    candidates: int
    archive: str


@dataclass(frozen=True)
class ImportResult:
    identities: int
    active_templates: int
    retired_templates: int
    candidates: int
    compatibility: str
    policy: GovernancePolicy


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _derive_subkeys(passphrase: str, salt: bytes) -> tuple[bytes, bytes]:
    """Derive (K_wrap, K_envelope) via pinned Argon2id; no fallback."""
    try:
        from argon2.low_level import Type, hash_secret_raw
    except ImportError as exc:
        raise UnsupportedKdfError(
            "argon2id unavailable at runtime; export/import is fail-closed"
        ) from exc
    raw = hash_secret_raw(
        secret=passphrase.encode("utf-8"),
        salt=salt,
        time_cost=_KDF_TIME_COST,
        memory_cost=_KDF_MEMORY_KIB,
        parallelism=_KDF_PARALLELISM,
        hash_len=64,
        type=Type.ID,
        version=19,
    )
    return raw[:32], raw[32:]


def _check_kdf_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("kdf_algorithm") != _KDF_ALGORITHM:
        raise UnsupportedKdfError(
            f"unsupported kdf_algorithm: {metadata.get('kdf_algorithm')!r}"
        )
    if metadata.get("kdf_version") != _KDF_VERSION:
        raise UnsupportedKdfError(
            f"unsupported kdf_version: {metadata.get('kdf_version')!r}"
        )
    for field_name, expected in (
        ("kdf_memory_cost_kib", _KDF_MEMORY_KIB),
        ("kdf_time_cost", _KDF_TIME_COST),
        ("kdf_parallelism", _KDF_PARALLELISM),
        ("export_format_version", EXPORT_FORMAT_VERSION),
    ):
        if metadata.get(field_name) != expected:
            raise UnsupportedKdfError(
                f"unsupported {field_name}: {metadata.get(field_name)!r}"
            )


def _envelope_aad(salt_hex: str) -> bytes:
    return build_canonical_aad(
        "export-envelope", f"v{EXPORT_FORMAT_VERSION}", salt_hex
    )


def _check_compatibility(
    archived: ModelMigrationManifest, runtime: ModelMigrationManifest
) -> str:
    stored = archived.to_dict()
    live = runtime.to_dict()
    for field_name in _HARD_FIELDS:
        if stored.get(field_name) != live.get(field_name):
            raise ModelIncompatibilityError(
                f"Model incompatibility in field {field_name!r}: "
                f"archive={stored.get(field_name)!r} != "
                f"runtime={live.get(field_name)!r}"
            )
    for field_name in ("detector_generation", "preprocessing_generation"):
        if stored.get(field_name) != live.get(field_name):
            return "MIGRATION_REQUIRED"
    return "COMPATIBLE"


def _key_owners(container: dict[str, Any]) -> dict[str, str]:
    """Map every record key_id to its owning identity_id."""
    owners: dict[str, str] = {}
    for template in list(container.get("active_templates", [])) + list(
        container.get("retired_templates", [])
    ):
        owners[str(template["key_id"])] = str(template["identity_id"])
    for candidate in container.get("candidates", []):
        owners[str(candidate["key_id"])] = str(candidate["identity_id"])
    return owners


def _evidence_records(candidate_id: str, raw: str) -> tuple[EvidenceRecord, ...]:
    """Per-entry validated evidence; garbage fails the whole export closed.

    Fail-closed is preserved (no partial archive), but refusal is a
    structured StoreError naming the offender — never an uncaught
    float()/int() crash mid-export.
    """
    try:
        parsed: object = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise StoreError(
            f"candidate {candidate_id}: evidence log is not a JSON list"
        ) from exc
    if not isinstance(parsed, list):
        raise StoreError(
            f"candidate {candidate_id}: evidence log is not a JSON list"
        )
    records: list[EvidenceRecord] = []
    for index, event in enumerate(parsed):
        if not isinstance(event, dict):
            raise StoreError(
                f"candidate {candidate_id}: evidence entry {index} not an object"
            )
        try:
            sequence_number = int(event.get("sequence_number", 0))
        except (TypeError, ValueError) as exc:
            raise StoreError(
                f"candidate {candidate_id}: evidence entry {index}"
                " has non-integer sequence_number"
            ) from exc
        raw_score = event.get("score")
        score: float | None = None
        if raw_score is not None:
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise StoreError(
                    f"candidate {candidate_id}: evidence entry {index}"
                    f" has non-numeric score {raw_score!r}"
                ) from exc
            if not (
                score == score and score not in (float("inf"), float("-inf"))
            ):
                raise StoreError(
                    f"candidate {candidate_id}: evidence entry {index}"
                    f" has non-finite score {raw_score!r}"
                )
        records.append(
            EvidenceRecord(
                event_type=str(event.get("event_type", "")),
                timestamp=str(event.get("timestamp", "")),
                sequence_number=sequence_number,
                score=score,
            )
        )
    return tuple(records)


def _collect_container(
    repo: Any, manifest: ModelMigrationManifest, policy: GovernancePolicy
) -> tuple[ExportContainer, list[dict[str, Any]], list[dict[str, Any]]]:
    """Read complete governance state; sidecars ride inside the envelope.

    Deletion-pending (non-active) identities are EXCLUDED with their full
    closure (templates, candidates, revisions, events, tombstones): a
    tombstone window is a deletion in flight, and exporting it would land
    an unmanageable (deleted, tombstone-less) identity on the destination
    whose rows/DEKs strand outside the erasure invariant. The source-side
    deletion continues to completion; the destination never learns it.
    """
    con: sqlite3.Connection | None = repo.connection
    if con is None:
        raise StoreError("repository not initialized")
    identities = tuple(
        ExportedIdentity(
            identity_id=str(row[0]),
            display_name=str(row[1]),
            status=str(row[2]),
            current_revision=int(row[3]),
        )
        for row in con.execute(
            "SELECT id, display_name, status, current_revision"
            " FROM identities WHERE status = 'active' ORDER BY id"
        ).fetchall()
    )
    live_ids = {identity.identity_id for identity in identities}
    template_cols = (
        "ft.id, ft.identity_id, ft.generation_id, ft.model_version,"
        " ft.embedding_dim, ft.revision_number, ft.revision_supersedes,"
        " ft.key_id, ft.embedding_blob, ft.exemplar_blob,"
        " ft.exemplar_crop_box, ft.exemplar_landmarks, ft.exemplar_margin,"
        " ft.quality_score, ft.utility_score"
    )
    active = tuple(
        repo._hydrate_template(tuple(row))
        for row in con.execute(
            f"SELECT {template_cols} FROM face_templates ft"
            " JOIN identities i ON i.id = ft.identity_id"
            " WHERE ft.status = 'active' AND i.status = 'active'"
            " ORDER BY ft.id"
        ).fetchall()
    )
    retired = tuple(
        repo._hydrate_template(tuple(row))
        for row in con.execute(
            f"SELECT {template_cols} FROM face_templates ft"
            " JOIN identities i ON i.id = ft.identity_id"
            " WHERE ft.status = 'retired' AND i.status = 'active'"
            " ORDER BY ft.id"
        ).fetchall()
    )
    candidates = tuple(
        CandidateTemplate(
            template_id=str(row[0]),
            identity_id=str(row[1]),
            generation_id=str(row[2]),
            status=CandidateStatus(str(row[3])),
            key_id=str(row[4]),
            encrypted_embedding=repo._parse_blob(
                bytes(row[5]), "candidate_templates", str(row[0])
            ),
            encrypted_exemplar=(
                repo._parse_blob(bytes(row[6]), "candidate_templates", str(row[0]))
                if row[6] is not None
                else None
            ),
            exemplar_crop_box=repo._json_crop(
                str(row[7]) if row[7] is not None else None
            ),
            exemplar_landmarks=repo._json_landmarks(
                str(row[8]) if row[8] is not None else None
            ),
            quality_score=float(row[9]),
            additional_corroboration_count=int(row[10]),
            evidence_log=_evidence_records(str(row[0]), str(row[11])),
            expires_at=str(row[12]),
            created_at=str(row[13]),
        )
        for row in con.execute(
            "SELECT c.id, c.identity_id, c.generation_id, c.status, c.key_id,"
            " c.embedding_blob, c.exemplar_blob, c.exemplar_crop_box,"
            " c.exemplar_landmarks, c.quality_score,"
            " c.additional_corroboration_count, c.evidence_log,"
            " c.expires_at, c.created_at"
            " FROM candidate_templates c"
            " JOIN identities i ON i.id = c.identity_id"
            " WHERE i.status = 'active' ORDER BY c.id"
        ).fetchall()
    )
    detail_rows = con.execute(
        "SELECT r.identity_id, r.revision, r.active_template_ids,"
        " r.retired_template_ids FROM template_revisions r"
        " JOIN identities i ON i.id = r.identity_id"
        " WHERE i.status = 'active'"
        " ORDER BY r.identity_id, r.revision"
    ).fetchall()
    revisions = tuple(
        TemplateRevision(
            revision=int(row[1]),
            template_id=f"{row[0]}@r{row[1]}",
            supersedes=None,
        )
        for row in detail_rows
    )
    revision_details = [
        {
            "identity_id": str(row[0]),
            "revision": int(row[1]),
            "active_template_ids": str(row[2]),
            "retired_template_ids": str(row[3]),
        }
        for row in detail_rows
    ]
    # Tombstones of live identities only; pending-tombstone rows belong
    # to excluded identities and never cross the boundary (see above).
    tombstones = tuple(
        TombstoneRecord(
            target_type=str(row[0]),
            target_id=str(row[1]),
            key_ids=tuple(json.loads(str(row[2]))),
            status=str(row[3]),
        )
        for row in con.execute(
            "SELECT d.target_type, d.target_id, d.key_ids_json, d.status"
            " FROM deletion_tombstones d ORDER BY d.id"
        ).fetchall()
        if str(row[1]) in live_ids or str(row[0]) != "identity"
    )
    # Anonymized match events: identity associations never leave the
    # source store (NULL on export, NULL on insert).
    match_events = [
        {
            "id": str(row[0]),
            "timestamp": str(row[1]),
            "sequence_number": int(row[2]),
            "status": str(row[3]),
            "decision_score": float(row[4]),
            "runner_up_score": float(row[5]) if row[5] is not None else None,
            "candidate_created": int(row[6]),
            "actor": row[7],
        }
        for row in con.execute(
            "SELECT id, timestamp, sequence_number, status, decision_score,"
            " runner_up_score, candidate_created, actor"
            " FROM match_events ORDER BY timestamp, sequence_number"
        ).fetchall()
    ]
    container = ExportContainer(
        metadata=ExportMetadata(kdf_salt_hex="", created_at=_utcnow()),
        policy=policy,
        model_manifest=manifest,
        identities=identities,
        active_templates=active,
        retired_templates=retired,
        candidates=candidates,
        revisions=revisions,
        tombstones=tombstones,
    )
    return container, revision_details, match_events


def export_identities(
    repo: Any,
    manifest: ModelMigrationManifest,
    path: str | Path,
    passphrase: str,
    policy: GovernancePolicy | None = None,
) -> ExportManifest:
    """Serialize complete governance state into an authenticated archive."""
    resolved_policy = policy or GovernancePolicy.provisional_v1()
    container, revision_details, match_events = _collect_container(
        repo, manifest, resolved_policy
    )
    container_dict = container.to_dict()
    container_dict["revision_details"] = revision_details
    container_dict["match_events"] = match_events
    # Inner layer: wrap each record DEK under K_wrap (AES-256-GCM,
    # key_id as associated data so wraps cannot be swapped).
    salt = os.urandom(_SALT_BYTES)
    k_wrap, k_envelope = _derive_subkeys(passphrase, salt)
    key_ids: set[str] = set()
    active_payloads = container_dict["active_templates"]
    retired_payloads = container_dict["retired_templates"]
    candidate_payloads = container_dict["candidates"]
    assert isinstance(active_payloads, list)
    assert isinstance(retired_payloads, list)
    assert isinstance(candidate_payloads, list)
    for template in active_payloads + retired_payloads:
        assert isinstance(template, dict)
        key_ids.add(str(template["key_id"]))
    for candidate in candidate_payloads:
        assert isinstance(candidate, dict)
        key_ids.add(str(candidate["key_id"]))
    source_provider: KeyProviderProtocol = repo.key_provider
    wrapped_deks: dict[str, dict[str, str]] = {}
    for key_id in sorted(key_ids):
        try:
            raw_dek = source_provider.get_key(key_id)
        except Exception as exc:
            raise StoreCorruptionError(
                f"export cannot wrap absent DEK: {key_id}"
            ) from exc
        nonce = os.urandom(12)
        sealed = AESGCM(k_wrap).encrypt(
            nonce, raw_dek, key_id.encode("utf-8")
        )
        wrapped_deks[key_id] = {
            "nonce_hex": nonce.hex(),
            "sealed_hex": sealed.hex(),
        }
    inner = json.dumps(
        {"container": container_dict, "wrapped_deks": wrapped_deks},
        sort_keys=True,
    ).encode("utf-8")
    envelope_nonce = os.urandom(EncryptedBlob.NONCE_BYTES)
    envelope_sealed = AESGCM(k_envelope).encrypt(
        envelope_nonce, inner, _envelope_aad(salt.hex())
    )
    archive = {
        "metadata": {
            "export_format_version": EXPORT_FORMAT_VERSION,
            "kdf_algorithm": _KDF_ALGORITHM,
            "kdf_version": _KDF_VERSION,
            "kdf_salt": salt.hex(),
            "kdf_memory_cost_kib": _KDF_MEMORY_KIB,
            "kdf_time_cost": _KDF_TIME_COST,
            "kdf_parallelism": _KDF_PARALLELISM,
            "created_at": _utcnow(),
        },
        "envelope": {
            "format_version": EncryptedBlob.FORMAT_VERSION,
            "cipher_id": EncryptedBlob.CIPHER_ID_AES_256_GCM,
            "nonce_hex": envelope_nonce.hex(),
            "ciphertext_hex": envelope_sealed.hex(),
        },
    }
    out = Path(path)
    out.write_bytes(json.dumps(archive).encode("utf-8"))
    return ExportManifest(
        identities=len(container.identities),
        active_templates=len(container.active_templates),
        retired_templates=len(container.retired_templates),
        candidates=len(container.candidates),
        archive=str(out),
    )


def import_identities(
    repo: Any,
    runtime_manifest: ModelMigrationManifest,
    path: str | Path,
    passphrase: str,
    dest_key_provider: KeyProviderProtocol,
) -> ImportResult:
    """Verify-then-insert an archive under destination custody."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise StoreError(f"export archive unreadable: {path}") from exc
    try:
        archive = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise StoreCorruptionError("export archive is not valid JSON") from exc
    if not isinstance(archive, dict):
        raise StoreCorruptionError("export archive must be a JSON object")
    metadata = archive.get("metadata")
    envelope_payload = archive.get("envelope")
    if not isinstance(metadata, dict) or not isinstance(envelope_payload, dict):
        raise StoreCorruptionError("export archive missing metadata/envelope")
    try:
        salt = bytes.fromhex(str(metadata["kdf_salt"]))
    except (KeyError, ValueError) as exc:
        raise UnsupportedKdfError("export archive missing kdf_salt") from exc
    # KDF-parameter check BEFORE any crypto (unsupported → exit 4).
    _check_kdf_metadata(
        {
            "kdf_algorithm": metadata.get("kdf_algorithm"),
            "kdf_version": metadata.get("kdf_version"),
            "kdf_memory_cost_kib": metadata.get("kdf_memory_cost_kib"),
            "kdf_time_cost": metadata.get("kdf_time_cost"),
            "kdf_parallelism": metadata.get("kdf_parallelism"),
            "export_format_version": metadata.get("export_format_version"),
        }
    )
    k_wrap, k_envelope = _derive_subkeys(passphrase, salt)
    try:
        blob = EncryptedBlob(
            format_version=int(envelope_payload["format_version"]),
            cipher_id=int(envelope_payload["cipher_id"]),
            nonce=bytes.fromhex(str(envelope_payload["nonce_hex"])),
            ciphertext=bytes.fromhex(str(envelope_payload["ciphertext_hex"])),
        )
    except (KeyError, ValueError) as exc:
        raise StoreCorruptionError("export envelope header corrupt") from exc
    try:
        inner = AESGCM(k_envelope).decrypt(
            blob.nonce, blob.ciphertext, _envelope_aad(str(metadata["kdf_salt"]))
        )
    except Exception as exc:
        raise StoreCorruptionError(
            "export envelope authentication failed: tampered archive"
        ) from exc
    try:
        wrapped = json.loads(inner.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise StoreCorruptionError("export inner package corrupt") from exc
    container_dict = wrapped.get("container")
    wrapped_deks = wrapped.get("wrapped_deks", {})
    if not isinstance(container_dict, dict):
        raise StoreCorruptionError("export container missing")
    # 9-field predicate BEFORE any DB write (exit 3, zero writes).
    archived_manifest = ModelMigrationManifest(
        **container_dict["model_manifest"]
    )
    compatibility = _check_compatibility(archived_manifest, runtime_manifest)
    # DEK re-homing: unwrap under K_wrap, then store under destination
    # custody through the protocol method (which seals under the
    # destination master key for file providers). The destination
    # identity owns the re-homed key so identity-scoped destruction
    # (delete → destroy_identity_keys) keeps working after import.
    key_owner = _key_owners(container_dict)
    new_key_ids: dict[str, str] = {}
    for old_key_id, wrapped_dek in wrapped_deks.items():
        try:
            sealed = bytes.fromhex(str(wrapped_dek["sealed_hex"]))
            nonce = bytes.fromhex(str(wrapped_dek["nonce_hex"]))
            raw_dek = AESGCM(k_wrap).decrypt(
                nonce, sealed, str(old_key_id).encode("utf-8")
            )
        except Exception as exc:
            raise StoreCorruptionError(
                f"DEK unwrap failed for {old_key_id}"
            ) from exc
        rewrap_nonce = os.urandom(12)
        rewrap_sealed = AESGCM(k_wrap).encrypt(
            rewrap_nonce, raw_dek, None
        )
        new_key_ids[str(old_key_id)] = dest_key_provider.unwrap_and_store_key(
            WrappedKey(
                wrapped_dek=rewrap_sealed[:-16],
                nonce=rewrap_nonce,
                tag=rewrap_sealed[-16:],
                key_id=str(old_key_id),
            ),
            k_wrap,
            owner_identity_id=key_owner.get(str(old_key_id)),
        )
    _insert_container(repo, container_dict, new_key_ids)
    # Policy restoration: the destination store has no policy table (the
    # runtime default is provisional_v1), so import never silently mutates
    # runtime behavior. The archived policy is handed back explicitly; the
    # caller adopts it by constructing its manager with it (fail-closed:
    # no implicit overwrite, no silent keep).
    restored_policy = GovernancePolicy(**container_dict["policy"])
    return ImportResult(
        identities=len(container_dict["identities"]),
        active_templates=len(container_dict["active_templates"]),
        retired_templates=len(container_dict["retired_templates"]),
        candidates=len(container_dict["candidates"]),
        compatibility=compatibility,
        policy=restored_policy,
    )


def _blob_wire(payload: dict[str, Any] | None) -> tuple[bytes, bytes] | None:
    if payload is None:
        return None
    header = (
        bytes([int(payload["format_version"]), int(payload["cipher_id"]), 0, 0])
        + bytes.fromhex(str(payload["nonce_hex"]))
    )
    return (
        header + bytes.fromhex(str(payload["ciphertext_hex"])),
        bytes.fromhex(str(payload["nonce_hex"])),
    )


def _insert_container(
    repo: Any,
    container: dict[str, Any],
    new_key_ids: dict[str, str],
) -> None:
    """Transactional insert of re-homed rows; ciphertexts preserved as-is."""
    con: sqlite3.Connection | None = repo.connection
    if con is None:
        raise StoreError("repository not initialized")
    # Duplicate import is an explicit refusal (exit 4), never a bare
    # IntegrityError traceback: check before opening the transaction.
    for identity in container["identities"]:
        exists = con.execute(
            "SELECT 1 FROM identities WHERE id = ?",
            (identity["identity_id"],),
        ).fetchone()
        if exists is not None:
            raise StoreError(
                "identity already exists in destination: "
                f"{identity['identity_id']}"
            )
    con.execute("BEGIN IMMEDIATE")
    try:
        for identity in container["identities"]:
            con.execute(
                "INSERT INTO identities (id, display_name, status,"
                " current_revision, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))",
                (
                    identity["identity_id"],
                    identity["display_name"],
                    identity["status"],
                    identity["current_revision"],
                ),
            )
        for template in container["active_templates"]:
            _insert_template_row(con, template, "active", new_key_ids)
        for template in container["retired_templates"]:
            _insert_template_row(
                con, template, "retired", new_key_ids, retired_at=_utcnow()
            )
        for candidate in container["candidates"]:
            _insert_candidate_row(con, candidate, new_key_ids)
        for detail in container.get("revision_details", []):
            con.execute(
                "INSERT INTO template_revisions (identity_id, revision,"
                " active_template_ids, retired_template_ids, policy_version,"
                " created_at, actor) VALUES (?, ?, ?, ?, 1,"
                " datetime('now'), 'operator')",
                (
                    detail["identity_id"],
                    detail["revision"],
                    detail["active_template_ids"],
                    detail["retired_template_ids"],
                ),
            )
        for event in container.get("match_events", []):
            con.execute(
                "INSERT INTO match_events (id, timestamp, sequence_number,"
                " status, decision_score, runner_up_score,"
                " matched_identity_id, candidate_created, actor)"
                " VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                (
                    event["id"],
                    event["timestamp"],
                    event["sequence_number"],
                    event["status"],
                    event["decision_score"],
                    event["runner_up_score"],
                    event["candidate_created"],
                    event["actor"],
                ),
            )
        con.commit()
    except Exception:
        con.rollback()
        raise


def _insert_template_row(
    con: sqlite3.Connection,
    template: dict[str, Any],
    status: str,
    new_key_ids: dict[str, str],
    retired_at: str | None = None,
) -> None:
    emb = _blob_wire(template["encrypted_embedding"])
    exe = _blob_wire(template["encrypted_exemplar"])
    assert emb is not None and exe is not None
    crop = template.get("exemplar_crop_box")
    landmarks = template.get("exemplar_landmarks")
    con.execute(
        "INSERT INTO face_templates (id, identity_id, generation_id,"
        " model_version, embedding_dim, revision_number, revision_supersedes,"
        " status, key_id, embedding_blob, embedding_nonce, exemplar_blob,"
        " exemplar_nonce, exemplar_crop_box, exemplar_landmarks,"
        " exemplar_margin, quality_score, utility_score,"
        " additional_corroboration_count, created_at, retired_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
        " datetime('now'), ?)",
        (
            template["template_id"],
            template["identity_id"],
            template["generation_id"],
            template["model_version"],
            template["embedding_dim"],
            template["revision"]["revision"],
            template["revision"]["supersedes"],
            status,
            new_key_ids[str(template["key_id"])],
            emb[0],
            emb[1],
            exe[0],
            exe[1],
            json.dumps(crop) if crop is not None else None,
            json.dumps(landmarks) if landmarks is not None else None,
            template.get("exemplar_margin", 0.0),
            template.get("quality_score", 0.0),
            template.get("utility_score", 0.0),
            0,
            retired_at,
        ),
    )


def _insert_candidate_row(
    con: sqlite3.Connection,
    candidate: dict[str, Any],
    new_key_ids: dict[str, str],
) -> None:
    emb = _blob_wire(candidate["encrypted_embedding"])
    exe = _blob_wire(candidate.get("encrypted_exemplar"))
    assert emb is not None
    crop = candidate.get("exemplar_crop_box")
    landmarks = candidate.get("exemplar_landmarks")
    con.execute(
        "INSERT INTO candidate_templates (id, identity_id, generation_id,"
        " status, key_id, embedding_blob, embedding_nonce, exemplar_blob,"
        " exemplar_nonce, exemplar_crop_box, exemplar_landmarks,"
        " quality_score, additional_corroboration_count, evidence_log,"
        " expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
        " ?, ?, ?, ?, ?)",
        (
            candidate["template_id"],
            candidate["identity_id"],
            candidate["generation_id"],
            candidate["status"],
            new_key_ids[str(candidate["key_id"])],
            emb[0],
            emb[1],
            exe[0] if exe else None,
            exe[1] if exe else None,
            json.dumps(crop) if crop is not None else None,
            json.dumps(landmarks) if landmarks is not None else None,
            candidate.get("quality_score", 0.0),
            candidate.get("additional_corroboration_count", 0),
            json.dumps(
                [
                    {
                        "event_type": e["event_type"],
                        "timestamp": e["timestamp"],
                        "sequence_number": e["sequence_number"],
                        "score": e.get("score"),
                    }
                    for e in candidate.get("evidence_log", [])
                ]
            ),
            candidate.get("expires_at", ""),
            candidate.get("created_at", ""),
        ),
    )
