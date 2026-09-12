"""Encrypted SQLite repository (S1B §§4-5, plan §7: schema + tombstones).

Every write runs under ``BEGIN IMMEDIATE`` with WAL + ``foreign_keys = ON``.
Biometric payloads are AEAD ciphertexts under per-record DEKs held exclusively
by the KeyProvider; SQLite stores opaque ``key_id`` strings only.

Deletion follows the 3-phase Journaled Tombstone Protocol: (1) insert
tombstone + mark deleted, (2) destroy DEKs, (3) purge rows + anonymize match
events + ``wal_checkpoint(TRUNCATE)`` with fail-closed result validation.
`initialize` runs `reconcile_tombstones`, treating already-absent keys as
idempotent success (crash-after-destroy recovery).
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from facecore.contracts.crypto import (
    EncryptedBlob,
    KeyNotFoundError,
    KeyProviderProtocol,
    StoreCorruptionError,
)
from facecore.contracts.template import FaceTemplate, TemplateRevision
from facecore.errors import StoreError
from facecore.storage.cipher import AeadCipher, build_canonical_aad

_SCHEMA_VERSION = 1
_TOMBSTONE_PENDING = "pending_key_destruction"
_TOMBSTONE_DESTROYED = "key_destroyed"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_truncate(con: sqlite3.Connection) -> None:
    """Fail-closed WAL truncation: busy/log residue raises StoreError (exit 4)."""
    res = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if res is None or res[0] != 0 or res[1] != 0:
        busy = res[0] if res else None
        log = res[1] if res else None
        raise StoreCorruptionError(
            "WAL checkpoint failed to truncate: "
            f"busy={busy}, log={log}; store is fail-closed"
        )


def trim_match_events(con: sqlite3.Connection, ceiling: int) -> int:
    """Trim match_events to ceiling rows (oldest first). Returns rows removed."""
    row = con.execute("SELECT COUNT(*) FROM match_events").fetchone()
    total = int(row[0]) if row else 0
    if total <= ceiling:
        return 0
    con.execute("BEGIN IMMEDIATE")
    try:
        con.execute(
            "DELETE FROM match_events WHERE id IN ("
            "SELECT id FROM match_events "
            "ORDER BY timestamp ASC, sequence_number ASC LIMIT ?)",
            (total - ceiling,),
        )
        con.commit()
    except Exception:
        con.rollback()
        raise
    return total - ceiling


def trim_revisions(
    con: sqlite3.Connection, identity_id: str, ceiling: int
) -> int:
    """Trim per-identity revision rows to ceiling. Returns rows removed."""
    count_row = con.execute(
        "SELECT COUNT(*) FROM template_revisions WHERE identity_id = ?",
        (identity_id,),
    ).fetchone()
    total = int(count_row[0]) if count_row else 0
    if total <= ceiling:
        return 0
    con.execute("BEGIN IMMEDIATE")
    try:
        con.execute(
            "DELETE FROM template_revisions WHERE id IN ("
            "SELECT id FROM template_revisions WHERE identity_id = ? "
            "ORDER BY revision ASC LIMIT ?)",
            (identity_id, total - ceiling),
        )
        con.commit()
    except Exception:
        con.rollback()
        raise
    return total - ceiling


def prune_backups(directory: Path, pattern: str, ceiling: int) -> list[str]:
    """Keep the newest `ceiling` backups; delete older. Returns pruned paths."""
    candidates = sorted(directory.glob(pattern), key=lambda p: p.name)
    if len(candidates) <= ceiling:
        return []
    pruned: list[str] = []
    for path in candidates[: len(candidates) - ceiling]:
        path.unlink()
        pruned.append(str(path))
    return pruned


class SQLiteRepository:
    def __init__(self, db_path: str, key_provider: KeyProviderProtocol) -> None:
        self._db_path = db_path
        self.key_provider = key_provider
        self.connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path)
        con.execute("PRAGMA journal_mode = WAL;")
        con.execute("PRAGMA synchronous = NORMAL;")
        con.execute("PRAGMA foreign_keys = ON;")
        con.execute("PRAGMA busy_timeout = 5000;")
        return con

    def _require(self) -> sqlite3.Connection:
        if self.connection is None:
            raise StoreError("repository not initialized")
        return self.connection

    def initialize(self) -> None:
        if self.connection is None:
            self.connection = self._connect()
        con = self._require()
        schema = (
            resources.files("facecore.storage.migrations")
            .joinpath("v1.sql")
            .read_text()
        )
        con.executescript(schema)
        self.reconcile_tombstones()

    def _seal(
        self, dek: bytes, plaintext: bytes, table: str, record: str, identity: str
    ) -> tuple[bytes, bytes]:
        blob = AeadCipher(dek).encrypt(
            plaintext, build_canonical_aad(table, record, identity)
        )
        header = bytes(
            [blob.format_version, blob.cipher_id, 0x00, 0x00]
        ) + blob.nonce
        return header + blob.ciphertext, blob.nonce

    @staticmethod
    def _parse_blob(sealed: bytes, table: str, record: str) -> EncryptedBlob:
        if len(sealed) < 4 + EncryptedBlob.NONCE_BYTES + 16:
            raise StoreCorruptionError(
                f"ciphertext too short for {table}:{record}"
            )
        version, cipher_id = sealed[0], sealed[1]
        try:
            return EncryptedBlob(
                format_version=version,
                cipher_id=cipher_id,
                nonce=sealed[4 : 4 + EncryptedBlob.NONCE_BYTES],
                ciphertext=sealed[4 + EncryptedBlob.NONCE_BYTES :],
            )
        except ValueError as exc:
            raise StoreCorruptionError(f"blob header rejected: {exc}") from exc

    def _open(
        self, dek: bytes, sealed: bytes, table: str, record: str, identity: str
    ) -> bytes:
        blob = self._parse_blob(sealed, table, record)
        return AeadCipher(dek).decrypt(
            blob, build_canonical_aad(table, record, identity)
        )

    def _reserve_identity_create(self, identity_id: str, template_id: str) -> None:
        """Reserve uniqueness before custody mutation; rollback is deliberate."""
        con = self._require()
        con.execute("BEGIN IMMEDIATE")
        try:
            identity = con.execute(
                "SELECT 1 FROM identities WHERE id = ?", (identity_id,)
            ).fetchone()
            if identity is not None:
                raise sqlite3.IntegrityError(f"identity already exists: {identity_id}")
            template = con.execute(
                "SELECT 1 FROM face_templates WHERE id = ?", (template_id,)
            ).fetchone()
            if template is not None:
                raise sqlite3.IntegrityError(
                    f"template already exists: {template_id}"
                )
        except Exception:
            con.rollback()
            raise
        con.rollback()

    def _reserve_revision_create(self, identity_id: str, template_id: str) -> None:
        """Validate append targets before creating a new external DEK."""
        con = self._require()
        con.execute("BEGIN IMMEDIATE")
        try:
            identity = con.execute(
                "SELECT status FROM identities WHERE id = ?", (identity_id,)
            ).fetchone()
            if identity is None or identity[0] != "active":
                raise StoreError(f"unknown identity: {identity_id}")
            template = con.execute(
                "SELECT 1 FROM face_templates WHERE id = ?", (template_id,)
            ).fetchone()
            if template is not None:
                raise sqlite3.IntegrityError(
                    f"template already exists: {template_id}"
                )
        except Exception:
            con.rollback()
            raise
        con.rollback()

    def _reserve_candidate_create(self, identity_id: str, candidate_id: str) -> None:
        """Validate candidate uniqueness before creating a new external DEK."""
        con = self._require()
        con.execute("BEGIN IMMEDIATE")
        try:
            identity = con.execute(
                "SELECT status FROM identities WHERE id = ?", (identity_id,)
            ).fetchone()
            if identity is None or identity[0] != "active":
                raise StoreError(f"unknown identity: {identity_id}")
            candidate = con.execute(
                "SELECT 1 FROM candidate_templates WHERE id = ?", (candidate_id,)
            ).fetchone()
            if candidate is not None:
                raise sqlite3.IntegrityError(
                    f"candidate already exists: {candidate_id}"
                )
        except Exception:
            con.rollback()
            raise
        con.rollback()

    def _cleanup_created_key(self, key_id: str) -> None:
        """Compensate every failed pre-commit write; absent is idempotent."""
        try:
            self.key_provider.destroy_key(key_id)
        except KeyNotFoundError:
            return

    def _get_dek(self, key_id: str) -> bytes:
        try:
            return self.key_provider.get_key(key_id)
        except StoreError:
            raise
        except Exception as exc:
            raise StoreCorruptionError(f"key retrieval failed: {key_id}") from exc

    def enroll_identity(
        self,
        identity_id: str,
        display_name: str,
        template: FaceTemplate,
        embedding: bytes,
        exemplar: bytes,
        record_id: str | None = None,
    ) -> str:
        con = self._require()
        tid = record_id or template.template_id
        self._reserve_identity_create(identity_id, tid)
        key_id: str | None = None
        try:
            key_id = self.key_provider.create_key(identity_id)
            dek = self._get_dek(key_id)
            enc_emb, emb_nonce = self._seal(
                dek, embedding, "face_templates", tid, identity_id
            )
            enc_exe, exe_nonce = self._seal(
                dek, exemplar, "face_templates", tid, identity_id
            )
            now = _utcnow()
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "INSERT INTO identities "
                "(id, display_name, status, current_revision,"
                " created_at, updated_at)"
                " VALUES (?, ?, 'active', 1, ?, ?)",
                (identity_id, display_name, now, now),
            )
            con.execute(
                "INSERT INTO face_templates "
                "(id, identity_id, generation_id, model_version, embedding_dim,"
                " revision_number, revision_supersedes, status, key_id,"
                " embedding_blob, embedding_nonce, exemplar_blob, exemplar_nonce,"
                " exemplar_crop_box, exemplar_landmarks, exemplar_margin,"
                " quality_score, utility_score, additional_corroboration_count,"
                " created_at, retired_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?,"
                " ?, ?, 0, ?, NULL)",
                (
                    tid,
                    identity_id,
                    template.generation_id,
                    template.model_version,
                    template.embedding_dim,
                    template.revision.revision,
                    template.revision.supersedes,
                    key_id,
                    enc_emb,
                    emb_nonce,
                    enc_exe,
                    exe_nonce,
                    json.dumps(template.exemplar_crop_box)
                    if template.exemplar_crop_box is not None
                    else None,
                    json.dumps(template.exemplar_landmarks)
                    if template.exemplar_landmarks is not None
                    else None,
                    template.exemplar_margin,
                    template.quality_score,
                    template.utility_score,
                    now,
                ),
            )
            con.execute(
                "INSERT INTO template_revisions "
                "(identity_id, revision, active_template_ids, retired_template_ids,"
                " policy_version, created_at, actor)"
                " VALUES (?, ?, ?, '[]', 1, ?, 'user')",
                (
                    identity_id,
                    template.revision.revision,
                    json.dumps([tid]),
                    now,
                ),
            )
            con.execute(
                "UPDATE identities SET current_revision = ? WHERE id = ?",
                (template.revision.revision, identity_id),
            )
            con.commit()
        except Exception:
            con.rollback()
            if key_id is not None:
                self._cleanup_created_key(key_id)
            raise
        return tid

    def append_revision(
        self,
        identity_id: str,
        template: FaceTemplate,
        embedding: bytes,
        exemplar: bytes,
    ) -> str:
        con = self._require()
        tid = template.template_id
        self._reserve_revision_create(identity_id, tid)
        key_id: str | None = None
        try:
            key_id = self.key_provider.create_key(identity_id)
            dek = self._get_dek(key_id)
            enc_emb, emb_nonce = self._seal(
                dek, embedding, "face_templates", tid, identity_id
            )
            enc_exe, exe_nonce = self._seal(
                dek, exemplar, "face_templates", tid, identity_id
            )
            now = _utcnow()
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status, current_revision FROM identities WHERE id = ?",
                (identity_id,),
            ).fetchone()
            if row is None or row[0] != "active":
                raise StoreError(f"identity deleted or tombstoned: {identity_id}")
            revision = max(int(row[1]) + 1, template.revision.revision)
            con.execute(
                "INSERT INTO face_templates "
                "(id, identity_id, generation_id, model_version, embedding_dim,"
                " revision_number, revision_supersedes, status, key_id,"
                " embedding_blob, embedding_nonce, exemplar_blob, exemplar_nonce,"
                " exemplar_crop_box, exemplar_landmarks, exemplar_margin,"
                " quality_score, utility_score, additional_corroboration_count,"
                " created_at, retired_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?,"
                " ?, ?, 0, ?, NULL)",
                (
                    tid,
                    identity_id,
                    template.generation_id,
                    template.model_version,
                    template.embedding_dim,
                    revision,
                    template.revision.supersedes,
                    key_id,
                    enc_emb,
                    emb_nonce,
                    enc_exe,
                    exe_nonce,
                    json.dumps(template.exemplar_crop_box)
                    if template.exemplar_crop_box is not None
                    else None,
                    json.dumps(template.exemplar_landmarks)
                    if template.exemplar_landmarks is not None
                    else None,
                    template.exemplar_margin,
                    template.quality_score,
                    template.utility_score,
                    now,
                ),
            )
            con.execute(
                "UPDATE identities SET current_revision = ?, updated_at = ?"
                " WHERE id = ?",
                (revision, now, identity_id),
            )
            active_ids = [
                r[0]
                for r in con.execute(
                    "SELECT id FROM face_templates"
                    " WHERE identity_id = ? AND status = 'active'",
                    (identity_id,),
                ).fetchall()
            ]
            con.execute(
                "INSERT INTO template_revisions "
                "(identity_id, revision, active_template_ids, retired_template_ids,"
                " policy_version, created_at, actor)"
                " VALUES (?, ?, ?, '[]', 1, ?, 'user')",
                (identity_id, revision, json.dumps(active_ids), now),
            )
            con.commit()
        except Exception:
            con.rollback()
            if key_id is not None:
                self._cleanup_created_key(key_id)
            raise
        return tid

    def add_candidate_record(
        self,
        candidate_id: str,
        identity_id: str,
        embedding: bytes,
        exemplar: bytes,
        expires_at: str,
    ) -> str:
        con = self._require()
        self._reserve_candidate_create(identity_id, candidate_id)
        key_id: str | None = None
        try:
            key_id = self.key_provider.create_key(identity_id)
            dek = self._get_dek(key_id)
            enc_emb, emb_nonce = self._seal(
                dek, embedding, "candidate_templates", candidate_id, identity_id
            )
            enc_exe, exe_nonce = self._seal(
                dek, exemplar, "candidate_templates", candidate_id, identity_id
            )
            now = _utcnow()
            con.execute("BEGIN IMMEDIATE")
            identity = con.execute(
                "SELECT status FROM identities WHERE id = ?", (identity_id,)
            ).fetchone()
            if identity is None or identity[0] != "active":
                raise StoreError(f"identity deleted or tombstoned: {identity_id}")
            con.execute(
                "INSERT INTO candidate_templates "
                "(id, identity_id, generation_id, status, key_id,"
                " embedding_blob, embedding_nonce, exemplar_blob, exemplar_nonce,"
                " exemplar_crop_box, exemplar_landmarks, quality_score,"
                " additional_corroboration_count, evidence_log,"
                " expires_at, created_at)"
                " VALUES (?, ?, 'G1', 'pending', ?, ?, ?, ?, ?,"
                " NULL, NULL, 0.9, 0, '[]', ?, ?)",
                (
                    candidate_id,
                    identity_id,
                    key_id,
                    enc_emb,
                    emb_nonce,
                    enc_exe,
                    exe_nonce,
                    expires_at,
                    now,
                ),
            )
            con.commit()
        except Exception:
            con.rollback()
            if key_id is not None:
                self._cleanup_created_key(key_id)
            raise
        return candidate_id

    @staticmethod
    def _json_crop(value: str | None) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        raw: object = json.loads(value)
        if not isinstance(raw, list) or len(raw) != 4:
            raise StoreCorruptionError("invalid exemplar crop geometry")
        return (
            float(str(raw[0])),
            float(str(raw[1])),
            float(str(raw[2])),
            float(str(raw[3])),
        )

    @staticmethod
    def _json_landmarks(
        value: str | None,
    ) -> tuple[tuple[float, float], ...] | None:
        if value is None:
            return None
        raw: object = json.loads(value)
        if not isinstance(raw, list):
            raise StoreCorruptionError("invalid exemplar landmark geometry")
        result: list[tuple[float, float]] = []
        for point in raw:
            if not isinstance(point, list) or len(point) != 2:
                raise StoreCorruptionError("invalid exemplar landmark point")
            result.append((float(str(point[0])), float(str(point[1]))))
        return tuple(result)

    def _hydrate_template(self, row: tuple[object, ...]) -> FaceTemplate:
        (
            template_id,
            identity_id,
            generation_id,
            model_version,
            embedding_dim,
            revision_number,
            revision_supersedes,
            key_id,
            embedding_blob,
            exemplar_blob,
            exemplar_crop_box,
            exemplar_landmarks,
            exemplar_margin,
            quality_score,
            utility_score,
        ) = row
        if not isinstance(embedding_blob, (bytes, bytearray)):
            raise StoreCorruptionError("embedding payload is not bytes")
        if not isinstance(exemplar_blob, (bytes, bytearray)):
            raise StoreCorruptionError("exemplar payload is not bytes")
        embedding_sealed = bytes(embedding_blob)
        exemplar_sealed = bytes(exemplar_blob)
        return FaceTemplate(
            template_id=str(template_id),
            identity_id=str(identity_id),
            model_version=str(model_version),
            embedding_dim=int(str(embedding_dim)),
            generation_id=str(generation_id),
            revision=TemplateRevision(
                revision=int(str(revision_number)),
                template_id=str(template_id),
                supersedes=(
                    str(revision_supersedes)
                    if revision_supersedes is not None
                    else None
                ),
            ),
            key_id=str(key_id),
            encrypted_embedding=self._parse_blob(
                embedding_sealed, "face_templates", str(template_id)
            ),
            encrypted_exemplar=self._parse_blob(
                exemplar_sealed, "face_templates", str(template_id)
            ),
            exemplar_crop_box=self._json_crop(
                str(exemplar_crop_box) if exemplar_crop_box is not None else None
            ),
            exemplar_landmarks=self._json_landmarks(
                str(exemplar_landmarks) if exemplar_landmarks is not None else None
            ),
            exemplar_margin=float(str(exemplar_margin)),
            quality_score=float(str(quality_score)),
            utility_score=float(str(utility_score)),
        )

    def list_active_templates(self) -> list[FaceTemplate]:
        con = self._require()
        rows = con.execute(
            "SELECT ft.id, ft.identity_id, ft.generation_id, ft.model_version,"
            " ft.embedding_dim, ft.revision_number, ft.revision_supersedes,"
            " ft.key_id, ft.embedding_blob, ft.exemplar_blob,"
            " ft.exemplar_crop_box, ft.exemplar_landmarks, ft.exemplar_margin,"
            " ft.quality_score, ft.utility_score"
            " FROM face_templates ft JOIN identities i ON i.id = ft.identity_id"
            " WHERE ft.status = 'active' AND i.status = 'active'"
            " AND NOT EXISTS (SELECT 1 FROM deletion_tombstones dt"
            " WHERE dt.target_type = 'identity' AND dt.target_id = i.id)"
            " ORDER BY ft.id"
        ).fetchall()
        return [self._hydrate_template(tuple(row)) for row in rows]

    def read_embedding(self, template_id: str) -> bytes:
        con = self._require()
        row = con.execute(
            "SELECT ft.identity_id, ft.key_id, ft.embedding_blob"
            " FROM face_templates ft JOIN identities i ON i.id = ft.identity_id"
            " WHERE ft.id = ? AND ft.status = 'active' AND i.status = 'active'"
            " AND NOT EXISTS (SELECT 1 FROM deletion_tombstones dt"
            " WHERE dt.target_type = 'identity' AND dt.target_id = i.id)",
            (template_id,),
        ).fetchone()
        if row is None:
            exists = con.execute(
                "SELECT i.status FROM face_templates ft"
                " JOIN identities i ON i.id = ft.identity_id WHERE ft.id = ?",
                (template_id,),
            ).fetchone()
            if exists is not None and exists[0] != "active":
                raise StoreError("identity deleted or tombstoned")
            raise StoreError(f"unknown template: {template_id}")
        identity_id, key_id, sealed = row
        dek = self._get_dek(str(key_id))
        return self._open(
            dek, bytes(sealed), "face_templates", template_id, str(identity_id)
        )

    def record_match_event(
        self,
        event_id: str,
        timestamp: str,
        sequence_number: int,
        status: str,
        score: float,
        matched_identity_id: str | None,
    ) -> None:
        con = self._require()
        con.execute("BEGIN IMMEDIATE")
        try:
            con.execute(
                "INSERT INTO match_events "
                "(id, timestamp, sequence_number, status, decision_score,"
                " runner_up_score, matched_identity_id, candidate_created, actor)"
                " VALUES (?, ?, ?, ?, ?, NULL, ?, 0, NULL)",
                (
                    event_id,
                    timestamp,
                    sequence_number,
                    status,
                    score,
                    matched_identity_id,
                ),
            )
            con.commit()
        except Exception:
            con.rollback()
            raise

    def revision_count(self, identity_id: str) -> int:
        con = self._require()
        return int(
            con.execute(
                "SELECT COUNT(*) FROM template_revisions WHERE identity_id = ?",
                (identity_id,),
            ).fetchone()[0]
        )

    def get_identity_status(self, identity_id: str) -> str | None:
        con = self._require()
        row = con.execute(
            "SELECT status FROM identities WHERE id = ?", (identity_id,)
        ).fetchone()
        return str(row[0]) if row else None

    def tombstone_key_ids(self, identity_id: str) -> list[str]:
        con = self._require()
        rows = con.execute(
            "SELECT key_ids_json FROM deletion_tombstones WHERE target_id = ?",
            (identity_id,),
        ).fetchall()
        key_ids: list[str] = []
        for (payload,) in rows:
            key_ids.extend(json.loads(payload))
        return key_ids

    def begin_delete_identity(self, identity_id: str) -> None:
        """Atomically snapshot key IDs and mark deletion before external work."""
        con = self._require()
        con.execute("BEGIN IMMEDIATE")
        try:
            identity = con.execute(
                "SELECT status FROM identities WHERE id = ?", (identity_id,)
            ).fetchone()
            if identity is None:
                raise StoreError(f"unknown identity: {identity_id}")
            if identity[0] == "deleted":
                raise StoreError(f"identity already being deleted: {identity_id}")
            rows = con.execute(
                "SELECT key_id FROM face_templates WHERE identity_id = ?",
                (identity_id,),
            ).fetchall()
            rows += con.execute(
                "SELECT key_id FROM candidate_templates WHERE identity_id = ?",
                (identity_id,),
            ).fetchall()
            key_ids = [str(row[0]) for row in rows]
            con.execute(
                "UPDATE identities SET status = 'deleted' WHERE id = ?",
                (identity_id,),
            )
            con.execute(
                "INSERT INTO deletion_tombstones "
                "(target_type, target_id, key_ids_json, status, created_at)"
                " VALUES ('identity', ?, ?, ?, ?)",
                (identity_id, json.dumps(key_ids), _TOMBSTONE_PENDING, _utcnow()),
            )
            con.commit()
        except Exception:
            con.rollback()
            raise

    def mark_tombstone_key_destroyed(self, identity_id: str) -> None:
        con = self._require()
        con.execute("BEGIN IMMEDIATE")
        try:
            con.execute(
                "UPDATE deletion_tombstones SET status = ? WHERE target_id = ?",
                (_TOMBSTONE_DESTROYED, identity_id),
            )
            con.commit()
        except Exception:
            con.rollback()
            raise

    def _purge_identity_rows(self, con: sqlite3.Connection, target_id: str) -> None:
        con.execute(
            "UPDATE match_events SET matched_identity_id = NULL"
            " WHERE matched_identity_id = ?",
            (target_id,),
        )
        for row in con.execute(
            "SELECT id, embedding_blob, exemplar_blob FROM face_templates"
            " WHERE identity_id = ?",
            (target_id,),
        ).fetchall():
            con.execute(
                "UPDATE face_templates SET embedding_blob = ?, exemplar_blob = ?"
                " WHERE id = ?",
                (os.urandom(len(row[1])), os.urandom(len(row[2])), row[0]),
            )
        con.execute(
            "DELETE FROM face_templates WHERE identity_id = ?", (target_id,)
        )
        for row in con.execute(
            "SELECT id, embedding_blob, exemplar_blob FROM candidate_templates"
            " WHERE identity_id = ?",
            (target_id,),
        ).fetchall():
            exemplar = (
                os.urandom(len(row[2])) if row[2] is not None else None
            )
            con.execute(
                "UPDATE candidate_templates SET embedding_blob = ?,"
                " exemplar_blob = ? WHERE id = ?",
                (os.urandom(len(row[1])), exemplar, row[0]),
            )
        con.execute(
            "DELETE FROM candidate_templates WHERE identity_id = ?", (target_id,)
        )
        con.execute(
            "DELETE FROM template_revisions WHERE identity_id = ?", (target_id,)
        )
        con.execute("DELETE FROM identities WHERE id = ?", (target_id,))

    def _destroy_keys_best_effort(self, key_ids: list[str]) -> None:
        """Destroy all keys; only explicit already-absent is idempotent."""
        for key_id in key_ids:
            try:
                self.key_provider.destroy_key(key_id)
            except KeyNotFoundError:
                # Crash-after-destroy is the documented idempotent branch.
                continue

    def reconcile_tombstones(self) -> None:
        con = self._require()
        rows = con.execute(
            "SELECT id, target_type, target_id, key_ids_json, status"
            " FROM deletion_tombstones"
        ).fetchall()
        for t_id, target_type, target_id, payload, status in rows:
            key_ids = list(json.loads(payload))
            if status == _TOMBSTONE_PENDING:
                self._destroy_keys_best_effort(key_ids)
                con.execute("BEGIN IMMEDIATE")
                try:
                    con.execute(
                        "UPDATE deletion_tombstones SET status = ? WHERE id = ?",
                        (_TOMBSTONE_DESTROYED, t_id),
                    )
                    con.commit()
                except Exception:
                    con.rollback()
                    raise
                status = _TOMBSTONE_DESTROYED
            if status == _TOMBSTONE_DESTROYED:
                con.execute("BEGIN IMMEDIATE")
                try:
                    if target_type == "identity":
                        self._purge_identity_rows(con, target_id)
                    else:
                        con.execute(
                            "DELETE FROM face_templates WHERE id = ?", (target_id,)
                        )
                        con.execute(
                            "DELETE FROM candidate_templates WHERE id = ?",
                            (target_id,),
                        )
                    con.execute(
                        "DELETE FROM deletion_tombstones WHERE id = ?", (t_id,)
                    )
                    con.commit()
                except Exception:
                    con.rollback()
                    raise
                checkpoint_truncate(con)

    def delete_identity(self, identity_id: str) -> None:
        self.begin_delete_identity(identity_id)
        key_ids = self.tombstone_key_ids(identity_id)
        self._destroy_keys_best_effort(key_ids)
        self.mark_tombstone_key_destroyed(identity_id)
        con = self._require()
        con.execute("BEGIN IMMEDIATE")
        try:
            self._purge_identity_rows(con, identity_id)
            con.execute(
                "DELETE FROM deletion_tombstones WHERE target_id = ?",
                (identity_id,),
            )
            con.commit()
        except Exception:
            con.rollback()
            raise
        checkpoint_truncate(con)
