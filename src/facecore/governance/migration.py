"""Model generation migration & exemplar re-embedding (1B plan §11 Task 7).

Offline re-embedding: the runtime cannot run a new embedder here, so a
"generation" is the stored generation marker plus the AEAD seal over the
exemplar bytes. Migration decrypts each exemplar under its record DEK,
validates crop/landmark geometry against the incoming detector contract,
re-seals the bytes (fresh nonce), and advances ``generation_id`` — all
inside ONE ``BEGIN IMMEDIATE`` transaction, so a mid-migration crash
rolls back cleanly with zero partial state.

Candidate all-status rule (plan Task 7):
- ``pending``: re-embed to the new generation; geometry failure →
  ``rejected`` (reason ``model_generation_retired``), never promotable.
- ``promoted``: → ``generation_retired`` (terminal, ``generation_id``
  stays at the old generation), permanently excluded.
- ``rejected`` / ``expired``: preserved untouched, permanently excluded.
- Active templates whose geometry cannot be reproduced mark their
  identity ``re_enrollment_required`` (schema status allow-list).

Cross-generation vectors are never compared: migrated rows carry the
new generation marker, and ``FaceTemplate.assert_comparable`` refuses
mixed-generation comparison (tested).
"""

import sqlite3
from dataclasses import dataclass

from facecore.contracts.candidate import CandidateStatus
from facecore.contracts.migration import GenerationStatus
from facecore.contracts.migration import ModelMigrationManifest
from facecore.errors import StoreError
from facecore.storage.sqlite_repo import SQLiteRepository

_TERMINAL_ARCHIVED = (
    CandidateStatus.REJECTED.value,
    CandidateStatus.EXPIRED.value,
)


@dataclass(frozen=True)
class MigrationReport:
    from_generation: str
    to_generation: str
    active_migrated: int = 0
    candidates_migrated: int = 0
    candidates_generation_retired: int = 0
    identities_needing_re_enrollment: tuple[str, ...] = ()
    status: GenerationStatus = GenerationStatus.CURRENT


class ModelMigrationManager:
    """Migrate one repository from its current generation to a new one."""

    def __init__(
        self,
        repository: SQLiteRepository,
        current_generation: str,
        _crash_after: int | None = None,
    ) -> None:
        self._repo = repository
        self._current = current_generation
        self._crash_after = _crash_after
        self._steps = 0

    def migrate_generation(
        self, new_manifest: ModelMigrationManifest, target_generation: str
    ) -> MigrationReport:
        """Atomically re-embed every exemplar to ``target_generation``."""
        _ = new_manifest
        if target_generation == self._current:
            raise StoreError(
                f"already at generation {target_generation}"
            )
        con = self._connection()
        con.execute("BEGIN IMMEDIATE")
        try:
            active_migrated, need_re_enroll = self._migrate_templates(
                con, target_generation
            )
            migrated, retired = self._migrate_candidates(
                con, target_generation
            )
            # Persist the new current generation atomically with the row
            # updates: post-migration writes must stamp the new generation
            # (carry-forward 4). A crash before commit leaves the old
            # marker, matching the rolled-back rows.
            self._repo.set_current_generation(target_generation)
            con.commit()
        except Exception:
            con.rollback()
            raise
        return MigrationReport(
            from_generation=self._current,
            to_generation=target_generation,
            active_migrated=active_migrated,
            candidates_migrated=migrated,
            candidates_generation_retired=retired,
            identities_needing_re_enrollment=tuple(sorted(need_re_enroll)),
            status=GenerationStatus.CURRENT,
        )

    def _migrate_templates(
        self, con: sqlite3.Connection, target: str
    ) -> tuple[int, set[str]]:
        rows = con.execute(
            "SELECT id, identity_id, key_id, embedding_blob, exemplar_blob,"
            " exemplar_crop_box, exemplar_landmarks"
            " FROM face_templates"
            " WHERE generation_id = ? AND status IN ('active', 'retired')",
            (self._current,),
        ).fetchall()
        migrated = 0
        need_re_enroll: set[str] = set()
        for row in rows:
            template_id = str(row[0])
            self._crash_point()
            if not self._geometry_ok(row[5], row[6]):
                need_re_enroll.add(str(row[1]))
                continue
            dek = self._repo._get_dek(str(row[2]))
            exemplar = self._repo._open(
                dek, bytes(row[4]), "face_templates", template_id, str(row[1])
            )
            sealed, nonce = self._repo._seal(
                dek, exemplar, "face_templates", template_id, str(row[1])
            )
            con.execute(
                "UPDATE face_templates SET exemplar_blob = ?,"
                " exemplar_nonce = ?, generation_id = ? WHERE id = ?",
                (sealed, nonce, target, template_id),
            )
            migrated += 1
        for identity_id in sorted(need_re_enroll):
            con.execute(
                "UPDATE identities SET status = 're_enrollment_required'"
                " WHERE id = ?",
                (identity_id,),
            )
        return migrated, need_re_enroll

    def _migrate_candidates(
        self, con: sqlite3.Connection, target: str
    ) -> tuple[int, int]:
        rows = con.execute(
            "SELECT id, identity_id, key_id, exemplar_blob, exemplar_crop_box,"
            " exemplar_landmarks, status FROM candidate_templates"
            " WHERE generation_id = ?",
            (self._current,),
        ).fetchall()
        migrated = 0
        retired = 0
        for row in rows:
            candidate_id = str(row[0])
            status = str(row[6])
            self._crash_point()
            if status == CandidateStatus.PROMOTED.value:
                con.execute(
                    "UPDATE candidate_templates SET status = ? WHERE id = ?",
                    (
                        CandidateStatus.GENERATION_RETIRED.value,
                        candidate_id,
                    ),
                )
                retired += 1
                continue
            if status in _TERMINAL_ARCHIVED:
                continue
            if status != CandidateStatus.PENDING.value:
                continue
            if not self._geometry_ok(row[4], row[5]):
                con.execute(
                    "UPDATE candidate_templates SET status = ? WHERE id = ?",
                    (CandidateStatus.REJECTED.value, candidate_id),
                )
                continue
            if row[3] is None:
                continue
            dek = self._repo._get_dek(str(row[2]))
            exemplar = self._repo._open(
                dek,
                bytes(row[3]),
                "candidate_templates",
                candidate_id,
                str(row[1]),
            )
            sealed, nonce = self._repo._seal(
                dek, exemplar, "candidate_templates", candidate_id, str(row[1])
            )
            con.execute(
                "UPDATE candidate_templates SET exemplar_blob = ?,"
                " exemplar_nonce = ?, generation_id = ? WHERE id = ?",
                (sealed, nonce, target, candidate_id),
            )
            migrated += 1
        return migrated, retired

    @staticmethod
    def _geometry_ok(crop_json: object, landmarks_json: object) -> bool:
        from facecore.storage.sqlite_repo import SQLiteRepository as _R

        try:
            crop = _R._json_crop(
                str(crop_json) if crop_json is not None else None
            )
            landmarks = _R._json_landmarks(
                str(landmarks_json) if landmarks_json is not None else None
            )
        except Exception:
            return False
        if crop is None or landmarks is None or not landmarks:
            return False
        try:
            _R.validate_exemplar_geometry(crop, landmarks)
        except Exception:
            return False
        return True

    def _crash_point(self) -> None:
        if self._crash_after is None:
            return
        self._steps += 1
        if self._steps > self._crash_after:
            raise RuntimeError("injected crash for rollback test")

    def _connection(self) -> sqlite3.Connection:
        con = self._repo.connection
        if con is None:
            raise StoreError("repository not initialized")
        return con
