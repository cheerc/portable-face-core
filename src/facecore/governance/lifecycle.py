"""Identity lifecycle operations (1B plan §11 Task 5).

Thin orchestration over the encrypted repository (Task 2). Every mutation
is atomic at the repository boundary; rendering (stable JSON, exit codes)
belongs to the CLI layer.

- ``add_identity`` is create-only: duplicate IDs fail without mutation.
- ``re_enroll`` atomically retires actives and installs the new photo as
  the sole active template. Retired rows keep their DEKs (rollback must
  stay decryptable); only ``delete_identity`` destroys keys.
- ``rollback`` restores revision *n*, retiring newer templates; refuses
  when the depth exceeds ``rollback_max_depth``.
- ``delete_identity`` executes the Journaled Tombstone Protocol via the
  repository: tombstone + DEK destruction + row overwrite + event
  anonymization + ``wal_checkpoint(TRUNCATE)``.
- ``reject_candidate`` / ``list_candidates`` are candidate-side recovery.
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass

from facecore.contracts.candidate import CandidateStatus
from facecore.contracts.policy import GovernancePolicy
from facecore.contracts.template import FaceTemplate, TemplateRevision
from facecore.errors import StoreError
from facecore.storage.sqlite_repo import SQLiteRepository

_MODEL_VERSION = "sface-2021dec-fp32"
_EMBEDDING_DIM = 128
_CROP_BOX = (0.0, 0.0, 112.0, 112.0)
_LANDMARKS = (
    (30.0, 30.0),
    (82.0, 30.0),
    (56.0, 56.0),
    (30.0, 82.0),
    (82.0, 82.0),
)


@dataclass(frozen=True)
class LifecycleResult:
    identity_id: str
    action: str
    revision: int | None = None
    template_id: str | None = None


def _new_template(
    identity_id: str, revision: int, supersedes: str | None
) -> FaceTemplate:
    template_id = f"t-{uuid.uuid4().hex[:8]}"
    return FaceTemplate(
        template_id=template_id,
        identity_id=identity_id,
        model_version=_MODEL_VERSION,
        embedding_dim=_EMBEDDING_DIM,
        generation_id="G1",
        revision=TemplateRevision(
            revision=revision, template_id=template_id, supersedes=supersedes
        ),
        key_id="lifecycle-managed",
        exemplar_crop_box=_CROP_BOX,
        exemplar_landmarks=_LANDMARKS,
        quality_score=0.9,
    )


class LifecycleManager:
    """Orchestrate identity lifecycle against an initialized repository."""

    def __init__(
        self,
        repository: SQLiteRepository,
        policy: GovernancePolicy | None = None,
    ) -> None:
        self._repo = repository
        self._policy = policy or GovernancePolicy.provisional_v1()

    @property
    def repository(self) -> SQLiteRepository:
        """Bound repository (export/import operate on the same store)."""
        return self._repo

    def add_identity(
        self,
        identity_id: str,
        display_name: str,
        embedding: bytes,
        exemplar: bytes,
    ) -> LifecycleResult:
        """Create-only enrollment storing the initial exemplar."""
        template = _new_template(identity_id, 1, None)
        try:
            self._repo.enroll_identity(
                identity_id, display_name, template, embedding, exemplar
            )
        except Exception as exc:
            raise StoreError(f"identity add failed: {identity_id}") from exc
        return LifecycleResult(
            identity_id=identity_id,
            action="added",
            revision=1,
            template_id=template.template_id,
        )

    def re_enroll(
        self, identity_id: str, embedding: bytes, exemplar: bytes
    ) -> LifecycleResult:
        """Atomically retire actives; install the new photo as sole active."""
        con = self._connection()
        if self._repo.get_identity_status(identity_id) != "active":
            raise StoreError(f"unknown identity: {identity_id}")
        active_ids = [
            row[0]
            for row in con.execute(
                "SELECT id FROM face_templates"
                " WHERE identity_id = ? AND status = 'active'",
                (identity_id,),
            ).fetchall()
        ]
        if not active_ids:
            raise StoreError(f"no active template for: {identity_id}")
        template = _new_template(identity_id, 1, active_ids[0])
        new_tid = self._repo.append_revision(
            identity_id, template, embedding, exemplar
        )
        con.execute("BEGIN IMMEDIATE")
        try:
            for old_id in active_ids:
                con.execute(
                    "UPDATE face_templates SET status = 'retired',"
                    " retired_at = datetime('now') WHERE id = ?",
                    (old_id,),
                )
            revision = int(
                con.execute(
                    "SELECT current_revision FROM identities WHERE id = ?",
                    (identity_id,),
                ).fetchone()[0]
            )
            # The revision row written by append_revision snapshots the
            # pre-retirement active set (old + new). Rewrite BOTH sides so
            # the generation snapshot is exact: only the new template is
            # active, the retired ones are retired. Otherwise a later
            # rollback to this revision would resurrect the old templates.
            con.execute(
                "UPDATE template_revisions"
                " SET active_template_ids = ?, retired_template_ids = ?"
                " WHERE identity_id = ? AND revision = ?",
                (
                    json.dumps([new_tid]),
                    json.dumps(active_ids),
                    identity_id,
                    revision,
                ),
            )
            con.commit()
        except Exception:
            con.rollback()
            raise
        return LifecycleResult(
            identity_id=identity_id,
            action="re_enrolled",
            revision=revision,
            template_id=new_tid,
        )

    def rollback(self, identity_id: str, to_revision: int) -> LifecycleResult:
        """Restore generation *n* as the exact recorded snapshot.

        Non-destructive switching: every recorded generation stays
        addressable. The active set becomes exactly the target snapshot —
        outsiders retire regardless of revision number. The depth gate
        bounds backward travel only; forward/lateral switches are gated
        by snapshot existence (fail-closed `revision not found`).
        """
        con = self._connection()
        row = con.execute(
            "SELECT status, current_revision FROM identities WHERE id = ?",
            (identity_id,),
        ).fetchone()
        if row is None or row[0] != "active":
            raise StoreError(f"unknown identity: {identity_id}")
        current = int(row[1])
        if to_revision < 1:
            raise StoreError(f"revision out of range: {to_revision}")
        # Rollback is non-destructive generation switching: any recorded
        # generation stays addressable even when the current pointer sits
        # below it (e.g. rollback(1) then rollback(2)). The depth gate
        # only bounds backward travel; forward switches need no gate.
        if to_revision < current:
            if current - to_revision > self._policy.rollback_max_depth:
                raise StoreError(
                    f"revision {to_revision} exceeds rollback_max_depth"
                    f" {self._policy.rollback_max_depth}"
                )
        if to_revision == current:
            return LifecycleResult(
                identity_id=identity_id,
                action="rolled_back",
                revision=current,
            )
        targets = con.execute(
            "SELECT revision, active_template_ids FROM template_revisions"
            " WHERE identity_id = ? AND revision = ? ORDER BY id",
            (identity_id, to_revision),
        ).fetchall()
        if not targets:
            raise StoreError(f"revision not found: {to_revision}")
        if len(targets) > 1:
            raise StoreError(
                f"ambiguous revision snapshot: {to_revision}"
            )
        target = targets[0]
        keep_ids: set[str] = set(json.loads(str(target[1])))
        # Exact-snapshot restore: EVERY active template outside the
        # target snapshot retires, regardless of its revision number.
        # (A stale active from an older generation would otherwise
        # survive alongside the restored set.)
        outsiders = [
            row[0]
            for row in con.execute(
                "SELECT id FROM face_templates WHERE identity_id = ?"
                " AND status = 'active'",
                (identity_id,),
            ).fetchall()
            if row[0] not in keep_ids
        ]
        con.execute("BEGIN IMMEDIATE")
        try:
            for tid in outsiders:
                con.execute(
                    "UPDATE face_templates SET status = 'retired',"
                    " retired_at = datetime('now') WHERE id = ?",
                    (tid,),
                )
            for tid in keep_ids:
                con.execute(
                    "UPDATE face_templates SET status = 'active',"
                    " retired_at = NULL WHERE id = ? AND status = 'retired'",
                    (tid,),
                )
            con.execute(
                "UPDATE identities SET current_revision = ? WHERE id = ?",
                (to_revision, identity_id),
            )
            con.commit()
        except Exception:
            con.rollback()
            raise
        return LifecycleResult(
            identity_id=identity_id,
            action="rolled_back",
            revision=to_revision,
        )

    def delete_identity(self, identity_id: str) -> LifecycleResult:
        """Cryptographic erasure via the Journaled Tombstone Protocol."""
        self._repo.delete_identity(identity_id)
        return LifecycleResult(identity_id=identity_id, action="deleted")

    def reject_candidate(self, candidate_id: str) -> LifecycleResult:
        """Mark a pending candidate rejected (no promotion path remains)."""
        con = self._connection()
        con.execute("BEGIN IMMEDIATE")
        try:
            row = con.execute(
                "SELECT identity_id, status FROM candidate_templates"
                " WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise StoreError(f"unknown candidate: {candidate_id}")
            if str(row[1]) != CandidateStatus.PENDING.value:
                raise StoreError(f"candidate not pending: {candidate_id}")
            con.execute(
                "UPDATE candidate_templates SET status = 'rejected'"
                " WHERE id = ?",
                (candidate_id,),
            )
            con.commit()
        except Exception:
            con.rollback()
            raise
        return LifecycleResult(
            identity_id=str(row[0]),
            action="candidate_rejected",
            template_id=candidate_id,
        )

    def list_candidates(
        self, identity_id: str | None = None
    ) -> list[dict[str, object]]:
        """List candidate rows (optionally scoped to one identity)."""
        con = self._connection()
        query = (
            "SELECT id, identity_id, status,"
            " additional_corroboration_count, expires_at, created_at"
            " FROM candidate_templates"
        )
        if identity_id is None:
            rows = con.execute(query + " ORDER BY created_at, id").fetchall()
        else:
            rows = con.execute(
                query + " WHERE identity_id = ? ORDER BY created_at, id",
                (identity_id,),
            ).fetchall()
        return [
            {
                "candidate_id": str(row[0]),
                "identity_id": str(row[1]),
                "status": str(row[2]),
                "additional_corroboration_count": int(row[3]),
                "expires_at": str(row[4]),
                "created_at": str(row[5]),
            }
            for row in rows
        ]

    def show_identity(self, identity_id: str) -> dict[str, object]:
        """Stable identity snapshot for the CLI show/status surface."""
        con = self._connection()
        row = con.execute(
            "SELECT id, status, current_revision, created_at, updated_at"
            " FROM identities WHERE id = ?",
            (identity_id,),
        ).fetchone()
        if row is None:
            raise StoreError(f"unknown identity: {identity_id}")
        active = [
            r[0]
            for r in con.execute(
                "SELECT id FROM face_templates"
                " WHERE identity_id = ? AND status = 'active' ORDER BY id",
                (identity_id,),
            ).fetchall()
        ]
        return {
            "identity_id": str(row[0]),
            "status": str(row[1]),
            "current_revision": int(row[2]),
            "created_at": str(row[3]),
            "updated_at": str(row[4]),
            "active_template_ids": active,
        }

    def _connection(self) -> sqlite3.Connection:
        con = self._repo.connection
        if con is None:
            raise StoreError("repository not initialized")
        return con
