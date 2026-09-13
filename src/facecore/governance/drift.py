"""Drift indicators and bounded response policy (1B plan §11 Task 8).

``DriftDetector`` tracks the active-template centroid against a reference:

- ``anchor``: the centroid first observed for the identity, pinned to the
  template that produced it. The anchor carries an observed-at timestamp
  and expires when (a) its template leaves the active set (evicted), or
  (b) it ages past ``retired_retention_days`` (provisional 90 days).
  Expired anchors are dropped — never held as unmanaged permanent
  embeddings.
- ``rolling``: once the anchor expires, the reference becomes current
  bank diversity (max pairwise cosine distance among actives). A
  diverging bank reads as drift without any permanent anchor.

``measure_identity_drift`` never crashes on eviction: a missing anchor
template resolves to ``rolling``, not ``None.embedding``.

``enforce_policy`` applies the bounded response triple on breach
(shift > ``drift_max_centroid_shift``): marks the identity
``re_enrollment_required`` in the store. Query-side downgrade
(matched → review + ``drift_boundary_exceeded``) is the ``identify()``
``drift_status`` hook; creation/promotion suspension is answered by
``is_suspended`` for the caller to gate on.
"""

import math
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np

from facecore.contracts.drift import (
    DriftMetrics,
    DriftPolicy,
    DriftReferenceKind,
    DriftStatus,
)
from facecore.contracts.policy import GovernancePolicy
from facecore.errors import StoreError
from facecore.storage.sqlite_repo import SQLiteRepository


class NonFiniteDriftInputError(ValueError):
    """Non-finite drift vector or shift rejected at the boundary."""


def _require_finite_vector(name: str, vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise NonFiniteDriftInputError(
            f"{name} contains non-finite components"
        )
    return values


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    clean_a = _require_finite_vector("reference", a)
    clean_b = _require_finite_vector("observed", b)
    denom = float(np.linalg.norm(clean_a) * np.linalg.norm(clean_b))
    if denom == 0.0:
        raise ValueError("zero-norm embedding cannot be scored")
    shift = float(1.0 - np.dot(clean_a, clean_b) / denom)
    if not math.isfinite(shift):
        raise NonFiniteDriftInputError("cosine shift is non-finite")
    return shift


def _centroid(vectors: list[np.ndarray]) -> np.ndarray:
    clean = [_require_finite_vector(f"vector[{i}]", v) for i, v in enumerate(vectors)]
    stacked = np.stack(clean)
    mean = np.mean(stacked, axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        raise ValueError("degenerate centroid has zero norm")
    result = mean / norm
    if not np.all(np.isfinite(result)):
        raise NonFiniteDriftInputError("centroid is non-finite")
    return result


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_days(observed_at: str, now: str) -> float:
    try:
        start = datetime.fromisoformat(observed_at)
        end = datetime.fromisoformat(now)
    except (ValueError, TypeError):
        return math.inf
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return (end - start).total_seconds() / 86400.0


class DriftDetector:
    """Measure per-identity drift against a lifecycle-safe reference."""

    def __init__(
        self,
        repository: SQLiteRepository,
        policy: GovernancePolicy | None = None,
        vector_reader: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self._repo = repository
        self._policy = policy or GovernancePolicy.provisional_v1()
        self._reader = vector_reader or self._default_reader
        # identity_id -> (anchor vector, observed_at iso, template_id)
        self._anchors: dict[str, tuple[np.ndarray, str, str]] = {}

    def measure_identity_drift(self, identity_id: str) -> DriftMetrics:
        """Return centroid shift with the live reference kind."""
        con = self._connection()
        rows = con.execute(
            "SELECT id FROM face_templates"
            " WHERE identity_id = ? AND status = 'active' ORDER BY id",
            (identity_id,),
        ).fetchall()
        if not rows:
            raise StoreError(f"no active template for: {identity_id}")
        active_ids = [str(row[0]) for row in rows]
        vectors = [self._reader(template_id) for template_id in active_ids]
        centroid = _centroid(vectors)
        now = _utcnow()
        anchor = self._anchors.get(identity_id)
        if anchor is not None:
            anchor_vector, observed_at, anchor_template = anchor
            alive = anchor_template in active_ids
            fresh = (
                _age_days(observed_at, now)
                <= self._policy.retired_retention_days
            )
            if alive and fresh:
                return DriftMetrics(
                    identity_id=identity_id,
                    centroid_shift=_cosine_distance(centroid, anchor_vector),
                    reference_kind=DriftReferenceKind.ANCHOR.value,
                    observed_at=now,
                )
            # Expired anchors are dropped, never retained.
            del self._anchors[identity_id]
        else:
            # First observation pins the anchor to the current centroid.
            self._anchors[identity_id] = (centroid, now, active_ids[0])
            return DriftMetrics(
                identity_id=identity_id,
                centroid_shift=0.0,
                reference_kind=DriftReferenceKind.ANCHOR.value,
                observed_at=now,
            )
        return DriftMetrics(
            identity_id=identity_id,
            centroid_shift=self._diversity(vectors),
            reference_kind=DriftReferenceKind.ROLLING.value,
            observed_at=now,
        )

    def reset_anchor(self, identity_id: str) -> None:
        """Drop the anchor (re-enroll calls this: next measure re-pins)."""
        self._anchors.pop(identity_id, None)

    def enforce_policy(
        self, identity_id: str, metrics: DriftMetrics
    ) -> DriftStatus:
        """Assess and, on breach, mark re_enrollment_required in store."""
        status = DriftPolicy().assess(metrics, self._policy)
        if status is DriftStatus.BOUNDARY_EXCEEDED:
            con = self._connection()
            con.execute("BEGIN IMMEDIATE")
            try:
                con.execute(
                    "UPDATE identities SET status = 're_enrollment_required'"
                    " WHERE id = ?",
                    (identity_id,),
                )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return status

    def is_suspended(self, identity_id: str) -> bool:
        """True when creation/promotion must halt for the identity."""
        return self._repo.get_identity_status(identity_id) == (
            "re_enrollment_required"
        )

    @staticmethod
    def _diversity(vectors: list[np.ndarray]) -> float:
        if len(vectors) < 2:
            return 0.0
        return max(
            _cosine_distance(a, b)
            for i, a in enumerate(vectors)
            for b in vectors[i + 1 :]
        )

    def _default_reader(self, template_id: str) -> np.ndarray:
        raw = self._repo.read_embedding(template_id)
        return np.frombuffer(raw, dtype=np.float64).copy()

    def _connection(self) -> sqlite3.Connection:
        con = self._repo.connection
        if con is None:
            raise StoreError("repository not initialized")
        return con
