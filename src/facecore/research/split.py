"""Prospective holdout and candidate freeze contracts and pure functions (E6).

Source of truth:
    - docs/specs/2026-09-16-phase2b-mac-recognition-research.md §8;
    - docs/plans/2026-09-16-phase2b-mac-recognition-execution-plan.md
      §11.2, §12 E6;
    - ADR 0010 (Phase 2B evidence isolation).

Invariants:
    - Candidate freeze must precede collection of future visits (prospective holdout).
    - Same visit cannot cross splits (must not appear in both development and holdout).
    - Already-used development visits cannot be marked or reused as holdout.
    - Candidate hashes (code SHA, profile digest, analysis digest, manifest digest)
      are frozen and immutable; candidate/hash swaps are rejected fail-closed.
    - Unreleased holdout data must not be subject to content analysis.
    - Single-evaluation rule: a holdout dataset can only be evaluated once for confirmation;
      re-evaluating new strategies or candidates on the same holdout is rejected.
    - Contamination is audited and recorded; original data is preserved.
    - Pure functions have no hidden disk/network side effects; custody is managed by recorder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from facecore.errors import FaceCoreError
from facecore.research.experiment import (
    STUDY_SCHEMA_VERSION,
    AttemptRecord,
    ExperimentManifest,
)


class SplitContaminationError(FaceCoreError):
    """Split integrity violation or contamination detected (fail-closed)."""

    exit_code = 4


class HoldoutSealedError(FaceCoreError):
    """Attempt/bundle is sealed holdout; read refused before authorized release."""

    exit_code = 4


@dataclass(frozen=True)
class CandidateFreeze:
    """Frozen pre-holdout candidate specification (§11.2)."""

    freeze_id: str
    manifest_digest: str
    code_sha: str
    profile_digest: str
    analysis_digest: str
    planned_visit_ids: tuple[str, ...]
    frozen_at_utc: str
    schema_version: str = STUDY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateFreeze:
        raise NotImplementedError

    def digest(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class HoldoutRelease:
    """Authorized release for sealed holdout evaluation (§11.2)."""

    release_id: str
    freeze_id: str
    freeze_digest: str
    operator_decision_id: str
    authorized_at_utc: str
    schema_version: str = STUDY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HoldoutRelease:
        raise NotImplementedError

    def digest(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class ContaminationRecord:
    """Audit record for split contamination or policy violations."""

    contamination_id: str
    experiment_id: str
    reason: str
    attempt_id: str | None = None
    visit_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    detected_at_utc: str = ""
    schema_version: str = STUDY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContaminationRecord:
        raise NotImplementedError


def freeze_candidate(
    manifest: ExperimentManifest,
    *,
    code_sha: str,
    profile_digest: str,
    analysis_digest: str,
    planned_visit_ids: tuple[str, ...],
    frozen_at_utc: str | None = None,
) -> CandidateFreeze:
    """Pure function: freeze candidate parameters before prospective collection."""
    raise NotImplementedError


def authorize_holdout(
    freeze: CandidateFreeze,
    *,
    operator_decision_id: str,
    authorized_at_utc: str | None = None,
) -> HoldoutRelease:
    """Pure function: authorize release of sealed holdout for evaluation."""
    raise NotImplementedError


def classify_split(
    visit_id: str,
    freeze: CandidateFreeze | None,
) -> str:
    """Classify visit into 'development' or 'holdout' based on candidate freeze."""
    raise NotImplementedError


def validate_candidate_freeze(
    manifest: ExperimentManifest,
    freeze: CandidateFreeze,
    *,
    existing_attempts: list[AttemptRecord] = (),
) -> list[str]:
    """Validate candidate freeze against manifest and existing attempts."""
    raise NotImplementedError


def validate_holdout_release(
    freeze: CandidateFreeze,
    release: HoldoutRelease,
) -> list[str]:
    """Validate that a release matches the candidate freeze."""
    raise NotImplementedError
