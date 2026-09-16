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
    - Single-evaluation rule: a holdout dataset can only be evaluated once for
      confirmation; re-evaluating new strategies or candidates on the same
      holdout is rejected.
    - Contamination is audited and recorded; original data is preserved.
    - Pure functions have no hidden disk/network side effects; custody is
      managed by recorder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Sequence

from facecore.errors import FaceCoreError
from facecore.research.experiment import (
    STUDY_SCHEMA_VERSION,
    AttemptRecord,
    ExperimentManifest,
    _require_iso8601,
    _require_study_schema,
)


class SplitContaminationError(FaceCoreError):
    """Split integrity violation or contamination detected (fail-closed)."""

    exit_code = 4


class HoldoutSealedError(FaceCoreError):
    """Attempt/bundle is sealed holdout; read refused before authorized release."""

    exit_code = 4


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CandidateFreeze:
    """Frozen pre-holdout candidate specification (§11.2)."""

    freeze_id: str
    experiment_id: str
    manifest_digest: str
    code_sha: str
    profile_digest: str
    analysis_digest: str
    planned_visit_ids: tuple[str, ...]
    frozen_at_utc: str
    schema_version: str = STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_study_schema(self.schema_version)
        if not self.freeze_id:
            raise ValueError("freeze_id must not be empty")
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if not self.manifest_digest:
            raise ValueError("manifest_digest must not be empty")
        if not self.code_sha:
            raise ValueError("code_sha must not be empty")
        if not self.profile_digest:
            raise ValueError("profile_digest must not be empty")
        if not self.analysis_digest:
            raise ValueError("analysis_digest must not be empty")
        if not self.planned_visit_ids:
            raise ValueError("planned_visit_ids must not be empty")
        if any(
            not isinstance(visit_id, str) or not visit_id or "/" in visit_id
            for visit_id in self.planned_visit_ids
        ):
            raise ValueError("planned_visit_ids must contain valid visit IDs")
        if len(set(self.planned_visit_ids)) != len(self.planned_visit_ids):
            raise ValueError("duplicate visit_id in planned_visit_ids")
        _require_iso8601(self.frozen_at_utc, "frozen_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "freeze_id": self.freeze_id,
            "experiment_id": self.experiment_id,
            "manifest_digest": self.manifest_digest,
            "code_sha": self.code_sha,
            "profile_digest": self.profile_digest,
            "analysis_digest": self.analysis_digest,
            "planned_visit_ids": list(self.planned_visit_ids),
            "frozen_at_utc": self.frozen_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateFreeze:
        _require_study_schema(data.get("schema_version", STUDY_SCHEMA_VERSION))
        return cls(
            freeze_id=data["freeze_id"],
            experiment_id=data["experiment_id"],
            manifest_digest=data["manifest_digest"],
            code_sha=data["code_sha"],
            profile_digest=data["profile_digest"],
            analysis_digest=data["analysis_digest"],
            planned_visit_ids=tuple(data["planned_visit_ids"]),
            frozen_at_utc=data["frozen_at_utc"],
            schema_version=data.get("schema_version", STUDY_SCHEMA_VERSION),
        )

    def digest(self) -> str:
        """Cryptographic hash of the canonical candidate freeze definition."""
        canonical = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class HoldoutRelease:
    """Authorized release for sealed holdout evaluation (§11.2)."""

    release_id: str
    experiment_id: str
    freeze_id: str
    freeze_digest: str
    operator_decision_id: str
    authorized_at_utc: str
    schema_version: str = STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_study_schema(self.schema_version)
        if not self.release_id:
            raise ValueError("release_id must not be empty")
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if not self.freeze_id:
            raise ValueError("freeze_id must not be empty")
        if not self.freeze_digest:
            raise ValueError("freeze_digest must not be empty")
        if not self.operator_decision_id:
            raise ValueError("operator_decision_id must not be empty")
        _require_iso8601(self.authorized_at_utc, "authorized_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "experiment_id": self.experiment_id,
            "freeze_id": self.freeze_id,
            "freeze_digest": self.freeze_digest,
            "operator_decision_id": self.operator_decision_id,
            "authorized_at_utc": self.authorized_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HoldoutRelease:
        _require_study_schema(data.get("schema_version", STUDY_SCHEMA_VERSION))
        return cls(
            release_id=data["release_id"],
            experiment_id=data["experiment_id"],
            freeze_id=data["freeze_id"],
            freeze_digest=data["freeze_digest"],
            operator_decision_id=data["operator_decision_id"],
            authorized_at_utc=data["authorized_at_utc"],
            schema_version=data.get("schema_version", STUDY_SCHEMA_VERSION),
        )

    def digest(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


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

    def __post_init__(self) -> None:
        _require_study_schema(self.schema_version)
        if not self.contamination_id:
            raise ValueError("contamination_id must not be empty")
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if not self.reason:
            raise ValueError("reason must not be empty")
        if not self.detected_at_utc:
            object.__setattr__(self, "detected_at_utc", _utc_now_iso())
        _require_iso8601(self.detected_at_utc, "detected_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contamination_id": self.contamination_id,
            "experiment_id": self.experiment_id,
            "reason": self.reason,
            "attempt_id": self.attempt_id,
            "visit_id": self.visit_id,
            "details": dict(self.details),
            "detected_at_utc": self.detected_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContaminationRecord:
        _require_study_schema(data.get("schema_version", STUDY_SCHEMA_VERSION))
        return cls(
            contamination_id=data["contamination_id"],
            experiment_id=data["experiment_id"],
            reason=data["reason"],
            attempt_id=data.get("attempt_id"),
            visit_id=data.get("visit_id"),
            details=dict(data.get("details", {})),
            detected_at_utc=data["detected_at_utc"],
            schema_version=data.get("schema_version", STUDY_SCHEMA_VERSION),
        )


def freeze_candidate(
    manifest: ExperimentManifest,
    *,
    code_sha: str,
    profile_digest: str,
    analysis_digest: str,
    planned_visit_ids: tuple[str, ...],
    frozen_at_utc: str | None = None,
) -> CandidateFreeze:
    """Pure function: freeze candidate parameters before prospective collection.

    Validation rules:
    - code_sha, profile_digest, analysis_digest must be non-empty.
    - planned_visit_ids must contain at least one non-empty visit_id,
      without duplicates.
    - If manifest carries a policy.profile_digest, it must match profile_digest.
    - Returns a CandidateFreeze instance with a deterministic freeze_id.
    """
    if not isinstance(manifest, ExperimentManifest):
        raise TypeError(f"expected ExperimentManifest, got {type(manifest).__name__}")
    if not code_sha or not isinstance(code_sha, str):
        raise ValueError("code_sha must not be empty")
    if len(code_sha) not in (40, 64) or any(
        char not in "0123456789abcdefABCDEF" for char in code_sha
    ):
        raise ValueError("code_sha must be a full 40- or 64-character hexadecimal SHA")
    if not profile_digest or not isinstance(profile_digest, str):
        raise ValueError("profile_digest must not be empty")
    if not analysis_digest or not isinstance(analysis_digest, str):
        raise ValueError("analysis_digest must not be empty")
    if not planned_visit_ids:
        raise ValueError("planned_visit_ids must not be empty")
    planned_visits = tuple(planned_visit_ids)
    if any(
        not isinstance(visit_id, str) or not visit_id or "/" in visit_id
        for visit_id in planned_visits
    ):
        raise ValueError("planned_visit_ids must contain valid visit IDs")
    if len(set(planned_visits)) != len(planned_visits):
        raise ValueError("duplicate visit_id in planned_visit_ids")

    software_section = manifest.sections.get("software", {})
    manifest_code_sha = software_section.get("code_sha")
    if manifest_code_sha and manifest_code_sha != code_sha:
        raise ValueError(
            f"code SHA mismatch: candidate {code_sha!r} != "
            f"manifest software {manifest_code_sha!r}"
        )

    policy_section = manifest.sections.get("policy", {})
    manifest_profile_digest = policy_section.get("profile_digest")
    if manifest_profile_digest and manifest_profile_digest != profile_digest:
        raise ValueError(
            "profile digest mismatch (profile_digest): candidate "
            f"{profile_digest!r} != manifest policy {manifest_profile_digest!r}"
        )

    frozen_at = frozen_at_utc or _utc_now_iso()
    raw_key = (
        f"{manifest.experiment_id}:{manifest.digest()}:{code_sha}:"
        f"{profile_digest}:{analysis_digest}:{','.join(planned_visits)}"
    )
    freeze_id = f"frz_{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:16]}"

    return CandidateFreeze(
        freeze_id=freeze_id,
        experiment_id=manifest.experiment_id,
        manifest_digest=manifest.digest(),
        code_sha=code_sha,
        profile_digest=profile_digest,
        analysis_digest=analysis_digest,
        planned_visit_ids=planned_visits,
        frozen_at_utc=frozen_at,
    )


def authorize_holdout(
    freeze: CandidateFreeze,
    *,
    operator_decision_id: str,
    authorized_at_utc: str | None = None,
) -> HoldoutRelease:
    """Pure function: authorize release of sealed holdout for evaluation."""
    if not isinstance(freeze, CandidateFreeze):
        raise TypeError(f"expected CandidateFreeze, got {type(freeze).__name__}")
    if not operator_decision_id or not isinstance(operator_decision_id, str):
        raise ValueError("operator_decision_id must be a non-empty string")

    authorized_at = authorized_at_utc or _utc_now_iso()
    raw_key = (
        f"{freeze.experiment_id}:{freeze.freeze_id}:{freeze.digest()}:"
        f"{operator_decision_id}"
    )
    release_id = f"rel_{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:16]}"

    return HoldoutRelease(
        release_id=release_id,
        experiment_id=freeze.experiment_id,
        freeze_id=freeze.freeze_id,
        freeze_digest=freeze.digest(),
        operator_decision_id=operator_decision_id,
        authorized_at_utc=authorized_at,
    )


def classify_split(
    visit_id: str,
    freeze: CandidateFreeze | None,
) -> str:
    """Classify visit into 'development' or 'holdout' based on candidate freeze."""
    if freeze and visit_id in freeze.planned_visit_ids:
        return "holdout"
    return "development"


def validate_candidate_freeze(
    manifest: ExperimentManifest,
    freeze: CandidateFreeze,
    *,
    existing_attempts: Sequence[AttemptRecord] = (),
) -> list[str]:
    """Validate candidate freeze against manifest and existing attempts."""
    errors: list[str] = []
    if freeze.experiment_id != manifest.experiment_id:
        errors.append(
            f"experiment_id mismatch: freeze {freeze.experiment_id} != "
            f"manifest {manifest.experiment_id}"
        )
    if freeze.manifest_digest != manifest.digest():
        errors.append(
            f"manifest digest mismatch: freeze {freeze.manifest_digest} != "
            f"manifest {manifest.digest()}"
        )
    manifest_code_sha = manifest.sections.get("software", {}).get("code_sha")
    if manifest_code_sha and freeze.code_sha != manifest_code_sha:
        errors.append(
            f"code SHA mismatch: freeze {freeze.code_sha} != "
            f"manifest software {manifest_code_sha}"
        )
    policy_prof = manifest.sections.get("policy", {}).get("profile_digest")
    if policy_prof and freeze.profile_digest != policy_prof:
        errors.append(
            f"profile digest mismatch: freeze {freeze.profile_digest} != "
            f"manifest policy {policy_prof}"
        )
    for att in existing_attempts:
        if att.visit_id in freeze.planned_visit_ids:
            errors.append(
                f"attempt {att.attempt_id} already exists for planned holdout visit "
                f"{att.visit_id}; freeze must precede visit collection"
            )
    return errors


def validate_holdout_release(
    freeze: CandidateFreeze,
    release: HoldoutRelease,
) -> list[str]:
    """Validate that a release matches the candidate freeze."""
    errors: list[str] = []
    if release.freeze_id != freeze.freeze_id:
        errors.append(
            f"freeze_id mismatch: release {release.freeze_id} != "
            f"freeze {freeze.freeze_id}"
        )
    if release.freeze_digest != freeze.digest():
        errors.append(
            f"freeze_digest mismatch: release {release.freeze_digest} != "
            f"freeze {freeze.digest()}"
        )
    return errors


def validate_holdout_candidate(
    freeze: CandidateFreeze,
    *,
    code_sha: str,
    profile_digest: str,
    analysis_digest: str,
    manifest_digest: str,
) -> list[str]:
    """Check candidate parameters against freeze; return list of mismatches."""
    mismatches: list[str] = []
    if code_sha != freeze.code_sha:
        mismatches.append(
            f"code_sha mismatch: expected {freeze.code_sha}, got {code_sha}"
        )
    if profile_digest != freeze.profile_digest:
        mismatches.append(
            f"profile_digest mismatch: expected {freeze.profile_digest}, "
            f"got {profile_digest}"
        )
    if analysis_digest != freeze.analysis_digest:
        mismatches.append(
            f"analysis_digest mismatch: expected {freeze.analysis_digest}, "
            f"got {analysis_digest}"
        )
    if manifest_digest != freeze.manifest_digest:
        mismatches.append(
            f"manifest_digest mismatch: expected {freeze.manifest_digest}, "
            f"got {manifest_digest}"
        )
    return mismatches
