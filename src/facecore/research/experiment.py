"""Phase 2B study contracts: manifest, attempt, label (E1, G1).

Source of truth:
    - docs/plans/2026-09-16-phase2b-mac-recognition-execution-plan.md
      §4 (manifest blocks), §11.2 (signatures), §12 E1;
    - Task: t-20260916021626405371-53424-11.

Hard boundaries:
    - AttemptRecord carries NO truth field: attempts count every accepted
      Start (open failure, cancel, crash) without knowing the answer.
    - EvaluationLabel travels evaluator-only: scorer, quality rank,
      acquisition prompts, support rules, and frame selection must never
      read it. Labels land in the encrypted sidecar, never in inference.
    - Study schema is explicitly versioned ("v2"); v1 session envelopes
      in records.py are untouched and stay readable/deletable. Unknown
      schema versions are refused fail-closed, never backfilled.

Only synthetic payloads in tests; never real faces; camera-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from typing import Any

#: Study envelope schema version. v1 is the Phase 2A session envelope in
#: records.py; v2 is the Phase 2B attempt/label generation.
STUDY_SCHEMA_VERSION = "v2"

MANIFEST_SECTIONS = (
    "identity",
    "software",
    "gallery",
    "policy",
    "capture",
    "privacy",
    "study",
    "analysis",
)

#: Operational states an attempt may rest in. Terminal inference outcomes
#: (matched/review/...) are SessionResult territory, not attempt states:
#: the attempt ledger answers "was it tried and what happened
#: operationally", never "was it recognized correctly".
ATTEMPT_STATUSES = frozenset(
    {
        "accepted",
        "open_error",
        "setup_error",
        "cancelled",
        "timeout",
        "completed",
        "error",
    }
)

LABEL_KINDS = frozenset({"enrolled", "unenrolled", "uncertain"})


def _require_iso8601(value: str | None, field_name: str) -> None:
    if not value or not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be ISO 8601, got {value!r}"
        ) from exc


def _require_study_schema(version: object) -> None:
    if version != STUDY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported study schema {version!r}; "
            f"expected {STUDY_SCHEMA_VERSION!r}"
        )


@dataclass(frozen=True)
class ExperimentManifest:
    """Frozen pre-collection manifest (§4 eight blocks).

    Sections travel as plain mappings so G3 can freeze new keys without
    a code change; only ``identity.experiment_id`` is structural.
    """

    sections: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [key for key in MANIFEST_SECTIONS if key not in self.sections]
        if missing:
            raise ValueError(f"manifest missing blocks: {missing}")
        identity = self.sections.get("identity")
        if not isinstance(identity, dict) or not identity.get("experiment_id"):
            raise ValueError("manifest identity must carry experiment_id")

    @property
    def experiment_id(self) -> str:
        identity = self.sections["identity"]
        assert isinstance(identity, dict)
        experiment_id = identity["experiment_id"]
        assert isinstance(experiment_id, str)
        return experiment_id

    def to_dict(self) -> dict[str, Any]:
        return {key: dict(self.sections[key]) for key in MANIFEST_SECTIONS}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentManifest:
        if not isinstance(data, dict):
            raise ValueError("manifest must be a JSON object")
        sections = {}
        for key in MANIFEST_SECTIONS:
            block = data.get(key)
            if not isinstance(block, dict):
                raise ValueError(f"manifest block {key!r} must be an object")
            sections[key] = dict(block)
        return cls(sections=sections)

    def digest(self) -> str:
        """Stable content hash: any frozen-input change changes the batch."""
        canonical = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class AttemptRecord:
    """One accepted research Start, durable before camera/model work.

    ``accepted_at_utc`` (durable encrypted write success) is the boundary:
    earlier refusals are start_denied, never recorded attempts.
    """

    experiment_id: str
    attempt_id: str
    participant_id: str
    visit_id: str
    condition_id: str
    attempt_index: int
    retry_of: str | None
    consent_ref: str
    requested_at_utc: str
    accepted_at_utc: str
    started_at_utc: str | None
    ended_at_utc: str | None
    operational_status: str
    error_code: str | None
    bundle_ref: str | None
    schema_version: str = STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_study_schema(self.schema_version)
        for name in (
            "experiment_id",
            "attempt_id",
            "participant_id",
            "visit_id",
            "condition_id",
            "consent_ref",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if "/" in self.attempt_id or "/" in self.experiment_id:
            raise ValueError("experiment/attempt ids must not contain '/'")
        if self.attempt_index < 1:
            raise ValueError("attempt_index must be >= 1")
        _require_iso8601(self.requested_at_utc, "requested_at_utc")
        _require_iso8601(self.accepted_at_utc, "accepted_at_utc")
        if self.started_at_utc is not None:
            _require_iso8601(self.started_at_utc, "started_at_utc")
        if self.ended_at_utc is not None:
            _require_iso8601(self.ended_at_utc, "ended_at_utc")
        if self.operational_status not in ATTEMPT_STATUSES:
            raise ValueError(
                f"unknown operational_status {self.operational_status!r}; "
                f"expected one of {sorted(ATTEMPT_STATUSES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "attempt_id": self.attempt_id,
            "participant_id": self.participant_id,
            "visit_id": self.visit_id,
            "condition_id": self.condition_id,
            "attempt_index": self.attempt_index,
            "retry_of": self.retry_of,
            "consent_ref": self.consent_ref,
            "requested_at_utc": self.requested_at_utc,
            "accepted_at_utc": self.accepted_at_utc,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "operational_status": self.operational_status,
            "error_code": self.error_code,
            "bundle_ref": self.bundle_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttemptRecord:
        _require_study_schema(data.get("schema_version", STUDY_SCHEMA_VERSION))
        return cls(
            experiment_id=data["experiment_id"],
            attempt_id=data["attempt_id"],
            participant_id=data["participant_id"],
            visit_id=data["visit_id"],
            condition_id=data["condition_id"],
            attempt_index=int(data["attempt_index"]),
            retry_of=data.get("retry_of"),
            consent_ref=data["consent_ref"],
            requested_at_utc=data["requested_at_utc"],
            accepted_at_utc=data["accepted_at_utc"],
            started_at_utc=data.get("started_at_utc"),
            ended_at_utc=data.get("ended_at_utc"),
            operational_status=data["operational_status"],
            error_code=data.get("error_code"),
            bundle_ref=data.get("bundle_ref"),
        )


@dataclass(frozen=True)
class EvaluationLabel:
    """Post-terminal ground truth, evaluator-only, revisioned.

    Corrections append a new revision; revision 1 is never rewritten, so
    a label fix cannot silently change the original inference record.
    """

    attempt_id: str
    revision: int
    kind: str
    identity_id: str | None
    actor_ref: str
    labeled_at: str
    schema_version: str = STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_study_schema(self.schema_version)
        if not self.attempt_id:
            raise ValueError("attempt_id must not be empty")
        if self.revision < 1:
            raise ValueError("revision must be >= 1")
        if self.kind not in LABEL_KINDS:
            raise ValueError(
                f"unknown label kind {self.kind!r}; "
                f"expected one of {sorted(LABEL_KINDS)}"
            )
        if self.kind == "enrolled" and not self.identity_id:
            raise ValueError("enrolled labels must carry identity_id")
        if self.kind != "enrolled" and self.identity_id is not None:
            raise ValueError(f"{self.kind} labels must not carry identity_id")
        if not self.actor_ref:
            raise ValueError("actor_ref must not be empty")
        _require_iso8601(self.labeled_at, "labeled_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "revision": self.revision,
            "kind": self.kind,
            "identity_id": self.identity_id,
            "actor_ref": self.actor_ref,
            "labeled_at": self.labeled_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationLabel:
        _require_study_schema(data.get("schema_version", STUDY_SCHEMA_VERSION))
        return cls(
            attempt_id=data["attempt_id"],
            revision=int(data["revision"]),
            kind=data["kind"],
            identity_id=data.get("identity_id"),
            actor_ref=data["actor_ref"],
            labeled_at=data["labeled_at"],
        )
