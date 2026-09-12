"""IdentificationResult frozen contract (spec section 11 example key set/order).

Phase-1A states are exactly ``matched`` / ``review`` / ``unknown`` /
``invalid_input``. ``review``, ``unknown`` and ``invalid_input`` carry
``identity: null`` and never a guessed ``display_name``.
``candidate_created`` is always ``False`` in Phase 1A (no Phase-1B governance).
"""

import json
from dataclasses import dataclass, field
from enum import Enum

from facecore import SCHEMA_VERSION

# Phase-1B governance reason codes (Task 8 bounded response policy).
# Additive string constants only; the frozen spec §11 key set/order in
# `IdentificationResult.to_json` below is untouched.
DRIFT_BOUNDARY_EXCEEDED = "drift_boundary_exceeded"
IDENTITY_RE_ENROLLMENT_REQUIRED = "identity_re_enrollment_required"


class ResultStatus(str, Enum):
    MATCHED = "matched"
    REVIEW = "review"
    UNKNOWN = "unknown"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class Decision:
    score: float | None
    runner_up_score: float | None
    threshold: float | None
    margin: float | None
    reason_codes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Quality:
    status: str
    reason_codes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IdentificationResult:
    status: ResultStatus
    identity: dict[str, object] | None
    decision: Decision
    quality: Quality
    model_version: str
    template_revision: int
    candidate_created: bool

    def __post_init__(self) -> None:
        try:
            status = (
                self.status
                if isinstance(self.status, ResultStatus)
                else ResultStatus(self.status)
            )
        except ValueError:
            raise ValueError(
                f"status must be one of four Phase-1A states, got {self.status!r}"
            ) from None
        object.__setattr__(self, "status", status)
        if self.status is not ResultStatus.MATCHED and self.identity is not None:
            raise ValueError(
                f"{self.status.value} must carry identity null, never a guess"
            )
        if self.candidate_created:
            raise ValueError("candidate_created is always false in Phase 1A")

    def to_json(self) -> str:
        """Serialize with the frozen spec section 11 key order."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": self.status.value,
            "identity": self.identity,
            "decision": {
                "score": self.decision.score,
                "runner_up_score": self.decision.runner_up_score,
                "threshold": self.decision.threshold,
                "margin": self.decision.margin,
                "reason_codes": list(self.decision.reason_codes),
            },
            "quality": {
                "status": self.quality.status,
                "reason_codes": list(self.quality.reason_codes),
            },
            "model_version": self.model_version,
            "template_revision": self.template_revision,
            "candidate_created": self.candidate_created,
        }
        return json.dumps(payload)
