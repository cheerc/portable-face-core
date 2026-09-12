"""Phase-1B candidate contracts (plan §7 schema: candidate_templates).

`CandidateTemplate` is created strictly through the confirmation-gated shadow
flow (Task 3): newly created candidates carry ``status=pending`` and
``additional_corroboration_count=0``. A seed event can never promote itself.
"""

from dataclasses import dataclass, field
from enum import Enum

from facecore.contracts.crypto import EncryptedBlob


class CandidateStatus(str, Enum):
    PENDING = "pending"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    EXPIRED = "expired"
    GENERATION_RETIRED = "generation_retired"


@dataclass(frozen=True)
class EvidenceRecord:
    event_type: str
    timestamp: str
    sequence_number: int
    score: float | None = None


@dataclass(frozen=True)
class CandidateTemplate:
    template_id: str
    identity_id: str
    generation_id: str
    key_id: str
    encrypted_embedding: EncryptedBlob
    status: CandidateStatus = CandidateStatus.PENDING
    additional_corroboration_count: int = 0
    encrypted_exemplar: EncryptedBlob | None = None
    exemplar_crop_box: tuple[float, float, float, float] | None = None
    exemplar_landmarks: tuple[tuple[float, float], ...] | None = None
    quality_score: float = 0.0
    evidence_log: tuple[EvidenceRecord, ...] = field(default_factory=tuple)
    expires_at: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.status, str) and not isinstance(
            self.status, CandidateStatus
        ):
            try:
                object.__setattr__(self, "status", CandidateStatus(self.status))
            except ValueError:
                raise ValueError(
                    f"status must be one of five lifecycle states, "
                    f"got {self.status!r}"
                ) from None
        if self.additional_corroboration_count < 0:
            raise ValueError(
                "additional_corroboration_count must be >= 0, "
                f"got {self.additional_corroboration_count}"
            )
