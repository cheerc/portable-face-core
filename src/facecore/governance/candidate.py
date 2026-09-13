"""Confirmation-gated shadow candidate pipeline (1B plan §11 Task 3).

Creation is strictly gated: (a) match score >= ``candidate_update_threshold``
(provisional 0.88), (b) quality ``accepted``, (c) explicit ``correct``
confirmation. ``not_me`` / cancelled / timeout / EOF never touch the database.

Pending observations live strictly in memory. The pipeline holds only the byte
length of a held observation — never the bytes themselves — so process exit
leaves zero unencrypted artifacts. Candidate insertion delegates DEK creation,
AEAD sealing, and geometry validation to `SQLiteRepository`.

`CandidateDecision` reports only facts this pipeline owns (the created
candidate's ID and seed status). It never fabricates key references or
ciphertext handles: those live exclusively behind the repository boundary,
which Task 3 does not modify.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from facecore.contracts.confirmation import ConfirmationRequest
from facecore.contracts.policy import GovernancePolicy
from facecore.contracts.result import IdentificationResult, ResultStatus
from facecore.storage.sqlite_repo import SQLiteRepository

# Provisional shadow-candidate expiry window (DEVIATION: no candidate TTL exists
# in plan §6, spec §9, or the dispatch; 7 days bounds unconfirmed biometric
# accumulation pending operator TTL ratification via Task 4/5 retention work).
CANDIDATE_EXPIRY_DAYS = 7


@dataclass(frozen=True)
class CandidateDecision:
    created: bool
    candidate_id: str | None
    promoted: bool = False
    reason: str = ""


class CandidatePipeline:
    """Evaluate one identification observation against the creation gate."""

    def __init__(
        self,
        repository: SQLiteRepository,
        policy: GovernancePolicy | None = None,
        generation_id: str | None = None,
    ) -> None:
        self._repository = repository
        self._policy = policy or GovernancePolicy.provisional_v1()
        if generation_id is not None and not generation_id:
            raise ValueError("generation_id must be non-empty")
        self._generation_id = generation_id
        self._pending_lengths: list[int] = []

    @property
    def generation_id(self) -> str:
        """Generation stamped on new candidates (store marker by default)."""
        if self._generation_id is not None:
            return self._generation_id
        return self._repository.get_current_generation()

    def hold_pending(self, observation: bytes) -> None:
        """Hold a pending observation in memory (length only, never the bytes)."""
        self._pending_lengths.append(len(observation))

    def discard_pending(self) -> None:
        self._pending_lengths.clear()

    def pending_count(self) -> int:
        return len(self._pending_lengths)

    def pending_bytes(self) -> int:
        return sum(self._pending_lengths)

    def evaluate_observation(
        self,
        result: IdentificationResult,
        decoded_face: bytes,
        confirmation: ConfirmationRequest,
    ) -> CandidateDecision:
        """Apply the strict creation gate; never promote on a seed event."""
        if result.status is not ResultStatus.MATCHED:
            return CandidateDecision(
                created=False, candidate_id=None, reason="non_matched_status"
            )
        score = result.decision.score
        if score is None or score < self._policy.candidate_update_threshold:
            return CandidateDecision(
                created=False, candidate_id=None, reason="below_update_threshold"
            )
        if result.quality.status != "accepted":
            return CandidateDecision(
                created=False, candidate_id=None, reason="quality_not_accepted"
            )
        if not confirmation.is_affirmative:
            return CandidateDecision(
                created=False, candidate_id=None, reason="confirmation_not_correct"
            )
        identity = result.identity
        if not isinstance(identity, dict) or not identity.get("id"):
            return CandidateDecision(
                created=False, candidate_id=None, reason="missing_identity"
            )
        identity_id = str(identity["id"])
        candidate_id = f"c-{identity_id}-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)
        self._repository.add_candidate_record(
            candidate_id,
            identity_id,
            decoded_face,
            decoded_face,
            (now + timedelta(days=CANDIDATE_EXPIRY_DAYS)).isoformat(),
            generation_id=self.generation_id,
            exemplar_crop_box=None,
            exemplar_landmarks=None,
            quality_score=score,
            additional_corroboration_count=0,
            evidence_log="[]",
            created_at=now.isoformat(),
        )
        return CandidateDecision(
            created=True,
            candidate_id=candidate_id,
            promoted=False,
            reason="seed_created_pending",
        )
