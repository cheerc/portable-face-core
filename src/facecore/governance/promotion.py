"""Promotion gate with exclusivity margin and generation check (plan Task 4).

Pure decision logic. Promotion strictly requires: (a) count >=
``additional_corroboration_min_events`` (provisional 1), (b) cross-identity
exclusivity margin >= ``promotion_margin`` (provisional 0.12), (c) candidate
generation equals the active model generation. Only ``pending`` candidates
are promotable; a seed event can never promote itself.

`build_promotion_template` constructs the active `FaceTemplate` carrying the
candidate's encrypted exemplar, crop, landmarks, key reference, and
generation. Persistence (status transition + row insert) belongs to the
caller (Task 5 lifecycle) using existing repository methods.
"""

import math
from dataclasses import dataclass

from facecore.contracts.candidate import CandidateStatus, CandidateTemplate
from facecore.contracts.policy import GovernancePolicy
from facecore.contracts.template import FaceTemplate, TemplateRevision


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    candidate_status: str
    reason: str


class PromotionManager:
    """Evaluate one candidate against the strict promotion gate."""

    def __init__(
        self,
        current_generation: str,
        policy: GovernancePolicy | None = None,
    ) -> None:
        self._generation = current_generation
        self._policy = policy or GovernancePolicy.provisional_v1()

    def evaluate(
        self, candidate: CandidateTemplate, margin: float
    ) -> PromotionDecision:
        """Return the gate verdict; never mutate candidate state."""
        if candidate.status is not CandidateStatus.PENDING:
            return PromotionDecision(
                promote=False,
                candidate_status=candidate.status.value,
                reason="not_pending",
            )
        if (
            candidate.additional_corroboration_count
            < self._policy.additional_corroboration_min_events
        ):
            return PromotionDecision(
                promote=False,
                candidate_status=candidate.status.value,
                reason="insufficient_corroboration",
            )
        if not math.isfinite(margin) or margin < self._policy.promotion_margin:
            return PromotionDecision(
                promote=False,
                candidate_status=candidate.status.value,
                reason="insufficient_margin",
            )
        if candidate.generation_id != self._generation:
            return PromotionDecision(
                promote=False,
                candidate_status=candidate.status.value,
                reason="generation_mismatch",
            )
        return PromotionDecision(
            promote=True,
            candidate_status=candidate.status.value,
            reason="promotion_approved",
        )

    @staticmethod
    def build_promotion_template(
        candidate: CandidateTemplate,
        new_template_id: str,
        revision: TemplateRevision,
        model_version: str,
        embedding_dim: int,
    ) -> FaceTemplate:
        """Carry the candidate's exemplar and crop into a new active template.

        Runtime properties (model version, embedding dimension) are explicit
        caller inputs: the engine never fabricates them from generation IDs.
        """
        return FaceTemplate(
            template_id=new_template_id,
            identity_id=candidate.identity_id,
            model_version=model_version,
            embedding_dim=embedding_dim,
            generation_id=candidate.generation_id,
            revision=revision,
            encrypted_exemplar=candidate.encrypted_exemplar,
            exemplar_crop_box=candidate.exemplar_crop_box,
            exemplar_landmarks=candidate.exemplar_landmarks,
            key_id=candidate.key_id,
            quality_score=candidate.quality_score,
        )
