"""Identification policy: band assignment over gallery scores (Task 7).

Per codex ruling on PR-F (frozen contract holds): aggregation_strategy is
NOT recorded in IdentificationResult — it lives versioned in PolicyProfile,
which identify() receives, so a Phase-1B change is detectable by policy
comparison. Deviation from the plan sentence is declared in the PR body.

Gallery embeddings arrive as an explicit mapping (identity_id -> vector);
the Repository guards identity existence, revision, and model_version.
"""

import numpy as np

from facecore.contracts.drift import DriftStatus
from facecore.contracts.policy import PolicyProfile
from facecore.contracts.result import (
    DRIFT_BOUNDARY_EXCEEDED,
    Decision,
    IdentificationResult,
    Quality,
    ResultStatus,
)
from facecore.repository.base import Repository


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        raise ValueError("zero-norm embedding cannot be scored")
    return float(np.dot(a, b) / denom)


def identify(
    probe: np.ndarray,
    repository: Repository,
    policy: PolicyProfile,
    *,
    gallery: dict[str, np.ndarray],
    model_version: str,
    quality: Quality | None = None,
    drift_status: DriftStatus | None = None,
) -> IdentificationResult:
    """Assign matched / review / unknown over one probe against the gallery.

    Task 8 bounded response: a breached ``drift_status`` downgrades an
    otherwise-matched outcome to ``review`` with
    ``drift_boundary_exceeded`` (identity stays null per the frozen
    review contract). Default ``None`` preserves legacy behavior.
    """
    match_t = policy.match_threshold
    review_t = policy.review_threshold
    margin_t = policy.margin_threshold
    if match_t is None or review_t is None or margin_t is None:
        raise ValueError("thresholds are swept, not chosen — set them before identify")
    scored: list[tuple[str, str, float]] = []
    for identity_id, vector in gallery.items():
        view = repository.get_identity(identity_id)
        if view is None:
            raise KeyError(f"gallery identity not enrolled: {identity_id}")
        if view.active_template.model_version != model_version:
            other = view.active_template.model_version
            raise ValueError(
                f"cross-model comparison refused: {other!r} vs {model_version!r}"
            )
        scored.append((identity_id, view.display_name, cosine_score(probe, vector)))
    scored.sort(key=lambda item: item[2], reverse=True)
    if not scored:
        return IdentificationResult(
            status=ResultStatus.UNKNOWN,
            identity=None,
            decision=Decision(
                score=None,
                runner_up_score=None,
                threshold=review_t,
                margin=None,
                reason_codes=["below_review_threshold"],
            ),
            quality=quality or Quality(status="accepted", reason_codes=[]),
            model_version=model_version,
            template_revision=1,
            candidate_created=False,
        )
    top_id, top_name, top_score = scored[0]
    if len(scored) == 1:
        runner_up: float | None = None
        margin: float | None = None
        extra = ["margin_unavailable_single_identity"]
    else:
        runner_up = scored[1][2]
        margin = top_score - runner_up
        extra = []
    breached = drift_status is DriftStatus.BOUNDARY_EXCEEDED
    if top_score >= match_t and (margin is None or margin >= margin_t):
        if breached:
            status = ResultStatus.REVIEW
            codes = ["match_threshold_met", DRIFT_BOUNDARY_EXCEEDED, *extra]
        else:
            status = ResultStatus.MATCHED
            codes = ["match_threshold_met", *extra]
    elif top_score >= match_t:
        status = ResultStatus.REVIEW
        codes = ["insufficient_margin"]
    elif top_score >= review_t:
        status = ResultStatus.REVIEW
        codes = ["below_match_threshold"]
    else:
        return IdentificationResult(
            status=ResultStatus.UNKNOWN,
            identity=None,
            decision=Decision(
                score=top_score,
                runner_up_score=runner_up,
                threshold=review_t,
                margin=margin,
                reason_codes=["below_review_threshold"],
            ),
            quality=quality or Quality(status="accepted", reason_codes=[]),
            model_version=model_version,
            template_revision=1,
            candidate_created=False,
        )
    identity: dict[str, object] | None = (
        {"id": top_id, "display_name": top_name, "metadata": {}}
        if status is ResultStatus.MATCHED
        else None
    )
    return IdentificationResult(
        status=status,
        identity=identity,
        decision=Decision(
            score=top_score,
            runner_up_score=runner_up,
            threshold=match_t if status is ResultStatus.MATCHED else review_t,
            margin=margin,
            reason_codes=codes,
        ),
        quality=quality or Quality(status="accepted", reason_codes=[]),
        model_version=model_version,
        template_revision=1,
        candidate_created=False,
    )
