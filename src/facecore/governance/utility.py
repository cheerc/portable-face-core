"""6-factor utility rescoring with exact plan §6 formulas (Task 4).

Pure functions over explicit `TemplateSignals`. Every component maps to
$[0.0, 1.0]$; positive weights sum to $1.00$ by construction; the composite
is outer-clamped, so scores are unconditionally in $[0.0, 1.0]$.
"""

import math
from dataclasses import dataclass


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class TemplateSignals:
    variance_laplacian: float
    detector_confidence: float
    additional_corroboration_count: int
    age_days: float
    cosine_to_centroid: float
    max_peer_cosine: float
    runner_up_similarity: float | None
    bank_size: int


class UtilityRescorer:
    """Score one template with the exact §6 component formulas."""

    W_Q = 0.30
    W_S = 0.30
    W_R = 0.20
    W_C = 0.20
    W_RED = 0.15
    W_O = 0.15

    @staticmethod
    def quality(signals: TemplateSignals) -> float:
        """U_Q: normalized sharpness blended with detector confidence."""
        return _clamp(
            0.5 * signals.variance_laplacian / 120.0
            + 0.5 * signals.detector_confidence
        )

    @staticmethod
    def support(signals: TemplateSignals) -> float:
        """U_S: corroboration count normalized to 3 events."""
        return _clamp(signals.additional_corroboration_count / 3.0)

    @staticmethod
    def recency(signals: TemplateSignals) -> float:
        """U_R: half-life decay over 180 days."""
        return _clamp(math.exp(-math.log(2.0) / 180.0 * signals.age_days))

    @staticmethod
    def coverage(signals: TemplateSignals) -> float:
        """U_C: (1 - cos)/2; single-template bank scores 1.0."""
        if signals.bank_size <= 1:
            return 1.0
        return _clamp((1.0 - signals.cosine_to_centroid) / 2.0)

    @staticmethod
    def redundancy(signals: TemplateSignals) -> float:
        """U_Red: max peer cosine clamped; single-template bank scores 0.0."""
        if signals.bank_size <= 1:
            return 0.0
        return _clamp(signals.max_peer_cosine)

    @staticmethod
    def outlier(signals: TemplateSignals) -> float:
        """U_O: runner-up proximity above 0.70 over a 0.15 band."""
        if signals.runner_up_similarity is None:
            return 0.0
        return _clamp((signals.runner_up_similarity - 0.70) / 0.15)

    @classmethod
    def score(cls, signals: TemplateSignals) -> float:
        """Composite utility, unconditionally in [0.0, 1.0]."""
        base = (
            cls.W_Q * cls.quality(signals)
            + cls.W_S * cls.support(signals)
            + cls.W_R * cls.recency(signals)
            + cls.W_C * cls.coverage(signals)
        )
        return _clamp(
            base - cls.W_RED * cls.redundancy(signals) - cls.W_O * cls.outlier(signals)
        )
