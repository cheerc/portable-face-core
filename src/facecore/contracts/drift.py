"""Phase-1B drift contracts (plan Task 8: bounded response policy).

Reference is lifecycle-safe: when the initial template is evicted and pruned,
detection transitions to rolling-centroid diversity instead of holding a
permanent unmanaged anchor. Breach downgrades ``matched`` to ``review`` with
``drift_boundary_exceeded`` and sets ``re_enrollment_required``.
"""

from dataclasses import dataclass
from enum import Enum

from facecore.contracts.policy import GovernancePolicy


class DriftStatus(str, Enum):
    WITHIN_BOUNDS = "within_bounds"
    BOUNDARY_EXCEEDED = "boundary_exceeded"


class DriftReferenceKind(str, Enum):
    ANCHOR = "anchor"
    ROLLING = "rolling"


@dataclass(frozen=True)
class DriftMetrics:
    identity_id: str
    centroid_shift: float
    reference_kind: str
    observed_at: str
    max_individual_distance: float | None = None

    def __post_init__(self) -> None:
        if self.centroid_shift < 0.0:
            raise ValueError(
                f"centroid_shift must be >= 0, got {self.centroid_shift}"
            )
        if self.reference_kind not in (
            DriftReferenceKind.ANCHOR.value,
            DriftReferenceKind.ROLLING.value,
        ):
            raise ValueError(
                "reference_kind must be 'anchor' or 'rolling', "
                f"got {self.reference_kind!r}"
            )


@dataclass(frozen=True)
class DriftPolicy:
    provisional: bool = True

    def assess(
        self, metrics: DriftMetrics, policy: GovernancePolicy | None = None
    ) -> DriftStatus:
        bound = (
            policy.drift_max_centroid_shift
            if policy is not None
            else GovernancePolicy.provisional_v1().drift_max_centroid_shift
        )
        if metrics.centroid_shift > bound:
            return DriftStatus.BOUNDARY_EXCEEDED
        return DriftStatus.WITHIN_BOUNDS
