"""PolicyProfile frozen defaults (plan Frozen Phase-1A defaults table).

Provisional and versioned: ``quality_policy_version: 1``. Threshold values are
swept by Task 10, never chosen here — hence ``None`` until swept.

Phase-1B append-only addition below: `GovernancePolicy` (plan §6,
``governance_policy_version: 1``). All operational values are provisional
pending the Operator Decision Manifest; `PolicyProfile` above is untouched.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyProfile:
    quality_policy_version: int
    detector_confidence_min: float
    face_min_shorter_side_px: int
    sharpness_min: float
    exposure_luma_range: tuple[float, float]
    exposure_clipped_fraction_max: float
    yaw_max_deg: float
    pitch_max_deg: float
    occluded_landmarks_max: int
    occluded_landmark_confidence_min: float
    aggregation_strategy: str
    match_threshold: float | None = None
    review_threshold: float | None = None
    margin_threshold: float | None = None

    @classmethod
    def frozen_v1(cls) -> "PolicyProfile":
        return cls(
            quality_policy_version=1,
            detector_confidence_min=0.90,
            face_min_shorter_side_px=112,
            sharpness_min=60.0,
            exposure_luma_range=(40.0, 215.0),
            exposure_clipped_fraction_max=0.05,
            yaw_max_deg=30.0,
            pitch_max_deg=20.0,
            occluded_landmarks_max=0,
            occluded_landmark_confidence_min=0.5,
            aggregation_strategy="single_template_passthrough",
        )

    def with_thresholds(
        self,
        *,
        match_threshold: float,
        review_threshold: float,
        margin_threshold: float,
    ) -> "PolicyProfile":
        if not review_threshold <= match_threshold:
            raise ValueError("review_threshold must not exceed match_threshold")
        return PolicyProfile(
            quality_policy_version=self.quality_policy_version,
            detector_confidence_min=self.detector_confidence_min,
            face_min_shorter_side_px=self.face_min_shorter_side_px,
            sharpness_min=self.sharpness_min,
            exposure_luma_range=self.exposure_luma_range,
            exposure_clipped_fraction_max=self.exposure_clipped_fraction_max,
            yaw_max_deg=self.yaw_max_deg,
            pitch_max_deg=self.pitch_max_deg,
            occluded_landmarks_max=self.occluded_landmarks_max,
            occluded_landmark_confidence_min=self.occluded_landmark_confidence_min,
            aggregation_strategy=self.aggregation_strategy,
            match_threshold=match_threshold,
            review_threshold=review_threshold,
            margin_threshold=margin_threshold,
        )


@dataclass(frozen=True)
class GovernancePolicy:
    """Phase-1B governance defaults (plan §6, version 1, all provisional).

    Ratified package-accepted per Operator Decision Manifest
    ``d-20260912041106780391-34``; the ``provisional`` flag stays ``True``
    until a superseding operator manifest marks these values binding.
    """

    governance_policy_version: int
    provisional: bool
    candidate_update_threshold: float
    additional_corroboration_min_events: int
    burst_suppression_min_interval_secs: float
    promotion_margin: float
    template_bank_capacity: int
    utility_weight_quality: float
    utility_weight_support: float
    utility_weight_recency: float
    utility_weight_coverage: float
    utility_penalty_redundancy: float
    utility_penalty_outlier: float
    drift_max_centroid_shift: float
    drift_max_initial_distance: float
    rollback_max_depth: int
    retired_retention_days: int
    revision_history_max_count: int
    match_events_max_count: int
    backup_max_count: int
    exemplar_margin: float

    @classmethod
    def provisional_v1(cls) -> "GovernancePolicy":
        return cls(
            governance_policy_version=1,
            provisional=True,
            candidate_update_threshold=0.88,
            additional_corroboration_min_events=1,
            burst_suppression_min_interval_secs=60.0,
            promotion_margin=0.12,
            template_bank_capacity=5,
            utility_weight_quality=0.30,
            utility_weight_support=0.30,
            utility_weight_recency=0.20,
            utility_weight_coverage=0.20,
            utility_penalty_redundancy=0.15,
            utility_penalty_outlier=0.15,
            drift_max_centroid_shift=0.20,
            drift_max_initial_distance=0.25,
            rollback_max_depth=5,
            retired_retention_days=90,
            revision_history_max_count=20,
            match_events_max_count=10000,
            backup_max_count=5,
            exemplar_margin=0.0,
        )
