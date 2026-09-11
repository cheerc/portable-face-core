"""PolicyProfile frozen defaults (plan Frozen Phase-1A defaults table).

Provisional and versioned: ``quality_policy_version: 1``. Threshold values are
swept by Task 10, never chosen here — hence ``None`` until swept.
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
