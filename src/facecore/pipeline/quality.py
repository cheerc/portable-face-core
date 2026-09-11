"""Quality gate: the quality_policy_version 1 seven-gate table (Task 4)."""

from dataclasses import dataclass

from facecore.contracts.policy import PolicyProfile


@dataclass(frozen=True)
class Landmark:
    confidence: float
    x: float
    y: float


@dataclass(frozen=True)
class QualityVerdict:
    status: str
    reason_codes: list[str]


def evaluate_quality(
    policy: PolicyProfile,
    *,
    detector_confidence: float,
    shorter_side_px: int,
    sharpness: float,
    mean_luma: float,
    clipped_fraction: float,
    yaw_deg: float,
    pitch_deg: float,
    landmarks: list[Landmark],
) -> QualityVerdict:
    """Run all seven gates; multiple failures return every applicable code."""
    codes: list[str] = []
    if detector_confidence < policy.detector_confidence_min:
        codes.append("quality_detector_confidence")
    if shorter_side_px < policy.face_min_shorter_side_px:
        codes.append("quality_face_too_small")
    if sharpness < policy.sharpness_min:
        codes.append("quality_blurry")
    low, high = policy.exposure_luma_range
    if (
        not (low <= mean_luma <= high)
        or clipped_fraction > policy.exposure_clipped_fraction_max
    ):
        codes.append("quality_exposure")
    if abs(yaw_deg) > policy.yaw_max_deg:
        codes.append("quality_pose_yaw")
    if abs(pitch_deg) > policy.pitch_max_deg:
        codes.append("quality_pose_pitch")
    occluded = sum(
        1 for lm in landmarks if lm.confidence < policy.occluded_landmark_confidence_min
    )
    if occluded > policy.occluded_landmarks_max:
        codes.append("quality_occluded")
    if codes:
        return QualityVerdict(status="rejected", reason_codes=codes)
    return QualityVerdict(status="accepted", reason_codes=[])
