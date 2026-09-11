"""Task 1 RED/GREEN: PolicyProfile frozen defaults (quality_policy_version 1)."""

import pytest

from facecore.contracts.policy import PolicyProfile


def test_frozen_quality_defaults_match_plan_table() -> None:
    policy = PolicyProfile.frozen_v1()
    assert policy.quality_policy_version == 1
    assert policy.detector_confidence_min == 0.90
    assert policy.face_min_shorter_side_px == 112
    assert policy.sharpness_min == 60.0
    assert policy.exposure_luma_range == (40.0, 215.0)
    assert policy.exposure_clipped_fraction_max == 0.05
    assert policy.yaw_max_deg == 30.0
    assert policy.pitch_max_deg == 20.0
    assert policy.occluded_landmarks_max == 0
    assert policy.occluded_landmark_confidence_min == 0.5
    assert policy.aggregation_strategy == "single_template_passthrough"


def test_thresholds_are_swept_not_chosen() -> None:
    policy = PolicyProfile.frozen_v1()
    assert policy.match_threshold is None
    assert policy.review_threshold is None
    assert policy.margin_threshold is None


def test_invalid_threshold_ordering_rejected() -> None:
    with pytest.raises(ValueError):
        PolicyProfile.frozen_v1().with_thresholds(
            match_threshold=0.5, review_threshold=0.7, margin_threshold=0.0
        )
