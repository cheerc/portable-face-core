"""Task 4 RED/GREEN: quality v1 — seven gates, all-codes, accepted."""

import pytest

from facecore.contracts.policy import PolicyProfile
from facecore.pipeline.quality import Landmark, evaluate_quality


def _good_kwargs() -> dict:
    return {
        "detector_confidence": 0.99,
        "shorter_side_px": 200,
        "sharpness": 120.0,
        "mean_luma": 128.0,
        "clipped_fraction": 0.0,
        "yaw_deg": 5.0,
        "pitch_deg": 3.0,
        "landmarks": [Landmark(0.9, 10.0, 10.0) for _ in range(5)],
    }


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        ({"detector_confidence": 0.5}, "quality_detector_confidence"),
        ({"shorter_side_px": 60}, "quality_face_too_small"),
        ({"sharpness": 10.0}, "quality_blurry"),
        ({"mean_luma": 10.0}, "quality_exposure"),
        ({"clipped_fraction": 0.5}, "quality_exposure"),
        ({"yaw_deg": 45.0}, "quality_pose_yaw"),
        ({"pitch_deg": 30.0}, "quality_pose_pitch"),
        (
            {"landmarks": [Landmark(0.1, 0.0, 0.0)] + [Landmark(0.9, 1.0, 1.0)] * 4},
            "quality_occluded",
        ),
    ],
)
def test_each_gate_fires_its_own_reason_code(mutate: dict, code: str) -> None:
    """Failing case from the plan, parameterized: blank must carry the code."""
    kwargs = {**_good_kwargs(), **mutate}
    verdict = evaluate_quality(PolicyProfile.frozen_v1(), **kwargs)
    assert verdict.reason_codes == [code], (
        f"expected [{code}], got {verdict.reason_codes}"
    )


def test_multiple_failures_return_all_codes_not_first_only() -> None:
    kwargs = {**_good_kwargs(), "sharpness": 10.0, "yaw_deg": 45.0}
    verdict = evaluate_quality(PolicyProfile.frozen_v1(), **kwargs)
    assert verdict.reason_codes == ["quality_blurry", "quality_pose_yaw"]


def test_passing_case_returns_accepted_with_empty_codes() -> None:
    verdict = evaluate_quality(PolicyProfile.frozen_v1(), **_good_kwargs())
    assert verdict.status == "accepted"
    assert verdict.reason_codes == []
