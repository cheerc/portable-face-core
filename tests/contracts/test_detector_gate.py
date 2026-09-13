"""§6-1 RED/GREEN: detector gate configurability.

Source of truth: acceptance report §3.2 + spec §7 + PolicyProfile @ bd2706d.
RED: no `with_detector_gate` entry point; CLI has no `--detector-gate`.
"""

import pytest

from facecore.contracts.policy import PolicyProfile


def test_frozen_default_gate_unchanged() -> None:
    assert PolicyProfile.frozen_v1().detector_confidence_min == 0.90


def test_with_detector_gate_configures() -> None:
    policy = PolicyProfile.frozen_v1().with_detector_gate(
        detector_confidence_min=0.8
    )
    assert policy.detector_confidence_min == 0.8
    # Frozen body untouched; thresholds preserved.
    assert PolicyProfile.frozen_v1().detector_confidence_min == 0.90
    assert policy.quality_policy_version == 1


def test_invalid_gate_rejected_structured() -> None:
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            PolicyProfile.frozen_v1().with_detector_gate(
                detector_confidence_min=bad
            )
