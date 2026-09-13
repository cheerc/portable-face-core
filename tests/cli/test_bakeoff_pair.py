"""M2 RED/GREEN: bakeoff --pair selection (pair1 frozen default, pair2 new).

RED: _resolve_pair_artifacts does not exist; --pair flag unknown.
Pair 1 selection byte-identical to the frozen wiring (2023mar fixed-640 +
fp32 manifest). Pair 2 selects 2026may (native dynamic shape) + int8bq.
"""

import pytest

from facecore.cli import _resolve_pair_artifacts
from facecore.contracts.manifest import (
    YUNET_2026MAY_SHA,
    ModelManifest,
)


def test_pair1_is_frozen_default_wiring() -> None:
    resolved = _resolve_pair_artifacts("pair1")
    assert resolved.detector_filename == "face_detection_yunet_2023mar.onnx"
    assert resolved.detector_input_size == 640
    assert (
        resolved.manifest_factory.__func__ is ModelManifest.sface_2021dec_fp32.__func__
    )
    assert resolved.embedder_filename == "face_recognition_sface_2021dec.onnx"


def test_pair2_selects_2026may_and_int8bq() -> None:
    resolved = _resolve_pair_artifacts("pair2")
    assert resolved.detector_filename == "face_detection_yunet_2026may.onnx"
    assert resolved.detector_sha256 == YUNET_2026MAY_SHA
    assert resolved.detector_input_size is None  # native dynamic shape
    assert (
        resolved.manifest_factory.__func__
        is ModelManifest.sface_2021dec_int8bq.__func__
    )
    assert (
        resolved.embedder_filename
        == "face_recognition_sface_2021dec_int8bq.onnx"
    )


def test_unknown_pair_rejected() -> None:
    with pytest.raises(ValueError, match="unknown pair"):
        _resolve_pair_artifacts("pair3")
