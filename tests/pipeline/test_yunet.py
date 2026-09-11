"""Task 5.5 RED/GREEN: YuNet ONNX adapter — decode, units, NMS, wiring."""

from pathlib import Path

import numpy as np
import pytest

from facecore.conformance.fixtures import make_exif_case
from facecore.pipeline.decode import decode_image
from facecore.pipeline.detect import enforce_single_face
from facecore.pipeline.yunet import YuNetDetector, decode_predictions, nms

MODELS = Path(__file__).resolve().parents[2] / "models"
YUNET_2023MAR = MODELS / "face_detection_yunet_2023mar.onnx"
YUNET_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def _needs_artifact() -> Path:
    pytest.importorskip("onnxruntime")
    if not YUNET_2023MAR.exists():
        pytest.skip("requires downloaded YuNet artifact")
    return YUNET_2023MAR


def test_decode_units_pixels_not_anchor_offsets() -> None:
    """Adapter contract: decode outputs pixel-space box+landmarks, converted
    from anchor-relative offsets (face_detect.cpp:207-223)."""
    stride = 8
    cols, rows = 2, 2
    n = cols * rows
    cls = np.full((n, 1), 0.99, dtype=np.float32)
    obj = np.full((n, 1), 0.99, dtype=np.float32)
    bbox = np.zeros((n, 4), dtype=np.float32)  # zero offset → anchor center
    kps = np.zeros((n, 10), dtype=np.float32)
    faces = decode_predictions(
        {"cls": cls, "obj": obj, "bbox": bbox, "kps": kps},
        stride=stride,
        cols=cols,
        rows=rows,
        score_threshold=0.5,
    )
    assert len(faces) == n
    # Anchor (c=1,r=0): cx=(1+0)*8=8, w=exp(0)*8=8 → x=8-4=4
    f = faces[1]
    assert f.box == pytest.approx((4.0, -4.0, 8.0, 8.0))
    # Landmarks: (0+c)*8, (0+r)*8 → (8, 0) for all five
    assert f.landmarks[0] == pytest.approx((8.0, 0.0))
    assert f.confidence == pytest.approx((0.99 * 0.99) ** 0.5)


def test_nms_keeps_highest_suppresses_overlap() -> None:
    from facecore.pipeline.detect import DetectedFace

    keep = nms(
        [
            DetectedFace(box=(0.0, 0.0, 10.0, 10.0), landmarks=(), confidence=0.9),
            DetectedFace(box=(1.0, 1.0, 10.0, 10.0), landmarks=(), confidence=0.5),
            DetectedFace(box=(100.0, 100.0, 10.0, 10.0), landmarks=(), confidence=0.8),
        ],
        nms_threshold=0.3,
        top_k=10,
    )
    assert [f.confidence for f in keep] == [0.9, 0.8]


def test_real_artifact_blank_image_yields_no_faces() -> None:
    """Real ORT run: blank input scores below threshold → zero faces."""
    path = _needs_artifact()
    detector = YuNetDetector(path, YUNET_SHA)
    decoded = decode_image(make_exif_case(1))
    faces = detector.detect(decoded, score_threshold=0.9)
    assert faces == []
    status, code, _ = enforce_single_face(faces)
    assert (status, code) == ("invalid_input", "input_no_face")


def test_decodable_enroll_builds_identity_or_reason_coded() -> None:
    """Task 5.5 acceptance: decodable image enrolls or returns reason-coded
    invalid_input — never a silent partial (codex acceptance criterion)."""
    from facecore.eval.session import EvaluationSession

    path = _needs_artifact()
    session = EvaluationSession(detector=YuNetDetector(path, YUNET_SHA))
    outcome = session.enroll(make_exif_case(1), "p1")
    assert outcome in ("enrolled", "invalid_input")
    assert session.identity_count() in (0, 1)
    if outcome == "invalid_input":
        assert session.identity_count() == 0
