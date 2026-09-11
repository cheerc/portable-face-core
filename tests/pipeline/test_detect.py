"""Task 5 RED/GREEN: exactly-one-face enforcement via detector adapter."""

from facecore.pipeline.detect import DetectedFace, enforce_single_face


def _face(confidence: float = 0.95) -> DetectedFace:
    return DetectedFace(
        box=(10.0, 10.0, 100.0, 100.0),
        landmarks=(
            (30.0, 40.0),
            (70.0, 40.0),
            (50.0, 60.0),
            (35.0, 80.0),
            (65.0, 80.0),
        ),
        confidence=confidence,
    )


def test_zero_faces_is_invalid_input_no_face() -> None:
    status, code, _ = enforce_single_face([])
    assert status == "invalid_input"
    assert code == "input_no_face"


def test_two_faces_refused_not_largest_picked() -> None:
    """Failing case from the plan: stub returns two faces, still must refuse."""
    status, code, _ = enforce_single_face([_face(0.99), _face(0.50)])
    assert status == "invalid_input"
    assert code == "input_multiple_faces"


def test_single_face_passes_through() -> None:
    face = _face()
    status, code, out = enforce_single_face([face])
    assert status == "ok"
    assert code is None
    assert out == face
