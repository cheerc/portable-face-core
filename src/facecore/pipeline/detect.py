"""Single-face detection contract + exactly-one-face enforcement (Task 5)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectedFace:
    box: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...]
    confidence: float


def enforce_single_face(
    faces: list[DetectedFace],
) -> tuple[str, str | None, DetectedFace | None]:
    """Count check before any downstream call; refuse on != 1 (no largest-pick)."""
    if len(faces) == 0:
        return ("invalid_input", "input_no_face", None)
    if len(faces) != 1:
        return ("invalid_input", "input_multiple_faces", None)
    return ("ok", None, faces[0])
