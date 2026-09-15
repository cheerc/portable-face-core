"""Task 5 RED/GREEN: deterministic versioned alignment."""

from facecore.pipeline.align import ALIGN_CONTRACT_VERSION, align_crop
from facecore.pipeline.detect import DetectedFace


def test_same_input_yields_byte_identical_crop() -> None:
    pixels = bytes((i * 7) % 256 for i in range(64 * 64 * 3))
    face = DetectedFace(
        box=(8.0, 8.0, 48.0, 48.0),
        landmarks=(
            (20.0, 24.0),
            (44.0, 24.0),
            (32.0, 36.0),
            (24.0, 48.0),
            (40.0, 48.0),
        ),
        confidence=0.95,
    )
    first = align_crop(pixels, 64, 64, face)
    second = align_crop(pixels, 64, 64, face)
    assert first.pixels == second.pixels
    assert first.contract_version == ALIGN_CONTRACT_VERSION == 2
    assert (first.width, first.height) == (112, 112)
