"""PR-2 RED/GREEN: landmark similarity alignment (Fix-1 PR-2, ADR 0009).

Contract: align_crop fits a SIMILARITY transform (rotation + uniform
scale + translation) from the five detected landmarks onto the canonical
112x112 template, replacing the raw-box crop + anisotropic resize.

Template provenance (recorded, never from memory):
OpenCV opencv/opencv@4.x,
modules/objdetect/src/face_recognize.cpp::getSimilarityTransformMatrix,
dst[5][2] = {(38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366),
(41.5493, 92.3655), (70.7299, 92.2041)}; warpAffine(..., INTER_LINEAR).
The resample here is BILINEAR (low-pass), replacing NEAREST (aliased,
phase-dependent — spike (ii) secondary cause).

RED requirement: on pre-fix code (box crop, no similarity path) the
landmark-position assertions fail; evidence in PR.
"""

import math

import numpy as np
import pytest

from facecore.pipeline.align import (
    ALIGN_SIZE,
    CANONICAL_TEMPLATE_5,
    align_crop,
    similarity_transform,
)
from facecore.pipeline.detect import DetectedFace


def test_template_provenance_values() -> None:
    assert CANONICAL_TEMPLATE_5 == (
        (38.2946, 51.6963),
        (73.5318, 51.5014),
        (56.0252, 71.7366),
        (41.5493, 92.3655),
        (70.7299, 92.2041),
    )


def test_similarity_identity_on_template_itself() -> None:
    mat = similarity_transform(CANONICAL_TEMPLATE_5, CANONICAL_TEMPLATE_5)
    for x, y in CANONICAL_TEMPLATE_5:
        assert mat[0][0] * x + mat[0][1] * y + mat[0][2] == pytest.approx(
            x, abs=1e-6
        )
        assert mat[1][0] * x + mat[1][1] * y + mat[1][2] == pytest.approx(
            y, abs=1e-6
        )


def test_similarity_recovers_known_transform() -> None:
    """Template through rotation+scale+shift must round-trip < 1px."""
    angle, scale, tx, ty = math.radians(10.0), 1.5, 37.0, -23.0
    ca, sa = math.cos(angle), math.sin(angle)
    src = tuple(
        (scale * (ca * x - sa * y) + tx, scale * (sa * x + ca * y) + ty)
        for x, y in CANONICAL_TEMPLATE_5
    )
    mat = similarity_transform(src, CANONICAL_TEMPLATE_5)
    for (sx, sy), (dx, dy) in zip(src, CANONICAL_TEMPLATE_5):
        assert mat[0][0] * sx + mat[0][1] * sy + mat[0][2] == pytest.approx(
            dx, abs=1.0
        )
        assert mat[1][0] * sx + mat[1][1] * sy + mat[1][2] == pytest.approx(
            dy, abs=1.0
        )


def _synthetic_face(size: int = 400) -> tuple[np.ndarray, tuple]:
    """Gray canvas, white disc, five dark landmark dots (face-free)."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size / 2.0
    r = size * 0.30
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r**2] = 240
    landmarks = (
        (cx - 0.35 * r, cy - 0.20 * r),
        (cx + 0.35 * r, cy - 0.20 * r),
        (cx, cy + 0.10 * r),
        (cx - 0.30 * r, cy + 0.50 * r),
        (cx + 0.30 * r, cy + 0.50 * r),
    )
    for lx, ly in landmarks:
        img[(xx - lx) ** 2 + (yy - ly) ** 2 <= (r * 0.07) ** 2] = 20
    return img, landmarks


def test_aligned_crop_centers_landmarks_on_template() -> None:
    """End to end: the five dots must land on the template within 8px."""
    img, landmarks = _synthetic_face()
    h, w, _ = img.shape
    face = DetectedFace(
        box=(w * 0.2, h * 0.2, w * 0.6, h * 0.6),
        landmarks=landmarks,
        confidence=0.99,
    )
    out = align_crop(img.tobytes(), w, h, face)
    assert (out.width, out.height) == (ALIGN_SIZE, ALIGN_SIZE)
    gray = (
        np.frombuffer(out.pixels, dtype=np.uint8).reshape(112, 112, 3).mean(axis=2)
    )
    for tx, ty in CANONICAL_TEMPLATE_5:
        ix, iy = int(round(tx)), int(round(ty))
        patch = gray[max(0, iy - 8) : iy + 9, max(0, ix - 8) : ix + 9]
        assert patch.min() < 100, f"template point {(tx, ty)} has no dot nearby"
