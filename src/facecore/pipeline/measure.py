"""Measured quality inputs from an aligned crop + detected face (Task 9).

Closes 5.5 N1: the enroll path now feeds evaluate_quality with measured
values under the frozen v1 policy instead of executing the count gate only.

- sharpness: variance of Laplacian over the aligned crop luma.
- exposure: mean luma + clipped-pixel fraction (<=2 or >=253).
- yaw/pitch: estimated from the five landmarks (eye-line roll for yaw
  proxy via eye-nose asymmetry; eye-mouth verticality for pitch proxy).
  Coarse by design — the bake-off may prove the frozen defaults wrong,
  which is the point of running it.
- landmark confidence: YuNet emits no per-landmark confidence, so all five
  are recorded at 1.0 (never occluded by this path); stated, not hidden.
"""

import numpy as np

from facecore.pipeline.align import AlignedCrop
from facecore.pipeline.detect import DetectedFace
from facecore.pipeline.quality import Landmark


def _luma(crop: AlignedCrop) -> np.ndarray:
    arr = np.frombuffer(crop.pixels, dtype=np.uint8).reshape(crop.height, crop.width, 3)
    mix: np.ndarray = (
        0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    ).astype(np.float64)
    return mix


def sharpness_of(crop: AlignedCrop) -> float:
    luma = _luma(crop)
    lap = (
        4 * luma[1:-1, 1:-1]
        - luma[:-2, 1:-1]
        - luma[2:, 1:-1]
        - luma[1:-1, :-2]
        - luma[1:-1, 2:]
    )
    return float(np.var(lap))


def exposure_of(crop: AlignedCrop) -> tuple[float, float]:
    luma = _luma(crop)
    clipped = float(np.mean((luma <= 2) | (luma >= 253)))
    return float(np.mean(luma)), clipped


def pose_of(face: DetectedFace) -> tuple[float, float]:
    """Estimate yaw/pitch degrees from five landmarks (coarse proxy)."""
    (re_x, re_y), (le_x, le_y), (n_x, n_y), (rm_x, rm_y), (lm_x, lm_y) = face.landmarks
    eye_dx = abs(le_x - re_x)
    if eye_dx < 1e-6:
        return 90.0, 90.0
    # Yaw proxy: nose deviation from the eye midpoint, scaled to degrees.
    yaw = abs(n_x - (re_x + le_x) / 2.0) / eye_dx * 90.0
    # Pitch proxy: nose deviation from the eye-mouth midline.
    eye_y = (re_y + le_y) / 2.0
    mouth_y = (rm_y + lm_y) / 2.0
    vertical = abs(mouth_y - eye_y)
    if vertical < 1e-6:
        return min(yaw, 90.0), 90.0
    pitch = abs(n_y - (eye_y + mouth_y) / 2.0) / vertical * 90.0
    return min(yaw, 90.0), min(pitch, 90.0)


def landmarks_of(face: DetectedFace) -> list[Landmark]:
    return [Landmark(confidence=1.0, x=x, y=y) for x, y in face.landmarks]
