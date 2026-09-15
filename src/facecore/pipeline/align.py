"""Deterministic versioned alignment: landmark similarity warp to 112x112.

Contract version 3 (PR-2, ADR 0009 Decision 2): the aligned crop is fit
from the five detected landmarks onto the canonical template with a
similarity transform (rotation + uniform scale + translation), resampled
BILINEAR (low-pass). Replaces the raw-box crop + anisotropic NEAREST
resize. Restores compliance with the architecture already specified in
the design document (§5: landmark alignment and normalized crop).

Canonical template provenance (never from memory):
OpenCV opencv/opencv@4.x,
modules/objdetect/src/face_recognize.cpp::getSimilarityTransformMatrix,
dst[5][2] used by FaceRecognizerSF::alignCrop (warpAffine INTER_LINEAR).
Contract version 2 (PR-1): detector input uniform scale + centered pad.
Contract version 1: nearest-neighbor resize, edge padding, RGB byte output.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from facecore.pipeline.detect import DetectedFace

ALIGN_CONTRACT_VERSION = 3
ALIGN_SIZE = 112

CANONICAL_TEMPLATE_5: tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
] = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


def similarity_transform(
    src: tuple[tuple[float, float], ...],
    dst: tuple[tuple[float, float], ...] = CANONICAL_TEMPLATE_5,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Fit similarity (rotation + uniform scale + translation), Umeyama.

    Mirrors OpenCV getSimilarityTransformMatrix (SVD, reflection-safe):
    demean both sets, A = dst_demean^T src_demean / 5, R = U D V^T,
    scale from singular values over source variance, t = dst_mean - s R
    src_mean. Returns the 2x3 row-major matrix mapping src -> dst.
    """
    s = np.asarray(src, dtype=np.float64)
    d = np.asarray(dst, dtype=np.float64)
    src_mean = s.mean(axis=0)
    dst_mean = d.mean(axis=0)
    s0 = s - src_mean
    d0 = d - dst_mean
    mat_a = d0.T @ s0 / 5.0
    u, vals, vt = np.linalg.svd(mat_a)
    det_a = float(np.linalg.det(mat_a))
    dd = np.array([1.0, -1.0 if det_a < 0 else 1.0])
    rank = int(np.sum(vals > vals.max() * 2 * np.finfo(float).eps))
    if rank == 1:
        if float(np.linalg.det(u) * np.linalg.det(vt)) > 0:
            rot = u @ vt
        else:
            rot = u @ np.diag([dd[0], -1.0]) @ vt
    else:
        rot = u @ np.diag(dd) @ vt
    var = float((s0[:, 0] ** 2).mean() + (s0[:, 1] ** 2).mean())
    scale = float((vals[0] * dd[0] + vals[1] * dd[1]) / var)
    rot_s = rot * scale
    trans = dst_mean - rot_s @ src_mean
    return (
        (float(rot_s[0, 0]), float(rot_s[0, 1]), float(trans[0])),
        (float(rot_s[1, 0]), float(rot_s[1, 1]), float(trans[1])),
    )


@dataclass(frozen=True)
class AlignedCrop:
    width: int
    height: int
    contract_version: int
    pixels: bytes


def align_crop(
    raw_rgb: bytes, width: int, height: int, face: DetectedFace
) -> AlignedCrop:
    """Warp the face onto the canonical template via landmark similarity."""
    if len(face.landmarks) != 5:
        raise ValueError(
            f"align_crop needs exactly 5 landmarks, got {len(face.landmarks)}"
        )
    img = Image.frombytes("RGB", (width, height), raw_rgb)
    mat = similarity_transform(face.landmarks)
    # Edge-pad the source so out-of-frame warp samples read edge pixels
    # (same convention as the v1 box crop), never PIL's default fill 0 —
    # fill 0 would inject black corners and trip the exposure clip gate.
    pad = max(1, round(max(width, height) * 0.25))
    padded = np.pad(
        np.asarray(img), ((pad, pad), (pad, pad), (0, 0)), mode="edge"
    )
    img = Image.fromarray(padded, mode="RGB")
    # Refit the matrix in padded coordinates: the same physical pixel sits
    # at x+pad, so T = R(x_pad - pad) + t and translation loses R·pad.
    a, b, c = mat[0]
    d, e, f = mat[1]
    c2 = c - a * pad - b * pad
    f2 = f - d * pad - e * pad
    # PIL AFFINE maps output -> input; invert the forward 2x3 matrix.
    det = a * e - b * d
    if det == 0.0:
        raise ValueError("degenerate landmark constellation (zero area)")
    inv = (
        e / det,
        -b / det,
        (b * f2 - e * c2) / det,
        -d / det,
        a / det,
        (d * c2 - a * f2) / det,
    )
    warped = img.transform(
        (ALIGN_SIZE, ALIGN_SIZE), Image.Transform.AFFINE, inv,
        resample=Image.Resampling.BILINEAR,
    )
    return AlignedCrop(
        width=ALIGN_SIZE,
        height=ALIGN_SIZE,
        contract_version=ALIGN_CONTRACT_VERSION,
        pixels=warped.tobytes(),
    )
