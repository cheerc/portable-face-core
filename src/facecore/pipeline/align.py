"""Deterministic versioned alignment: box crop, pad, resize to 112x112 (Task 5).

Contract version 2 (PR-1, ADR 0009 Decision 3): detector input uses uniform
scale + centered zero padding, so restored box/landmark geometry feeding
this crop is aspect-preserving. Crop math itself unchanged from version 1.
Contract version 1: nearest-neighbor resize, edge padding, RGB byte output.
"""

from dataclasses import dataclass

from PIL import Image

from facecore.pipeline.detect import DetectedFace

ALIGN_CONTRACT_VERSION = 2
ALIGN_SIZE = 112


@dataclass(frozen=True)
class AlignedCrop:
    width: int
    height: int
    contract_version: int
    pixels: bytes


def align_crop(
    raw_rgb: bytes, width: int, height: int, face: DetectedFace
) -> AlignedCrop:
    """Crop the face box (clamped, edge-padded) and resize deterministically."""
    img = Image.frombytes("RGB", (width, height), raw_rgb)
    x, y, w, h = face.box
    left, top = int(x), int(y)
    right, bottom = int(x + w), int(y + h)
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        import numpy as np

        arr = np.asarray(img)
        arr = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )
        img = Image.fromarray(arr, mode="RGB")
        left += pad_left
        top += pad_top
        right += pad_left
        bottom += pad_top
    crop = img.crop((left, top, right, bottom))
    resized = crop.resize((ALIGN_SIZE, ALIGN_SIZE), Image.Resampling.NEAREST)
    return AlignedCrop(
        width=ALIGN_SIZE,
        height=ALIGN_SIZE,
        contract_version=ALIGN_CONTRACT_VERSION,
        pixels=resized.tobytes(),
    )
