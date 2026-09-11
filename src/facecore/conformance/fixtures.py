"""Deterministic face-free fixture generator (first consumer: Task 3 EXIF cases).

Generates in-memory JPEG bytes only — callers write them to the gitignored
build directory, never into the repository. Task 12 reuses this generator
for the full Layer-A set.
"""

import io

from PIL import Image

# Upright 4x3 face-free pattern: R channel ramps with x, G ramps with y,
# B is a checkerboard — asymmetric under every transpose so each EXIF
# orientation is distinguishable before exif_transpose.
_W, _H = 4, 3


def _upright_image() -> Image.Image:
    img = Image.new("RGB", (_W, _H))
    px = img.load()
    assert px is not None
    for y in range(_H):
        for x in range(_W):
            px[x, y] = ((x * 64) % 256, (y * 80) % 256, 255 if (x + y) % 2 else 0)
    return img


def upright_expected() -> bytes:
    """Committed expected value: raw RGB bytes of the upright pattern."""
    return _upright_image().tobytes()


# Physical storage for EXIF orientation `n`: the INVERSE of the transform
# exif_transpose applies for `n`, so transposing restores the upright array.
# Self-inverse ops (2,3,4,5,7) store the same transpose; 6 <-> 8 are swapped:
# exif_transpose does ROTATE_270 for 6 and ROTATE_90 for 8.
def _stored_for_orientation(orientation: int) -> Image.Image:
    upright = _upright_image()
    transpose = Image.Transpose
    return {
        1: upright,
        2: upright.transpose(transpose.FLIP_LEFT_RIGHT),
        3: upright.transpose(transpose.ROTATE_180),
        4: upright.transpose(transpose.FLIP_TOP_BOTTOM),
        5: upright.transpose(transpose.TRANSPOSE),
        6: upright.transpose(transpose.ROTATE_90),
        7: upright.transpose(transpose.TRANSVERSE),
        8: upright.transpose(transpose.ROTATE_270),
    }[orientation]


def make_exif_case(orientation: int) -> bytes:
    """Lossless PNG bytes tagged with EXIF orientation 1..8 (face-free).

    PNG (not JPEG): even quality-100 JPEG is lossy and cannot round-trip
    the high-contrast pattern byte-exactly. The plan requires the eight
    orientation cases, not a specific container.
    """
    if orientation not in range(1, 9):
        raise ValueError(f"EXIF orientation must be 1..8, got {orientation}")
    stored = _stored_for_orientation(orientation)
    exif = stored.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    stored.save(buf, format="PNG", exif=exif)
    return buf.getvalue()
