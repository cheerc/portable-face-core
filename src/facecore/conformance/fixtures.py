"""Deterministic face-free fixture generator (Task 3 + Task 12 Layer-A set).

Generates in-memory PNG bytes and synthetic arrays only — callers write
them to the gitignored build directory, never into the repository.
Covers the full twelve-case Layer-A set (spec section 6, plan Task 12):
decoding, orientation, color order, resize, crop, padding, interpolation,
tensor layout, scaling, normalization, embedding normalization, numeric encoding.
"""

import hashlib
import io
import struct

import numpy as np
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


# =========================================================================
# Layer-A 12 Deterministic Conformance Cases (Task 12)
# =========================================================================

CASE_NAMES: tuple[str, ...] = (
    "decoding",
    "orientation",
    "color_order",
    "resize",
    "crop",
    "padding",
    "interpolation",
    "tensor_layout",
    "scaling",
    "normalization",
    "embedding_normalization",
    "numeric_encoding",
)


def generate_decoding_case() -> dict[str, object]:
    """Case 1: decoding — decode in-memory PNG bytes to deterministic RGB array."""
    from facecore.pipeline.decode import decode_image

    png_bytes = make_exif_case(1)
    decoded = decode_image(png_bytes)
    return {
        "case": "decoding",
        "width": decoded.width,
        "height": decoded.height,
        "color_order": decoded.color_order,
        "pixel_count": decoded.width * decoded.height,
        "sha256": hashlib.sha256(decoded.pixels).hexdigest(),
        "bytes": list(decoded.pixels),
    }


def generate_orientation_case() -> dict[str, object]:
    """Case 2: orientation — EXIF orientations 1..8 all decode to upright array."""
    from facecore.pipeline.decode import decode_image

    expected = upright_expected()
    matched = [decode_image(make_exif_case(i)).pixels == expected for i in range(1, 9)]
    return {
        "case": "orientation",
        "orientations": list(range(1, 9)),
        "all_upright": all(matched),
        "upright_sha256": hashlib.sha256(expected).hexdigest(),
    }


def generate_color_order_case() -> dict[str, object]:
    """Case 3: color order — verify explicit RGB channel ordering on 2x2 pattern."""
    from facecore.pipeline.decode import decode_image

    img = Image.new("RGB", (2, 2))
    img.putpixel((0, 0), (255, 0, 0))
    img.putpixel((1, 0), (0, 255, 0))
    img.putpixel((0, 1), (0, 0, 255))
    img.putpixel((1, 1), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    decoded = decode_image(buf.getvalue())
    raw = list(decoded.pixels)
    pixels = [raw[i : i + 3] for i in range(0, len(raw), 3)]
    return {
        "case": "color_order",
        "channels": "RGB",
        "width": 2,
        "height": 2,
        "pixels": pixels,
    }


def generate_resize_case() -> dict[str, object]:
    """Case 4: resize — deterministic nearest-neighbor 2x2 -> 4x4 upsampling."""
    img = Image.fromarray(np.array([[10, 20], [30, 40]], dtype=np.uint8), mode="L")
    resized = img.resize((4, 4), Image.Resampling.NEAREST)
    return {
        "case": "resize",
        "method": "NEAREST",
        "input_size": [2, 2],
        "output_size": [4, 4],
        "matrix": np.asarray(resized).tolist(),
    }


def generate_crop_case() -> dict[str, object]:
    """Case 5: crop — deterministic slice of 4x4 array."""
    arr = np.arange(16, dtype=np.uint8).reshape((4, 4))
    cropped = arr[1:3, 1:3]
    return {
        "case": "crop",
        "input_size": [4, 4],
        "box": [1, 1, 2, 2],
        "cropped": cropped.tolist(),
    }


def generate_padding_case() -> dict[str, object]:
    """Case 6: padding — edge padding mirroring align_crop."""
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    padded = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
    return {
        "case": "padding",
        "mode": "edge",
        "pad_width": [[1, 1], [1, 1]],
        "output": padded.tolist(),
    }


def generate_interpolation_case(*, mode: str = "BILINEAR") -> dict[str, object]:
    """Case 7: interpolation — bilinear 2x2 -> 4x4 interpolation."""
    resample = getattr(Image.Resampling, mode)
    im = Image.fromarray(np.array([[0, 100], [100, 200]], dtype=np.uint8))
    resized = im.resize((4, 4), resample)
    return {
        "case": "interpolation",
        "method": mode,
        "input_size": [2, 2],
        "output_size": [4, 4],
        "output": np.asarray(resized).tolist(),
    }


def generate_tensor_layout_case() -> dict[str, object]:
    """Case 8: tensor layout — HWC -> NCHW transpose."""
    arr = np.arange(18, dtype=np.uint8).reshape((2, 3, 3))
    t = np.transpose(arr, (2, 0, 1))[None]
    return {
        "case": "tensor_layout",
        "source_layout": "HWC",
        "target_layout": "NCHW",
        "input_shape": list(arr.shape),
        "output_shape": list(t.shape),
        "data": t.tolist(),
    }


def generate_scaling_case() -> dict[str, object]:
    """Case 9: scaling — uint8 to float32 raw and unit-scale conversions."""
    arr = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    f = arr.astype(np.float32)
    f_norm = f / 255.0
    return {
        "case": "scaling",
        "input_uint8": arr.tolist(),
        "raw_float32": [float(x) for x in f.tolist()],
        "unit_scale_float32": [float(x) for x in f_norm.tolist()],
    }


def generate_normalization_case() -> dict[str, object]:
    """Case 10: normalization — zero-mean and standardized (x-127.5)/128."""
    vals = [0.0, 127.5, 255.0]
    mean = 127.5
    std = 128.0
    zero_mean = [x - mean for x in vals]
    standardized = [(x - mean) / std for x in vals]
    return {
        "case": "normalization",
        "input": vals,
        "mean": mean,
        "std": std,
        "zero_mean": zero_mean,
        "standardized": standardized,
    }


def generate_embedding_normalization_case() -> dict[str, object]:
    """Case 11: embedding normalization — L2 norm and normalized vector."""
    vec = [3.0, 4.0, 0.0, 0.0]
    norm = float(np.linalg.norm(vec))
    normalized = [x / norm for x in vec]
    return {
        "case": "embedding_normalization",
        "input": vec,
        "input_norm": norm,
        "normalized": normalized,
        "output_norm": float(np.linalg.norm(normalized)),
    }


def generate_numeric_encoding_case() -> dict[str, object]:
    """Case 12: numeric encoding — IEEE-754 binary32 little-endian hex."""
    vals = [0.0, 1.0, -1.0, 0.5, -0.5, 0.125, 0.363, 1.128]
    hex_le = [struct.pack("<f", v).hex() for v in vals]
    return {
        "case": "numeric_encoding",
        "format": "IEEE-754-binary32",
        "byte_order": "little-endian",
        "values": vals,
        "hex_le": hex_le,
    }


def generate_case(case_name: str, **kwargs: object) -> dict[str, object]:
    """Dispatch to the named case generator."""
    generators = {
        "decoding": generate_decoding_case,
        "orientation": generate_orientation_case,
        "color_order": generate_color_order_case,
        "resize": generate_resize_case,
        "crop": generate_crop_case,
        "padding": generate_padding_case,
        "interpolation": generate_interpolation_case,
        "tensor_layout": generate_tensor_layout_case,
        "scaling": generate_scaling_case,
        "normalization": generate_normalization_case,
        "embedding_normalization": generate_embedding_normalization_case,
        "numeric_encoding": generate_numeric_encoding_case,
    }
    if case_name not in generators:
        raise ValueError(
            f"unknown case name {case_name!r}; expected one of {CASE_NAMES}"
        )
    if kwargs:
        if case_name == "interpolation":
            mode = str(kwargs.get("mode", "BILINEAR"))
            return generate_interpolation_case(mode=mode)
        raise ValueError(f"case {case_name!r} does not accept keyword arguments")
    return generators[case_name]()
