"""Decode, EXIF orientation, and color-order normalization (Task 3)."""

import io
from dataclasses import dataclass

from PIL import Image, ImageOps

from facecore.errors import InputDecodeError


@dataclass(frozen=True)
class DecodedImage:
    width: int
    height: int
    color_order: str
    pixels: bytes


def decode_image(data: bytes) -> DecodedImage:
    """Decode bytes to deterministic RGB with EXIF orientation applied."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            oriented = ImageOps.exif_transpose(img)
            rgb = oriented.convert("RGB")
            return DecodedImage(
                width=rgb.width,
                height=rgb.height,
                color_order="RGB",
                pixels=rgb.tobytes(),
            )
    except InputDecodeError:
        raise
    except Exception as exc:
        raise InputDecodeError(f"undecodable image bytes: {exc}") from exc
