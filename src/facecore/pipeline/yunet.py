"""YuNet ONNX detector adapter (Task 5.5): DecodedImage -> DetectedFace.

Postprocessing mirrors OpenCV face_detect.cpp:174-245 (one-hand source):
score = sqrt(clamp(cls) * clamp(obj)), thresholded at decode time;
box cx=(c+dx)*stride, cy=(r+dy)*stride, w=exp(dw)*stride, h=exp(dh)*stride,
x1=cx-w/2, y1=cy-h/2; landmarks (kps+c)*stride, (kps+r)*stride.
Units after decode are input-image pixels; the adapter scales back to the
source DecodedImage pixels. NMS is score-ordered IoU suppression + top_k.

Input: RGB DecodedImage -> BGR numpy -> pad to /32 -> NCHW float32 raw
[0,255] (blobFromImage defaults, face_detect.cpp:136-148).
2023mar takes fixed 640x640 (measured: other shapes rejected); the adapter
letterboxes with ONE uniform scale plus CENTERED zero padding (ADR 0009
Decision 1: aspect preserved end to end), then restores boxes/landmarks
through that same scale and padding offset. 2026may symbolic dims ride
the same path when fed its native shape.
"""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from facecore.errors import ModelIntegrityError
from facecore.pipeline.decode import DecodedImage
from facecore.pipeline.detect import DetectedFace

STRIDES = (8, 16, 32)
FIXED_INPUT = 640


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_predictions(
    outputs: dict[str, np.ndarray],
    *,
    stride: int,
    cols: int,
    rows: int,
    score_threshold: float,
) -> list[DetectedFace]:
    """Decode one stride level from anchor-relative offsets to pixel boxes."""
    cls = outputs["cls"].reshape(-1)
    obj = outputs["obj"].reshape(-1)
    bbox = outputs["bbox"].reshape(-1, 4)
    kps = outputs["kps"].reshape(-1, 10)
    faces: list[DetectedFace] = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            cls_score = min(max(float(cls[idx]), 0.0), 1.0)
            obj_score = min(max(float(obj[idx]), 0.0), 1.0)
            score = (cls_score * obj_score) ** 0.5
            if score < score_threshold:
                continue
            cx = (c + float(bbox[idx, 0])) * stride
            cy = (r + float(bbox[idx, 1])) * stride
            w = float(np.exp(bbox[idx, 2])) * stride
            h = float(np.exp(bbox[idx, 3])) * stride
            landmarks = tuple(
                (
                    (float(kps[idx, 2 * n]) + c) * stride,
                    (float(kps[idx, 2 * n + 1]) + r) * stride,
                )
                for n in range(5)
            )
            faces.append(
                DetectedFace(
                    box=(cx - w / 2.0, cy - h / 2.0, w, h),
                    landmarks=landmarks,
                    confidence=score,
                )
            )
    return faces


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms(
    faces: list[DetectedFace], *, nms_threshold: float, top_k: int
) -> list[DetectedFace]:
    """Score-ordered IoU suppression, keep top_k (mirrors NMSBoxes usage)."""
    ordered = sorted(faces, key=lambda f: f.confidence, reverse=True)
    kept: list[DetectedFace] = []
    for face in ordered:
        if all(_iou(face.box, other.box) <= nms_threshold for other in kept):
            kept.append(face)
        if len(kept) >= top_k:
            break
    return kept


def _letterbox_to(
    bgr: np.ndarray, size: int
) -> tuple[np.ndarray, float, float, float]:
    """Uniform-scale + CENTERED zero-pad onto size x size.

    Returns (canvas, scale, pad_x, pad_y) where a 640-space coordinate
    restores as (x - pad_x) / scale. Centered + 0 locked by
    test_letterbox_invariance (plan open-question 2).
    """
    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    small = np.asarray(
        Image.fromarray(bgr).resize((nw, nh), Image.Resampling.BILINEAR)
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x = (size - nw) / 2.0
    pad_y = (size - nh) / 2.0
    y0, x0 = int(round(pad_y)), int(round(pad_x))
    canvas[y0 : y0 + nh, x0 : x0 + nw] = small
    return canvas, scale, pad_x, pad_y


def _pad_to(image: np.ndarray, divisor: int = 32) -> np.ndarray:
    h, w = image.shape[:2]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if not pad_h and not pad_w:
        return image
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


class YuNetDetector:
    def __init__(
        self,
        artifact_path: Path,
        expected_sha256: str,
        *,
        input_size: int | None = FIXED_INPUT,
    ) -> None:
        actual = _sha256_of(artifact_path)
        if actual != expected_sha256:
            raise ModelIntegrityError(
                "detector artifact hash mismatch: "
                f"disk {actual} != manifest {expected_sha256!r}"
            )
        import onnxruntime as ort  # type: ignore[import-untyped]

        self._session = ort.InferenceSession(
            str(artifact_path), providers=["CPUExecutionProvider"]
        )
        self._input_size = input_size

    def detect(
        self,
        decoded: DecodedImage,
        *,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> list[DetectedFace]:
        rgb = np.frombuffer(decoded.pixels, dtype=np.uint8).reshape(
            decoded.height, decoded.width, 3
        )
        bgr = rgb[:, :, ::-1]
        if self._input_size is not None:
            letterboxed, scale, pad_x, pad_y = _letterbox_to(bgr, self._input_size)
        else:
            letterboxed, scale, pad_x, pad_y = bgr, 1.0, 0.0, 0.0
        padded = _pad_to(letterboxed)
        tensor = np.transpose(padded, (2, 0, 1))[None].astype(np.float32)
        raw = self._session.run(None, {"input": tensor})
        outputs = dict(
            zip([o.name for o in self._session.get_outputs()], raw, strict=True)
        )
        faces: list[DetectedFace] = []
        pad_h, pad_w = padded.shape[:2]
        for stride in STRIDES:
            # Grid mirrors OpenCV: cols=padW/stride, rows=padH/stride.
            cols, rows = pad_w // stride, pad_h // stride
            faces.extend(
                decode_predictions(
                    {
                        "cls": outputs[f"cls_{stride}"],
                        "obj": outputs[f"obj_{stride}"],
                        "bbox": outputs[f"bbox_{stride}"],
                        "kps": outputs[f"kps_{stride}"],
                    },
                    stride=stride,
                    cols=cols,
                    rows=rows,
                    score_threshold=score_threshold,
                )
            )
        kept = nms(faces, nms_threshold=nms_threshold, top_k=top_k)
        return [
            DetectedFace(
                box=(
                    (f.box[0] - pad_x) / scale,
                    (f.box[1] - pad_y) / scale,
                    f.box[2] / scale,
                    f.box[3] / scale,
                ),
                landmarks=tuple(
                    ((x - pad_x) / scale, (y - pad_y) / scale)
                    for x, y in f.landmarks
                ),
                confidence=f.confidence,
            )
            for f in kept
        ]
