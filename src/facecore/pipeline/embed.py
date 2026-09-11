"""ONNX embedding with manifest integrity gate (Task 6).

Preprocessing contract (pinned Phase 1 against the downloaded artifact +
OpenCV face_recognize.cpp:58): the aligned crop is RGB, raw [0,255], zero
mean; the net input is NCHW float32. L2 normalization is embedder duty
(OpenCV normalizes in match(); raw fc1 norms are != 1.0, measured).
"""

import hashlib
from pathlib import Path

import numpy as np

from facecore.contracts.manifest import ModelManifest
from facecore.errors import ModelIntegrityError
from facecore.pipeline.align import AlignedCrop


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Embedder:
    def __init__(self, manifest: ModelManifest, artifact_path: Path) -> None:
        actual = _sha256_of(artifact_path)
        # verify_sha256 False (incl. recorded-None) becomes ModelIntegrityError
        # (exit 3) here — never a warning, never continued (codex PR-C note 1).
        if not manifest.verify_sha256(actual):
            raise ModelIntegrityError(
                f"artifact hash mismatch for {manifest.model_id}: "
                f"disk sha256 {actual} does not match manifest "
                f"{manifest.weight_sha256!r}"
            )
        import onnxruntime as ort  # type: ignore[import-untyped]

        self._manifest = manifest
        self._session = ort.InferenceSession(
            str(artifact_path), providers=["CPUExecutionProvider"]
        )

    @property
    def model_version(self) -> str:
        return self._manifest.model_id

    def embed(self, crop: AlignedCrop) -> tuple[np.ndarray, str]:
        arr = np.frombuffer(crop.pixels, dtype=np.uint8).reshape(
            crop.height, crop.width, 3
        )
        tensor = np.transpose(arr, (2, 0, 1))[None].astype(np.float32)
        raw = self._session.run(["fc1"], {"data": tensor})[0][0]
        norm = float(np.linalg.norm(raw))
        if norm == 0.0:
            raise ModelIntegrityError(
                f"zero-norm embedding from {self._manifest.model_id}"
            )
        return raw / norm, self._manifest.model_id


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
    model_version_a: str,
    model_version_b: str,
) -> float:
    """Cosine score; cross-model_version comparison raises, never scores."""
    if model_version_a != model_version_b:
        raise ValueError(
            "cross-model comparison refused: "
            f"{model_version_a!r} vs {model_version_b!r}"
        )
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        raise ValueError("zero-norm embedding cannot be compared")
    return float(np.dot(a, b) / denom)
