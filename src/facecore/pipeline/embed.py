"""ONNX embedding with manifest integrity gate (Task 6).

前處理由 manifest 決定：SFace 保持 RGB raw [0,255]、zero mean；
ArcFace RGB 使用 manifest mean／scale。兩者皆為 NCHW float32，輸出
以 L2 normalization 正規化；模型 IO 名稱從 session 讀取。
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
        tensor = (tensor - self._manifest.input_mean) / self._manifest.input_scale
        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        raw = self._session.run([output_name], {input_name: tensor})[0][0]
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
