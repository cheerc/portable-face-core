"""Task 6 RED/GREEN: ONNX embedding with manifest integrity gate."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from facecore.contracts.manifest import ModelManifest
from facecore.errors import ModelIntegrityError
from facecore.pipeline.align import AlignedCrop
from facecore.pipeline.embed import Embedder, cosine_similarity

MODELS = Path(__file__).resolve().parents[2] / "models"
BUILD = Path(__file__).resolve().parents[2] / "build" / "embed-fixtures"

FP32_SHA = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"


def _manifest(tmp_artifact: Path) -> ModelManifest:
    base = ModelManifest.sface_2021dec_fp32()
    return ModelManifest(
        model_id=base.model_id,
        source_repo=base.source_repo,
        artifact_url=base.artifact_url,
        retrieval_date=base.retrieval_date,
        code_license=base.code_license,
        weight_license=base.weight_license,
        license_locator=base.license_locator,
        weight_sha256=FP32_SHA,
        provenance=base.provenance,
        provenance_note=base.provenance_note,
        embedding_dim=base.embedding_dim,
        input_width=base.input_width,
        input_height=base.input_height,
        mobile_usability=base.mobile_usability,
    )


def _crop() -> AlignedCrop:
    rng = np.random.default_rng(7)
    pixels = (rng.random((112, 112, 3)) * 255).astype(np.uint8).tobytes()
    return AlignedCrop(width=112, height=112, contract_version=1, pixels=pixels)


def test_tampered_artifact_hash_mismatch_raises_before_inference() -> None:
    """Failing case from the plan: tampered bytes must raise ModelIntegrityError."""
    pytest.importorskip("onnxruntime")
    src = MODELS / "face_recognition_sface_2021dec.onnx"
    pytest.skip("requires downloaded artifact") if not src.exists() else None
    BUILD.mkdir(parents=True, exist_ok=True)
    tampered = BUILD / "sface_tampered.onnx"
    shutil.copy(src, tampered)
    raw = bytearray(tampered.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    tampered.write_bytes(bytes(raw))
    with pytest.raises(ModelIntegrityError) as exc_info:
        Embedder(_manifest(tampered), tampered)
    assert type(exc_info.value).exit_code == 3


def test_embedding_l2_norm_is_one_within_1e_6() -> None:
    pytest.importorskip("onnxruntime")
    src = MODELS / "face_recognition_sface_2021dec.onnx"
    if not src.exists():
        pytest.skip("requires downloaded artifact")
    embedder = Embedder(_manifest(src), src)
    embedding, model_version = embedder.embed(_crop())
    assert model_version == "face_recognition_sface_2021dec"
    assert abs(float(np.linalg.norm(embedding)) - 1.0) <= 1e-6


def test_cross_model_version_comparison_raises() -> None:
    pytest.importorskip("onnxruntime")
    src = MODELS / "face_recognition_sface_2021dec.onnx"
    if not src.exists():
        pytest.skip("requires downloaded artifact")
    embedder = Embedder(_manifest(src), src)
    a, _ = embedder.embed(_crop())
    with pytest.raises(ValueError):
        cosine_similarity(a, a, "face_recognition_sface_2021dec", "other-model")


def test_no_network_during_inference(socket_blocker) -> None:
    pytest.importorskip("onnxruntime")
    src = MODELS / "face_recognition_sface_2021dec.onnx"
    if not src.exists():
        pytest.skip("requires downloaded artifact")
    embedder = Embedder(_manifest(src), src)
    embedder.embed(_crop())  # must not touch the network
