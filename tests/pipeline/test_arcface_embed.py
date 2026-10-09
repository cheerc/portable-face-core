"""ArcFace manifest、前處理與真實權重的 test-first 契約。"""

import hashlib
from types import SimpleNamespace

import numpy as np

from facecore.contracts.manifest import ModelManifest
from facecore.pipeline.align import AlignedCrop
from facecore.pipeline.embed import Embedder


def test_arcface_manifest_is_complete():
    manifest = ModelManifest.arcface_w600k_r50_fp32()
    assert manifest.model_id == "arcface-w600k-r50-fp32"
    assert (
        manifest.weight_sha256
        == "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
    )
    assert (
        manifest.weight_license
        == "ALL models are available for non-commercial research purposes only"
    )
    assert manifest.redistribution == "無；權重不進 Git、不散佈"
    assert "WebFace600K" in manifest.provenance_note
    for field in (
        manifest.code_license,
        manifest.license_locator,
        manifest.mobile_usability,
    ):
        assert field
    assert manifest.embedding_dim == 512


def test_arcface_rgb_normalization_and_session_io(tmp_path, monkeypatch):
    import onnxruntime as ort
    from dataclasses import replace

    artifact = tmp_path / "synthetic.onnx"
    artifact.write_bytes(b"synthetic model fixture")
    manifest = replace(
        ModelManifest.arcface_w600k_r50_fp32(),
        weight_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
    )
    captured = {}

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="input.1")]

        def get_outputs(self):
            return [SimpleNamespace(name="683")]

        def run(self, outputs, feeds):
            captured.update(outputs=outputs, feeds=feeds)
            return [np.array([[3.0, 4.0]], dtype=np.float32)]

    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: Session())
    crop = AlignedCrop(
        112,
        112,
        3,
        np.tile(np.array([0, 128, 255], dtype=np.uint8), (112, 112, 1)).tobytes(),
    )
    vector, version = Embedder(manifest, artifact).embed(crop)
    assert version == manifest.model_id
    assert captured["outputs"] == ["683"]
    tensor = captured["feeds"]["input.1"]
    assert tensor.shape == (1, 3, 112, 112)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(
        tensor[0, :, 0, 0], [-1, (128 - 127.5) / 127.5, 1], atol=1e-7
    )
    np.testing.assert_allclose(vector, [0.6, 0.8], atol=1e-7)
