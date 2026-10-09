"""G3 真入口模型 gate、factory 注入與 SFace eval 隔離。"""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from facecore.live.capture import FakeCapture
from facecore.live.contracts import FramePacket, ResearchProfile
from facecore.research import cli

ROOT = Path(__file__).resolve().parents[2]
ARC = "arcface-w600k-r50-fp32"
SFACE = "face_recognition_sface_2021dec"


def profile(threshold):
    payload = json.loads((ROOT / "profiles/g3-v1.json").read_text())
    payload["match_threshold"] = threshold
    return ResearchProfile.from_dict(payload)


def live(tmp_path, p, **kwargs):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(p.to_dict()))
    capture = FakeCapture(
        frames=[
            FramePacket(
                sequence=1, captured_ns=0, rgb=np.zeros((16, 16, 3), dtype=np.uint8)
            )
        ]
    )
    return cli.cmd_live(
        profile_path=path,
        store=tmp_path / "store",
        key_dir=tmp_path / "keys",
        device="0",
        session_id="arcface-test",
        models=tmp_path / "models",
        gallery_dir=tmp_path / "gallery",
        mode="demo",
        record_consent=False,
        image_consent=False,
        capture_factory=lambda _: capture,
        detector_factory=lambda _: object(),
        **kwargs,
    )


@pytest.mark.parametrize("tampered", [False, True])
def test_default_live_r50_missing_or_wrong_hash_has_model_failure(
    tmp_path, capsys, tampered
):
    models = tmp_path / "models"
    models.mkdir()
    if tampered:
        (models / "w600k_r50.onnx").write_bytes(b"wrong R50")
    # 其他模型存在不能觸發 fallback。
    (models / "w600k_mbf.onnx").write_bytes(b"mbf")
    (models / "face_recognition_sface_2021dec.onnx").write_bytes(b"sface")
    rc = live(tmp_path, profile(0.60))
    error = capsys.readouterr().err
    assert rc == 2
    assert "模型檔載入失敗" in error
    assert "w600k_r50" in error or ARC in error
    assert "face_recognition_sface_2021dec" not in error


@pytest.mark.parametrize("model,threshold", [(ARC, 0.363), (SFACE, 0.60)])
def test_live_rejects_injected_model_threshold_mismatch(
    tmp_path, capsys, model, threshold
):
    rc = live(
        tmp_path,
        profile(threshold),
        embedder_factory=lambda _: SimpleNamespace(model_version=model),
    )
    assert rc == 2
    assert "模型與 profile 不相容" in capsys.readouterr().err


def test_live_rejects_embedder_without_model_identity(tmp_path, capsys):
    rc = live(tmp_path, profile(0.60), embedder_factory=lambda _: object())
    assert rc == 2
    assert "模型與 profile 不相容" in capsys.readouterr().err


def test_arcface_profile_only_changes_identity_and_match_threshold():
    base = json.loads((ROOT / "profiles/g3-v1.json").read_text())
    arc = json.loads((ROOT / "profiles/g3-v1-arcface-r50.json").read_text())
    assert arc.pop("embedding_model_version") == ARC
    assert arc.pop("profile_version") == "g3-v1-arcface-r50"
    assert arc.pop("match_threshold") == 0.60
    base.pop("profile_version")
    base.pop("match_threshold")
    assert arc == base


def test_eval_helper_default_still_uses_sface_and_legacy_generation(
    tmp_path, monkeypatch
):
    from facecore.pipeline import embed
    from facecore.live import frame_pipeline

    def constructor(manifest, artifact):
        assert manifest.model_id == SFACE
        assert artifact.name == "face_recognition_sface_2021dec.onnx"
        return SimpleNamespace(model_version=manifest.model_id)

    def gallery(*args, **kwargs):
        assert kwargs["embedder"].model_version == SFACE
        assert kwargs["generation"] == "gen-1"
        return SimpleNamespace(model_version=SFACE)

    monkeypatch.setattr(embed, "Embedder", constructor)
    monkeypatch.setattr(frame_pipeline, "build_gallery_from_folder", gallery)
    context = cli._build_true_context(
        tmp_path,
        None,
        profile(0.363),
        detector_factory=lambda _: object(),
        gallery_dir=tmp_path,
    )
    assert context.model_version == SFACE
    assert context.policy.match_threshold == 0.363


def test_arcface_gallery_generation_and_profile_identity(tmp_path, monkeypatch):
    from facecore.live import frame_pipeline

    seen = {}

    def gallery(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(model_version=ARC)

    monkeypatch.setattr(frame_pipeline, "build_gallery_from_folder", gallery)
    p = replace(profile(0.60), embedding_model_version=ARC)
    context = cli._build_true_context(
        tmp_path,
        None,
        p,
        detector_factory=lambda _: object(),
        embedder_factory=lambda _: SimpleNamespace(model_version=ARC),
        gallery_dir=tmp_path,
    )
    assert context.model_version == ARC
    assert seen["generation"] == "arcface-112-rgb-minus127.5-div127.5+align3"


def test_correct_threshold_does_not_mask_profile_backend_mismatch(tmp_path, capsys):
    p = replace(profile(0.60), embedding_model_version=SFACE)
    rc = live(
        tmp_path, p, embedder_factory=lambda _: SimpleNamespace(model_version=ARC)
    )
    assert rc == 2
    assert "模型與 profile 不相容" in capsys.readouterr().err
