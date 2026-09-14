"""Phase 2A Task A tests: true camera-device wiring (non-fake branch).

Source of truth: task t-20260914144956431776-76424-50;
governing decision: d-20260914144615650283-7.

RED contract (must fail before implementation exists):
- 非 fake device 被拒：a non-fake device with models+corpus present
  must NOT be refused (must run the true pipeline, not exit 2).
GREEN keeps --device fake regression green; repo holds synthetic only.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from facecore.live.capture import FakeCapture
from facecore.live.contracts import FramePacket
from facecore.research.cli import cmd_live


def _profile_dict(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "ta-test-v1",
        "timeout_ms": 5000,
        "sample_interval_ms": 200,
        "max_frames": 25,
        "queue_limit": 1,
        "required_support": 3,
        "min_support_interval_ms": 200,
        "match_threshold": 0.45,
        "review_threshold": 0.30,
        "margin_threshold": 0.10,
        "detector_version": "yunet-test",
        "quality_policy_version": "q-test-v1",
        "continuity_max_center_delta_ratio": 0.50,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return path


def _synthetic_png(path: Path, seed: int) -> None:
    from PIL import Image

    arr = np.full((200, 200, 3), 120, dtype=np.uint8)
    arr[::2, ::2] = 160
    arr[1::2, 1::2] = 80
    arr[20:80, 20:80, 0] = (150 + seed) % 256
    Image.fromarray(arr, mode="RGB").save(path)


def _corpus_manifest(tmp_path: Path) -> Path:
    p1 = tmp_path / "enroll_01.png"
    p2 = tmp_path / "enroll_02.png"
    _synthetic_png(p1, 1)
    _synthetic_png(p2, 2)
    manifest = {
        "files": [
            {
                "path": str(p1),
                "role": "enrollment",
                "identity": "person-01",
                "conditions": [],
            },
            {
                "path": str(p2),
                "role": "enrollment",
                "identity": "person-02",
                "conditions": [],
            },
        ]
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(manifest))
    return path


def _mock_detector_embedder() -> tuple[MagicMock, MagicMock]:
    from facecore.pipeline.detect import DetectedFace

    landmarks = (
        (50.0, 60.0),
        (110.0, 60.0),
        (80.0, 90.0),
        (60.0, 115.0),
        (100.0, 115.0),
    )
    face = DetectedFace(
        box=(20.0, 20.0, 120.0, 120.0),
        landmarks=landmarks,
        confidence=0.99,
    )
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [face]
    mock_embedder = MagicMock()
    mock_embedder.model_version = "sface_2021dec"
    mock_embedder.embed.side_effect = lambda crop: (
        np.array([1.0, 0.0], dtype=np.float32),
        "sface_2021dec",
    )
    return mock_detector, mock_embedder


def _packet(seq: int) -> FramePacket:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[0, 0, 0] = seq % 256
    return FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=rgb)


# ---------------------------------------------------------------------------
# RED: non-fake device with models+corpus must run the true pipeline
# ---------------------------------------------------------------------------


def test_true_pipeline_runs_with_injected_capture_and_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from facecore.research import cli as cli_module

    profile_path = _profile_dict(tmp_path)
    corpus = _corpus_manifest(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    detector, embedder = _mock_detector_embedder()
    frames = [_packet(seq) for seq in range(1, 5)]

    monkeypatch_capture = FakeCapture(frames=frames)
    rc = cmd_live(
        profile_path=profile_path,
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        device="0",
        session_id="sess-ta-001",
        record_consent=True,
        image_consent=True,
        models=models,
        corpus=corpus,
        capture_factory=lambda _device: monkeypatch_capture,
        detector_factory=lambda _models: detector,
        embedder_factory=lambda _models: embedder,
    )
    assert rc == 0
    manifest = json.loads(
        (tmp_path / "store" / "sess-ta-001" / "manifest.json").read_text()
    )
    assert manifest["status"] == "committed"
    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["session_id"] == "sess-ta-001"
    assert payload["generation"] == cli_module.TRUE_PIPELINE_GENERATION
    # True gallery digest is frozen from the built gallery, never the fake one.
    assert payload["gallery_digest"] != "cli-fake-gallery"
    assert "facecore.research.cli" not in payload.get("note", "")


def test_missing_models_or_corpus_fail_clear_exit_2(tmp_path: Path) -> None:
    profile_path = _profile_dict(tmp_path)
    rc = cmd_live(
        profile_path=profile_path,
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        device="0",
        session_id="sess-ta-002",
        record_consent=True,
        image_consent=True,
    )
    assert rc == 2
    assert not (tmp_path / "store" / "sess-ta-002").exists()


def test_fake_path_untouched_by_wiring(tmp_path: Path) -> None:
    profile_path = _profile_dict(tmp_path)
    rc = cmd_live(
        profile_path=profile_path,
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        device="fake",
        session_id="sess-ta-003",
        record_consent=True,
        image_consent=True,
    )
    assert rc == 0
    assert (tmp_path / "store" / "sess-ta-003" / "manifest.json").is_file()
