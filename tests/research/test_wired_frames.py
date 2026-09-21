"""Task t-3 tests: wired frame staging + per-frame ID ledger.

RED contract:
- 真機路徑存幀：frames sampled through the controller land encrypted
  (frame_count > 0, decryptable, dims-capped).
- 逐幀 ID 落盤：record envelope carries per-frame best-match entries;
  review band keeps matched_identity None.
- replay 真 gallery：cmd_replay accepts --models/--corpus and replays a
  true bundle (no generation_mismatch).
Repo holds synthetic only.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from facecore.live.capture import FakeCapture
from facecore.live.contracts import FramePacket
from facecore.research.cli import cmd_live, cmd_replay


def _profile_dict(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "t3-test-v1",
        "timeout_ms": 5000,
        "sample_interval_ms": 200,
        "max_frames": 26,
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
            {"path": str(p1), "role": "enrollment",
             "identity": "person-01", "conditions": []},
            {"path": str(p2), "role": "enrollment",
             "identity": "person-02", "conditions": []},
        ]
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(manifest))
    return path


def _mock_detector_embedder() -> tuple[MagicMock, MagicMock]:
    from facecore.pipeline.detect import DetectedFace

    landmarks = ((50.0, 60.0), (110.0, 60.0), (80.0, 90.0),
                 (60.0, 115.0), (100.0, 115.0))
    face = DetectedFace(box=(20.0, 20.0, 120.0, 120.0),
                        landmarks=landmarks, confidence=0.99)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [face]
    mock_embedder = MagicMock()
    mock_embedder.model_version = "sface_2021dec"
    mock_embedder.embed.side_effect = lambda crop: (
        np.array([1.0, 0.0], dtype=np.float32), "sface_2021dec")
    return mock_detector, mock_embedder


def _packet(seq: int) -> FramePacket:
    # NOTE (T8 lesson): true-path sessions start on the live monotonic
    # clock, so injected frames must stamp it too — never seq-derived
    # stamps, which read as time-travel. 200x200 mid-tone pattern per
    # the T2 scorer fixture (fits the mock 120x120 face box).
    import time as _time

    rgb = np.full((200, 200, 3), 120, dtype=np.uint8)
    rgb[::2, ::2] = 160
    rgb[1::2, 1::2] = 80
    rgb[0, 0, 0] = seq % 256
    return FramePacket(
        sequence=seq,
        captured_ns=_time.monotonic_ns() + seq * 200_000_000,
        rgb=rgb,
    )


def test_true_path_stages_frames_encrypted(tmp_path: Path) -> None:
    profile_path = _profile_dict(tmp_path)
    corpus = _corpus_manifest(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    detector, embedder = _mock_detector_embedder()
    frames = [_packet(seq) for seq in range(1, 6)]
    rc = cmd_live(
        profile_path=profile_path,
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        device="0",
        session_id="sess-t3-001",
        record_consent=True,
        image_consent=True,
        models=models,
        corpus=corpus,
        capture_factory=lambda _device: FakeCapture(frames=frames),
        detector_factory=lambda _models: detector,
        embedder_factory=lambda _models: embedder,
    )
    assert rc == 0
    manifest = json.loads(
        (tmp_path / "store" / "sess-t3-001" / "manifest.json").read_text())
    assert manifest["status"] == "committed"
    assert manifest["frame_count"] > 0
    assert manifest["frame_count"] <= 25


def test_per_frame_ledger_in_envelope_review_keeps_no_identity(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timezone
    from facecore.research.recorder import ResearchRecorder

    profile_path = _profile_dict(tmp_path)
    corpus = _corpus_manifest(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    detector, embedder = _mock_detector_embedder()
    frames = [_packet(seq) for seq in range(1, 6)]
    rc = cmd_live(
        profile_path=profile_path,
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        device="0",
        session_id="sess-t3-002",
        record_consent=True,
        image_consent=True,
        models=models,
        corpus=corpus,
        capture_factory=lambda _device: FakeCapture(frames=frames),
        detector_factory=lambda _models: detector,
        embedder_factory=lambda _models: embedder,
    )
    assert rc == 0

    def _clock() -> datetime:
        return datetime.now(timezone.utc)

    rec = ResearchRecorder(
        store_root=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        clock=_clock,
    )
    record = rec.read_record("sess-t3-002")
    assert len(record.frame_scores) > 0
    first = record.frame_scores[0]
    assert first.top_identity in ("person-01", "person-02")
    assert 0.0 <= first.top_score <= 1.0
    assert first.margin is not None
    if record.result.status != "matched":
        assert record.result.matched_identity is None
    # Staged frames decrypt back to packets.
    packet = rec.read_frame("sess-t3-002", 0)
    assert packet.sequence >= 1


def test_replay_with_true_gallery_models(tmp_path: Path) -> None:
    profile_path = _profile_dict(tmp_path)
    corpus = _corpus_manifest(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    detector, embedder = _mock_detector_embedder()
    frames = [_packet(seq) for seq in range(1, 6)]
    assert cmd_live(
        profile_path=profile_path,
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        device="0",
        session_id="sess-t3-003",
        record_consent=True,
        image_consent=True,
        models=models,
        corpus=corpus,
        capture_factory=lambda _device: FakeCapture(frames=frames),
        detector_factory=lambda _models: detector,
        embedder_factory=lambda _models: embedder,
    ) == 0
    rc = cmd_replay(
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        session_id="sess-t3-003",
        profile_path=profile_path,
        models=models,
        corpus=corpus,
        detector_factory=lambda _models: detector,
        embedder_factory=lambda _models: embedder,
    )
    assert rc == 0


def test_fake_path_still_stages_no_frames(tmp_path: Path) -> None:
    profile_path = _profile_dict(tmp_path)
    assert cmd_live(
        profile_path=profile_path,
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        device="fake",
        session_id="sess-t3-004",
        record_consent=True,
        image_consent=True,
    ) == 0
    manifest = json.loads(
        (tmp_path / "store" / "sess-t3-004" / "manifest.json").read_text())
    assert manifest["frame_count"] == 0
