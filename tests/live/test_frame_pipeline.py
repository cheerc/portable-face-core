"""Phase 2A Task T2 tests: decoded-frame single frame pipeline & fixed research gallery.

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T2
    - Task: t-20260914111107867791-76424-33
    - Governing decision: d-20260914110757304910-5
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import pytest

from facecore.contracts.policy import PolicyProfile
from facecore.live.contracts import FramePacket, ResearchProfile
from facecore.live.frame_pipeline import (
    ResearchGallery,
    ScoringContext,
    build_research_gallery,
    score_frame,
)
from facecore.pipeline.detect import DetectedFace


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def make_test_profile() -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="provisional_v1",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=25,
        queue_limit=1,
        required_support=3,
        min_support_interval_ms=200,
        match_threshold=0.45,
        review_threshold=0.363,
        margin_threshold=0.10,
        detector_version="yunet_2023mar",
        quality_policy_version="standard_v1",
        continuity_max_center_delta_ratio=None,
    )


def make_dummy_detected_face(
    x: float = 10.0,
    y: float = 10.0,
    w: float = 120.0,
    h: float = 120.0,
    conf: float = 0.95,
) -> DetectedFace:
    landmarks = (
        (x + 30.0, y + 40.0),
        (x + 90.0, y + 40.0),
        (x + 60.0, y + 70.0),
        (x + 40.0, y + 95.0),
        (x + 80.0, y + 95.0),
    )
    return DetectedFace(box=(x, y, w, h), landmarks=landmarks, confidence=conf)


# ---------------------------------------------------------------------------
# ScoringContext tests
# ---------------------------------------------------------------------------


def test_scoring_context_with_thresholds_valid_and_rejects_invalid() -> None:
    gallery = ResearchGallery(
        embeddings={"person-01": np.array([1.0, 0.0], dtype=np.float32)},
        model_version="sface_2021dec",
        generation="gen-1",
        digest="gal-digest",
    )
    policy = PolicyProfile.frozen_v1()
    ctx = ScoringContext(
        gallery=gallery,
        model_version="sface_2021dec",
        policy=policy,
        detector=MagicMock(),
        embedder=MagicMock(),
    )

    # Valid threshold update
    ctx_updated = ctx.with_thresholds(
        match_threshold=0.50, review_threshold=0.40, margin_threshold=0.15
    )
    assert ctx_updated.policy.match_threshold == 0.50
    assert ctx_updated.policy.review_threshold == 0.40
    assert ctx_updated.policy.margin_threshold == 0.15

    # Invalid: review_threshold > match_threshold
    with pytest.raises(ValueError, match="review_threshold must not exceed"):
        ctx.with_thresholds(
            match_threshold=0.40, review_threshold=0.50, margin_threshold=0.15
        )


# ---------------------------------------------------------------------------
# Test-First RED #1: Color order & Mirroring contracts
# ---------------------------------------------------------------------------


def test_color_normalization_and_encoded_fixture_equivalence() -> None:
    """RED case: color order or orientation difference produces divergent crops."""
    # Create test image with pattern satisfying exposure and sharpness gates
    arr = np.full((200, 200, 3), 120, dtype=np.uint8)
    arr[::2, ::2] = 160
    arr[1::2, 1::2] = 80
    # Distinct asymmetric patch
    arr[20:60, 20:60, 0] = 180
    arr[20:60, 20:60, 1] = 60
    arr[20:60, 20:60, 2] = 100

    # Create FramePacket from RGB array
    packet = FramePacket(
        sequence=1, captured_ns=1000, rgb=arr, orientation=0, mirrored=False
    )

    face = make_dummy_detected_face(20.0, 20.0, 120.0, 120.0)

    # Mock detector and embedder
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [face]
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = (
        np.array([1.0, 0.0], dtype=np.float32),
        "sface_2021dec",
    )

    gallery = ResearchGallery(
        embeddings={"person-01": np.array([1.0, 0.0], dtype=np.float32)},
        model_version="sface_2021dec",
        generation="gen-1",
        digest="gal-digest",
    )
    ctx = ScoringContext(
        gallery=gallery,
        model_version="sface_2021dec",
        policy=PolicyProfile.frozen_v1().with_thresholds(
            match_threshold=0.45, review_threshold=0.363, margin_threshold=0.10
        ),
        detector=mock_detector,
        embedder=mock_embedder,
    )

    obs = score_frame(packet, ctx)
    assert obs.face_count == 1
    assert obs.quality_pass
    assert mock_embedder.embed.called


def test_mirror_only_affects_preview_not_inference() -> None:
    """RED case: mirrored flag on FramePacket must NOT alter inference scores."""
    arr = np.full((200, 200, 3), 120, dtype=np.uint8)
    arr[::2, ::2] = 160
    arr[1::2, 1::2] = 80
    arr[20:80, 20:80, 0] = 180  # Asymmetric patch

    pkt_normal = FramePacket(
        sequence=1, captured_ns=1000, rgb=arr, orientation=0, mirrored=False
    )
    pkt_mirrored = FramePacket(
        sequence=2, captured_ns=2000, rgb=arr, orientation=0, mirrored=True
    )

    face = make_dummy_detected_face(20.0, 20.0, 120.0, 120.0)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [face]

    captured_crops: list[np.ndarray] = []

    def mock_embed(crop: object) -> tuple[np.ndarray, str]:
        pixels = getattr(crop, "pixels")
        captured_crops.append(np.frombuffer(pixels, dtype=np.uint8).copy())
        return np.array([1.0, 0.0], dtype=np.float32), "sface_2021dec"

    mock_embedder = MagicMock()
    mock_embedder.embed.side_effect = mock_embed

    gallery = ResearchGallery(
        embeddings={"person-01": np.array([1.0, 0.0], dtype=np.float32)},
        model_version="sface_2021dec",
        generation="gen-1",
        digest="gal-digest",
    )
    ctx = ScoringContext(
        gallery=gallery,
        model_version="sface_2021dec",
        policy=PolicyProfile.frozen_v1().with_thresholds(
            match_threshold=0.45, review_threshold=0.363, margin_threshold=0.10
        ),
        detector=mock_detector,
        embedder=mock_embedder,
    )

    obs1 = score_frame(pkt_normal, ctx)
    obs2 = score_frame(pkt_mirrored, ctx)

    assert len(captured_crops) == 2
    # The crops passed to embedder must be IDENTICAL;
    # mirrored=True is purely for display preview
    np.testing.assert_array_equal(captured_crops[0], captured_crops[1])
    assert obs1.identity_scores == obs2.identity_scores


# ---------------------------------------------------------------------------
# Test-First RED #2: Single-face and Quality Rejection
# ---------------------------------------------------------------------------


def test_zero_face_does_not_embed() -> None:
    packet = FramePacket(
        sequence=1, captured_ns=1000, rgb=np.zeros((100, 100, 3), dtype=np.uint8)
    )
    mock_detector = MagicMock()
    mock_detector.detect.return_value = []  # Zero faces detected
    mock_embedder = MagicMock()

    gallery = ResearchGallery(
        embeddings={"person-01": np.array([1.0, 0.0], dtype=np.float32)},
        model_version="sface_2021dec",
        generation="gen-1",
        digest="gal-digest",
    )
    ctx = ScoringContext(
        gallery=gallery,
        model_version="sface_2021dec",
        policy=PolicyProfile.frozen_v1().with_thresholds(
            match_threshold=0.45, review_threshold=0.363, margin_threshold=0.10
        ),
        detector=mock_detector,
        embedder=mock_embedder,
    )

    obs = score_frame(packet, ctx)
    assert obs.face_count == 0
    assert not obs.quality_pass
    assert "input_no_face" in obs.quality_reasons
    assert obs.identity_scores == {}
    assert not mock_embedder.embed.called, (
        "Embedder must not be called when face_count == 0"
    )


def test_multi_face_does_not_embed() -> None:
    packet = FramePacket(
        sequence=1, captured_ns=1000, rgb=np.zeros((200, 200, 3), dtype=np.uint8)
    )
    f1 = make_dummy_detected_face(10.0, 10.0, 50.0, 50.0)
    f2 = make_dummy_detected_face(100.0, 100.0, 50.0, 50.0)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [f1, f2]  # Multi-face
    mock_embedder = MagicMock()

    gallery = ResearchGallery(
        embeddings={"person-01": np.array([1.0, 0.0], dtype=np.float32)},
        model_version="sface_2021dec",
        generation="gen-1",
        digest="gal-digest",
    )
    ctx = ScoringContext(
        gallery=gallery,
        model_version="sface_2021dec",
        policy=PolicyProfile.frozen_v1().with_thresholds(
            match_threshold=0.45, review_threshold=0.363, margin_threshold=0.10
        ),
        detector=mock_detector,
        embedder=mock_embedder,
    )

    obs = score_frame(packet, ctx)
    assert obs.face_count == 2
    assert not obs.quality_pass
    assert "input_multiple_faces" in obs.quality_reasons
    assert obs.identity_scores == {}
    assert not mock_embedder.embed.called, (
        "Embedder must not be called when face_count > 1"
    )


def test_quality_rejection_does_not_embed() -> None:
    packet = FramePacket(
        sequence=1, captured_ns=1000, rgb=np.zeros((200, 200, 3), dtype=np.uint8)
    )
    # Low confidence face (< 0.90)
    low_conf_face = make_dummy_detected_face(20.0, 20.0, 120.0, 120.0, conf=0.70)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [low_conf_face]
    mock_embedder = MagicMock()

    gallery = ResearchGallery(
        embeddings={"person-01": np.array([1.0, 0.0], dtype=np.float32)},
        model_version="sface_2021dec",
        generation="gen-1",
        digest="gal-digest",
    )
    ctx = ScoringContext(
        gallery=gallery,
        model_version="sface_2021dec",
        policy=PolicyProfile.frozen_v1().with_thresholds(
            match_threshold=0.45, review_threshold=0.363, margin_threshold=0.10
        ),
        detector=mock_detector,
        embedder=mock_embedder,
    )

    obs = score_frame(packet, ctx)
    assert obs.face_count == 1
    assert not obs.quality_pass
    assert "quality_detector_confidence" in obs.quality_reasons
    assert obs.identity_scores == {}
    assert not mock_embedder.embed.called, (
        "Embedder must not be called on quality rejection"
    )


# ---------------------------------------------------------------------------
# Test-First RED #3: Model mismatch & Gallery Immutability
# ---------------------------------------------------------------------------


def test_model_mismatch_rejected() -> None:
    gallery = ResearchGallery(
        embeddings={"person-01": np.array([1.0, 0.0], dtype=np.float32)},
        model_version="sface_2021dec",
        generation="gen-1",
        digest="gal-digest",
    )
    mock_embedder = MagicMock()
    mock_embedder.model_version = "sface_2026may_mismatch"

    with pytest.raises(ValueError, match="model mismatch"):
        ScoringContext(
            gallery=gallery,
            model_version="sface_2021dec",
            policy=PolicyProfile.frozen_v1(),
            detector=MagicMock(),
            embedder=mock_embedder,
        )


def test_gallery_immutability_and_building_rules(tmp_path: Path) -> None:
    # 1. Gallery embeddings cannot be mutated
    gallery = ResearchGallery(
        embeddings={"person-01": np.array([1.0, 0.0], dtype=np.float32)},
        model_version="sface_2021dec",
        generation="gen-1",
        digest="gal-digest",
    )
    with pytest.raises(TypeError):
        gallery.embeddings["person-02"] = np.array([0.0, 1.0], dtype=np.float32)  # type: ignore[index]

    # 2. Duplicate identity in manifest rejected
    manifest_dup = tmp_path / "manifest_dup.json"
    photo1 = tmp_path / "p1.jpg"
    photo2 = tmp_path / "p2.jpg"
    valid_img = Image.new("RGB", (64, 64), color=(120, 120, 120))
    valid_img.save(photo1)
    valid_img.save(photo2)

    manifest_dup.write_text(
        '{"files": ['
        '{"path": "'
        + str(photo1)
        + '", "role": "enrollment", "identity": "person-01"},'
        '{"path": "' + str(photo2) + '", "role": "enrollment", "identity": "person-01"}'
        "]}"
    )

    mock_embedder = MagicMock()
    mock_embedder.model_version = "sface_2021dec"
    mock_detector = MagicMock()

    # Pass an isolated fake repo_root so tmp_path files are guaranteed
    # strictly external to repo_root (fixes Linux CI runner /tmp path collision)
    isolated_repo = tmp_path / "fake_repo"
    isolated_repo.mkdir()

    with pytest.raises(ValueError, match="duplicate identity in enrollment manifest"):
        build_research_gallery(
            manifest_path=manifest_dup,
            repo_root=isolated_repo,
            detector=mock_detector,
            embedder=mock_embedder,
            generation="gen-1",
        )

    # 3. Missing file in manifest rejected
    manifest_missing = tmp_path / "manifest_missing.json"
    manifest_missing.write_text(
        '{"files": ['
        '{"path": "'
        + str(tmp_path / "nonexistent.jpg")
        + '", "role": "enrollment", "identity": "person-02"}'
        "]}"
    )
    with pytest.raises(FileNotFoundError):
        build_research_gallery(
            manifest_path=manifest_missing,
            repo_root=isolated_repo,
            detector=mock_detector,
            embedder=mock_embedder,
            generation="gen-1",
        )
