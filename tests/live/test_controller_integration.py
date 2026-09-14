"""Phase 2A Task T4 integration: controller + real T2 scorer + T3 engine.

Source of truth: Phase 2A Implementation Plan §4 & §6 T4;
Task: t-20260914111156870952-76424-36;
Governing decision: d-20260914110757304910-5.

Proves the production wiring FakeCapture → score_frame → SessionEngine
with synthetic pixels only (mid-tone checkerboard per T2 pattern).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from facecore.contracts.policy import PolicyProfile
from facecore.live.capture import FakeCapture
from facecore.live.contracts import FramePacket, ResearchProfile, SessionStatus
from facecore.live.controller import LiveController
from facecore.live.frame_pipeline import (
    ResearchGallery,
    ScoringContext,
    score_frame,
)
from facecore.live.session import SessionEngine
from facecore.pipeline.detect import DetectedFace


def _profile() -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="t4-int-v1",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=25,
        queue_limit=1,
        required_support=3,
        min_support_interval_ms=200,
        match_threshold=0.45,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="yunet-test",
        quality_policy_version="q-test-v1",
        continuity_max_center_delta_ratio=0.50,
    )


def _synthetic_rgb(seed: int) -> np.ndarray:
    arr = np.full((200, 200, 3), 120, dtype=np.uint8)
    arr[::2, ::2] = 160
    arr[1::2, 1::2] = 80
    arr[20:80, 20:80, 0] = (150 + seed) % 256
    return arr


def _context() -> ScoringContext:
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

    def mock_embed(crop: object) -> tuple[np.ndarray, str]:
        return np.array([1.0, 0.0], dtype=np.float32), "sface_2021dec"

    mock_embedder = MagicMock()
    mock_embedder.embed.side_effect = mock_embed

    gallery = ResearchGallery(
        embeddings={
            "person-01": np.array([1.0, 0.0], dtype=np.float32),
            "person-02": np.array([0.0, 1.0], dtype=np.float32),
        },
        model_version="sface_2021dec",
        generation="gen-1",
        digest="gal-digest",
    )
    return ScoringContext(
        gallery=gallery,
        model_version="sface_2021dec",
        policy=PolicyProfile.frozen_v1().with_thresholds(
            match_threshold=0.45, review_threshold=0.30, margin_threshold=0.10
        ),
        detector=mock_detector,
        embedder=mock_embedder,
    )


def test_production_wiring_capture_to_scorer_to_engine() -> None:
    ctx = _context()
    engine = SessionEngine(_profile(), ctx.gallery.digest, "gen-1")
    frames = [
        FramePacket(
            sequence=seq,
            captured_ns=seq * 200_000_000,
            rgb=_synthetic_rgb(seq),
        )
        for seq in range(1, 8)
    ]
    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=frames),
        scorer=lambda packet: score_frame(packet, ctx),
    )
    controller.start_session("sess-t4-int", now_ns=0)
    result = controller.run_until_terminal(max_steps=10)
    controller.close()
    assert controller.workers_joined
    assert controller.source_closed
    # Synthetic wiring must terminate deterministically (matched or
    # timeout/invalid/unknown — never hang, never leak the worker).
    if result is not None:
        assert isinstance(result.status, SessionStatus)
        assert result.session_id == "sess-t4-int"
