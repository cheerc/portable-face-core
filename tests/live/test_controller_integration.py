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
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionStatus,
)
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
        max_frames=26,
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


def test_square_capture_mapping_reaches_scorer_before_inference() -> None:
    """E7-B r2 F2: rectangular capture is cropped before score_frame."""
    from facecore.live.qt_window import crop_frame

    ctx = _context()
    engine = SessionEngine(_profile(), ctx.gallery.digest, "gen-1")
    original = np.full((200, 240, 3), 120, dtype=np.uint8)
    original[::2, ::2] = 160
    original[1::2, 1::2] = 80
    frame = FramePacket(sequence=1, captured_ns=0, rgb=original)
    seen_shapes: list[tuple[int, ...]] = []
    mappings: list[dict[str, object]] = []

    def transform(packet: FramePacket) -> FramePacket:
        cropped, mapping = crop_frame(packet.rgb)
        mappings.append(mapping.to_dict())
        return FramePacket(
            sequence=packet.sequence,
            captured_ns=packet.captured_ns,
            rgb=cropped,
            orientation=packet.orientation,
            mirrored=packet.mirrored,
        )

    def scorer(packet: FramePacket) -> FrameObservation:
        seen_shapes.append(packet.rgb.shape)
        return score_frame(packet, ctx)

    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=[frame]),
        scorer=scorer,
        frame_transform=transform,
    )
    controller.start_session("sess-e7-square", now_ns=0)
    result = controller.run_until_terminal(max_steps=5)
    controller.close()

    assert result is not None
    assert seen_shapes == [(200, 200, 3)]
    assert mappings == [
        {
            "x": 20,
            "y": 0,
            "size": 200,
            "frame_w": 240,
            "frame_h": 200,
            "mirrored_preview": False,
        }
    ]


def test_sink_receives_original_packet_while_scorer_gets_square() -> None:
    """E7-B r4 S1 RED: staging/preview keep full frame, scorer gets square."""
    from facecore.live.qt_window import crop_frame

    ctx = _context()
    engine = SessionEngine(_profile(), ctx.gallery.digest, "gen-1")
    original = np.full((200, 240, 3), 120, dtype=np.uint8)
    frame = FramePacket(sequence=1, captured_ns=0, rgb=original)
    seen_shapes: list[tuple[int, ...]] = []
    sunk_shapes: list[tuple[int, ...]] = []

    def transform(packet: FramePacket) -> FramePacket:
        cropped, _ = crop_frame(packet.rgb)
        return FramePacket(
            sequence=packet.sequence,
            captured_ns=packet.captured_ns,
            rgb=cropped,
            orientation=packet.orientation,
            mirrored=packet.mirrored,
        )

    def scorer(packet: FramePacket) -> FrameObservation:
        seen_shapes.append(packet.rgb.shape)
        return score_frame(packet, ctx)

    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=[frame]),
        scorer=scorer,
        frame_sink=lambda packet: sunk_shapes.append(packet.rgb.shape),
        frame_transform=transform,
    )
    controller.start_session("sess-e7-split", now_ns=0)
    result = controller.run_until_terminal(max_steps=5)
    controller.close()

    assert result is not None
    assert seen_shapes == [(200, 200, 3)]
    assert sunk_shapes == [(200, 240, 3)]
