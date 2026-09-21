"""Behavior-level tests for downstream 25-bound constants and contracts (Issue #84).

RED evidence: these tests must FAIL on the current base (173bda7e) and turn
GREEN with the downstream bounds alignment fix:
1. 26th frame stages in recorder and reads in SessionTrace.
2. 25-frame run under 26-frame profile replays as early-stop (not full).
3. Trace overflow surfaces out of controller._append_live_trace.
4. Guard test asserting both MAX constants >= ceil(5000/200)+1.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
)
from facecore.live.controller import LiveController
from facecore.live.session import SessionEngine
from facecore.research.diagnostics import (
    FrameDiagnostics,
    FrameTraceEntry,
    MAX_TRACE_ENTRIES,
)
from facecore.research.recorder import (
    MAX_FRAMES_PER_SESSION,
    ResearchRecorder,
)
from facecore.research.records import ConsentRecord
from facecore.research.replay import _window_for


def _now() -> datetime:
    return datetime(2026, 9, 21, 12, 0, 0, tzinfo=timezone.utc)


def _consent(session_id: str = "sess-bounds-001") -> ConsentRecord:
    return ConsentRecord(
        session_id=session_id,
        participant_id="part-01",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-21T12:00:00+00:00",
        record_expires_at_utc="2026-10-21T12:00:00+00:00",
        image_expires_at_utc="2026-09-28T12:00:00+00:00",
    )


def _frame(seq: int) -> FramePacket:
    return FramePacket(
        sequence=seq,
        captured_ns=(seq - 1) * 200_000_000,
        rgb=np.full((16, 16, 3), 100, dtype=np.uint8),
    )


def _trace_entry(seq: int) -> FrameTraceEntry:
    return FrameTraceEntry(
        sequence=seq,
        captured_ns=(seq - 1) * 200_000_000,
        processed_ns=(seq - 1) * 200_000_000 + 1_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(2.0, 2.0, 4.0, 4.0),
        identity_score_pairs=(("id-01", 0.8),),
        quality_rank=0.9,
        model_generation="gen-1",
        gallery_digest="gal-1",
        diagnostics=FrameDiagnostics(
            sharpness=0.9,
            brightness=0.5,
            contrast=0.5,
            illumination_uniformity=0.8,
            crop_symmetry_ratio=0.5,
            edge_headroom_ratio=0.2,
            edge_chin_margin_ratio=0.2,
            edge_left_margin_ratio=0.2,
            edge_right_margin_ratio=0.2,
            pose_pitch=0.0,
            pose_yaw=0.0,
            pose_roll=0.0,
            eye_distance_px=30.0,
            interocular_distance_ratio=0.3,
            eye_open_ratio=0.8,
            mouth_closed_ratio=0.9,
            motion_blur_score=0.1,
            occlusion_score=0.0,
        ),
    )


def _profile(max_frames: int = 26) -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="bound-test-v1",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=max_frames,
        queue_limit=1,
        required_support=3,
        min_support_interval_ms=200,
        match_threshold=0.45,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="det-1",
        quality_policy_version="qual-1",
        continuity_max_center_delta_ratio=None,
    )


# ---------------------------------------------------------------------------
# RED Test 1: 26th frame stages + traces
# ---------------------------------------------------------------------------


def test_26th_frame_stages_and_traces(tmp_path: Path) -> None:
    """26th frame must stage in recorder and load in SessionTrace without raising."""
    store = tmp_path / "store"
    key_dir = tmp_path / "keys"
    store.mkdir()
    key_dir.mkdir()

    recorder = ResearchRecorder(store_root=store, key_dir=key_dir, clock=_now)
    session_id = "sess-26th-001"
    recorder.begin(session_id, _consent(session_id))

    # Stage 26 frames: pre-fix raises ValueError on 26th frame (frame_count >= 25)
    for seq in range(1, 27):
        recorder.append_frame(_frame(seq))

    # Trace 26 entries: pre-fix read_trace raises ValueError in
    # SessionTrace.__post_init__
    attempt_id = f"att-{session_id}"
    for seq in range(1, 27):
        recorder.append_trace(attempt_id, _trace_entry(seq))

    trace = recorder.read_trace(attempt_id)
    assert len(trace.entries) == 26


# ---------------------------------------------------------------------------
# RED Test 2: 25-frame run replays early-stop
# ---------------------------------------------------------------------------


def test_25_frame_run_replays_early_stop() -> None:
    """25-frame session under 26-frame profile must replay as early-stop, not full."""
    profile = _profile(max_frames=26)
    # 25 frames covering 4800ms (not reaching 26 frames, not reaching 5000ms deadline)
    window = _window_for(
        frame_count=25,
        coverage_ns=4_800_000_000,
        profile=profile,
    )
    # Pre-fix: min(26, 25) == 25, so full_frames is True -> returns "full" (defect!)
    # Post-fix: frame_count >= profile.max_frames (25 >= 26 is False) -> "early-stop"
    assert window == "early-stop"


# ---------------------------------------------------------------------------
# RED Test 3: trace overflow surfaces
# ---------------------------------------------------------------------------


def test_trace_overflow_surfaces() -> None:
    """Trace overflow in append_trace must re-raise, not be swallowed silently."""
    mock_engine = MagicMock(spec=SessionEngine)
    mock_engine.profile = _profile()
    mock_capture = MagicMock()
    mock_scorer = MagicMock()

    mock_recorder = MagicMock()
    # Simulate overflow exception from trace recorder
    mock_recorder.append_trace.side_effect = ValueError(
        "trace exceeded max 26 observations, got 27"
    )

    controller = LiveController(
        engine=mock_engine,
        source=mock_capture,
        scorer=mock_scorer,
        trace_recorder=mock_recorder,
        trace_attempt_id="att-overflow-001",
    )

    obs = FrameObservation(
        sequence=27,
        captured_ns=5_200_000_000,
        processed_ns=5_201_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(0.0, 0.0, 1.0, 1.0),
        identity_scores={"id1": 0.9},
        quality_rank=0.9,
        model_generation="gen-1",
        gallery_digest="gal-1",
    )

    # Pre-fix: controller._append_live_trace catches all ValueError
    # and passes (DID NOT RAISE!)
    # Post-fix: only "already exists" re-drive dupe is swallowed; overflow raises!
    with pytest.raises(ValueError, match="exceeded max"):
        controller._append_live_trace(obs)


# ---------------------------------------------------------------------------
# RED Test 4: Guard test asserting both MAX constants >= ceil(5000/200)+1
# ---------------------------------------------------------------------------


def test_guard_both_constants_cover_fixed_window() -> None:
    """Guard: downstream bounds must be >= ceil(timeout_ms/sample_interval_ms)+1.

    Any new 25-bound constant must join this test.
    """
    required_min = math.ceil(5000 / 200) + 1  # 26
    # Pre-fix: MAX_FRAMES_PER_SESSION == 25, MAX_TRACE_ENTRIES == 25 (both fail!)
    assert MAX_FRAMES_PER_SESSION >= required_min, (
        f"MAX_FRAMES_PER_SESSION ({MAX_FRAMES_PER_SESSION}) must be >= {required_min}"
    )
    assert MAX_TRACE_ENTRIES >= required_min, (
        f"MAX_TRACE_ENTRIES ({MAX_TRACE_ENTRIES}) must be >= {required_min}"
    )
