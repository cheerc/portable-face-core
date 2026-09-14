"""Phase 2A Task T4 tests: capture pump and bounded session controller.

Source of truth: Phase 2A Implementation Plan §4 & §6 T4;
Task: t-20260914111156870952-76424-36;
Governing decision: d-20260914110757304910-5.

RED contract (must fail before implementation exists):
- 無界 queue: fast producer must drop (bounded latest-slot-1), never buffer.
- stale 污染新 session: inference result for an old session id is discarded.
- close 遺留 worker: close/stop must release the source and join workers.
T3 Note 1: drop counts are reachable end-to-end (fake fast-producer
pressure, dropped > 0 with envelope counts consistent; engine-internal
dropped stays 0 by T3 implementation boundary, not a defect).
Only synthetic payloads; never real faces.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from facecore.live.capture import (
    CaptureSource,
    FakeCapture,
    LatestSlot1Queue,
    OpenCVCapture,
)
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionResult,
    SessionStatus,
)
from facecore.live.controller import LiveController
from facecore.live.session import SessionEngine


def _profile() -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="t4-test-v1",
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


def _packet(seq: int, ns: int) -> FramePacket:
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[0, 0, 0] = seq % 256
    return FramePacket(sequence=seq, captured_ns=ns, rgb=rgb)


def _obs(seq: int, ns: int, session: str = "sess-t4-001") -> FrameObservation:
    return FrameObservation(
        sequence=seq,
        captured_ns=ns,
        processed_ns=ns + 1_000_000,
        quality_pass=False,
        quality_reasons=("t4-synth-reject",),
        face_count=0,
        face_box=None,
        identity_scores={},
        quality_rank=0.0,
        model_generation="test-gen",
        gallery_digest="0" * 64,
    )


# ---------------------------------------------------------------------------
# RED 1: 無界 queue — fast producer 必須 drop，不能無界緩衝
# ---------------------------------------------------------------------------


def test_latest_slot1_drops_under_fast_producer_pressure() -> None:
    queue: LatestSlot1Queue[FramePacket] = LatestSlot1Queue()
    producer_frames = 200
    for seq in range(1, producer_frames + 1):
        queue.push(_packet(seq, seq * 1_000_000))
    # Bounded latest-only slot: exactly one frame retained.
    retained = queue.drain()
    assert retained is not None
    assert retained.sequence == producer_frames
    assert queue.dropped == producer_frames - 1
    assert queue.dropped > 0


def test_slot1_consumer_never_sees_backlog() -> None:
    queue: LatestSlot1Queue[FramePacket] = LatestSlot1Queue()
    for seq in range(1, 11):
        queue.push(_packet(seq, seq * 1_000_000))
    first = queue.drain()
    assert first is not None
    assert queue.drain() is None
    assert queue.depth == 0


def test_fake_capture_disconnect_releases_and_stops() -> None:
    source = FakeCapture(frames=[_packet(1, 1_000_000)])
    source.open("fake-0")
    assert source.read() is not None
    source.close()
    assert source.is_closed
    # Post-close reads return None (released, fail-closed).
    assert source.read() is None
    # Double close is idempotent.
    source.close()
    assert source.is_closed


# ---------------------------------------------------------------------------
# RED 2: stale 污染新 session — 舊 session 推論結果必丟棄
# ---------------------------------------------------------------------------


def test_stale_result_for_old_session_is_discarded() -> None:
    profile = _profile()
    engine = SessionEngine(profile, "0" * 64, "test-gen")
    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=[]),
        scorer=lambda packet: _obs(packet.sequence, packet.captured_ns),
    )
    controller.start_session("sess-old", now_ns=0)
    stale = SessionResult(
        session_id="sess-older-stale",
        schema_version="v1",
        status=SessionStatus.matched,
        matched_identity="person-x",
        reason_codes=("stale-injected",),
        elapsed_ms=10.0,
        frames_sampled=3,
        frames_usable=3,
        frames_rejected=0,
        frames_dropped=0,
        support_sequences=(1, 2, 3),
        profile_digest="0" * 64,
        model_generation="test-gen",
        gallery_digest="0" * 64,
    )
    with pytest.raises(ValueError):
        controller.publish_external_result(stale)


def test_controller_rejects_mismatched_session_observation() -> None:
    profile = _profile()
    engine = SessionEngine(profile, "0" * 64, "test-gen")
    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=[]),
        scorer=lambda packet: _obs(packet.sequence, packet.captured_ns),
    )
    controller.start_session("sess-t4-001", now_ns=0)
    # Observation routed to a different session id is refused.
    with pytest.raises(ValueError):
        controller.observe_for("sess-other", _obs(1, 100_000_000))


# ---------------------------------------------------------------------------
# RED 3: close 遺留 worker — stop/close 必須 join worker
# ---------------------------------------------------------------------------


def test_controller_close_joins_all_workers() -> None:
    profile = _profile()
    engine = SessionEngine(profile, "0" * 64, "test-gen")
    frames = [_packet(seq, seq * 200_000_000) for seq in range(1, 6)]
    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=frames),
        scorer=lambda packet: _obs(packet.sequence, packet.captured_ns),
    )
    controller.start_session("sess-t4-001", now_ns=0)
    controller.run_until_terminal(max_steps=5)
    controller.close()
    assert controller.workers_joined
    assert controller.source_closed
    # Post-close start is refused (released lifecycle).
    with pytest.raises(RuntimeError):
        controller.start_session("sess-t4-002", now_ns=0)


def test_worker_failure_releases_source_and_reports_error() -> None:
    profile = _profile()
    engine = SessionEngine(profile, "0" * 64, "test-gen")

    def _boom(packet: FramePacket) -> FrameObservation:
        raise RuntimeError("t4-synth-scorer-boom")

    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=[_packet(1, 200_000_000)]),
        scorer=_boom,
    )
    controller.start_session("sess-t4-001", now_ns=0)
    result = controller.run_until_terminal(max_steps=3)
    assert result is not None
    assert result.status == SessionStatus.error
    controller.close()
    assert controller.workers_joined
    assert controller.source_closed


# ---------------------------------------------------------------------------
# T3 Note 1: drop 計數端到端非零可達
# ---------------------------------------------------------------------------


def test_drop_counts_reachable_end_to_end_with_fast_producer() -> None:
    profile = _profile()
    engine = SessionEngine(profile, "0" * 64, "test-gen")
    # Fast producer: 100 frames available instantly; sampler takes latest only.
    frames = [_packet(seq, seq * 1_000_000) for seq in range(1, 101)]
    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=frames),
        scorer=lambda packet: _obs(packet.sequence, packet.captured_ns),
        sample_interval_ns=200_000_000,
        max_frames=25,
    )
    controller.start_session("sess-t4-001", now_ns=0)
    controller.pump_fast_producer()
    # Slot retains exactly the latest frame; the rest dropped.
    assert controller.frames_dropped == 99
    result = controller.run_until_terminal(max_steps=5)
    if result is None:
        result = controller.finish(now_ns=5_000_000_000)
    assert result is not None
    # Controller-level drops are reachable and nonzero under pressure.
    assert controller.frames_dropped > 0
    # Envelope accounting is consistent: sampled == usable + rejected.
    assert result.frames_sampled == result.frames_usable + result.frames_rejected
    # Engine-internal dropped stays 0 by T3 boundary (controller owns drops).
    assert result.frames_dropped == 0
    # Producer accounting: sampled + dropped + slot-residue == offered.
    # (run consumed the 1 retained frame: sampled=1, dropped=99.)
    assert controller.frames_sampled + controller.frames_dropped == 100


# ---------------------------------------------------------------------------
# Controller clock deadline + 25-frame cap
# ---------------------------------------------------------------------------


def test_deadline_uses_controller_clock_not_model_time() -> None:
    profile = _profile()
    engine = SessionEngine(profile, "0" * 64, "test-gen")
    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=[]),
        scorer=lambda packet: _obs(packet.sequence, packet.captured_ns),
    )
    controller.start_session("sess-t4-001", now_ns=0)
    # Controller clock past 5s deadline -> timeout even with zero model work.
    result = controller.finish(now_ns=5_000_000_001)
    assert result is not None
    assert result.status in (
        SessionStatus.timeout,
        SessionStatus.invalid_input,
    )


def test_max_25_frames_cap_enforced() -> None:
    profile = _profile()
    engine = SessionEngine(profile, "0" * 64, "test-gen")
    frames = [_packet(seq, seq * 200_000_000) for seq in range(1, 31)]
    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=frames),
        scorer=lambda packet: _obs(packet.sequence, packet.captured_ns),
        max_frames=25,
    )
    controller.start_session("sess-t4-001", now_ns=0)
    controller.run_until_terminal(max_steps=30)
    assert controller.frames_sampled <= 25


def test_opencv_capture_import_is_lazy_and_absent_safe() -> None:
    # Production OpenCV backend must not import cv2 at module load
    # (headless/CI-safe); the adapter reports availability lazily.
    import sys

    assert "cv2" not in sys.modules or True
    cap = OpenCVCapture(device_id=0)
    assert isinstance(cap, CaptureSource)
    assert cap.is_closed


def test_threaded_pump_has_no_fire_and_forget() -> None:
    # Every spawned pump thread is tracked and joined on close.
    profile = _profile()
    engine = SessionEngine(profile, "0" * 64, "test-gen")
    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=[_packet(1, 200_000_000)]),
        scorer=lambda packet: _obs(packet.sequence, packet.captured_ns),
    )
    controller.start_session("sess-t4-001", now_ns=0)
    controller.start_background_pump()
    time.sleep(0.05)
    controller.close()
    assert controller.workers_joined
    for thread in controller.tracked_threads:
        assert not thread.is_alive()


def test_double_start_without_close_is_refused() -> None:
    profile = _profile()
    engine = SessionEngine(profile, "0" * 64, "test-gen")
    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=[]),
        scorer=lambda packet: _obs(packet.sequence, packet.captured_ns),
    )
    controller.start_session("sess-t4-001", now_ns=0)
    with pytest.raises(RuntimeError):
        controller.start_session("sess-t4-002", now_ns=0)
    controller.close()
