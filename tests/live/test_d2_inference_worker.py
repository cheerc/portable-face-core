"""D2 RED: inference runs on a worker, the UI thread only paints.

Source of truth:
    - PLAN-mac-face-demo-handoff.md §D2 (capture and inference on a
      worker, Qt main thread only updates the UI; latest-frame bounded
      queue; worker stop/join/device-release under one lifecycle; the
      preview must not depend on quality-pass or research persistence).
    - D2 lead ruling (2026-09-29): preview returns via LatestSlot1Queue
      drained by the existing 20 ms UI timer — NOT a Qt signal, because
      an unverifiable-by-tests change may not enter; research staging
      moves to the worker WITH inference so the preview is genuinely
      decoupled from encrypted disk writes.

What these tests pin, and what they deliberately do NOT claim:

- They prove the UI thread is not BLOCKED by inference. They do NOT
  prove real Qt event-loop concurrency — no test in this repo drives
  `QApplication.exec()` or `processEvents()` (verified by grep). Real
  parallel behaviour is a D4 on-device measurement.

- The camera-open read is currently synchronous on the UI thread too:
  `start_background_pump` is only called by `run_with_timeout` and
  `run_background_and_join`, never by the Qt window. So "capture is
  already on a worker" is true for the headless paths only.

Camera-free throughout: FakeCapture, synthetic frames, a gated scorer,
and an offscreen Qt app.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np
import pytest

from facecore.live.capture import FakeCapture, LatestSlot1Queue
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionStatus,
)
from facecore.live.controller import LiveController
from facecore.live.session import SessionEngine

# The frozen g3-v1 recognition parameters (D0 §6).
G3_MATCH = 0.363
G3_MARGIN = 0.10
G3_SUPPORT = 3
G3_TIMEOUT_MS = 5000


def _profile(**overrides: Any) -> ResearchProfile:
    base: dict[str, Any] = {
        "schema_version": "v1",
        "profile_version": "d2-v1",
        "timeout_ms": G3_TIMEOUT_MS,
        "sample_interval_ms": 200,
        "max_frames": 26,
        "queue_limit": 1,
        "required_support": G3_SUPPORT,
        "min_support_interval_ms": 200,
        "match_threshold": G3_MATCH,
        "review_threshold": 0.30,
        "margin_threshold": G3_MARGIN,
        "detector_version": "yunet",
        "quality_policy_version": "1",
        "continuity_max_center_delta_ratio": 0.5,
    }
    base.update(overrides)
    return ResearchProfile(**base)  # type: ignore[arg-type]


def _packet(seq: int, captured_ns: int) -> FramePacket:
    return FramePacket(
        sequence=seq,
        captured_ns=captured_ns,
        rgb=np.full((16, 16, 3), 150, dtype=np.uint8),
    )


def _face_frames(n: int, *, first_ns: int = 0, step_ns: int = 200_000_000):
    return [_packet(i, first_ns + (i - 1) * step_ns) for i in range(1, n + 1)]


def _matching_scorer(packet: FramePacket) -> FrameObservation:
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 52_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(0.0, 0.0, 2.0, 2.0),
        identity_scores={"enroll-23": 0.65, "enroll-10": 0.29},
        quality_rank=0.9,
        model_generation="gen-1",
        gallery_digest="digest-d2",
    )


class GatedScorer:
    """Scorer that blocks on an Event until released (fault injection).

    This is the D2 equivalent of test_pump_release_race.py's
    BlockingSource: a deterministic way to hold inference open without
    a real camera or a real wall-clock delay. While the gate is shut,
    the UI thread must remain responsive.
    """

    def __init__(self, inner: Any = None) -> None:
        self._inner = inner or _matching_scorer
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.calls = 0
        self.scoring_thread: str | None = None

    def __call__(self, packet: FramePacket) -> FrameObservation:
        self.calls += 1
        self.scoring_thread = threading.current_thread().name
        self.entered.set()
        # Bounded wait: a stuck gate must fail the test, not hang it.
        if not self.gate.wait(timeout=10.0):
            raise AssertionError("scorer gate never released; test is wedged")
        return self._inner(packet)

    def release(self) -> None:
        self.gate.set()


# ---------------------------------------------------------------------------
# Scope 1: inference runs off the UI thread
# ---------------------------------------------------------------------------


def test_inference_runs_on_a_worker_not_the_calling_thread() -> None:
    """D2 scope 1: the scorer must not run on the caller's thread.

    Called from the main thread (as the Qt timer callback does today);
    the scorer must observe a different thread.
    """
    scorer = GatedScorer()
    scorer.release()  # let it through; we only care about the thread
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine, FakeCapture(_face_frames(6)), scorer, preview_sink=lambda p: None
    )
    controller.start_session("d2-thread", 0, device_id="fake")

    caller = threading.current_thread().name
    controller.start_inference_worker()
    try:
        deadline = time.monotonic() + 5.0
        while scorer.calls == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert scorer.calls > 0, "worker never scored a frame"
        assert scorer.scoring_thread is not None
        assert scorer.scoring_thread != caller, (
            f"inference stayed on the caller thread {caller!r}; "
            f"scorer ran on {scorer.scoring_thread!r}"
        )
    finally:
        scorer.release()
        controller.close()


def test_preview_is_produced_by_the_worker_for_the_ui_to_drain() -> None:
    """D2 scope 1+5: the worker hands preview frames back, it does not paint.

    The preview channel is a bounded LatestSlot1Queue the UI drains —
    the worker must not touch a UI object.
    """
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine,
        FakeCapture(_face_frames(10)),
        _matching_scorer,
        preview_sink=lambda packet: None,
    )
    controller.start_session("d2-preview", 0, device_id="fake")
    controller.start_inference_worker()
    try:
        terminal = controller.wait_for_terminal(timeout_s=10.0)
        assert terminal is not None
        # Something reached the preview channel for the UI to paint.
        assert controller.preview_queue.depth > 0, (
            "worker produced no preview frame; the UI would stay blank"
        )
    finally:
        controller.close()


# ---------------------------------------------------------------------------
# Scope 2: the preview channel stays bounded (latest-frame, never a backlog)
# ---------------------------------------------------------------------------


def test_preview_channel_is_bounded_and_keeps_only_the_latest() -> None:
    """D2 scope 2: a slow UI must not accumulate a frame backlog.

    The camera produces far more frames than the UI can paint; the
    channel must stay at depth <= 1 and count the drops, exactly like
    the capture queue.
    """
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine,
        FakeCapture(_face_frames(200)),
        _matching_scorer,
        preview_sink=lambda packet: None,
    )
    controller.start_session("d2-bounded", 0, device_id="fake")
    controller.start_inference_worker()
    try:
        controller.wait_for_terminal(timeout_s=15.0)
        queue = controller.preview_queue
        assert queue.depth <= 1, f"preview channel grew to {queue.depth} frames"
        assert queue.dropped > 0, (
            "the UI never drained and nothing was dropped; the channel "
            "is not actually latest-only"
        )
    finally:
        controller.close()


# ---------------------------------------------------------------------------
# Scope 3 + 7: lifecycle — bounded exit, no worker accumulation, no leak
# ---------------------------------------------------------------------------


def test_worker_exits_in_bounded_time_on_close() -> None:
    """D2 scope 3/7: close() must not wait forever on an inference worker.

    Without a bounded exit, closing the window would freeze the UI for
    the full join timeout on every round.
    """
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine,
        FakeCapture(_face_frames(200)),
        _matching_scorer,
        preview_sink=lambda packet: None,
    )
    controller.start_session("d2-bounded-exit", 0, device_id="fake")
    controller.start_inference_worker()

    started = time.monotonic()
    controller.close()
    elapsed = time.monotonic() - started
    assert controller.workers_joined, "an inference worker survived close()"
    assert elapsed < 5.0, f"close() blocked for {elapsed:.1f}s"


def test_close_releases_the_camera_source() -> None:
    """D2 scope 3/7: the camera must never stay occupied (lead's top concern)."""
    source = FakeCapture(_face_frames(10))
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine, source, _matching_scorer, preview_sink=lambda p: None
    )
    controller.start_session("d2-release", 0, device_id="fake")
    controller.start_inference_worker()
    controller.wait_for_terminal(timeout_s=10.0)
    controller.close()
    assert source.is_closed, "camera source left open after close()"


def test_repeated_rounds_do_not_accumulate_workers() -> None:
    """D2 scope 7: ten rounds must not leave ten threads behind."""
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine,
        FakeCapture(_face_frames(6)),
        _matching_scorer,
        preview_sink=lambda packet: None,
    )
    baseline = len(threading.enumerate())
    for round_index in range(10):
        controller.start_session(f"d2-accum-{round_index}", 0, device_id="fake")
        controller.start_inference_worker()
        controller.wait_for_terminal(timeout_s=10.0)
        controller.close()
        controller = LiveController(
            SessionEngine(_profile(), "digest-d2", "gen-1"),
            FakeCapture(_face_frames(6)),
            _matching_scorer,
            preview_sink=lambda packet: None,
        )
    after = len(threading.enumerate())
    assert after <= baseline + 1, (
        f"threads grew {baseline} -> {after}; workers are accumulating"
    )


# ---------------------------------------------------------------------------
# Scope 7: the UI thread is not blocked by inference (fault injection)
# ---------------------------------------------------------------------------


def test_ui_thread_progresses_while_inference_is_blocked() -> None:
    """D2 scope 7: a blocked scorer must not freeze the UI thread.

    This is the fault-injection acceptance: while the scorer is held
    open on the worker, the UI thread must still be able to run its
    tick work. It deliberately does NOT claim real Qt event-loop
    concurrency — no repo test drives exec()/processEvents() — that is a
    D4 on-device measurement.
    """
    scorer = GatedScorer()
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine,
        FakeCapture(_face_frames(30)),
        scorer,
        preview_sink=lambda packet: None,
    )
    controller.start_session("d2-ui-free", 0, device_id="fake")
    controller.start_inference_worker()
    try:
        assert scorer.entered.wait(timeout=5.0), "worker never entered the scorer"

        # The UI thread is free to do its own work while the worker is
        # parked inside the scorer. Simulate the timer tick's work.
        ticks = 0
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            controller.current_thread_can_read_state()
            ticks += 1
            time.sleep(0.005)
        assert ticks > 10, f"UI thread managed only {ticks} ticks while blocked"
        # Cancel must remain callable from the UI thread mid-inference.
        assert controller.state_is_running()
        controller.cancel_inference()
    finally:
        scorer.release()
        controller.close()


def test_cancel_during_inference_terminates_the_round() -> None:
    """D2 scope 7: cancel must complete even mid-inference."""
    scorer = GatedScorer()
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine,
        FakeCapture(_face_frames(30)),
        scorer,
        preview_sink=lambda packet: None,
    )
    controller.start_session("d2-cancel-mid", 0, device_id="fake")
    controller.start_inference_worker()
    try:
        assert scorer.entered.wait(timeout=5.0)
        controller.cancel_inference()
        scorer.release()
        terminal = controller.wait_for_terminal(timeout_s=10.0)
        assert terminal is not None
        assert terminal.status == SessionStatus.cancelled
    finally:
        controller.close()


# ---------------------------------------------------------------------------
# Scope 5: the preview does not depend on research persistence
# ---------------------------------------------------------------------------


def test_preview_survives_a_failing_research_sink() -> None:
    """D2 scope 5: preview updates must not wait on encrypted disk writes.

    A research sink that raises or is slow must not stop preview frames
    from reaching the UI — that is the whole point of splitting the two
    paths.
    """
    def _boom(packet: FramePacket) -> None:
        raise RuntimeError("recorder disk is unavailable")

    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine,
        FakeCapture(_face_frames(10)),
        _matching_scorer,
        research_sink=_boom,
        preview_sink=lambda packet: None,
    )
    controller.start_session("d2-preview-independent", 0, device_id="fake")
    controller.start_inference_worker()
    try:
        controller.wait_for_terminal(timeout_s=10.0)
        assert controller.preview_queue.depth > 0, (
            "a failing research sink stopped the preview entirely"
        )
    finally:
        controller.close()


def test_closed_source_does_not_wait_out_the_jitter_bar() -> None:
    """D2 scope 3: a closed source ends the round, and this test says how
    far it can and cannot prove that.

    Mutation-tested twice, and both attempts failed to discriminate:

    - Relaxing test_qt_window.py's `_consecutive_dry == 1` to `>= 1` and
      then reverting the closed-source short-circuit left every assertion
      green. The counter is now worker-maintained, so the UI sees a
      settled value.
    - Asserting elapsed time instead also passed against the mutation:
      FakeCapture's closed read returns None immediately, so all three
      jitter-bar iterations complete in microseconds. The ~100 ms cost of
      the bar only exists at a real camera's 30 fps cadence, which this
      repo has no way to reproduce (no test drives a real capture source).

    So the timing claim is NOT proven here and no fake protection is left
    standing. What this does pin is the behaviour that is observable: a
    closed source ends the round as a terminal instead of hanging or
    being retried. The short-circuit itself remains verified by reading
    the diff and is a D4 on-device item alongside the rest of the
    camera-cadence questions.
    """
    class _ClosedSource(FakeCapture):
        def __init__(self, frames: list[FramePacket]) -> None:
            super().__init__(frames)
            self.closed_flag = False

        def close(self) -> None:
            self.closed_flag = True
            super().close()

        @property
        def is_closed(self) -> bool:
            return self.closed_flag

    source = _ClosedSource(_face_frames(20))
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine, source, _matching_scorer, preview_sink=lambda p: None
    )
    controller.start_session("d2-closed", 0, device_id="fake")
    controller.start_inference_worker()
    source.closed_flag = True  # the camera went away before any frame

    started = time.monotonic()
    terminal = controller.wait_for_terminal(timeout_s=10.0)
    elapsed = time.monotonic() - started
    try:
        assert terminal is not None, "a closed source never ended the round"
        # Bounded and prompt: the round must not hang or spin.
        assert elapsed < 1.0, f"closed source took {elapsed*1000:.0f}ms to end"
    finally:
        controller.close()


def test_failing_research_sink_does_not_kill_the_inference_worker() -> None:
    """D2 scope 5: a disk failure degrades staging, it does not stop inference.

    Found while implementing: an uncaught research-sink exception killed
    the worker thread outright, so the camera stayed open, the preview
    froze, and no terminal was ever produced. A side effect must not be
    able to end a round.
    """
    def _boom(packet: FramePacket) -> None:
        raise RuntimeError("recorder disk is unavailable")

    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine,
        FakeCapture(_face_frames(10)),
        _matching_scorer,
        research_sink=_boom,
        preview_sink=lambda packet: None,
    )
    controller.start_session("d2-sink-failure", 0, device_id="fake")
    controller.start_inference_worker()
    try:
        terminal = controller.wait_for_terminal(timeout_s=10.0)
        assert terminal is not None, (
            "the worker died with the sink; the round never terminated"
        )
        assert controller.research_sink_errors, (
            "a contained staging failure must still be recorded"
        )
        # Inference continued past the failing frames.
        assert terminal.frames_sampled > 1
    finally:
        controller.close()


def test_slow_research_sink_does_not_delay_preview_frames() -> None:
    """D2 scope 5: preview frames must appear before slow disk work ends.

    The research sink sleeps 200 ms per frame; the UI must already be
    able to see preview frames while that is still running.
    """
    def _slow(packet: FramePacket) -> None:
        time.sleep(0.2)

    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine,
        FakeCapture(_face_frames(30)),
        _matching_scorer,
        research_sink=_slow,
        preview_sink=lambda packet: None,
    )
    controller.start_session("d2-slow-disk", 0, device_id="fake")
    controller.start_inference_worker()
    try:
        # The first preview frame must be available well before the
        # 200 ms-per-frame research sink could possibly have finished.
        deadline = time.monotonic() + 3.0
        while controller.preview_queue.depth == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert controller.preview_queue.depth > 0, (
            "preview waited for the slow research sink"
        )
    finally:
        controller.close()


# ---------------------------------------------------------------------------
# Negative: D1 timing must be untouched, and no unbounded queue may appear
# ---------------------------------------------------------------------------


def test_d1_first_frame_anchor_still_arms_under_the_worker() -> None:
    """D1 semantics must survive workerization (lead will check this).

    The recognition window still arms at the first frame the camera
    delivers, and the countdown clamp still depends on that flag.
    """
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine,
        FakeCapture(_face_frames(8, first_ns=2_000_000_000)),
        _matching_scorer,
        preview_sink=lambda packet: None,
    )
    controller.start_session("d2-anchor", 0, device_id="fake")
    controller.start_inference_worker()
    try:
        controller.wait_for_terminal(timeout_s=10.0)
        assert controller.recognition_anchored, (
            "the first-frame anchor never fired under the worker"
        )
        assert engine.recognition_anchored
    finally:
        controller.close()


def test_preview_channel_never_holds_more_than_one_frame() -> None:
    """Anti-regression: the channel is slot-1, not an accumulating queue."""
    queue = LatestSlot1Queue[FramePacket]()
    for i in range(50):
        queue.push(_packet(i + 1, i * 1_000_000))
    assert queue.depth <= 1
    assert queue.dropped >= 49


# ---------------------------------------------------------------------------
# The opt-in wiring: only a real event loop gets worker-driven inference
# ---------------------------------------------------------------------------


def _gui_window(source: Any, **overrides: Any) -> Any:
    """A window configured the way the real GUI path configures it."""
    from datetime import datetime, timedelta, timezone

    from facecore.live.desktop import DesktopSession
    from facecore.research.records import ConsentRecord

    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    desktop = DesktopSession(
        engine=engine,
        source=source,
        scorer=_matching_scorer,
        session_id="d2-gui",
        release_source_on_terminal=False,
    )
    now = datetime.now(timezone.utc)
    consent = ConsentRecord(
        session_id="d2-gui",
        participant_id="synthetic-01",
        record_consent=True,
        image_consent=True,
        consented_at_utc=now.isoformat(),
        record_expires_at_utc=(now + timedelta(days=1)).isoformat(),
        image_expires_at_utc=(now + timedelta(days=1)).isoformat(),
    )
    from facecore.live.qt_window import QtResearchWindow

    return QtResearchWindow(
        desktop, consent=consent, offscreen=True,
        background_inference=True, **overrides,
    )


def test_real_gui_path_starts_an_inference_worker(qt_app: Any) -> None:
    """D2 scope 1: the GUI path is the one that gets the worker.

    The CLI passes background_inference=not qt_offscreen, so a real event
    loop no longer runs inference inside its timer callback.
    """
    window = _gui_window(FakeCapture(_face_frames(20)))
    try:
        window.record_consent_checkbox.setChecked(True)
        window.image_consent_checkbox.setChecked(True)
        window.start_clicked()
        # A short source means the round may already be over; what matters
        # is that a worker was created and tracked at all, not that it is
        # still running by the time we look.
        thread = window.desktop._controller._inference_thread
        assert thread is not None, (
            "the GUI path started a round without an inference worker"
        )
        assert thread.name == "d2-inference"
        assert thread in window.desktop._controller.tracked_threads, (
            "the worker must be tracked so close() joins it"
        )
    finally:
        window.close()
    assert window.desktop._controller.workers_joined


def test_offscreen_driver_keeps_the_synchronous_tick(qt_app: Any) -> None:
    """D2: the deterministic driver must stay synchronous.

    Flipping the worker on for every path would turn every existing
    timing assertion into a race, so the opt-in is load-bearing.
    """
    from datetime import datetime, timedelta, timezone

    from facecore.live.desktop import DesktopSession
    from facecore.live.qt_window import QtResearchWindow
    from facecore.research.records import ConsentRecord

    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    desktop = DesktopSession(
        engine=engine,
        source=FakeCapture(_face_frames(6)),
        scorer=_matching_scorer,
        session_id="d2-offscreen",
    )
    now = datetime.now(timezone.utc)
    consent = ConsentRecord(
        session_id="d2-offscreen",
        participant_id="synthetic-01",
        record_consent=True,
        image_consent=True,
        consented_at_utc=now.isoformat(),
        record_expires_at_utc=(now + timedelta(days=1)).isoformat(),
        image_expires_at_utc=(now + timedelta(days=1)).isoformat(),
    )
    window = QtResearchWindow(
        desktop, consent=consent, offscreen=True, background_inference=False
    )
    try:
        window.record_consent_checkbox.setChecked(True)
        window.image_consent_checkbox.setChecked(True)
        window.start_clicked()
        assert desktop._controller._inference_thread is None, (
            "the synchronous driver must not spawn a worker"
        )
        window.process_until_terminal()
        assert desktop.terminal is not None, (
            "the synchronous driver must still reach a terminal"
        )
    finally:
        window.close()


@pytest.fixture(scope="module")
def qt_app() -> Any:
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication as ActualQApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return ActualQApplication.instance() or ActualQApplication([])
