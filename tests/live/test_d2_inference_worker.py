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

from facecore.live.capture import CaptureSource, FakeCapture, LatestSlot1Queue
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


class BlockingSource(CaptureSource):
    """read() parks until released; records close-during-read per thread.

    D2's third fault-injection double, alongside GatedScorer (blocked
    inference) and FakeCapture (instant reads). FakeCapture can never
    block, so a close-during-read regression is invisible to it; this one
    makes the failure deterministic without a camera, the same way
    test_pump_release_race.py does for issue #64.
    """

    def __init__(self) -> None:
        self._closed = True
        self._opened = False
        self._in_read = 0
        self._readers: set[str] = set()
        self._gate = threading.Event()
        self._close_while_reading = False
        self._close_while: dict[str, bool] = {}
        self._lock = threading.Lock()
        self._seq = 0

    def open(self, device_id: str) -> None:
        self._closed = False
        self._opened = True

    def read(self) -> FramePacket | None:
        with self._lock:
            if self._closed or not self._opened:
                return None
            self._in_read += 1
            self._readers.add(threading.current_thread().name)
        if not self._gate.wait(timeout=10):
            with self._lock:
                self._in_read -= 1
            return None
        with self._lock:
            self._in_read -= 1
            self._readers.discard(threading.current_thread().name)
            if self._closed:
                return None
            self._seq += 1
            return FramePacket(
                sequence=self._seq,
                captured_ns=self._seq * 200_000_000,
                rgb=np.zeros((16, 16, 3), dtype=np.uint8),
            )

    def close(self) -> None:
        with self._lock:
            if self._in_read > 0:
                self._close_while_reading = True
                for name in self._readers:
                    self._close_while[name] = True
            self._closed = True
        self._gate.set()

    def arm(self) -> None:
        """Make the next read park again (the first close consumed the gate)."""
        self._gate.clear()

    def wait_until_reading(self, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while self.in_read == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        return self.in_read > 0

    def wait_closed(self, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while not self.is_closed and time.monotonic() < deadline:
            time.sleep(0.01)
        return self.is_closed

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def in_read(self) -> int:
        with self._lock:
            return self._in_read

    @property
    def close_while_reading(self) -> bool:
        with self._lock:
            return self._close_while_reading

    @property
    def close_while(self) -> dict[str, bool]:
        with self._lock:
            return dict(self._close_while)


# ---------------------------------------------------------------------------
# B1 (reviewer blocking): close() must never release the source while a
# worker is inside a native read. Both workers carry the issue #64
# signature; closeEvent -> close() reaches this on the real GUI path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("thread_name", ["d2-inference", "t4-capture-pump"])
def test_close_never_releases_the_source_while_a_worker_is_reading(
    thread_name: str,
) -> None:
    """B1: the join-timeout release is the bug, not the join timeout.

    The worker's read is parked longer than the 5 s join budget, so
    close() proceeds to its release step with that worker still inside
    read(). close() must then SKIP the release and let the worker's own
    exit release instead — on AVFoundation releasing under an in-flight
    native read segfaults (issue #64).

    Mutation-tested: reverting the joined guard (and the per-worker
    finally) turns this red. FakeCapture can never block, which is
    exactly why this double exists.
    """
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    source = BlockingSource()
    controller = LiveController(engine, source, _matching_scorer)
    controller.start_session("d2-close-race", 0, device_id="fake")
    if thread_name == "t4-capture-pump":
        controller.start_background_pump()
    else:
        controller.start_inference_worker()
    try:
        assert source.wait_until_reading(), f"{thread_name} never entered read()"
        source.arm()  # close() will consume this gate
        started = time.monotonic()
        controller.close()
        elapsed = time.monotonic() - started
        assert source.close_while_reading is False, (
            f"close() released the source while {thread_name} was still "
            "reading — close-during-read segfaults AVFoundation (issue #64)"
        )
        # Bounded: close() must not wait forever for a read that never
        # returns. (The budget itself is not what's under test.)
        assert elapsed < 15.0, f"close() took {elapsed:.1f}s"
        assert controller.workers_joined is False, (
            "the read is still parked; the worker cannot have exited yet"
        )
    finally:
        # Let the parked read return so the worker runs its own exit path.
        source._gate.set()
    assert source.wait_closed(timeout_s=5.0), (
        "no worker released the source on exit — the camera is leaked"
    )
    assert controller.workers_joined is True, "the worker never exited"


def test_close_with_both_workers_reading_still_skips_the_release() -> None:
    """B1: the two #64 workers must guard each other, not just the caller.

    Found by the B1 probe after the first fix went green. With a capture
    pump AND an inference worker alive, close() skips its own release
    correctly — but the inference worker then hit the join timeout and
    ran its own exit release WHILE the pump was still parked in a read.
    The caller's guard was not enough: each worker's exit path has to
    defer to a live sibling too.

    This is the shape the GUI can actually reach (headless drivers start
    a pump and never an inference worker, so the parametrized cases above
    cannot see it).
    """
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    source = BlockingSource()
    controller = LiveController(engine, source, _matching_scorer)
    controller.start_session("d2-both-workers", 0, device_id="fake")
    controller.start_background_pump()
    controller.start_inference_worker()
    try:
        assert source.wait_until_reading(), "no worker entered read()"
        source.arm()  # the first close() consumes this gate
        controller.close()
        assert source.close_while_reading is False, (
            "close released while a worker was still reading: "
            f"{source.close_while}"
        )
    finally:
        source._gate.set()
    # The last worker out must still release — the guard must defer,
    # never drop, the release.
    assert source.wait_closed(timeout_s=10.0), (
        "no worker released the source on exit — the camera is leaked"
    )


def test_close_stops_the_worker_it_cannot_make_stop_itself() -> None:
    """B1/N1: close() must tell the inference worker to stop.

    close() used to set only _pump_stop while the inference worker polls
    _inference_stop. With a healthy scorer the worker reached its
    terminal on its own, so the bug was invisible.

    Two things are arranged here so the stop signal is the ONLY way out:

    - The scorer always rejects, so B never locks and the worker has no
      terminal of its own to finish on. (A matching scorer locks B at
      the 3rd frame and exits immediately, which is why the obvious
      version of this test passed in 0.18 s under the mutation.)
    - Scoring costs 200 ms per frame, so even with 26 frames queued the
      worker cannot get through the round inside the 5 s join budget.
      The stop flag is polled between frames, so it ends the worker
      after the frame in flight.

    Mutation-tested: removing `self._inference_stop.set()` from close()
    leaves the worker running, `workers_joined` False.
    """
    class _SlowRejectingScorer:
        """Always rejects, slowly: no terminal, and no fast exit."""

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, packet: FramePacket) -> FrameObservation:
            self.calls += 1
            time.sleep(0.2)
            return FrameObservation(
                sequence=packet.sequence,
                captured_ns=packet.captured_ns,
                processed_ns=packet.captured_ns + 200_000_000,
                quality_pass=True,
                quality_reasons=(),
                face_count=1,
                face_box=(0.0, 0.0, 2.0, 2.0),
                identity_scores={"enroll-23": 0.10},
                quality_rank=0.9,
                model_generation="gen-1",
                gallery_digest="digest-d2",
            )

    scorer = _SlowRejectingScorer()
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(
        engine, FakeCapture(_face_frames(400)), scorer
    )
    controller.start_session("d2-close-stops-worker", 0, device_id="fake")
    controller.start_inference_worker()
    try:
        deadline = time.monotonic() + 5.0
        while scorer.calls < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert scorer.calls >= 1, "the worker never started scoring"
        controller.close()
        assert controller.workers_joined is True, (
            "close() did not stop the inference worker: the flag it sets "
            "is not the one the worker polls, so the join fell back to "
            "its 5 s timeout with the worker still running"
        )
    finally:
        controller.close()


def test_the_workers_own_exit_releases_the_source_the_caller_skipped() -> None:
    """B1: the skipped release is a handoff, not a leak.

    close() may not release while a read is in flight, so the source
    stays open at that moment. It must NOT stay open afterwards: the
    worker releases it on its own way out. Asserted after the read is
    released, so it is not the "close() did it" tautology.
    """
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    source = BlockingSource()
    controller = LiveController(engine, source, _matching_scorer)
    controller.start_session("d2-close-handoff", 0, device_id="fake")
    controller.start_inference_worker()
    try:
        assert source.wait_until_reading(), "the worker never entered read()"
        source.arm()
        controller.close()
        assert source.is_closed is False, (
            "close() must skip the release while a read is in flight"
        )
    finally:
        source._gate.set()
    assert source.wait_closed(timeout_s=5.0), (
        "the source was never released: close() skipped it and no worker "
        "took over — that is a leaked camera"
    )


def test_close_releases_normally_when_no_read_is_in_flight() -> None:
    """B1 must not cost the ordinary case its release.

    A worker that has already exited leaves nothing to protect, so
    close() releases immediately — the guard must not turn every close
    into "skip and wait for a worker that has already gone".
    """
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    source = BlockingSource()
    controller = LiveController(engine, source, _matching_scorer)
    controller.start_session("d2-close-clean", 0, device_id="fake")
    controller.start_inference_worker()
    source._gate.set()  # reads return immediately; the round runs to terminal
    assert controller.wait_for_terminal(timeout_s=10.0) is not None
    controller.close()
    assert source.is_closed, "close() left the camera open after a clean round"
    assert source.close_while_reading is False


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
        engine, FakeCapture(_face_frames(6)), scorer
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
        engine, source, _matching_scorer
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
        )
    after = len(threading.enumerate())
    assert after <= baseline + 1, (
        f"threads grew {baseline} -> {after}; workers are accumulating"
    )


# ---------------------------------------------------------------------------
# Scope 7: the UI thread is not blocked by inference (fault injection)
# ---------------------------------------------------------------------------


def test_ui_thread_stays_responsive_while_inference_is_blocked() -> None:
    """D2 scope 7: a blocked scorer must not freeze the UI thread.

    The fault-injection acceptance: while the scorer is held open on the
    worker, the UI thread must still be able to run its tick work. It
    deliberately does NOT claim real Qt event-loop concurrency — no repo
    test drives exec()/processEvents() — that is a D4 on-device
    measurement.

    RED for the reviewer's S2. The previous version called
    `controller.current_thread_can_read_state()` in a loop without
    checking the return value, and that method is
    `return self._terminal is None or self._terminal is not None` — a
    tautology. It only ever measured "the main thread can run a while
    loop", so it stayed green even when a mutation held `self._lock`
    across the scorer call (a real UI freeze). The helper was also
    deleted: a method that cannot fail is not a check.

    What is measured instead is UI work that genuinely takes
    `self._lock` — the same lock an inference-side freeze would hold —
    with each call bounded so a real freeze fails the test on elapsed
    time rather than hanging it.
    """
    scorer = GatedScorer()
    engine = SessionEngine(_profile(), "digest-d2", "gen-1")
    controller = LiveController(engine, FakeCapture(_face_frames(30)), scorer)
    controller.start_session("d2-ui-free", 0, device_id="fake")
    controller.start_inference_worker()
    try:
        assert scorer.entered.wait(timeout=5.0), "worker never entered the scorer"

        # The UI tick's real reads. wait_for_terminal takes the lock and
        # workers_joined reads the tracked-thread list under it; both are
        # what a scorer-side freeze would block. timeout_s=0 makes each
        # call a poll instead of a wait, so a contended lock shows up as
        # elapsed time on a probe below rather than as a hang here.
        polls = 0
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            controller.wait_for_terminal(timeout_s=0)
            polls += 1
            assert controller.workers_joined is False, (
                "the worker is parked inside the scorer; it cannot be joined"
            )
            assert controller.state_is_running(), "the round stopped on its own"
            assert controller.frames_sampled >= 1, (
                "the worker never sampled a frame the UI could read"
            )
            time.sleep(0.005)
        assert polls > 10, f"UI thread managed only {polls} polls while blocked"

        # The lock a freeze would hold, probed the same way. RLock by
        # construction (a worker may re-enter from a worker-side call),
        # so "acquired" cannot be confused with "free".
        probe_lock = controller._lock
        started = time.monotonic()
        acquired = probe_lock.acquire(timeout=1.0)
        probe_elapsed = time.monotonic() - started
        if acquired:
            probe_lock.release()
        assert acquired, (
            f"the UI thread could not take the controller lock in "
            f"{probe_elapsed:.2f}s while a scorer was parked — the UI is "
            "blocked behind the worker"
        )
        assert probe_elapsed < 0.5, (
            f"the controller lock took {probe_elapsed:.2f}s; the worker is "
            "holding it across inference, which freezes the UI"
        )

        # Cancel must remain callable from the UI thread mid-inference.
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


def test_closed_source_terminates_on_the_first_dry_read() -> None:
    """D2 scope 3: a source that goes away ends the round immediately.

    RED for the reviewer's S1: the previous version of this test
    (b43186d) asserted `elapsed < 1.0` after setting closed_flag right
    after start_inference_worker(). The worker had already matched 3
    frames off FakeCapture and returned, so the closed-source branch
    never ran, the assertion measured how fast 3 frames matched, and the
    test stayed green even when a closed source never terminated at all.

    This version cannot make that mistake. The source is closed BEFORE
    the worker starts, so no frame can ever match; the branch under test
    is the only path to a terminal. And the fast path is asserted
    structurally — read_count == 1 — instead of by elapsed time, so it
    discriminates regardless of machine speed.

    Honest limit (unchanged, and it is a real one): this proves the
    branch ends the round after one dry read. It does NOT measure what
    the ~100 ms jitter bar costs at a real camera's 30 fps cadence,
    which only exists with real hardware — that part stays a D4
    on-device item. What is now genuinely pinned is the part this
    environment CAN pin: a closed source cannot hang, and cannot burn
    further reads.
    """
    class _DrySource(CaptureSource):
        """Closed source whose read costs time, so a read count is visible."""

        def __init__(self, per_read_s: float) -> None:
            self._closed = True
            self.read_count = 0
            self._per_read_s = per_read_s

        def open(self, device_id: str) -> None:
            self._closed = False

        def read(self) -> FramePacket | None:
            time.sleep(self._per_read_s)
            self.read_count += 1
            return None

        def close(self) -> None:
            self._closed = True

        @property
        def is_closed(self) -> bool:
            return self._closed

    # required_support far above anything this source can deliver: the
    # engine must never lock B, so "matched" is not reachable at all.
    profile = _profile(required_support=99)
    engine = SessionEngine(profile, "digest-d2", "gen-1")
    source = _DrySource(per_read_s=0.05)
    controller = LiveController(engine, source, _matching_scorer)
    controller.start_session("d2-closed", 0, device_id="fake")
    source.close()  # the camera went away before the worker ever read
    controller.start_inference_worker()
    try:
        started = time.monotonic()
        terminal = controller.wait_for_terminal(timeout_s=10.0)
        elapsed = time.monotonic() - started
        assert terminal is not None, (
            "a closed source never ended the round — the round hangs"
        )
        assert source.read_count == 1, (
            f"closed source took {source.read_count} dry reads; the fast "
            "path must conclude on the first one (the 3-read jitter bar is "
            "only for a live source that briefly reads empty)"
        )
        assert elapsed < 1.0, f"closed source took {elapsed*1000:.0f}ms to end"
        assert controller.frames_sampled == 0
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
