"""Fix #81: headless fixed-window must reach deadline via caller repetition.

Root cause: ``cli.py:602`` calls ``desktop.run_until_terminal(max_steps=50)``
once.  A real camera at ~30 fps produces ~6 gated frames per 200 ms sample,
consuming ~150 steps for a 5 s window — but only 50 are available.  The step
budget exhausts first, and ``_finalize_collection`` stamps
``source_exhausted / incomplete``.

Qt already works because ``process_once`` (timer tick) calls
``run_until_terminal(50)`` repeatedly.  The fix mirrors this for headless.

Camera contract: ``_RealtimeCamera`` paces at ``1/fps`` per read using a real
``time.sleep``, so ``captured_ns = time.monotonic_ns()`` naturally advances and
the ``processed_ns >= captured_ns`` contract in ``FrameObservation`` holds
without synthetic clocks.
"""

from __future__ import annotations

import threading

import numpy as np

from facecore.live.capture import CaptureSource, FakeCapture
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionStatus,
)
from facecore.live.desktop import DesktopSession
from facecore.live.session import SessionEngine
from facecore.research.diagnostics import (
    FrameTraceEntry,
    SessionTrace,
)
from facecore.research.records import CollectionWindow, ConsentRecord
from facecore.research.replay import evaluate_arms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class DeterministicCadenceCamera(CaptureSource):
    """Paces at 1/fps synthetically without wall-clock sleep.

    Latency-independent: advances captured_ns by 1/fps (in nanoseconds)
    per read, starting from start_ns. Zero time.sleep(), completely
    independent of host scheduler jitter or CI runner load.
    Counts reads to allow asserting exact step budget bounds.
    """

    def __init__(
        self,
        fps: int = 30,
        max_frames: int = 10_000,
        start_ns: int = 0,
    ) -> None:
        self.fps = fps
        self.frame_interval_ns = 1_000_000_000 // fps
        self._max = max_frames
        self._start_ns = start_ns
        self._seq = 0
        self.read_count = 0
        self._opened = False
        self._closed = False
        self._lock = threading.Lock()

    def open(self, device_id: str) -> None:
        with self._lock:
            self._seq = 0
            self.read_count = 0
            self._opened = True
            self._closed = False

    def read(self) -> FramePacket | None:
        with self._lock:
            self.read_count += 1
            if self._closed or not self._opened or self._seq >= self._max:
                return None
            self._seq += 1
            captured_ns = self._start_ns + (self._seq - 1) * self.frame_interval_ns
            rgb = np.full((200, 200, 3), 120, dtype=np.uint8)
            return FramePacket(
                sequence=self._seq,
                captured_ns=captured_ns,
                rgb=rgb,
            )

    def close(self) -> None:
        with self._lock:
            self._opened = False
            self._closed = True

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed


# Backwards compatibility alias for existing test references
_RealtimeCamera = DeterministicCadenceCamera


def _profile(
    timeout_ms: int = 5000,
    sample_interval_ms: int = 200,
    max_frames: int = 25,
) -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="fix81-v1",
        timeout_ms=timeout_ms,
        sample_interval_ms=sample_interval_ms,
        max_frames=max_frames,
        queue_limit=1,
        required_support=3,
        min_support_interval_ms=200,
        match_threshold=0.45,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="det-fix81",
        quality_policy_version="qual-fix81",
        continuity_max_center_delta_ratio=0.50,
    )


def _deadline_profile() -> ResearchProfile:
    """5s window with a frame cap that cannot fire before the deadline.

    Production profile (timeout_ms=5000, sample_interval_ms=200,
    max_frames=25) samples [0, 4800ms]: the cap fires before the deadline
    by design (see issue #84 — a profile-contract inconsistency that #81
    must not touch). Deadline-termination tests must therefore use 26+
    sample slots; a dedicated low-cap positive control below pins the
    max_frames_reached incomplete branch instead.
    """
    return _profile(max_frames=26)


def _matched_scorer(packet: FramePacket) -> FrameObservation:
    """Every frame is quality-pass with a strong person-01 match."""
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(20.0, 20.0, 120.0, 120.0),
        identity_scores={"person-01": 0.85, "person-02": 0.20},
        quality_rank=0.5,
        model_generation="gen-fix81",
        gallery_digest="gal-fix81",
    )


class _TraceCollector:
    """Minimal trace recorder mock: collects FrameTraceEntry in memory."""

    def __init__(self) -> None:
        self.entries: list[FrameTraceEntry] = []

    def append_trace(self, attempt_id: str, entry: FrameTraceEntry) -> None:
        self.entries.append(entry)


def _consent(session_id: str = "sess-fix81") -> ConsentRecord:
    return ConsentRecord(
        session_id=session_id,
        participant_id="part-synth",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-01-01T00:00:00+00:00",
        record_expires_at_utc="2026-12-31T23:59:59+00:00",
        image_expires_at_utc="2026-12-31T23:59:59+00:00",
    )


def _make_desktop(
    camera: CaptureSource,
    *,
    profile: ResearchProfile | None = None,
    trace_collector: _TraceCollector | None = None,
    start_ns: int | None = None,
) -> tuple[DesktopSession, int, ResearchProfile]:
    if profile is None:
        profile = _profile()
    if start_ns is None:
        start_ns = 1_000_000_000_000
    if hasattr(camera, "_start_ns"):
        camera._start_ns = start_ns
    engine = SessionEngine(profile, "gal-fix81", "gen-fix81")
    desktop = DesktopSession(
        engine=engine,
        source=camera,
        scorer=_matched_scorer,
        session_id="sess-fix81",
        sample_interval_ns=int(profile.sample_interval_ms * 1_000_000),
        max_frames=profile.max_frames,
        fixed_seconds=True,
        trace_recorder=trace_collector,
        trace_attempt_id="att-fix81" if trace_collector is not None else None,
    )
    desktop.on_start(_consent(), now_ns=start_ns, device_id="0")
    return desktop, start_ns, profile


def _headless_loop(
    desktop: DesktopSession,
    max_steps: int = 50,
) -> int:
    """Exact mirror of production cli.py:602-604 caller loop.

    Zero wall-clock escape hatches. Returns the total call count to prove
    termination is deterministic, bounded, and state-driven.
    """
    calls = 0
    while desktop.state == "running":
        desktop.run_until_terminal(max_steps=max_steps)
        calls += 1
    return calls


def _build_window(
    ctrl: object, start_ns: int, profile: ResearchProfile
) -> CollectionWindow:
    deadline_ns = start_ns + int(profile.timeout_ms * 1_000_000)
    return CollectionWindow(
        session_id="sess-fix81",
        collection_start_ns=start_ns,
        collection_deadline_ns=deadline_ns,
        collection_end_ns=ctrl._last_sampled_ns(),  # type: ignore[attr-defined]
        collection_stop_reason=ctrl.collection_stop_reason,  # type: ignore[attr-defined]
        collection_complete=ctrl.collection_complete,  # type: ignore[attr-defined]
        frames_sampled=ctrl.frames_sampled,  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHeadlessFixedWindowDeadline:
    """Fix #81: headless caller must repeat run_until_terminal."""

    def test_headless_30fps_reaches_deadline(self) -> None:
        """Caller loop with 30 fps source reaches deadline_reached."""
        tc = _TraceCollector()
        camera = _RealtimeCamera(fps=30)
        desktop, start_ns, profile = _make_desktop(
            camera, profile=_deadline_profile(), trace_collector=tc
        )

        calls = _headless_loop(desktop)

        ctrl = desktop._controller
        assert ctrl.collection_stop_reason == "deadline_reached"
        assert ctrl.collection_complete is True
        assert desktop.state == "terminal"
        assert ctrl.frames_sampled >= 20
        assert calls >= 2

        window = _build_window(ctrl, start_ns, profile)
        trace = SessionTrace(
            schema_version="v2",
            attempt_id="att-fix81",
            manifest_digest=profile.profile_digest(),
            session_start_ns=start_ns,
            deadline_ns=start_ns + int(profile.timeout_ms * 1_000_000),
            session_end_ns=ctrl._last_sampled_ns(),
            collection_stop_reason=ctrl.collection_stop_reason,
            is_complete=ctrl.collection_complete,
            entries=tuple(tc.entries),
            terminal_result=desktop.terminal,
        )
        arm_a, arm_b = evaluate_arms(trace, profile, window=window)
        assert arm_a.collection_extent == "full"
        assert arm_b.collection_extent == "full"

    def test_headless_60fps_reaches_deadline(self) -> None:
        """Second cadence: 60 fps also reaches deadline — no magic constant."""
        camera = _RealtimeCamera(fps=60)
        desktop, start_ns, profile = _make_desktop(
            camera, profile=_deadline_profile()
        )

        calls = _headless_loop(desktop)

        ctrl = desktop._controller
        assert ctrl.collection_stop_reason == "deadline_reached"
        assert ctrl.collection_complete is True
        assert desktop.state == "terminal"
        assert calls >= 4

    def test_eof_before_deadline_still_incomplete(self) -> None:
        """Source truly runs out before deadline → source_exhausted.

        Proves that incomplete terminal exits caller loop without any
        wall-clock escape hatch (F1/F2 acceptance).
        """
        camera = _RealtimeCamera(fps=30, max_frames=10)
        desktop, start_ns, profile = _make_desktop(camera)

        calls = _headless_loop(desktop)

        ctrl = desktop._controller
        assert ctrl.collection_stop_reason == "source_exhausted"
        assert ctrl.collection_complete is False
        assert desktop.state == "terminal"
        assert calls <= 5

    def test_max_frames_before_deadline_still_incomplete(self) -> None:
        """Max frames cap reached before deadline → max_frames_reached.

        Proves that max_frames_reached incomplete terminal exits caller loop
        deterministically with state == terminal (F1 acceptance).
        """
        prof = _profile(timeout_ms=5000, sample_interval_ms=200, max_frames=5)
        camera = _RealtimeCamera(fps=30)
        desktop, start_ns, profile = _make_desktop(camera, profile=prof)

        calls = _headless_loop(desktop)

        ctrl = desktop._controller
        assert ctrl.collection_stop_reason == "max_frames_reached"
        assert ctrl.collection_complete is False
        assert desktop.state == "terminal"
        assert calls <= 10

    def test_single_call_bounded(self) -> None:
        """A single run_until_terminal(50) is bounded by the step budget.

        Proves the shared API contract: one call consumes at most 50
        pump-consume iterations (asserted via read_count <= 50) and then
        returns control to the caller while state remains running — the
        Qt timer tick model. Latency-independent: counts steps, not seconds.
        """
        camera = DeterministicCadenceCamera(fps=30)
        desktop, start_ns, profile = _make_desktop(camera)

        desktop.run_until_terminal(max_steps=50)

        assert camera.read_count <= 50
        assert desktop.state == "running"
        assert desktop._controller.collection_stop_reason == "in_progress"

    def test_production_step_budget_matches_qt_tick(self) -> None:
        """Headless and Qt share the same max_steps=50 caller budget.

        Guards the F1-adjacent invariant the bounded test relies on: if
        either caller ever diverges from 50, this fails at review time
        instead of silently changing tick granularity.
        """
        import inspect
        from pathlib import Path

        from facecore.research import cli as cli_module
        import facecore.live.qt_window as qt_module

        cli_src = inspect.getsource(cli_module.cmd_live)
        assert "run_until_terminal(max_steps=50)" in cli_src
        assert qt_module.__file__ is not None
        qt_src = Path(qt_module.__file__).read_text(encoding="utf-8")
        assert "run_until_terminal(max_steps=50)" in qt_src

    def test_cancel_pre_lock_stops_collection(self) -> None:
        """Deterministic pre-lock Cancel stops immediately with cancelled terminal."""
        camera = DeterministicCadenceCamera(fps=30)
        desktop, start_ns, profile = _make_desktop(camera)

        # Drive 1 step (1 sample; profile requires 3 supports to lock B).
        desktop.run_until_terminal(max_steps=1)
        assert desktop.state == "running"
        assert desktop.inference_terminal is None
        assert desktop._controller.collection_stop_reason == "in_progress"

        result = desktop.on_cancel(start_ns + 50_000_000)

        assert desktop.state == "terminal"
        assert result.status == SessionStatus.cancelled
        assert desktop.collection_stop_reason == "cancelled"
        assert desktop.collection_complete is False

        calls = _headless_loop(desktop)
        assert calls == 0

    def test_cancel_post_lock_stops_remaining_collection(self) -> None:
        """Post-lock Cancel stops remaining fixed-window collection."""
        camera = DeterministicCadenceCamera(fps=30)
        desktop, start_ns, profile = _make_desktop(camera)

        # Drive 20 steps (3 samples taken, locking B with matched status).
        desktop.run_until_terminal(max_steps=20)
        assert desktop.state == "running"
        assert desktop.inference_terminal is not None
        assert desktop.inference_terminal.status == SessionStatus.matched
        assert desktop._controller.collection_stop_reason == "in_progress"

        result = desktop.on_cancel(start_ns + 1_000_000_000)

        assert desktop.state == "terminal"
        assert result.status == SessionStatus.matched
        assert desktop.collection_stop_reason == "cancelled"
        assert desktop.collection_complete is False

        calls = _headless_loop(desktop)
        assert calls == 0

    def test_transient_dropped_frame_does_not_abort_collection(self) -> None:
        """N1: transient frame drop on an open camera does not abort collection."""

        class _FlakyCamera(DeterministicCadenceCamera):
            def __init__(self, fps: int = 30) -> None:
                super().__init__(fps=fps)
                self._dropped_seqs = {3, 10}

            def read(self) -> FramePacket | None:
                with self._lock:
                    if self._closed or not self._opened:
                        return None
                    self.read_count += 1
                    self._seq += 1
                    if self._seq in self._dropped_seqs:
                        return None
                    captured_ns = (
                        self._start_ns + (self._seq - 1) * self.frame_interval_ns
                    )
                    rgb = np.full((200, 200, 3), 120, dtype=np.uint8)
                    return FramePacket(
                        sequence=self._seq,
                        captured_ns=captured_ns,
                        rgb=rgb,
                    )

        camera = _FlakyCamera(fps=30)
        desktop, start_ns, profile = _make_desktop(
            camera, profile=_deadline_profile()
        )

        calls = _headless_loop(desktop)

        ctrl = desktop._controller
        assert ctrl.collection_stop_reason == "deadline_reached"
        assert ctrl.collection_complete is True
        assert desktop.state == "terminal"
        assert calls >= 2

    def test_gated_out_frame_drained_from_queue_resets_dry_counter(self) -> None:
        """B4: frames drained from queue but gated out must not count as dry.

        When camera read returns None (pump returns False), a packet drained
        from the slot-1 queue proves the source stream is active. Even if
        not sample_due (gated out), _consume_one returns a non-None consumed
        packet, resetting _consecutive_dry to 0.
        """
        camera = FakeCapture([])
        desktop, start_ns, profile = _make_desktop(camera)
        ctrl = desktop._controller

        # First sample at t=0
        p1 = FramePacket(
            sequence=1,
            captured_ns=start_ns,
            rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        ctrl._queue.push(p1)
        consumed, _ = ctrl._consume_one()
        assert consumed is not None
        assert ctrl.frames_sampled == 1

        # Second packet at t=50ms (< 200ms sample interval) is gated out.
        p2 = FramePacket(
            sequence=2,
            captured_ns=start_ns + 50_000_000,
            rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        ctrl._queue.push(p2)

        consumed2, _ = ctrl._consume_one()
        assert consumed2 is not None
        assert consumed2.sequence == 2
        assert ctrl.frames_sampled == 1

    def test_gated_out_queue_drain_prevents_premature_source_exhaustion(
        self,
    ) -> None:
        """B4: stream of gated-out queue packets prevents premature source_exhausted."""
        camera = FakeCapture([])
        desktop, start_ns, profile = _make_desktop(camera)
        ctrl = desktop._controller

        # Sample 1
        ctrl._queue.push(
            FramePacket(
                sequence=1,
                captured_ns=start_ns,
                rgb=np.zeros((10, 10, 3), dtype=np.uint8),
            )
        )
        desktop.run_until_terminal(max_steps=1)
        assert ctrl.frames_sampled == 1
        assert ctrl._consecutive_dry == 0

        # Feed 5 gated-out packets (< 200ms) one by one
        for i in range(2, 7):
            ctrl._queue.push(
                FramePacket(
                    sequence=i,
                    captured_ns=start_ns + (i - 1) * 20_000_000,
                    rgb=np.zeros((10, 10, 3), dtype=np.uint8),
                )
            )
            desktop.run_until_terminal(max_steps=1)
            assert ctrl._consecutive_dry == 0
            assert ctrl.collection_stop_reason == "in_progress"
            assert desktop.state == "running"

        # When the queue is genuinely empty and camera has no further frames,
        # caller loop legitimately concludes source_exhausted without hanging.
        calls = _headless_loop(desktop)

        assert ctrl.collection_stop_reason == "source_exhausted"
        assert ctrl.collection_complete is False
        assert desktop.state == "terminal"
        assert calls <= 5
