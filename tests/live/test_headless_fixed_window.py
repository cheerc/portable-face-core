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
import time

import numpy as np

from facecore.live.capture import CaptureSource
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


class _RealtimeCamera(CaptureSource):
    """Paces at real wall-clock cadence (1/fps per read).

    NOT paced to the 200 ms sample interval — that is the bug's
    masking mechanism (``_SteppedCamera`` sleeps 200 ms, making 50
    steps cover 10 s).  This camera sleeps only ``1/fps`` (~33 ms at
    30 fps), so 50 steps span only ~1.7 s — far short of a 5 s window.
    """

    def __init__(self, fps: int = 30, max_frames: int = 10_000) -> None:
        self._period_s = 1.0 / fps
        self._max = max_frames
        self._seq = 0
        self._lock = threading.Lock()
        self._opened = False
        self._closed = False

    def open(self, device_id: str) -> None:
        with self._lock:
            self._seq = 0
            self._opened = True
            self._closed = False

    def read(self) -> FramePacket | None:
        with self._lock:
            if self._closed or not self._opened or self._seq >= self._max:
                return None
        time.sleep(self._period_s)
        with self._lock:
            if self._closed or not self._opened:
                return None
            self._seq += 1
            rgb = np.full((200, 200, 3), 120, dtype=np.uint8)
            return FramePacket(
                sequence=self._seq,
                captured_ns=time.monotonic_ns(),
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


def _matched_scorer(packet: FramePacket) -> FrameObservation:
    """Every frame is quality-pass with a strong person-01 match."""
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=time.monotonic_ns(),
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
) -> tuple[DesktopSession, int, ResearchProfile]:
    if profile is None:
        profile = _profile()
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
    start_ns = time.monotonic_ns()
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
            camera, trace_collector=tc
        )

        calls = _headless_loop(desktop)

        ctrl = desktop._controller
        assert ctrl.collection_stop_reason == "deadline_reached"
        assert ctrl.collection_complete is True
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
        desktop, start_ns, profile = _make_desktop(camera)

        calls = _headless_loop(desktop)

        ctrl = desktop._controller
        assert ctrl.collection_stop_reason == "deadline_reached"
        assert ctrl.collection_complete is True
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
        """A single run_until_terminal(50) returns well before deadline.

        Proves the shared API contract: bounded by step budget, not by
        the profile deadline — Qt timer tick is never stretched to 5 s.
        """
        camera = _RealtimeCamera(fps=30)
        desktop, start_ns, profile = _make_desktop(camera)

        t0 = time.monotonic()
        desktop.run_until_terminal(max_steps=50)
        elapsed = time.monotonic() - t0

        assert elapsed < 3.0
        assert desktop.state == "running"

    def test_cancel_during_collection(self) -> None:
        """Cancel stops collection immediately and sets state to terminal."""
        camera = _RealtimeCamera(fps=30)
        desktop, start_ns, profile = _make_desktop(camera)

        desktop.run_until_terminal(max_steps=10)
        assert desktop.state == "running"
        result = desktop.on_cancel(time.monotonic_ns())

        assert desktop.state == "terminal"
        assert result.status == SessionStatus.cancelled
        assert desktop.collection_stop_reason == "cancelled"

        calls = _headless_loop(desktop)
        assert calls == 0

    def test_transient_dropped_frame_does_not_abort_collection(self) -> None:
        """N1: transient frame drop on an open camera does not abort collection."""
        class _FlakyCamera(CaptureSource):
            def __init__(self, fps: int = 30) -> None:
                self._period_s = 1.0 / fps
                self._seq = 0
                self._lock = threading.Lock()
                self._opened = False
                self._closed = False
                self._dropped_seqs = {3, 10}

            def open(self, device_id: str) -> None:
                with self._lock:
                    self._seq = 0
                    self._opened = True
                    self._closed = False

            def read(self) -> FramePacket | None:
                with self._lock:
                    if self._closed or not self._opened:
                        return None
                time.sleep(self._period_s)
                with self._lock:
                    if self._closed or not self._opened:
                        return None
                    self._seq += 1
                    if self._seq in self._dropped_seqs:
                        return None
                    rgb = np.full((200, 200, 3), 120, dtype=np.uint8)
                    return FramePacket(
                        sequence=self._seq,
                        captured_ns=time.monotonic_ns(),
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

        camera = _FlakyCamera(fps=30)
        desktop, start_ns, profile = _make_desktop(camera)

        calls = _headless_loop(desktop)

        ctrl = desktop._controller
        assert ctrl.collection_stop_reason == "deadline_reached"
        assert ctrl.collection_complete is True
        assert calls >= 2
