"""Bounded live session controller: pump, scoring, engine, lifecycle (T4).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T4;
    - Task: t-20260914111156870952-76424-36;
    - Governing decision: d-20260914110757304910-5.

Hard boundaries:
    - The controller owns the pump thread, the latest-slot-1 queue, the
      200ms sample gate, the 25-frame cap, controller-clock deadline, and
      drop counters. The T3 SessionEngine owns evidence accumulation;
      its internal ``frames_dropped`` stays 0 by T3 implementation
      boundary (T3 Note 1) — end-to-end drops live on this controller.
    - Results for a stale (non-current) session id are discarded, never
      delivered to the UI layer.
    - Stop/disconnect/worker-failure/close releases the source and joins
      every spawned worker. Fire-and-forget threads are forbidden: each
      spawned thread is tracked and joined on close.
    - No product entry point is exposed here (T7 owns CLI/desktop).
"""

from __future__ import annotations

from collections.abc import Callable
import threading
import time

from facecore.live.capture import CaptureSource, LatestSlot1Queue
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    SessionResult,
    SessionStatus,
)
from facecore.live.session import SessionEngine

Scorer = Callable[[FramePacket], FrameObservation]


class LiveController:
    """Owns one bounded session run: pump → score → engine observe."""

    def __init__(
        self,
        engine: SessionEngine,
        source: CaptureSource,
        scorer: Scorer,
        *,
        sample_interval_ns: int = 200_000_000,
        max_frames: int = 25,
    ) -> None:
        profile = engine.profile
        if sample_interval_ns <= 0:
            raise ValueError(
                f"sample_interval_ns must be positive, got {sample_interval_ns}"
            )
        if max_frames <= 0 or max_frames > profile.max_frames:
            raise ValueError(
                f"max_frames must be in [1, {profile.max_frames}], "
                f"got {max_frames}"
            )
        self._engine = engine
        self._source = source
        self._scorer = scorer
        self._sample_interval_ns = sample_interval_ns
        self._max_frames = max_frames

        self._queue: LatestSlot1Queue[FramePacket] = LatestSlot1Queue()
        self._session_id: str | None = None
        self._session_start_ns: int | None = None
        self._device_id: str | None = None
        self._frames_sampled = 0
        self._next_sample_ns: int | None = None
        self._last_sequence = 0
        self._terminal: SessionResult | None = None
        self._closed = False

        self._lock = threading.Lock()
        self._pump_thread: threading.Thread | None = None
        self._pump_stop = threading.Event()
        self._tracked_threads: list[threading.Thread] = []

    # -- lifecycle ---------------------------------------------------------
    def start_session(
        self, session_id: str, now_ns: int, device_id: str = "default"
    ) -> None:
        """Open the source and start engine accumulation for one session."""
        with self._lock:
            if self._closed:
                raise RuntimeError("controller is closed; cannot start session")
            if self._session_id is not None:
                raise RuntimeError(
                    f"session {self._session_id!r} still active; "
                    "close the previous session first"
                )
            if not session_id:
                raise ValueError("session_id must not be empty")
            if now_ns < 0:
                raise ValueError(f"now_ns must be >= 0, got {now_ns}")
            self._source.open(device_id)
            self._device_id = device_id
            self._engine.start(session_id, now_ns)
            self._session_id = session_id
            self._session_start_ns = now_ns
            self._frames_sampled = 0
            self._next_sample_ns = now_ns
            self._last_sequence = 0
            self._terminal = None
            self._pump_stop.clear()

    def _require_active(self) -> str:
        if self._session_id is None or self._session_start_ns is None:
            raise RuntimeError("no active session")
        return self._session_id

    # -- pump ----------------------------------------------------------------
    def _pump_once(self) -> bool:
        """Move one source packet into the slot-1 queue. False when dry."""
        packet = self._source.read()
        if packet is None:
            return False
        self._queue.push(packet)
        return True

    def pump_fast_producer(self) -> None:
        """Drain the whole source into the slot-1 queue (pressure test)."""
        self._require_active()
        while self._pump_once():
            pass

    def _sample_due(self, packet: FramePacket) -> bool:
        assert self._next_sample_ns is not None
        assert self._session_start_ns is not None
        if self._frames_sampled >= self._max_frames:
            return False
        if packet.sequence <= self._last_sequence:
            return False
        # Sample gate: first frame immediate, then every 200ms by capture
        # clock, latest-only (older gated-out frames fall out of the slot).
        if self._frames_sampled == 0:
            return True
        return packet.captured_ns >= self._next_sample_ns

    def _consume_one(self) -> SessionResult | None:
        """Score one queued packet and feed the engine. None when idle."""
        session_id = self._require_active()
        packet = self._queue.drain()
        if packet is None:
            return self._terminal
        if not self._sample_due(packet):
            return self._terminal
        self._last_sequence = packet.sequence
        self._frames_sampled += 1
        assert self._session_start_ns is not None
        # Advance the sample gate from this packet's capture clock.
        self._next_sample_ns = packet.captured_ns + self._sample_interval_ns
        try:
            observation = self._scorer(packet)
        except Exception as exc:
            self._terminal = self._engine.finish(
                self._controller_now_ns(), reason="timeout"
            )
            # Surface scorer failure as an engine error terminal.
            self._terminal = SessionResult(
                session_id=session_id,
                schema_version="v1",
                status=SessionStatus.error,
                matched_identity=None,
                reason_codes=(f"scorer_failure: {type(exc).__name__}",),
                elapsed_ms=self._terminal.elapsed_ms,
                frames_sampled=self._terminal.frames_sampled,
                frames_usable=self._terminal.frames_usable,
                frames_rejected=self._terminal.frames_rejected,
                frames_dropped=self._terminal.frames_dropped,
                support_sequences=(),
                profile_digest=self._terminal.profile_digest,
                model_generation=self._terminal.model_generation,
                gallery_digest=self._terminal.gallery_digest,
            )
            self._release_source()
            return self._terminal
        if observation.sequence != packet.sequence:
            raise ValueError(
                "scorer returned observation for "
                f"sequence {observation.sequence}, expected {packet.sequence}"
            )
        result = self._engine.observe(observation)
        if result is not None:
            self._terminal = result
            self._release_source()
        return self._terminal

    def run_until_terminal(self, max_steps: int = 100) -> SessionResult | None:
        """Pump + consume until the engine terminates or steps exhaust."""
        self._require_active()
        for _ in range(max_steps):
            if self._terminal is not None:
                return self._terminal
            if not self._pump_once():
                drained = self._consume_one()
                if drained is not None:
                    return drained
                # Source dry and queue empty: conclude at controller clock.
                return self.finish(self._controller_now_ns())
            terminal = self._consume_one()
            if terminal is not None:
                return terminal
        return self._terminal

    # -- session routing (stale-result guard) ----------------------------------
    def observe_for(
        self, session_id: str, observation: FrameObservation
    ) -> SessionResult | None:
        """Route one observation; refuse ids that are not the live session."""
        live = self._require_active()
        if session_id != live:
            raise ValueError(
                f"stale observation for session {session_id!r}; "
                f"live session is {live!r} — discarded"
            )
        result = self._engine.observe(observation)
        if result is not None:
            self._terminal = result
            self._release_source()
        return result

    def publish_external_result(self, result: SessionResult) -> None:
        """Accept an inference result only when it names the live session."""
        live = self._require_active()
        if result.session_id != live:
            raise ValueError(
                f"stale external result for session {result.session_id!r}; "
                f"live session is {live!r} — discarded, never shown in UI"
            )
        self._terminal = result
        self._release_source()

    # -- clock / finish / close --------------------------------------------------
    def _controller_now_ns(self) -> int:
        # Deadline uses the controller clock (monotonic), never the moment
        # a model happens to finish.
        return time.monotonic_ns()

    def finish(self, now_ns: int) -> SessionResult:
        """Conclude the session at the controller clock."""
        self._require_active()
        if self._terminal is not None:
            return self._terminal
        assert self._session_start_ns is not None
        deadline = self._session_start_ns + int(
            self._engine.profile.timeout_ms * 1_000_000
        )
        effective = max(now_ns, self._session_start_ns)
        if effective > deadline:
            effective = now_ns
        self._terminal = self._engine.finish(effective)
        self._release_source()
        return self._terminal

    def start_background_pump(self) -> None:
        """Start the tracked pump thread (joined on close; never detached)."""
        self._require_active()

        def _pump_loop() -> None:
            while not self._pump_stop.is_set():
                if not self._pump_once():
                    break

        thread = threading.Thread(
            target=_pump_loop, name="t4-capture-pump", daemon=True
        )
        with self._lock:
            self._tracked_threads.append(thread)
        thread.start()
        with self._lock:
            self._pump_thread = thread

    def _release_source(self) -> None:
        try:
            self._source.close()
        except Exception:
            pass

    def close(self) -> None:
        """Stop the pump, release the source, join every tracked worker."""
        with self._lock:
            if self._closed:
                self._join_tracked_locked()
                return
            self._closed = True
            self._pump_stop.set()
        self._release_source()
        with self._lock:
            self._join_tracked_locked()
            self._pump_thread = None

    def _join_tracked_locked(self) -> None:
        for thread in self._tracked_threads:
            if thread.is_alive():
                # Never join the current thread (deadlock guard).
                if thread is not threading.current_thread():
                    thread.join(timeout=5.0)

    # -- diagnostics -------------------------------------------------------------
    @property
    def frames_sampled(self) -> int:
        return self._frames_sampled

    @property
    def frames_dropped(self) -> int:
        # End-to-end drops live here (slot-1 overwrites); the T3 engine
        # internal counter stays 0 by implementation boundary.
        return self._queue.dropped

    @property
    def workers_joined(self) -> bool:
        with self._lock:
            return all(not t.is_alive() for t in self._tracked_threads)

    @property
    def source_closed(self) -> bool:
        return self._source.is_closed

    @property
    def tracked_threads(self) -> list[threading.Thread]:
        with self._lock:
            return list(self._tracked_threads)

    @property
    def current_session_id(self) -> str | None:
        return self._session_id
