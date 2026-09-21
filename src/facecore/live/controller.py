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
    """Owns one bounded session run: pump → score → engine observe.

    Phase 2B E3 fixed-window separation: in ``fixed_seconds`` mode the
    B inference terminal locks on first terminal, but the collector keeps
    sampling to the original deadline. Post-lock frames feed arm A and
    diagnostics only — they are never re-sent to B. Cancel/close/revoke/
    multi-face/continuity-unknown/error stop the collector incomplete.
    """

    def __init__(
        self,
        engine: SessionEngine,
        source: CaptureSource,
        scorer: Scorer,
        *,
        sample_interval_ns: int = 200_000_000,
        max_frames: int = 25,
        frame_sink: Callable[[FramePacket], None] | None = None,
        frame_transform: Callable[[FramePacket], FramePacket] | None = None,
        fixed_seconds: bool = False,
        trace_recorder: object | None = None,
        trace_attempt_id: str | None = None,
    ) -> None:
        """frame_sink (t-3): optional per-sampled-frame staging hook.

        Called with each sampled packet AFTER scoring succeeds and BEFORE
        engine observe, so encrypted staging (recorder.append_frame) sees
        exactly the frames the engine scored. None keeps prior behavior.

        frame_transform (E7-B): optional input transformation applied before
        the scorer (e.g. square capture geometry per Appendix A).

        fixed_seconds (E3): when True, B inference terminal locks but the
        collector continues to the original deadline for arm A + trace.
        trace_recorder/trace_attempt_id (E3): when both set, each scored
        observation is persisted via ``append_trace`` on the live path.
        """
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
        if (trace_recorder is None) != (trace_attempt_id is None):
            raise ValueError(
                "trace_recorder and trace_attempt_id must be given together"
            )
        self._engine = engine
        self._source = source
        self._scorer = scorer
        self._sample_interval_ns = sample_interval_ns
        self._max_frames = (
            profile.max_frames if fixed_seconds else max_frames
        )
        self._frame_sink = frame_sink
        self._frame_transform = frame_transform
        self._fixed_seconds = fixed_seconds
        self._trace_recorder = trace_recorder
        self._trace_attempt_id = trace_attempt_id
        self._scored_observations: list[FrameObservation] = []

        self._queue: LatestSlot1Queue[FramePacket] = LatestSlot1Queue()
        self._session_id: str | None = None
        self._session_start_ns: int | None = None
        self._device_id: str | None = None
        self._frames_sampled = 0
        self._next_sample_ns: int | None = None
        self._last_sequence = 0
        self._terminal: SessionResult | None = None
        self._closed = False

        # E3 fixed-window collector state: B locks once; the collector
        # continues independently until deadline / cap / stop signal.
        self._inference_terminal: SessionResult | None = None
        self._post_lock_observations: list[FrameObservation] = []
        self._collector_complete = False
        self._collector_stop_reason = "in_progress"
        self._collector_safety_flags: list[str] = []
        self._collection_cancelled = False
        self._consecutive_dry = 0

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
            self._scored_observations = []
            self._inference_terminal = None
            self._post_lock_observations = []
            self._collector_complete = False
            self._collector_stop_reason = "in_progress"
            self._collector_safety_flags = []
            self._collection_cancelled = False
            self._consecutive_dry = 0
            self._pump_stop.clear()

    def _require_active(self) -> str:
        if self._session_id is None or self._session_start_ns is None:
            raise RuntimeError("no active session")
        return self._session_id

    # -- pump ----------------------------------------------------------------
    def _source_closed(self) -> bool:
        """Closed-source check tolerant of property/method test doubles.

        The ``CaptureSource`` ABC declares ``is_closed`` as a property, but
        older test doubles expose a plain method of the same name. A bound
        method object is always truthy, which would silently skip the closed
        fast path on such doubles. Evaluate callables instead of trusting
        raw truthiness (B3).
        """
        closed = getattr(self._source, "is_closed", False)
        if callable(closed):
            try:
                return bool(closed())
            except Exception:
                return False
        return bool(closed)

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

    def _consume_one(self) -> tuple[FramePacket | None, SessionResult | None]:
        """Score one queued packet and feed the engine.

        Returns (consumed, terminal) so the caller can distinguish all four
        outcomes B2/B4 require: terminal ready, packet consumed but pre-B
        (no terminal), gated-out packet (drained from queue, not dry), and
        queue empty.
        """
        session_id = self._require_active()
        packet = self._queue.drain()
        if packet is None:
            return None, None
        if not self._sample_due(packet):
            return packet, self._terminal
        self._last_sequence = packet.sequence
        self._frames_sampled += 1
        assert self._session_start_ns is not None
        # Advance the sample gate from this packet's capture clock.
        self._next_sample_ns = packet.captured_ns + self._sample_interval_ns
        score_packet = (
            self._frame_transform(packet)
            if self._frame_transform is not None
            else packet
        )
        try:
            observation = self._scorer(score_packet)
        except Exception as exc:
            # A transform failure (e.g. capture geometry drift) refuses the
            # scorer input: propagate so the caller fails closed instead of
            # committing a bundle with unreconstructible geometry.
            if score_packet is not packet:
                raise
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
            # Stop the background pump BEFORE releasing the source: a
            # native read blocked in the pump thread must not race the
            # close (AVFoundation segfaults on close-during-read).
            # No-op when no background pump is running (sync path).
            self._stop_and_release()
            return packet, self._terminal
        if observation.sequence != packet.sequence:
            raise ValueError(
                "scorer returned observation for "
                f"sequence {observation.sequence}, expected {packet.sequence}"
            )
        self._scored_observations.append(observation)
        if self._frame_sink is not None:
            # E7-B Appendix A.7: staging/preview keep the original full
            # frame plus mapping; only the scorer input is transformed.
            self._frame_sink(packet)
        self._append_live_trace(observation)
        if self._fixed_seconds and self._inference_terminal is not None:
            # Fixed-window: B already locked. This frame belongs to the
            # collector (arm A + diagnostics) only — never back into B.
            self._post_lock_observations.append(observation)
            self._collect_safety_flags(observation)
            return packet, self._inference_terminal
        result = self._engine.observe(observation)
        if result is not None:
            if self._fixed_seconds:
                # B locks here; the collector continues to the deadline.
                self._inference_terminal = result
                self._terminal = result
                return packet, self._terminal
            self._terminal = result
            self._stop_and_release()
        return packet, self._terminal

    def _append_live_trace(self, observation: FrameObservation) -> None:
        """Persist one scored observation to the encrypted trace sidecar."""
        if self._trace_recorder is None or self._trace_attempt_id is None:
            return
        from facecore.research.diagnostics import FrameTraceEntry

        entry = FrameTraceEntry.from_observation(
            observation, staged_index=self._frames_sampled - 1
        )
        append = getattr(self._trace_recorder, "append_trace", None)
        if append is None:
            raise AttributeError(
                "trace_recorder has no append_trace method"
            )
        try:
            append(self._trace_attempt_id, entry)
        except ValueError:
            # Duplicate sequence on re-drive: keep first write, stay live.
            pass

    def _collect_safety_flags(self, observation: FrameObservation) -> None:
        """Post-lock safety monitoring: multi-face never goes unnoticed."""
        if observation.face_count > 1:
            for reason in observation.quality_reasons:
                if reason not in self._collector_safety_flags:
                    self._collector_safety_flags.append(reason)
            if not observation.quality_reasons:
                if "input_multiple_faces" not in self._collector_safety_flags:
                    self._collector_safety_flags.append("input_multiple_faces")

    def cancel_collection(self, now_ns: int) -> SessionResult:
        """Stop the collector immediately: incomplete, terminal preserved."""
        self._require_active()
        self._collection_cancelled = True
        self._collector_complete = False
        self._collector_stop_reason = "cancelled"
        if self._inference_terminal is not None:
            self._terminal = self._inference_terminal
        else:
            self._terminal = self._engine.finish(now_ns, reason="cancelled")
            self._inference_terminal = self._terminal
        self._stop_and_release()
        return self._terminal

    def run_until_terminal(self, max_steps: int = 100) -> SessionResult | None:
        """Pump + consume until the engine terminates or steps exhaust."""
        self._require_active()
        for _ in range(max_steps):
            if self._fixed_seconds:
                if self._collector_stop_reason != "in_progress":
                    return self._terminal
                if self._collection_should_stop():
                    self._finalize_collection()
                    return self._terminal
            elif self._terminal is not None:
                return self._terminal
            if not self._pump_once():
                consumed, terminal = self._consume_one()
                if consumed is not None:
                    # B2: a drained+scored packet is never dry, even when the
                    # engine has not yet locked B (terminal None pre-lock).
                    # Transient AVFoundation read failure on an open camera
                    # must not retire the collector while queue still feeds.
                    self._consecutive_dry = 0
                    if self._fixed_seconds:
                        if self._collection_should_stop():
                            self._finalize_collection()
                        continue
                    if terminal is not None:
                        return terminal
                    continue
                self._consecutive_dry += 1
                closed = self._source_closed()
                if self._consecutive_dry < 3 and not closed:
                    continue
                # Source dry (>= 3 consecutive empty reads after empty queue,
                # or source explicitly closed) and queue empty: conclude at
                # controller clock. The 3-read bar (~100ms at 30fps) is a
                # transient-drop tolerance, not a time semantic: it only
                # delays the finalize by a bounded camera-jitter window, and
                # the resulting stop_reason is still decided by
                # _collection_should_stop / _finalize_collection state.
                if self._fixed_seconds:
                    self._finalize_collection()
                    return self._terminal
                return self.finish(self._controller_now_ns())
            self._consecutive_dry = 0
            _consumed, terminal = self._consume_one()
            if terminal is not None:
                if self._fixed_seconds:
                    if self._collection_should_stop():
                        self._finalize_collection()
                    continue
                return terminal
        if self._fixed_seconds:
            if self._collection_should_stop():
                self._finalize_collection()
        return self._terminal

    def _collection_should_stop(self) -> bool:
        """Collector stops on cancel, cap, deadline, or source exhaustion."""
        if self._collection_cancelled:
            return True
        if self._frames_sampled >= self._max_frames:
            return True
        if self._session_start_ns is None:
            return True
        deadline = self._session_start_ns + int(
            self._engine.profile.timeout_ms * 1_000_000
        )
        last_ns = self._last_sampled_ns()
        return last_ns is not None and last_ns >= deadline

    def _last_sampled_ns(self) -> int | None:
        if self._scored_observations:
            return self._scored_observations[-1].captured_ns
        return None

    @property
    def last_consumed_ns(self) -> int | None:
        """Newest consumed capture stamp (session-clock domain, read-only)."""
        return self._last_sampled_ns()

    def _finalize_collection(self) -> None:
        """Seal collector evidence without rewriting the B terminal."""
        if self._collector_stop_reason != "in_progress":
            return
        if self._collection_cancelled:
            self._collector_stop_reason = "cancelled"
        elif self._frames_sampled >= self._max_frames:
            last_ns = self._last_sampled_ns()
            deadline = (self._session_start_ns or 0) + int(
                self._engine.profile.timeout_ms * 1_000_000
            )
            if last_ns is not None and last_ns >= deadline:
                self._collector_stop_reason = "deadline_reached"
                self._collector_complete = True
            else:
                self._collector_stop_reason = "max_frames_reached"
                self._collector_complete = False
        else:
            last_ns = self._last_sampled_ns()
            deadline = (self._session_start_ns or 0) + int(
                self._engine.profile.timeout_ms * 1_000_000
            )
            if last_ns is not None and last_ns >= deadline:
                self._collector_stop_reason = "deadline_reached"
                self._collector_complete = True
            else:
                self._collector_stop_reason = "source_exhausted"
                self._collector_complete = False
        if self._terminal is None and self._inference_terminal is not None:
            self._terminal = self._inference_terminal
        self._stop_and_release()

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
            self._stop_and_release()
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
        self._stop_and_release()

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
        self._stop_and_release()
        return self._terminal

    def run_with_timeout(self, timeout_ns: int) -> SessionResult:
        """Background pump + synchronous consume with true timeout semantics.

        T7 N1 (claimed by T8): the background pump fills the slot-1 queue
        while the foreground consumes. On timeout the pump is stopped,
        the worker joined, the source released, and a timeout terminal is
        returned. An earlier engine terminal wins immediately (also
        stopping and joining the pump). Either way no worker is left
        behind and no fire-and-forget thread escapes.
        """
        live = self._require_active()
        if timeout_ns <= 0:
            raise ValueError(f"timeout_ns must be positive, got {timeout_ns}")
        assert self._session_start_ns is not None
        deadline_ns = self._session_start_ns + timeout_ns
        self.start_background_pump()
        try:
            while True:
                if self._terminal is not None:
                    return self._terminal
                _consumed, terminal = self._consume_one()
                if terminal is not None:
                    return terminal
                if self._controller_now_ns() >= deadline_ns:
                    break
                time.sleep(0.005)
        finally:
            self._stop_pump_and_join()
        if self._terminal is not None:
            return self._terminal
        self._terminal = self._engine.finish(deadline_ns)
        self._stop_and_release()
        assert self._terminal.session_id == live
        return self._terminal

    def _stop_pump_and_join(self) -> None:
        self._pump_stop.set()
        with self._lock:
            self._join_tracked_locked()

    def _stop_and_release(self) -> None:
        """Stop the pump, join it (bounded), then release — iff safe.

        The pump thread closes the source itself on exit, so a release
        here is usually a no-op. It is SKIPPED while the pump is still
        alive: releasing under an in-flight native read segfaults
        AVFoundation (issue #64). The pump's own finally-close covers
        the skipped case whenever its read returns. With no background
        pump (sync path) the release always runs.
        Must NOT be called while holding self._lock (see close()).

        Honest residual: if the native read NEVER returns, the source
        stays open and the (daemon) pump thread stays alive — verified
        locally (closed False, joined False until the gate opens).
        Rationale: disconnect makes AVFoundation reads fail-return, so
        the stuck-forever case is driver-hang-only; crashing the process
        (status quo) is strictly worse than leaking one daemon thread.
        A watchdog for this residual is future work, not this PR.
        """
        self._stop_pump_and_join()
        with self._lock:
            joined = all(not t.is_alive() for t in self._tracked_threads)
        if joined:
            self._release_source()

    def start_background_pump(self) -> None:
        """Start the tracked pump thread (joined on close; never detached)."""
        self._require_active()

        def _pump_loop() -> None:
            try:
                while not self._pump_stop.is_set():
                    if not self._pump_once():
                        break
            finally:
                # The pump thread owns the source close on its way out:
                # close and read never run concurrently on two threads.
                self._release_source()

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
        """Stop the pump, join it, then release the source.

        Ordering (issue #64): the pump thread closes the source itself
        on exit; this release is a no-op then. Never release while a
        native read may still be in flight.
        """
        with self._lock:
            if self._closed:
                self._join_tracked_locked()
                return
            self._closed = True
            self._pump_stop.set()
            self._join_tracked_locked()
        self._release_source()
        with self._lock:
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

    @property
    def scored_observations(self) -> list[FrameObservation]:
        """Copy of scored observations in sample order (t-3 ledger source)."""
        return list(self._scored_observations)

    # -- E3 fixed-window collector evidence (read-only) -------------------------
    @property
    def inference_terminal(self) -> SessionResult | None:
        """B locked terminal: never rewritten by post-lock collector frames."""
        return self._inference_terminal

    @property
    def post_lock_observations(self) -> list[FrameObservation]:
        """Collector frames after B locked (arm A + diagnostics only)."""
        return list(self._post_lock_observations)

    @property
    def collection_complete(self) -> bool:
        """True only when the collector reached the original deadline."""
        return self._collector_complete

    @property
    def collection_stop_reason(self) -> str:
        return self._collector_stop_reason

    @property
    def collection_safety_flags(self) -> list[str]:
        """Post-lock safety signals (e.g. multi-face) spotted by monitor."""
        return list(self._collector_safety_flags)
