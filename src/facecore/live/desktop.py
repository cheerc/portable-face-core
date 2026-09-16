"""Headless-safe desktop session bindings (Phase 2A §4 & §6 T7).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T7;
    - Task: t-20260914111211897569-76424-38;
    - Governing decision: d-20260914110757304910-5;
    - UI selection: D1 §11.1 (pyside6 primary, LGPLv3 dynamic linking).

Hard boundaries:
    - This module never imports a GUI toolkit at module load: it owns the
      controller lifecycle plus UI event bindings (start/cancel/countdown/
      face-box/quality display model/terminal labeling) and stays
      headless-testable. Real-window wiring belongs to T7's runbook and a
      future thin view layer; no real-device evidence is claimed here.
    - Start requires an explicit ConsentRecord with both record and image
      consent; without them the session never opens (fail-closed).
    - Identity display rule: display_identity() returns a name only when
      the terminal status is matched; every other band shows no name.
    - Ground truth never enters the scorer/engine path: label_terminal()
      annotates the label sidecar only (consumed by report.summarize).
    - Close releases the source and joins every worker (no orphan).
    - Research watermark is always present (research, never认证).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from facecore.live.capture import CaptureSource
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    SessionResult,
    SessionStatus,
)
from facecore.live.controller import LiveController
from facecore.live.session import SessionEngine
from facecore.research.records import ConsentRecord

DesktopState = Literal["idle", "running", "terminal", "labeled", "closed"]

WATERMARK = "研究原型 · research prototype — 非身份认证 · not for identification"


class DesktopSession:
    """UI event bindings over one LiveController run (headless-testable)."""

    def __init__(
        self,
        engine: SessionEngine,
        source: CaptureSource,
        scorer: Callable[[FramePacket], FrameObservation],
        session_id: str,
        *,
        sample_interval_ns: int = 200_000_000,
        max_frames: int = 25,
        frame_sink: Callable[[FramePacket], None] | None = None,
        fixed_seconds: bool = False,
        trace_recorder: object | None = None,
        trace_attempt_id: str | None = None,
    ) -> None:
        if not session_id:
            raise ValueError("session_id must not be empty")
        self._controller = LiveController(
            engine=engine,
            source=source,
            scorer=scorer,
            sample_interval_ns=sample_interval_ns,
            max_frames=max_frames,
            frame_sink=frame_sink,
            fixed_seconds=fixed_seconds,
            trace_recorder=trace_recorder,
            trace_attempt_id=trace_attempt_id,
        )
        self._engine = engine
        self._session_id = session_id
        self._state: DesktopState = "idle"
        self._terminal: SessionResult | None = None
        self._label: str | None = None
        self._recording = False

    # -- UI events -----------------------------------------------------------
    @property
    def state(self) -> DesktopState:
        return self._state

    @property
    def watermark(self) -> str:
        return WATERMARK

    @property
    def label(self) -> str | None:
        return self._label

    @property
    def recording(self) -> bool:
        """Recording indicator: on from successful start until close."""
        return self._recording

    def on_start(
        self, consent: ConsentRecord, now_ns: int, device_id: str = "default"
    ) -> None:
        """Start event: requires explicit dual consent (checkbox-mapped).

        Fix (a): device_id is passed through to the capture open call so
        the requested --device reaches the camera (no fallback-0).
        """
        if self._state != "idle":
            raise RuntimeError(f"cannot start from state {self._state!r}")
        if consent is None or not isinstance(consent, ConsentRecord):
            raise PermissionError("explicit consent record is required")
        if consent.session_id != self._session_id:
            raise ValueError("consent session id does not match")
        if not consent.record_consent:
            raise PermissionError("record consent absent; refusing to start")
        if not consent.image_consent:
            raise PermissionError("image consent absent; refusing to start")
        self._controller.start_session(self._session_id, now_ns, device_id=device_id)
        self._state = "running"
        self._recording = True

    def on_cancel(self, now_ns: int) -> SessionResult:
        """Cancel event: stops immediately with cancelled terminal."""
        if self._state != "running":
            raise RuntimeError(f"cannot cancel from state {self._state!r}")
        cancelled = self._engine.finish(now_ns, reason="cancelled")
        self._terminal = cancelled
        self._state = "terminal"
        self._recording = False
        self._controller.close()
        return cancelled

    def run_until_terminal(self, max_steps: int = 100) -> SessionResult | None:
        """Drive pump→score→engine synchronously (headless/test path)."""
        if self._state != "running":
            raise RuntimeError(f"cannot run from state {self._state!r}")
        terminal = self._controller.run_until_terminal(max_steps=max_steps)
        if terminal is None:
            terminal = self._controller.finish(
                self._controller._controller_now_ns()
            )
        self._terminal = terminal
        self._state = "terminal"
        self._recording = False
        return terminal

    def run_background_and_join(self, timeout_s: float = 10.0) -> SessionResult:
        """T4 N2: background-pump + synchronous-consume dual-mode case.

        The background pump fills the slot-1 queue while the foreground
        consumes; both modes share the single queue and the single
        tracked worker, which close() joins. Proves no race loses the
        terminal or leaks the worker.
        """
        if self._state != "running":
            raise RuntimeError(f"cannot run from state {self._state!r}")
        self._controller.start_background_pump()
        terminal = self._controller.run_until_terminal(max_steps=200)
        if terminal is None:
            terminal = self._controller.finish(
                self._controller._controller_now_ns()
            )
        self._terminal = terminal
        self._state = "terminal"
        self._recording = False
        _ = timeout_s
        return terminal

    def display_identity(self) -> str | None:
        """Identity display rule: matched shows the name; else nothing."""
        if self._terminal is None:
            return None
        if self._terminal.status == SessionStatus.matched:
            return self._terminal.matched_identity
        return None

    def display_band(self) -> str:
        """Quality/decision band for the face-box overlay (no names)."""
        if self._terminal is None:
            return "running"
        return self._terminal.status.value

    def countdown_ms_remaining(self, now_ns: int) -> int:
        """5-second countdown from the controller clock."""
        start_ns = self._controller._session_start_ns
        if start_ns is None:
            return int(self._engine.profile.timeout_ms)
        remaining_ns = (
            start_ns + int(self._engine.profile.timeout_ms * 1_000_000) - now_ns
        )
        return max(0, remaining_ns // 1_000_000)

    def label_terminal(self, ground_truth: str | None) -> None:
        """Operator post-terminal labeling: sidecar only, never scorer input."""
        if self._state != "terminal":
            raise RuntimeError(
                f"can only label a terminal session, not {self._state!r}"
            )
        self._label = ground_truth
        self._state = "labeled"

    def close(self) -> None:
        """UI close event: release source, join workers, end lifecycle."""
        self._controller.close()
        self._recording = False
        if self._state != "labeled":
            self._state = "closed"

    # -- diagnostics (delegated, read-only) ------------------------------------
    @property
    def source_closed(self) -> bool:
        return self._controller.source_closed

    @property
    def workers_joined(self) -> bool:
        return self._controller.workers_joined

    @property
    def terminal(self) -> SessionResult | None:
        return self._terminal

    @property
    def observations(self) -> list[FrameObservation]:
        """Scored observations in sample order (t-3 ledger source)."""
        return self._controller.scored_observations

    def cancel_collection(self, now_ns: int) -> SessionResult:
        """E3: stop the fixed-window collector immediately (incomplete)."""
        if self._state != "running":
            raise RuntimeError(f"cannot cancel from state {self._state!r}")
        cancelled = self._controller.cancel_collection(now_ns)
        self._terminal = cancelled
        self._state = "terminal"
        self._recording = False
        return cancelled

    # -- E3 fixed-window collector evidence (read-only passthrough) ------------
    @property
    def inference_terminal(self) -> SessionResult | None:
        return self._controller.inference_terminal

    @property
    def post_lock_observations(self) -> list[FrameObservation]:
        return self._controller.post_lock_observations

    @property
    def collection_complete(self) -> bool:
        return self._controller.collection_complete

    @property
    def collection_stop_reason(self) -> str:
        return self._controller.collection_stop_reason

    @property
    def collection_safety_flags(self) -> list[str]:
        return self._controller.collection_safety_flags
