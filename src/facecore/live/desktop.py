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
from datetime import datetime, timezone
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
from facecore.research.experiment import EvaluationLabel
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
        # Ruling: max_frames default kept at 25 by design (task -21/-24 ruling).
        # In fixed mode profile.max_frames wins; in non-fixed mode 25 is a
        # valid execution-layer clamp pinned by test_execution_max_frames_cap_enforced.
        max_frames: int = 25,
        frame_sink: Callable[[FramePacket], None] | None = None,
        frame_transform: Callable[[FramePacket], FramePacket] | None = None,
        fixed_seconds: bool = False,
        trace_recorder: object | None = None,
        trace_attempt_id: str | None = None,
        label_recorder: object | None = None,
        label_attempt_id: str | None = None,
        label_actor_ref: str = "desktop-operator",
        release_source_on_terminal: bool = True,
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
            frame_transform=frame_transform,
            fixed_seconds=fixed_seconds,
            trace_recorder=trace_recorder,
            trace_attempt_id=trace_attempt_id,
            release_source_on_terminal=release_source_on_terminal,
        )
        self._engine = engine
        self._session_id = session_id
        self._state: DesktopState = "idle"
        self._terminal: SessionResult | None = None
        self._label: str | None = None
        self._deleted = False
        self._delete_failed = False
        self._label_recorder = label_recorder
        self._label_attempt_id = label_attempt_id
        self._label_actor_ref = label_actor_ref
        if (label_recorder is None) != (label_attempt_id is None):
            raise ValueError(
                "label_recorder and label_attempt_id must be given together"
            )
        if not label_actor_ref:
            raise ValueError("label_actor_ref must not be empty")
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

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def deleted(self) -> bool:
        """True once the operator deleted the session chain via the UI."""
        return self._deleted

    @property
    def delete_failed(self) -> bool:
        """True if the operator attempted deletion but the deletion failed."""
        return self._delete_failed

    def mark_deleted(self) -> None:
        """Flag the session chain deleted: stops any later commit path."""
        self._deleted = True
        self._delete_failed = False
        self._recording = False

    def mark_delete_failed(self) -> None:
        """Flag that deletion failed: stops any later commit path."""
        self._deleted = False
        self._delete_failed = True
        self._recording = False

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
        if self._controller._fixed_seconds:
            cancelled = self._controller.cancel_collection(now_ns)
        else:
            cancelled = self._engine.finish(now_ns, reason="cancelled")
            self._controller.close()
        self._terminal = cancelled
        self._state = "terminal"
        self._recording = False
        return cancelled

    def run_until_terminal(self, max_steps: int = 100) -> SessionResult | None:
        """Drive pump→score→engine synchronously (headless/test path)."""
        if self._state != "running":
            raise RuntimeError(f"cannot run from state {self._state!r}")
        terminal = self._controller.run_until_terminal(max_steps=max_steps)
        if terminal is None:
            if (
                self._controller._fixed_seconds
                and self._controller.collection_stop_reason == "in_progress"
            ):
                return None
            terminal = self._controller.finish(
                self._controller._controller_now_ns()
            )
        self._terminal = terminal
        if (
            self._controller._fixed_seconds
            and self._controller.collection_stop_reason == "in_progress"
        ):
            # Fixed-window mode keeps the view running after B locks so Cancel
            # can stop the remaining collector before its original deadline.
            self._state = "running"
            self._recording = True
        else:
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
        if (
            self._controller._fixed_seconds
            and self._controller.collection_stop_reason == "in_progress"
        ):
            # Fixed-window mode keeps the view running after B locks so Cancel
            # can stop the remaining collector before its original deadline.
            self._state = "running"
            self._recording = True
        else:
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

    def configure_label_persistence(
        self,
        recorder: object,
        attempt_id: str,
        *,
        actor_ref: str = "desktop-operator",
    ) -> None:
        """Attach the evaluator-only label sidecar before terminal labeling."""
        if self._state not in ("idle", "terminal"):
            raise RuntimeError(
                f"cannot configure label persistence in state {self._state!r}"
            )
        if not attempt_id:
            raise ValueError("attempt_id must not be empty")
        if not actor_ref:
            raise ValueError("actor_ref must not be empty")
        self._label_recorder = recorder
        self._label_attempt_id = attempt_id
        self._label_actor_ref = actor_ref

    def label_terminal(
        self,
        ground_truth: str | None,
        *,
        kind: str | None = None,
        labeled_at_utc: str | None = None,
    ) -> None:
        """Persist evaluator-only label; it never enters scorer/engine inputs.

        A non-None identity is an ``enrolled`` label.  ``kind="unenrolled"``
        explicitly labels an unknown sample without displaying a guessed name.
        Existing callers without a recorder remain in-memory compatible.
        """
        if self._state != "terminal":
            raise RuntimeError(
                f"can only label a terminal session, not {self._state!r}"
            )
        if kind is None:
            kind = "enrolled" if ground_truth is not None else "unenrolled"
        if kind not in {"enrolled", "unenrolled", "uncertain"}:
            raise ValueError(f"unsupported label kind {kind!r}")
        if kind == "enrolled" and not ground_truth:
            raise ValueError("enrolled label requires identity")
        if kind != "enrolled" and ground_truth is not None:
            raise ValueError(f"{kind} label must not carry identity")

        if self._label_recorder is not None and self._label_attempt_id is not None:
            read_history = getattr(self._label_recorder, "read_label_history")
            write_label = getattr(self._label_recorder, "write_label")
            try:
                history = read_history(self._label_attempt_id)
            except KeyError:
                history = []
            revision = (history[-1].revision + 1) if history else 1
            timestamp = labeled_at_utc or datetime.now(timezone.utc).isoformat()
            write_label(
                EvaluationLabel(
                    attempt_id=self._label_attempt_id,
                    revision=revision,
                    kind=kind,
                    identity_id=ground_truth if kind == "enrolled" else None,
                    actor_ref=self._label_actor_ref,
                    labeled_at=timestamp,
                )
            )
        self._label = ground_truth
        self._state = "labeled"

    def close(self) -> None:
        """UI close event: release source, join workers, end lifecycle."""
        self._controller.close()
        self._recording = False
        if self._state != "labeled":
            self._state = "closed"

    def detach(self) -> None:
        """Discard a finished round without releasing the shared source.

        G3 W2 round handoff: stops the pump and joins workers, but the
        camera handle stays open for the next round built over the same
        source. Only the live window's current round is closed with
        close() (which releases the source).
        """
        self._controller.close_without_source()
        self._recording = False

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
    def source(self) -> CaptureSource:
        """Shared capture source (read-only; G3 W2 standby preview)."""
        return self._controller.source

    @property
    def scorer(self) -> Callable[[FramePacket], FrameObservation]:
        """Scoring function (read-only; G3 W2 standby face trigger)."""
        return self._controller.scorer

    @property
    def observations(self) -> list[FrameObservation]:
        """Scored observations in sample order (t-3 ledger source)."""
        return self._controller.scored_observations

    @property
    def controller_consumed_ns(self) -> int | None:
        """Newest consumed capture stamp (session-clock domain, read-only)."""
        return self._controller.last_consumed_ns

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
