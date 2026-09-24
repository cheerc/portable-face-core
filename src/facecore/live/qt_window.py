"""Minimal optional Qt research window and square capture geometry (E7-B).

The module is intentionally outside the core import path.  PySide6 is loaded
only when this optional ``research-ui`` module is imported; ``facecore.live``
remains usable without the GUI dependency.

The capture contract is Appendix A of the Phase 2B research specification:
center crop with ``S=min(W,H)``, one mapping for preview and crop, mirror only
for preview, and no identity-dependent geometry.  The first-round recorder
keeps the finite original frame and authenticates the mapping sidecar.

Synthetic/offscreen tests only; this module does not open a camera by itself.
"""

# mypy: disable-error-code=unused-ignore

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any

import numpy as np

from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    SessionResult,
    SessionStatus,
)
from facecore.live.desktop import DesktopSession
from facecore.research.records import ConsentRecord


@dataclass(frozen=True)
class RoundComplete:
    """One labeled G3 round awaiting record commit (G3 W4).

    Produced by the window when the operator presses 正確／錯誤 (label
    already persisted to the round sidecar); consumed by the CLI tail,
    which commits the encrypted bundle, closes the round attempt, and
    appends the results.csv row. Observations carry the scored frames
    for top1/top2/margin reduction.
    """

    session_id: str
    attempt_id: str | None
    terminal: SessionResult
    observations: tuple[FrameObservation, ...]
    label_kind: str
    label_identity: str | None
    profile_version: str
    started_utc: str


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"crop_mapping {name} must be an integer")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"crop_mapping {name} must be bool")
    return value


@dataclass(frozen=True)
class CropMapping:
    """Mapping from an oriented source frame to its center square."""

    x: int
    y: int
    size: int
    frame_w: int
    frame_h: int
    mirrored_preview: bool = False

    def __post_init__(self) -> None:
        if self.frame_w <= 0 or self.frame_h <= 0:
            raise ValueError("frame dimensions must be positive")
        if self.size <= 0:
            raise ValueError("crop size must be positive")
        if self.x < 0 or self.y < 0:
            raise ValueError("crop origin must be non-negative")
        if self.x + self.size > self.frame_w or self.y + self.size > self.frame_h:
            raise ValueError("crop is outside the source frame")
        if self.size != min(self.frame_w, self.frame_h):
            raise ValueError("crop size must equal the shorter frame side")

    def to_dict(self) -> dict[str, object]:
        """Serialize the frozen mapping using the E7 premise schema."""
        return {
            "x": self.x,
            "y": self.y,
            "size": self.size,
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "mirrored_preview": self.mirrored_preview,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CropMapping:
        required = {"x", "y", "size", "frame_w", "frame_h", "mirrored_preview"}
        if set(data) != required:
            raise ValueError("crop_mapping schema mismatch")
        return cls(
            x=_require_int(data["x"], "x"),
            y=_require_int(data["y"], "y"),
            size=_require_int(data["size"], "size"),
            frame_w=_require_int(data["frame_w"], "frame_w"),
            frame_h=_require_int(data["frame_h"], "frame_h"),
            mirrored_preview=_require_bool(
                data["mirrored_preview"], "mirrored_preview"
            ),
        )


def center_square_crop(
    frame_w: int, frame_h: int, *, mirrored_preview: bool = False
) -> CropMapping:
    """Return the center-square mapping for a positive oriented frame."""
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("frame dimensions must be positive")
    size = min(frame_w, frame_h)
    return CropMapping(
        x=(frame_w - size) // 2,
        y=(frame_h - size) // 2,
        size=size,
        frame_w=frame_w,
        frame_h=frame_h,
        mirrored_preview=mirrored_preview,
    )


def crop_frame(
    frame: np.ndarray, *, mirrored_preview: bool = False
) -> tuple[np.ndarray, CropMapping]:
    """Crop an RGB frame without resizing and return its geometry mapping."""
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be numpy.ndarray")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (H, W, 3)")
    if frame.dtype != np.uint8:
        raise ValueError("frame must use uint8 pixels")
    height, width = frame.shape[:2]
    mapping = center_square_crop(width, height, mirrored_preview=mirrored_preview)
    cropped = frame[
        mapping.y : mapping.y + mapping.size,
        mapping.x : mapping.x + mapping.size,
        :,
    ]
    return np.ascontiguousarray(cropped), mapping


def preview_frame(frame: np.ndarray, mapping: CropMapping) -> np.ndarray:
    """Apply preview-only mirror to an already cropped frame."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (H, W, 3)")
    expected = (mapping.size, mapping.size, 3)
    if frame.shape != expected:
        raise ValueError(f"frame shape {frame.shape} does not match {expected}")
    if mapping.mirrored_preview:
        return np.ascontiguousarray(frame[:, ::-1, :])
    return np.ascontiguousarray(frame)


def crop_packet(
    packet: FramePacket, *, mirrored_preview: bool = False
) -> tuple[FramePacket, CropMapping]:
    """Apply center-square crop to a FramePacket without mutating sequence/time."""
    cropped, mapping = crop_frame(packet.rgb, mirrored_preview=mirrored_preview)
    return (
        FramePacket(
            sequence=packet.sequence,
            captured_ns=packet.captured_ns,
            rgb=cropped,
            orientation=packet.orientation,
            mirrored=packet.mirrored,
        ),
        mapping,
    )


_QT_WINDOW_FACTORY: Any

try:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised without extra
    _QT_IMPORT_ERROR = exc

    class _QtResearchWindowUnavailable:
        """Helpful failure when the optional research-ui extra is absent."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "QtResearchWindow requires the optional research-ui extra "
                "(pyside6==6.11.2)"
            ) from _QT_IMPORT_ERROR

    _QT_WINDOW_FACTORY = _QtResearchWindowUnavailable

else:

    class _QtResearchWindow(QMainWindow):  # type: ignore[misc]
        """Small offscreen-testable Qt view over a real DesktopSession."""

        # G3 W2 continuous modes. "single" preserves the one-round legacy
        # behavior (no next_session factory); the continuous modes run the
        # spec §2 loop: standby → round → result → key → standby.
        _MODE_SINGLE = "single"
        _MODE_STANDBY = "standby"
        _MODE_RUNNING = "running"
        _MODE_RESULT = "result"

        def __init__(
            self,
            desktop: DesktopSession,
            *,
            consent: ConsentRecord,
            recorder: object | None = None,
            attempt_id: str | None = None,
            session_id: str | None = None,
            device_id: str = "default",
            mirrored_preview: bool = False,
            offscreen: bool = False,
            clock_ns: Callable[[], int] = time.monotonic_ns,
            clock_advance: Callable[[], None] | None = None,
            next_session: Callable[
                [], tuple[DesktopSession, ConsentRecord, str | None]
            ]
            | None = None,
            camera_options: list[tuple[int, str]] | None = None,
        ) -> None:
            super().__init__()
            self.desktop = desktop
            self.consent = consent
            self.recorder = recorder
            self.attempt_id = attempt_id
            self.session_id = session_id or desktop.session_id
            self.device_id = device_id
            self._mirrored_preview = mirrored_preview
            self._clock_ns = clock_ns
            self._clock_advance = clock_advance
            self._crop_mapping: CropMapping | None = None
            self._preview_image: QImage | None = None
            self._timer = QTimer(self)
            self._timer.setInterval(20)
            self._timer.timeout.connect(self.process_once)
            # G3 W2: standby driver (preview + face-trigger) for the
            # continuous loop. Idle unless enter_standby() starts it.
            self._standby_timer = QTimer(self)
            self._standby_timer.setInterval(100)
            self._standby_timer.timeout.connect(self._standby_tick)
            self._next_session = next_session
            self._mode = (
                self._MODE_SINGLE if next_session is None else self._MODE_STANDBY
            )
            self._result_text = ""
            # G3 W4: labeled rounds awaiting record commit (consumed by
            # the CLI tail after the window closes).
            self.completed_rounds: list[RoundComplete] = []
            self._round_started_utc: str | None = None
            # G3 W6: camera picker options as (opencv index, label).
            # None means no picker (legacy behavior); an empty list means
            # no camera was found (startup refuses with 找不到相機).
            self._camera_options = list(camera_options or [])

            if (recorder is None) != (attempt_id is None):
                raise ValueError("recorder and attempt_id must be given together")
            if recorder is not None and attempt_id is not None:
                desktop.configure_label_persistence(
                    recorder, attempt_id, actor_ref="qt-operator"
                )

            if offscreen:
                self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            self.setWindowTitle("FaceCore Research")
            self._build_ui()
            self._set_status("待開始 · ready")

        def _build_ui(self) -> None:
            root = QWidget(self)
            layout = QVBoxLayout(root)
            self.watermark_label = QLabel(self.desktop.watermark)
            self.watermark_label.setObjectName("researchWatermark")
            self.device_label = QLabel(f"裝置 · device: {self.device_id}")
            self.device_label.setObjectName("device")
            self.ttl_label = QLabel(
                f"TTL: record {self.consent.record_expires_at_utc} · "
                f"image {self.consent.image_expires_at_utc}"
            )
            self.ttl_label.setObjectName("ttl")
            self.status_label = QLabel()
            self.status_label.setObjectName("status")
            remaining_ms = self.desktop.countdown_ms_remaining(self._clock_ns())
            self.countdown_label = QLabel(f"倒數 · countdown: {remaining_ms} ms")
            self.countdown_label.setObjectName("countdown")
            self.saved_state_label = QLabel("未保存 · not saved")
            self.saved_state_label.setObjectName("savedState")
            self.identity_label = QLabel()
            self.identity_label.setObjectName("identity")
            self.guide_label = QLabel("方形引導框 · square guide: 待採集")
            self.guide_label.setObjectName("guide")
            # G3 W6: camera picker. First row is the unselected prompt so
            # no camera is preselected; the operator must pick one.
            self.camera_combo = QComboBox()
            self.camera_combo.setObjectName("cameraPicker")
            self.camera_combo.addItem("請選擇相機", None)
            for cam_index, cam_label in self._camera_options:
                self.camera_combo.addItem(cam_label, cam_index)
            self.camera_combo.currentIndexChanged.connect(
                self._camera_picked
            )
            self.preview_label = QLabel("synthetic preview")
            self.preview_label.setMinimumSize(240, 240)
            self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            consent_row = QHBoxLayout()
            self.record_consent_checkbox = QCheckBox("record consent")
            self.image_consent_checkbox = QCheckBox("image consent")
            self.record_consent_checkbox.setChecked(self.consent.record_consent)
            self.image_consent_checkbox.setChecked(self.consent.image_consent)
            consent_row.addWidget(self.record_consent_checkbox)
            consent_row.addWidget(self.image_consent_checkbox)

            controls = QHBoxLayout()
            self.start_button = QPushButton("Start")
            self.cancel_button = QPushButton("Cancel")
            self.delete_button = QPushButton("Delete")
            # G3 W3 spec §2 step 6: the only labeling keys are the
            # operator 正確／錯誤 buttons. The legacy research label
            # buttons are gone (their enrolled default took the system
            # prediction as the answer); programmatic labeling stays via
            # label_enrolled (explicit identity required) / label_unknown.
            self.correct_button = QPushButton("正確")
            self.incorrect_button = QPushButton("錯誤")
            self.start_button.clicked.connect(self.start_clicked)
            self.cancel_button.clicked.connect(self.cancel_clicked)
            self.delete_button.clicked.connect(self.delete_clicked)
            self.correct_button.clicked.connect(
                lambda _checked=False: self.press_correct()
            )
            self.incorrect_button.clicked.connect(
                lambda _checked=False: self.press_incorrect()
            )
            controls.addWidget(self.start_button)
            controls.addWidget(self.cancel_button)
            controls.addWidget(self.delete_button)
            controls.addWidget(self.correct_button)
            controls.addWidget(self.incorrect_button)

            layout.addWidget(self.watermark_label)
            layout.addWidget(self.device_label)
            layout.addWidget(self.ttl_label)
            layout.addWidget(self.status_label)
            layout.addWidget(self.countdown_label)
            layout.addWidget(self.saved_state_label)
            layout.addWidget(self.identity_label)
            layout.addWidget(self.guide_label)
            layout.addWidget(self.preview_label)
            layout.addWidget(self.camera_combo)
            layout.addLayout(consent_row)
            layout.addLayout(controls)
            self.setCentralWidget(root)
            self.cancel_button.setEnabled(False)
            self.delete_button.setEnabled(True)
            self.correct_button.setEnabled(False)
            self.incorrect_button.setEnabled(False)

        def _camera_picked(self, row: int) -> None:
            """G3 W6: operator picks a camera; nothing starts by itself."""
            picked = self.camera_combo.itemData(row)
            if picked is None:
                return
            self.device_id = str(picked)
            self.device_label.setText(f"裝置 · device: {self.device_id}")

        def selected_camera_index(self) -> int | None:
            """Picked OpenCV index, or None when the prompt row is current."""
            picked = self.camera_combo.currentData()
            return int(picked) if picked is not None else None

        def show_startup_error(self, message: str) -> None:
            """G3 W6: show a Chinese startup failure and stay put.

            No crash, no silent continue: timers stay stopped and Start
            stays disabled until the operator closes the window.
            """
            self._timer.stop()
            self._standby_timer.stop()
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.correct_button.setEnabled(False)
            self.incorrect_button.setEnabled(False)
            self._set_status(message)

        @property
        def mode(self) -> str:
            """G3 W2 continuous-loop mode (single/standby/running/result)."""
            return self._mode

        @property
        def result_text(self) -> str:
            """G3 W2 last result display text (empty outside result mode)."""
            return self._result_text

        @property
        def crop_mapping(self) -> CropMapping | None:
            return self._crop_mapping

        @property
        def preview_image(self) -> QImage | None:
            return self._preview_image

        def _set_status(self, text: str) -> None:
            self.status_label.setText(text)

        def _set_countdown(self) -> None:
            remaining = self.desktop.countdown_ms_remaining(self._clock_ns())
            self.countdown_label.setText(f"倒數 · countdown: {remaining} ms")

        def _refresh_saved_state(self) -> None:
            if self.recorder is not None and self.attempt_id is not None:
                read_mapping = getattr(self.recorder, "read_crop_mapping")
                try:
                    read_mapping(self.attempt_id)
                except Exception:
                    self.saved_state_label.setText("未保存 · not saved")
                    return
                self.saved_state_label.setText("已保存 · saved")
                return
            self.saved_state_label.setText("未保存 · not saved")

        def _set_guide(self) -> None:
            if self._crop_mapping is None:
                self.guide_label.setText("方形引導框 · square guide: 待採集")
                return
            mapping = self._crop_mapping
            self.guide_label.setText(
                "方形引導框 · square guide: "
                f"x={mapping.x} y={mapping.y} S={mapping.size} "
                f"({mapping.frame_w}x{mapping.frame_h})"
            )

        def start_clicked(self) -> None:
            """Start the existing DesktopSession after both consent checks."""
            if self._next_session is not None and self._mode != self._MODE_STANDBY:
                # Continuous loop: manual Start only fires from standby;
                # standby auto-trigger and key flow own the transitions.
                return
            if not (
                self.record_consent_checkbox.isChecked()
                and self.image_consent_checkbox.isChecked()
            ):
                self._set_status("需要 record consent 與 image consent")
                return
            try:
                self.desktop.on_start(
                    self.consent,
                    now_ns=self._clock_ns(),
                    device_id=self.device_id,
                )
            except Exception as exc:
                self._set_status(f"start refused: {type(exc).__name__}")
                return
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self._set_status("採集中 · collecting")
            self._set_countdown()
            if self._next_session is not None:
                self._mode = self._MODE_RUNNING
                self._round_started_utc = datetime.now(timezone.utc).isoformat()
                self._standby_timer.stop()
            self._timer.start()

        def cancel_clicked(self) -> None:
            """Cancel normal inference or the post-terminal fixed collector."""
            if self.desktop.state != "running":
                return
            try:
                if self.desktop.inference_terminal is not None:
                    result = self.desktop.cancel_collection(self._clock_ns())
                else:
                    result = self.desktop.on_cancel(self._clock_ns())
            except Exception as exc:
                self._set_status(f"cancel failed: {type(exc).__name__}")
                return
            self._update_terminal(result)

        def process_once(self) -> None:
            """Drive one bounded synchronous controller step for Qt tests."""
            if self.desktop.state != "running":
                self._timer.stop()
                return
            try:
                result = self.desktop.run_until_terminal(max_steps=50)
            except Exception as exc:
                self._set_status(f"processing failed: {type(exc).__name__}")
                self.desktop.close()
                self._timer.stop()
                # Geometry/staging failures are fail-closed signals, not
                # swallowed UI noise: surface the status and propagate so
                # the CLI refuses the commit instead of masking drift.
                raise
            if self._clock_advance is not None:
                self._clock_advance()
            self._set_countdown()
            if result is not None:
                self._update_terminal(result)
            if self.desktop.state != "running":
                self._timer.stop()
                if (
                    self._next_session is not None
                    and self.desktop.state == "terminal"
                ):
                    self._enter_result(result)

        def process_until_terminal(self, max_steps: int = 200) -> None:
            """Drive synthetic events without opening a camera."""
            if self._next_session is not None:
                # Continuous loop: pump standby until a round starts, then
                # run the round to terminal.
                for _ in range(max_steps):
                    if self._mode != self._MODE_STANDBY:
                        break
                    self._standby_tick()
            for _ in range(max_steps):
                if self.desktop.state != "running":
                    break
                self.process_once()

        def enter_standby(self) -> None:
            """Enter standby: preview + square guide, waiting for a face.

            G3 W2 spec §2 steps 3-4. Discards the finished round (without
            releasing the shared camera handle) and builds the next round
            via the factory. Requires the continuous loop (next_session).
            """
            if self._next_session is None:
                raise RuntimeError(
                    "enter_standby requires the continuous loop "
                    "(next_session factory)"
                )
            if self._mode == self._MODE_RUNNING:
                return
            if self._camera_options and self.selected_camera_index() is None:
                # G3 W6: no preselected camera; the operator must pick one
                # from the dropdown before standby starts.
                self._set_status("請選擇相機")
                return
            if self.desktop.state in ("terminal", "labeled"):
                self.desktop.detach()
            try:
                desktop, consent, attempt_id = self._next_session()
            except Exception as exc:
                # Fail-closed: a round that cannot be built (e.g. attempt
                # persistence refused) must not silently advance.
                self._set_status(f"standby failed: {type(exc).__name__}")
                return
            self.desktop = desktop
            self.consent = consent
            # The round owns its crop-mapping sidecar: without a fresh
            # attempt the preview must not write to the previous round's.
            self.attempt_id = attempt_id
            self._result_text = ""
            self._round_started_utc = None
            try:
                # Standby owns the preview: open the shared source now so
                # ticks can render frames; the round's start_session
                # re-opens idempotently (same camera, never rebuilt).
                self.desktop.source.open(self.device_id)
            except Exception as exc:
                self._set_status(f"standby failed: {type(exc).__name__}")
                return
            self._mode = self._MODE_STANDBY
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.correct_button.setEnabled(False)
            self.incorrect_button.setEnabled(False)
            self._set_status("請站到鏡頭前")
            self._standby_timer.start()

        def _standby_tick(self) -> None:
            """One standby step: render preview, start a round on a face."""
            if self._mode != self._MODE_STANDBY:
                return
            try:
                packet = self.desktop.source.read()
            except Exception as exc:
                self._set_status(f"standby failed: {type(exc).__name__}")
                self._standby_timer.stop()
                return
            if packet is None:
                return
            try:
                # set_frame persists the shared crop mapping before
                # rendering, so a freshly built round (no mapping stored
                # yet) previews instead of failing on the read-back.
                self.set_frame(packet.rgb)
            except Exception as exc:
                self._set_status(f"standby failed: {type(exc).__name__}")
                self._standby_timer.stop()
                return
            try:
                observation = self.desktop.scorer(packet)
            except Exception as exc:
                self._set_status(f"standby failed: {type(exc).__name__}")
                self._standby_timer.stop()
                return
            if observation.face_count >= 1:
                self.start_clicked()

        def _enter_result(self, result: Any) -> None:
            """Show the round result and arm the 正確／錯誤 keys."""
            text = self._format_result(result)
            self._result_text = text
            self._set_status(text)
            self._mode = self._MODE_RESULT
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.correct_button.setEnabled(True)
            self.incorrect_button.setEnabled(True)

        def _format_result(self, result: Any) -> str:
            """Spec §2 step 5 display text for one terminal result."""
            if result is None:
                return "辨識未完成"
            status = result.status
            if (
                status == SessionStatus.matched
                and result.matched_identity is not None
            ):
                score: float | None = None
                for obs in self.desktop.observations:
                    if result.matched_identity in obs.identity_scores:
                        score = obs.identity_scores[result.matched_identity]
                        break
                if score is None:
                    return str(result.matched_identity)
                return f"{result.matched_identity} {score:.2f}"
            if status in (SessionStatus.timeout, SessionStatus.unknown):
                return "找不到此註冊人員"
            reason = (
                result.reason_codes[0]
                if result.reason_codes
                else status.value
            )
            return f"{status.value}：{reason}"

        def press_correct(self) -> None:
            """G3 W3 正確 key: persist the operator verdict, back to standby.

            Correct endorses what is shown: a matched round records the
            shown identity (operator-confirmed, kind enrolled); a
            not-found round records unenrolled (operator confirms absent).
            """
            self._press_key(correct=True)

        def press_incorrect(self) -> None:
            """G3 W3 錯誤 key: persist the operator verdict, back to standby.

            Incorrect never records the system prediction: the label is
            uncertain with no identity, so a misrecognition is never
            auto-recorded as correct.
            """
            self._press_key(correct=False)

        def _press_key(self, *, correct: bool) -> None:
            if self._next_session is None:
                raise RuntimeError(
                    "label keys require the continuous loop "
                    "(next_session factory)"
                )
            if self._mode != self._MODE_RESULT:
                return
            if not self.desktop.has_label_persistence:
                # Fail-closed: a verdict that cannot be persisted must not
                # be silently dropped by advancing to the next round.
                self._set_status("標註未綁定，無法落盤")
                return
            try:
                if correct:
                    identity = self.desktop.display_identity()
                    if identity is None:
                        self.desktop.label_terminal(None, kind="unenrolled")
                        label_kind, label_identity = "unenrolled", None
                    else:
                        self.desktop.label_terminal(identity, kind="enrolled")
                        label_kind, label_identity = "enrolled", identity
                else:
                    self.desktop.label_terminal(None, kind="uncertain")
                    label_kind, label_identity = "uncertain", None
            except Exception as exc:
                self._set_status(f"標註失敗：{type(exc).__name__}")
                return
            terminal = self.desktop.terminal
            if terminal is not None:
                # G3 W4: queue the labeled round for record commit; the
                # CLI tail commits after the window closes.
                self.completed_rounds.append(
                    RoundComplete(
                        session_id=self.desktop.session_id,
                        attempt_id=self.attempt_id,
                        terminal=terminal,
                        observations=tuple(self.desktop.observations),
                        label_kind=label_kind,
                        label_identity=label_identity,
                        profile_version=self.desktop.profile_version,
                        started_utc=self._round_started_utc or "",
                    )
                )
            self.correct_button.setEnabled(False)
            self.incorrect_button.setEnabled(False)
            self._set_status("已標註 · labeled")
            self._refresh_saved_state()
            self.enter_standby()

        def _update_terminal(self, result: Any) -> None:
            if (
                self.desktop.state == "running"
                and self.desktop.inference_terminal is not None
            ):
                self._set_status(
                    f"辨識已鎖定 · collecting until deadline ({result.status.value})"
                )
                return
            self.cancel_button.setEnabled(False)
            self._set_status(result.status.value)
            identity = self.desktop.display_identity()
            self.identity_label.setText(identity or "")

        def delete_clicked(self) -> None:
            """Delete the session bundle plus linked attempts (operator action).

            Atomic from the caller's view: the recorder removes the session
            bundle directory and cascades to linked attempts, then the
            desktop is flagged deleted so the CLI never commits afterwards.
            A False return or any error fails closed: no deleted flag, no
            success status, and the CLI never reports rc0 for it.
            """
            self._timer.stop()
            self._standby_timer.stop()
            try:
                self.desktop.close()
            except Exception as exc:
                self.desktop.mark_delete_failed()
                self._set_status(f"delete failed: {type(exc).__name__}")
                return
            try:
                delete_bundle = getattr(self.recorder, "delete", None)
                if delete_bundle is None:
                    raise AttributeError("recorder has no delete method")
                if delete_bundle(self.session_id) is not True:
                    raise RuntimeError("recorder.delete reported incomplete")
            except Exception as exc:
                self.desktop.mark_delete_failed()
                self._set_status(f"delete failed: {type(exc).__name__}")
                return
            self.desktop.mark_deleted()
            self._set_status("已刪除 · deleted")
            self._refresh_saved_state()

        def label_enrolled(self, identity: str | None = None) -> None:
            """Persist an enrolled evaluator label without feeding inference.

            G3 W3: the identity must come from the caller (operator key
            press or explicit research input). A missing identity refuses
            instead of taking the system prediction as the answer.
            """
            if identity is None:
                self._set_status("標註需要指定身份，不採用系統預測")
                return
            self.desktop.label_terminal(identity, kind="enrolled")
            self._set_status("已標註 · labeled")
            self._refresh_saved_state()

        def label_unknown(self) -> None:
            """Persist an unenrolled label without displaying a guessed name."""
            self.desktop.label_terminal(None, kind="unenrolled")
            self._set_status("已標註 unknown · labeled")
            self._refresh_saved_state()

        def set_frame(self, frame: np.ndarray) -> CropMapping:
            """Update preview and persist the shared crop mapping sidecar.

            Direct-drive path (tests, manual use): computes the mapping from
            the given frame, persists it, and renders the overlay.
            """
            mapping = center_square_crop(
                int(frame.shape[1]),
                int(frame.shape[0]),
                mirrored_preview=self._mirrored_preview,
            )
            if self.recorder is not None and self.attempt_id is not None:
                record_mapping = getattr(self.recorder, "record_crop_mapping")
                record_mapping(self.attempt_id, mapping.to_dict())
                persisted = getattr(self.recorder, "read_crop_mapping")(self.attempt_id)
                if persisted != mapping.to_dict():
                    raise ValueError(
                        "preview mapping diverged from persisted capture mapping"
                    )
            return self.render_full_frame(frame, mapping)

        def render_full_frame(
            self, frame: np.ndarray, mapping: CropMapping | None = None
        ) -> CropMapping:
            """Render the original full frame with the guide overlay.

            A.7 first round: the preview shows the full source frame with
            the square guide drawn from the same mapping the scorer
            consumed. Never re-crops: the scorer-side capture adapter owns
            the mapping, and this sink only reads it back for the overlay.
            """
            resolved = mapping
            if resolved is None:
                resolved = center_square_crop(
                    int(frame.shape[1]),
                    int(frame.shape[0]),
                    mirrored_preview=self._mirrored_preview,
                )
            if self.recorder is not None and self.attempt_id is not None:
                persisted = getattr(self.recorder, "read_crop_mapping")(self.attempt_id)
                if persisted != resolved.to_dict():
                    raise ValueError(
                        "preview mapping diverged from persisted capture mapping"
                    )
            self._crop_mapping = resolved
            self._set_guide()
            self._refresh_saved_state()
            shown = self._overlay_guide(frame, resolved)
            height, width = shown.shape[:2]
            image = QImage(
                shown.data,
                width,
                height,
                int(shown.strides[0]),
                QImage.Format.Format_RGB888,
            ).copy()
            self._preview_image = image
            self.preview_label.setPixmap(QPixmap.fromImage(image))
            return resolved

        def _overlay_guide(
            self, frame: np.ndarray, mapping: CropMapping
        ) -> np.ndarray:
            """Draw the square guide over the full frame (same mapping)."""
            # Overlay contract: render the full source frame and outline the
            # exact crop rectangle instead of showing only the cropped pixmap.
            overlay = np.ascontiguousarray(frame).copy()
            x0, y0, size = mapping.x, mapping.y, mapping.size
            x1, y1 = x0 + size, y0 + size
            overlay[y0:y1, x0 : x0 + 1, :] = (0, 255, 0)
            overlay[y0:y1, x1 - 1 : x1, :] = (0, 255, 0)
            overlay[y0 : y0 + 1, x0:x1, :] = (0, 255, 0)
            overlay[y1 - 1 : y1, x0:x1, :] = (0, 255, 0)
            if mapping.mirrored_preview:
                overlay = np.ascontiguousarray(overlay[:, ::-1, :])
            return overlay

        def closeEvent(self, event: Any) -> None:
            self._timer.stop()
            self._standby_timer.stop()
            self.desktop.close()
            event.accept()

    _QT_WINDOW_FACTORY = _QtResearchWindow


QtResearchWindow = _QT_WINDOW_FACTORY
