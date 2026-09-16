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

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from facecore.live.desktop import DesktopSession
from facecore.research.records import ConsentRecord


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
    mapping = center_square_crop(
        width, height, mirrored_preview=mirrored_preview
    )
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


try:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QCheckBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised without extra
    _QT_IMPORT_ERROR = exc

    class _QtResearchWindow:
        """Helpful failure when the optional research-ui extra is absent."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "QtResearchWindow requires the optional research-ui extra "
                "(pyside6==6.11.2)"
            ) from _QT_IMPORT_ERROR

else:

    class _QtResearchWindow(QMainWindow):  # type: ignore[no-redef]
        """Small offscreen-testable Qt view over a real DesktopSession."""

        def __init__(
            self,
            desktop: DesktopSession,
            *,
            consent: ConsentRecord,
            recorder: object | None = None,
            attempt_id: str | None = None,
            device_id: str = "default",
            mirrored_preview: bool = False,
            offscreen: bool = False,
            clock_ns: Callable[[], int] = time.monotonic_ns,
        ) -> None:
            super().__init__()
            self.desktop = desktop
            self.consent = consent
            self.recorder = recorder
            self.attempt_id = attempt_id
            self.device_id = device_id
            self._mirrored_preview = mirrored_preview
            self._clock_ns = clock_ns
            self._crop_mapping: CropMapping | None = None
            self._preview_image: QImage | None = None
            self._timer = QTimer(self)
            self._timer.setInterval(20)
            self._timer.timeout.connect(self.process_once)

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
            self.status_label = QLabel()
            self.status_label.setObjectName("status")
            self.identity_label = QLabel()
            self.identity_label.setObjectName("identity")
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
            self.enrolled_label_button = QPushButton("Label enrolled")
            self.unknown_label_button = QPushButton("Label unknown")
            self.start_button.clicked.connect(self.start_clicked)
            self.cancel_button.clicked.connect(self.cancel_clicked)
            self.enrolled_label_button.clicked.connect(
                lambda _checked=False: self.label_enrolled()
            )
            self.unknown_label_button.clicked.connect(
                lambda _checked=False: self.label_unknown()
            )
            controls.addWidget(self.start_button)
            controls.addWidget(self.cancel_button)
            controls.addWidget(self.enrolled_label_button)
            controls.addWidget(self.unknown_label_button)

            layout.addWidget(self.watermark_label)
            layout.addWidget(self.status_label)
            layout.addWidget(self.identity_label)
            layout.addWidget(self.preview_label)
            layout.addLayout(consent_row)
            layout.addLayout(controls)
            self.setCentralWidget(root)
            self.cancel_button.setEnabled(False)
            self.enrolled_label_button.setEnabled(False)
            self.unknown_label_button.setEnabled(False)

        @property
        def crop_mapping(self) -> CropMapping | None:
            return self._crop_mapping

        @property
        def preview_image(self) -> QImage | None:
            return self._preview_image

        def _set_status(self, text: str) -> None:
            self.status_label.setText(text)

        def start_clicked(self) -> None:
            """Start the existing DesktopSession after both consent checks."""
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
                result = self.desktop.run_until_terminal(max_steps=1)
            except Exception as exc:
                self._set_status(f"processing failed: {type(exc).__name__}")
                self.desktop.close()
                self._timer.stop()
                return
            if result is not None:
                self._update_terminal(result)
            if self.desktop.state != "running":
                self._timer.stop()

        def process_until_terminal(self, max_steps: int = 200) -> None:
            """Drive synthetic events without opening a camera."""
            for _ in range(max_steps):
                if self.desktop.state != "running":
                    break
                self.process_once()

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
            self.enrolled_label_button.setEnabled(identity is not None)
            self.unknown_label_button.setEnabled(True)

        def label_enrolled(self, identity: str | None = None) -> None:
            """Persist an enrolled evaluator label without feeding inference."""
            if identity is None:
                identity = self.desktop.display_identity()
            if identity is None:
                return
            self.desktop.label_terminal(identity, kind="enrolled")
            self.enrolled_label_button.setEnabled(False)
            self.unknown_label_button.setEnabled(False)
            self._set_status("已標註 · labeled")

        def label_unknown(self) -> None:
            """Persist an unenrolled label without displaying a guessed name."""
            self.desktop.label_terminal(None, kind="unenrolled")
            self.enrolled_label_button.setEnabled(False)
            self.unknown_label_button.setEnabled(False)
            self._set_status("已標註 unknown · labeled")

        def set_frame(self, frame: np.ndarray) -> CropMapping:
            """Update preview and persist the shared crop mapping sidecar."""
            cropped, mapping = crop_frame(
                frame, mirrored_preview=self._mirrored_preview
            )
            self._crop_mapping = mapping
            if self.recorder is not None and self.attempt_id is not None:
                record_mapping = getattr(self.recorder, "record_crop_mapping")
                record_mapping(self.attempt_id, mapping.to_dict())
            shown = preview_frame(cropped, mapping)
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
            return mapping

        def closeEvent(self, event: Any) -> None:
            self._timer.stop()
            self.desktop.close()
            event.accept()


QtResearchWindow = _QtResearchWindow
