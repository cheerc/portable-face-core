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
from pathlib import Path
import time
from typing import Any, TYPE_CHECKING

import numpy as np

from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    SessionResult,
    SessionStatus,
)
from facecore.live.desktop import DesktopSession
from facecore.research.records import ConsentRecord

if TYPE_CHECKING:
    from facecore.research.recorder import ResearchRecorder


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
    # D7-A W1: the gallery this round ran against. The gallery is loaded
    # once at App startup, so every round carries the same report — that
    # is the point: the demo row records which gallery produced it.
    # Optional because record-mode callers construct this without one.
    gallery_load_report: Any | None = None


@dataclass(frozen=True)
class _SyntheticObservation:
    """Minimal observation shape for failure classification tests."""

    face_count: int = 0
    quality_pass: bool = False
    quality_reasons: tuple[str, ...] = ()


def _quality_reason_text(observations: tuple[Any, ...]) -> str:
    """Collect the real quality reasons across a round's observations.

    A quality rejection is actionable only if the operator learns WHICH
    one — 「品質不合格」 alone cannot be acted on, while the concrete
    reasons (exposure, blur, …) can. Returns a parenthesised suffix, or
    an empty string when the round carried no usable reason text.
    """
    reasons: set[str] = set()
    for obs in observations:
        reasons.update(getattr(obs, "quality_reasons", ()) or ())
    if not reasons:
        return ""
    return f"（{'、'.join(sorted(reasons))}）"


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

        # G3 R1 Start-gated modes. "single" preserves the one-round legacy
        # behavior (no next_session factory); the continuous modes run the
        # R1 spec §2 loop: ready → running → result → ready. The camera
        # opens only on Start and releases at every terminal; nothing
        # previews or auto-starts from a pick or a key press.
        _MODE_SINGLE = "single"
        _MODE_READY = "ready"
        _MODE_RUNNING = "running"
        _MODE_RESULT = "result"

        def __init__(
            self,
            desktop: DesktopSession,
            *,
            consent: ConsentRecord,
            recorder: ResearchRecorder | None = None,
            attempt_id: str | None = None,
            session_id: str | None = None,
            device_id: str = "default",
            mirrored_preview: bool = False,
            offscreen: bool = False,
            clock_ns: Callable[[], int] = time.monotonic_ns,
            clock_advance: Callable[[], None] | None = None,
            next_session: Callable[[], tuple[DesktopSession, ConsentRecord, str | None]]
            | None = None,
            camera_options: list[tuple[int, str]] | None = None,
            results_csv: Path | None = None,
            background_inference: bool = False,
            gallery: Any = None,
            enrollment_dir: Path | None = None,
            load_report: Any = None,
            demo_results_csv: Path | None = None,
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
            self._next_session = next_session
            self._mode = self._MODE_SINGLE if next_session is None else self._MODE_READY
            self._result_text = ""
            # G3 W4: labeled rounds awaiting record commit (consumed by
            # the CLI tail after the window closes).
            self.completed_rounds: list[RoundComplete] = []
            self._round_started_utc: str | None = None
            # G3 R1 PR-A change 2: per-round clock anchor (this round's
            # open-time clock value); re-taken on every _start_round.
            self._round_start_ns: int | None = None
            # D2: guards the Ready->Result transition to run exactly once
            # per round, since the terminal is now observed on a tick
            # that may find the desktop already terminal.
            self._result_entered = False
            # G3 W6: camera picker options as (opencv index, label).
            # None means no picker (legacy behavior); an empty list means
            # no camera was found (startup refuses with 找不到相機).
            self._camera_options = list(camera_options or [])
            # G3 W8: results.csv path for commit-on-label (None keeps the
            # W4 memory-queue behavior for callers without a csv target).
            self._results_csv = results_csv
            # D3b: plaintext demo result file for the non-recording mode.
            self.demo_results_csv = demo_results_csv
            # D2: opt-in worker-driven inference (see _start_round).
            self._background_inference = background_inference
            # D3: gallery, enrollment folder and startup loading report.
            self.gallery = gallery
            self.enrollment_dir = enrollment_dir
            self.load_report = load_report or getattr(gallery, "load_report", None)

            if (recorder is None) != (attempt_id is None):
                raise ValueError("recorder and attempt_id must be given together")
            if recorder is not None and attempt_id is not None:
                desktop.configure_label_persistence(
                    recorder, attempt_id, actor_ref="qt-operator"
                )
            if demo_results_csv is not None and recorder is None:
                # D3b: demo mode has no encrypted sidecar, so the round's
                # verdict needs the plaintext sink for the guard below to
                # accept it. The rounds the next_session factory builds carry
                # this via their own DesktopSession(demo_label_sink=...); this
                # call covers the initial desktop and any caller that hands us
                # an unconfigured one. Commander decision
                # d-20260929193128013174-12 item 2.
                desktop.configure_demo_label_persistence(
                    demo_results_csv, actor_ref="qt-operator"
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
            self.enrollment_label = QLabel()
            self.enrollment_label.setObjectName("enrollmentReport")
            self._update_enrollment_ui()
            self.status_label = QLabel()
            self.status_label.setObjectName("status")
            # D7-A #141: the header-mismatch message is several lines long.
            # A QLabel does not wrap by default, and it renders newlines as
            # blanks, so without this the five-line explanation collapses
            # into one unreadable strip that runs past the window edge —
            # which would leave the operator with the same six characters
            # the fix was meant to replace.
            self.status_label.setWordWrap(True)
            remaining_ms = self.desktop.countdown_ms_remaining(self._clock_ns())
            self.countdown_label = QLabel(f"倒數 · countdown: {remaining_ms} ms")
            self.countdown_label.setObjectName("countdown")
            self.saved_state_label = QLabel("未保存 · not saved")
            self.saved_state_label.setObjectName("savedState")
            self.identity_label = QLabel()
            self.identity_label.setObjectName("identity")
            self.guide_label = QLabel("方形引導框 · square guide: 待採集")
            self.guide_label.setObjectName("guide")

            # D3: result verification panel (candidate thumbnail, ID, scores,
            # frames, failure reason)
            self.candidate_thumbnail_label = QLabel("無縮圖")
            self.candidate_thumbnail_label.setObjectName("candidateThumbnail")
            self.candidate_thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.candidate_thumbnail_label.setMinimumSize(120, 120)

            self.result_identity_label = QLabel()
            self.result_identity_label.setObjectName("resultIdentity")

            self.scores_label = QLabel()
            self.scores_label.setObjectName("resultScores")

            self.frames_label = QLabel()
            self.frames_label.setObjectName("resultFrames")

            self.failure_reason_label = QLabel()
            self.failure_reason_label.setObjectName("failureReason")

            # G3 W6: camera picker. First row is the unselected prompt so
            # no camera is preselected; the operator must pick one.
            self.camera_combo = QComboBox()
            self.camera_combo.setObjectName("cameraPicker")
            self.camera_combo.addItem("請選擇相機", None)
            for cam_index, cam_label in self._camera_options:
                self.camera_combo.addItem(cam_label, cam_index)
            self.camera_combo.currentIndexChanged.connect(self._camera_picked)
            self.preview_label = QLabel("synthetic preview")
            self.preview_label.setMinimumSize(240, 240)
            self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # G3-w P3: the two per-round dropdowns are removed (spec
            # §4.1). Ground truth now comes from the operator's
            # annotation buttons: ✓ writes the derived values, ✗ opens
            # the disclosure area below, skip writes an explicit value.
            # The per-round values live in `_annot_probe_kind` /
            # `_annot_identity`, reset at every round start so no value
            # can leak from the previous round (decision: no residual).
            # Combos never return: free text is rejected (a mistyped
            # identity is a silent ground-truth error), every option is
            # a closed domain — gallery keys, non-target stems, or the
            # runbook-defined `outsider` marker.
            self._annot_probe_kind = ""
            self._annot_identity = ""
            self._nontarget_dir: str | None = None
            self._nontarget_stems: list[str] = []

            self.ground_truth_combo = QComboBox()
            self.ground_truth_combo.setObjectName("groundTruthPicker")
            self.nontarget_dir_button = QPushButton("選 non-target 資料夾")
            self.nontarget_dir_button.setObjectName("nontargetDirPicker")
            # Visibility is owned by the disclosure parent alone: hiding
            # the children here as well would keep them hidden when the
            # parent is shown (Qt does not re-show an explicitly hidden
            # child with its parent), leaving the disclosure area empty
            # with no selectable menu.
            self.nontarget_dir_button.clicked.connect(
                self._pick_nontarget_dir
            )
            self.ground_truth_combo.currentIndexChanged.connect(
                self._ground_truth_picked
            )

            disclosure_row = QHBoxLayout()
            disclosure_row.addWidget(QLabel("實際是誰"))
            disclosure_row.addWidget(self.ground_truth_combo)
            disclosure_row.addWidget(self.nontarget_dir_button)
            self.disclosure_widget = QWidget()
            self.disclosure_widget.setLayout(disclosure_row)
            self.disclosure_widget.setVisible(False)

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
            # G3-w P3: skip is a peer of ✓／✗ (spec §4.2), not a field
            # inside the disclosure area. It is unavailable on
            # `invalid_input` rounds (spec §4.5) and never shown for
            # `cancelled`／`error` (spec §4.6, no buttons at all).
            self.skip_button = QPushButton("略過")
            self.skip_button.setObjectName("skipVerdict")
            # D2b (operator decision d-20260929173733098323-10): with the
            # lens kept open after a round, the operator needs an explicit
            # way to run another round on the SAME open camera, and an
            # explicit way to shut it. 再辨識 never re-opens the lens;
            # 停止相機 is the only in-loop release.
            self.recognize_again_button = QPushButton("再次辨識")
            self.recognize_again_button.setObjectName("recognizeAgain")
            self.stop_camera_button = QPushButton("停止相機")
            self.stop_camera_button.setObjectName("stopCamera")
            self.start_button.clicked.connect(self.start_clicked)
            self.cancel_button.clicked.connect(self.cancel_clicked)
            self.delete_button.clicked.connect(self.delete_clicked)
            self.recognize_again_button.clicked.connect(
                self.recognize_again_clicked
            )
            self.stop_camera_button.clicked.connect(self.stop_camera_clicked)
            self.correct_button.clicked.connect(
                lambda _checked=False: self.press_correct()
            )
            self.incorrect_button.clicked.connect(
                lambda _checked=False: self.press_incorrect()
            )
            self.skip_button.clicked.connect(
                lambda _checked=False: self.press_skip()
            )
            controls.addWidget(self.start_button)
            controls.addWidget(self.cancel_button)
            controls.addWidget(self.delete_button)
            controls.addWidget(self.correct_button)
            controls.addWidget(self.incorrect_button)
            controls.addWidget(self.skip_button)
            controls.addWidget(self.recognize_again_button)
            controls.addWidget(self.stop_camera_button)

            layout.addWidget(self.watermark_label)
            layout.addWidget(self.device_label)
            layout.addWidget(self.ttl_label)
            layout.addWidget(self.enrollment_label)
            layout.addWidget(self.status_label)
            layout.addWidget(self.countdown_label)
            layout.addWidget(self.saved_state_label)
            layout.addWidget(self.identity_label)
            layout.addWidget(self.guide_label)
            layout.addWidget(self.preview_label)
            layout.addWidget(self.candidate_thumbnail_label)
            layout.addWidget(self.result_identity_label)
            layout.addWidget(self.scores_label)
            layout.addWidget(self.frames_label)
            layout.addWidget(self.failure_reason_label)
            layout.addWidget(self.camera_combo)
            layout.addWidget(self.disclosure_widget)
            layout.addLayout(consent_row)
            layout.addLayout(controls)
            self.setCentralWidget(root)
            self.cancel_button.setEnabled(False)
            self.delete_button.setEnabled(True)
            self.correct_button.setEnabled(False)
            self.incorrect_button.setEnabled(False)
            self.skip_button.setEnabled(False)
            self.skip_button.setVisible(self._next_session is not None)
            # D2b: both lens controls are meaningful only in the
            # Start-gated loop, and only once a round has opened the
            # camera. They are hidden in the legacy single-round path.
            self.recognize_again_button.setVisible(self._next_session is not None)
            self.stop_camera_button.setVisible(self._next_session is not None)
            self.recognize_again_button.setEnabled(False)
            self.stop_camera_button.setEnabled(False)

        def _refresh_ground_truth_options(self) -> None:
            """Rebuild the disclosure combo for the current round result.

            Closed domain, never free text (spec §4.3): gallery keys,
            runtime non-target folder stems, and the runbook-defined
            `outsider` marker as the no-roster path. `outsider` is the
            runbook-defined marker, never invented here.
            """
            combo = self.ground_truth_combo
            combo.blockSignals(True)
            try:
                combo.clear()
                gallery = getattr(self, "gallery", None)
                embeddings = getattr(gallery, "embeddings", None) or {}
                # One option set for every status: gallery keys (grid 2
                # corrections and the matched identity itself), runtime
                # non-target stems, and the runbook-defined `outsider`
                # marker (grid 3 and the no-roster path, spec §4.3). No
                # per-status branching: the operator — not the system
                # result — decides which one this round was.
                for identity in sorted(embeddings):
                    combo.addItem(identity, ("target", identity))
                for stem in self._nontarget_stems:
                    combo.addItem(stem, ("nontarget", stem))
                combo.addItem("outsider（隨機路人）", ("nontarget", "outsider"))
            finally:
                combo.blockSignals(False)

        def _ground_truth_picked(self, _row: int = -1) -> None:
            """Store the operator's disclosure pick as this round's truth.

            The pick is a (probe_kind, identity) pair, never a bare
            identity: the two columns are one fact, so they are stored
            together and cannot drift apart between pick and write.
            """
            data = self.ground_truth_combo.currentData()
            if isinstance(data, tuple) and len(data) == 2:
                self._annot_probe_kind, self._annot_identity = data
            else:
                self._annot_probe_kind, self._annot_identity = "", ""

        def _pick_nontarget_dir(self) -> None:
            """Let the operator point at a non-target folder (spec §4.4).

            The folder is enumerated at runtime; no filename is ever
            hard-coded or written into the repo. Changing the folder
            contents takes effect without any code change.
            """
            from pathlib import Path

            from PySide6.QtWidgets import QFileDialog

            from facecore.live.frame_pipeline import GALLERY_IMAGE_SUFFIXES

            picked = QFileDialog.getExistingDirectory(
                self, "選 non-target 資料夾"
            )
            if not picked:
                return
            exts = {e.lower() for e in GALLERY_IMAGE_SUFFIXES}
            stems = sorted(
                p.stem
                for p in Path(picked).iterdir()
                if p.is_file() and p.suffix.lower() in exts
            )
            self._nontarget_dir = picked
            self._nontarget_stems = stems
            self._refresh_ground_truth_options()
            self._set_status(f"已選 non-target 資料夾（{len(stems)} 個候選）")

        def _probe_kind_value(self) -> str:
            """The `probe_kind` to record on the next round.

            The value comes from the operator's button annotation, not
            from combo selection (the per-round combos were removed in
            P3). The write-time-read property the writer_wiring guard
            protects still holds: the annotation is read here, at write
            time, never latched earlier.
            """
            return self._annot_probe_kind

        def _presenting_identity_value(self) -> str:
            """The `presenting_identity` to record on the next round.

            Same source change as `_probe_kind_value`: the value comes
            from the operator's button annotation, not from combo
            selection. Empty means 「沒有記錄」, not a value.
            """
            return self._annot_identity

        def _camera_picked(self, row: int) -> None:
            """G3 R1: a pick only routes the device; nothing opens.

            R1 spec §2-1: enumeration and selection must not open a
            video stream, fetch frames, or build a preview. The round
            starts only when the operator presses Start.
            """
            picked = self.camera_combo.itemData(row)
            if picked is None:
                return
            self.device_id = str(picked)
            self.device_label.setText(f"裝置 · device: {self.device_id}")
            if self._next_session is not None and self._mode == self._MODE_READY:
                self._set_status("已選相機，按 Start 開始")
                self._refresh_start_enabled()

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
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.correct_button.setEnabled(False)
            self.incorrect_button.setEnabled(False)
            self._set_status(message)

        def _update_enrollment_ui(self) -> None:
            """Update enrollment report with expected/loaded counts and failures."""
            if self.load_report is not None:
                rep = self.load_report
                if rep.failures:
                    fails_str = "、".join(
                        f"{f.filename}（{f.reason}）" for f in rep.failures
                    )
                    head = (
                        f"註冊組：應載入 {rep.expected_count} 人，"
                        f"實際載入 {rep.loaded_count} 人"
                    )
                    detail = f"註冊失敗（{len(rep.failures)} 檔）：{fails_str}"
                    self.enrollment_label.setText(f"{head}\n{detail}")
                else:
                    self.enrollment_label.setText(
                        f"註冊組：應載入 {rep.expected_count} 人，"
                        f"實際載入 {rep.loaded_count} 人（全部成功）"
                    )
            elif (
                self.gallery is not None
                and getattr(self.gallery, "embeddings", None)
            ):
                count = len(self.gallery.embeddings)
                self.enrollment_label.setText(
                    f"註冊組：應載入 {count} 人，實際載入 {count} 人（全部成功）"
                )
            else:
                self.enrollment_label.setText("註冊組：未載入")

        def find_identity_photo(self, ident: str) -> Path | None:
            """Locate the local enrollment photo file for identity (stem).

            Directly reads the original enrollment photo file without copying.
            Returns None if not found or unreadable.
            """
            if not ident:
                return None
            if self.gallery is not None:
                sources = getattr(self.gallery, "identity_sources", None)
                if sources and ident in sources:
                    path = sources[ident]
                    if isinstance(path, Path) and path.is_file():
                        return path
            enr_dir = self.enrollment_dir
            if enr_dir is None and self.load_report is not None:
                enr_dir = self.load_report.source_dir
            if enr_dir is not None and enr_dir.is_dir():
                for suffix in (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tiff",
                    ".tif",
                    ".webp",
                ):
                    candidate = enr_dir / f"{ident}{suffix}"
                    if candidate.is_file():
                        return candidate
                    candidate_upper = enr_dir / f"{ident}{suffix.upper()}"
                    if candidate_upper.is_file():
                        return candidate_upper
                try:
                    for p in enr_dir.iterdir():
                        if p.is_file() and p.stem == ident:
                            return p
                except Exception:
                    pass
            return None

        def _display_thumbnail(self, photo_path: Path | None) -> None:
            """Display local candidate thumbnail directly without copying."""
            if photo_path is not None and photo_path.is_file():
                try:
                    pixmap = QPixmap(str(photo_path))
                    if not pixmap.isNull():
                        scaled = pixmap.scaled(
                            120,
                            120,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        self.candidate_thumbnail_label.setPixmap(scaled)
                        self.candidate_thumbnail_label.setToolTip(
                            f"註冊照片：{photo_path.name}"
                        )
                        return
                except Exception:
                    pass
            self.candidate_thumbnail_label.clear()
            self.candidate_thumbnail_label.setText("無縮圖")

        def _clear_result_panel(self) -> None:
            """Clear result details panel when returning to ready or starting."""
            self.candidate_thumbnail_label.clear()
            self.candidate_thumbnail_label.setText("無縮圖")
            self.result_identity_label.setText("")
            self.scores_label.setText("")
            self.frames_label.setText("")
            self.failure_reason_label.setText("")

        def _update_result_details(self, result: Any) -> None:
            """Update verification result details (scores, frames, causes)."""
            if result is None:
                self.result_identity_label.setText("身分 · identity: 辨識未完成")
                self.scores_label.setText("Top 1: 無 · Top 2: 無 · 差距: 無")
                self.frames_label.setText("有效幀／所需幀: 0/0")
                self.failure_reason_label.setText("失敗原因: 辨識未完成")
                self._display_thumbnail(None)
                return

            profile = getattr(
                self.desktop,
                "profile",
                getattr(getattr(self.desktop, "_engine", None), "profile", None),
            )
            required_support = getattr(profile, "required_support", 3)
            review_thresh = getattr(profile, "review_threshold", 0.30)

            observations = tuple(self.desktop.observations)
            status = result.status
            is_matched = (
                status == SessionStatus.matched
                and result.matched_identity is not None
            )

            top1_ident: str | None = None
            top1_score: float | None = None
            top2_ident: str | None = None
            top2_score: float | None = None
            margin: float | None = None
            effective_frames = 0
            thumbnail_ident: str | None = None

            if is_matched:
                matched_id = str(result.matched_identity)
                thumbnail_ident = matched_id
                self.result_identity_label.setText(f"身分 · identity: {matched_id}")
                self.identity_label.setText(matched_id)
                self.failure_reason_label.setText("失敗原因: 無（辨識成功）")

                scored_obs = [
                    o
                    for o in observations
                    if o.identity_scores and matched_id in o.identity_scores
                ]
                if scored_obs:
                    scored_obs.sort(
                        key=lambda o: (
                            o.identity_scores.get(matched_id, 0.0),
                            getattr(o, "quality_rank", 0.0),
                        ),
                        reverse=True,
                    )
                    best_obs = scored_obs[0]
                    sorted_scores = sorted(
                        best_obs.identity_scores.items(),
                        key=lambda it: it[1],
                        reverse=True,
                    )
                    top1_ident, top1_score = sorted_scores[0]
                    if len(sorted_scores) > 1:
                        top2_ident, top2_score = sorted_scores[1]
                        margin = top1_score - top2_score
                else:
                    top1_ident = matched_id
                    top1_score = None

                support_count = sum(
                    1
                    for o in observations
                    if o.identity_scores
                    and getattr(o, "quality_pass", False)
                    and getattr(o, "face_count", 0) == 1
                    and max(o.identity_scores.items(), key=lambda it: it[1])[0]
                    == matched_id
                )
                effective_frames = max(support_count, required_support)
            else:
                self.identity_label.setText("")
                usable_frames = [
                    o
                    for o in observations
                    if getattr(o, "quality_pass", False)
                    and getattr(o, "face_count", 0) == 1
                    and o.identity_scores
                ]
                if usable_frames:
                    usable_frames.sort(
                        key=lambda o: (
                            getattr(o, "quality_rank", 0.0),
                            -getattr(o, "sequence", 0),
                        ),
                        reverse=True,
                    )
                    best_frame = usable_frames[0]
                    sorted_scores = sorted(
                        best_frame.identity_scores.items(),
                        key=lambda it: it[1],
                        reverse=True,
                    )
                    top1_ident, top1_score = sorted_scores[0]
                    thumbnail_ident = top1_ident
                    if len(sorted_scores) > 1:
                        top2_ident, top2_score = sorted_scores[1]
                        margin = top1_score - top2_score

                    support_code = next(
                        (
                            c
                            for c in result.reason_codes
                            if c.startswith("support_") and "_of_" in c
                        ),
                        None,
                    )
                    if support_code:
                        try:
                            effective_frames = int(support_code.split("_")[1])
                        except (ValueError, IndexError):
                            effective_frames = sum(
                                1
                                for o in usable_frames
                                if o.identity_scores.get(top1_ident, 0.0)
                                >= review_thresh
                            )
                    else:
                        effective_frames = sum(
                            1
                            for o in usable_frames
                            if o.identity_scores.get(top1_ident, 0.0)
                            >= review_thresh
                        )
                    self.result_identity_label.setText(
                        f"候選 · candidate: {top1_ident}"
                    )
                else:
                    self.result_identity_label.setText("身分 · identity: 無候選")
                    thumbnail_ident = None

                failure_text = self.classify_failure(
                    observations=observations,
                    reason_codes=tuple(result.reason_codes),
                )
                self.failure_reason_label.setText(f"失敗原因: {failure_text}")

            if top1_ident is not None and top1_score is not None:
                top1_text = f"Top 1: {top1_ident}（相似度 {top1_score:.3f}）"
                if top2_ident is not None and top2_score is not None:
                    top2_text = f"Top 2: {top2_ident}（相似度 {top2_score:.3f}）"
                    margin_text = (
                        f"差距: {margin:.3f}" if margin is not None else "差距: 無"
                    )
                    self.scores_label.setText(
                        f"{top1_text} · {top2_text} · {margin_text}"
                    )
                else:
                    self.scores_label.setText(
                        f"{top1_text} · Top 2: 無 · 差距: 無"
                    )
            elif top1_ident is not None:
                self.scores_label.setText(
                    f"Top 1: {top1_ident} · Top 2: 無 · 差距: 無"
                )
            else:
                self.scores_label.setText("Top 1: 無 · Top 2: 無 · 差距: 無")

            self.frames_label.setText(
                f"有效幀／所需幀: {effective_frames}/{required_support}"
            )

            photo_path = (
                self.find_identity_photo(thumbnail_ident)
                if thumbnail_ident
                else None
            )
            self._display_thumbnail(photo_path)

        @property
        def mode(self) -> str:
            """G3 R1 continuous-loop mode (single/ready/running/result)."""
            return self._mode

        @property
        def result_text(self) -> str:
            """G3 W2 last result display text (empty outside result mode)."""
            return self._result_text

        @property
        def round_start_ns(self) -> int | None:
            """G3 R1 PR-A: this round's clock anchor (None before start)."""
            return self._round_start_ns

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
            """Start one round: continuous loop builds it, single reuses."""
            if self._next_session is not None:
                # G3 R1: every round starts from Ready via an explicit
                # operator Start. A pick alone never opens the camera.
                self._start_gated_round()
                return
            self._start_round()

        def _record_unlabeled_round(self) -> None:
            """D2b: persist the round the operator declined to label.

            Commander point (二). 再次辨識 lets the operator move on
            without a verdict, and a round without a verdict is still a
            round that happened — dropping it silently would make the
            demo file disagree with what the operator saw. So it is
            written with label_kind="unlabeled".

            This value is DEMO-ONLY. The research ledger's label_kind
            column is the existing enrolled／unenrolled／uncertain
            enumeration used by report.py and analysis.py, and this
            method never touches it: it writes the plaintext demo file,
            which decision -11 item 3 established neither of those
            modules reads. In record mode an unlabeled round keeps its
            pre-existing path instead — cmd_live's close-out calls
            recorder.abort(..., reason="unlabeled_close") — so research
            statistics are unaffected in both modes.
            """
            if self.demo_results_csv is None:
                return
            terminal = self.desktop.terminal
            if terminal is None:
                return
            from facecore.research.cli import (
                DemoLogHeaderMismatch,
                append_g3_demo_results_csv,
            )

            round_complete = RoundComplete(
                session_id=self.desktop.session_id,
                attempt_id=self.attempt_id,
                terminal=terminal,
                observations=tuple(self.desktop.observations),
                label_kind="unlabeled",
                label_identity=None,
                profile_version=self.desktop.profile_version,
                started_utc=self._round_started_utc or "",
                # D7-A W1: same report as the labeled path — both write
                # demo rows, so both must record which gallery they ran
                # against. It is the App-startup report, unchanged per round.
                gallery_load_report=self.load_report,
            )
            try:
                append_g3_demo_results_csv(
                    self.demo_results_csv,
                    round_complete,
                    required_support=getattr(
                        self.desktop.profile, "required_support", 0
                    ),
                    labeled_at_utc=datetime.now(timezone.utc).isoformat(),
                    # D7-A W3: forward the round's own tallies and profile so
                    # the row carries what this round already computed.
                    event_counts=self.desktop.event_counts(),
                    support_clears=self.desktop.support_clear_reasons(),
                    profile=self.desktop.profile,
                    # D7-A W0-b: the operator's ground truth for this
                    # round. Read at WRITE time, not at round start, so
                    # changing the pick before labeling is honoured — the
                    # unlabeled path below needs it for the same reason.
                    probe_kind=self._probe_kind_value(),
                    presenting_identity=self._presenting_identity_value(),
                    gallery_load_report=self.load_report,
                    # G3-w P3: the rerun path is a deliberate
                    # non-judgement, recorded as an explicit value so it
                    # cannot collide with a missing verdict (spec §5.3).
                    operator_verdict="skipped",
                )
            except OSError as exc:
                # Fail-closed: a round we cannot record must not look
                # like a round that was recorded.
                #
                # D7-A #141: a header mismatch is not a "write failed" —
                # it names the operator's next step, so the message has to
                # REACH him. A bare `except OSError:` discarded it and left
                # six characters on screen that distinguish nothing. Keep
                # the generic wording for every other OSError (disk full,
                # permissions): those have no per-file remedy to offer.
                if isinstance(exc, DemoLogHeaderMismatch):
                    self._set_status(f"紀錄寫入失敗 · {exc}")
                else:
                    self._set_status("紀錄寫入失敗")
                return
            self.completed_rounds.append(round_complete)

        def recognize_again_clicked(self) -> None:
            """D2b 再次辨識: run the next round on the SAME open camera.

            The lens is already open (that is the point of D2b), so this
            must NOT re-open it: the round is handed off through
            detach(), which stops the previous round's workers and keeps
            the shared source, exactly as R1 §2-5 already did between
            labeled rounds.

            Commander point (二): if the operator skips the verdict, the
            finished round is recorded as unlabeled rather than dropped.
            In record mode the existing close-out aborts it instead; both
            are pre-existing, non-silent paths.
            """
            if self._next_session is None:
                return
            if self._mode != self._MODE_RESULT:
                return
            if self.desktop.state == "terminal":
                # Never labeled. Record it before the desktop is detached.
                self._record_unlabeled_round()
            self._timer.stop()
            self._start_gated_round()

        def stop_camera_clicked(self) -> None:
            """D2b 停止相機: shut the lens, clear the view, back to Ready.

            This is the only in-loop release. The camera PICK is kept, so
            the next Start re-opens the same device — the operator does
            not have to choose it again.
            """
            if self._next_session is None:
                return
            self._timer.stop()
            if self.desktop.state in ("terminal", "labeled", "closed"):
                self.desktop.detach()
            try:
                self.desktop.close()
            except Exception:
                pass
            self._clear_preview()
            self._clear_result_panel()
            self._result_text = ""
            self.recognize_again_button.setEnabled(False)
            self.stop_camera_button.setEnabled(False)
            self._mode = self._MODE_READY
            self._refresh_start_enabled()
            self._set_status("相機已關閉 · camera stopped")

        def _refresh_start_enabled(self) -> None:
            """Enable Start only when a camera is picked (continuous loop)."""
            if self._next_session is None or self._mode != self._MODE_READY:
                return
            picked = (
                self.selected_camera_index() is not None
                if self._camera_options
                else True
            )
            self.start_button.setEnabled(picked)

        def _start_gated_round(self) -> None:
            """Build a fresh round for the picked camera and start it."""
            if self._next_session is None:
                raise RuntimeError(
                    "gated rounds require the continuous loop (next_session factory)"
                )
            # D2b: 再辨識 starts the next round straight from Result, so
            # the gate is Ready OR Result. Ready is the operator's Start;
            # Result is the operator's 再次辨識 on the still-open camera.
            if self._mode not in (self._MODE_READY, self._MODE_RESULT):
                return
            if not (
                self.record_consent_checkbox.isChecked()
                and self.image_consent_checkbox.isChecked()
            ):
                self._set_status("需要 record consent 與 image consent")
                return
            if (
                self._camera_options
                and self._mode == self._MODE_READY
                and self.selected_camera_index() is None
            ):
                self._set_status("請選擇相機")
                return
            if self.desktop.state in ("terminal", "labeled", "closed"):
                # detach() stops this round's workers and KEEPS the shared
                # source: the camera opened by the previous round is still
                # open, which is what lets 再辨識 skip the reopen.
                self.desktop.detach()
            try:
                desktop, consent, attempt_id = self._next_session()
            except Exception as exc:
                self._set_status(f"建輪失敗：{type(exc).__name__}")
                return
            self.desktop = desktop
            self.consent = consent
            self.attempt_id = attempt_id
            self._result_text = ""
            self._round_started_utc = None
            # D2b: the previous round's result is gone the moment the next
            # one starts, so no stale result text, panels or countdown can
            # bleed into it. (At Result itself nothing is cleared — the
            # operator is still reading it against the live preview.)
            self._clear_result_panel()
            self.identity_label.setText("")
            self.recognize_again_button.setEnabled(False)
            self.stop_camera_button.setEnabled(False)
            # D2b: coming from Result the camera is already open, so the
            # round must NOT re-open it. From Ready the operator's Start
            # is the thing that opens the lens, as before.
            starting_over_open_camera = self._mode == self._MODE_RESULT
            self._mode = self._MODE_RUNNING
            # R1 §2-3: the open happens inside on_start (see
            # _start_round); the 5 s window anchors right after it.
            self._start_round(reuse_open_source=starting_over_open_camera)

        def _start_round(self, *, reuse_open_source: bool = False) -> None:
            """Start the current round desktop (single Start or trigger)."""
            if not (
                self.record_consent_checkbox.isChecked()
                and self.image_consent_checkbox.isChecked()
            ):
                self._set_status("需要 record consent 與 image consent")
                return
            try:
                # D1: this clock reading is the ROUND anchor (what the
                # operator pressed Start at) and is recorded as
                # round_start_ns. It is deliberately NOT the recognition
                # window anchor: the camera opens inside on_start, below,
                # and the engine re-arms the 5 s recognition window at the
                # first captured frame (LiveController._anchor_first_frame
                # -> SessionEngine.anchor_recognition).
                #
                # The pre-D1 comment here claimed this read happened
                # "after the source open", which it never did — the open
                # is the statement below. D0 recorded the discrepancy;
                # D1 removed the trap. The 5 s budget is evidence time
                # only: see docs/mac-demo-baseline-d0.md §7 for the
                # 23/23 field rounds that spent it waiting instead.
                round_start_ns = self._clock_ns()
                self.desktop.on_start(
                    self.consent,
                    now_ns=round_start_ns,
                    device_id=self.device_id,
                    reuse_open_source=reuse_open_source,
                )
            except Exception as exc:
                # G3 W7: a wrong camera pick fails here in Chinese (worst
                # case the operator closes the app and picks again).
                self._set_status(f"相機開啟失敗：{type(exc).__name__}")
                return
            self._round_start_ns = round_start_ns
            self._result_entered = False
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self._set_status("採集中 · collecting")
            self._set_countdown()
            if self._next_session is not None:
                self._mode = self._MODE_RUNNING
                self._round_started_utc = datetime.now(timezone.utc).isoformat()
            # D2: inference, camera reads and research staging run on a
            # worker; the timer below only paints.
            #
            # `background_inference` is opt-in: a real GUI event loop
            # needs it (D2's whole point), while the synchronous
            # `process_until_terminal` driver and the headless CLI paths
            # keep driving inference in-line, where their determinism
            # and step budgets are already pinned. Flipping it on
            # unconditionally would make every existing timing assertion
            # a race.
            if self._background_inference:
                self.desktop.start_inference_worker()
            self._timer.start()

        def cancel_clicked(self) -> None:
            """R1 §2-6: Cancel stops the round, closes the lens, clears.

            D2: the inference worker must be told to stop FIRST, otherwise
            it keeps scoring against an engine the UI has already closed
            out, and the round can land as `timeout` instead of the
            `cancelled` the operator actually asked for.
            """
            if self.desktop.state != "running":
                return
            try:
                self.desktop.cancel_inference()
                if self.desktop.inference_terminal is not None:
                    result = self.desktop.cancel_collection(self._clock_ns())
                else:
                    result = self.desktop.on_cancel(self._clock_ns())
            except Exception as exc:
                self._set_status(f"cancel failed: {type(exc).__name__}")
                return
            self.desktop.release_source()
            self._clear_preview()
            self._result_entered = False
            if self._next_session is not None:
                self.enter_ready(status="已取消 · cancelled")
            else:
                self._update_terminal(result)

        def process_once(self) -> None:
            """D2: in background mode this tick only paints.

            With `background_inference` enabled, inference, camera reads
            and research staging run on the controller's worker, and the
            frame that used to be scored, encrypted and written to disk
            inside this call now happens on another thread — so a slow
            model or a slow disk no longer stalls the window (D2 scope 1).

            Without it the tick keeps its pre-D2 synchronous behavior, so
            the headless CLI and the deterministic offscreen drivers are
            untouched.
            """
            if not self._background_inference:
                self._process_once_sync()
                return
            self._drain_preview()
            if self._clock_advance is not None:
                self._clock_advance()
            if self.desktop.state == "running":
                self._set_countdown()
            # Adopt the worker's terminal if it has landed. This is what
            # moves the desktop out of "running", so it must run even on
            # the tick that observes the round already finished — the
            # terminal transition below has to be reached exactly once.
            self.desktop.wait_for_terminal(timeout_s=0.0)
            if self.desktop.state == "terminal" and not self._result_entered:
                self._result_entered = True
                result = self.desktop.terminal
                if result is not None:
                    self._update_terminal(result)
                if self._next_session is not None:
                    # D2b: _enter_result restarts the timer, because the
                    # camera stays open and the preview must keep
                    # updating while the operator reads the result.
                    self._enter_result(result)
                else:
                    self._timer.stop()
            elif self.desktop.state == "running":
                pass
            elif self._mode != self._MODE_RESULT:
                # D2b: at result the timer is deliberately still running
                # (continuous preview). Only a non-result, non-running
                # state stops it.
                self._timer.stop()

        def _process_once_sync(self) -> None:
            """Pre-D2 tick: drive one bounded synchronous controller step."""
            if self.desktop.state != "running":
                self._timer.stop()
                return
            try:
                # G3 R1 PR-A change 3: bounded Qt steps must not finish
                # the round before its 5 s window elapses; exhaustion
                # leaves the round running for later ticks.
                result = self.desktop.run_until_terminal(
                    max_steps=50, finish_on_exhaust=False
                )
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
                if (
                    self._next_session is not None
                    and self.desktop.state == "terminal"
                ):
                    # D2b: _enter_result keeps the lens open and restarts
                    # the timer (continuous preview).
                    self._enter_result(result)
                else:
                    self._timer.stop()

        def _drain_preview(self) -> bool:
            """D2: paint the newest worker-produced frame, if any.

            The channel holds at most one frame, so a UI that ticks slower
            than the camera captures simply skips frames instead of
            building a backlog. Called only from the UI thread, which is
            the only place QPixmap may be touched.
            """
            packet = self.desktop.drain_preview()
            if packet is None:
                return False
            try:
                self.render_full_frame(packet.rgb)
            except Exception:
                # A preview paint failure must not stop the round; the
                # next tick will pick up a newer frame.
                return False
            return True

        def process_until_terminal(self, max_steps: int = 200) -> None:
            """D2: drive synthetic events to a terminal, waiting on the worker.

            Inference no longer happens inside `process_once` (it happens
            on the worker), so a synchronous caller that only ticks would
            spin `max_steps` times against a terminal that never lands.
            Each iteration therefore waits briefly for the worker's
            terminal before ticking the UI again — the same bounded,
            latency-independent handshake the offscreen tests need.
            """
            for _ in range(max_steps):
                if self.desktop.state != "running":
                    break
                if self.desktop.wait_for_terminal(timeout_s=0.05) is None:
                    # The worker is still running; keep the UI ticking so
                    # preview and countdown still update.
                    self.process_once()
                    continue
                self.process_once()
                break

        def enter_ready(self, status: str | None = None) -> None:
            """Enter Ready: lens shut, no preview, Start armed (R1 §2-2).

            Keeps the picked camera but never opens it; the preview
            area shows no prior face. Requires the continuous loop
            (next_session factory).

            D2b: "lens shut" is now this method's own responsibility, not
            something _enter_result already guaranteed. Before D2b the
            result path released the camera, so the detach() below left
            nothing open; with the lens held open across rounds, a bare
            detach() would have leaked the handle whenever a round ended
            in Ready (after a verdict, or a Cancel). So the release is
            explicit here.
            """
            if self._next_session is None:
                raise RuntimeError(
                    "enter_ready requires the continuous loop (next_session factory)"
                )
            if self.desktop.state in ("terminal", "labeled", "closed"):
                # D2b: stop this round's workers AND shut the lens. Under
                # D2 the source was already released at result, so this
                # was detach-only; holding the camera across rounds makes
                # the release load-bearing.
                self.desktop.detach()
                try:
                    self.desktop.close()
                except Exception:
                    pass
            self._clear_preview()
            self._mode = self._MODE_READY
            self._refresh_start_enabled()
            self.cancel_button.setEnabled(False)
            self.correct_button.setEnabled(False)
            self.incorrect_button.setEnabled(False)
            self.recognize_again_button.setEnabled(False)
            self.stop_camera_button.setEnabled(False)
            if status is None:
                if self._camera_options and self.selected_camera_index() is None:
                    status = "請選擇相機"
                else:
                    status = "已選相機，按 Start 開始"
            self._set_status(status)

        # G3 W2-W8 standby entry removed by R1 PR-B: selecting a camera
        # must not open a stream or auto-start a round. Kept as a
        # fail-loud alias so old callers break visibly, not silently.
        def enter_standby(self) -> None:
            """Removed: use enter_ready + Start (R1 Start-gated)."""
            raise RuntimeError(
                "enter_standby was removed by R1 Start-gating; "
                "use enter_ready then start_clicked"
            )

        def _clear_preview(self) -> None:
            """Drop the photo pixmap and its cache (R1 §2-4/6)."""
            self._preview_image = None
            self._crop_mapping = None
            try:
                self.preview_label.clear()
            except Exception:
                pass
            self.identity_label.setText("")
            self._clear_result_panel()
            self._set_guide()

        def _enter_result(self, result: Any) -> None:
            """Show the result with the lens still open (D2b).

            Supersedes R1 §2-4 for this window only (operator decision
            d-20260929173733098323-10). The old contract released the
            source and cleared the pixmap before any result text showed,
            because at that point there was no way to run another round.

            Now the camera stays open so the operator can read the
            result against a live preview, press 正確／錯誤, and then
            either 再次辨識 (same open camera — no reopen) or 停止相機
            (the only in-loop release). The photo is therefore NOT
            cleared: it is a live view now, not a stale capture, and
            clearing it is exactly the R1 §2-6 behaviour D2b removes.

            The desktop stays terminal (not closed) so the keys can
            label, and D3a's result panels are still populated.
            """
            text = self._format_result(result)
            self._result_text = text
            self._set_status(text)
            self._update_result_details(result)
            self._mode = self._MODE_RESULT
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.correct_button.setEnabled(True)
            self.incorrect_button.setEnabled(True)
            # G3-w P3: skip is unavailable on `invalid_input` rounds
            # (spec §4.5) — there is no identification to decline, only
            # the missed-person flag the ✓／✗ buttons record. The
            # disclosure area starts hidden on every round (progressive
            # disclosure: it appears only after ✗).
            skippable = result is None or (
                result.status != SessionStatus.invalid_input
            )
            self.skip_button.setEnabled(skippable)
            self.skip_button.setVisible(skippable)
            self.disclosure_widget.setVisible(False)
            # G3-w P3: no residual — the new round starts with empty
            # annotation. Whatever the previous round picked stays with
            # that round's row.
            self._annot_probe_kind = ""
            self._annot_identity = ""
            # D2b: the lens is open, so both controls are available. The
            # preview keeps ticking — the worker is still running and
            # still publishing frames into the preview channel.
            self.recognize_again_button.setEnabled(True)
            self.stop_camera_button.setEnabled(True)
            self._timer.start()

        def _format_result(self, result: Any) -> str:
            """R1 §2-4 display text: matched / not-found / diagnosable.

            A round with usable frames but no recognition in 5 s shows
            找不到此註冊人員. Rounds with no frames, no faces, or all
            quality rejections show their own cause — never a verified
            absence from the gallery, never zero_usable_frames as a
            misrecognition.
            """
            if result is None:
                return "辨識未完成"
            status = result.status
            if status == SessionStatus.matched and result.matched_identity is not None:
                score: float | None = None
                for obs in self.desktop.observations:
                    if result.matched_identity in obs.identity_scores:
                        score = obs.identity_scores[result.matched_identity]
                        break
                if score is None:
                    return str(result.matched_identity)
                return f"{result.matched_identity} {score:.2f}"
            # D1: a non-match terminal is classified by its REASONS, not
            # by its status alone. `insufficient_evidence` is written as
            # timeout (the window did elapse) while meaning "a face was
            # seen and the best frame even reached match level, but the
            # 3-frame rule never closed". Short-circuiting on status
            # showed 「找不到此註冊人員」 — the exact opposite of the
            # evidence, and worse than pre-D1, which at least said the
            # camera produced nothing. classify_failure now owns every
            # non-match sentence, including the plain ran-out-of-window
            # case, so this branch would only be a way to skip it.
            return self.classify_failure(
                observations=tuple(self.desktop.observations),
                reason_codes=tuple(result.reason_codes),
            )

        @staticmethod
        def classify_failure(
            *,
            observations: tuple[Any, ...] = (),
            reason_codes: tuple[str, ...] = (),
        ) -> str:
            """Chinese cause text for a non-match terminal (R1 §2-4, D1).

            Reason codes are consulted FIRST, before the observation
            heuristics. D1 split the terminal reasons so an operator can
            tell "nobody was recognised" from "a face was visible but the
            evidence was too thin", and that split is worthless if the
            status-only branches answer first: an `insufficient_evidence`
            round reaches the screen carrying the fact that a face was
            seen and the best single frame even reached match level, and
            showing 「找不到此註冊人員」 for it asserts the opposite.

            Every D1 code has prose here. An unrecognised code never
            reaches the operator as a raw token — the screen is not a
            place to print a machine identifier at someone.
            """
            codes = set(reason_codes)

            # D1: the camera delivered nothing at all. Same event the
            # empty-observations branch below already described as
            # 「相機無影格」, so it must not get a second, different
            # description.
            if "no_frames_captured" in codes and not observations:
                return "未取得可辨識影格：相機無影格"

            # D1: a face was visible and scored, but the multi-frame rule
            # never closed. The operator needs to know a person WAS there
            # — this is the case most easily misread as "not enrolled".
            if "insufficient_evidence" in codes:
                support = next(
                    (c for c in reason_codes if c.startswith("support_")), ""
                )
                detail = f"（{support.replace('_', ' ')}）" if support else ""
                return f"已看見人臉，但多幀確認未成立{detail}"

            # D1: frames arrived and every one was rejected. Say which
            # cause, using the real quality reasons where we have them.
            if "input_multiple_faces" in codes or "multiple_faces_detected" in codes:
                return "未取得可辨識影格：畫面有多張人臉"
            if "all_frames_rejected_mixed_causes" in codes:
                return "未取得可辨識影格：部分無臉、部分品質不合格"
            if "all_frames_rejected_no_face" in codes:
                return "未取得可辨識影格：未偵測到人臉"
            if "all_frames_rejected_quality" in codes:
                joined = _quality_reason_text(observations)
                return f"未取得可辨識影格：品質拒絕{joined}"

            if not observations:
                return "未取得可辨識影格：相機無影格"
            if any(getattr(o, "face_count", 0) > 1 for o in observations):
                return "未取得可辨識影格：畫面有多張人臉"
            faced = [o for o in observations if o.face_count >= 1]
            if not faced:
                return "未取得可辨識影格：未偵測到人臉"
            reasons: set[str] = set()
            for obs in faced:
                reasons.update(obs.quality_reasons or ())
            if reasons and all(not getattr(o, "quality_pass", False) for o in faced):
                joined = "、".join(sorted(reasons))
                return f"未取得可辨識影格：品質拒絕（{joined}）"
            # A ran-the-whole-window round with nothing special to report
            # is the genuine "did not find them" answer; saying anything
            # softer would be a false reassurance.
            if codes & {
                "deadline_exceeded",
                "best_baseline_unknown",
                "best_baseline_review",
            }:
                return "找不到此註冊人員"
            return "未完成辨識：無法判定原因"

        def format_result_for_test(self, kind: str) -> str:
            """Test hook routing one failure kind through _format_result."""
            if kind == "no_frame":
                return self.classify_failure(observations=())
            if kind == "no_face":
                obs = _SyntheticObservation(
                    face_count=0, quality_reasons=("no_face_detected",)
                )
                return self.classify_failure(observations=(obs,))
            if kind == "quality_rejected":
                obs = _SyntheticObservation(
                    face_count=1, quality_reasons=("blur_too_high",), quality_pass=False
                )
                return self.classify_failure(observations=(obs,))
            raise ValueError(f"unknown failure kind {kind!r}")

        def press_correct(self) -> None:
            """G3-w P3 正確 key: derive ground truth, persist, back to Ready.

            Correct endorses what is shown (spec §4.2): a matched round
            records the shown identity (`target`); a not-found round
            records `nontarget` + `outsider` — the operator confirms that
            nobody enrolled is there, and `outsider` is the runbook-defined
            marker for that fact. Zero extra input.
            """
            from facecore.live.contracts import SessionStatus

            terminal = self.desktop.terminal
            if terminal is not None and terminal.status == SessionStatus.matched:
                identity = self.desktop.display_identity()
                self._annot_probe_kind = "target"
                # The system already named them and the operator agrees:
                # the identity is the shown one, never a guess.
                self._annot_identity = identity or ""
            else:
                self._annot_probe_kind = "nontarget"
                self._annot_identity = "outsider"
            self._press_key(correct=True)

        def press_incorrect(self) -> None:
            """G3-w P3 錯誤 key: open the disclosure area (spec §4.2).

            First press only reveals the input area — the operator has
            not judged anything yet. The write happens on
            `confirm_incorrect`, so an accidental ✗ never records a
            wrong identity. Pressing ✗ again after picking is the same
            as confirming.
            """
            if not self.disclosure_widget.isVisible():
                self._refresh_ground_truth_options()
                self.disclosure_widget.setVisible(True)
                self._set_status("請選擇實際是誰，再按一次錯誤確認")
                return
            self.confirm_incorrect()

        def confirm_incorrect(self) -> None:
            """Write the ✗ verdict with the disclosure pick."""
            if not self._annot_probe_kind:
                self._set_status("請先選擇實際是誰")
                return
            self._press_key(correct=False)

        def press_skip(self) -> None:
            """G3-w P3 略過 key: an explicit non-judgement (spec §4.2).

            `skipped` is a value, not an absence: it records that the
            operator deliberately declined this round. Per decision, the
            recommended skip leaves `probe_kind`/`presenting_identity`
            empty and never rewrites stored CSVs.

            The write goes through `_record_unlabeled_round` — the same
            path as 再次辨識 — so the row carries `label_kind="unlabeled"`
            and the SOP's labeled-round formula does not count a skipped
            round as answered. No third writer is added: the writer_wiring
            exact-two guard stays in force.
            """
            from facecore.live.contracts import SessionStatus

            terminal = self.desktop.terminal
            if terminal is not None and (
                terminal.status == SessionStatus.invalid_input
            ):
                return
            self._annot_probe_kind = ""
            self._annot_identity = ""
            self._record_unlabeled_round()
            self.correct_button.setEnabled(False)
            self.incorrect_button.setEnabled(False)
            self.skip_button.setEnabled(False)
            self._set_status("已略過 · skipped")
            self._refresh_saved_state()
            self.enter_ready(status="已略過 · skipped")

        def _press_key(self, *, correct: bool) -> None:
            if self._next_session is None:
                raise RuntimeError(
                    "label keys require the continuous loop (next_session factory)"
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
                round_complete = RoundComplete(
                    session_id=self.desktop.session_id,
                    attempt_id=self.attempt_id,
                    terminal=terminal,
                    observations=tuple(self.desktop.observations),
                    label_kind=label_kind,
                    label_identity=label_identity,
                    profile_version=self.desktop.profile_version,
                    started_utc=self._round_started_utc or "",
                    # D7-A W1: the App-startup gallery report, forwarded so
                    # the demo row records which gallery this round used.
                    gallery_load_report=self.load_report,
                )
                if self._results_csv is not None and self.recorder is not None:
                    # G3 W8: commit on label (bundle + attempt + csv row)
                    # so a crash before close loses nothing labeled.
                    from facecore.research.cli import commit_g3_rounds

                    committed, failed = commit_g3_rounds(
                        self.recorder, self._results_csv, [round_complete]
                    )
                    if failed > 0 or committed != 1:
                        self._set_status("紀錄寫入失敗")
                        return
                elif self.demo_results_csv is not None:
                    # D3b: demo mode. The verdict row is appended to the
                    # plaintext demo file; recorder.commit is NOT called,
                    # because there is no bundle and no attempt ledger to
                    # commit into. The round still queues so the tail can
                    # count it, and a write failure still refuses the
                    # advance (fail-closed, same as record mode).
                    from facecore.research.cli import (
                        DemoLogHeaderMismatch,
                        append_g3_demo_results_csv,
                    )

                    try:
                        append_g3_demo_results_csv(
                            self.demo_results_csv,
                            round_complete,
                            required_support=getattr(
                                self.desktop.profile, "required_support", 0
                            ),
                            labeled_at_utc=datetime.now(timezone.utc).isoformat(),
                            # D7-A W3: same forwarding as the unlabeled
                            # path — both write demo rows from this round.
                            event_counts=self.desktop.event_counts(),
                            support_clears=self.desktop.support_clear_reasons(),
                            profile=self.desktop.profile,
                            # D7-A W0-b: same ground truth as the
                            # unlabeled path. Both write demo rows, so
                            # omitting it here would leave every labeled
                            # round empty while the unlabeled ones were
                            # recorded — the two paths must agree.
                            probe_kind=self._probe_kind_value(),
                            presenting_identity=self._presenting_identity_value(),
                            # D7-A W1: the App-startup gallery report.
                            gallery_load_report=self.load_report,
                            # G3-w P3: the verdict is the button just
                            # pressed — correct/incorrect (spec §5.1).
                            # `skipped` never flows through this path: both
                            # non-judgement exits (略過 button and 再次辨識)
                            # write via `_record_unlabeled_round` with
                            # `label_kind="unlabeled"`.
                            operator_verdict=(
                                "correct" if correct else "incorrect"
                            ),
                        )
                    except OSError as exc:
                        # D7-A #141: same as the unlabeled path — the
                        # header-mismatch text is the operator's next step.
                        if isinstance(exc, DemoLogHeaderMismatch):
                            self._set_status(f"紀錄寫入失敗 · {exc}")
                        else:
                            self._set_status("紀錄寫入失敗")
                        return
                # Queue the labeled round (committed above when a csv
                # target exists; otherwise the CLI tail commits on close).
                self.completed_rounds.append(round_complete)
            self.correct_button.setEnabled(False)
            self.incorrect_button.setEnabled(False)
            self._set_status("已標註 · labeled")
            self._refresh_saved_state()
            # R1 §2-5: back to Ready with the pick kept and the lens
            # shut; the next round needs another Start.
            self.enter_ready(status="已標註 · labeled")

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
            self._update_result_details(result)

        def delete_clicked(self) -> None:
            """Delete the session bundle plus linked attempts (operator action).

            Atomic from the caller's view: the recorder removes the session
            bundle directory and cascades to linked attempts, then the
            desktop is flagged deleted so the CLI never commits afterwards.
            A False return or any error fails closed: no deleted flag, no
            success status, and the CLI never reports rc0 for it.
            """
            self._timer.stop()
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

        def _overlay_guide(self, frame: np.ndarray, mapping: CropMapping) -> np.ndarray:
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
            # R1 §2-6: closing the window always releases the lens.
            self._timer.stop()
            self.desktop.close()
            event.accept()

    _QT_WINDOW_FACTORY = _QtResearchWindow


QtResearchWindow = _QT_WINDOW_FACTORY
