"""Phase 2B E7-B tests: minimal Qt research window and square capture contract.

Source of truth:
    - docs/specs/2026-09-16-phase2b-mac-recognition-research.md §8 and Appendix A;
    - docs/plans/2026-09-16-phase2b-mac-recognition-execution-plan.md §12 E7;
    - E7-B dispatch premises (crop_mapping schema, fixed orientation assumption,
      ``live --ui {fake,qt}``, ``--qt-offscreen``, NOTICE, research-ui extra).

All tests use synthetic frames, FakeCapture, and an offscreen Qt application.
No camera, real faces, gallery, embeddings, or photos are used.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from facecore.live.capture import CaptureSource, FakeCapture
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionStatus,
)
from facecore.live.desktop import DesktopSession
from facecore.live.session import SessionEngine
from facecore.research.experiment import (
    AttemptRecord,
    EvaluationLabel,
    ExperimentManifest,
)
from facecore.research.records import ConsentRecord
from facecore.research.recorder import ResearchRecorder
from facecore.live.qt_window import (
    CropMapping,
    QtResearchWindow,
    center_square_crop,
    crop_frame,
    preview_frame,
)
from tests.live.test_headless_fixed_window import DeterministicCadenceCamera

# Qt helpers are imported in the fixture so the default verify job can run
# geometry/parser tests without the optional research-ui dependency.
QApplication: Any
QTest: Any
Qt: Any


TEST_PROFILE_DIGEST = "0" * 64


def _profile() -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="qt-test-v1",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=25,
        queue_limit=1,
        required_support=1,
        min_support_interval_ms=1,
        match_threshold=0.45,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="det-qt-test",
        quality_policy_version="quality-qt-test",
        continuity_max_center_delta_ratio=0.5,
    )


def _packet(seq: int, width: int = 5, height: int = 3) -> FramePacket:
    frame = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    return FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=frame)


def _matching_scorer(packet: FramePacket) -> FrameObservation:
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(0.0, 0.0, 2.0, 2.0),
        identity_scores={"person-synth-01": 0.9, "person-synth-02": 0.1},
        quality_rank=0.9,
        model_generation="gen-qt-test",
        gallery_digest="gallery-qt-test",
    )


def _review_scorer(packet: FramePacket) -> FrameObservation:
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(0.0, 0.0, 2.0, 2.0),
        identity_scores={"person-synth-01": 0.32, "person-synth-02": 0.31},
        quality_rank=0.9,
        model_generation="gen-qt-test",
        gallery_digest="gallery-qt-test",
    )


def _consent(session_id: str = "qt-session") -> ConsentRecord:
    return ConsentRecord(
        session_id=session_id,
        participant_id="participant-synth-01",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-16T08:00:00Z",
        record_expires_at_utc="2026-10-16T08:00:00Z",
        image_expires_at_utc="2026-09-23T08:00:00Z",
    )


def _manifest() -> ExperimentManifest:
    return ExperimentManifest.from_dict(
        {
            "identity": {"experiment_id": "exp-qt"},
            "software": {},
            "gallery": {},
            "policy": {"profile_digest": TEST_PROFILE_DIGEST},
            "capture": {"orientation": "fixed-test-assumption"},
            "privacy": {},
            "study": {},
            "analysis": {},
        }
    )


def _attempt(attempt_id: str = "attempt-qt") -> AttemptRecord:
    return AttemptRecord(
        experiment_id="exp-qt",
        attempt_id=attempt_id,
        participant_id="participant-synth-01",
        visit_id="visit-qt-01",
        condition_id="condition-qt",
        attempt_index=1,
        retry_of=None,
        consent_ref="qt-session",
        requested_at_utc="2026-09-16T08:00:00Z",
        accepted_at_utc="2026-09-16T08:00:01Z",
        started_at_utc=None,
        ended_at_utc=None,
        operational_status="accepted",
        error_code=None,
        bundle_ref=None,
    )


def _recorder(tmp_path: Path) -> ResearchRecorder:
    def clock() -> datetime:
        return datetime(2026, 9, 16, 8, tzinfo=timezone.utc)

    return ResearchRecorder(tmp_path / "store", tmp_path / "keys", clock=clock)


@pytest.fixture(scope="module")
def qt_app() -> Any:
    pytest.importorskip("PySide6.QtWidgets")
    global QApplication, QTest, Qt
    from PySide6.QtCore import Qt as ActualQt
    from PySide6.QtTest import QTest as ActualQTest
    from PySide6.QtWidgets import QApplication as ActualQApplication

    QApplication = ActualQApplication
    QTest = ActualQTest
    Qt = ActualQt
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return ActualQApplication.instance() or ActualQApplication([])


class _AdvancingClock:
    """Deterministic advancing clock for Qt offscreen tests.

    Starts at start_ns (aligned with session start <= frame 1 captured_ns).
    Each call to advance() increments by step_ns (default 200 ms, matching
    the profile sample interval and frame cadence).
    Latency-independent: zero wall-clock sleep, independent of host OS scheduler.
    """

    def __init__(self, start_ns: int = 0, step_ns: int = 200_000_000) -> None:
        self.now_ns = start_ns
        self.step_ns = step_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance(self) -> None:
        self.now_ns += self.step_ns


class TestSquareCaptureGeometry:
    """Appendix A.2/A.3/A.4 geometry is independent of identity and preview mirror."""

    def test_center_square_mapping_portrait_and_landscape(self) -> None:
        portrait = center_square_crop(640, 800)
        assert portrait == CropMapping(
            x=0,
            y=80,
            size=640,
            frame_w=640,
            frame_h=800,
            mirrored_preview=False,
        )
        landscape = center_square_crop(800, 640, mirrored_preview=True)
        assert landscape.to_dict() == {
            "x": 80,
            "y": 0,
            "size": 640,
            "frame_w": 800,
            "frame_h": 640,
            "mirrored_preview": True,
        }

    def test_crop_and_preview_share_mapping_without_nonuniform_resize(self) -> None:
        frame = np.arange(5 * 3 * 3, dtype=np.uint8).reshape(3, 5, 3)
        cropped, mapping = crop_frame(frame, mirrored_preview=True)
        assert mapping.x == 1
        assert mapping.y == 0
        assert mapping.size == 3
        np.testing.assert_array_equal(cropped, frame[:, 1:4, :])
        np.testing.assert_array_equal(
            preview_frame(cropped, mapping), cropped[:, ::-1, :]
        )

    def test_invalid_geometry_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            center_square_crop(0, 10)
        with pytest.raises(ValueError, match="shape"):
            crop_frame(np.zeros((3, 3), dtype=np.uint8))


class TestQtResearchWindow:
    """Offscreen Qt events drive real DesktopSession/controller bindings."""

    def test_start_cancel_and_close_join_workers(self, qt_app: QApplication) -> None:
        profile = _profile()
        desktop = DesktopSession(
            engine=SessionEngine(profile, "gallery-qt-test", "gen-qt-test"),
            source=FakeCapture(frames=[_packet(1), _packet(2), _packet(3)]),
            scorer=_matching_scorer,
            session_id="qt-session",
            fixed_seconds=True,
        )
        clock = _AdvancingClock()
        window = QtResearchWindow(
            desktop,
            consent=_consent(),
            offscreen=True,
            clock_ns=clock,
            clock_advance=clock.advance,
        )
        window.show()
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        assert desktop.state == "running"
        # Drive one event to lock B, then cancel the remaining fixed collection.
        window.process_once()
        assert desktop.inference_terminal is not None
        QTest.mouseClick(window.cancel_button, Qt.MouseButton.LeftButton)
        assert desktop.state == "terminal"
        assert desktop.terminal is not None
        assert desktop.source_closed
        assert desktop.workers_joined
        window.close()
        assert desktop.source_closed
        assert desktop.workers_joined

    def test_cancel_before_inference_lock_stops_collection_and_timer(
        self, qt_app: QApplication
    ) -> None:
        """C2 Scenario 1: Cancel before B locks terminates and stops timer.

        Proves that operator cancellation before inference terminal is reached
        results in a SessionStatus.cancelled terminal, sets collection_stop_reason
        to 'cancelled', transitions desktop.state to 'terminal', and stops
        the Qt processing timer.
        """
        profile = _profile()
        desktop = DesktopSession(
            engine=SessionEngine(profile, "gallery-qt-test", "gen-qt-test"),
            source=FakeCapture(frames=[_packet(1), _packet(2), _packet(3)]),
            scorer=_review_scorer,
            session_id="qt-session-pre-cancel",
            fixed_seconds=True,
        )
        clock = _AdvancingClock()
        window = QtResearchWindow(
            desktop,
            consent=_consent("qt-session-pre-cancel"),
            offscreen=True,
            clock_ns=clock,
            clock_advance=clock.advance,
        )
        window.show()
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        assert desktop.state == "running"
        assert window._timer.isActive() is True
        assert desktop.inference_terminal is None

        QTest.mouseClick(window.cancel_button, Qt.MouseButton.LeftButton)
        assert desktop.state == "terminal"
        assert desktop.terminal is not None
        assert desktop.terminal.status == SessionStatus.cancelled
        assert desktop.collection_stop_reason == "cancelled"
        assert desktop.collection_complete is False

        window.process_once()
        assert window._timer.isActive() is False
        window.close()

    def test_incomplete_terminal_stops_timer_and_covers_closed_source_fast_path(
        self, qt_app: QApplication
    ) -> None:
        """C2 Scenario 2: Incomplete terminal stops Qt timer.

        Covers closed source fast path. Proves that when collection stops
        due to source exhaustion (including explicit closed-source fast path
        where source.is_closed is True) or reaching max_frames cap before
        deadline:
        1. desktop.state transitions to 'terminal' (F1 fix).
        2. Qt timer stops on process_once (preventing infinite busy ticks).
        """
        # Part A: Source exhaustion via closed-source fast path (B3/B5)
        source_a = FakeCapture(frames=[])
        desktop_a = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test"),
            source=source_a,
            scorer=_matching_scorer,
            session_id="qt-session-closed-fast",
            fixed_seconds=True,
        )
        clock_a = _AdvancingClock()
        window_a = QtResearchWindow(
            desktop_a,
            consent=_consent("qt-session-closed-fast"),
            offscreen=True,
            clock_ns=clock_a,
            clock_advance=clock_a.advance,
        )
        window_a.show()
        QTest.mouseClick(window_a.start_button, Qt.MouseButton.LeftButton)
        assert desktop_a.state == "running"
        assert window_a._timer.isActive() is True

        # Close source BEFORE processing: triggers closed fast path on 1st read
        source_a.close()
        assert source_a.is_closed is True

        window_a.process_once()
        # Branch-specific proof: fast path triggered on dry read 1 (dry == 1 < 3)
        assert desktop_a._controller._consecutive_dry == 1
        assert desktop_a.state == "terminal"
        assert desktop_a.collection_stop_reason == "source_exhausted"
        assert desktop_a.collection_complete is False
        assert window_a._timer.isActive() is False
        window_a.close()

        # Part A2: Closed-source fast path with callable method duck-typing (B3)
        class _MethodClosedCapture(CaptureSource):
            def __init__(self) -> None:
                self._closed = False

            def open(self, device_id: str) -> None:
                self._closed = False

            def read(self) -> FramePacket | None:
                return None

            def close(self) -> None:
                self._closed = True

            def is_closed(self) -> bool:
                return self._closed

        source_m = _MethodClosedCapture()
        desktop_m = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test"),
            source=source_m,
            scorer=_matching_scorer,
            session_id="qt-session-closed-method",
            fixed_seconds=True,
        )
        clock_m = _AdvancingClock()
        window_m = QtResearchWindow(
            desktop_m,
            consent=_consent("qt-session-closed-method"),
            offscreen=True,
            clock_ns=clock_m,
            clock_advance=clock_m.advance,
        )
        window_m.show()
        QTest.mouseClick(window_m.start_button, Qt.MouseButton.LeftButton)
        source_m.close()
        assert source_m.is_closed() is True

        window_m.process_once()
        assert desktop_m._controller._consecutive_dry == 1
        assert desktop_m.state == "terminal"
        assert desktop_m.collection_stop_reason == "source_exhausted"
        assert window_m._timer.isActive() is False
        window_m.close()

        # Part B: Max frames reached before deadline
        low_cap_prof = ResearchProfile(
            schema_version="v1",
            profile_version="qt-lowcap",
            timeout_ms=5000,
            sample_interval_ms=200,
            max_frames=1,
            queue_limit=1,
            required_support=1,
            min_support_interval_ms=1,
            match_threshold=0.45,
            review_threshold=0.30,
            margin_threshold=0.10,
            detector_version="det-qt-test",
            quality_policy_version="quality-qt-test",
            continuity_max_center_delta_ratio=0.5,
        )
        desktop_b = DesktopSession(
            engine=SessionEngine(low_cap_prof, "gallery-qt-test", "gen-qt-test"),
            source=FakeCapture(frames=[_packet(1), _packet(2)]),
            scorer=_matching_scorer,
            session_id="qt-session-max-frames",
            max_frames=1,
            fixed_seconds=True,
        )
        clock_b = _AdvancingClock()
        window_b = QtResearchWindow(
            desktop_b,
            consent=_consent("qt-session-max-frames"),
            offscreen=True,
            clock_ns=clock_b,
            clock_advance=clock_b.advance,
        )
        window_b.show()
        QTest.mouseClick(window_b.start_button, Qt.MouseButton.LeftButton)
        window_b.process_once()

        assert desktop_b.state == "terminal"
        assert desktop_b.collection_stop_reason == "max_frames_reached"
        assert desktop_b.collection_complete is False
        assert window_b._timer.isActive() is False
        window_b.close()

    def test_in_progress_collection_retains_timer_after_inference_lock(
        self, qt_app: QApplication
    ) -> None:
        """C2 Scenario 3: Timer remains active while collection is in progress.

        Proves that in fixed-window mode, after inference B locks (inference_terminal
        is matched), desktop.state remains 'running' because collection is still
        in progress, and the Qt timer is NOT prematurely stopped.
        """
        clock = _AdvancingClock()
        camera = DeterministicCadenceCamera(fps=30)
        profile = _profile()
        desktop = DesktopSession(
            engine=SessionEngine(profile, "gallery-qt-test", "gen-qt-test"),
            source=camera,
            scorer=_matching_scorer,
            session_id="qt-session-in-progress",
            sample_interval_ns=int(profile.sample_interval_ms * 1_000_000),
            max_frames=profile.max_frames,
            fixed_seconds=True,
        )
        window = QtResearchWindow(
            desktop,
            consent=_consent("qt-session-in-progress"),
            offscreen=True,
            clock_ns=clock,
            clock_advance=clock.advance,
        )
        window.show()
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        assert desktop.state == "running"
        assert window._timer.isActive() is True

        window.process_once()
        assert desktop.inference_terminal is not None
        assert desktop.inference_terminal.status == SessionStatus.matched

        assert desktop.state == "running"
        assert desktop.collection_stop_reason == "in_progress"
        assert window._timer.isActive() is True
        window.close()

    def test_label_persists_to_encrypted_research_sidecar(
        self, qt_app: QApplication, tmp_path: Path
    ) -> None:
        recorder = _recorder(tmp_path)
        manifest = _manifest()
        attempt = _attempt()
        recorder.begin_attempt(manifest, attempt, _consent())
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test"),
            source=FakeCapture(frames=[_packet(1)]),
            scorer=_matching_scorer,
            session_id="qt-session",
            label_recorder=recorder,
            label_attempt_id=attempt.attempt_id,
        )
        window = QtResearchWindow(
            desktop,
            consent=_consent(),
            recorder=recorder,
            attempt_id=attempt.attempt_id,
            offscreen=True,
            clock_ns=lambda: 0,
        )
        window.show()
        window.set_frame(_packet(1).rgb)
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        window.process_until_terminal()
        assert desktop.display_identity() == "person-synth-01"
        QTest.mouseClick(window.enrolled_label_button, Qt.MouseButton.LeftButton)
        stored = recorder.read_label(attempt.attempt_id)
        assert stored == EvaluationLabel(
            attempt_id=attempt.attempt_id,
            revision=1,
            kind="enrolled",
            identity_id="person-synth-01",
            actor_ref="qt-operator",
            labeled_at=stored.labeled_at,
        )
        assert window.crop_mapping is not None
        assert (
            recorder.read_crop_mapping(attempt.attempt_id)
            == window.crop_mapping.to_dict()
        )
        window.close()

    def test_review_band_does_not_display_guessed_identity(
        self, qt_app: QApplication
    ) -> None:
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test"),
            source=FakeCapture(frames=[_packet(1)]),
            scorer=_review_scorer,
            session_id="qt-session-review",
        )
        window = QtResearchWindow(
            desktop,
            consent=_consent("qt-session-review"),
            offscreen=True,
            clock_ns=lambda: 0,
        )
        window.show()
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        window.process_until_terminal()
        assert desktop.terminal is not None
        assert desktop.terminal.status != SessionStatus.matched
        assert desktop.display_identity() is None
        assert window.identity_label.text() == ""
        window.close()

    def test_crop_mapping_is_stored_with_fixed_orientation_assumption(
        self, qt_app: QApplication, tmp_path: Path
    ) -> None:
        recorder = _recorder(tmp_path)
        recorder.begin_attempt(_manifest(), _attempt("attempt-geometry"), _consent())
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test"),
            source=FakeCapture(frames=[_packet(1, width=5, height=3)]),
            scorer=_matching_scorer,
            session_id="qt-session-geometry",
        )
        window = QtResearchWindow(
            desktop,
            consent=_consent("qt-session-geometry"),
            recorder=recorder,
            attempt_id="attempt-geometry",
            offscreen=True,
            clock_ns=lambda: 0,
        )
        window.set_frame(_packet(1, width=5, height=3).rgb)
        assert window.crop_mapping is not None
        assert window.crop_mapping.to_dict().keys() == {
            "x", "y", "size", "frame_w", "frame_h", "mirrored_preview"
        }
        assert (
            recorder.read_crop_mapping("attempt-geometry")
            == window.crop_mapping.to_dict()
        )
        window.close()

    def test_preview_overlay_draws_square_guide_on_full_frame(
        self, qt_app: QApplication, tmp_path: Path
    ) -> None:
        """E7-B r4 S3 RED: guide overlay uses the same crop mapping."""
        recorder = _recorder(tmp_path)
        recorder.begin_attempt(_manifest(), _attempt("attempt-overlay"), _consent())
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test"),
            source=FakeCapture(frames=[_packet(1, width=5, height=3)]),
            scorer=_matching_scorer,
            session_id="qt-session-overlay",
        )
        window = QtResearchWindow(
            desktop,
            consent=_consent("qt-session-overlay"),
            recorder=recorder,
            attempt_id="attempt-overlay",
            offscreen=True,
            clock_ns=lambda: 0,
        )
        frame = _packet(1, width=5, height=3).rgb
        mapping = window.set_frame(frame)
        assert window.preview_image is not None
        # Overlay keeps the full source shape, not the cropped square.
        assert (window.preview_image.width(), window.preview_image.height()) == (5, 3)
        # Guide rectangle matches the persisted mapping exactly.
        assert (mapping.x, mapping.y, mapping.size) == (1, 0, 3)
        assert window.guide_label.text().find("x=1 y=0 S=3") >= 0
        window.close()

    def test_terminal_delete_removes_bundle_and_stops_commit(
        self, qt_app: QApplication, tmp_path: Path
    ) -> None:
        """E7-B r5 T2 RED: Delete removes session bundle, no later commit."""
        recorder = _recorder(tmp_path)
        manifest = _manifest()
        attempt = _attempt("attempt-delete")
        consent = _consent("qt-session-delete")
        recorder.begin_attempt(manifest, attempt, consent)
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test"),
            source=FakeCapture(frames=[_packet(1)]),
            scorer=_matching_scorer,
            session_id="qt-session-delete",
            label_recorder=recorder,
            label_attempt_id=attempt.attempt_id,
        )
        window = QtResearchWindow(
            desktop,
            consent=consent,
            recorder=recorder,
            attempt_id=attempt.attempt_id,
            device_id="cam-test-delete",
            offscreen=True,
            clock_ns=lambda: 0,
        )
        window.show()
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        window.process_until_terminal()
        assert desktop.terminal is not None
        QTest.mouseClick(window.delete_button, Qt.MouseButton.LeftButton)
        assert desktop.deleted is True
        assert recorder.session_bundle_exists("qt-session-delete") is False
        assert recorder.list_attempts(experiment_id="exp-qt") == []
        window.close()

    def test_delete_failure_does_not_report_success(
        self, qt_app: QApplication, tmp_path: Path
    ) -> None:
        """E7-B r6 U2 RED: recorder delete False must not mark deleted."""

        class _FailingDeleteRecorder(ResearchRecorder):
            def delete(self, session_id: str) -> bool:
                return False

        now = datetime(2026, 9, 16, 8, tzinfo=timezone.utc)
        recorder = _FailingDeleteRecorder(
            tmp_path / "store", tmp_path / "keys", clock=lambda: now
        )
        manifest = _manifest()
        attempt = _attempt("attempt-delete-fail")
        consent = _consent("qt-session-delete-fail")
        recorder.begin_attempt(manifest, attempt, consent)
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test"),
            source=FakeCapture(frames=[_packet(1)]),
            scorer=_matching_scorer,
            session_id="qt-session-delete-fail",
            label_recorder=recorder,
            label_attempt_id=attempt.attempt_id,
        )
        window = QtResearchWindow(
            desktop,
            consent=consent,
            recorder=recorder,
            attempt_id=attempt.attempt_id,
            device_id="cam-test-delete-fail",
            offscreen=True,
            clock_ns=lambda: 0,
        )
        window.show()
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        window.process_until_terminal()
        assert desktop.terminal is not None
        QTest.mouseClick(window.delete_button, Qt.MouseButton.LeftButton)
        assert desktop.deleted is False
        assert desktop.delete_failed is True
        assert "delete failed" in window.status_label.text()
        window.close()

    def test_minimal_ux_controls_and_countdown(
        self, qt_app: QApplication, tmp_path: Path
    ) -> None:
        """E7-B r2 F4: plan §12 minimal UX controls exist and render countdown."""
        recorder = _recorder(tmp_path)
        manifest = _manifest()
        attempt = _attempt("attempt-ux")
        consent = _consent("qt-session-ux")
        recorder.begin_attempt(manifest, attempt, consent)
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test"),
            source=FakeCapture(frames=[_packet(1)]),
            scorer=_matching_scorer,
            session_id="qt-session-ux",
            label_recorder=recorder,
            label_attempt_id=attempt.attempt_id,
        )
        window = QtResearchWindow(
            desktop,
            consent=consent,
            recorder=recorder,
            attempt_id=attempt.attempt_id,
            device_id="cam-test-ux",
            offscreen=True,
            clock_ns=lambda: 0,
        )
        # Check controls exist per plan §12 E7 minimal set
        assert hasattr(window, "device_label")
        assert "cam-test-ux" in window.device_label.text()
        assert hasattr(window, "ttl_label")
        assert "30" in window.ttl_label.text() or "2026" in window.ttl_label.text()
        assert hasattr(window, "countdown_label")
        assert hasattr(window, "saved_state_label")
        assert hasattr(window, "delete_button")
        assert hasattr(window, "guide_label")

        # Countdown rendering
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        countdown_text = window.countdown_label.text()
        assert "5000" in countdown_text or "5.0" in countdown_text

    def test_countdown_ticks_down_with_live_clock(
        self, qt_app: QApplication, tmp_path: Path
    ) -> None:
        """E7-B r4 S2 RED: countdown must not freeze at the start value."""
        recorder = _recorder(tmp_path)
        manifest = _manifest()
        attempt = _attempt("attempt-countdown")
        consent = _consent("qt-session-countdown")
        recorder.begin_attempt(manifest, attempt, consent)
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test"),
            source=FakeCapture(frames=[_packet(1)]),
            scorer=_matching_scorer,
            session_id="qt-session-countdown",
            label_recorder=recorder,
            label_attempt_id=attempt.attempt_id,
        )
        now_ns = [0]
        window = QtResearchWindow(
            desktop,
            consent=consent,
            recorder=recorder,
            attempt_id=attempt.attempt_id,
            device_id="cam-test-countdown",
            offscreen=True,
            clock_ns=lambda: now_ns[0],
        )
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        first = window.countdown_label.text()
        now_ns[0] = 2_000_000_000
        window.process_once()
        second = window.countdown_label.text()
        assert first != second
        assert "3000" in second
        window.close()

        # Delete action closes session and purges attempt
        QTest.mouseClick(window.delete_button, Qt.MouseButton.LeftButton)
        assert desktop.state in ("closed", "terminal")
        assert recorder.list_attempts(experiment_id="exp-qt") == []
        window.close()


def test_research_ui_extra_and_notice_are_declared() -> None:
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    text = pyproject.read_text()
    assert "research-ui = [" in text
    assert '"pyside6==6.11.2"' in text
    notice = pyproject.parent / "NOTICE"
    assert notice.is_file()
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in notice.read_text()


def test_cli_qt_flags_route_without_opening_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parser exposes the frozen Qt route while preserving fake default."""
    import facecore.research.cli as research_cli

    called: dict[str, object] = {}

    def fake_cmd_live(**kwargs: object) -> int:
        called.update(kwargs)
        return 17

    monkeypatch.setattr(research_cli, "cmd_live", fake_cmd_live)
    profile = Path("synthetic-profile.json")
    rc = research_cli.main(
        [
            "live",
            "--profile",
            str(profile),
            "--store",
            "/tmp/synthetic-research-store",
            "--device",
            "fake",
            "--session",
            "synthetic-session",
            "--record-consent",
            "--image-consent",
            "--ui",
            "qt",
            "--qt-offscreen",
        ]
    )
    assert rc == 17
    assert called["ui"] == "qt"
    assert called["qt_offscreen"] is True


def test_default_mypy_allows_optional_qt_module() -> None:
    """Default verify stays independent from the research-ui extra."""
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    text = pyproject.read_text()
    assert 'module = "PySide6.*"' in text
    assert "ignore_missing_imports = true" in text
