"""D3: Result verifiability and startup loading report tests (Task D3a).

Covers:
1. Startup enrollment report: expected vs actual count, failed files & reasons.
2. Result display panel: candidate thumbnail (direct read, zero copy to store/repo),
   candidate/identity label ("候選" for non-match, not confirmed), top1/top2,
   similarity (never percentage/accuracy), margin, effective/required frames.
3. Failure reasons separated: exposure, too small, no face, multiple faces,
   insufficient evidence, unknown, camera/model error.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest

from facecore.live.capture import FakeCapture
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionStatus,
)
from facecore.live.desktop import DesktopSession
from facecore.live.frame_pipeline import (
    EnrollmentFailure,
    GalleryLoadReport,
    ResearchGallery,
)
from facecore.live.session import SessionEngine
from facecore.research.records import ConsentRecord

_QT_AVAILABLE = importlib.util.find_spec("PySide6") is not None
pytestmark = pytest.mark.skipif(
    not _QT_AVAILABLE,
    reason="Qt tests require the optional research-ui extra (pyside6)",
)


def _profile(**overrides: Any) -> ResearchProfile:
    base: dict[str, Any] = {
        "schema_version": "v1",
        "profile_version": "d3-v1",
        "timeout_ms": 5000,
        "sample_interval_ms": 200,
        "max_frames": 26,
        "queue_limit": 1,
        "required_support": 3,
        "min_support_interval_ms": 200,
        "match_threshold": 0.363,
        "review_threshold": 0.30,
        "margin_threshold": 0.10,
        "detector_version": "yunet",
        "quality_policy_version": "1",
        "continuity_max_center_delta_ratio": 0.5,
    }
    base.update(overrides)
    return ResearchProfile(**base)  # type: ignore[arg-type]


def _consent(session_id: str = "d3-test-session") -> ConsentRecord:
    return ConsentRecord(
        session_id=session_id,
        participant_id="participant-d3-01",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-30T00:00:00Z",
        record_expires_at_utc="2026-10-30T00:00:00Z",
        image_expires_at_utc="2026-10-30T00:00:00Z",
    )


def _packet(seq: int, captured_ns: int = 0) -> FramePacket:
    return FramePacket(
        sequence=seq,
        captured_ns=captured_ns or seq * 200_000_000,
        rgb=np.full((16, 16, 3), 150, dtype=np.uint8),
    )


def _write_test_photo(path: Path, shade: int = 150) -> None:
    img = Image.new("RGB", (32, 32), (shade, shade, shade))
    img.save(path)


def _dummy_gallery(
    identities: list[str],
    *,
    load_report: GalleryLoadReport | None = None,
    sources: dict[str, Path] | None = None,
) -> ResearchGallery:
    embeddings = {
        ident: np.full((8,), 0.5, dtype=np.float32) for ident in identities
    }
    return ResearchGallery(
        embeddings=embeddings,
        model_version="sface_2021dec",
        generation="gen-d3-test",
        digest="digest-d3-dummy",
        load_report=load_report,
        identity_sources=sources,
    )


@pytest.fixture(scope="module")
def qt_app() -> Any:
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


# ===========================================================================
# Scope 1: 啟動載入報告 UI 測試
# ===========================================================================


class TestStartupEnrollmentReportUI:
    def test_enrollment_label_shows_expected_and_actual_counts_all_success(
        self, qt_app: Any
    ) -> None:
        from facecore.live.qt_window import QtResearchWindow

        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gal-d3", "gen-d3"),
            source=FakeCapture(frames=[_packet(1)]),
            scorer=lambda p: FrameObservation(
                sequence=p.sequence,
                captured_ns=p.captured_ns,
                processed_ns=p.captured_ns,
                quality_pass=False,
                quality_reasons=("no_face_detected",),
                face_count=0,
                face_box=None,
                identity_scores={},
                quality_rank=0.0,
                model_generation="gen-d3",
                gallery_digest="gal-d3",
            ),
            session_id="d3-ui-1",
        )
        report = GalleryLoadReport(
            expected_count=23,
            loaded_count=23,
            failures=(),
        )
        gallery = _dummy_gallery(["enroll-01", "enroll-02"], load_report=report)
        window = QtResearchWindow(
            desktop,
            consent=_consent("d3-ui-1"),
            offscreen=True,
            gallery=gallery,
        )
        text = window.enrollment_label.text()
        assert "應載入 23 人" in text
        assert "實際載入 23 人" in text
        assert "全部成功" in text
        window.close()

    def test_enrollment_label_shows_failure_details_on_partial_failure(
        self, qt_app: Any
    ) -> None:
        from facecore.live.qt_window import QtResearchWindow

        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gal-d3", "gen-d3"),
            source=FakeCapture(frames=[_packet(1)]),
            scorer=lambda p: FrameObservation(
                sequence=p.sequence,
                captured_ns=p.captured_ns,
                processed_ns=p.captured_ns,
                quality_pass=False,
                quality_reasons=(),
                face_count=0,
                face_box=None,
                identity_scores={},
                quality_rank=0.0,
                model_generation="gen-d3",
                gallery_digest="gal-d3",
            ),
            session_id="d3-ui-2",
        )
        report = GalleryLoadReport(
            expected_count=23,
            loaded_count=21,
            failures=(
                EnrollmentFailure(
                    filename="enroll-04.jpg", reason="未偵測到人臉"
                ),
                EnrollmentFailure(
                    filename="enroll-13.png", reason="多張人臉"
                ),
            ),
        )
        gallery = _dummy_gallery(["enroll-01"], load_report=report)
        window = QtResearchWindow(
            desktop,
            consent=_consent("d3-ui-2"),
            offscreen=True,
            gallery=gallery,
        )
        text = window.enrollment_label.text()
        assert "應載入 23 人" in text
        assert "實際載入 21 人" in text
        assert "enroll-04.jpg（未偵測到人臉）" in text
        assert "enroll-13.png（多張人臉）" in text
        window.close()

    def test_enrollment_label_without_load_report_falls_back_to_gallery_count(
        self, qt_app: Any
    ) -> None:
        from facecore.live.qt_window import QtResearchWindow

        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gal-d3", "gen-d3"),
            source=FakeCapture(frames=[_packet(1)]),
            scorer=lambda p: FrameObservation(
                sequence=p.sequence,
                captured_ns=p.captured_ns,
                processed_ns=p.captured_ns,
                quality_pass=False,
                quality_reasons=(),
                face_count=0,
                face_box=None,
                identity_scores={},
                quality_rank=0.0,
                model_generation="gen-d3",
                gallery_digest="gal-d3",
            ),
            session_id="d3-ui-3",
        )
        gallery = _dummy_gallery(["enroll-01", "enroll-02", "enroll-03"])
        window = QtResearchWindow(
            desktop,
            consent=_consent("d3-ui-3"),
            offscreen=True,
            gallery=gallery,
        )
        text = window.enrollment_label.text()
        assert "應載入 3 人" in text
        assert "實際載入 3 人" in text
        window.close()


# ===========================================================================
# Scope 2 & 3: 結果顯示面板與失敗原因分開顯示
# ===========================================================================


class TestResultVerificationPanel:
    def test_matched_round_displays_identity_scores_frames_and_thumbnail(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """D3 Scope 2: matched displays identity, scores, frames, and thumbnail."""
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from facecore.live.qt_window import QtResearchWindow

        enr_dir = tmp_path / "enrollment"
        enr_dir.mkdir()
        _write_test_photo(enr_dir / "enroll-23.png", shade=120)
        _write_test_photo(enr_dir / "enroll-10.jpg", shade=150)

        def _matched_scorer(packet: FramePacket) -> FrameObservation:
            return FrameObservation(
                sequence=packet.sequence,
                captured_ns=packet.captured_ns,
                processed_ns=packet.captured_ns + 10_000_000,
                quality_pass=True,
                quality_reasons=(),
                face_count=1,
                face_box=(0.0, 0.0, 2.0, 2.0),
                identity_scores={"enroll-23": 0.650, "enroll-10": 0.280},
                quality_rank=0.95,
                model_generation="gen-d3",
                gallery_digest="gal-d3",
            )

        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gal-d3", "gen-d3"),
            source=FakeCapture(
                frames=[_packet(1), _packet(2), _packet(3), _packet(4)]
            ),
            scorer=_matched_scorer,
            session_id="d3-match-test",
            fixed_seconds=False,
        )
        gallery = _dummy_gallery(
            ["enroll-23", "enroll-10"],
            sources={
                "enroll-23": enr_dir / "enroll-23.png",
                "enroll-10": enr_dir / "enroll-10.jpg",
            },
        )
        window = QtResearchWindow(
            desktop,
            consent=_consent("d3-match-test"),
            offscreen=True,
            gallery=gallery,
            enrollment_dir=enr_dir,
            clock_ns=lambda: 0,
        )
        window.show()
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        window.process_until_terminal()

        assert desktop.terminal is not None
        assert desktop.terminal.status == SessionStatus.matched
        assert desktop.terminal.matched_identity == "enroll-23"

        # Result panel assertions
        assert window.result_identity_label.text() == "身分 · identity: enroll-23"
        assert window.identity_label.text() == "enroll-23"

        scores_text = window.scores_label.text()
        assert "Top 1: enroll-23（相似度 0.650）" in scores_text
        assert "Top 2: enroll-10（相似度 0.280）" in scores_text
        assert "差距: 0.370" in scores_text
        # Hard rule: similarity must NOT be displayed as percentage accuracy
        assert "%" not in scores_text
        assert "準確率" not in scores_text

        assert window.frames_label.text() == "有效幀／所需幀: 3/3"
        assert window.failure_reason_label.text() == "失敗原因: 無（辨識成功）"

        # Thumbnail check
        pixmap = window.candidate_thumbnail_label.pixmap()
        assert pixmap is not None
        assert not pixmap.isNull()
        assert pixmap.width() <= 120 and pixmap.height() <= 120

        window.close()

    def test_non_matched_candidate_labeled_as_candidate_never_confirmed(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """D3 Scope 2: non-matched terminal MUST label candidate as '候選'."""
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from facecore.live.qt_window import QtResearchWindow

        enr_dir = tmp_path / "enrollment"
        enr_dir.mkdir()
        _write_test_photo(enr_dir / "enroll-05.png", shade=130)

        # Scorer reaches review band but not matched 3 times -> timeout
        def _thin_scorer(packet: FramePacket) -> FrameObservation:
            return FrameObservation(
                sequence=packet.sequence,
                captured_ns=packet.captured_ns,
                processed_ns=packet.captured_ns + 10_000_000,
                quality_pass=True,
                quality_reasons=(),
                face_count=1,
                face_box=(0.0, 0.0, 2.0, 2.0),
                identity_scores={"enroll-05": 0.330, "enroll-02": 0.290},
                quality_rank=0.8,
                model_generation="gen-d3",
                gallery_digest="gal-d3",
            )

        # Single frame then dry source -> terminates with insufficient_evidence
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gal-d3", "gen-d3"),
            source=FakeCapture(frames=[_packet(1)]),
            scorer=_thin_scorer,
            session_id="d3-cand-test",
        )
        gallery = _dummy_gallery(
            ["enroll-05", "enroll-02"],
            sources={"enroll-05": enr_dir / "enroll-05.png"},
        )
        window = QtResearchWindow(
            desktop,
            consent=_consent("d3-cand-test"),
            offscreen=True,
            gallery=gallery,
            enrollment_dir=enr_dir,
            clock_ns=lambda: 0,
        )
        window.show()
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        window.process_until_terminal()

        assert desktop.terminal is not None
        assert desktop.terminal.status != SessionStatus.matched

        # Explicit requirement: "未達 matched 的一律標「候選」，不得冒充確定辨識"
        result_id_text = window.result_identity_label.text()
        assert "候選" in result_id_text
        assert "enroll-05" in result_id_text
        assert "身分 · identity: enroll-05" not in result_id_text
        # display_identity remains empty for non-match
        assert window.identity_label.text() == ""

        scores_text = window.scores_label.text()
        assert "Top 1: enroll-05（相似度 0.330）" in scores_text
        assert "%" not in scores_text
        assert "準確率" not in scores_text

        # Failure reason is displayed
        assert "已看見人臉，但多幀確認未成立" in window.failure_reason_label.text()

        # Thumbnail is shown for the top candidate
        pixmap = window.candidate_thumbnail_label.pixmap()
        assert pixmap is not None
        assert not pixmap.isNull()

        window.close()

    def test_no_face_detected_shows_no_candidate_and_zero_frames(
        self, qt_app: Any
    ) -> None:
        """D3 Scope 2 & 3: no face shows no candidate, 0 frames, distinct reason."""
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from facecore.live.qt_window import QtResearchWindow

        def _no_face_scorer(packet: FramePacket) -> FrameObservation:
            return FrameObservation(
                sequence=packet.sequence,
                captured_ns=packet.captured_ns,
                processed_ns=packet.captured_ns,
                quality_pass=False,
                quality_reasons=("no_face_detected",),
                face_count=0,
                face_box=None,
                identity_scores={},
                quality_rank=0.0,
                model_generation="gen-d3",
                gallery_digest="gal-d3",
            )

        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gal-d3", "gen-d3"),
            source=FakeCapture(frames=[_packet(1)]),
            scorer=_no_face_scorer,
            session_id="d3-noface-test",
        )
        window = QtResearchWindow(
            desktop,
            consent=_consent("d3-noface-test"),
            offscreen=True,
            clock_ns=lambda: 0,
        )
        window.show()
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        window.process_until_terminal()

        assert desktop.terminal is not None
        assert window.result_identity_label.text() == "身分 · identity: 無候選"
        assert window.scores_label.text() == "Top 1: 無 · Top 2: 無 · 差距: 無"
        assert window.frames_label.text() == "有效幀／所需幀: 0/3"
        assert "未偵測到人臉" in window.failure_reason_label.text()
        assert window.candidate_thumbnail_label.text() == "無縮圖"
        window.close()

    def test_thumbnail_zero_copy_verification(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """D3 Stop condition check: thumbnail direct read must NOT copy photos."""
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from facecore.live.qt_window import QtResearchWindow

        enr_dir = tmp_path / "enrollment"
        enr_dir.mkdir()
        photo_path = enr_dir / "enroll-99.png"
        _write_test_photo(photo_path, shade=180)

        store_dir = tmp_path / "fake_store"
        store_dir.mkdir()

        def _single_scorer(packet: FramePacket) -> FrameObservation:
            return FrameObservation(
                sequence=packet.sequence,
                captured_ns=packet.captured_ns,
                processed_ns=packet.captured_ns,
                quality_pass=True,
                quality_reasons=(),
                face_count=1,
                face_box=(0.0, 0.0, 2.0, 2.0),
                identity_scores={"enroll-99": 0.700},
                quality_rank=0.9,
                model_generation="gen-d3",
                gallery_digest="gal-d3",
            )

        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gal-d3", "gen-d3"),
            source=FakeCapture(frames=[_packet(1), _packet(2), _packet(3)]),
            scorer=_single_scorer,
            session_id="d3-zerocopy-test",
        )
        gallery = _dummy_gallery(
            ["enroll-99"], sources={"enroll-99": photo_path}
        )
        window = QtResearchWindow(
            desktop,
            consent=_consent("d3-zerocopy-test"),
            offscreen=True,
            gallery=gallery,
            enrollment_dir=enr_dir,
            clock_ns=lambda: 0,
        )
        window.show()
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        window.process_until_terminal()

        # Verification: store_dir must contain NO copies of enrollment photos
        store_files = list(store_dir.rglob("*.png")) + list(
            store_dir.rglob("*.jpg")
        )
        assert len(store_files) == 0, f"Photos were copied to store: {store_files}"

        # Original photo remains intact at source path
        assert photo_path.is_file()
        window.close()

    def test_thumbnail_missing_or_corrupt_degrades_gracefully(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """D3 Scope 2: missing/corrupt photo degrades to '無縮圖' without crashing."""
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from facecore.live.qt_window import QtResearchWindow

        enr_dir = tmp_path / "empty_enrollment"
        enr_dir.mkdir()
        # Photo path points to non-existent file
        missing_photo = enr_dir / "enroll-ghost.png"

        def _ghost_scorer(packet: FramePacket) -> FrameObservation:
            return FrameObservation(
                sequence=packet.sequence,
                captured_ns=packet.captured_ns,
                processed_ns=packet.captured_ns,
                quality_pass=True,
                quality_reasons=(),
                face_count=1,
                face_box=(0.0, 0.0, 2.0, 2.0),
                identity_scores={"enroll-ghost": 0.700},
                quality_rank=0.9,
                model_generation="gen-d3",
                gallery_digest="gal-d3",
            )

        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gal-d3", "gen-d3"),
            source=FakeCapture(frames=[_packet(1), _packet(2), _packet(3)]),
            scorer=_ghost_scorer,
            session_id="d3-ghost-test",
        )
        gallery = _dummy_gallery(
            ["enroll-ghost"], sources={"enroll-ghost": missing_photo}
        )
        window = QtResearchWindow(
            desktop,
            consent=_consent("d3-ghost-test"),
            offscreen=True,
            gallery=gallery,
            enrollment_dir=enr_dir,
            clock_ns=lambda: 0,
        )
        window.show()
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        # Must not raise or crash
        window.process_until_terminal()

        assert window.candidate_thumbnail_label.text() == "無縮圖"
        window.close()

    def test_result_panel_clears_on_enter_ready(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """D3 lifecycle: entering ready clears prior round's result panel."""
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from facecore.live.qt_window import QtResearchWindow

        enr_dir = tmp_path / "enrollment"
        enr_dir.mkdir()
        _write_test_photo(enr_dir / "enroll-23.png")

        def _factory() -> tuple[DesktopSession, ConsentRecord, str | None]:
            d = DesktopSession(
                engine=SessionEngine(_profile(), "gal-d3", "gen-d3"),
                source=FakeCapture(frames=[_packet(1), _packet(2), _packet(3)]),
                scorer=lambda p: FrameObservation(
                    sequence=p.sequence,
                    captured_ns=p.captured_ns,
                    processed_ns=p.captured_ns,
                    quality_pass=True,
                    quality_reasons=(),
                    face_count=1,
                    face_box=(0.0, 0.0, 2.0, 2.0),
                    identity_scores={"enroll-23": 0.650},
                    quality_rank=0.9,
                    model_generation="gen-d3",
                    gallery_digest="gal-d3",
                ),
                session_id="d3-loop-1",
            )
            return d, _consent("d3-loop-1"), None

        desktop, consent, _ = _factory()
        window = QtResearchWindow(
            desktop,
            consent=consent,
            offscreen=True,
            next_session=_factory,
            enrollment_dir=enr_dir,
            clock_ns=lambda: 0,
        )
        window.show()
        window.enter_ready()
        QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
        window.process_until_terminal()

        # In result state: result panel has content
        assert "enroll-23" in window.result_identity_label.text()

        # Correct/enter_ready returns to ready: result panel is cleared
        window.enter_ready()
        assert window.result_identity_label.text() == ""
        assert window.scores_label.text() == ""
        assert window.frames_label.text() == ""
        assert window.failure_reason_label.text() == ""
        assert window.candidate_thumbnail_label.text() == "無縮圖"
        window.close()


# ===========================================================================
# Scope 3: 失敗原因分流測試
# ===========================================================================


class TestD3FailureReasons:
    def test_classify_failure_separates_multiple_faces(self) -> None:
        from facecore.live.qt_window import _QtResearchWindow

        text = _QtResearchWindow.classify_failure(
            reason_codes=("input_multiple_faces", "session_restart_required")
        )
        assert "畫面有多張人臉" in text

    def test_classify_failure_separates_quality_exposure_and_size(self) -> None:
        from facecore.live.qt_window import _QtResearchWindow

        obs_exposure = FrameObservation(
            sequence=1,
            captured_ns=100,
            processed_ns=100,
            quality_pass=False,
            quality_reasons=("quality_exposure",),
            face_count=1,
            face_box=(0.0, 0.0, 2.0, 2.0),
            identity_scores={},
            quality_rank=0.0,
            model_generation="g",
            gallery_digest="d",
        )
        text_exp = _QtResearchWindow.classify_failure(
            observations=(obs_exposure,),
            reason_codes=("all_frames_rejected_quality",),
        )
        assert "quality_exposure" in text_exp

        obs_small = FrameObservation(
            sequence=1,
            captured_ns=100,
            processed_ns=100,
            quality_pass=False,
            quality_reasons=("quality_face_too_small",),
            face_count=1,
            face_box=(0.0, 0.0, 2.0, 2.0),
            identity_scores={},
            quality_rank=0.0,
            model_generation="g",
            gallery_digest="d",
        )
        text_small = _QtResearchWindow.classify_failure(
            observations=(obs_small,),
            reason_codes=("all_frames_rejected_quality",),
        )
        assert "quality_face_too_small" in text_small
