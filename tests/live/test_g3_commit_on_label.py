"""G3 W8 RED: commit on key press + config store/key effective.

Source of truth: commander post-merge follow-up (close-out tail
commits once at close; keys only append to memory) + spec §7-4 spirit
(no labeled round lost on crash).

All tests use synthetic frames, FakeCapture, an offscreen Qt
application, and a tmp ResearchRecorder. No camera, real faces,
gallery, embeddings, or photos.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from facecore.live.capture import FakeCapture
from facecore.live.contracts import FramePacket
from facecore.live.desktop import DesktopSession
from facecore.live.session import SessionEngine
from facecore.live.qt_window import QtResearchWindow
from tests.live.test_qt_window import (
    _AdvancingClock,
    _attempt,
    _consent,
    _manifest,
    _matching_scorer,
    _profile,
    _recorder,
)


@pytest.fixture(scope="module")
def qt_app() -> Any:
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication as ActualQApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return ActualQApplication.instance() or ActualQApplication([])


def _face_frames(n: int = 60) -> list[FramePacket]:
    frames = []
    for seq in range(1, n + 1):
        frame = np.full((8, 8, 3), 150, dtype=np.uint8)
        frames.append(
            FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=frame)
        )
    return frames


class _RoundFactory:
    def __init__(self, tmp_path: Path, source: FakeCapture, scorer: Any) -> None:
        self.recorder = _recorder(tmp_path)
        self.manifest = _manifest()
        self.source = source
        self.scorer = scorer
        self.counter = 0
        self.attempt_ids: list[str] = []
        self.results_csv = tmp_path / "results.csv"

    def __call__(self) -> tuple[DesktopSession, Any, str]:
        self.counter += 1
        session_id = f"g3w8-round-{self.counter}"
        attempt = _attempt(f"attempt-g3w8-{self.counter}")
        consent = _consent(session_id)
        self.recorder.begin_attempt(self.manifest, attempt, consent)
        self.recorder.begin(session_id, consent)
        self.attempt_ids.append(attempt.attempt_id)
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test"),
            source=self.source,
            scorer=self.scorer,
            session_id=session_id,
            release_source_on_terminal=False,
            label_recorder=self.recorder,
            label_attempt_id=attempt.attempt_id,
        )
        return desktop, consent, attempt.attempt_id


def _open_window(qt_app: Any, factory: _RoundFactory) -> QtResearchWindow:
    first_desktop, first_consent, first_attempt = factory()
    clock = _AdvancingClock()
    window = QtResearchWindow(
        first_desktop,
        consent=first_consent,
        recorder=factory.recorder,
        attempt_id=first_attempt,
        offscreen=True,
        clock_ns=clock,
        clock_advance=clock.advance,
        next_session=factory,
        results_csv=factory.results_csv,
    )
    window.show()
    window.enter_standby()
    return window


class TestCommitOnLabel:
    def test_two_labeled_rounds_persist_before_close(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """N labeled rounds, window still open: store + csv hold N rows."""
        factory = _RoundFactory(
            tmp_path, FakeCapture(_face_frames()), _matching_scorer
        )
        window = _open_window(qt_app, factory)
        window.process_until_terminal(max_steps=200)
        window.press_correct()
        assert window.mode == "standby"
        window.process_until_terminal(max_steps=200)
        window.press_incorrect()
        assert window.mode == "standby"
        # Window NEVER closed here: crash now must not lose the rounds.
        with open(factory.results_csv, newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        for round_ in window.completed_rounds:
            record = factory.recorder.read_record(round_.session_id)
            assert record.result.status.value == round_.terminal.status.value
        window.close()

    def test_failed_commit_stays_on_result(self, qt_app: Any, tmp_path: Path) -> None:
        """A commit failure refuses to advance (no silent loss)."""
        factory = _RoundFactory(
            tmp_path, FakeCapture(_face_frames()), _matching_scorer
        )
        window = _open_window(qt_app, factory)
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        # Sabotage the bundle commit: drop the staged session first.
        factory.recorder.abort(window.desktop.session_id, reason="test-sabotage")
        window.press_correct()
        assert window.mode == "result"
        window.close()


class TestConfigStoreEffective:
    def test_config_store_dir_receives_bundle(self, tmp_path: Path) -> None:
        """config store_dir/key_dir override the passed store paths."""
        import json as _json

        from facecore.research.cli import cmd_live

        profile = tmp_path / "profile.json"
        profile.write_text(
            _json.dumps(
                {
                    "schema_version": "v1",
                    "profile_version": "g3w8-cli-test",
                    "timeout_ms": 5000,
                    "sample_interval_ms": 200,
                    "max_frames": 26,
                    "queue_limit": 1,
                    "required_support": 1,
                    "min_support_interval_ms": 1,
                    "match_threshold": 0.363,
                    "review_threshold": 0.30,
                    "margin_threshold": 0.10,
                    "detector_version": "det-cli-test",
                    "quality_policy_version": "qual-cli-test",
                    "continuity_max_center_delta_ratio": 0.5,
                }
            )
        )
        config_path = tmp_path / "g3-local.json"
        config_path.write_text(
            _json.dumps(
                {
                    "enrollment_dir": str(tmp_path / "enroll-group"),
                    "models_dir": str(tmp_path / "models"),
                    "store_dir": str(tmp_path / "cfg-store"),
                    "key_dir": str(tmp_path / "cfg-keys"),
                }
            )
        )
        rc = cmd_live(
            profile_path=profile,
            store=tmp_path / "ignored-store",
            key_dir=tmp_path / "ignored-keys",
            device="fake",
            session_id="g3w8-cli-config",
            record_consent=True,
            image_consent=True,
            config=config_path,
        )
        assert rc == 0
        assert (tmp_path / "cfg-store" / "g3w8-cli-config").is_dir()
        assert not (tmp_path / "ignored-store").exists()


class TestPickStartsLoop:
    def test_pick_camera_starts_standby_and_rounds(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """CLI order: options → enter_standby (unpicked) → pick → loop."""
        factory = _RoundFactory(
            tmp_path, FakeCapture(_face_frames()), _matching_scorer
        )
        first_desktop, first_consent, first_attempt = factory()
        clock = _AdvancingClock()
        window = QtResearchWindow(
            first_desktop,
            consent=first_consent,
            recorder=factory.recorder,
            attempt_id=first_attempt,
            offscreen=True,
            clock_ns=clock,
            clock_advance=clock.advance,
            next_session=factory,
            results_csv=factory.results_csv,
            camera_options=[(0, "Inner Cam"), (1, "相機 1")],
        )
        window.show()
        # Unpicked: standby refuses with a prompt, timer stays stopped.
        window.enter_standby()
        assert window.status_label.text() == "請選擇相機"
        assert window._standby_timer.isActive() is False
        # Pick: standby starts at once (the BLOCKING fix).
        window.camera_combo.setCurrentIndex(1)
        assert window.mode == "standby"
        assert window._standby_timer.isActive() is True
        assert window.desktop.source.is_closed is False
        # A face opens round 1; the key commits it and round 2 follows.
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        window.press_correct()
        assert window.mode == "standby"
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        window.press_correct()
        assert window.mode == "standby"
        import csv as _csv

        with open(factory.results_csv, newline="") as handle:
            rows = list(_csv.DictReader(handle))
        assert len(rows) == 2
        window.close()
