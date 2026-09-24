"""G3 W3 RED: operator correct/incorrect keys persist per-round labels.

Source of truth: docs/specs/2026-09-24-g3-local-test-app.md §2 step 6
(labels come only from key presses, never from system prediction, never
flow back into recognition) and §7 item 5 (a misrecognition can be marked
incorrect and is never auto-recorded as correct).

All tests use synthetic frames, FakeCapture, an offscreen Qt application,
and a tmp ResearchRecorder. No camera, real faces, gallery, embeddings,
or photos.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from facecore.live.capture import FakeCapture
from facecore.live.contracts import FrameObservation, FramePacket
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


def _low_scorer(packet: FramePacket) -> FrameObservation:
    """Face present but below review: round runs to deadline → unknown."""
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(0.0, 0.0, 2.0, 2.0),
        identity_scores={"person-synth-01": 0.20, "person-synth-02": 0.19},
        quality_rank=0.9,
        model_generation="gen-qt-test",
        gallery_digest="gallery-qt-test",
    )


def _face_frames(n: int = 60) -> list[FramePacket]:
    frames = []
    for seq in range(1, n + 1):
        frame = np.full((8, 8, 3), 150, dtype=np.uint8)
        frames.append(
            FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=frame)
        )
    return frames


class _RoundFactory:
    """Build recorder-bound round sessions over one shared source."""

    def __init__(self, tmp_path: Path, source: FakeCapture, scorer: Any) -> None:
        self.recorder = _recorder(tmp_path)
        self.manifest = _manifest()
        self.source = source
        self.scorer = scorer
        self.counter = 0
        self.attempt_ids: list[str] = []

    def __call__(self) -> tuple[DesktopSession, Any, str]:
        self.counter += 1
        session_id = f"g3w3-round-{self.counter}"
        attempt = _attempt(f"attempt-g3w3-{self.counter}")
        consent = _consent(session_id)
        self.recorder.begin_attempt(self.manifest, attempt, consent)
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


def _open_window(
    qt_app: Any, factory: _RoundFactory, first_scorer: Any = _matching_scorer
) -> QtResearchWindow:
    factory.scorer = first_scorer
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
    )
    window.show()
    window.enter_standby()
    return window


class TestG3LabelButtons:
    def test_incorrect_on_matched_never_records_prediction(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Misrecognition marked incorrect: label carries no identity."""
        factory = _RoundFactory(tmp_path, FakeCapture(_face_frames()), None)
        window = _open_window(qt_app, factory)
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        shown = window.result_text
        assert shown.startswith("person-synth-01")
        round_attempt = factory.attempt_ids[-1]
        window.press_incorrect()
        assert window.mode == "standby"
        stored = factory.recorder.read_label(round_attempt)
        assert stored.kind == "uncertain"
        assert stored.identity_id is None
        assert stored.revision == 1
        window.close()

    def test_correct_on_matched_records_shown_identity(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Correct key endorses the shown identity (operator answer)."""
        factory = _RoundFactory(tmp_path, FakeCapture(_face_frames()), None)
        window = _open_window(qt_app, factory)
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        round_attempt = factory.attempt_ids[-1]
        window.press_correct()
        assert window.mode == "standby"
        stored = factory.recorder.read_label(round_attempt)
        assert stored.kind == "enrolled"
        assert stored.identity_id == "person-synth-01"
        window.close()

    def test_not_found_round_keys(self, qt_app: Any, tmp_path: Path) -> None:
        """Not-found round: correct confirms absent, incorrect flags miss."""
        factory = _RoundFactory(tmp_path, FakeCapture(_face_frames()), None)
        window = _open_window(qt_app, factory, first_scorer=_low_scorer)
        factory.scorer = _low_scorer
        window.process_until_terminal(max_steps=300)
        assert window.mode == "result"
        assert window.result_text == "找不到此註冊人員"
        round_attempt = factory.attempt_ids[-1]
        window.press_correct()
        assert window.mode == "standby"
        stored = factory.recorder.read_label(round_attempt)
        assert stored.kind == "unenrolled"
        assert stored.identity_id is None
        window.close()

    def test_key_without_binding_refuses_fail_closed(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """No label persistence bound: key refuses, round is not lost."""
        source = FakeCapture(_face_frames())
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test"),
            source=source,
            scorer=_matching_scorer,
            session_id="g3w3-unbound-0",
            release_source_on_terminal=False,
        )
        counter = 0

        def next_session() -> tuple[DesktopSession, Any, None]:
            nonlocal counter
            counter += 1
            sibling = DesktopSession(
                engine=SessionEngine(
                    _profile(), "gallery-qt-test", "gen-qt-test"
                ),
                source=source,
                scorer=_matching_scorer,
                session_id=f"g3w3-unbound-{counter}",
                release_source_on_terminal=False,
            )
            return sibling, _consent(f"g3w3-unbound-{counter}"), None

        clock = _AdvancingClock()
        window = QtResearchWindow(
            desktop,
            consent=_consent("g3w3-unbound-0"),
            offscreen=True,
            clock_ns=clock,
            clock_advance=clock.advance,
            next_session=next_session,
        )
        window.show()
        window.enter_standby()
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        window.press_correct()
        # Refused: stays on result, never advances, nothing persisted.
        assert window.mode == "result"
        assert desktop.label is None
        window.close()

    def test_label_enrolled_without_identity_refuses(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Legacy enrolled path no longer takes the system prediction."""
        factory = _RoundFactory(tmp_path, FakeCapture(_face_frames()), None)
        window = _open_window(qt_app, factory)
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        round_attempt = factory.attempt_ids[-1]
        window.label_enrolled()
        with pytest.raises(KeyError):
            factory.recorder.read_label(round_attempt)
        window.close()
