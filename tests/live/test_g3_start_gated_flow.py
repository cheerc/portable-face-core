"""G3 R1 PR-A RED: per-round data + clock + stepping (Start-gated).

Source of truth: docs/specs/2026-09-24-g3-local-test-app.md R1 §2-3
(Start-gated: clock from this round's post-open monotonic clock) and §3
(per-round staging/crop/trace bound to the round's attempt/session),
plus docs/plans/2026-09-24-g3-start-gated-repair-plan.md D1 manifest
(frozen changes 2/3/4) and PR-A acceptance.

All tests use synthetic frames, FakeCapture, an offscreen Qt
application, and a tmp ResearchRecorder. No camera, real faces,
gallery, embeddings, or photos.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from facecore.live.capture import FakeCapture
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
)
from facecore.live.desktop import DesktopSession
from facecore.live.session import SessionEngine
from facecore.live.qt_window import QtResearchWindow
from tests.live.test_qt_window import (
    _AdvancingClock,
    _attempt,
    _consent,
    _manifest,
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
        frame = np.full((16, 16, 3), 150, dtype=np.uint8)
        frames.append(
            FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=frame)
        )
    return frames


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
        model_generation="gen-g3a-test",
        gallery_digest="gallery-g3a-test",
    )


def _low_quality_scorer(packet: FramePacket) -> FrameObservation:
    """Face present but unusable: the round must not end before 5 s."""
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=False,
        quality_reasons=("no_face_detected",),
        face_count=0,
        face_box=None,
        identity_scores={},
        quality_rank=0.0,
        model_generation="gen-g3a-test",
        gallery_digest="gallery-g3a-test",
    )


class _RoundFactory:
    """Recorder-bound round sessions mirroring the production factory."""

    def __init__(self, tmp_path: Any, source: FakeCapture) -> None:
        self.recorder = _recorder(tmp_path)
        self.manifest = _manifest()
        self.source = source
        self.counter = 0
        self.attempt_ids: list[str] = []

    def __call__(self) -> tuple[DesktopSession, Any, str]:
        self.counter += 1
        session_id = f"g3a-round-{self.counter}"
        attempt = _attempt(f"attempt-g3a-{self.counter}")
        consent = _consent(session_id)
        self.recorder.begin_attempt(self.manifest, attempt, consent)
        self.recorder.begin(session_id, consent)
        self.attempt_ids.append(attempt.attempt_id)
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-g3a-test", "gen-g3a-test"),
            source=self.source,
            scorer=_matching_scorer,
            session_id=session_id,
            release_source_on_terminal=False,
            label_recorder=self.recorder,
            label_attempt_id=attempt.attempt_id,
        )
        return desktop, consent, attempt.attempt_id


class TestG3StartGatedFlow:
    def test_round_stages_own_frames_and_mapping(
        self, qt_app: Any, tmp_path: Any
    ) -> None:
        """PR-A change 4: the round (not the initial session) owns frames.

        After one standby-triggered round reaches terminal, the round's
        session must hold staged frames and the crop mapping must be
        persisted under the round's attempt id.
        """
        source = FakeCapture(_face_frames())
        factory = _RoundFactory(tmp_path, source)
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
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        round_session_id = window.desktop.session_id
        round_attempt_id = window.attempt_id
        state = factory.recorder._active.get(round_session_id)
        assert state is not None, "round session must be active for staging"
        assert state.frame_count > 0, "round must stage its own frames"
        mapping = factory.recorder.read_crop_mapping(round_attempt_id)
        assert mapping["frame_w"] == 16
        assert mapping["frame_h"] == 16
        window.close()

    def test_round_clock_anchors_at_round_open(
        self, qt_app: Any, tmp_path: Any
    ) -> None:
        """PR-A change 2: the 5 s window starts at this round's open.

        The window records each round's start (taken after the round's
        source open) as ``round_start_ns``. The second round's anchor
        must be later than the first round's: rounds must not share one
        pre-anchored clock value (the pre-R1 bug anchored every round
        at a single ``start_ns`` taken before the factory existed).
        """
        source = FakeCapture(_face_frames(120))
        factory = _RoundFactory(tmp_path, source)
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
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        first_anchor = window.round_start_ns
        assert first_anchor is not None
        window.press_correct()
        assert window.mode == "standby"
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        second_anchor = window.round_start_ns
        assert second_anchor is not None
        assert second_anchor > first_anchor, (
            "round 2 clock must re-anchor at its own open, "
            f"got {second_anchor} <= {first_anchor}"
        )
        window.close()

    def test_low_quality_round_does_not_finish_before_deadline(
        self, qt_app: Any, tmp_path: Any
    ) -> None:
        """PR-A change 3: bounded steps must not finish before 5 s.

        With only unusable frames, a 50-step drive must leave the round
        running (no terminal), so the UI can keep ticking until the real
        deadline. A step-exhausted early finish (invalid_input with
        zero_usable_frames_collected) fails this test.
        """
        source = FakeCapture(_face_frames())
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-g3a-test", "gen-g3a-test"),
            source=source,
            scorer=_low_quality_scorer,
            session_id="g3a-slow-round",
            release_source_on_terminal=False,
        )
        desktop.on_start(_consent("g3a-slow-round"), now_ns=0, device_id="fake")
        terminal = desktop.run_until_terminal(
            max_steps=50, finish_on_exhaust=False
        )
        assert terminal is None, (
            "50 bounded steps must not finish a round whose 5 s window "
            "has not elapsed"
        )
        assert desktop.state == "running"
        desktop.close()
