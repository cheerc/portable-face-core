"""G3 W2 RED: Qt continuous mode — standby → round → result → key → standby.

Source of truth: docs/specs/2026-09-24-g3-local-test-app.md §2 (behavior
contract steps 3-6) and §7 item 3 (continuous rounds, key advances to the
next round, camera never reopened).

All tests use synthetic frames, FakeCapture, and an offscreen Qt
application. No camera, real faces, gallery, embeddings, or photos.
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
    ResearchProfile,
)
from facecore.live.desktop import DesktopSession
from facecore.live.session import SessionEngine
from facecore.live.qt_window import QtResearchWindow
from tests.live.test_qt_window import (
    _AdvancingClock,
    _consent,
)


@pytest.fixture(scope="module")
def qt_app() -> Any:
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication as ActualQApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return ActualQApplication.instance() or ActualQApplication([])


def _w2_profile() -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="g3-v1-test",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=26,
        queue_limit=1,
        required_support=1,
        min_support_interval_ms=1,
        match_threshold=0.363,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="det-g3w2-test",
        quality_policy_version="quality-g3w2-test",
        continuity_max_center_delta_ratio=0.5,
    )


def _face_packet(seq: int) -> FramePacket:
    frame = np.full((8, 8, 3), 150, dtype=np.uint8)
    return FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=frame)


def _empty_packet(seq: int) -> FramePacket:
    frame = np.full((8, 8, 3), 20, dtype=np.uint8)
    return FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=frame)


def _no_face_scorer(packet: FramePacket) -> FrameObservation:
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
        model_generation="gen-g3w2-test",
        gallery_digest="gallery-g3w2-test",
    )


def _g3_matching_scorer(packet: FramePacket) -> FrameObservation:
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
        model_generation="gen-g3w2-test",
        gallery_digest="gallery-g3w2-test",
    )


def _low_score_scorer(packet: FramePacket) -> FrameObservation:
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
        model_generation="gen-g3w2-test",
        gallery_digest="gallery-g3w2-test",
    )


def _brightness_scorer(packet: FramePacket) -> FrameObservation:
    """Bright frame = face present; dark frame = no face (trigger test)."""
    if int(packet.rgb.mean()) > 100:
        return _g3_matching_scorer(packet)
    return _no_face_scorer(packet)


class _OpenCloseCounter(FakeCapture):
    """FakeCapture wrapper counting open/close calls (camera-reopen evidence)."""

    def __init__(self, frames: list[FramePacket]) -> None:
        super().__init__(frames=frames)
        self.open_calls = 0
        self.close_calls = 0

    def open(self, device_id: str) -> None:
        self.open_calls += 1
        super().open(device_id)

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def _make_desktop(
    source: FakeCapture, session_id: str, scorer: Any = _g3_matching_scorer
) -> tuple[DesktopSession, Any]:
    return (
        DesktopSession(
            engine=SessionEngine(
                _w2_profile(), "gallery-g3w2-test", "gen-g3w2-test"
            ),
            source=source,
            scorer=scorer,
            session_id=session_id,
            release_source_on_terminal=False,
        ),
        _consent(session_id),
    )


class TestQtContinuousMode:
    def test_two_rounds_return_to_standby_without_source_close(
        self, qt_app: Any
    ) -> None:
        """Two full rounds via key presses; source never closed mid-loop."""
        frames = [_face_packet(seq) for seq in range(1, 60)]
        source = _OpenCloseCounter(frames=frames)
        counter = 0

        def next_session() -> tuple[DesktopSession, Any]:
            nonlocal counter
            counter += 1
            return _make_desktop(source, f"g3-round-{counter}")

        clock = _AdvancingClock()
        first_desktop, first_consent = _make_desktop(source, "g3-round-0")
        window = QtResearchWindow(
            first_desktop,
            consent=first_consent,
            offscreen=True,
            clock_ns=clock,
            clock_advance=clock.advance,
            next_session=next_session,
        )
        window.show()
        # Round 1: standby detects face → round starts → terminal matched.
        window.enter_standby()
        assert window.mode == "standby"
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        assert window.result_text.startswith("person-synth-01")
        # Key press returns to standby without closing the source.
        window.press_correct()
        assert window.mode == "standby"
        assert source.close_calls == 0
        assert source.is_closed is False
        # Round 2 runs on a fresh session over the same open source.
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        assert window.result_text.startswith("person-synth-01")
        window.press_incorrect()
        assert window.mode == "standby"
        assert source.close_calls == 0
        assert source.is_closed is False
        window.close()

    def test_timeout_round_shows_not_found_text(self, qt_app: Any) -> None:
        """A round with no usable frames shows the not-found message.

        The face trigger itself is covered by the two-round test; here the
        round is started manually (as if a face appeared) and then only
        below-threshold faces arrive, so the bounded session runs to the
        deadline without a match (spec §2 step 5: timeout/unknown →
        not-found text).
        """
        frames = [_face_packet(seq) for seq in range(1, 60)]
        source = _OpenCloseCounter(frames=frames)

        def next_session() -> tuple[DesktopSession, Any]:
            return _make_desktop(source, "g3-timeout-1", scorer=_low_score_scorer)

        clock = _AdvancingClock()
        first_desktop, first_consent = _make_desktop(
            source, "g3-timeout-0", scorer=_low_score_scorer
        )
        window = QtResearchWindow(
            first_desktop,
            consent=first_consent,
            offscreen=True,
            clock_ns=clock,
            clock_advance=clock.advance,
            next_session=next_session,
        )
        window.show()
        window.enter_standby()
        # Manual start: the trigger (face appears) is covered elsewhere;
        # this round then sees only faceless frames until it terminates.
        window.start_clicked()
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        assert window.result_text == "找不到此註冊人員"
        window.press_correct()
        assert window.mode == "standby"
        window.close()

    def test_placeholder_keys_do_not_write_labels(self, qt_app: Any) -> None:
        """W2 placeholder keys return to standby without label sidecars."""
        frames = [_face_packet(seq) for seq in range(1, 60)]
        source = _OpenCloseCounter(frames=frames)

        def next_session() -> tuple[DesktopSession, Any]:
            return _make_desktop(source, "g3-nolabel-1")

        clock = _AdvancingClock()
        first_desktop, first_consent = _make_desktop(source, "g3-nolabel-0")
        window = QtResearchWindow(
            first_desktop,
            consent=first_consent,
            offscreen=True,
            clock_ns=clock,
            clock_advance=clock.advance,
            next_session=next_session,
        )
        window.show()
        window.enter_standby()
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        # Placeholder keys must not touch the label sidecar: the desktop
        # stays terminal-state adjacent (never labeled) and no recorder
        # is involved (recorder=None in this continuous path).
        window.press_correct()
        assert window.mode == "standby"
        assert window.desktop.label is None
        assert window.desktop.state == "idle"
        window.close()
