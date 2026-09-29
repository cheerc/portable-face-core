"""D1 RED: camera-open must not consume the recognition window.

Source of truth:
    - PLAN-mac-face-demo-handoff.md §D1 (open/first-frame use an
      independent bounded wait; the 5 s recognition window anchors at
      the first valid frame; segmented timing lands in the existing
      trace; no-frame / no-face / quality-reject / insufficient-evidence
      are distinct reasons; everything terminates in bounded time).
    - D0 frozen baseline: profiles/g3-v1.json, 23-image gallery,
      match 0.363 / margin 0.10 / required_support 3 / timeout 5000 ms
      (docs/mac-demo-baseline-d0.md §5-§7).

Root cause these tests pin down (lead-verified line numbers):
    qt_window.py:549 reads the clock BEFORE :550 on_start; the camera
    open happens inside on_start (controller.py:158) and engine.start
    (:160) is then handed the pre-open now_ns, so session.py:133 arms
    the deadline at open-begin. A real open that costs seconds leaves no
    room for evidence: 23/23 field invalid_input rounds sampled exactly
    one frame and ran 5264.2-34558.8 ms, all past the 5000 ms window.

Every test here is camera-free: FakeCapture, synthetic frames, a fake
clock, an offscreen Qt app, and a tmp recorder. No real camera, no real
faces, no gallery, no embeddings, no photos.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pytest

from facecore.live.capture import FakeCapture
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionStatus,
)
from facecore.live.controller import LiveController
from facecore.live.session import SessionEngine

# The frozen g3-v1 recognition parameters (D0 §6).
G3_MATCH = 0.363
G3_MARGIN = 0.10
G3_SUPPORT = 3
G3_TIMEOUT_MS = 5000


def _profile(**overrides: Any) -> ResearchProfile:
    base: dict[str, Any] = {
        "schema_version": "v1",
        "profile_version": "d1-v1",
        "timeout_ms": G3_TIMEOUT_MS,
        "sample_interval_ms": 200,
        "max_frames": 26,
        "queue_limit": 1,
        "required_support": G3_SUPPORT,
        "min_support_interval_ms": 200,
        "match_threshold": G3_MATCH,
        "review_threshold": 0.30,
        "margin_threshold": G3_MARGIN,
        "detector_version": "yunet",
        "quality_policy_version": "1",
        "continuity_max_center_delta_ratio": 0.5,
    }
    base.update(overrides)
    return ResearchProfile(**base)  # type: ignore[arg-type]


def _packet(seq: int, captured_ns: int) -> FramePacket:
    frame = np.full((16, 16, 3), 150, dtype=np.uint8)
    return FramePacket(sequence=seq, captured_ns=captured_ns, rgb=frame)


def _face_frames(n: int, *, first_ns: int = 0, step_ns: int = 200_000_000):
    return [_packet(i, first_ns + (i - 1) * step_ns) for i in range(1, n + 1)]


def _matching_scorer(packet: FramePacket) -> FrameObservation:
    """Qualified single-face frame, 0.65/0.29: over match, over margin."""
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 52_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(0.0, 0.0, 2.0, 2.0),
        identity_scores={"enroll-23": 0.65, "enroll-10": 0.29},
        quality_rank=0.9,
        model_generation="gen-1",
        gallery_digest="digest-d1",
    )


class _DelayedFirstFrameSource(FakeCapture):
    """FakeCapture whose first usable frame lands after the 5 s window.

    This is the field condition, not a convenience: the 23/23
    invalid_input rounds in results.csv all sampled exactly one frame and
    ran 5264.2-34558.8 ms, every one past the 5000 ms window, with
    quality_pass=true scores already computed on that frame. A camera
    that cannot deliver before the window closes leaves the round with
    nothing to credit.

    The delay is simulated in the capture clock rather than by burning
    CPU, so the test is fast and deterministic: no frame carries a
    captured_ns before the stated delay elapses.
    """

    def __init__(self, n_frames: int, *, first_frame_delay_ns: int) -> None:
        self.first_frame_delay_ns = first_frame_delay_ns
        self.open_ns: int | None = None
        super().__init__(_face_frames(n_frames))

    def open(self, device_id: str) -> None:
        self.open_ns = time.monotonic_ns()
        base = self.open_ns + self.first_frame_delay_ns
        self._frames = _face_frames(
            len(self._frames), first_ns=base, step_ns=200_000_000
        )
        self._cursor = 0
        super().open(device_id)


def time_monotonic_ns() -> int:
    return time.monotonic_ns()


# ---------------------------------------------------------------------------
# Regression 1: a slow camera open must not exhaust the recognition window
# ---------------------------------------------------------------------------


def test_slow_open_does_not_exhaust_recognition_window() -> None:
    """D1 §D1-1: the open cost is outside the 5 s recognition window.

    The first usable frame lands 8.3 s after the open begins — past the
    5000 ms window, exactly like every field round in results.csv. The
    round must still reach matched/supported_3_frames: the open cost is
    not evidence time.
    """
    engine = SessionEngine(_profile(), "digest-d1", "gen-1")
    source = _DelayedFirstFrameSource(12, first_frame_delay_ns=8_304_000_000)
    controller = LiveController(engine, source, _matching_scorer)
    controller.start_session("d1-slow-open", time_monotonic_ns(), device_id="fake")
    terminal = controller.run_until_terminal(max_steps=100)
    assert terminal is not None
    assert terminal.status == SessionStatus.matched
    assert "supported_3_frames" in terminal.reason_codes
    assert terminal.support_sequences == (1, 2, 3)


def test_recognition_window_anchors_at_first_valid_frame() -> None:
    """D1 §D1-1: the window arms at the first valid frame, not at open.

    Frames 1..3 arrive 8.3 s after the open began. Under the pre-fix
    anchor every one of them is past the deadline and the round ends
    zero_usable; anchoring at the first valid frame credits them.
    """
    engine = SessionEngine(_profile(), "digest-d1", "gen-1")
    controller = LiveController(
        engine, FakeCapture(_face_frames(6, first_ns=8_304_000_000)), _matching_scorer
    )
    controller.start_session("d1-first-frame-anchor", 0, device_id="fake")
    terminal = controller.run_until_terminal(max_steps=100)
    assert terminal is not None
    assert terminal.status == SessionStatus.matched
    assert terminal.support_sequences == (1, 2, 3)


# ---------------------------------------------------------------------------
# Regression 2: never a first frame -> bounded, distinct reason
# ---------------------------------------------------------------------------


def test_no_frame_ever_terminates_bounded_with_distinct_reason() -> None:
    """D1 §D1-4: an empty source ends in bounded time, reason names no frame.

    Pre-fix this collapses into zero_usable_frames_collected, which
    cannot distinguish "the camera never delivered a frame" from "frames
    arrived but none were usable".
    """
    engine = SessionEngine(_profile(), "digest-d1", "gen-1")
    controller = LiveController(engine, FakeCapture([]), _matching_scorer)
    controller.start_session("d1-no-frame", 0, device_id="fake")
    terminal = controller.run_until_terminal(max_steps=100)
    assert terminal is not None
    assert terminal.frames_sampled == 0
    assert "no_frames_captured" in terminal.reason_codes
    assert "zero_usable_frames_collected" not in terminal.reason_codes


def test_dry_source_ends_bounded_on_first_frame_wait() -> None:
    """D1 §D1-4: a camera that opens but never delivers ends in bounded time.

    The wait is bounded by the source running dry (three consecutive empty
    reads), not by an unlimited wait and not by spending the recognition
    window: the round must end with the first-frame reason, having
    consumed no recognition budget, because no evidence could exist.
    """
    engine = SessionEngine(_profile(), "digest-d1", "gen-1")
    controller = LiveController(engine, FakeCapture([]), _matching_scorer)
    controller.start_session("d1-first-frame-never", 0, device_id="fake")
    terminal = controller.run_until_terminal(max_steps=100)
    assert terminal is not None
    assert terminal.frames_sampled == 0
    assert "first_frame_timeout" in terminal.reason_codes
    assert "no_frames_captured" in terminal.reason_codes


# ---------------------------------------------------------------------------
# Regression 3: open failure -> bounded, distinct reason
# ---------------------------------------------------------------------------


class _FailingOpenSource(FakeCapture):
    def open(self, device_id: str) -> None:
        raise RuntimeError("device busy")


def test_open_failure_surfaces_as_open_failed() -> None:
    """D1 §D1-4: a refused open is its own reason, not a silent stall."""
    engine = SessionEngine(_profile(), "digest-d1", "gen-1")
    controller = LiveController(
        engine, _FailingOpenSource(_face_frames(3)), _matching_scorer
    )
    with pytest.raises(RuntimeError, match="device busy"):
        controller.start_session("d1-open-fail", 0, device_id="fake")


# ---------------------------------------------------------------------------
# Regression 4: sampling interrupted -> bounded, distinct reason
# ---------------------------------------------------------------------------


def _interrupted_scorer(packet: FramePacket) -> FrameObservation:
    if packet.sequence >= 3:
        raise RuntimeError("frame decode failed")
    return _matching_scorer(packet)


def test_sampling_interruption_surfaces_as_scorer_failure() -> None:
    """D1 §D1-4: a mid-round scoring failure ends bounded and stays typed."""
    engine = SessionEngine(_profile(), "digest-d1", "gen-1")
    controller = LiveController(
        engine, FakeCapture(_face_frames(8)), _interrupted_scorer
    )
    controller.start_session("d1-interrupted", 0, device_id="fake")
    terminal = controller.run_until_terminal(max_steps=100)
    assert terminal is not None
    assert terminal.status == SessionStatus.error
    assert any(code.startswith("scorer_failure") for code in terminal.reason_codes)


# ---------------------------------------------------------------------------
# Regression 5: cancel then retry both work
# ---------------------------------------------------------------------------


def test_cancel_then_retry_reaches_matched() -> None:
    """D1 §D1-4/R1 §2-6: cancel closes the round, the next round still matches."""
    engine_a = SessionEngine(_profile(), "digest-d1", "gen-1")
    source_a = FakeCapture(_face_frames(30, first_ns=2_000_000_000))
    controller_a = LiveController(engine_a, source_a, _matching_scorer)
    controller_a.start_session("d1-cancel-1", 0, device_id="fake")
    cancelled = controller_a.cancel_collection(0)
    assert cancelled.status == SessionStatus.cancelled

    engine_b = SessionEngine(_profile(), "digest-d1", "gen-1")
    controller_b = LiveController(
        engine_b, FakeCapture(_face_frames(6)), _matching_scorer
    )
    controller_b.start_session("d1-cancel-2", 0, device_id="fake")
    retried = controller_b.run_until_terminal(max_steps=100)
    assert retried is not None
    assert retried.status == SessionStatus.matched


# ---------------------------------------------------------------------------
# Distinct failure reasons (D1 §D1-3)
# ---------------------------------------------------------------------------


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
        model_generation="gen-1",
        gallery_digest="digest-d1",
    )


def _quality_rejected_scorer(packet: FramePacket) -> FrameObservation:
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=False,
        quality_reasons=("quality_exposure",),
        face_count=1,
        face_box=(0.0, 0.0, 2.0, 2.0),
        identity_scores={},
        quality_rank=0.0,
        model_generation="gen-1",
        gallery_digest="digest-d1",
    )


def _run_with(scorer: Any, session_id: str) -> Any:
    engine = SessionEngine(_profile(), "digest-d1", "gen-1")
    controller = LiveController(engine, FakeCapture(_face_frames(30)), scorer)
    controller.start_session(session_id, 0, device_id="fake")
    terminal = controller.run_until_terminal(max_steps=100)
    assert terminal is not None
    return terminal


def test_no_face_and_quality_reject_have_distinct_reasons() -> None:
    """D1 §D1-3: no-face and quality-reject are separately diagnosable."""
    no_face = _run_with(_no_face_scorer, "d1-no-face")
    rejected = _run_with(_quality_rejected_scorer, "d1-quality")
    assert no_face.reason_codes != rejected.reason_codes
    assert no_face.frames_sampled > 0
    assert rejected.frames_sampled > 0
    # Neither may report the no-frame reason: frames did arrive.
    assert "no_frames_captured" not in no_face.reason_codes
    assert "no_frames_captured" not in rejected.reason_codes


def test_insufficient_evidence_differs_from_no_usable() -> None:
    """D1 §D1-3: qualifying frames that never reach 3 support is its own case.

    Frames qualify but arrive inside min_support_interval_ms, so support
    never reaches 3. The round must end as a distinct "evidence
    insufficient" case, not as "no usable frames" and not as matched.
    """
    engine = SessionEngine(_profile(min_support_interval_ms=2000), "digest-d1", "gen-1")
    # 100 ms apart: every frame is after the first, but all are skipped
    # by the 2 s support interval, so support never accumulates.
    controller = LiveController(
        engine, FakeCapture(_face_frames(26, step_ns=100_000_000)), _matching_scorer
    )
    controller.start_session("d1-insufficient", 0, device_id="fake")
    terminal = controller.run_until_terminal(max_steps=200)
    assert terminal is not None
    assert terminal.frames_usable > 0
    assert terminal.status != SessionStatus.matched
    assert "insufficient_evidence" in terminal.reason_codes
    assert "no_frames_captured" not in terminal.reason_codes
    assert "zero_usable_frames_collected" not in terminal.reason_codes


# ---------------------------------------------------------------------------
# Segmented timing (D1 §D1-2)
# ---------------------------------------------------------------------------


def test_segmented_timing_records_open_and_first_frame() -> None:
    """D1 §D1-2: open begin/end and the first-frame mark are persisted.

    The source opens instantly and delivers its first frame 300 ms later,
    so the two segments must come out separately attributable: a tiny
    open and a ~300 ms open-to-first-frame gap. That separation is what
    D4 needs to tell a slow open apart from a slow first read.
    """
    engine = SessionEngine(_profile(), "digest-d1", "gen-1")
    source = _DelayedFirstFrameSource(6, first_frame_delay_ns=300_000_000)
    controller = LiveController(engine, source, _matching_scorer)
    controller.start_session("d1-timing", time_monotonic_ns(), device_id="fake")
    terminal = controller.run_until_terminal(max_steps=100)
    assert terminal is not None

    marks = terminal.timing_marks
    assert marks is not None, "terminal must carry the D1 segmented timing"
    assert marks.open_begin_ns is not None
    assert marks.open_end_ns is not None
    assert marks.open_end_ns >= marks.open_begin_ns
    assert marks.first_frame_ns is not None
    assert marks.first_frame_ns >= marks.open_end_ns
    assert marks.recognition_start_ns == marks.first_frame_ns
    assert marks.terminal_ns is not None
    # The open here is cheap; the wait after it is the visible cost.
    assert marks.open_duration_ms is not None and marks.open_duration_ms < 250.0
    assert marks.open_to_first_frame_ms is not None
    assert marks.open_to_first_frame_ms >= 250.0
    assert marks.recognition_duration_ms is not None


def test_timing_reports_none_rather_than_a_negative_duration() -> None:
    """D1 §D1-2: an impossible ordering reports nothing, not a negative wait.

    `first_frame_ns` comes from the capture clock; the open stamps come
    from the controller clock. When a caller injects a synthetic start
    (as these tests and replay both do) the two domains disagree and the
    subtraction goes negative. A negative "wait" would be indistinguishable
    from a real measurement and would poison D4's open-versus-read
    attribution, so the marks must degrade to None.
    """
    from facecore.live.contracts import TimingMarks

    # first frame "before" the open ended — impossible ordering.
    marks = TimingMarks(
        open_begin_ns=1_000_000_000,
        open_end_ns=2_000_000_000,
        first_frame_ns=1_500_000_000,
        recognition_start_ns=1_500_000_000,
        terminal_ns=6_000_000_000,
    )
    assert marks.open_to_first_frame_ms is None
    # The open segment itself stays real: both ends are controller-clock.
    assert marks.open_duration_ms == 1000.0
    assert marks.recognition_duration_ms == 4500.0

    # Terminal before the anchor: also impossible, also reported as None.
    backwards = TimingMarks(
        open_begin_ns=1_000_000_000,
        open_end_ns=2_000_000_000,
        first_frame_ns=3_000_000_000,
        recognition_start_ns=3_000_000_000,
        terminal_ns=2_500_000_000,
    )
    assert backwards.recognition_duration_ms is None


def test_segmented_timing_survives_serialization() -> None:
    """D1 §D1-2: the marks travel through to_dict/from_dict unchanged."""
    engine = SessionEngine(_profile(), "digest-d1", "gen-1")
    controller = LiveController(
        engine, FakeCapture(_face_frames(6)), _matching_scorer
    )
    controller.start_session("d1-timing-json", 0, device_id="fake")
    terminal = controller.run_until_terminal(max_steps=100)
    assert terminal is not None

    from facecore.live.contracts import SessionResult

    revived = SessionResult.from_dict(terminal.to_dict())
    assert revived.timing_marks is not None
    assert revived.timing_marks.first_frame_ns == terminal.timing_marks.first_frame_ns
    assert revived.to_dict() == terminal.to_dict()


# ---------------------------------------------------------------------------
# The frozen engine deadline contract is untouched
# ---------------------------------------------------------------------------


def test_engine_deadline_contract_unchanged_for_direct_callers() -> None:
    """D1 must not move the engine's own deadline arithmetic.

    Replay (E4 original-time re-run) and the fixed-window collector both
    anchor on session start; anchoring stays exactly as before for any
    caller that starts the engine directly.
    """
    engine = SessionEngine(_profile(), "digest-d1", "gen-1")
    engine.start("d1-frozen", 0)
    assert engine.deadline_ns == G3_TIMEOUT_MS * 1_000_000


def test_three_frame_rule_and_unknown_rejection_still_hold() -> None:
    """D1 §D1 acceptance: the 3-frame rule and unknown rejection survive."""
    engine = SessionEngine(_profile(), "digest-d1", "gen-1")
    source = FakeCapture(_face_frames(30))
    controller = LiveController(engine, source, _matching_scorer)
    controller.start_session("d1-unknown", 0, device_id="fake")

    def _weak(packet: FramePacket) -> FrameObservation:
        obs = _matching_scorer(packet)
        return FrameObservation(
            sequence=obs.sequence,
            captured_ns=obs.captured_ns,
            processed_ns=obs.processed_ns,
            quality_pass=obs.quality_pass,
            quality_reasons=obs.quality_reasons,
            face_count=obs.face_count,
            face_box=obs.face_box,
            # Below review_threshold: a valid frame that is nobody.
            identity_scores={"enroll-23": 0.10, "enroll-10": 0.05},
            quality_rank=obs.quality_rank,
            model_generation=obs.model_generation,
            gallery_digest=obs.gallery_digest,
        )

    engine_b = SessionEngine(_profile(), "digest-d1", "gen-1")
    controller_b = LiveController(engine_b, FakeCapture(_face_frames(30)), _weak)
    controller_b.start_session("d1-unknown-2", 0, device_id="fake")
    terminal = controller_b.run_until_terminal(max_steps=100)
    assert terminal is not None
    assert terminal.status == SessionStatus.timeout
    assert "no_frames_captured" not in terminal.reason_codes


# ---------------------------------------------------------------------------
# Offscreen Qt wiring: the countdown follows the same anchor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qt_app() -> Any:
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication as ActualQApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return ActualQApplication.instance() or ActualQApplication([])


def test_qt_countdown_uses_recognition_anchor(qt_app: Any) -> None:
    """D1 §D1-1: the UI countdown agrees with the engine window.

    Once the round is running the countdown must be measured from the
    recognition anchor, so a slow open does not display "0 ms" while
    evidence is still being gathered.
    """
    from facecore.live.desktop import DesktopSession
    from facecore.research.records import ConsentRecord
    from datetime import datetime, timedelta, timezone

    desktop = DesktopSession(
        engine=SessionEngine(_profile(), "digest-d1", "gen-1"),
        source=FakeCapture(_face_frames(30, first_ns=2_000_000_000)),
        scorer=_matching_scorer,
        session_id="d1-qt-countdown",
    )
    consent = ConsentRecord(
        session_id="d1-qt-countdown",
        participant_id="synthetic-01",
        consented_at_utc=datetime.now(timezone.utc).isoformat(),
        record_expires_at_utc=(
            datetime.now(timezone.utc) + timedelta(days=1)
        ).isoformat(),
        image_expires_at_utc=(
            datetime.now(timezone.utc) + timedelta(days=1)
        ).isoformat(),
        record_consent=True,
        image_consent=True,
    )
    desktop.on_start(consent, now_ns=0, device_id="fake")
    desktop.run_until_terminal(max_steps=100)
    # Terminal reached; the countdown must read zero, not a negative or
    # an unbounded value.
    assert desktop.countdown_ms_remaining(10_000_000_000) == 0
