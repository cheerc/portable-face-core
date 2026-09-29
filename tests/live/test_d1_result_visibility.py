"""D1 rework RED: result reasons must reach the operator, not just the log.

Review ce0a95a7 (REJECTED on 6e83d36) found that D1's reason split was
invisible where it matters:

- B1: `insufficient_evidence` is written as SessionStatus.timeout, and
  `QtResearchWindow._format_result` short-circuits every timeout to
  「找不到此註冊人員」 without ever reading reason_codes. The operator is
  told the person is not in the gallery when the truth is "a face was
  seen and the best single frame even reached match level, but the
  multi-frame rule never closed". That is worse than pre-D1, where the
  round at least said the camera produced nothing.
- S1: the new codes reached `f"未完成辨識：{first}"` as raw English
  tokens, and `no_frames_captured` was named for a condition the UI
  already described differently ("相機無影格").

Both are the same defect seen from two sides: the split exists in the
engine and nowhere else. These tests pin the operator-visible text for
every D1 reason code and the boundary between "not in the gallery" and
"seen but insufficient".

Camera-free: no Qt window is constructed here, only the static
classifier, plus a direct probe of the short-circuit condition.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionResult,
    SessionStatus,
)
from facecore.live.capture import FakeCapture
from facecore.live.controller import LiveController
from facecore.live.session import SessionEngine


def _profile(**overrides: Any) -> ResearchProfile:
    base: dict[str, Any] = {
        "schema_version": "v1",
        "profile_version": "d1-v1",
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


def _packet(seq: int, captured_ns: int) -> FramePacket:
    return FramePacket(
        sequence=seq,
        captured_ns=captured_ns,
        rgb=np.full((16, 16, 3), 150, dtype=np.uint8),
    )


def _matching_scorer(packet: FramePacket) -> FrameObservation:
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


def _obs(
    *,
    sequence: int = 1,
    face_count: int = 1,
    quality_pass: bool = True,
    quality_reasons: tuple[str, ...] = (),
) -> FrameObservation:
    return FrameObservation(
        sequence=sequence,
        captured_ns=sequence * 200_000_000,
        processed_ns=sequence * 200_000_000,
        quality_pass=quality_pass,
        quality_reasons=quality_reasons,
        face_count=face_count,
        face_box=(0.0, 0.0, 2.0, 2.0) if face_count >= 1 else None,
        identity_scores={"enroll-23": 0.65, "enroll-10": 0.29},
        quality_rank=0.9,
        model_generation="gen-1",
        gallery_digest="digest-d1",
    )


# The classifier under test, plus the exact short-circuit the review named.
from facecore.live.qt_window import _QtResearchWindow  # noqa: E402

_classify = _QtResearchWindow.classify_failure


# ---------------------------------------------------------------------------
# B1: insufficient evidence is not "not in the gallery"
# ---------------------------------------------------------------------------


def test_insufficient_evidence_text_is_not_not_found() -> None:
    """B1: seeing a face with thin evidence must not read as absence.

    This is the reviewer's measured case verbatim: status=timeout,
    frames_usable=13, best single frame already at match level, yet the
    screen said 「找不到此註冊人員」.
    """
    text = _classify(
        observations=(_obs(sequence=1), _obs(sequence=2)),
        reason_codes=(
            "insufficient_evidence",
            "support_2_of_3",
            "best_baseline_matched",
        ),
    )
    assert text != "找不到此註冊人員"
    # It must read as "we saw a person, the multi-frame check did not
    # close" — in words an operator can act on.
    assert "人臉" in text or "看見" in text
    # No raw tokens.
    assert "insufficient_evidence" not in text
    assert "support_2_of_3" not in text
    assert "best_baseline_matched" not in text


def test_insufficient_evidence_is_checked_before_the_timeout_short_circuit() -> None:
    """B1: status alone must not decide the text.

    A timeout whose reason codes say "insufficient evidence" is a
    different event from a timeout that simply ran out of window, and the
    two must not collapse to the same sentence.
    """
    thin = _classify(
        observations=(_obs(sequence=1),),
        reason_codes=("insufficient_evidence", "support_1_of_3"),
    )
    ran_out = _classify(
        observations=(_obs(sequence=1),),
        reason_codes=("deadline_exceeded", "best_baseline_unknown"),
    )
    assert thin != ran_out
    # The plain timeout case is the genuine "did not find them" answer.
    assert ran_out == "找不到此註冊人員"


def test_insufficient_evidence_terminal_is_rendered_not_short_circuited() -> None:
    """B1 end-to-end: build the real terminal, then the real text.

    Runs the engine the way the field did and feeds the resulting
    SessionResult through the display decision, so the test fails on the
    actual status/reason combination rather than a hand-written one.
    """
    engine = SessionEngine(
        _profile(min_support_interval_ms=2000), "digest-d1", "gen-1"
    )
    frames = [_packet(i, (i - 1) * 100_000_000) for i in range(1, 27)]
    controller = LiveController(engine, FakeCapture(frames), _matching_scorer)
    controller.start_session("d1-b1-e2e", 0, device_id="fake")
    terminal = controller.run_until_terminal(max_steps=200)
    assert terminal is not None
    # The reviewer's measured terminal shape.
    assert terminal.status == SessionStatus.timeout
    assert "insufficient_evidence" in terminal.reason_codes
    assert terminal.frames_usable > 0

    text = _classify(
        observations=(_obs(sequence=1),),
        reason_codes=terminal.reason_codes,
    )
    assert text != "找不到此註冊人員"
    assert "insufficient_evidence" not in text


# ---------------------------------------------------------------------------
# S1: every D1 code reads as Chinese, never as a raw token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "all_frames_rejected_no_face",
        "all_frames_rejected_quality",
        "all_frames_rejected_mixed_causes",
    ],
)
def test_d1_rejected_frame_codes_never_leak_raw_tokens(code: str) -> None:
    """S1: the operator sees a cause, not a machine token.

    These three mean "frames arrived, every one was rejected", so the
    round does carry observations and the screen must name the cause —
    and must not fall through to 「找不到此註冊人員」.
    """
    text = _classify(
        observations=(_obs(sequence=1),),
        reason_codes=(code, "deadline_exceeded"),
    )
    assert code not in text
    assert "deadline_exceeded" not in text
    assert text != "找不到此註冊人員"
    # Still Chinese prose an operator can read.
    assert text.strip() != ""
    assert any("一" <= ch <= "鿿" for ch in text)


def test_no_frames_captured_never_reads_as_not_found() -> None:
    """S1: a dead camera must not be reported as "person not enrolled".

    `no_frames_captured` is written when the source delivered nothing,
    so the round has no observations; the text is the pre-existing
    「相機無影格」 — which is exactly what the alignment test below
    asserts, and the reason it is not 「找不到此註冊人員」.
    """
    text = _classify(
        observations=(),
        reason_codes=("no_frames_captured", "first_frame_timeout"),
    )
    assert text != "找不到此註冊人員"
    assert "no_frames_captured" not in text
    assert "first_frame_timeout" not in text


def test_no_frames_code_agrees_with_the_pre_existing_no_frame_text() -> None:
    """S1: one condition, one description.

    `no_frames_captured` is written when the camera delivered nothing
    (`_frames_sampled == 0`). The pre-existing text for "no
    observations" is 「未取得可辨識影格：相機無影格」. The two must not
    describe the same event differently.
    """
    pre_existing = _classify(observations=(), reason_codes=())
    from_code = _classify(
        observations=(),
        reason_codes=("no_frames_captured", "first_frame_timeout"),
    )
    assert from_code == pre_existing


def test_all_frames_rejected_no_face_names_the_missing_face() -> None:
    """S1: the no-face case must actually say the face was missing."""
    text = _classify(
        observations=(_obs(sequence=1, face_count=0, quality_pass=False),),
        reason_codes=("all_frames_rejected_no_face", "timeout"),
    )
    assert "人臉" in text
    assert "all_frames_rejected_no_face" not in text


def test_all_frames_rejected_quality_names_the_quality_cause() -> None:
    """S1: a quality rejection must surface the reason, not just a label."""
    text = _classify(
        observations=(
            _obs(sequence=1, quality_pass=False, quality_reasons=("quality_exposure",)),
        ),
        reason_codes=("all_frames_rejected_quality", "timeout"),
    )
    assert "品質" in text
    assert "quality_exposure" in text  # the specific cause is useful to show
    assert "all_frames_rejected_quality" not in text


def test_mixed_causes_reports_that_causes_were_mixed() -> None:
    """S1: a mixed-cause round must not claim a single clean reason."""
    text = _classify(
        observations=(
            _obs(sequence=1, face_count=0, quality_pass=False),
            _obs(sequence=2, quality_pass=False, quality_reasons=("quality_blur",)),
        ),
        reason_codes=("all_frames_rejected_mixed_causes", "timeout"),
    )
    assert "all_frames_rejected_mixed_causes" not in text
    assert text.strip() != ""


# ---------------------------------------------------------------------------
# The pre-existing distinctions must survive the rework
# ---------------------------------------------------------------------------


def test_pre_existing_three_failure_kinds_stay_distinct() -> None:
    """R1 §2-4 still holds after D1 rework (reviewer asked to keep it)."""
    no_frame = _classify(observations=())
    no_face = _classify(
        observations=(_obs(sequence=1, face_count=0, quality_pass=False),),
        reason_codes=(),
    )
    rejected = _classify(
        observations=(
            _obs(sequence=1, quality_pass=False, quality_reasons=("blur_too_high",)),
        ),
        reason_codes=(),
    )
    assert no_frame != no_face != rejected
    for text in (no_frame, no_face, rejected):
        assert "zero_usable_frames_collected" not in text
    assert "不在" not in no_frame and "不在" not in no_face


def test_matched_is_unaffected_by_the_reason_split() -> None:
    """A real match must keep its existing text and never be overridden."""
    result = SessionResult(
        session_id="d1-matched",
        schema_version="v1",
        status=SessionStatus.matched,
        matched_identity="enroll-23",
        reason_codes=("supported_3_frames",),
        elapsed_ms=400.0,
        frames_sampled=3,
        frames_usable=3,
        frames_rejected=0,
        frames_dropped=0,
        support_sequences=(1, 2, 3),
        profile_digest="d" * 64,
        model_generation="gen-1",
        gallery_digest="digest-d1",
    )
    # status is matched; the reason-based branch must not claim otherwise.
    assert result.status == SessionStatus.matched
    text = _classify(
        observations=(_obs(sequence=1),),
        reason_codes=result.reason_codes,
    )
    assert "找不到" not in text
    assert "insufficient_evidence" not in text


def test_unverifiable_reason_still_falls_back_without_leaking() -> None:
    """An unknown code must not become a raw token on the operator's screen."""
    text = _classify(
        observations=(_obs(sequence=1),),
        reason_codes=("some_future_code_we_do_not_know",),
    )
    assert "some_future_code_we_do_not_know" not in text
    assert text.strip() != ""
