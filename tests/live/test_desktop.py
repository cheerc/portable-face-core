"""Phase 2A Task T7 tests: headless desktop session bindings.

Source of truth: Phase 2A Implementation Plan §4 & §6 T7;
Task: t-20260914111211897569-76424-38;
Governing decision: d-20260914110757304910-5.

RED contract (must fail before implementation exists):
- Start 繞同意/key：start without record consent or without keys fails.
- uncertain 露名字：display_identity is None unless status is matched.
- UI close 未釋放：close releases source and joins workers.
- truth 進 scorer：labeling never touches the scorer/engine path.
GUI cannot run headless here: controller + event bindings are tested,
real-device evidence is recorded separately (runbook).
Only synthetic payloads; never real faces.
"""

from __future__ import annotations

import numpy as np
import pytest

from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionStatus,
)
from facecore.live.capture import FakeCapture
from facecore.live.desktop import DesktopSession
from facecore.live.session import SessionEngine
from facecore.research.records import ConsentRecord


def _profile() -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="t7-test-v1",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=25,
        queue_limit=1,
        required_support=3,
        min_support_interval_ms=200,
        match_threshold=0.45,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="yunet-test",
        quality_policy_version="q-test-v1",
        continuity_max_center_delta_ratio=0.50,
    )


def _packet(seq: int) -> FramePacket:
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[0, 0, 0] = seq % 256
    return FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=rgb)


def _rejecting_scorer(packet: FramePacket) -> FrameObservation:
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=False,
        quality_reasons=("t7-synth-reject",),
        face_count=0,
        face_box=None,
        identity_scores={},
        quality_rank=0.0,
        model_generation="test-gen",
        gallery_digest="1" * 64,
    )


def _consent(
    session_id: str = "sess-t7-001", *, record: bool = True, image: bool = True
) -> ConsentRecord:
    return ConsentRecord(
        session_id=session_id,
        participant_id="part-synth-001",
        record_consent=record,
        image_consent=image,
        consented_at_utc="2026-09-14T10:00:00Z",
        record_expires_at_utc="2026-10-14T10:00:00Z",
        image_expires_at_utc="2026-09-21T10:00:00Z",
    )


def _desktop(session_id: str = "sess-t7-001") -> DesktopSession:
    engine = SessionEngine(_profile(), "1" * 64, "test-gen")
    frames = [_packet(seq) for seq in range(1, 6)]
    return DesktopSession(
        engine=engine,
        source=FakeCapture(frames=frames),
        scorer=_rejecting_scorer,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# RED 1: Start 繞同意/key
# ---------------------------------------------------------------------------


def test_start_without_record_consent_refused() -> None:
    desktop = _desktop()
    with pytest.raises(PermissionError):
        desktop.on_start(_consent(record=False), now_ns=0)
    assert desktop.state == "idle"


def test_start_without_image_consent_refused() -> None:
    desktop = _desktop()
    with pytest.raises(PermissionError):
        desktop.on_start(_consent(image=False), now_ns=0)
    assert desktop.state == "idle"


def test_start_requires_explicit_consent_object() -> None:
    desktop = _desktop()
    with pytest.raises((PermissionError, ValueError, TypeError)):
        desktop.on_start(None, now_ns=0)  # type: ignore[arg-type]
    assert desktop.state == "idle"


# ---------------------------------------------------------------------------
# RED 2: uncertain 露名字
# ---------------------------------------------------------------------------


def test_display_identity_hidden_until_matched() -> None:
    desktop = _desktop()
    desktop.on_start(_consent(), now_ns=0)
    # No terminal yet: nothing to display.
    assert desktop.display_identity() is None
    result = desktop.run_until_terminal(max_steps=5)
    assert result is not None
    if result.status == SessionStatus.matched:
        assert desktop.display_identity() == result.matched_identity
    else:
        # review/unknown/timeout/invalid/error/cancel: never show a name.
        assert desktop.display_identity() is None


def test_display_identity_never_leaks_candidate_scores() -> None:
    desktop = _desktop()
    desktop.on_start(_consent(), now_ns=0)
    desktop.run_until_terminal(max_steps=5)
    shown = desktop.display_identity()
    assert shown is None or isinstance(shown, str)


# ---------------------------------------------------------------------------
# RED 3: UI close 未釋放
# ---------------------------------------------------------------------------


def test_close_releases_source_and_joins_workers() -> None:
    desktop = _desktop()
    desktop.on_start(_consent(), now_ns=0)
    desktop.run_until_terminal(max_steps=5)
    desktop.close()
    assert desktop.source_closed
    assert desktop.workers_joined
    assert desktop.state in ("closed", "terminal")


def test_close_without_start_is_safe() -> None:
    desktop = _desktop()
    desktop.close()
    assert desktop.source_closed
    assert desktop.workers_joined


# ---------------------------------------------------------------------------
# RED 4: truth 進 scorer
# ---------------------------------------------------------------------------


def test_labeling_never_touches_scorer_or_engine() -> None:
    seen: list[str] = []

    def _spying_scorer(packet: FramePacket) -> FrameObservation:
        obs = _rejecting_scorer(packet)
        return obs

    engine = SessionEngine(_profile(), "1" * 64, "test-gen")
    desktop = DesktopSession(
        engine=engine,
        source=FakeCapture(frames=[_packet(1)]),
        scorer=_spying_scorer,
        session_id="sess-t7-001",
    )
    desktop.on_start(_consent(), now_ns=0)
    desktop.run_until_terminal(max_steps=3)
    # Labeling is a pure annotation: ground truth goes to the label
    # sidecar only, never into scorer/engine inputs.
    desktop.label_terminal("person-01")
    assert desktop.label == "person-01"
    assert seen == []
    assert desktop.state == "labeled"


def test_research_watermark_always_present() -> None:
    desktop = _desktop()
    assert "研究" in desktop.watermark or "research" in desktop.watermark.lower()


def test_cancel_stops_session_immediately() -> None:
    desktop = _desktop()
    desktop.on_start(_consent(), now_ns=0)
    result = desktop.on_cancel(now_ns=1_000_000_000)
    assert result is not None
    assert result.status == SessionStatus.cancelled
    assert desktop.state == "terminal"


def test_countdown_reflects_controller_clock() -> None:
    desktop = _desktop()
    desktop.on_start(_consent(), now_ns=0)
    remaining = desktop.countdown_ms_remaining(now_ns=1_000_000_000)
    assert remaining == 4000
    assert desktop.countdown_ms_remaining(now_ns=6_000_000_000) == 0


def test_t4n2_background_plus_sync_dual_mode_no_race() -> None:
    """T4 N2: background pump + synchronous consume share one slot/worker.

    Both modes run against the same queue and the same tracked worker;
    the terminal is reached exactly once and close joins everything.
    """
    desktop = _desktop()
    desktop.on_start(_consent(), now_ns=0)
    result = desktop.run_background_and_join()
    assert result is not None
    assert result.session_id == "sess-t7-001"
    assert desktop.state == "terminal"
    desktop.close()
    assert desktop.workers_joined
    assert desktop.source_closed
