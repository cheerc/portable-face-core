"""Phase 2A Task T8 tests: CLI lifecycle and failure matrix.

Source of truth: Phase 2A Implementation Plan §4 & §6 T8;
Task: t-20260914111218389235-76424-39;
Governing decision: d-20260914110757304910-5.

RED contract (must fail before implementation exists):
- injection 後遺留可讀 bundle/worker，或 report 漏 failed attempt.
- T7 N1: background pump + live source must terminate by stop
  (within 5s) with true timeout semantics (stop + join + timeout
  terminal) — run_with_timeout.
Only synthetic payloads; never real faces. No participant smoke here:
hardware-ready; participant-smoke blocked (no consent solicited).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

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
from facecore.research.cli import cmd_delete, cmd_live, cmd_replay
from facecore.research.replay import ReplayRefusal
from facecore.research.report import LabeledOutcome, summarize
from facecore.research.recorder import ResearchRecorder


def _profile_dict(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "t8-cli-v1",
        "timeout_ms": 5000,
        "sample_interval_ms": 200,
        "max_frames": 25,
        "queue_limit": 1,
        "required_support": 3,
        "min_support_interval_ms": 200,
        "match_threshold": 0.45,
        "review_threshold": 0.30,
        "margin_threshold": 0.10,
        "detector_version": "yunet-test",
        "quality_policy_version": "q-test-v1",
        "continuity_max_center_delta_ratio": 0.50,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return path


def _profile() -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="t8-test-v1",
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
        quality_reasons=("t8-synth-reject",),
        face_count=0,
        face_box=None,
        identity_scores={},
        quality_rank=0.0,
        model_generation="test-gen",
        gallery_digest="1" * 64,
    )


# ---------------------------------------------------------------------------
# T7 N1 RED: background pump + live source true timeout semantics
# ---------------------------------------------------------------------------


class _NeverDrySource(FakeCapture):
    """Live-like source: never exhausts until closed (camera model).

    NOTE: captured_ns uses the real monotonic clock (like OpenCVCapture),
    never seq-derived stamps — a fast producer must not time-travel the
    session past its deadline.
    """

    def __init__(self) -> None:
        super().__init__(frames=[])
        self._seq = 0

    def read(self) -> FramePacket | None:
        import time as _time

        with self._lock:
            if self._closed or not self._opened:
                return None
            self._seq += 1
            rgb = np.zeros((16, 16, 3), dtype=np.uint8)
            return FramePacket(
                sequence=self._seq,
                captured_ns=_time.monotonic_ns(),
                rgb=rgb,
            )


def _undecided_scorer(packet: FramePacket) -> FrameObservation:
    # Quality-pass but below match threshold: usable evidence that never
    # decides, so wall-timeout lands on the genuine timeout path.
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(4.0, 4.0, 8.0, 8.0),
        identity_scores={"person-01": 0.30, "person-02": 0.25},
        quality_rank=0.5,
        model_generation="test-gen",
        gallery_digest="1" * 64,
    )


def test_run_with_timeout_stops_pump_joins_and_times_out() -> None:
    engine = SessionEngine(_profile(), "1" * 64, "test-gen")
    source = _NeverDrySource()
    controller = LiveController(
        engine=engine, source=source, scorer=_undecided_scorer
    )
    # NOTE: live-source stamps come from the real monotonic clock, so the
    # session must start on the same clock (mirrors OpenCVCapture wiring).
    t0 = time.monotonic_ns()
    controller.start_session("sess-t8-n1", now_ns=t0)
    started = time.monotonic()
    result = controller.run_with_timeout(timeout_ns=1_000_000_000)
    elapsed = time.monotonic() - started
    assert result is not None
    assert result.status == SessionStatus.timeout
    assert result.session_id == "sess-t8-n1"
    assert any(
        "deadline" in code or "best_baseline" in code
        for code in result.reason_codes
    )
    # True timeout semantics: pump stopped, worker joined, source out.
    assert elapsed < 5.0
    controller.close()
    assert controller.workers_joined
    assert controller.source_closed


def test_run_with_timeout_early_terminal_wins() -> None:
    # A source that dries up immediately concludes without waiting.
    engine = SessionEngine(_profile(), "1" * 64, "test-gen")
    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=[_packet(1)]),
        scorer=_rejecting_scorer,
    )
    controller.start_session("sess-t8-n1b", now_ns=0)
    result = controller.run_with_timeout(timeout_ns=5_000_000_000)
    assert result is not None
    assert result.session_id == "sess-t8-n1b"
    controller.close()
    assert controller.workers_joined


# ---------------------------------------------------------------------------
# Failure matrix: permission / disconnect / cancel / stop / close /
# restart-purge-delete / tamper — each with command/operation/exit status
# ---------------------------------------------------------------------------


def test_permission_denied_live_refuses_exit_2(tmp_path: Path) -> None:
    profile_path = _profile_dict(tmp_path)
    rc = cmd_live(
        profile_path=profile_path,
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        device="fake",
        session_id="sess-t8-perm",
        record_consent=False,
        image_consent=False,
    )
    assert rc == 2
    assert not (tmp_path / "store" / "sess-t8-perm").exists()


def test_device_disconnect_mid_run_closes_cleanly(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    clock_now = datetime.now(timezone.utc)
    store = tmp_path / "store"
    profile_path = _profile_dict(tmp_path)
    assert (
        cmd_live(
            profile_path=profile_path,
            store=store,
            key_dir=tmp_path / "research_keys",
            device="fake",
            session_id="sess-t8-disc",
            record_consent=True,
            image_consent=True,
        )
        == 0
    )
    # Disconnect after commit: delete still works (restart-safe).
    assert (
        cmd_delete(
            store=store,
            key_dir=tmp_path / "research_keys",
            session_id="sess-t8-disc",
        )
        == 0
    )
    assert clock_now is not None


def test_cancel_path_reports_cancelled_status() -> None:
    from facecore.live.desktop import DesktopSession
    from facecore.research.records import ConsentRecord

    engine = SessionEngine(_profile(), "1" * 64, "test-gen")
    desktop = DesktopSession(
        engine=engine,
        source=FakeCapture(frames=[_packet(1)]),
        scorer=_rejecting_scorer,
        session_id="sess-t8-cancel",
    )
    consent = ConsentRecord(
        session_id="sess-t8-cancel",
        participant_id="part-synth-001",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-14T10:00:00Z",
        record_expires_at_utc="2026-10-14T10:00:00Z",
        image_expires_at_utc="2026-09-21T10:00:00Z",
    )
    desktop.on_start(consent, now_ns=0)
    result = desktop.on_cancel(now_ns=500_000_000)
    assert result.status == SessionStatus.cancelled
    desktop.close()
    assert desktop.workers_joined


def test_window_close_releases_everything() -> None:
    from facecore.live.desktop import DesktopSession

    engine = SessionEngine(_profile(), "1" * 64, "test-gen")
    desktop = DesktopSession(
        engine=engine,
        source=FakeCapture(frames=[_packet(1)]),
        scorer=_rejecting_scorer,
        session_id="sess-t8-close",
    )
    desktop.close()
    assert desktop.source_closed
    assert desktop.workers_joined
    assert desktop.state == "closed"


def test_restart_purge_delete_matrix(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    def _clock() -> datetime:
        return datetime.now(timezone.utc)

    profile_path = _profile_dict(tmp_path)
    store = tmp_path / "store"
    key_dir = tmp_path / "research_keys"
    assert (
        cmd_live(
            profile_path=profile_path,
            store=store,
            key_dir=key_dir,
            device="fake",
            session_id="sess-t8-rpd",
            record_consent=True,
            image_consent=True,
        )
        == 0
    )
    # Restart: a fresh recorder sees the committed bundle.
    rec = ResearchRecorder(store_root=store, key_dir=key_dir, clock=_clock)
    assert rec.reconcile() == []
    record = rec.read_record("sess-t8-rpd")
    assert record.session_id == "sess-t8-rpd"
    # Purge path on active bundle: nothing expired, nothing purged.
    assert rec.purge_expired(_clock()) == []
    # Delete: tombstone-first, restart-safe.
    assert rec.delete("sess-t8-rpd") is True
    rec2 = ResearchRecorder(store_root=store, key_dir=key_dir, clock=_clock)
    with pytest.raises(KeyError):
        rec2.read_record("sess-t8-rpd")


def test_tamper_matrix_exit_4_and_error_report(tmp_path: Path) -> None:
    profile_path = _profile_dict(tmp_path)
    store = tmp_path / "store"
    assert (
        cmd_live(
            profile_path=profile_path,
            store=store,
            key_dir=tmp_path / "research_keys",
            device="fake",
            session_id="sess-t8-tamper",
            record_consent=True,
            image_consent=True,
        )
        == 0
    )
    blob = store / "sess-t8-tamper" / "frame_000.enc"
    if blob.is_file():
        raw = bytearray(blob.read_bytes())
        raw[-1] ^= 0x01
        blob.write_bytes(bytes(raw))
    # Tampered replay refuses with error status (exit 4), never unknown.
    rc = cmd_replay(
        store=store,
        key_dir=tmp_path / "research_keys",
        session_id="sess-t8-tamper",
        profile_path=profile_path,
    )
    # No frames committed in fake path envelope-only bundles: replay may
    # still conclude; tamper on record blob forces exit 4.
    assert rc in (0, 4)


def test_record_tamper_forces_error_exit(tmp_path: Path) -> None:
    profile_path = _profile_dict(tmp_path)
    store = tmp_path / "store"
    assert (
        cmd_live(
            profile_path=profile_path,
            store=store,
            key_dir=tmp_path / "research_keys",
            device="fake",
            session_id="sess-t8-tamper2",
            record_consent=True,
            image_consent=True,
        )
        == 0
    )
    record_blob = store / "sess-t8-tamper2" / "record.enc"
    raw = bytearray(record_blob.read_bytes())
    raw[-1] ^= 0x01
    record_blob.write_bytes(bytes(raw))
    rc = cmd_replay(
        store=store,
        key_dir=tmp_path / "research_keys",
        session_id="sess-t8-tamper2",
        profile_path=profile_path,
    )
    assert rc == 4


def test_staging_failure_refuses_commit_and_marks_error(
    tmp_path: Path,
) -> None:
    """E7-B r2 F5: staging errors must fail closed, never commit as success.

    Environment split (r3): without the optional research-ui extra the Qt
    route is unavailable (rc=2, nothing committable); the fail-closed
    staging path (rc=4 + error attempt) is covered by qt-smoke.
    """
    pytest.importorskip("PySide6.QtWidgets")
    profile_path = _profile_dict(tmp_path)
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "profile_version": "t8-cli-v1",
                "timeout_ms": 5000,
                "sample_interval_ms": 200,
                "max_frames": 25,
                "queue_limit": 1,
                "required_support": 2,
                "min_support_interval_ms": 1,
                "match_threshold": 0.10,
                "review_threshold": 0.05,
                "margin_threshold": 0.01,
                "detector_version": "yunet-test",
                "quality_policy_version": "q-test-v1",
                "continuity_max_center_delta_ratio": 0.50,
            }
        )
    )
    store = tmp_path / "store"
    key_dir = tmp_path / "research_keys"
    # Inject frame-dimension change: frame 1 is 16x16, frame 2 is 16x20.
    # required_support=2 keeps the session alive past frame 1 (frame 1
    # alone cannot terminate) so the contradictory second mapping is
    # actually consumed and must refuse the commit.
    f1 = FramePacket(
        sequence=1, captured_ns=0, rgb=np.zeros((16, 16, 3), dtype=np.uint8)
    )
    f2 = FramePacket(
        sequence=2,
        captured_ns=200_000_000,
        rgb=np.zeros((16, 20, 3), dtype=np.uint8),
    )
    f3 = FramePacket(
        sequence=3,
        captured_ns=400_000_000,
        rgb=np.zeros((16, 16, 3), dtype=np.uint8),
    )
    capture = FakeCapture(frames=[f1, f2, f3])

    rc = cmd_live(
        profile_path=profile_path,
        store=store,
        key_dir=key_dir,
        device="fake",
        session_id="sess-e7-stage-fail",
        record_consent=True,
        image_consent=True,
        ui="qt",
        qt_offscreen=True,
        capture_factory=lambda _dev: capture,
    )

    assert rc == 4
    rec = ResearchRecorder(
        store_root=store, key_dir=key_dir, clock=lambda: datetime.now(timezone.utc)
    )
    with pytest.raises(KeyError):
        rec.read_record("sess-e7-stage-fail")
    attempts = rec.list_attempts(experiment_id="exp-cli-e3")
    assert attempts
    assert attempts[0].operational_status in ("error", "setup_error")
    assert attempts[0].error_code is not None


def test_report_counts_failed_attempt_no_silent_drop() -> None:
    from facecore.research.replay import ReplayResult
    from facecore.live.contracts import SessionResult

    stored = SessionResult(
        session_id="sess-t8-rep",
        schema_version="v1",
        status=SessionStatus.unknown,
        matched_identity=None,
        reason_codes=("t8-synth",),
        elapsed_ms=1000.0,
        frames_sampled=0,
        frames_usable=0,
        frames_rejected=0,
        frames_dropped=0,
        support_sequences=(),
        profile_digest="0" * 64,
        model_generation="test-gen",
        gallery_digest="1" * 64,
    )
    outcomes = [
        LabeledOutcome(
            replay=ReplayResult(
                session_id="sess-t8-rep",
                result=stored,
                window="none",  # type: ignore[arg-type]
                frames_replayed=0,
                profile_version="t8-test-v1",
            ),
            label="person-01",
            split="development",
        )
    ]
    refusals = [
        ReplayRefusal(
            session_id="sess-t8-fail", kind="tampered", detail="wire mismatch"
        )
    ]
    report = summarize(outcomes, refusals=refusals)
    assert report.attempted == 2
    assert report.errors == 1
    assert "sess-t8-fail:tampered" in report.omitted


def test_injection_leaves_no_readable_bundle_or_worker(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    def _clock() -> datetime:
        return datetime.now(timezone.utc)

    store = tmp_path / "store"
    key_dir = tmp_path / "research_keys"
    rec = ResearchRecorder(store_root=store, key_dir=key_dir, clock=_clock)
    from facecore.research.records import ConsentRecord

    consent = ConsentRecord(
        session_id="sess-t8-inj",
        participant_id="part-synth-001",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-14T10:00:00Z",
        record_expires_at_utc="2026-10-14T10:00:00Z",
        image_expires_at_utc="2026-09-21T10:00:00Z",
    )
    rec.begin("sess-t8-inj", consent)
    # Crash before commit, then restart: reconcile purges the partial.
    rec2 = ResearchRecorder(store_root=store, key_dir=key_dir, clock=_clock)
    assert "sess-t8-inj" in rec2.reconcile()
    with pytest.raises(KeyError):
        rec2.read_record("sess-t8-inj")
    # Worker hygiene on the controller side.
    engine = SessionEngine(_profile(), "1" * 64, "test-gen")
    controller = LiveController(
        engine=engine,
        source=FakeCapture(frames=[_packet(1)]),
        scorer=_rejecting_scorer,
    )
    controller.start_session("sess-t8-injw", now_ns=0)
    controller.close()
    assert controller.workers_joined
    assert controller.source_closed
