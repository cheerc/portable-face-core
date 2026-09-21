"""Phase 2A Task T5 tests: consent-gated encrypted recorder with TTL and deletion.

Source of truth: Phase 2A Implementation Plan §4 & §6 T5;
Task: t-20260914111130097503-76424-35;
Governing decision: d-20260914110757304910-5.

RED contract (must fail before implementation exists):
- 無同意落盤：append without image consent must refuse, zero frame files.
- 7 天到期仍可解密：image decrypt after 7d TTL expiry must fail closed.
- 刪除後重啟可讀：rebuilt recorder after delete must refuse reads.
- tamper 被接受：bit-flipped frame must be rejected on replay.
Only synthetic payloads; never real faces.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from facecore.live.contracts import (
    FramePacket,
    ResearchProfile,
    SessionResult,
    SessionStatus,
)
from facecore.research.keys import ResearchKeyProvider
from facecore.research.recorder import ClockRollbackError, ResearchRecorder
from facecore.research.records import ConsentRecord


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _consent(
    session_id: str = "sess-t5-001",
    *,
    record: bool = True,
    image: bool = True,
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


def _profile() -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="t5-test-v1",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=26,
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


def _frame(seq: int = 1) -> FramePacket:
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[0, 0, 0] = seq % 256
    return FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=rgb)


def _result(session_id: str = "sess-t5-001") -> SessionResult:
    return SessionResult(
        session_id=session_id,
        schema_version="v1",
        status=SessionStatus.unknown,
        matched_identity=None,
        reason_codes=("t5-synth",),
        elapsed_ms=1000.0,
        frames_sampled=5,
        frames_usable=5,
        frames_rejected=0,
        frames_dropped=0,
        support_sequences=(),
        profile_digest="0" * 64,
        model_generation="test-gen",
        gallery_digest="1" * 64,
    )


class _Clock:
    """Injectable wall clock (S2N3: virtual clock + real filesystem)."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _recorder(tmp_path: Path, clock: _Clock) -> ResearchRecorder:
    return ResearchRecorder(
        store_root=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        clock=clock,
    )


# ---------------------------------------------------------------------------
# RED 1: 無同意落盤
# ---------------------------------------------------------------------------


def test_append_without_image_consent_refuses_and_writes_nothing(
    tmp_path: Path,
) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent(image=False))
    with pytest.raises(PermissionError):
        rec.append_frame(_frame(1))
    sess_dir = tmp_path / "store" / "sess-t5-001"
    assert list(sess_dir.glob("frame_*.enc")) == []


def test_begin_without_record_consent_refuses(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    with pytest.raises(PermissionError):
        rec.begin("sess-t5-001", _consent(record=False))


# ---------------------------------------------------------------------------
# RED 2: 7 天到期影像仍可解密
# ---------------------------------------------------------------------------


def test_image_undecryptable_after_7d_expiry_record_still_readable(
    tmp_path: Path,
) -> None:
    t0 = _utc("2026-09-14T10:00:00Z")
    clock = _Clock(t0)
    rec = _recorder(tmp_path, clock)
    consent = _consent()
    rec.begin("sess-t5-001", consent)
    rec.append_frame(_frame(1))
    rec.commit(_result())

    # Day 8: image TTL (7d) expired, record TTL (30d) still active.
    clock.now = t0 + timedelta(days=8)
    rec2 = _recorder(tmp_path, clock)
    with pytest.raises(KeyError):
        rec2.read_frame("sess-t5-001", 0)
    # Record envelope must remain readable.
    record = rec2.read_record("sess-t5-001")
    assert record.session_id == "sess-t5-001"


def test_record_unreadable_after_30d_expiry(tmp_path: Path) -> None:
    t0 = _utc("2026-09-14T10:00:00Z")
    clock = _Clock(t0)
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    rec.append_frame(_frame(1))
    rec.commit(_result())

    clock.now = t0 + timedelta(days=31)
    rec2 = _recorder(tmp_path, clock)
    with pytest.raises(KeyError):
        rec2.read_record("sess-t5-001")


# ---------------------------------------------------------------------------
# RED 3: 刪除後重啟可讀
# ---------------------------------------------------------------------------


def test_delete_is_tombstone_first_reentrant_and_restart_safe(
    tmp_path: Path,
) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    rec.append_frame(_frame(1))
    rec.commit(_result())

    assert rec.delete("sess-t5-001") is True
    # Re-entrant: second delete is idempotent success.
    assert rec.delete("sess-t5-001") is True

    # Rebuilt recorder after restart must refuse reads.
    rec2 = _recorder(tmp_path, clock)
    with pytest.raises(KeyError):
        rec2.read_record("sess-t5-001")
    with pytest.raises(KeyError):
        rec2.read_frame("sess-t5-001", 0)


# ---------------------------------------------------------------------------
# RED 4: tamper 被接受
# ---------------------------------------------------------------------------


def test_tampered_frame_rejected_on_replay(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    rec.append_frame(_frame(1))
    rec.commit(_result())

    blob = tmp_path / "store" / "sess-t5-001" / "frame_000.enc"
    raw = bytearray(blob.read_bytes())
    raw[-1] ^= 0x01
    blob.write_bytes(bytes(raw))

    rec2 = _recorder(tmp_path, clock)
    with pytest.raises(ValueError):
        rec2.read_frame("sess-t5-001", 0)


def test_tampered_manifest_rejected(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    rec.commit(_result())

    manifest = tmp_path / "store" / "sess-t5-001" / "manifest.json"
    manifest.write_text(manifest.read_text().replace("committed", "tampered"))

    rec2 = _recorder(tmp_path, clock)
    with pytest.raises(ValueError):
        rec2.read_record("sess-t5-001")


# ---------------------------------------------------------------------------
# Consent lifecycle / bounds
# ---------------------------------------------------------------------------


def test_revoke_clears_uncommitted_images(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    rec.append_frame(_frame(1))
    rec.append_frame(_frame(2))
    rec.revoke_image_consent("sess-t5-001")
    sess_dir = tmp_path / "store" / "sess-t5-001"
    assert list(sess_dir.glob("frame_*.enc")) == []
    with pytest.raises(PermissionError):
        rec.append_frame(_frame(3))


def test_abort_on_multiface_clears_uncommitted_images(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    rec.append_frame(_frame(1))
    rec.abort("sess-t5-001", reason="input_multiple_faces")
    sess_dir = tmp_path / "store" / "sess-t5-001"
    assert list(sess_dir.glob("frame_*.enc")) == []
    assert not (sess_dir / "manifest.json").exists()


def test_image_cap_26_frames(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    for seq in range(1, 27):
        rec.append_frame(_frame(seq))
    with pytest.raises(ValueError):
        rec.append_frame(_frame(27))


def test_uncommitted_bundle_hidden_until_commit(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    rec.append_frame(_frame(1))
    rec2 = _recorder(tmp_path, clock)
    with pytest.raises(KeyError):
        rec2.read_record("sess-t5-001")


def test_clock_rollback_flags_error_and_halts_new_writes(
    tmp_path: Path,
) -> None:
    t0 = _utc("2026-09-14T10:00:00Z")
    clock = _Clock(t0)
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    rec.append_frame(_frame(1))

    # Wall clock jumps backwards: flag error, halt new writes.
    clock.now = t0 - timedelta(hours=1)
    with pytest.raises(ClockRollbackError):
        rec.append_frame(_frame(2))
    with pytest.raises(ClockRollbackError):
        rec.begin("sess-t5-002", _consent(session_id="sess-t5-002"))


def test_startup_reconcile_purges_partial_bundle(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    rec.append_frame(_frame(1))
    # Crash before commit: only manifest.json.tmp exists.
    rec2 = _recorder(tmp_path, clock)
    purged = rec2.reconcile()
    assert "sess-t5-001" in purged
    assert not (tmp_path / "store" / "sess-t5-001").exists()


def test_key_namespace_isolated_per_session(tmp_path: Path) -> None:
    kp = ResearchKeyProvider(tmp_path / "research_keys")
    rk_a, ik_a = kp.create_session_keys("sess-a")
    rk_b, ik_b = kp.create_session_keys("sess-b")
    assert len({rk_a, ik_a, rk_b, ik_b}) == 4
    assert kp.get_key(rk_a) != kp.get_key(ik_a)


def test_purge_expired_on_write_path(tmp_path: Path) -> None:
    t0 = _utc("2026-09-14T10:00:00Z")
    clock = _Clock(t0)
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent(session_id="sess-t5-001"))
    rec.append_frame(_frame(1))
    rec.commit(_result("sess-t5-001"))

    # Day 8: starting a new session triggers purge of expired images.
    clock.now = t0 + timedelta(days=8)
    rec.begin("sess-t5-002", _consent(session_id="sess-t5-002"))
    assert not (tmp_path / "store" / "sess-t5-001" / "frame_000.enc").exists()
    assert (tmp_path / "store" / "sess-t5-001" / "record.enc").exists()
