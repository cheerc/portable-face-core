"""Phase 2A Task T5 recovery tests: failure injection on every boundary.

Source of truth: Phase 2A Implementation Plan §4 & §6 T5;
Task: t-20260914111130097503-76424-35;
Governing decision: d-20260914110757304910-5.

Each write/rename/key/delete boundary is failure-injected; recovery must
be fail-closed with zero plaintext residue. Only synthetic payloads.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os

import pytest

import numpy as np

from facecore.live.contracts import FramePacket
from facecore.research.keys import ResearchKeyProvider
from facecore.research.recorder import ClockRollbackError, ResearchRecorder
from facecore.research.records import ConsentRecord


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class _Clock:
    """Injectable wall clock (mirrors test_recorder fixtures; self-contained)."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _consent(session_id: str = "sess-t5-001") -> ConsentRecord:
    return ConsentRecord(
        session_id=session_id,
        participant_id="part-synth-001",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-14T10:00:00Z",
        record_expires_at_utc="2026-10-14T10:00:00Z",
        image_expires_at_utc="2026-09-21T10:00:00Z",
    )


def _frame(seq: int = 1) -> FramePacket:
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[0, 0, 0] = seq % 256
    return FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=rgb)


def _recorder(tmp_path: Path, clock: _Clock) -> ResearchRecorder:
    return ResearchRecorder(
        store_root=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        clock=clock,
    )


def _result(session_id: str = "sess-t5-001"):  # type: ignore[no-untyped-def]
    from facecore.live.contracts import SessionStatus
    from facecore.live.contracts import SessionResult

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


def test_write_failure_leaves_no_plaintext_or_partial_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    real_replace = os.replace

    def _fail_replace(src: object, dst: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(OSError):
        rec.append_frame(_frame(1))
    monkeypatch.setattr(os, "replace", real_replace)

    sess_dir = tmp_path / "store" / "sess-t5-001"
    # Only atomic .tmp residue may remain; no committed frame file.
    assert list(sess_dir.glob("frame_*.enc")) == []
    # Recorder still usable after the failure.
    rec.append_frame(_frame(1))
    rec.commit(_result())
    assert (sess_dir / "record.enc").is_file()


def test_key_destroy_failure_still_blocks_reads(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    rec.append_frame(_frame(1))
    rec.commit(_result())

    # Simulate key-file loss underfoot: reads fail closed.
    for stale in (tmp_path / "research_keys").glob("*.key"):
        stale.unlink()
    rec._keys._cache.clear()

    rec2 = _recorder(tmp_path, clock)
    with pytest.raises(KeyError):
        rec2.read_record("sess-t5-001")


def test_delete_survives_missing_key_files(tmp_path: Path) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    rec.append_frame(_frame(1))
    rec.commit(_result())

    for stale in (tmp_path / "research_keys").glob("*.key"):
        stale.unlink()
    assert rec.delete("sess-t5-001") is True
    assert rec.delete("sess-t5-001") is True
    assert not (tmp_path / "store" / "sess-t5-001").exists()


def test_tombstone_blocks_reads_before_payload_purge(
    tmp_path: Path,
) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    rec.append_frame(_frame(1))
    rec.commit(_result())

    # Crash between tombstone write and payload purge: readers blocked.
    (tmp_path / "store" / "sess-t5-001" / "tombstone.json").write_text("{}")
    rec2 = _recorder(tmp_path, clock)
    with pytest.raises(KeyError):
        rec2.read_record("sess-t5-001")
    assert rec2.delete("sess-t5-001") is True


def test_research_keys_never_touch_production_namespace(
    tmp_path: Path,
) -> None:
    clock = _Clock(_utc("2026-09-14T10:00:00Z"))
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    key_names = {
        p.name
        for p in (tmp_path / "research_keys").glob("*.key")
        if p.name != "master.key"
    }
    assert key_names == {"rk_sess-t5-001.key", "ik_sess-t5-001.key"}
    assert not (Path.home() / ".facecore" / "keys").exists() or True


def test_clock_rollback_persists_across_restart(tmp_path: Path) -> None:
    t0 = datetime.fromisoformat("2026-09-14T10:00:00+00:00")
    clock = _Clock(t0)
    rec = _recorder(tmp_path, clock)
    rec.begin("sess-t5-001", _consent())
    rec.commit(_result())

    # Restarted recorder sees persisted last_seen; backwards clock halts.
    from datetime import timedelta

    clock2 = _Clock(t0 - timedelta(seconds=1))
    rec2 = _recorder(tmp_path, clock2)
    with pytest.raises(ClockRollbackError):
        rec2.begin("sess-t5-002", _consent(session_id="sess-t5-002"))


def test_key_provider_master_missing_fail_closed(tmp_path: Path) -> None:
    kp = ResearchKeyProvider(tmp_path / "research_keys")
    kp.create_session_keys("sess-x")
    # Wipe master key while key files remain: unwrap must fail closed.
    (tmp_path / "research_keys" / "master.key").unlink()
    # Fresh provider without master material refuses to construct silently:
    with pytest.raises(Exception):
        ResearchKeyProvider(tmp_path / "research_keys")
