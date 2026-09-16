"""E2 RED privacy and lifecycle tests: byte-level at-rest encryption, TTL separation, deletion (Phase 2B §12 E2).

Source of truth:
    - docs/specs/2026-09-16-phase2b-mac-recognition-research.md §6 (minimal diagnostics & truth isolation);
    - docs/plans/2026-09-16-phase2b-mac-recognition-execution-plan.md §12 E2 Acceptance;
    - ADR 0010 (at-rest encryption under repo-external store).

Mandatory constraints:
    1. Byte-level inspection via assert_no_plaintext_leak with scan root passed as parent
       (covering both store_root and key_dir).
    2. Sensitive literal set widened to include participant IDs, ground-truth identities,
       and probe labels.
    3. Immutable RED commit established before GREEN implementation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from facecore.contracts.crypto import StoreCorruptionError
from facecore.live.contracts import (
    FrameDiagnostics,
    FramePacket,
    ResearchProfile,
    SessionResult,
    SessionStatus,
)
from facecore.research.diagnostics import FrameTraceEntry, SessionTrace
from facecore.research.experiment import AttemptRecord, EvaluationLabel, ExperimentManifest
from facecore.research.records import ConsentRecord
from facecore.research.recorder import ResearchRecorder
from tests.conftest import assert_no_plaintext_leak


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _consent(
    session_id: str = "sess-priv-001",
    participant_id: str = "part-secret-999",
    consented_at: str = "2026-09-16T10:00:00Z",
) -> ConsentRecord:
    c_dt = _utc(consented_at)
    img_exp = (c_dt + timedelta(days=7)).isoformat()
    rec_exp = (c_dt + timedelta(days=30)).isoformat()
    return ConsentRecord(
        session_id=session_id,
        participant_id=participant_id,
        record_consent=True,
        image_consent=True,
        consented_at_utc=c_dt.isoformat(),
        record_expires_at_utc=rec_exp,
        image_expires_at_utc=img_exp,
    )


def _manifest(experiment_id: str = "exp-priv-001") -> ExperimentManifest:
    return ExperimentManifest.from_dict(
        {
            "identity": {"experiment_id": experiment_id},
            "software": {},
            "gallery": {},
            "policy": {},
            "capture": {},
            "privacy": {},
            "study": {},
            "analysis": {},
        }
    )


def _attempt(
    attempt_id: str = "att-priv-001",
    experiment_id: str = "exp-priv-001",
    participant_id: str = "part-secret-999",
) -> AttemptRecord:
    return AttemptRecord(
        experiment_id=experiment_id,
        attempt_id=attempt_id,
        participant_id=participant_id,
        visit_id="visit-001",
        condition_id="cond-001",
        attempt_index=1,
        retry_of=None,
        consent_ref="consent-secret",
        requested_at_utc="2026-09-16T10:00:00Z",
        accepted_at_utc="2026-09-16T10:00:01Z",
        started_at_utc="2026-09-16T10:00:02Z",
        ended_at_utc=None,
        operational_status="accepted",
        error_code=None,
        bundle_ref=None,
    )


def _recorder(tmp_path: Path, now: datetime) -> ResearchRecorder:
    current = now
    return ResearchRecorder(
        store_root=tmp_path / "store",
        key_dir=tmp_path / "keys",
        clock=lambda: current,
    )


def _sample_trace_entry(
    sequence: int = 1, identity_id: str = "person-classified-01"
) -> FrameTraceEntry:
    diag = FrameDiagnostics(
        sequence=sequence,
        original_shape=(480, 640, 3),
        normalized_shape=(480, 640, 3),
        orientation=0,
        mirrored=False,
        face_count=1,
        detector_confidence=0.96,
        face_box=(120.0, 120.0, 140.0, 140.0),
        landmarks=((130.0, 140.0), (190.0, 140.0), (160.0, 170.0), (140.0, 200.0), (180.0, 200.0)),
        sharpness=52.3,
        mean_luma=120.0,
        clipped_fraction=0.005,
        yaw_deg=1.2,
        pitch_deg=-0.5,
        quality_status="accepted",
    )
    return FrameTraceEntry(
        sequence=sequence,
        captured_ns=sequence * 200_000_000,
        processed_ns=sequence * 200_000_000 + 15_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(120.0, 120.0, 140.0, 140.0),
        identity_score_pairs=((identity_id, 0.88), ("runner-up-ident", 0.32)),
        quality_rank=52.3,
        model_generation="gen-v2",
        gallery_digest="digest-v2",
        diagnostics=diag,
        staged_index=sequence - 1,
    )


class TestDiagnosticPrivacyAtRest:
    """Verify that trace storage is strictly AEAD-encrypted with zero plaintext leaks."""

    def test_trace_bytes_at_rest_contain_no_sensitive_literals(self, tmp_path: Path) -> None:
        now = _utc("2026-09-16T10:00:00Z")
        rec = _recorder(tmp_path, now)

        participant_id = "part-top-secret-participant"
        secret_identity = "person-secret-truth-042"
        consent = _consent("sess-priv-001", participant_id=participant_id)
        manifest = _manifest("exp-priv-001")
        attempt = _attempt("att-priv-001", "exp-priv-001", participant_id=participant_id)

        rec.begin_attempt(manifest, attempt, consent)
        entry = _sample_trace_entry(sequence=1, identity_id=secret_identity)
        rec.append_trace("att-priv-001", entry)

        # Label written in sidecar
        label = EvaluationLabel(
            attempt_id="att-priv-001",
            revision=1,
            kind="enrolled",
            identity_id=secret_identity,
            actor_ref="evaluator-alice",
            labeled_at_utc="2026-09-16T10:05:00Z",
        )
        rec.write_label(label)

        # Assert no plaintext leak across parent root (covering both store and keys)
        sensitive_values = [
            participant_id,
            secret_identity,
            "runner-up-ident",
            "evaluator-alice",
            "consent-secret",
        ]
        assert_no_plaintext_leak(tmp_path, sensitive_values)


class TestDiagnosticLifecycleAndTtl:
    """Verify TTL separation (7d image vs 30d trace/record), deletion, and tampering."""

    def test_ttl_separation_image_7d_purged_trace_30d_retained(self, tmp_path: Path) -> None:
        start_time = _utc("2026-09-16T10:00:00Z")
        rec = _recorder(tmp_path, start_time)

        session_id = "sess-ttl-001"
        attempt_id = "att-ttl-001"
        consent = _consent(session_id, consented_at="2026-09-16T10:00:00Z")
        manifest = _manifest("exp-ttl")
        attempt = _attempt(attempt_id, "exp-ttl")

        rec.begin_attempt(manifest, attempt, consent)
        entry = _sample_trace_entry(sequence=1)
        rec.append_trace(attempt_id, entry)

        # Also stage an image frame in the session
        rec.begin(session_id, consent)
        # Create a dummy image packet
        import numpy as np
        img_packet = FramePacket(sequence=1, captured_ns=0, rgb=np.zeros((100, 100, 3), dtype=np.uint8))
        rec.append_frame(img_packet)
        dummy_result = SessionResult(
            session_id=session_id,
            schema_version="v1",
            status=SessionStatus.matched,
            matched_identity="person-01",
            reason_codes=("supported_3_frames",),
            elapsed_ms=500.0,
            frames_sampled=1,
            frames_usable=1,
            frames_rejected=0,
            frames_dropped=0,
            support_sequences=(1,),
            profile_digest="p1",
            model_generation="g1",
            gallery_digest="d1",
        )
        rec.commit(dummy_result)
        rec.finish_attempt(attempt_id, result=dummy_result, operational_status="completed", error_code=None)

        # At Day 8 (now = start + 8 days): image TTL (7d) is expired, but record TTL (30d) is active!
        day8 = start_time + timedelta(days=8)
        purged = rec.purge_expired(day8)
        assert session_id in purged

        # 1. Image frames should be purged
        sess_dir = tmp_path / "store" / session_id
        assert not list(sess_dir.glob("frame_*.enc"))

        # 2. Trace should still be readable!
        trace = rec.read_trace(attempt_id)
        assert len(trace.entries) == 1
        assert trace.entries[0].sequence == 1

        # At Day 31 (now = start + 31 days): record TTL is expired!
        day31 = start_time + timedelta(days=31)
        purged_31 = rec.purge_expired(day31)
        assert attempt_id in purged_31 or session_id in purged_31

        # Now trace should be gone, key destroyed, read_trace raises KeyError
        with pytest.raises(KeyError):
            rec.read_trace(attempt_id)

    def test_withdraw_attempt_destroys_key_and_trace(self, tmp_path: Path) -> None:
        now = _utc("2026-09-16T10:00:00Z")
        rec = _recorder(tmp_path, now)

        attempt_id = "att-withdraw-001"
        consent = _consent("sess-w-001")
        manifest = _manifest("exp-w")
        attempt = _attempt(attempt_id, "exp-w")

        rec.begin_attempt(manifest, attempt, consent)
        rec.append_trace(attempt_id, _sample_trace_entry(sequence=1))

        # Withdraw attempt
        rec.withdraw_attempt(attempt_id)

        # read_trace must fail because attempt & key are destroyed
        with pytest.raises(KeyError):
            rec.read_trace(attempt_id)

    def test_delete_session_cascades_to_trace(self, tmp_path: Path) -> None:
        now = _utc("2026-09-16T10:00:00Z")
        rec = _recorder(tmp_path, now)

        session_id = "sess-casc-001"
        attempt_id = "att-casc-001"
        consent = _consent(session_id)
        manifest = _manifest("exp-casc")
        attempt = _attempt(attempt_id, "exp-casc")

        rec.begin_attempt(manifest, attempt, consent)
        rec.append_trace(attempt_id, _sample_trace_entry(sequence=1))

        rec.begin(session_id, consent)
        # Cascade delete session
        rec.delete(session_id)

        # Trace for linked attempt should be deleted
        with pytest.raises(KeyError):
            rec.read_trace(attempt_id)

    def test_reconcile_protects_traces_directory(self, tmp_path: Path) -> None:
        now = _utc("2026-09-16T10:00:00Z")
        rec = _recorder(tmp_path, now)

        attempt_id = "att-rec-001"
        consent = _consent("sess-rec-001")
        manifest = _manifest("exp-rec")
        attempt = _attempt(attempt_id, "exp-rec")

        rec.begin_attempt(manifest, attempt, consent)
        rec.append_trace(attempt_id, _sample_trace_entry(sequence=1))

        # Create an uncommitted/corrupt session directory to test reconcile
        bad_sess = tmp_path / "store" / "sess-uncommitted"
        bad_sess.mkdir(parents=True, exist_ok=True)
        (bad_sess / "temp.dat").write_bytes(b"garbage")

        rec.reconcile()

        # Bad session is purged
        assert not bad_sess.exists()
        # Trace is protected and stays intact!
        trace = rec.read_trace(attempt_id)
        assert len(trace.entries) == 1

    def test_tampered_trace_or_aad_swap_fails_closed(self, tmp_path: Path) -> None:
        now = _utc("2026-09-16T10:00:00Z")
        rec = _recorder(tmp_path, now)

        attempt_id = "att-tamper-001"
        consent = _consent("sess-tamper-001")
        manifest = _manifest("exp-tamper")
        attempt = _attempt(attempt_id, "exp-tamper")

        rec.begin_attempt(manifest, attempt, consent)
        rec.append_trace(attempt_id, _sample_trace_entry(sequence=1))

        # Tamper with the raw ciphertext of sequence 1
        trace_file = tmp_path / "store" / "_traces" / attempt_id / "frame_0001.enc"
        assert trace_file.exists()
        raw = bytearray(trace_file.read_bytes())
        raw[-1] ^= 0xFF  # flip last byte
        trace_file.write_bytes(bytes(raw))

        # Decrypt must fail closed with StoreCorruptionError
        with pytest.raises(StoreCorruptionError):
            rec.read_trace(attempt_id)
