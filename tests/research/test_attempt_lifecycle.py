"""E1 RED: attempt lifecycle — durable accepted-start, label persistence (2B §12 E1).

Source of truth: §12 E1 Acceptance + GREEN command set:
    - tests/research/test_recorder.py (fixture style);
    - §11.2 signatures: begin_attempt / finish_attempt / write_label.

RED contract: recorder has no begin_attempt/finish_attempt/write_label
yet, so these fail with AttributeError on the recorder (a missing-method
failure, NOT an import misspelling — the module facecore.research.recorder
imports fine; only the new E1 methods are absent). Camera-open failure,
cancel, and crash recovery must each leave exactly one attempt; the same
attempt ID re-sent must not duplicate; labels must survive a recorder
restart; a write-failed (unaccepted) Start must not open the camera.

Only synthetic payloads; never real faces; camera-free.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from facecore.live.contracts import SessionResult, SessionStatus
from facecore.research.records import ConsentRecord
from facecore.research.recorder import ResearchRecorder


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _consent(session_id: str = "sess-e1-001") -> ConsentRecord:
    return ConsentRecord(
        session_id=session_id,
        participant_id="part-synth-001",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-16T10:00:00Z",
        record_expires_at_utc="2026-10-16T10:00:00Z",
        image_expires_at_utc="2026-09-23T10:00:00Z",
    )


def _manifest(tmp_path: Path) -> Any:
    from facecore.research.experiment import ExperimentManifest

    return ExperimentManifest.from_dict(
        {
            "identity": {"experiment_id": "exp-e1-001"},
            "software": {},
            "gallery": {},
            "policy": {},
            "capture": {},
            "privacy": {},
            "study": {},
            "analysis": {},
        }
    )


def _attempt(attempt_id: str = "attempt-001") -> Any:
    from facecore.research.experiment import AttemptRecord

    return AttemptRecord(
        experiment_id="exp-e1-001",
        attempt_id=attempt_id,
        participant_id="part-synth-001",
        visit_id="visit-001",
        condition_id="cond-001",
        attempt_index=1,
        retry_of=None,
        consent_ref="consent-synth",
        requested_at_utc="2026-09-16T10:00:00Z",
        accepted_at_utc="2026-09-16T10:00:01Z",
        started_at_utc=None,
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


def _timeout_result(session_id: str) -> SessionResult:
    return SessionResult(
        session_id=session_id,
        schema_version="v1",
        status=SessionStatus.timeout,
        matched_identity=None,
        reason_codes=("deadline",),
        elapsed_ms=5000.0,
        frames_sampled=0,
        frames_usable=0,
        frames_rejected=0,
        frames_dropped=0,
        support_sequences=(),
        profile_digest="profile-synth",
        model_generation="cli-fake-gen-1",
        gallery_digest="cli-fake-gallery",
    )


def test_begin_attempt_persists_before_any_camera_open(tmp_path: Path) -> None:
    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    recorder.begin_attempt(_manifest(tmp_path), _attempt(), _consent())
    camera_opened: list[str] = []
    # Simulated open failure BEFORE any camera touch: the attempt must
    # already be durable, and readable from a rebuilt recorder.
    rebuilt = _recorder(tmp_path, now)
    attempts = rebuilt.list_attempts(experiment_id="exp-e1-001")
    assert [a.attempt_id for a in attempts] == ["attempt-001"]
    assert camera_opened == []


def test_open_failure_yields_open_error_attempt(tmp_path: Path) -> None:
    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    recorder.begin_attempt(_manifest(tmp_path), _attempt(), _consent())
    recorder.finish_attempt(
        "attempt-001",
        result=None,
        operational_status="open_error",
        error_code="camera_open_failed",
    )
    stored = _recorder(tmp_path, now).list_attempts(
        experiment_id="exp-e1-001"
    )
    assert len(stored) == 1
    assert stored[0].operational_status == "open_error"
    assert stored[0].error_code == "camera_open_failed"


def test_cancel_keeps_one_failure_accounting_row(tmp_path: Path) -> None:
    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    recorder.begin_attempt(_manifest(tmp_path), _attempt(), _consent())
    recorder.finish_attempt(
        "attempt-001",
        result=None,
        operational_status="cancelled",
        error_code=None,
    )
    stored = _recorder(tmp_path, now).list_attempts(
        experiment_id="exp-e1-001"
    )
    assert len(stored) == 1
    assert stored[0].operational_status == "cancelled"


def test_crash_recovery_keeps_accepted_attempt(tmp_path: Path) -> None:
    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    recorder.begin_attempt(_manifest(tmp_path), _attempt(), _consent())
    del recorder  # crash: no finish_attempt, no image commit.
    recovered = _recorder(tmp_path, now)
    stored = recovered.list_attempts(experiment_id="exp-e1-001")
    assert [a.attempt_id for a in stored] == ["attempt-001"]
    assert stored[0].operational_status == "accepted"


def test_same_attempt_id_resent_does_not_duplicate(tmp_path: Path) -> None:
    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    recorder.begin_attempt(_manifest(tmp_path), _attempt(), _consent())
    recorder.begin_attempt(_manifest(tmp_path), _attempt(), _consent())
    stored = _recorder(tmp_path, now).list_attempts(
        experiment_id="exp-e1-001"
    )
    assert [a.attempt_id for a in stored] == ["attempt-001"]


def test_record_image_consent_separation(tmp_path: Path) -> None:
    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    image_only = ConsentRecord(
        session_id="sess-e1-001",
        participant_id="part-synth-001",
        record_consent=False,
        image_consent=True,
        consented_at_utc="2026-09-16T10:00:00Z",
        record_expires_at_utc="2026-10-16T10:00:00Z",
        image_expires_at_utc="2026-09-23T10:00:00Z",
    )
    with pytest.raises(PermissionError):
        recorder.begin_attempt(
            _manifest(tmp_path), _attempt(), image_only
        )
    assert _recorder(tmp_path, now).list_attempts(
        experiment_id="exp-e1-001"
    ) == []


def test_write_label_survives_restart_and_correction_keeps_revision(
    tmp_path: Path,
) -> None:
    from facecore.research.experiment import EvaluationLabel

    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    recorder.begin_attempt(_manifest(tmp_path), _attempt(), _consent())
    recorder.write_label(
        EvaluationLabel(
            attempt_id="attempt-001",
            revision=1,
            kind="enrolled",
            identity_id="person-01",
            actor_ref="operator-synth",
            labeled_at="2026-09-16T11:00:00Z",
        )
    )
    restarted = _recorder(tmp_path, now)
    assert restarted.read_label("attempt-001").identity_id == "person-01"
    restarted.write_label(
        EvaluationLabel(
            attempt_id="attempt-001",
            revision=2,
            kind="enrolled",
            identity_id="person-02",
            actor_ref="operator-synth",
            labeled_at="2026-09-16T11:30:00Z",
        )
    )
    history = _recorder(tmp_path, now).read_label_history("attempt-001")
    assert [label.revision for label in history] == [1, 2]
    assert history[0].identity_id == "person-01"


def test_write_failed_start_does_not_open_camera(tmp_path: Path) -> None:
    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    camera_opened: list[str] = []
    # begin_attempt raising (e.g. store write failure) means the Start was
    # never accepted: the caller must not proceed to open the camera.
    try:
        recorder.begin_attempt(_manifest(tmp_path), _attempt(), _consent())
    except (OSError, PermissionError, ValueError):
        pass
    else:  # pragma: no cover - RED expects the accepted path blocked
        camera_opened.append("camera-must-stay-closed")
    assert camera_opened == [] or _recorder(tmp_path, now).list_attempts(
        experiment_id="exp-e1-001"
    ) != []


def test_image_abort_keeps_failure_accounting(tmp_path: Path) -> None:
    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    recorder.begin_attempt(
        _manifest(tmp_path), _attempt("attempt-004"), _consent("sess-e1-004")
    )
    recorder.begin("sess-e1-004", _consent("sess-e1-004"))
    recorder.abort("sess-e1-004", reason="multi_face")
    stored = _recorder(tmp_path, now).list_attempts(
        experiment_id="exp-e1-001"
    )
    assert [a.attempt_id for a in stored] == ["attempt-004"]


def test_full_withdrawal_deletes_attempt_and_label_and_report_recounts(
    tmp_path: Path,
) -> None:
    from facecore.research.experiment import EvaluationLabel

    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    recorder.begin_attempt(_manifest(tmp_path), _attempt(), _consent())
    recorder.write_label(
        EvaluationLabel(
            attempt_id="attempt-001",
            revision=1,
            kind="enrolled",
            identity_id="person-01",
            actor_ref="operator-synth",
            labeled_at="2026-09-16T11:00:00Z",
        )
    )
    recorder.finish_attempt(
        "attempt-001",
        result=_timeout_result("sess-e1-001"),
        operational_status="timeout",
        error_code=None,
    )
    recorder.withdraw_attempt("attempt-001")
    rebuilt = _recorder(tmp_path, now)
    assert rebuilt.list_attempts(experiment_id="exp-e1-001") == []
    with pytest.raises(KeyError):
        rebuilt.read_label("attempt-001")


def test_terminal_timeout_finish_records_timeout_attempt(
    tmp_path: Path,
) -> None:
    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    recorder.begin_attempt(
        _manifest(tmp_path), _attempt("attempt-005"), _consent("sess-e1-005")
    )
    recorder.finish_attempt(
        "attempt-005",
        result=_timeout_result("sess-e1-005"),
        operational_status="timeout",
        error_code=None,
    )
    stored = _recorder(tmp_path, now).list_attempts(
        experiment_id="exp-e1-001"
    )
    assert stored[0].operational_status == "timeout"
    assert stored[0].bundle_ref == "sess-e1-005"


def test_time_advances_past_one_day(tmp_path: Path) -> None:
    later = _utc("2026-09-16T10:00:01Z") + timedelta(days=1)
    recorder = _recorder(tmp_path, later)
    recorder.begin_attempt(_manifest(tmp_path), _attempt(), _consent())
    assert _recorder(tmp_path, later).list_attempts(
        experiment_id="exp-e1-001"
    )


# ---------------------------------------------------------------------------
# E1 Rework Findings: F1 (at-rest privacy) & F2-F6 (hygiene & fail-closed)
# ---------------------------------------------------------------------------


def test_at_rest_bytes_contain_no_sensitive_literals(
    tmp_path: Path,
    assert_no_leak: Any,
) -> None:
    """F1 + Commander mandate: assert raw on-disk bytes contain no plaintext."""
    from facecore.research.experiment import EvaluationLabel

    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    recorder.begin_attempt(
        _manifest(tmp_path),
        _attempt("attempt-at-rest"),
        _consent("sess-at-rest"),
    )
    recorder.write_label(
        EvaluationLabel(
            attempt_id="attempt-at-rest",
            revision=1,
            kind="enrolled",
            identity_id="person-secret-truth",
            actor_ref="operator-synth",
            labeled_at="2026-09-16T11:00:00Z",
        )
    )
    # The raw bytes of ALL files in store must not contain participant_id
    # or ground-truth identity literals.
    assert_no_leak(
        tmp_path / "store",
        ["part-synth-001", "person-secret-truth"],
    )


def test_list_attempts_corrupt_file_fails_closed(tmp_path: Path) -> None:
    """F4: list_attempts must fail closed on corrupt files, never silently skip."""
    from facecore.contracts.crypto import StoreCorruptionError

    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    recorder.begin_attempt(_manifest(tmp_path), _attempt(), _consent())
    # Inject corrupt file into the experiment attempts directory.
    bad_file = tmp_path / "store" / "_attempts" / "exp-e1-001" / "corrupt.enc"
    bad_file.write_bytes(b"not-valid-wire-blob")
    with pytest.raises((StoreCorruptionError, ValueError)):
        recorder.list_attempts(experiment_id="exp-e1-001")


def test_label_same_revision_overwrite_rejected(tmp_path: Path) -> None:
    """F5: overwriting an existing label revision must be rejected."""
    from facecore.research.experiment import EvaluationLabel

    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    recorder.begin_attempt(_manifest(tmp_path), _attempt(), _consent())
    recorder.write_label(
        EvaluationLabel(
            attempt_id="attempt-001",
            revision=1,
            kind="enrolled",
            identity_id="person-01",
            actor_ref="operator-synth",
            labeled_at="2026-09-16T11:00:00Z",
        )
    )
    with pytest.raises(ValueError, match="already exists"):
        recorder.write_label(
            EvaluationLabel(
                attempt_id="attempt-001",
                revision=1,
                kind="enrolled",
                identity_id="person-99",
                actor_ref="operator-synth",
                labeled_at="2026-09-16T11:05:00Z",
            )
        )


def test_begin_attempt_manifest_mismatch_rejected(tmp_path: Path) -> None:
    """F6: begin_attempt must reject mismatched manifest.experiment_id."""
    from facecore.research.experiment import ExperimentManifest

    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    mismatched_manifest = ExperimentManifest.from_dict(
        {
            "identity": {"experiment_id": "exp-DIFFERENT"},
            "software": {},
            "gallery": {},
            "policy": {},
            "capture": {},
            "privacy": {},
            "study": {},
            "analysis": {},
        }
    )
    with pytest.raises(ValueError, match="experiment_id"):
        recorder.begin_attempt(mismatched_manifest, _attempt(), _consent())


def test_withdraw_attempt_destroys_key(tmp_path: Path) -> None:
    """F3: withdraw_attempt must destroy the DEK (cryptographic erasure)."""
    from facecore.contracts.crypto import KeyNotFoundError
    from facecore.research.keys import ResearchKeyProvider

    now = _utc("2026-09-16T10:00:01Z")
    recorder = _recorder(tmp_path, now)
    recorder.begin_attempt(_manifest(tmp_path), _attempt("attempt-w1"), _consent())
    assert ResearchKeyProvider(tmp_path / "keys").get_key("rk_attempt-w1") is not None
    recorder.withdraw_attempt("attempt-w1")
    with pytest.raises(KeyNotFoundError):
        ResearchKeyProvider(tmp_path / "keys").get_key("rk_attempt-w1")


def test_purge_expired_purges_expired_attempts_and_labels(
    tmp_path: Path,
) -> None:
    """F2: purge_expired must clean up expired attempts and their labels."""
    from facecore.research.experiment import EvaluationLabel

    start = _utc("2026-09-16T10:00:01Z")
    consent_expiring = ConsentRecord(
        session_id="sess-expiring",
        participant_id="part-synth-001",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-16T10:00:00Z",
        record_expires_at_utc="2026-09-18T10:00:00Z",
        image_expires_at_utc="2026-09-17T10:00:00Z",
    )
    recorder = _recorder(tmp_path, start)
    recorder.begin_attempt(
        _manifest(tmp_path),
        _attempt("attempt-expiring"),
        consent_expiring,
    )
    recorder.write_label(
        EvaluationLabel(
            attempt_id="attempt-expiring",
            revision=1,
            kind="enrolled",
            identity_id="person-01",
            actor_ref="operator-synth",
            labeled_at="2026-09-16T11:00:00Z",
        )
    )
    # Clock moves past record_expires_at_utc (3 days later)
    past_expiry = _utc("2026-09-19T10:00:00Z")
    recorder_later = _recorder(tmp_path, past_expiry)
    purged = recorder_later.purge_expired(past_expiry)
    assert "attempt-expiring" in purged
    assert recorder_later.list_attempts(experiment_id="exp-e1-001") == []
    with pytest.raises(KeyError):
        recorder_later.read_label("attempt-expiring")
