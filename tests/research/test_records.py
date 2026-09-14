"""Phase 2A Task T1 tests: research records and consent envelopes.

Source of truth: Phase 2A Implementation Plan §4 & §6 T1;
Task: t-20260914110810002830-76424-31;
Governing decision: d-20260914110757304910-5.
"""

from __future__ import annotations

import pytest

from facecore.live.contracts import SessionResult, SessionStatus
from facecore.research.records import ConsentRecord, ResearchSessionRecord


def test_consent_record_dual_permissions_and_expiry() -> None:
    consent = ConsentRecord(
        session_id="sess-001",
        participant_id="part-999",
        record_consent=True,
        image_consent=False,  # image denied, record granted
        consented_at_utc="2026-09-14T10:00:00Z",
        record_expires_at_utc="2026-10-14T10:00:00Z",  # 30 days
        image_expires_at_utc="2026-09-21T10:00:00Z",  # 7 days
        guardian_consent_verified=False,
    )
    assert consent.record_consent is True
    assert consent.image_consent is False
    assert consent.record_expires_at_utc > consent.image_expires_at_utc

    # JSON roundtrip
    d = consent.to_dict()
    restored = ConsentRecord.from_dict(d)
    assert restored == consent


@pytest.mark.parametrize(
    "bad_ts",
    [
        "",
        "not-a-timestamp",
        "2026/09/14",
    ],
)
def test_consent_record_rejects_malformed_timestamps(bad_ts: str) -> None:
    with pytest.raises(ValueError):
        ConsentRecord(
            session_id="sess-001",
            participant_id="part-999",
            record_consent=True,
            image_consent=True,
            consented_at_utc=bad_ts,
            record_expires_at_utc="2026-10-14T10:00:00Z",
            image_expires_at_utc="2026-09-21T10:00:00Z",
        )


def test_research_session_record_isolated_label_envelope() -> None:
    consent = ConsentRecord(
        session_id="sess-002",
        participant_id="part-001",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-14T10:00:00Z",
        record_expires_at_utc="2026-10-14T10:00:00Z",
        image_expires_at_utc="2026-09-21T10:00:00Z",
    )
    result = SessionResult(
        session_id="sess-002",
        schema_version="v1",
        status=SessionStatus.matched,
        matched_identity="person-23",
        reason_codes=("supported_3_frames",),
        elapsed_ms=850.0,
        frames_sampled=4,
        frames_usable=4,
        frames_rejected=0,
        frames_dropped=0,
        support_sequences=(1, 2, 3),
        profile_digest="p-hash",
        model_generation="gen-1",
        gallery_digest="g-hash",
    )

    record = ResearchSessionRecord(
        session_id="sess-002",
        schema_version="v1",
        consent=consent,
        result=result,
        ground_truth_label="person-23",  # Operator external evaluation label
        notes="Normal lighting, clean frontal angle",
    )

    assert record.ground_truth_label == "person-23"
    assert record.result.matched_identity == "person-23"

    # JSON roundtrip
    d = record.to_dict()
    restored = ResearchSessionRecord.from_dict(d)
    assert restored.session_id == record.session_id
    assert restored.ground_truth_label == record.ground_truth_label
    assert restored.result.matched_identity == "person-23"
    assert restored.consent.image_consent is True


def test_research_session_record_mismatched_session_id_rejected() -> None:
    consent = ConsentRecord(
        session_id="sess-AAA",
        participant_id="part-001",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-14T10:00:00Z",
        record_expires_at_utc="2026-10-14T10:00:00Z",
        image_expires_at_utc="2026-09-21T10:00:00Z",
    )
    result = SessionResult(
        session_id="sess-BBB",  # Mismatch with consent
        schema_version="v1",
        status=SessionStatus.unknown,
        matched_identity=None,
        reason_codes=(),
        elapsed_ms=5000.0,
        frames_sampled=25,
        frames_usable=25,
        frames_rejected=0,
        frames_dropped=0,
        support_sequences=(),
        profile_digest="p-hash",
        model_generation="gen-1",
        gallery_digest="g-hash",
    )

    with pytest.raises(ValueError, match="session_id mismatch"):
        ResearchSessionRecord(
            session_id="sess-AAA",
            schema_version="v1",
            consent=consent,
            result=result,
            ground_truth_label="unknown",
        )
