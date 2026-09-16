"""E1 RED: experiment / attempt / label schema contracts (Phase 2B §4 & §11.2).

Source of truth:
    - docs/plans/2026-09-16-phase2b-mac-recognition-execution-plan.md
      §4 (manifest fields), §11.2 (signatures), §12 E1;
    - Task: t-20260916021626405371-53424-11.

RED contract (must fail before E1 implementation exists):
- ModuleNotFoundError: facecore.research.experiment does not exist yet.
- The import path below follows §11.1 exactly (research/experiment.py);
  a RED failure must read as a missing module, not a misspelled import.

Only synthetic payloads; never real faces; camera-free.
"""

from __future__ import annotations

from typing import Any

import pytest


def _manifest_dict() -> dict[str, Any]:
    return {
        "identity": {
            "experiment_id": "exp-e1-001",
            "schema_version": "v2",
            "decision_refs": ["d-test-1"],
            "task_refs": ["t-test-1"],
            "owner": "lead-synth",
            "custodian": "custodian-synth",
        },
        "software": {
            "code_sha": "0" * 40,
            "python": "3.14.0",
            "os": "darwin-test",
            "arch": "arm64",
            "ort_version": "ort-test",
            "model_checksums": {"yunet": "abc123"},
            "align_contract": 3,
            "generation": "sface-112-rgb+align3",
        },
        "gallery": {
            "corpus_manifest_digest": "digest-synth",
            "templates_per_identity": 1,
            "gallery_digest": "gallery-synth",
            "identity_count": 2,
            "consent_ref": "consent-synth",
        },
        "policy": {
            "profile": {"profile_version": "e1-test-v1"},
            "profile_digest": "profile-synth",
        },
        "capture": {
            "requested_device": "fake",
            "resolved_device": "fake",
            "shape": [16, 16],
            "collection_mode": "early-stop",
        },
        "privacy": {
            "store_kind": "external-synth",
            "key_namespace": "research-synth",
            "record_consent": True,
            "image_consent": True,
            "record_ttl_days": 30,
            "image_ttl_days": 7,
        },
        "study": {
            "participant_ids": ["part-synth-001"],
            "participant_types": {"part-synth-001": "enrolled"},
            "visits": ["visit-001"],
            "planned_attempts": 1,
            "retry_policy": "no-retry",
            "split": {"visit-001": "development"},
        },
        "analysis": {
            "arms": ["A", "B"],
            "primary_outcome": "correct_enrolled",
            "trigger_version": "t01-t12-test",
            "paired_exclusion": "none",
        },
    }


def test_manifest_from_dict_roundtrip_and_digest() -> None:
    from facecore.research.experiment import ExperimentManifest

    manifest = ExperimentManifest.from_dict(_manifest_dict())
    assert manifest.experiment_id == "exp-e1-001"
    assert ExperimentManifest.from_dict(manifest.to_dict()) == manifest
    first = manifest.digest()
    assert manifest.digest() == first
    altered = _manifest_dict()
    altered["policy"] = {"profile": {"profile_version": "e1-test-v2"}}
    assert ExperimentManifest.from_dict(altered).digest() != first


def test_manifest_missing_required_block_rejected() -> None:
    from facecore.research.experiment import ExperimentManifest

    incomplete = _manifest_dict()
    del incomplete["study"]
    with pytest.raises(ValueError):
        ExperimentManifest.from_dict(incomplete)


def test_attempt_record_has_no_truth_field() -> None:
    from facecore.research.experiment import AttemptRecord

    attempt = AttemptRecord(
        experiment_id="exp-e1-001",
        attempt_id="attempt-001",
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
    assert "truth" not in attempt.to_dict()
    with pytest.raises(TypeError):
        AttemptRecord(
            experiment_id="exp-e1-001",
            attempt_id="attempt-002",
            participant_id="part-synth-001",
            visit_id="visit-001",
            condition_id="cond-001",
            attempt_index=2,
            retry_of=None,
            consent_ref="consent-synth",
            requested_at_utc="2026-09-16T10:00:00Z",
            accepted_at_utc="2026-09-16T10:00:02Z",
            started_at_utc=None,
            ended_at_utc=None,
            operational_status="accepted",
            error_code=None,
            bundle_ref=None,
            truth="person-01",  # type: ignore[call-arg]
        )


def test_attempt_record_rejects_unknown_status() -> None:
    from facecore.research.experiment import AttemptRecord

    with pytest.raises(ValueError):
        AttemptRecord(
            experiment_id="exp-e1-001",
            attempt_id="attempt-003",
            participant_id="part-synth-001",
            visit_id="visit-001",
            condition_id="cond-001",
            attempt_index=3,
            retry_of=None,
            consent_ref="consent-synth",
            requested_at_utc="2026-09-16T10:00:00Z",
            accepted_at_utc="2026-09-16T10:00:03Z",
            started_at_utc=None,
            ended_at_utc=None,
            operational_status="miraculously_better",
            error_code=None,
            bundle_ref=None,
        )


def test_evaluation_label_enrolled_requires_identity() -> None:
    from facecore.research.experiment import EvaluationLabel

    label = EvaluationLabel(
        attempt_id="attempt-001",
        revision=1,
        kind="enrolled",
        identity_id="person-01",
        actor_ref="operator-synth",
        labeled_at="2026-09-16T11:00:00Z",
    )
    assert EvaluationLabel.from_dict(label.to_dict()) == label
    with pytest.raises(ValueError):
        EvaluationLabel(
            attempt_id="attempt-001",
            revision=1,
            kind="enrolled",
            identity_id=None,
            actor_ref="operator-synth",
            labeled_at="2026-09-16T11:00:00Z",
        )
    with pytest.raises(ValueError):
        EvaluationLabel(
            attempt_id="attempt-002",
            revision=1,
            kind="unenrolled",
            identity_id="person-01",
            actor_ref="operator-synth",
            labeled_at="2026-09-16T11:00:00Z",
        )


def test_evaluation_label_rejects_unknown_kind() -> None:
    from facecore.research.experiment import EvaluationLabel

    with pytest.raises(ValueError):
        EvaluationLabel(
            attempt_id="attempt-001",
            revision=1,
            kind="probably_fine",
            identity_id=None,
            actor_ref="operator-synth",
            labeled_at="2026-09-16T11:00:00Z",
        )
