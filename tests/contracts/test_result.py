"""Task 1 RED/GREEN: IdentificationResult frozen contract."""

import json

import pytest

from facecore import SCHEMA_VERSION
from facecore.contracts.result import (
    Decision,
    IdentificationResult,
    Quality,
    ResultStatus,
)


def _matched_kwargs() -> dict:
    return {
        "status": "matched",
        "identity": {"id": "person-001", "display_name": "Test Person", "metadata": {}},
        "decision": Decision(
            score=0.82,
            runner_up_score=None,
            threshold=0.76,
            margin=None,
            reason_codes=["match_threshold_met"],
        ),
        "quality": Quality(status="accepted", reason_codes=[]),
        "model_version": "sface-2021dec-fp32",
        "template_revision": 1,
        "candidate_created": False,
    }


def test_rejects_authenticated_status() -> None:
    """Failing case from the plan: 'authenticated' is never a Phase-1A state."""
    with pytest.raises(ValueError):
        IdentificationResult(
            status="authenticated",
            **{k: v for k, v in _matched_kwargs().items() if k != "status"},
        )  # type: ignore[arg-type]


def test_matched_serializes_to_spec_key_set_in_frozen_order() -> None:
    result = IdentificationResult(**_matched_kwargs())
    payload = json.loads(result.to_json())
    assert list(payload.keys()) == [
        "schema_version",
        "status",
        "identity",
        "decision",
        "quality",
        "model_version",
        "template_revision",
        "candidate_created",
    ]
    assert payload["schema_version"] == 1
    assert payload["identity"] == {
        "id": "person-001",
        "display_name": "Test Person",
        "metadata": {},
    }


def test_non_matched_states_carry_null_identity_and_no_guess() -> None:
    for status in ("review", "unknown", "invalid_input"):
        result = IdentificationResult(
            status=status,  # type: ignore[arg-type]
            identity=None,
            decision=Decision(
                score=None,
                runner_up_score=None,
                threshold=None,
                margin=None,
                reason_codes=[],
            ),
            quality=Quality(status="accepted", reason_codes=[]),
            model_version="sface-2021dec-fp32",
            template_revision=1,
            candidate_created=False,
        )
        payload = json.loads(result.to_json())
        assert payload["identity"] is None
        assert (
            "display_name" not in payload["identity"] if payload["identity"] else True
        )


def test_candidate_created_is_always_false_in_phase_1a() -> None:
    with pytest.raises(ValueError):
        IdentificationResult(**{**_matched_kwargs(), "candidate_created": True})


def test_template_revision_is_one_for_one_shot_identity() -> None:
    result = IdentificationResult(**_matched_kwargs())
    assert result.template_revision == 1
    assert SCHEMA_VERSION == 1


def test_result_status_enum_has_exactly_four_states() -> None:
    assert {s.value for s in ResultStatus} == {
        "matched",
        "review",
        "unknown",
        "invalid_input",
    }
