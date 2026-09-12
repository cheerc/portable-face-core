"""Task 3 RED/GREEN: confirmation-gated shadow candidate pipeline.

Source of truth: 1B plan §11 Task 3.
RED: ``ModuleNotFoundError: No module named 'facecore.governance.candidate'``.
"""

from pathlib import Path

import pytest

from facecore.contracts.confirmation import (
    ActorType,
    ConfirmationRequest,
    ConfirmationVerdict,
)
from facecore.contracts.result import Decision, IdentificationResult, Quality
from facecore.governance.candidate import CandidateDecision, CandidatePipeline
from facecore.storage.key_provider import InMemoryKeyProvider
from facecore.storage.sqlite_repo import SQLiteRepository


def _result(
    score: float | None = 0.90,
    quality: str = "accepted",
    identity_id: str = "person-001",
) -> IdentificationResult:
    return IdentificationResult(
        status="matched",
        identity={"id": identity_id, "display_name": "Test", "metadata": {}},
        decision=Decision(
            score=score,
            runner_up_score=None,
            threshold=0.76,
            margin=None,
            reason_codes=[],
        ),
        quality=Quality(status=quality, reason_codes=[]),
        model_version="sface-2021dec-fp32",
        template_revision=1,
        candidate_created=False,
    )


def _confirm(
    verdict: ConfirmationVerdict = ConfirmationVerdict.CORRECT,
) -> ConfirmationRequest:
    return ConfirmationRequest(
        request_id="r-1",
        verdict=verdict,
        actor=ActorType.USER,
        created_at="2026-09-12T00:00:00+08:00",
    )


def _open_repo(tmp_path: Path) -> SQLiteRepository:
    return SQLiteRepository(str(tmp_path / "facecore.db"), InMemoryKeyProvider())


def _enroll(repo: SQLiteRepository, identity_id: str = "person-001") -> None:
    from facecore.contracts.template import FaceTemplate, TemplateRevision

    template_id = f"t-{identity_id}-1"
    repo.enroll_identity(
        identity_id,
        "Test Person",
        FaceTemplate(
            template_id=template_id,
            identity_id=identity_id,
            model_version="sface-2021dec-fp32",
            embedding_dim=4,
            revision=TemplateRevision(
                revision=1, template_id=template_id, supersedes=None
            ),
        ),
        b"e" * 16,
        b"x" * 8,
        template_id,
    )


def _pipeline(tmp_path: Path) -> tuple[CandidatePipeline, SQLiteRepository]:
    repo = _open_repo(tmp_path)
    repo.initialize()
    _enroll(repo)
    return CandidatePipeline(repo), repo


def test_correct_high_quality_match_creates_pending_candidate(
    tmp_path: Path,
) -> None:
    pipeline, repo = _pipeline(tmp_path)
    decision = pipeline.evaluate_observation(
        _result(), b"face-bytes", _confirm(ConfirmationVerdict.CORRECT)
    )
    assert isinstance(decision, CandidateDecision)
    assert decision.created is True
    assert decision.candidate_id is not None
    con = repo.connection
    assert con is not None
    row = con.execute(
        "SELECT status, additional_corroboration_count"
        " FROM candidate_templates WHERE id = ?",
        (decision.candidate_id,),
    ).fetchone()
    assert row == ("pending", 0)
    assert (
        con.execute("SELECT COUNT(*) FROM candidate_templates").fetchone()[0]
        == 1
    )


@pytest.mark.parametrize(
    "verdict",
    [
        ConfirmationVerdict.NOT_ME,
        ConfirmationVerdict.CANCELLED,
        ConfirmationVerdict.TIMEOUT,
        ConfirmationVerdict.EOF,
    ],
)
def test_not_me_creates_no_candidate(
    tmp_path: Path, verdict: ConfirmationVerdict
) -> None:
    pipeline, repo = _pipeline(tmp_path)
    before = set(tmp_path.rglob("*"))
    decision = pipeline.evaluate_observation(
        _result(), b"face-bytes", _confirm(verdict)
    )
    assert decision.created is False
    assert decision.candidate_id is None
    con = repo.connection
    assert con is not None
    assert (
        con.execute("SELECT COUNT(*) FROM candidate_templates").fetchone()[0]
        == 0
    )
    assert set(tmp_path.rglob("*")) - before <= {
        tmp_path / "facecore.db",
        tmp_path / "facecore.db-wal",
        tmp_path / "facecore.db-shm",
    }


@pytest.mark.parametrize("score", [0.87, 0.80, None])
def test_below_threshold_creates_no_candidate(
    tmp_path: Path, score: float | None
) -> None:
    pipeline, repo = _pipeline(tmp_path)
    decision = pipeline.evaluate_observation(
        _result(score=score), b"face-bytes", _confirm(ConfirmationVerdict.CORRECT)
    )
    assert decision.created is False
    assert decision.candidate_id is None
    con = repo.connection
    assert con is not None
    assert (
        con.execute("SELECT COUNT(*) FROM candidate_templates").fetchone()[0]
        == 0
    )


def test_rejected_quality_creates_no_candidate(tmp_path: Path) -> None:
    pipeline, repo = _pipeline(tmp_path)
    decision = pipeline.evaluate_observation(
        _result(quality="rejected"),
        b"face-bytes",
        _confirm(ConfirmationVerdict.CORRECT),
    )
    assert decision.created is False
    assert decision.candidate_id is None


def test_review_status_creates_no_candidate(tmp_path: Path) -> None:
    pipeline, repo = _pipeline(tmp_path)
    result = IdentificationResult(
        status="review",
        identity=None,
        decision=Decision(
            score=0.95,
            runner_up_score=None,
            threshold=0.76,
            margin=None,
            reason_codes=[],
        ),
        quality=Quality(status="accepted", reason_codes=[]),
        model_version="sface-2021dec-fp32",
        template_revision=1,
        candidate_created=False,
    )
    decision = pipeline.evaluate_observation(
        result, b"face-bytes", _confirm(ConfirmationVerdict.CORRECT)
    )
    assert decision.created is False
    assert decision.candidate_id is None


def test_pending_observation_is_memory_only(tmp_path: Path) -> None:
    pipeline, _ = _pipeline(tmp_path)
    assert pipeline.pending_count() == 0
    pipeline.hold_pending(b"face-bytes")
    assert pipeline.pending_count() == 1
    assert pipeline.pending_bytes() == len(b"face-bytes")
    pipeline.discard_pending()
    assert pipeline.pending_count() == 0
    files = {p.name for p in tmp_path.rglob("*")}
    assert not any(name.endswith((".png", ".jpg", ".npy", ".bin")) for name in files)


def test_seed_event_cannot_promote_itself(tmp_path: Path) -> None:
    pipeline, _ = _pipeline(tmp_path)
    decision = pipeline.evaluate_observation(
        _result(), b"face-bytes", _confirm(ConfirmationVerdict.CORRECT)
    )
    assert decision.created is True
    assert decision.candidate_id is not None
    assert decision.promoted is False


def test_confirm_learning_cli_rejects_not_me_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from facecore import cli as cli_module

    monkeypatch.chdir(tmp_path)
    code = cli_module.main(
        ["confirm-learning", "--verdict", "not_me", "--score", "0.95"]
    )
    assert code == 0
    assert not (tmp_path / "facecore.db").exists()


def test_confirm_learning_cli_requires_explicit_correct(tmp_path: Path) -> None:
    from facecore import cli as cli_module

    code = cli_module.main(["confirm-learning", "--verdict", "bogus", "--score", "0.9"])
    assert code == 2
    assert not (tmp_path / "facecore.db").exists()


def test_confirm_learning_cli_correct_reports_no_creation(tmp_path: Path) -> None:
    from facecore import cli as cli_module

    code = cli_module.main(
        ["confirm-learning", "--verdict", "correct", "--score", "0.95"]
    )
    assert code == 0
    assert not (tmp_path / "facecore.db").exists()


def test_candidate_row_round_trips_through_repository(tmp_path: Path) -> None:
    pipeline, repo = _pipeline(tmp_path)
    decision = pipeline.evaluate_observation(
        _result(), b"face-bytes-16!!", _confirm(ConfirmationVerdict.CORRECT)
    )
    assert decision.created is True
    con = repo.connection
    assert con is not None
    row = con.execute(
        "SELECT id, identity_id, status, additional_corroboration_count,"
        " key_id, embedding_blob, exemplar_blob FROM candidate_templates"
    ).fetchone()
    assert row is not None
    assert row[2] == "pending"
    assert row[3] == 0
    assert isinstance(row[4], str) and row[4].startswith("key-")
    assert isinstance(row[5], bytes) and isinstance(row[6], bytes)
