"""Task 7 RED/GREEN: model generation migration & exemplar re-embedding.

Source of truth: 1B plan §11 Task 7.
RED: ``ModuleNotFoundError: No module named 'facecore.governance.migration'``.
Failing case from the plan: a promoted candidate retains its G1 status
and is permitted to match against G2 templates
(``assert candidate.status == 'generation_retired' (got 'promoted')``).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from facecore.contracts.candidate import CandidateStatus
from facecore.contracts.migration import ModelMigrationManifest
from facecore.governance.lifecycle import LifecycleManager
from facecore.governance.migration import MigrationReport, ModelMigrationManager
from facecore.storage.key_provider import InMemoryKeyProvider
from facecore.storage.sqlite_repo import SQLiteRepository


def _manifest(**overrides: object) -> ModelMigrationManifest:
    base: dict[str, object] = {
        "embedder_artifact_hash": "0" * 64,
        "detector_generation": "yunet-2023mar",
        "preprocessing_generation": "sface-112-rgb",
        "tensor_layout": "NCHW",
        "normalization_contract": "scale=1/128;mean=127.5;std=128",
        "embedding_dimension": 128,
        "numerical_precision": "fp32",
        "quantization_type": "none",
        "execution_runtime": "onnxruntime-cpu-arm64",
    }
    base.update(overrides)
    return ModelMigrationManifest(**base)  # type: ignore[arg-type]


def _open_repo(
    tmp_path: Path, name: str = "facecore.db"
) -> tuple[SQLiteRepository, LifecycleManager]:
    provider = InMemoryKeyProvider()
    repo = SQLiteRepository(str(tmp_path / name), provider)
    repo.initialize()
    return repo, LifecycleManager(repo)


def _seed_all_statuses(repo: SQLiteRepository) -> None:
    manager = LifecycleManager(repo)
    manager.add_identity("person-001", "Test Person", b"e" * 64, b"x" * 64)
    manager.re_enroll("person-001", b"n" * 64, b"m" * 64)
    for candidate_id, status in (
        ("c-pending", "pending"),
        ("c-promoted", "promoted"),
        ("c-rejected", "rejected"),
        ("c-expired", "expired"),
    ):
        repo.add_candidate_record(
            candidate_id,
            "person-001",
            b"e" * 16,
            b"x" * 8,
            "2030-01-01T00:00:00Z",
            generation_id="G1",
            exemplar_crop_box=(0.0, 0.0, 112.0, 112.0),
            exemplar_landmarks=((30.0, 30.0), (82.0, 30.0)),
            quality_score=0.9,
            additional_corroboration_count=1,
            evidence_log="[]",
        )
        if status != "pending":
            con = repo.connection
            assert con is not None
            con.execute(
                "UPDATE candidate_templates SET status = ? WHERE id = ?",
                (status, candidate_id),
            )
            con.commit()


def test_promoted_candidate_transitions_to_generation_retired(
    tmp_path: Path,
) -> None:
    repo, _ = _open_repo(tmp_path)
    _seed_all_statuses(repo)
    manager = ModelMigrationManager(repo, current_generation="G1")
    report = manager.migrate_generation(_manifest(), target_generation="G2")
    assert isinstance(report, MigrationReport)
    statuses = {
        row[0]: row[1]
        for row in repo.connection.execute(  # type: ignore[union-attr]
            "SELECT id, status FROM candidate_templates"
        ).fetchall()
    }
    assert statuses["c-promoted"] == CandidateStatus.GENERATION_RETIRED.value
    assert statuses["c-rejected"] == "rejected"
    assert statuses["c-expired"] == "expired"


def test_pending_candidate_re_embedded_to_new_generation(
    tmp_path: Path,
) -> None:
    repo, _ = _open_repo(tmp_path)
    _seed_all_statuses(repo)
    manager = ModelMigrationManager(repo, current_generation="G1")
    report = manager.migrate_generation(_manifest(), target_generation="G2")
    assert report.candidates_migrated >= 1
    row = repo.connection.execute(  # type: ignore[union-attr]
        "SELECT status, generation_id FROM candidate_templates"
        " WHERE id = 'c-pending'"
    ).fetchone()
    assert row[0] == "pending"
    assert row[1] == "G2"


def test_pending_candidate_bad_geometry_rejected(tmp_path: Path) -> None:
    repo, _ = _open_repo(tmp_path)
    _seed_all_statuses(repo)
    con = repo.connection
    assert con is not None
    con.execute(
        "UPDATE candidate_templates SET exemplar_crop_box = ?"
        " WHERE id = 'c-pending'",
        ('[0.0, 0.0, -5.0, 4.0]',),
    )
    con.commit()
    manager = ModelMigrationManager(repo, current_generation="G1")
    manager.migrate_generation(_manifest(), target_generation="G2")
    row = con.execute(
        "SELECT status FROM candidate_templates WHERE id = 'c-pending'"
    ).fetchone()
    assert row[0] == "rejected"


def test_active_templates_migrated_atomically(tmp_path: Path) -> None:
    repo, _ = _open_repo(tmp_path)
    _seed_all_statuses(repo)
    manager = ModelMigrationManager(repo, current_generation="G1")
    report = manager.migrate_generation(_manifest(), target_generation="G2")
    assert report.active_migrated >= 1
    generations = {
        row[0]
        for row in repo.connection.execute(  # type: ignore[union-attr]
            "SELECT generation_id FROM face_templates WHERE status = 'active'"
        ).fetchall()
    }
    assert generations == {"G2"}


def test_mid_migration_crash_rolls_back_cleanly(tmp_path: Path) -> None:
    repo, _ = _open_repo(tmp_path)
    _seed_all_statuses(repo)
    manager = ModelMigrationManager(
        repo, current_generation="G1", _crash_after=1
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        manager.migrate_generation(_manifest(), target_generation="G2")
    # No partial state: every row still G1, every status untouched.
    generations = {
        row[0]
        for row in repo.connection.execute(  # type: ignore[union-attr]
            "SELECT generation_id FROM face_templates"
        ).fetchall()
    }
    assert generations == {"G1"}
    statuses = {
        row[0]: row[1]
        for row in repo.connection.execute(  # type: ignore[union-attr]
            "SELECT id, status FROM candidate_templates"
        ).fetchall()
    }
    assert statuses["c-promoted"] == "promoted"


def test_unreproducible_geometry_requires_re_enrollment(
    tmp_path: Path,
) -> None:
    repo, manager_lc = _open_repo(tmp_path)
    created = manager_lc.add_identity(
        "person-001", "Test Person", b"e" * 64, b"x" * 64
    )
    assert created.template_id is not None
    con = repo.connection
    assert con is not None
    con.execute(
        "UPDATE face_templates SET exemplar_crop_box = ? WHERE id = ?",
        ('[0.0, 0.0, -5.0, 4.0]', created.template_id),
    )
    con.commit()
    manager = ModelMigrationManager(repo, current_generation="G1")
    report = manager.migrate_generation(_manifest(), target_generation="G2")
    assert "person-001" in report.identities_needing_re_enrollment
    assert repo.get_identity_status("person-001") == "re_enrollment_required"


def test_cli_migrate_model_round_trip(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root / "src"),
        "FACECORE_DB": str(tmp_path / "cli.db"),
    }

    def _run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "facecore.cli", *args],
            capture_output=True,
            text=True,
            cwd=repo_root,
            env=env,
        )

    photo = tmp_path / "photo.bin"
    photo.write_bytes(bytes(range(128)))
    assert _run(
        "identity", "add", "--id", "person-001",
        "--display-name", "Test", "--photo", str(photo),
    ).returncode == 0
    proc = _run("migration", "status")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1])[
        "current_generation"
    ] == "G1"
    proc = _run("migration", "migrate-model", "--to-generation", "G2")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert payload["to_generation"] == "G2"
    assert payload["active_migrated"] >= 1
    proc = _run("migration", "status")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1])[
        "current_generation"
    ] == "G2"
    # Same-generation migration is a structured refusal (exit 4).
    proc = _run("migration", "migrate-model", "--to-generation", "G2")
    assert proc.returncode == 4, proc.stderr


def test_cross_generation_comparison_never_direct(tmp_path: Path) -> None:
    """G1 rows and G2 rows are never compared as vectors."""
    from facecore.contracts.template import FaceTemplate

    repo, _ = _open_repo(tmp_path)
    _seed_all_statuses(repo)
    manager = ModelMigrationManager(repo, current_generation="G1")
    manager.migrate_generation(_manifest(), target_generation="G2")
    templates = repo.list_active_templates()
    assert templates, "expected migrated actives"
    for template in templates:
        assert isinstance(template, FaceTemplate)
        assert template.generation_id == "G2"
        with pytest.raises(ValueError, match="cross-generation"):
            template.assert_comparable(
                FaceTemplate(
                    template_id="g1-probe",
                    identity_id="probe",
                    model_version=template.model_version,
                    embedding_dim=template.embedding_dim,
                    generation_id="G1",
                    revision=template.revision,
                )
            )
