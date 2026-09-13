"""Carry-forward 4 RED/GREEN: generation-aware stamping (option a).

Source of truth: PR-E r0 P2 (post-migration writes stamp G1) + PR #24
gate (option a selected; option b falsified — migration really produces
G2 while _new_template keeps writing G1).

RED: ``AttributeError: 'SQLiteRepository' object has no attribute
'get_current_generation'``.
"""

from pathlib import Path

import pytest

from facecore.governance.lifecycle import LifecycleManager
from facecore.governance.migration import ModelMigrationManager
from facecore.storage.key_provider import InMemoryKeyProvider
from facecore.storage.sqlite_repo import SQLiteRepository


def _open_repo(tmp_path: Path) -> SQLiteRepository:
    repo = SQLiteRepository(str(tmp_path / "facecore.db"), InMemoryKeyProvider())
    repo.initialize()
    return repo


def _manifest() -> object:
    from facecore.contracts.migration import ModelMigrationManifest

    return ModelMigrationManifest(
        embedder_artifact_hash="0" * 64,
        detector_generation="yunet-2023mar",
        preprocessing_generation="sface-112-rgb",
        tensor_layout="NCHW",
        normalization_contract="scale=1/128;mean=127.5;std=128",
        embedding_dimension=128,
        numerical_precision="fp32",
        quantization_type="none",
        execution_runtime="onnxruntime-cpu-arm64",
    )  # type: ignore[arg-type]


def test_fresh_store_defaults_to_g1(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    assert repo.get_current_generation() == "G1"


def test_unmigrated_writes_stamp_g1(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    manager = LifecycleManager(repo)
    result = manager.add_identity("person-001", "Test", b"e" * 64, b"x" * 64)
    assert result.template_id is not None
    con = repo.connection
    assert con is not None
    row = con.execute(
        "SELECT generation_id FROM face_templates WHERE id = ?",
        (result.template_id,),
    ).fetchone()
    assert row is not None and row[0] == "G1"


def test_post_migration_writes_stamp_new_generation(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    LifecycleManager(repo).add_identity(
        "person-001", "Test", b"e" * 64, b"x" * 64
    )
    migration = ModelMigrationManager(repo, current_generation="G1")
    report = migration.migrate_generation(_manifest(), "G2")  # type: ignore[arg-type]
    assert report.to_generation == "G2"
    assert repo.get_current_generation() == "G2"
    manager = LifecycleManager(repo)
    result = manager.add_identity("person-002", "Test2", b"f" * 64, b"y" * 64)
    assert result.template_id is not None
    con = repo.connection
    assert con is not None
    row = con.execute(
        "SELECT generation_id FROM face_templates WHERE id = ?",
        (result.template_id,),
    ).fetchone()
    assert row is not None and row[0] == "G2"


def test_post_migration_reenroll_stamps_new_generation(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    LifecycleManager(repo).add_identity(
        "person-001", "Test", b"e" * 64, b"x" * 64
    )
    migration = ModelMigrationManager(repo, current_generation="G1")
    migration.migrate_generation(_manifest(), "G2")  # type: ignore[arg-type]
    manager = LifecycleManager(repo)
    result = manager.re_enroll("person-001", b"n" * 64, b"m" * 64)
    assert result.template_id is not None
    con = repo.connection
    assert con is not None
    row = con.execute(
        "SELECT generation_id FROM face_templates WHERE id = ?",
        (result.template_id,),
    ).fetchone()
    assert row is not None and row[0] == "G2"


def test_generation_persists_across_reopen(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    LifecycleManager(repo).add_identity(
        "person-001", "Test", b"e" * 64, b"x" * 64
    )
    ModelMigrationManager(repo, current_generation="G1").migrate_generation(
        _manifest(),  # type: ignore[arg-type]
        "G2",
    )
    reopened = SQLiteRepository(
        str(tmp_path / "facecore.db"), InMemoryKeyProvider()
    )
    reopened.initialize()
    assert reopened.get_current_generation() == "G2"


def test_candidate_pipeline_defaults_to_store_generation(tmp_path: Path) -> None:
    from facecore.governance.candidate import CandidatePipeline

    repo = _open_repo(tmp_path)
    LifecycleManager(repo).add_identity(
        "person-001", "Test", b"e" * 64, b"x" * 64
    )
    ModelMigrationManager(repo, current_generation="G1").migrate_generation(
        _manifest(),  # type: ignore[arg-type]
        "G2",
    )
    pipeline = CandidatePipeline(repo)
    assert pipeline.generation_id == "G2"


def test_explicit_generation_overrides_store_default(tmp_path: Path) -> None:
    from facecore.governance.candidate import CandidatePipeline

    repo = _open_repo(tmp_path)
    assert LifecycleManager(repo, generation_id="G9") is not None
    assert (
        CandidatePipeline(repo, generation_id="G9").generation_id == "G9"
    )
    with pytest.raises(ValueError, match="generation_id"):
        LifecycleManager(repo, generation_id="")
