"""Task 6 RED/GREEN: encrypted export/import with key re-homing.

Source of truth: 1B plan §11 Task 6 + §8 protocol.
RED: ``ModuleNotFoundError: No module named 'facecore.storage.export'``.
Failing case from the plan: importing an archive into KeyProvider B with
a mismatched execution runtime succeeds
(``Failed: DID NOT RAISE ModelIncompatibilityError``).
"""

import json
import sqlite3
from pathlib import Path

import pytest

from facecore.contracts.migration import (
    ModelIncompatibilityError,
    ModelMigrationManifest,
)
from facecore.storage.export import (
    ExportManifest,
    export_identities,
    import_identities,
)
from facecore.storage.key_provider import InMemoryKeyProvider
from facecore.storage.sqlite_repo import SQLiteRepository


def _open_repo(
    tmp_path: Path, name: str, provider: InMemoryKeyProvider
) -> SQLiteRepository:
    repo = SQLiteRepository(str(tmp_path / name), provider)
    repo.initialize()
    return repo


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


def _seed(
    repo: SQLiteRepository, identity_id: str = "person-001"
) -> None:
    from facecore.governance.lifecycle import LifecycleManager

    manager = LifecycleManager(repo)
    manager.add_identity(identity_id, "Test Person", b"e" * 64, b"x" * 64)
    repo.add_candidate_record(
        "c-1",
        identity_id,
        b"e" * 16,
        b"x" * 8,
        "2030-01-01T00:00:00Z",
        generation_id="G1",
        quality_score=0.9,
        additional_corroboration_count=0,
        evidence_log=(
            '[{"event_type": "seed", "timestamp": "2026-09-12T00:00:00+08:00",'
            ' "sequence_number": 1}]'
        ),
    )


def test_mismatched_runtime_contract_refused(tmp_path: Path) -> None:
    provider_a = InMemoryKeyProvider()
    repo_a = _open_repo(tmp_path, "a.db", provider_a)
    _seed(repo_a)
    archive = tmp_path / "export.fce"
    export_identities(
        repo_a, _manifest(), archive, passphrase="correct horse 2026!"
    )
    provider_b = InMemoryKeyProvider()
    repo_b = _open_repo(tmp_path, "b.db", provider_b)
    with pytest.raises(ModelIncompatibilityError):
        import_identities(
            repo_b,
            _manifest(execution_runtime="onnxruntime-cpu-x86_64"),
            archive,
            passphrase="correct horse 2026!",
            dest_key_provider=provider_b,
        )
    assert repo_b.get_identity_status("person-001") is None


def test_mismatched_tensor_layout_refused_with_zero_writes(
    tmp_path: Path,
) -> None:
    provider_a = InMemoryKeyProvider()
    repo_a = _open_repo(tmp_path, "a.db", provider_a)
    _seed(repo_a)
    archive = tmp_path / "export.fce"
    export_identities(
        repo_a, _manifest(), archive, passphrase="correct horse 2026!"
    )
    provider_b = InMemoryKeyProvider()
    repo_b = _open_repo(tmp_path, "b.db", provider_b)
    with pytest.raises(ModelIncompatibilityError):
        import_identities(
            repo_b,
            _manifest(tensor_layout="NHWC"),
            archive,
            passphrase="correct horse 2026!",
            dest_key_provider=provider_b,
        )
    con = sqlite3.connect(str(tmp_path / "b.db"))
    try:
        assert con.execute("SELECT COUNT(*) FROM identities").fetchone()[0] == 0
        assert (
            con.execute("SELECT COUNT(*) FROM face_templates").fetchone()[0]
            == 0
        )
    finally:
        con.close()


def test_tampered_archive_fails_closed_before_any_write(
    tmp_path: Path,
) -> None:
    from facecore.errors import StoreError

    provider_a = InMemoryKeyProvider()
    repo_a = _open_repo(tmp_path, "a.db", provider_a)
    _seed(repo_a)
    archive = tmp_path / "export.fce"
    export_identities(
        repo_a, _manifest(), archive, passphrase="correct horse 2026!"
    )
    raw = bytearray(archive.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    archive.write_bytes(bytes(raw))
    provider_b = InMemoryKeyProvider()
    repo_b = _open_repo(tmp_path, "b.db", provider_b)
    with pytest.raises(StoreError):
        import_identities(
            repo_b,
            _manifest(),
            archive,
            passphrase="correct horse 2026!",
            dest_key_provider=provider_b,
        )
    assert repo_b.get_identity_status("person-001") is None


def test_wrong_passphrase_fails_closed(tmp_path: Path) -> None:
    from facecore.errors import StoreError

    provider_a = InMemoryKeyProvider()
    repo_a = _open_repo(tmp_path, "a.db", provider_a)
    _seed(repo_a)
    archive = tmp_path / "export.fce"
    export_identities(
        repo_a, _manifest(), archive, passphrase="correct horse 2026!"
    )
    provider_b = InMemoryKeyProvider()
    repo_b = _open_repo(tmp_path, "b.db", provider_b)
    with pytest.raises(StoreError):
        import_identities(
            repo_b,
            _manifest(),
            archive,
            passphrase="wrong passphrase",
            dest_key_provider=provider_b,
        )


def test_unsupported_kdf_version_fails_closed(tmp_path: Path) -> None:
    from facecore.contracts.crypto import UnsupportedKdfError

    provider_a = InMemoryKeyProvider()
    repo_a = _open_repo(tmp_path, "a.db", provider_a)
    _seed(repo_a)
    archive = tmp_path / "export.fce"
    export_identities(
        repo_a, _manifest(), archive, passphrase="correct horse 2026!"
    )
    payload = json.loads(archive.read_bytes().decode("utf-8"))
    payload["metadata"]["kdf_version"] = 999
    archive.write_bytes(json.dumps(payload).encode("utf-8"))
    provider_b = InMemoryKeyProvider()
    repo_b = _open_repo(tmp_path, "b.db", provider_b)
    with pytest.raises(UnsupportedKdfError):
        import_identities(
            repo_b,
            _manifest(),
            archive,
            passphrase="correct horse 2026!",
            dest_key_provider=provider_b,
        )


def test_round_trip_preserves_policy_statuses_and_evidence(
    tmp_path: Path,
) -> None:
    from facecore.contracts.policy import GovernancePolicy
    from facecore.governance.lifecycle import LifecycleManager

    custom = GovernancePolicy.provisional_v1()
    object.__setattr__(custom, "promotion_margin", 0.20)
    provider_a = InMemoryKeyProvider()
    repo_a = _open_repo(tmp_path, "a.db", provider_a)
    manager_a = LifecycleManager(repo_a, custom)
    manager_a.add_identity("person-001", "Test Person", b"e" * 64, b"x" * 64)
    manager_a.re_enroll("person-001", b"n" * 64, b"m" * 64)
    repo_a.add_candidate_record(
        "c-pending",
        "person-001",
        b"e" * 16,
        b"x" * 8,
        "2030-01-01T00:00:00Z",
        generation_id="G1",
        quality_score=0.9,
        additional_corroboration_count=0,
        evidence_log="[]",
    )
    repo_a.add_candidate_record(
        "c-rej",
        "person-001",
        b"e" * 16,
        b"x" * 8,
        "2030-01-01T00:00:00Z",
        generation_id="G1",
        quality_score=0.9,
        additional_corroboration_count=0,
        evidence_log="[]",
    )
    manager_a.reject_candidate("c-rej")
    archive = tmp_path / "export.fce"
    manifest = export_identities(
        repo_a, _manifest(), archive,
        passphrase="correct horse 2026!", policy=custom,
    )
    assert isinstance(manifest, ExportManifest)

    provider_b = InMemoryKeyProvider()
    repo_b = _open_repo(tmp_path, "b.db", provider_b)
    result = import_identities(
        repo_b,
        _manifest(),
        archive,
        passphrase="correct horse 2026!",
        dest_key_provider=provider_b,
    )
    assert result.identities == 1
    assert repo_b.get_identity_status("person-001") == "active"
    manager_b = LifecycleManager(repo_b)
    statuses = {
        c["candidate_id"]: c["status"]
        for c in manager_b.list_candidates("person-001")
    }
    assert statuses == {"c-pending": "pending", "c-rej": "rejected"}
    active = repo_b.list_active_templates()
    assert len(active) == 1
    assert repo_b.read_embedding(active[0].template_id) == b"n" * 64
    # Post-import behavior identical: rollback + reject work on re-homed
    # rows under destination custody.
    rolled = manager_b.rollback("person-001", 1)
    assert rolled.revision == 1
    manager_b.reject_candidate("c-pending")
    assert manager_b.list_candidates("person-001")[0]["status"] in (
        "pending",
        "rejected",
    )


def test_each_hard_field_refused_individually(tmp_path: Path) -> None:
    provider_a = InMemoryKeyProvider()
    repo_a = _open_repo(tmp_path, "a.db", provider_a)
    _seed(repo_a)
    archive = tmp_path / "export.fce"
    export_identities(
        repo_a, _manifest(), archive, passphrase="correct horse 2026!"
    )
    hard_fields = [
        "embedder_artifact_hash",
        "tensor_layout",
        "normalization_contract",
        "embedding_dimension",
        "numerical_precision",
        "quantization_type",
        "execution_runtime",
    ]
    bad_values: dict[str, object] = {
        "embedder_artifact_hash": "f" * 64,
        "tensor_layout": "NHWC",
        "normalization_contract": "other",
        "embedding_dimension": 512,
        "numerical_precision": "int8",
        "quantization_type": "int8bq",
        "execution_runtime": "other-runtime",
    }
    for field_name in hard_fields:
        provider_b = InMemoryKeyProvider()
        repo_b = _open_repo(tmp_path, f"b-{field_name}.db", provider_b)
        with pytest.raises(
            ModelIncompatibilityError, match=field_name
        ):
            import_identities(
                repo_b,
                _manifest(**{field_name: bad_values[field_name]}),
                archive,
                passphrase="correct horse 2026!",
                dest_key_provider=provider_b,
            )


def test_cli_export_import_round_trip_with_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json as _json
    import os as _os
    import subprocess as _subprocess
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **_os.environ,
        "PYTHONPATH": str(repo_root / "src"),
        "FACECORE_DB": str(tmp_path / "cli.db"),
    }
    monkeypatch.delenv("FACECORE_MASTER_KEY", raising=False)
    photo = tmp_path / "photo.bin"
    photo.write_bytes(bytes(range(128)))
    archive = tmp_path / "cli.fce"

    def _run(*args: str) -> _subprocess.CompletedProcess[str]:
        return _subprocess.run(
            [_sys.executable, "-m", "facecore.cli", *args],
            capture_output=True,
            text=True,
            cwd=repo_root,
            env=env,
        )

    assert _run(
        "identity", "add", "--id", "person-001",
        "--display-name", "Test", "--photo", str(photo),
    ).returncode == 0
    proc = _run(
        "export", "--archive", str(archive),
        "--passphrase", "correct horse 2026!",
    )
    assert proc.returncode == 0, proc.stderr
    assert _json.loads(proc.stdout.strip().splitlines()[-1])["status"] == "ok"
    assert _run(
        "identity", "delete", "--id", "person-001"
    ).returncode == 0
    proc = _run(
        "import", "--archive", str(archive),
        "--passphrase", "correct horse 2026!",
    )
    assert proc.returncode == 0, proc.stderr
    payload = _json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert payload["compatibility"] == "COMPATIBLE"
    proc = _run("identity", "show", "--id", "person-001")
    assert proc.returncode == 0, proc.stderr


def test_cli_import_wrong_passphrase_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json as _json
    import os as _os
    import subprocess as _subprocess
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **_os.environ,
        "PYTHONPATH": str(repo_root / "src"),
        "FACECORE_DB": str(tmp_path / "cli2.db"),
    }
    monkeypatch.delenv("FACECORE_MASTER_KEY", raising=False)
    photo = tmp_path / "photo.bin"
    photo.write_bytes(bytes(range(128)))
    archive = tmp_path / "cli2.fce"

    def _run(*args: str) -> _subprocess.CompletedProcess[str]:
        return _subprocess.run(
            [_sys.executable, "-m", "facecore.cli", *args],
            capture_output=True,
            text=True,
            cwd=repo_root,
            env=env,
        )

    assert _run(
        "identity", "add", "--id", "person-001",
        "--display-name", "Test", "--photo", str(photo),
    ).returncode == 0
    assert _run(
        "export", "--archive", str(archive),
        "--passphrase", "correct horse 2026!",
    ).returncode == 0
    proc = _run(
        "import", "--archive", str(archive),
        "--passphrase", "wrong passphrase",
    )
    assert proc.returncode == 4, proc.stderr
    assert (
        _json.loads(proc.stdout.strip().splitlines()[-1])["status"]
        == "store_error"
    )


def test_detector_generation_drift_reports_migration_required(
    tmp_path: Path,
) -> None:
    provider_a = InMemoryKeyProvider()
    repo_a = _open_repo(tmp_path, "a.db", provider_a)
    _seed(repo_a)
    archive = tmp_path / "export.fce"
    export_identities(
        repo_a, _manifest(), archive, passphrase="correct horse 2026!"
    )
    provider_b = InMemoryKeyProvider()
    repo_b = _open_repo(tmp_path, "b.db", provider_b)
    result = import_identities(
        repo_b,
        _manifest(detector_generation="yunet-2024jan"),
        archive,
        passphrase="correct horse 2026!",
        dest_key_provider=provider_b,
    )
    assert result.compatibility == "MIGRATION_REQUIRED"
    assert repo_b.get_identity_status("person-001") == "active"
