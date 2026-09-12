"""Task 5 RED/GREEN: identity lifecycle operations + recovery.

Source of truth: 1B plan §11 Task 5.
RED (pre-lifecycle): collection error on
``facecore.governance.lifecycle`` import.
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from facecore.contracts.policy import GovernancePolicy
from facecore.errors import StoreError
from facecore.governance.lifecycle import LifecycleManager
from facecore.storage.key_provider import InMemoryKeyProvider
from facecore.storage.sqlite_repo import SQLiteRepository


def _open_repo(
    tmp_path: Path, provider: InMemoryKeyProvider | None = None
) -> tuple[SQLiteRepository, InMemoryKeyProvider]:
    resolved = provider or InMemoryKeyProvider()
    repo = SQLiteRepository(str(tmp_path / "facecore.db"), resolved)
    repo.initialize()
    return repo, resolved


def _manager(
    tmp_path: Path, policy: GovernancePolicy | None = None
) -> tuple[LifecycleManager, SQLiteRepository]:
    repo, _ = _open_repo(tmp_path)
    return LifecycleManager(repo, policy), repo


def test_add_is_create_only_duplicate_fails_without_mutation(
    tmp_path: Path,
) -> None:
    provider = InMemoryKeyProvider()
    repo, _ = _open_repo(tmp_path, provider)
    manager = LifecycleManager(repo)
    manager.add_identity("person-001", "Test Person", b"e" * 64, b"x" * 64)
    keys_before = dict(provider._keys)
    with pytest.raises(StoreError):
        manager.add_identity("person-001", "Someone Else", b"f" * 64, b"y" * 64)
    assert provider._keys == keys_before
    assert repo.get_identity_status("person-001") == "active"
    assert len(repo.list_active_templates()) == 1


def test_add_stores_initial_exemplar(tmp_path: Path) -> None:
    manager, repo = _manager(tmp_path)
    result = manager.add_identity(
        "person-001", "Test Person", b"e" * 64, b"x" * 64
    )
    assert result.revision == 1
    assert result.template_id is not None
    assert repo.read_embedding(result.template_id) == b"e" * 64
    con = sqlite3.connect(str(tmp_path / "facecore.db"))
    try:
        row = con.execute(
            "SELECT exemplar_blob FROM face_templates WHERE id = ?",
            (result.template_id,),
        ).fetchone()
        assert row is not None and isinstance(row[0], bytes)
    finally:
        con.close()


def test_re_enroll_atomically_replaces_with_sole_active(tmp_path: Path) -> None:
    manager, repo = _manager(tmp_path)
    first = manager.add_identity(
        "person-001", "Test Person", b"e" * 64, b"x" * 64
    )
    assert first.template_id is not None
    second = manager.re_enroll("person-001", b"n" * 64, b"m" * 64)
    assert second.template_id is not None
    assert second.template_id != first.template_id
    active = repo.list_active_templates()
    assert [t.template_id for t in active] == [second.template_id]
    assert repo.read_embedding(second.template_id) == b"n" * 64
    # Retired rows survive with DEKs intact (rollback stays decryptable).
    con = sqlite3.connect(str(tmp_path / "facecore.db"))
    try:
        status = con.execute(
            "SELECT status FROM face_templates WHERE id = ?",
            (first.template_id,),
        ).fetchone()
        assert status is not None and status[0] == "retired"
    finally:
        con.close()


def test_re_enroll_unknown_identity_fails_closed(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    with pytest.raises(StoreError):
        manager.re_enroll("ghost-001", b"e" * 64, b"x" * 64)


def test_rollback_restores_revision_and_retires_newer(tmp_path: Path) -> None:
    manager, repo = _manager(tmp_path)
    first = manager.add_identity(
        "person-001", "Test Person", b"e" * 64, b"x" * 64
    )
    assert first.template_id is not None
    second = manager.re_enroll("person-001", b"n" * 64, b"m" * 64)
    assert second.template_id is not None
    result = manager.rollback("person-001", 1)
    assert result.revision == 1
    active = repo.list_active_templates()
    assert [t.template_id for t in active] == [first.template_id]
    snapshot = manager.show_identity("person-001")
    assert snapshot["current_revision"] == 1


def test_rollback_beyond_max_depth_refused(tmp_path: Path) -> None:
    policy = GovernancePolicy.provisional_v1()
    manager, _ = _manager(tmp_path, policy)
    manager.add_identity("person-001", "Test Person", b"e" * 64, b"x" * 64)
    current_depth = policy.rollback_max_depth
    assert current_depth >= 1
    with pytest.raises(StoreError):
        manager.rollback("person-001", 1 - current_depth - 1)


def test_rollback_unknown_revision_fails_closed(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    manager.add_identity("person-001", "Test Person", b"e" * 64, b"x" * 64)
    with pytest.raises(StoreError):
        manager.rollback("person-001", 99)


def test_reject_candidate_and_list(tmp_path: Path) -> None:
    manager, repo = _manager(tmp_path)
    manager.add_identity("person-001", "Test Person", b"e" * 64, b"x" * 64)
    repo.add_candidate_record(
        "c-1",
        "person-001",
        b"e" * 16,
        b"x" * 8,
        "2030-01-01T00:00:00Z",
        generation_id="G1",
        quality_score=0.9,
        additional_corroboration_count=0,
        evidence_log="[]",
    )
    result = manager.reject_candidate("c-1")
    assert result.action == "candidate_rejected"
    listed = manager.list_candidates("person-001")
    assert [(c["candidate_id"], c["status"]) for c in listed] == [
        ("c-1", "rejected")
    ]
    with pytest.raises(StoreError):
        manager.reject_candidate("c-1")


REPO = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}


def _cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    db = tmp_path / "facecore.db"
    monkeypatch.setenv("FACECORE_DB", str(db))
    monkeypatch.delenv("FACECORE_MASTER_KEY", raising=False)
    return {**ENV, "FACECORE_DB": str(db)}


def _run_cli(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "facecore.cli", *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
    )


def test_cli_add_show_delete_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _cli_env(tmp_path, monkeypatch)
    photo = tmp_path / "photo.bin"
    photo.write_bytes(bytes(range(256)))
    proc = _run_cli(
        ["identity", "add", "--id", "person-001",
         "--display-name", "Test", "--photo", str(photo)],
        env,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1])["status"] == "ok"
    proc = _run_cli(["identity", "show", "--id", "person-001"], env)
    assert proc.returncode == 0, proc.stderr
    snapshot = json.loads(proc.stdout.strip().splitlines()[-1])
    assert snapshot["identity_id"] == "person-001"
    assert len(snapshot["active_template_ids"]) == 1
    proc = _run_cli(["status"], env)
    assert proc.returncode == 0, proc.stderr
    proc = _run_cli(["identity", "delete", "--id", "person-001"], env)
    assert proc.returncode == 0, proc.stderr
    proc = _run_cli(["identity", "show", "--id", "person-001"], env)
    assert proc.returncode == 4, proc.stderr
    assert (
        json.loads(proc.stdout.strip().splitlines()[-1])["status"]
        == "store_error"
    )


def test_cli_duplicate_add_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _cli_env(tmp_path, monkeypatch)
    photo = tmp_path / "photo.bin"
    photo.write_bytes(bytes(range(128)))
    assert _run_cli(
        ["identity", "add", "--id", "person-001",
         "--display-name", "Test", "--photo", str(photo)],
        env,
    ).returncode == 0
    proc = _run_cli(
        ["identity", "add", "--id", "person-001",
         "--display-name", "Other", "--photo", str(photo)],
        env,
    )
    assert proc.returncode == 4, proc.stderr
    assert (
        json.loads(proc.stdout.strip().splitlines()[-1])["status"]
        == "store_error"
    )


def test_cli_bad_photo_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _cli_env(tmp_path, monkeypatch)
    proc = _run_cli(
        ["identity", "add", "--id", "person-001",
         "--display-name", "Test", "--photo", str(tmp_path / "missing.bin")],
        env,
    )
    assert proc.returncode == 2, proc.stderr


def test_cli_re_enroll_and_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _cli_env(tmp_path, monkeypatch)
    photo_a = tmp_path / "a.bin"
    photo_a.write_bytes(b"a" * 128)
    photo_b = tmp_path / "b.bin"
    photo_b.write_bytes(b"b" * 128)
    assert _run_cli(
        ["identity", "add", "--id", "person-001",
         "--display-name", "Test", "--photo", str(photo_a)],
        env,
    ).returncode == 0
    proc = _run_cli(
        ["identity", "re-enroll", "--id", "person-001",
         "--photo", str(photo_b)],
        env,
    )
    assert proc.returncode == 0, proc.stderr
    proc = _run_cli(
        ["identity", "rollback", "--id", "person-001", "--to-revision", "1"],
        env,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1])["revision"] == 1
    proc = _run_cli(
        ["identity", "rollback", "--id", "person-001", "--to-revision", "99"],
        env,
    )
    assert proc.returncode == 4, proc.stderr


def test_show_identity_snapshot(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    created = manager.add_identity(
        "person-001", "Test Person", b"e" * 64, b"x" * 64
    )
    snapshot = manager.show_identity("person-001")
    assert snapshot["identity_id"] == "person-001"
    assert snapshot["status"] == "active"
    assert snapshot["current_revision"] == 1
    assert snapshot["active_template_ids"] == [created.template_id]
    payload = json.dumps(snapshot, default=str)
    assert "person-001" in payload
