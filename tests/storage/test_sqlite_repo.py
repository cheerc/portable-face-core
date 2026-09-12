"""Task 2 RED/GREEN: SQLite repo + journaled tombstones (S1B §§4-5, plan §7).

RED: ``ModuleNotFoundError: No module named 'facecore.storage.sqlite_repo'``.
"""

import sqlite3
from pathlib import Path

import pytest

from facecore.contracts.crypto import KeyProviderProtocol
from facecore.contracts.template import FaceTemplate, TemplateRevision
from facecore.errors import StoreError
from facecore.storage.key_provider import InMemoryKeyProvider
from facecore.storage.sqlite_repo import SQLiteRepository


def _open_repo(
    tmp_path: Path, provider: KeyProviderProtocol | None = None
) -> SQLiteRepository:
    resolved: KeyProviderProtocol = provider or InMemoryKeyProvider()
    return SQLiteRepository(str(tmp_path / "facecore.db"), resolved)


def _template(identity: str = "person-001", template: str = "t-1") -> FaceTemplate:
    return FaceTemplate(
        template_id=template,
        identity_id=identity,
        model_version="sface-2021dec-fp32",
        embedding_dim=4,
        revision=TemplateRevision(revision=1, template_id=template, supersedes=None),
    )


def _payload(template_id: str) -> tuple[bytes, bytes, str]:
    return (b"e" * 16, b"x" * 32, template_id)


def test_enroll_identify_and_key_id_only_in_sqlite(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    active = repo.list_active_templates()
    assert [t.template_id for t in active] == ["t-1"]
    con = sqlite3.connect(str(tmp_path / "facecore.db"))
    try:
        cols = {row[1] for row in con.execute("PRAGMA table_info(face_templates)")}
        assert "encrypted_embedding" not in cols
        assert "encrypted_dek" not in cols
        row = con.execute(
            "SELECT key_id, embedding_blob, exemplar_blob FROM face_templates"
        ).fetchone()
        assert row[0].startswith("key-")
        assert isinstance(row[1], bytes) and isinstance(row[2], bytes)
    finally:
        con.close()


def test_tombstone_reconciliation_after_crash(tmp_path: Path) -> None:
    """Plan RED case: crash after key destruction leaves orphaned rows."""
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    repo.begin_delete_identity("person-001")
    key_ids = repo.tombstone_key_ids("person-001")
    assert len(key_ids) == 1
    # Simulate crash after key destruction but before Step 3 purge.
    repo.key_provider.destroy_key(key_ids[0])
    crashed = _open_repo(tmp_path, repo.key_provider)
    crashed.initialize()
    assert crashed.get_identity_status("person-001") is None
    assert crashed.list_active_templates() == []
    with pytest.raises(StoreError):
        crashed.key_provider.get_key(key_ids[0])


def test_crash_before_key_destruction_recovers(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    repo.begin_delete_identity("person-001")
    key_ids = repo.tombstone_key_ids("person-001")
    recovered = _open_repo(tmp_path, repo.key_provider)
    recovered.initialize()
    assert recovered.get_identity_status("person-001") is None
    with pytest.raises(StoreError):
        recovered.key_provider.get_key(key_ids[0])


def test_crash_mid_batch_destruction_recovers(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    repo.add_candidate_record(
        "c-1", "person-001", b"e" * 16, b"x" * 8, expires_at="2030-01-01T00:00:00Z"
    )
    repo.begin_delete_identity("person-001")
    key_ids = repo.tombstone_key_ids("person-001")
    assert len(key_ids) == 2
    repo.key_provider.destroy_key(key_ids[0])
    recovered = _open_repo(tmp_path, repo.key_provider)
    recovered.initialize()
    assert recovered.get_identity_status("person-001") is None
    for key_id in key_ids:
        with pytest.raises(StoreError):
            recovered.key_provider.get_key(key_id)


def test_crash_after_tombstone_key_destroyed_purges(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    repo.begin_delete_identity("person-001")
    key_ids = repo.tombstone_key_ids("person-001")
    for key_id in key_ids:
        repo.key_provider.destroy_key(key_id)
    repo.mark_tombstone_key_destroyed("person-001")
    recovered = _open_repo(tmp_path, repo.key_provider)
    recovered.initialize()
    assert recovered.get_identity_status("person-001") is None
    assert recovered.list_active_templates() == []


def test_checkpoint_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity(
        "person-001", "Test Person", _template(), *_payload("t-1")
    )
    monkeypatch.setattr(
        "facecore.storage.sqlite_repo.checkpoint_truncate",
        lambda con: (_ for _ in ()).throw(StoreError("busy")),
    )
    with pytest.raises(StoreError):
        repo.delete_identity("person-001")


def test_corrupt_ciphertext_raises_store_error(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    con = sqlite3.connect(str(tmp_path / "facecore.db"))
    try:
        con.execute("UPDATE face_templates SET embedding_blob = zeroblob(64)")
        con.commit()
    finally:
        con.close()
    with pytest.raises(StoreError):
        repo.read_embedding("t-1")


def test_match_event_anonymized_on_delete(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    repo.record_match_event(
        "evt-1",
        "2026-09-12T00:00:00+08:00",
        1,
        "matched",
        0.9,
        "person-001",
    )
    repo.delete_identity("person-001")
    con = sqlite3.connect(str(tmp_path / "facecore.db"))
    try:
        row = con.execute(
            "SELECT matched_identity_id FROM match_events WHERE id = 'evt-1'"
        ).fetchone()
        assert row[0] is None
    finally:
        con.close()


def test_schema_has_no_key_material_columns(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    con = sqlite3.connect(str(tmp_path / "facecore.db"))
    try:
        for table in ("face_templates", "candidate_templates"):
            cols = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
            assert "encrypted_dek" not in cols
            assert "encrypted_embedding" not in cols
            assert "key_id" in cols
    finally:
        con.close()
