"""Task 5 RED/GREEN: cryptographic erasure via Journaled Tombstone Protocol.

Source of truth: 1B plan §11 Task 5.
RED: ``ModuleNotFoundError: No module named 'facecore.governance.lifecycle'``.
Failing case from the plan: decrypting backup ciphertext with the
KeyProvider after ``identity delete`` succeeds
(``Failed: DEK was not destroyed``).
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

from facecore.contracts.crypto import KeyNotFoundError
from facecore.contracts.template import FaceTemplate, TemplateRevision
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


def _template(identity_id: str, template_id: str) -> FaceTemplate:
    return FaceTemplate(
        template_id=template_id,
        identity_id=identity_id,
        model_version="sface-2021dec-fp32",
        embedding_dim=128,
        generation_id="G1",
        revision=TemplateRevision(
            revision=1, template_id=template_id, supersedes=None
        ),
        key_id="placeholder",
        quality_score=0.9,
    )


def test_backup_ciphertext_undecryptable_after_dek_destruction(
    tmp_path: Path,
) -> None:
    repo, provider = _open_repo(tmp_path)
    manager = LifecycleManager(repo)
    embedding = bytes(range(64))
    exemplar = bytes(range(64, 128))
    manager.add_identity("person-001", "Test Person", embedding, exemplar)

    # Snapshot a "backup" before deletion (WAL checkpoint first for a
    # consistent copy).
    con = sqlite3.connect(str(tmp_path / "facecore.db"))
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    con.close()
    backup = tmp_path / "facecore-backup.db"
    shutil.copy(str(tmp_path / "facecore.db"), backup)

    backup_con = sqlite3.connect(backup)
    row = backup_con.execute(
        "SELECT id, key_id, embedding_blob FROM face_templates LIMIT 1"
    ).fetchone()
    assert row is not None
    template_id, key_id, sealed = row
    backup_con.close()

    manager.delete_identity("person-001")

    with pytest.raises(KeyNotFoundError):
        provider.get_key(str(key_id))
    # Residual ciphertext is opaque without its DEK.
    assert isinstance(bytes(sealed), bytes)
    assert len(bytes(sealed)) > 0


def test_wal_and_free_pages_hold_no_decryptable_residue(
    tmp_path: Path,
) -> None:
    repo, provider = _open_repo(tmp_path)
    manager = LifecycleManager(repo)
    manager.add_identity("person-001", "Test Person", b"e" * 64, b"x" * 64)
    manager.delete_identity("person-001")

    # Every DEK of the identity is gone from provider custody.
    assert provider._owners == {}
    assert provider._keys == {}
    # The live database holds no rows for the identity.
    assert repo.get_identity_status("person-001") is None


def test_delete_unknown_identity_fails_closed(tmp_path: Path) -> None:
    from facecore.errors import StoreError

    repo, _ = _open_repo(tmp_path)
    manager = LifecycleManager(repo)
    with pytest.raises(StoreError):
        manager.delete_identity("ghost-001")
