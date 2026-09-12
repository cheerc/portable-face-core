"""Task 2 RED/GREEN: retention stress (10k events + 20 revisions + 5 backups).

RED: ``ModuleNotFoundError: No module named 'facecore.storage.sqlite_repo'``
(via the ``prune_backups`` import path exercised below).
"""

import sqlite3
from pathlib import Path

from facecore.contracts.crypto import KeyProviderProtocol
from facecore.contracts.template import FaceTemplate, TemplateRevision
from facecore.storage.key_provider import InMemoryKeyProvider
from facecore.storage.sqlite_repo import (
    SQLiteRepository,
    prune_backups,
    trim_match_events,
    trim_revisions,
)


def _open_repo(tmp_path: Path) -> SQLiteRepository:
    provider: KeyProviderProtocol = InMemoryKeyProvider()
    return SQLiteRepository(str(tmp_path / "facecore.db"), provider)


def _template(identity: str, revision: int) -> FaceTemplate:
    template_id = f"t-{identity}-{revision}"
    return FaceTemplate(
        template_id=template_id,
        identity_id=identity,
        model_version="sface-2021dec-fp32",
        embedding_dim=4,
        revision=TemplateRevision(
            revision=revision, template_id=template_id, supersedes=None
        ),
    )


def test_event_revision_backup_ceilings(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity(
        "person-001",
        "Test Person",
        _template("person-001", 1),
        b"e" * 16,
        b"x" * 8,
        "t-person-001-1",
    )
    for n in range(10050):
        repo.record_match_event(
            f"evt-{n:05d}",
            "2026-09-12T00:00:00+08:00",
            n,
            "unknown",
            0.1,
            None,
        )
    assert repo.connection is not None
    assert trim_match_events(repo.connection, 10000) == 50
    con = repo.connection
    count_row = con.execute("SELECT COUNT(*) FROM match_events").fetchone()
    assert count_row is not None and int(count_row[0]) == 10000

    for revision in range(2, 26):
        repo.append_revision(
            "person-001", _template("person-001", revision), b"e" * 16, b"x" * 8
        )
    assert trim_revisions(con, "person-001", 20) == 5
    assert repo.revision_count("person-001") == 20

    backups = [tmp_path / f"backup-{n:02d}.db" for n in range(6)]
    for path in backups:
        path.write_bytes(b"0" * 8)
    pruned = prune_backups(tmp_path, "backup-*.db", 5)
    assert pruned == [str(backups[0])]
    assert len(list(tmp_path.glob("backup-*.db"))) == 5
    assert (tmp_path / "facecore.db").exists()

    con2 = sqlite3.connect(str(tmp_path / "facecore.db"))
    try:
        assert con2.execute("SELECT COUNT(*) FROM face_templates").fetchone()[0] >= 1
    finally:
        con2.close()


def test_retention_helpers_use_immediate_write_transactions(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity(
        "person-001",
        "Test Person",
        _template("person-001", 1),
        b"e" * 16,
        b"x" * 8,
        "t-person-001-1",
    )
    repo.record_match_event(
        "evt-1", "2026-09-12T00:00:00+08:00", 1, "unknown", 0.1, None
    )
    repo.record_match_event(
        "evt-2", "2026-09-12T00:00:01+08:00", 2, "unknown", 0.1, None
    )
    assert repo.connection is not None
    statements: list[str] = []
    repo.connection.set_trace_callback(statements.append)
    trim_match_events(repo.connection, 1)
    assert any(
        statement.strip().upper() == "BEGIN IMMEDIATE"
        for statement in statements
    )

    repo.append_revision(
        "person-001", _template("person-001", 2), b"e" * 16, b"x" * 8
    )
    statements.clear()
    trim_revisions(repo.connection, "person-001", 1)
    assert any(
        statement.strip().upper() == "BEGIN IMMEDIATE"
        for statement in statements
    )
