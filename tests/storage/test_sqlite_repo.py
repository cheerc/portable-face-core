"""Task 2 RED/GREEN: SQLite repo + journaled tombstones (S1B §§4-5, plan §7).

RED: ``ModuleNotFoundError: No module named 'facecore.storage.sqlite_repo'``.
"""

import sqlite3
import threading
from pathlib import Path

import pytest

from facecore.contracts.crypto import KeyProviderProtocol, StoreCorruptionError
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


class _DestroyFailureProvider(InMemoryKeyProvider):
    """Real in-memory custody with an injected operational destroy failure."""

    def destroy_key(self, key_id: str) -> None:
        raise StoreError("key store unavailable")


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
    with pytest.raises(StoreCorruptionError) as caught:
        repo.read_embedding("t-1")
    assert caught.value.exit_code == 4


def test_destroy_failure_leaves_pending_tombstone_and_rows(
    tmp_path: Path,
) -> None:
    provider = _DestroyFailureProvider()
    repo = _open_repo(tmp_path, provider)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    with pytest.raises(StoreError, match="key store unavailable"):
        repo.delete_identity("person-001")
    con = repo.connection
    assert con is not None
    tombstone = con.execute(
        "SELECT status FROM deletion_tombstones WHERE target_id = ?",
        ("person-001",),
    ).fetchone()
    assert tombstone == ("pending_key_destruction",)
    assert con.execute(
        "SELECT COUNT(*) FROM face_templates WHERE identity_id = ?",
        ("person-001",),
    ).fetchone()[0] == 1
    assert repo.get_identity_status("person-001") == "deleted"


def test_deleted_identity_is_unreadable_after_step_one(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    repo.begin_delete_identity("person-001")
    assert repo.list_active_templates() == []
    with pytest.raises(StoreError, match="deleted or tombstoned"):
        repo.read_embedding("t-1")


def test_add_candidate_persists_governance_metadata(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    repo.add_candidate_record(
        "c-governed",
        "person-001",
        b"e" * 16,
        b"x" * 8,
        "2030-01-01T00:00:00Z",
        generation_id="G2",
        exemplar_crop_box=(2.0, 3.0, 90.0, 91.0),
        exemplar_landmarks=((1.0, 2.0), (3.0, 4.0)),
        quality_score=0.77,
        additional_corroboration_count=2,
        evidence_log='[{"event_type":"seed","sequence_number":9}]',
        created_at="2026-09-12T00:00:00+08:00",
    )
    con = repo.connection
    assert con is not None
    row = con.execute(
        "SELECT generation_id, exemplar_crop_box, exemplar_landmarks,"
        " quality_score, additional_corroboration_count, evidence_log, created_at"
        " FROM candidate_templates WHERE id = 'c-governed'"
    ).fetchone()
    assert row == (
        "G2",
        "[2.0, 3.0, 90.0, 91.0]",
        "[[1.0, 2.0], [3.0, 4.0]]",
        0.77,
        2,
        '[{"event_type":"seed","sequence_number":9}]',
        "2026-09-12T00:00:00+08:00",
    )


@pytest.mark.parametrize("geometry", ["not-json", "[1, 2, \"bad\", 4]", "[1, 2, 3]"])
def test_corrupt_persisted_geometry_is_structured_exit_4(
    tmp_path: Path, geometry: str
) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    template = FaceTemplate(
        template_id="t-geometry",
        identity_id="person-001",
        model_version="sface-2021dec-fp32",
        embedding_dim=4,
        revision=TemplateRevision(
            revision=1, template_id="t-geometry", supersedes=None
        ),
        exemplar_crop_box=(1.0, 2.0, 3.0, 4.0),
        exemplar_landmarks=((1.0, 2.0),),
    )
    repo.enroll_identity(
        "person-001", "Test Person", template, b"e" * 16, b"x" * 8, "t-geometry"
    )
    con = repo.connection
    assert con is not None
    con.execute(
        "UPDATE face_templates SET exemplar_crop_box = ? WHERE id = ?",
        (geometry, "t-geometry"),
    )
    con.commit()
    with pytest.raises(StoreCorruptionError) as caught:
        repo.list_active_templates()
    assert caught.value.exit_code == 4


@pytest.mark.parametrize("offset", [2, 3])
def test_nonzero_reserved_blob_header_is_structured_exit_4(
    tmp_path: Path, offset: int
) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    con = repo.connection
    assert con is not None
    row = con.execute(
        "SELECT embedding_blob FROM face_templates WHERE id = 't-1'"
    ).fetchone()
    assert row is not None
    corrupted = bytearray(row[0])
    corrupted[offset] = 1
    con.execute(
        "UPDATE face_templates SET embedding_blob = ? WHERE id = 't-1'",
        (bytes(corrupted),),
    )
    con.commit()
    with pytest.raises(StoreCorruptionError) as caught:
        repo.read_embedding("t-1")
    assert caught.value.exit_code == 4


def test_candidate_ciphertext_is_overwritten_before_delete(tmp_path: Path) -> None:
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    repo.add_candidate_record(
        "c-1", "person-001", b"e" * 16, b"x" * 8, "2030-01-01T00:00:00Z"
    )
    con = repo.connection
    assert con is not None
    original = con.execute(
        "SELECT embedding_blob, exemplar_blob FROM candidate_templates WHERE id = 'c-1'"
    ).fetchone()
    assert original is not None
    con.execute(
        "CREATE TABLE candidate_delete_observed "
        "(embedding_blob BLOB NOT NULL, exemplar_blob BLOB NOT NULL)"
    )
    con.execute(
        "CREATE TRIGGER observe_candidate_delete BEFORE DELETE ON candidate_templates "
        "BEGIN INSERT INTO candidate_delete_observed VALUES "
        "(OLD.embedding_blob, OLD.exemplar_blob); END"
    )
    repo.delete_identity("person-001")
    observed = con.execute(
        "SELECT embedding_blob, exemplar_blob FROM candidate_delete_observed"
    ).fetchone()
    assert observed is not None
    assert observed[0] != original[0]
    assert observed[1] != original[1]


def test_failed_duplicate_enrollment_leaves_key_custody_unchanged(
    tmp_path: Path,
) -> None:
    provider = InMemoryKeyProvider()
    repo = _open_repo(tmp_path, provider)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    before = len(provider._keys)
    with pytest.raises(sqlite3.IntegrityError):
        repo.enroll_identity("person-001", "Duplicate", _template(), *_payload("t-2"))
    assert len(provider._keys) == before


def test_failed_append_and_candidate_do_not_leak_keys(tmp_path: Path) -> None:
    provider = InMemoryKeyProvider()
    repo = _open_repo(tmp_path, provider)
    repo.initialize()
    repo.enroll_identity("person-001", "Test Person", _template(), *_payload("t-1"))
    before = len(provider._keys)
    with pytest.raises(StoreError, match="unknown identity"):
        repo.append_revision(
            "person-404", _template("person-404", "t-404"), b"e", b"x"
        )
    assert len(provider._keys) == before
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_candidate_record(
            "c-1", "person-001", b"e" * 16, b"x", "2030-01-01T00:00:00Z"
        )
        repo.add_candidate_record(
            "c-1", "person-001", b"e" * 16, b"x", "2030-01-01T00:00:00Z"
        )
    assert len(provider._keys) == before + 1


def test_deletion_serializes_against_inflight_append_without_key_leak(
    tmp_path: Path,
) -> None:
    provider = InMemoryKeyProvider()
    delete_repo = _open_repo(tmp_path, provider)
    delete_repo.initialize()
    delete_repo.enroll_identity(
        "person-001", "Test Person", _template(), *_payload("t-1")
    )
    seal_reached = threading.Event()
    step_one_done = threading.Event()
    append_done = threading.Event()
    append_result: list[str] = []
    append_error: list[BaseException] = []

    def append_worker() -> None:
        append_repo = _open_repo(tmp_path, provider)
        append_repo.initialize()
        original_seal = append_repo._seal
        first_call = True

        def hooked_seal(
            dek: bytes,
            plaintext: bytes,
            table: str,
            record: str,
            identity: str,
        ) -> tuple[bytes, bytes]:
            nonlocal first_call
            result = original_seal(dek, plaintext, table, record, identity)
            if first_call:
                first_call = False
                seal_reached.set()
                assert step_one_done.wait(timeout=5)
            return result

        append_repo._seal = hooked_seal  # type: ignore[method-assign]
        try:
            append_result.append(
                append_repo.append_revision(
                    "person-001",
                    _template("person-001", "t-2"),
                    b"e" * 16,
                    b"x" * 16,
                )
            )
        except BaseException as exc:
            append_error.append(exc)
        finally:
            append_done.set()

    worker = threading.Thread(target=append_worker)
    worker.start()
    assert seal_reached.wait(timeout=5)
    delete_repo.begin_delete_identity("person-001")
    step_one_done.set()
    assert append_done.wait(timeout=5)
    worker.join(timeout=5)
    key_ids = delete_repo.tombstone_key_ids("person-001")
    delete_repo._destroy_keys_best_effort(key_ids)
    delete_repo.mark_tombstone_key_destroyed("person-001")
    con = delete_repo.connection
    assert con is not None
    con.execute("BEGIN IMMEDIATE")
    delete_repo._purge_identity_rows(con, "person-001")
    con.execute(
        "DELETE FROM deletion_tombstones WHERE target_id = ?", ("person-001",)
    )
    con.commit()
    assert append_result == []
    assert len(append_error) == 1
    assert isinstance(append_error[0], StoreError)
    assert provider._keys == {}


def test_repository_hydrates_supplied_generation_geometry_and_revision(
    tmp_path: Path,
) -> None:
    template = FaceTemplate(
        template_id="t-9",
        identity_id="person-001",
        model_version="sface-custom-fp32",
        embedding_dim=128,
        revision=TemplateRevision(revision=7, template_id="t-9", supersedes="t-8"),
        generation_id="G9",
        exemplar_crop_box=(2.0, 3.0, 90.0, 91.0),
        exemplar_landmarks=((1.0, 2.0), (3.0, 4.0)),
        quality_score=0.91,
        utility_score=0.72,
    )
    repo = _open_repo(tmp_path)
    repo.initialize()
    repo.enroll_identity(
        "person-001", "Test Person", template, b"e" * 16, b"x" * 32, "t-9"
    )
    hydrated = repo.list_active_templates()[0]
    assert hydrated.generation_id == "G9"
    assert hydrated.model_version == "sface-custom-fp32"
    assert hydrated.embedding_dim == 128
    assert hydrated.revision.revision == 7
    assert hydrated.revision.supersedes == "t-8"
    assert hydrated.exemplar_crop_box == (2.0, 3.0, 90.0, 91.0)
    assert hydrated.exemplar_landmarks == ((1.0, 2.0), (3.0, 4.0))
    assert hydrated.key_id is not None
    assert hydrated.encrypted_exemplar is not None


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
