"""Task 2 RED/GREEN: KeyProvider custody (S1B §3: FileKeyProvider + gating).

RED: ``ModuleNotFoundError: No module named 'facecore.storage.key_provider'``.
"""

import os
import stat
from pathlib import Path

import pytest

from facecore.errors import StoreError
from facecore.storage.key_provider import FileKeyProvider, InMemoryKeyProvider


def test_in_memory_round_trip_and_idempotent_destroy() -> None:
    provider = InMemoryKeyProvider()
    key_id = provider.create_key("person-001")
    assert provider.get_key(key_id) == provider.get_key(key_id)
    provider.destroy_key(key_id)
    provider.destroy_key(key_id)
    with pytest.raises(StoreError):
        provider.get_key(key_id)


def test_in_memory_destroy_identity_keys_idempotent() -> None:
    provider = InMemoryKeyProvider()
    provider.create_key("person-001")
    provider.destroy_identity_keys("person-001")
    provider.destroy_identity_keys("person-001")


def test_file_provider_new_store_auto_generates_master_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FACECORE_MASTER_KEY", raising=False)
    key_dir = tmp_path / "keys"
    monkeypatch.setattr(
        "facecore.storage.key_provider.default_key_dir",
        lambda: key_dir,
    )
    monkeypatch.setattr(
        "facecore.storage.key_provider.default_master_key_path",
        lambda: tmp_path / "master.key",
    )
    provider = FileKeyProvider()
    key_id = provider.create_key("person-001")
    assert len(provider.get_key(key_id)) == 32
    master = tmp_path / "master.key"
    assert stat.S_IMODE(os.stat(master).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(key_dir).st_mode) == 0o700


def test_file_provider_existing_store_missing_key_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FACECORE_MASTER_KEY", raising=False)
    key_dir = tmp_path / "keys"
    monkeypatch.setattr(
        "facecore.storage.key_provider.default_key_dir",
        lambda: key_dir,
    )
    master = tmp_path / "master.key"
    monkeypatch.setattr(
        "facecore.storage.key_provider.default_master_key_path",
        lambda: master,
    )
    first = FileKeyProvider()
    first.create_key("person-001")
    assert master.exists()
    master.unlink()
    with pytest.raises(StoreError):
        FileKeyProvider()


def test_wrap_rehoming_moves_dek_between_providers() -> None:
    source = InMemoryKeyProvider()
    dest = InMemoryKeyProvider()
    key_id = source.create_key("person-001")
    wrapping_key = b"w" * 32
    wrapped = source.wrap_key(key_id, wrapping_key)
    new_id = dest.unwrap_and_store_key(wrapped, wrapping_key)
    assert dest.get_key(new_id) == source.get_key(key_id)
