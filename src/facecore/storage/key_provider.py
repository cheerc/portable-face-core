"""KeyProvider implementations (S1B §3: sole DEK custody + fail-closed gating).

`InMemoryKeyProvider` serves unit tests. `FileKeyProvider` is the macOS local
provider: DEKs encrypted at rest under a master KEK (``FACECORE_MASTER_KEY``
or ``~/.facecore/master.key`` 0600, key dir 0700). Strict missing-key gating:
a new master key is auto-generated only for a provably empty store; an
existing store with a missing key fails closed with `StoreError` (exit 4).
Destruction is idempotent; files are random-overwritten + fsync before unlink.
"""

import json
import os
import stat
import uuid
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from facecore.contracts.crypto import (
    KeyNotFoundError,
    StoreCorruptionError,
    WrappedKey,
)
from facecore.errors import StoreError


def default_key_dir() -> Path:
    """Resolve the documented key-directory override before the default."""
    override = os.environ.get("FACECORE_KEY_DIR")
    if override is not None:
        if not override.strip():
            raise StoreError("FACECORE_KEY_DIR must not be empty")
        return Path(override).expanduser()
    return Path.home() / ".facecore" / "keys"


def default_master_key_path() -> Path:
    return Path.home() / ".facecore" / "master.key"


def _load_master_key(master_path: Path) -> bytes:
    env = os.environ.get("FACECORE_MASTER_KEY")
    if env:
        try:
            return bytes.fromhex(env.strip())
        except ValueError as exc:
            raise StoreCorruptionError(
                "FACECORE_MASTER_KEY is not valid hex"
            ) from exc
    if master_path.exists():
        return master_path.read_bytes()
    raise KeyNotFoundError(
        "Master key missing for existing repository; store is fail-closed"
    )


class InMemoryKeyProvider:
    """Test-only KeyProvider: DEKs in a dict, no filesystem touch."""

    def __init__(self) -> None:
        self._keys: dict[str, bytes] = {}
        self._owners: dict[str, str] = {}

    def create_key(self, identity_id: str) -> str:
        key_id = f"key-{identity_id}-{uuid.uuid4().hex[:8]}"
        self._keys[key_id] = AESGCM.generate_key(bit_length=256)
        self._owners[key_id] = identity_id
        return key_id

    def get_key(self, key_id: str) -> bytes:
        try:
            return self._keys[key_id]
        except KeyError:
            raise KeyNotFoundError(f"key not found: {key_id}") from None

    def destroy_key(self, key_id: str) -> None:
        self._keys.pop(key_id, None)
        self._owners.pop(key_id, None)

    def destroy_identity_keys(self, identity_id: str) -> None:
        for key_id in [
            k for k, owner in self._owners.items() if owner == identity_id
        ]:
            self._keys.pop(key_id, None)
            del self._owners[key_id]

    def wrap_key(self, key_id: str, wrapping_key: bytes) -> WrappedKey:
        nonce = os.urandom(12)
        sealed = AESGCM(wrapping_key).encrypt(nonce, self.get_key(key_id), None)
        return WrappedKey(
            wrapped_dek=sealed[:-16],
            nonce=nonce,
            tag=sealed[-16:],
            key_id=key_id,
        )

    def unwrap_and_store_key(
        self,
        wrapped_key: WrappedKey,
        unwrapping_key: bytes,
        owner_identity_id: str | None = None,
    ) -> str:
        raw = AESGCM(unwrapping_key).decrypt(
            wrapped_key.nonce,
            wrapped_key.wrapped_dek + wrapped_key.tag,
            None,
        )
        key_id = (
            f"key-rehomed-{wrapped_key.key_id}-{uuid.uuid4().hex[:8]}".replace(
                "/", "-"
            )
        )
        self._keys[key_id] = raw
        # Re-homed keys belong to the destination identity when known;
        # otherwise retain the source key_id as the audit trail.
        self._owners[key_id] = (
            owner_identity_id
            if owner_identity_id is not None
            else wrapped_key.key_id
        )
        return key_id


class FileKeyProvider:
    """macOS file-backed KeyProvider (S1B §3 Fork 3A)."""

    def __init__(
        self,
        key_dir: Path | None = None,
        master_key_path: Path | None = None,
        db_path: Path | None = None,
    ) -> None:
        self._key_dir = key_dir if key_dir is not None else default_key_dir()
        self._master_path = (
            master_key_path
            if master_key_path is not None
            else default_master_key_path()
        )
        self._db_path = (
            db_path if db_path is not None else self._key_dir.parent / "facecore.db"
        )
        env = os.environ.get("FACECORE_MASTER_KEY")
        if env is None and not self._master_path.exists():
            if self._store_has_state():
                raise KeyNotFoundError(
                    "Master key missing for existing repository; "
                    "store is fail-closed"
                )
            self._master_path.parent.mkdir(parents=True, exist_ok=True)
            self._master_path.write_bytes(os.urandom(32))
            os.chmod(self._master_path, 0o600)
        self._key_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._key_dir, 0o700)
        try:
            self._master_key = _load_master_key(self._master_path)
        except StoreError:
            raise
        if len(self._master_key) != 32:
            raise StoreError("Master key must be 32 bytes")
        if env is None:
            mode = stat.S_IMODE(os.stat(self._master_path).st_mode)
            if mode != 0o600:
                raise StoreError(
                    f"master.key permissions must be 0600, got {mode:o}"
                )

    def _store_has_state(self) -> bool:
        """A DB file alone is existing state; missing KEK must not regenerate."""
        if self._db_path.exists():
            return True
        if not self._key_dir.exists():
            return False
        return any(self._key_dir.iterdir())

    def _key_path(self, key_id: str) -> Path:
        safe = key_id.replace("/", "-")
        return self._key_dir / f"{safe}.key"

    def _read_entry(self, key_id: str) -> dict[str, str]:
        path = self._key_path(key_id)
        if not path.exists():
            raise KeyNotFoundError(f"key not found: {key_id}")
        try:
            parsed: object = json.loads(path.read_bytes().decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise StoreCorruptionError(f"key file corrupt: {key_id}") from exc
        if not isinstance(parsed, dict):
            raise StoreCorruptionError(f"key file corrupt: {key_id}")
        entry: dict[str, str] = {str(k): str(v) for k, v in parsed.items()}
        return entry

    def create_key(self, identity_id: str) -> str:
        key_id = f"key-{identity_id}-{uuid.uuid4().hex[:8]}"
        dek = AESGCM.generate_key(bit_length=256)
        nonce = os.urandom(12)
        sealed = AESGCM(self._master_key).encrypt(
            nonce, dek, key_id.encode("utf-8")
        )
        entry = {
            "identity_id": identity_id,
            "nonce": nonce.hex(),
            "encrypted_dek": sealed.hex(),
        }
        path = self._key_path(key_id)
        path.write_text(json.dumps(entry))
        os.chmod(path, 0o600)
        return key_id

    def get_key(self, key_id: str) -> bytes:
        entry = self._read_entry(key_id)
        try:
            nonce = bytes.fromhex(entry["nonce"])
            sealed = bytes.fromhex(entry["encrypted_dek"])
        except (KeyError, ValueError) as exc:
            raise StoreCorruptionError(f"key file corrupt: {key_id}") from exc
        try:
            return AESGCM(self._master_key).decrypt(
                nonce, sealed, key_id.encode("utf-8")
            )
        except Exception as exc:
            raise StoreCorruptionError(f"key unwrap failed: {key_id}") from exc

    def destroy_key(self, key_id: str) -> None:
        path = self._key_path(key_id)
        if not path.exists():
            return
        length = path.stat().st_size
        with open(path, "r+b") as handle:
            handle.write(os.urandom(length))
            handle.flush()
            os.fsync(handle.fileno())
        path.unlink()

    def destroy_identity_keys(self, identity_id: str) -> None:
        if not self._key_dir.exists():
            return
        for path in list(self._key_dir.glob("*.key")):
            try:
                entry = json.loads(path.read_bytes().decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if entry.get("identity_id") == identity_id:
                self.destroy_key(path.stem)

    def wrap_key(self, key_id: str, wrapping_key: bytes) -> WrappedKey:
        raw = self.get_key(key_id)
        nonce = os.urandom(12)
        sealed = AESGCM(wrapping_key).encrypt(nonce, raw, None)
        return WrappedKey(
            wrapped_dek=sealed[:-16],
            nonce=nonce,
            tag=sealed[-16:],
            key_id=key_id,
        )

    def unwrap_and_store_key(
        self,
        wrapped_key: WrappedKey,
        unwrapping_key: bytes,
        owner_identity_id: str | None = None,
    ) -> str:
        raw = AESGCM(unwrapping_key).decrypt(
            wrapped_key.nonce,
            wrapped_key.wrapped_dek + wrapped_key.tag,
            None,
        )
        key_id = f"key-rehomed-{uuid.uuid4().hex[:8]}"
        nonce = os.urandom(12)
        sealed = AESGCM(self._master_key).encrypt(
            nonce, raw, key_id.encode("utf-8")
        )
        entry = {
            "identity_id": (
                owner_identity_id
                if owner_identity_id is not None
                else wrapped_key.key_id
            ),
            "nonce": nonce.hex(),
            "encrypted_dek": sealed.hex(),
        }
        path = self._key_path(key_id)
        path.write_text(json.dumps(entry))
        os.chmod(path, 0o600)
        return key_id
