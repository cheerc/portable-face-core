"""Isolated research-session DEK custodian (Phase 2A §4 & §6 T5).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T5;
    - Task: t-20260914111130097503-76424-35;
    - Governing decision: d-20260914110757304910-5;
    - Boundary rules verified by S2 probe
      (experiments/mac_live_recorder_probe.py): the probe verified the
      boundary rules only; recorder construction is deferred to this module.

Hard boundaries:
    - Dedicated research key directory; never the production
      ``~/.facecore/keys`` namespace, never an identity DEK.
    - Per session, per category dual keys: ``rk_{session_id}`` (record,
      30-day TTL) / ``ik_{session_id}`` (image, 7-day TTL).
    - DEKs at rest are sealed under a research master KEK (0600); key
      directory is 0700; destruction is CSPRNG-overwrite + fsync + unlink.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from facecore.contracts.crypto import KeyNotFoundError, StoreCorruptionError
from facecore.errors import StoreError


def _research_key_id(kind: str, session_id: str) -> str:
    if kind not in ("rk", "ik"):
        raise ValueError(f"unknown research key kind {kind!r}")
    if not session_id or "/" in session_id:
        raise ValueError(f"invalid session_id {session_id!r}")
    return f"{kind}_{session_id}"


class ResearchKeyProvider:
    """Sole custodian of research session DEKs, isolated from identity keys."""

    def __init__(
        self,
        key_dir: Path,
        master_key_path: Path | None = None,
    ) -> None:
        self._key_dir = key_dir
        self._master_path = (
            master_key_path
            if master_key_path is not None
            else key_dir / "master.key"
        )
        env = os.environ.get("FACECORE_RESEARCH_MASTER_KEY")
        if env is None and not self._master_path.exists():
            if self._store_has_state():
                raise KeyNotFoundError(
                    "Research master key missing for existing store; "
                    "store is fail-closed"
                )
            self._master_path.parent.mkdir(parents=True, exist_ok=True)
            self._master_path.write_bytes(os.urandom(32))
            os.chmod(self._master_path, 0o600)
        self._key_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._key_dir, 0o700)
        self._master_key = self._load_master_key(env)
        if len(self._master_key) != 32:
            raise StoreError("Research master key must be 32 bytes")
        if env is None:
            mode = stat.S_IMODE(os.stat(self._master_path).st_mode)
            if mode != 0o600:
                raise StoreError(
                    f"research master.key permissions must be 0600, got {mode:o}"
                )
        self._cache: dict[str, bytes] = {}

    def _load_master_key(self, env: str | None) -> bytes:
        if env is not None:
            try:
                return bytes.fromhex(env.strip())
            except ValueError as exc:
                raise StoreCorruptionError(
                    "FACECORE_RESEARCH_MASTER_KEY is not valid hex"
                ) from exc
        return self._master_path.read_bytes()

    def _store_has_state(self) -> bool:
        if not self._key_dir.exists():
            return False
        return any(self._key_dir.iterdir())

    def _key_path(self, key_id: str) -> Path:
        safe = key_id.replace("/", "-")
        return self._key_dir / f"{safe}.key"

    def _seal(self, dek: bytes, key_id: str) -> bytes:
        nonce = os.urandom(12)
        sealed = AESGCM(self._master_key).encrypt(
            nonce, dek, key_id.encode("utf-8")
        )
        entry = {
            "nonce_hex": nonce.hex(),
            "sealed_hex": sealed.hex(),
        }
        return json.dumps(entry).encode("utf-8")

    def _unseal(self, key_id: str) -> bytes:
        path = self._key_path(key_id)
        if not path.exists():
            raise KeyNotFoundError(f"research key not found: {key_id}")
        try:
            entry = json.loads(path.read_bytes().decode("utf-8"))
            nonce = bytes.fromhex(entry["nonce_hex"])
            sealed = bytes.fromhex(entry["sealed_hex"])
        except (ValueError, UnicodeDecodeError, KeyError) as exc:
            raise StoreCorruptionError(
                f"research key file corrupt: {key_id}"
            ) from exc
        try:
            return AESGCM(self._master_key).decrypt(
                nonce, sealed, key_id.encode("utf-8")
            )
        except Exception as exc:
            raise StoreCorruptionError(
                f"research key unwrap failed: {key_id}"
            ) from exc

    def create_session_keys(self, session_id: str) -> tuple[str, str]:
        """Create the per-session ``rk_``/``ik_`` DEK pair; return key ids."""
        record_key_id = _research_key_id("rk", session_id)
        image_key_id = _research_key_id("ik", session_id)
        for key_id in (record_key_id, image_key_id):
            dek = AESGCM.generate_key(bit_length=256)
            path = self._key_path(key_id)
            path.write_bytes(self._seal(dek, key_id))
            os.chmod(path, 0o600)
            self._cache[key_id] = dek
        return record_key_id, image_key_id

    def get_key(self, key_id: str) -> bytes:
        """Retrieve a DEK; fail closed when absent."""
        cached = self._cache.get(key_id)
        if cached is not None:
            return cached
        dek = self._unseal(key_id)
        self._cache[key_id] = dek
        return dek

    def destroy_key(self, key_id: str) -> None:
        """Destroy one DEK: CSPRNG-overwrite + fsync + unlink. Idempotent."""
        self._cache.pop(key_id, None)
        path = self._key_path(key_id)
        if path.is_file():
            size = path.stat().st_size
            with path.open("r+b") as handle:
                handle.write(os.urandom(size))
                handle.flush()
                os.fsync(handle.fileno())
            path.unlink(missing_ok=True)

    def destroy_session_keys(self, session_id: str) -> None:
        """Destroy both ``rk_`` and ``ik_`` DEKs of one session. Idempotent."""
        self.destroy_key(_research_key_id("rk", session_id))
        self.destroy_key(_research_key_id("ik", session_id))
