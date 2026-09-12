"""Phase-1B crypto contracts (S1B manifest §2-§3: versioned AEAD + KeyProvider).

`EncryptedBlob` carries the 4-byte versioned header contract: ``format_version``
(``0x01``), ``cipher_id`` (``0x01`` = AES-256-GCM), a fresh 96-bit CSPRNG nonce
per write, and the ciphertext (with its 16-byte GCM tag appended).
`KeyProviderProtocol` is the sole custodian of DEKs; SQLite stores only opaque
`key_id` strings (see `KeyReference`). No key material ever enters the database.
"""

from typing import Protocol, runtime_checkable

from facecore.errors import StoreError


class KeyNotFoundError(StoreError):
    """Raised when a DEK is absent from KeyProvider custody (exit code 4)."""


class StoreCorruptionError(StoreError):
    """Raised on tampering, checkpoint failure, or auth-tag mismatch (exit 4)."""


class UnsupportedKdfError(StoreError):
    """Raised when export/import meets an unknown KDF algorithm/version (exit 4)."""


@runtime_checkable
class KeyProviderProtocol(Protocol):
    def create_key(self, identity_id: str) -> str:
        """Generate a fresh 256-bit DEK, store it, return opaque key_id."""
        ...

    def get_key(self, key_id: str) -> bytes:
        """Retrieve the raw 256-bit DEK. Raise KeyNotFoundError if missing."""
        ...

    def destroy_key(self, key_id: str) -> None:
        """Permanently destroy a single DEK. Idempotent (no-op if absent)."""
        ...

    def destroy_identity_keys(self, identity_id: str) -> None:
        """Permanently destroy all DEKs of an identity. Idempotent."""
        ...

    def wrap_key(self, key_id: str, wrapping_key: bytes) -> "WrappedKey":
        """Wrap a DEK under an external key (export key re-homing)."""
        ...

    def unwrap_and_store_key(
        self, wrapped_key: "WrappedKey", unwrapping_key: bytes
    ) -> str:
        """Unwrap a DEK into destination custody, return new key_id."""
        ...


class EncryptedBlob:
    """Versioned AEAD ciphertext blob (S1B §2 wire format, header version 1)."""

    FORMAT_VERSION = 1
    CIPHER_ID_AES_256_GCM = 1
    NONCE_BYTES = 12

    def __init__(
        self,
        *,
        format_version: int,
        cipher_id: int,
        nonce: bytes,
        ciphertext: bytes,
    ) -> None:
        if format_version != self.FORMAT_VERSION:
            raise ValueError(
                f"unsupported blob format_version {format_version!r}; "
                f"contract supports {self.FORMAT_VERSION}"
            )
        if cipher_id != self.CIPHER_ID_AES_256_GCM:
            raise ValueError(
                f"unsupported cipher_id {cipher_id!r}; "
                "contract supports AES-256-GCM only"
            )
        if len(nonce) != self.NONCE_BYTES:
            raise ValueError(
                f"nonce must be {self.NONCE_BYTES} bytes, got {len(nonce)}"
            )
        self.format_version = format_version
        self.cipher_id = cipher_id
        self.nonce = nonce
        self.ciphertext = ciphertext

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "cipher_id": self.cipher_id,
            "nonce_hex": self.nonce.hex(),
            "ciphertext_hex": self.ciphertext.hex(),
        }


class KeyReference:
    """Opaque SQLite-side pointer to a DEK held exclusively by KeyProvider."""

    def __init__(self, *, key_id: str) -> None:
        if not key_id:
            raise ValueError("key_id must be a non-empty opaque reference")
        self.key_id = key_id


class WrappedKey:
    """A DEK wrapped under an external key for export key re-homing (§8)."""

    def __init__(
        self,
        *,
        wrapped_dek: bytes,
        nonce: bytes,
        tag: bytes,
        key_id: str,
    ) -> None:
        self.wrapped_dek = wrapped_dek
        self.nonce = nonce
        self.tag = tag
        self.key_id = key_id
