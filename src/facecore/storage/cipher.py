"""Application-layer AEAD cipher (S1B §2: AES-256-GCM + bounded AAD + 4B header).

Wire format per encrypted blob: 4-byte header (``format_version=0x01``,
``cipher_id=0x01``, reserved ``0x00 0x00``) + 12-byte CSPRNG nonce +
ciphertext (GCM tag appended). `EncryptedBlob` (contracts) carries the
parsed form; this module owns encrypt/decrypt against live DEKs.
"""

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from facecore.contracts.crypto import EncryptedBlob, StoreCorruptionError

MAX_AAD_FIELD_BYTES = 65535
_AAD_PREFIX = b"facecore:v1:"


def build_canonical_aad(table: str, record_id: str, identity_id: str) -> bytes:
    """Canonical length-prefixed AAD with explicit byte-length validation."""
    parts = []
    for name, value in (
        ("table", table),
        ("record_id", record_id),
        ("identity_id", identity_id),
    ):
        encoded = value.encode("utf-8")
        if len(encoded) > MAX_AAD_FIELD_BYTES:
            raise ValueError(
                f"AAD field {name!r} length {len(encoded)} exceeds maximum "
                f"{MAX_AAD_FIELD_BYTES} bytes"
            )
        parts.append(len(encoded).to_bytes(2, "big") + encoded)
    return _AAD_PREFIX + b"".join(parts)


class AeadCipher:
    """AES-256-GCM cipher bound to one 256-bit DEK."""

    def __init__(self, dek: bytes) -> None:
        if len(dek) != 32:
            raise ValueError(f"DEK must be 32 bytes, got {len(dek)}")
        self._aead = AESGCM(dek)

    @classmethod
    def generate(cls) -> "AeadCipher":
        return cls(AESGCM.generate_key(bit_length=256))

    def encrypt(self, plaintext: bytes, aad: bytes) -> EncryptedBlob:
        nonce = os.urandom(EncryptedBlob.NONCE_BYTES)
        ciphertext = self._aead.encrypt(nonce, plaintext, aad)
        return EncryptedBlob(
            format_version=EncryptedBlob.FORMAT_VERSION,
            cipher_id=EncryptedBlob.CIPHER_ID_AES_256_GCM,
            nonce=nonce,
            ciphertext=ciphertext,
        )

    def decrypt(self, blob: EncryptedBlob, aad: bytes) -> bytes:
        try:
            return self._aead.decrypt(blob.nonce, blob.ciphertext, aad)
        except InvalidTag as exc:
            raise StoreCorruptionError(
                "AEAD authentication failed: tampered ciphertext or AAD"
            ) from exc
