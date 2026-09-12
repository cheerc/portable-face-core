"""Task 2 RED/GREEN: AEAD cipher contract (S1B §2: versioned AES-256-GCM).

RED: ``ModuleNotFoundError: No module named 'facecore.storage.cipher'``.
"""

from facecore.storage.cipher import (
    MAX_AAD_FIELD_BYTES,
    AeadCipher,
    build_canonical_aad,
)

import pytest


def test_canonical_aad_disambiguates_delimiters() -> None:
    a = build_canonical_aad("face_templates", "a:b", "c")
    b = build_canonical_aad("face_templates", "a", "b:c")
    assert a != b
    assert a.startswith(b"facecore:v1:")


def test_canonical_aad_rejects_overlength_field() -> None:
    with pytest.raises(ValueError):
        build_canonical_aad("t", "r" * (MAX_AAD_FIELD_BYTES + 1), "i")


def test_nonce_uniqueness_across_encryptions() -> None:
    cipher = AeadCipher.generate()
    nonces = {
        cipher.encrypt(b"\x00" * 32, build_canonical_aad("t", "r", "i")).nonce
        for _ in range(64)
    }
    assert len(nonces) == 64


def test_round_trip_and_tamper_fails_closed() -> None:
    cipher = AeadCipher.generate()
    aad = build_canonical_aad("face_templates", "tmpl-001", "person-001")
    blob = cipher.encrypt(b"\x00" * 512, aad)
    assert blob.format_version == 1
    assert blob.cipher_id == 1
    assert cipher.decrypt(blob, aad) == b"\x00" * 512
    tampered_aad = build_canonical_aad("face_templates", "tmpl-002", "person-001")
    with pytest.raises(Exception):
        cipher.decrypt(blob, tampered_aad)
