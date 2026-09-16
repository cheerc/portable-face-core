"""Shared fixtures and assertion helpers (offline assertion, at-rest privacy)."""

from collections.abc import Callable, Sequence
from pathlib import Path
import socket

import pytest


@pytest.fixture
def socket_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any socket creation: inference must never touch the network."""

    def _blocked(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access attempted during offline inference")

    monkeypatch.setattr(socket, "socket", _blocked)


def assert_no_plaintext_leak(
    root: Path, sensitive_values: Sequence[str | bytes]
) -> None:
    """Assert no sensitive strings/bytes appear in any raw file at rest under root.

    Reusable across E1 (attempt/label), E2 (trace), E5 (case packet), E6 (freeze).
    """
    assert root.exists(), f"scan target {root} does not exist"
    forbidden: list[bytes] = []
    for val in sensitive_values:
        if isinstance(val, str):
            if val:
                forbidden.append(val.encode("utf-8"))
        elif isinstance(val, (bytes, bytearray)):
            if val:
                forbidden.append(bytes(val))

    leaks: list[str] = []
    for p in root.rglob("*"):
        if p.is_file():
            raw = p.read_bytes()
            for pattern in forbidden:
                if pattern in raw:
                    leaks.append(f"{p}: found forbidden literal {pattern!r}")
    if leaks:
        raise AssertionError(
            "Plaintext at-rest leak detected:\n" + "\n".join(leaks)
        )


@pytest.fixture
def assert_no_leak() -> Callable[[Path, Sequence[str | bytes]], None]:
    """Fixture wrapping assert_no_plaintext_leak for test injection."""
    return assert_no_plaintext_leak
