"""Shared fixtures: socket-blocking (Task 6 offline assertion)."""

import socket

import pytest


@pytest.fixture
def socket_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any socket creation: inference must never touch the network."""

    def _blocked(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access attempted during offline inference")

    monkeypatch.setattr(socket, "socket", _blocked)
