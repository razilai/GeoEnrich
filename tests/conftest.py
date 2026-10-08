import socket

import pytest


@pytest.fixture(autouse=True)
def _block_sockets(monkeypatch):
    """No test may open a network connection."""

    def blocked(*args, **kwargs):
        raise RuntimeError("tests may not open sockets")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
