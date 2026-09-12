"""Bounded event-driven peer disconnect observation for one HTTP exchange."""

from __future__ import annotations

import selectors
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from polylogue.daemon.execution import CancellationHandle


@contextmanager
def observe_peer_disconnect(connection: socket.socket) -> Iterator[CancellationHandle]:
    """Wake the request owner on EOF without a polling timer or orphan watcher."""
    disconnected = CancellationHandle()
    wake_read, wake_write = socket.socketpair()

    def watch() -> None:
        with selectors.DefaultSelector() as selector:
            selector.register(connection, selectors.EVENT_READ)
            selector.register(wake_read, selectors.EVENT_READ)
            for key, _events in selector.select():
                if key.fileobj is wake_read:
                    return
                try:
                    closed = not connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                except (ConnectionError, OSError):
                    closed = True
                if closed:
                    disconnected.cancel()
                # The endpoint permits one exchange, not pipelined requests.
                # Extra bytes are never consumed by the operation owner.
                return

    thread = threading.Thread(target=watch, name="operation-peer", daemon=True)
    thread.start()
    try:
        yield disconnected
    finally:
        try:
            wake_write.send(b"x")
            thread.join(timeout=1.0)
        finally:
            wake_read.close()
            wake_write.close()
