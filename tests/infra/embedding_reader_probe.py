"""Physical SQLite lifetime observations for synthetic embedding routes."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue.storage.sqlite.connection_profile import open_readonly_connection


class EmbeddingReadProbe:
    """Retain actual native cursors and verify their close ordering and DB fds."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.monkeypatch = monkeypatch
        self.handles: list[sqlite3.Connection] = []
        self.cursors: list[TrackedCursor] = []

    def open(self, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = open_readonly_connection(*args, **kwargs)
        native_close = conn.close

        def execute(*args: Any, **kwargs: Any) -> sqlite3.Cursor:
            cursor = conn.cursor(factory=TrackedCursor)
            self.cursors.append(cursor)
            try:
                return cursor.execute(*args, **kwargs)
            except BaseException:
                cursor.close()
                raise

        def close() -> None:
            native_close()
            assert all(cursor.settled for cursor in self.cursors if cursor.connection is conn)

        self.monkeypatch.setattr(conn, "execute", execute)
        self.monkeypatch.setattr(conn, "close", close)
        self.handles.append(conn)
        return conn

    def assert_settled(self) -> None:
        assert self.handles
        assert all(cursor.settled for cursor in self.cursors)
        for conn in self.handles:
            with pytest.raises(sqlite3.ProgrammingError):
                conn.execute("SELECT 1")
        descriptors = Path("/proc/self/fd")
        if not descriptors.is_dir():
            # Darwin has no procfs. Native cursor ordering and closed-handle
            # checks above still exercise the full behavioral route there.
            return
        retained = []
        for descriptor in descriptors.iterdir():
            try:
                target = Path(os.readlink(descriptor).removesuffix(" (deleted)"))
            except FileNotFoundError:
                continue
            if target.is_relative_to(self.root) and target.name.endswith((".db", ".db-wal", ".db-shm")):
                retained.append(target.name)
        assert retained == [], retained


class TrackedCursor(sqlite3.Cursor):
    """A native SQLite cursor whose settlement must precede native parent close."""

    settled = False

    def close(self) -> None:
        if not self.settled:
            # Access on a closed connection raises: this is an ordering proof,
            # before super.close, rather than ProgrammingError after c.close.
            _ = self.connection.in_transaction
            super().close()
            self.settled = True
