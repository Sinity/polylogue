"""Controlled settlement failure around an actual SQLite connection."""

from __future__ import annotations

import sqlite3
import threading
from typing import Any


class SettlementHandle:
    """Keep real kernel locks until the original owner can finish cleanup."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.allow_cleanup = threading.Event()
        self.cleanup_started = threading.Event()
        self.continue_cleanup: threading.Event | None = None
        self.calls: list[tuple[str, threading.Thread]] = []
        self.owner = threading.current_thread()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.connection, name)

    @property
    def row_factory(self) -> Any:
        return self.connection.row_factory

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        self.connection.row_factory = value

    def rollback(self) -> None:
        self.calls.append(("rollback", threading.current_thread()))
        assert threading.current_thread() is self.owner
        self.cleanup_started.set()
        if self.continue_cleanup is not None:
            self.continue_cleanup.wait()
        if not self.allow_cleanup.is_set():
            raise OSError("synthetic rollback remains unsettled")
        self.connection.rollback()

    def close(self) -> None:
        self.cleanup_started.set()
        self.calls.append(("close", threading.current_thread()))
        assert threading.current_thread() is self.owner
        if not self.allow_cleanup.is_set():
            raise OSError("synthetic close remains unsettled")
        self.connection.close()
