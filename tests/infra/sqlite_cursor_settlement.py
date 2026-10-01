"""Controlled close failure on an actual native SQLite statement."""

import sqlite3
import threading
from typing import Any


class ControlledCursor(sqlite3.Cursor):
    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        self.creator = threading.current_thread()
        self.allow_cleanup = threading.Event()
        self.allow_cleanup.set()
        self.close_attempts = 0

    def close(self) -> None:
        assert threading.current_thread() is self.creator
        self.close_attempts += 1
        if not self.allow_cleanup.is_set():
            raise OSError("synthetic native cursor remains unsettled")
        super().close()


class UnhashableCursor(ControlledCursor):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnhashableCursor)


class BackupCursorFault:
    """Real backup plus retained native statement at the copy boundary."""

    def __init__(self, connection: sqlite3.Connection, *, on_target: bool, fail_copy: bool) -> None:
        self.connection = connection
        self.on_target = on_target
        self.fail_copy = fail_copy
        self.cursor: ControlledCursor | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.connection, name)

    @property
    def row_factory(self) -> Any:
        return self.connection.row_factory

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        self.connection.row_factory = value

    def backup(self, target: sqlite3.Connection) -> None:
        self.connection.backup(target)
        if not self.on_target and any(row[2] for row in target.execute("PRAGMA database_list") if row[1] == "main"):
            return
        connection = target if self.on_target else self.connection
        self.cursor = connection.cursor(factory=ControlledCursor)
        self.cursor.execute("SELECT 1 UNION ALL SELECT 2")
        assert next(self.cursor)[0] == 1
        self.cursor.allow_cleanup.clear()
        if self.fail_copy:
            raise OSError("synthetic failure after physical SQLite backup")
