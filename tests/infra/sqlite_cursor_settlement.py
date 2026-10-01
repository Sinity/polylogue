"""Controlled close failure on an actual native SQLite statement."""

import sqlite3
import threading


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
