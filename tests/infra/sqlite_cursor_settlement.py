"""Controlled close failures on actual native SQLite cursors."""

import sqlite3
import threading


class ControlledCursor(sqlite3.Cursor):
    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        self.creator = threading.current_thread()
        self.allow_cleanup = threading.Event()
        self.allow_cleanup.set()
        self.close_attempts = 0
        self.cleanup_failure: BaseException = OSError("synthetic native cursor remains unsettled")

    def close(self) -> None:
        assert threading.current_thread() is self.creator
        self.close_attempts += 1
        if not self.allow_cleanup.is_set():
            raise self.cleanup_failure
        super().close()


class UnhashableCursor(ControlledCursor):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnhashableCursor)


class ConstructorStatementCursor(ControlledCursor):
    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        self.execute("SELECT 1 UNION ALL SELECT 2")
        assert next(self)[0] == 1
        raise ValueError("synthetic failure after native cursor construction SQL")


class BeforeNativeInitCursor(sqlite3.Cursor):
    def __init__(self, connection: sqlite3.Connection) -> None:
        raise ValueError("synthetic failure before native cursor initialization")


class InvalidReturnCursor(sqlite3.Cursor):
    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        self.execute("SELECT 1 UNION ALL SELECT 2")
        next(self)
        return 17  # type: ignore[return-value]  # Deliberate violation of Python's constructor contract.
