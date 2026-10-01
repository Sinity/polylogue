"""Controlled close failure on an actual native SQLite statement."""

import sqlite3
import threading
from typing import Any

from polylogue.storage.io_phase_metrics import _MeasuredConnection


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


class ControlledConnection(_MeasuredConnection):
    """Inject terminal faults on the actual connection registered by its owner."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.creator = threading.current_thread()
        self.rollback_failure: BaseException | None = None
        self.close_failure: BaseException | None = None
        self.rollback_attempts = 0
        self.close_attempts = 0

    def rollback(self) -> None:
        assert threading.current_thread() is self.creator
        self.rollback_attempts += 1
        if self.rollback_failure is not None:
            raise self.rollback_failure
        super().rollback()

    def close(self) -> None:
        assert threading.current_thread() is self.creator
        self.close_attempts += 1
        if self.close_failure is not None:
            raise self.close_failure
        super().close()


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
