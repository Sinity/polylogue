"""Controlled close failure on an actual native SQLite statement."""

import sqlite3
import threading
from builtins import BaseExceptionGroup
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from polylogue.storage.io_phase_metrics import _MeasuredConnection


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


def sqlite_factory_targets_database(database: str | Path, paths: Iterable[str | Path]) -> bool:
    """Select the actual anchored writable fixture handle without reopening it."""
    token = str(database)
    if token == ":memory:":
        return False
    if token.startswith("file:"):
        uri = urlsplit(token)
        if parse_qs(uri.query).get("mode") != ["rw"]:
            return False
        selected = Path(unquote(uri.path))
    else:
        selected = Path(database)
    # stat/samefile never opens/closes the SQLite inode, preserving native
    # POSIX locks. This selector grants no writer permission or close proof.
    return any(selected.samefile(Path(path)) for path in paths)


def control_archive_connections(monkeypatch: pytest.MonkeyPatch, *paths: str | Path) -> None:
    """Control actual writable factory bindings, before Native registration."""
    from polylogue.storage.sqlite import connection_profile
    from polylogue.storage.sqlite.archive_tiers import archive

    destinations = {destination for path in paths for destination in (Path(path), Path(path).resolve())}
    for module in (archive, connection_profile):
        original = module.connect_measured

        def controlled(
            database: str | Path, *args: Any, _original: Any = original, **kwargs: Any
        ) -> sqlite3.Connection:
            if sqlite_factory_targets_database(database, destinations):
                return sqlite3.connect(database, *args, factory=ControlledConnection, **kwargs)
            return cast(sqlite3.Connection, _original(database, *args, **kwargs))

        monkeypatch.setattr(module, "connect_measured", controlled)


class BackupCursorFault(_MeasuredConnection):
    """Actual native backup owner retaining a statement at the copy boundary."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.on_target = False
        self.fail_copy = False
        self.retained_cursor: ControlledCursor | None = None

    def backup(self, target: sqlite3.Connection, **kwargs: Any) -> None:
        super().backup(target, **kwargs)
        if not self.on_target and any(row[2] for row in target.execute("PRAGMA database_list") if row[1] == "main"):
            return
        connection = target if self.on_target else self
        self.retained_cursor = connection.cursor(factory=ControlledCursor)
        self.retained_cursor.execute("SELECT 1 UNION ALL SELECT 2")
        assert next(self.retained_cursor)[0] == 1
        self.retained_cursor.allow_cleanup.clear()
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


class SettlementConnection(_MeasuredConnection):
    """Fault the native handle that the production factory actually registers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.owner = threading.current_thread()
        self.allow_cleanup = threading.Event()
        self.allow_cleanup.set()
        self.cleanup_started = threading.Event()
        self.continue_cleanup: threading.Event | None = None
        self.calls: list[tuple[str, threading.Thread]] = []

    def rollback(self) -> None:
        self.calls.append(("rollback", threading.current_thread()))
        assert threading.current_thread() is self.owner
        self.cleanup_started.set()
        if self.continue_cleanup is not None:
            self.continue_cleanup.wait()
        if not self.allow_cleanup.is_set():
            raise OSError("synthetic rollback remains unsettled")
        super().rollback()

    def close(self) -> None:
        self.cleanup_started.set()
        self.calls.append(("close", threading.current_thread()))
        assert threading.current_thread() is self.owner
        if not self.allow_cleanup.is_set():
            raise OSError("synthetic close remains unsettled")
        super().close()


def arm_settlement(connection: sqlite3.Connection) -> SettlementConnection:
    """Arm the factory-created connection without replacing its identity."""
    assert isinstance(connection, SettlementConnection)
    connection.allow_cleanup.clear()
    return connection


def settle_fault_connections(handles: list[SettlementConnection]) -> None:
    """Settle only a control's actual registered handles on their creator."""
    from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_on_current_thread

    for handle in handles:
        handle.allow_cleanup.set()
    identities = {id(handle) for handle in handles}
    attempted: set[int] = set()
    failures: list[BaseException] = []
    for owner in retained_native_sql_owners_on_current_thread():
        if owner._connection_identity not in identities:
            continue
        terminal = owner._terminal_parent or owner
        if id(terminal) not in attempted:
            attempted.add(id(terminal))
            try:
                terminal.close()
            except BaseException as error:
                failures.append(error)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("Controlled native cleanup failed", failures)


@pytest.fixture(autouse=True)
def native_settlement_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use actual measured subclasses in modules with terminal fault controls."""
    original = sqlite3.connect

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        factory = kwargs.get("factory", sqlite3.Connection)
        if factory in (sqlite3.Connection, _MeasuredConnection):
            kwargs["factory"] = SettlementConnection
        return cast(sqlite3.Connection, original(*args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", connect)


def settlement_owner_summary(owners: Any) -> tuple[tuple[object, ...], ...]:
    """Fixed physical-owner metadata for discriminating failed census controls."""
    return tuple(
        (
            type(owner).__name__,
            type(getattr(owner, "_terminal_parent", None)).__name__,
            bool(getattr(owner, "_settled", False)),
            getattr(owner, "connection", None) is not None,
            bool(getattr(owner, "close_required", False)),
            tuple(type(dependency).__name__ for dependency in getattr(owner, "_lifetime_dependencies", ())),
        )
        for owner in owners
    )
