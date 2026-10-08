"""Bounded process counters for observable storage I/O phase boundaries.

``begin`` covers explicit SQL BEGIN on measured handles. SQLite's implicit
BEGIN, internal busy-handler sleeps, and internal fsyncs are not exposed by
the stdlib connection API and are named as unavailable below.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import weakref
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TypeVar, cast, overload

from polylogue.storage.sqlite.write_lease import UnleasedWriteError, current_write_lease, require_write_lease

Tier = Literal["source", "index", "embeddings", "user", "audit", "ops"]
Phase = Literal[
    "connection_create",
    "begin",
    "commit",
    "rollback",
    "checkpoint",
    "blob_file_fsync",
    "blob_directory_fsync",
]

_TIERS = frozenset({"source", "index", "embeddings", "user", "audit", "ops"})
_COUNTS: dict[tuple[Tier, Phase, bool, bool], tuple[int, int]] = {}
_LOCK = threading.Lock()
_CursorT = TypeVar("_CursorT", bound=sqlite3.Cursor)


@dataclass(frozen=True, slots=True)
class IoPhaseSample:
    tier: Tier
    phase: Phase
    inside_writer_lease: bool
    succeeded: bool
    count: int
    elapsed_ns: int


def tier_for_path(path: str | Path) -> Tier | None:
    name = str(path).split("?", 1)[0].removeprefix("file:")
    stem = Path(name).name.removesuffix(".db")
    return cast(Tier, stem) if stem in _TIERS else None


def _inside_writer_lease() -> bool:
    lease = current_write_lease()
    if lease is None:
        return False
    try:
        # Echo the held lease's own archive_root: this sampler has no
        # archive of its own to assert, it only wants to know whether a
        # lease is held. Passing None here would otherwise read as an
        # omission on an archive-bound lease and misreport every sample as
        # unowned; the lease API treats a missing archive root as an omitted
        # identity on an archive-bound lease.
        return require_write_lease("I/O phase ownership sample", archive_root=lease.archive_root) is not None
    except UnleasedWriteError:
        return False


def record_io_phase(
    tier: Tier | None,
    phase: Phase,
    elapsed_ns: int,
    *,
    succeeded: bool,
    inside_writer_lease: bool | None = None,
) -> None:
    if tier is None:
        return
    owned = _inside_writer_lease() if inside_writer_lease is None else inside_writer_lease
    key = (tier, phase, owned, succeeded)
    with _LOCK:
        count, total = _COUNTS.get(key, (0, 0))
        _COUNTS[key] = (count + 1, total + elapsed_ns)


def io_phase_snapshot() -> tuple[IoPhaseSample, ...]:
    with _LOCK:
        values = tuple(_COUNTS.items())
    return tuple(
        IoPhaseSample(tier, phase, owned, succeeded, count, elapsed_ns)
        for (tier, phase, owned, succeeded), (count, elapsed_ns) in sorted(values)
    )


def io_phase_process_snapshot() -> dict[str, object]:
    """Export one process's counters for explicit worker receipt aggregation."""
    return {
        "scope": "process",
        "pid": os.getpid(),
        "samples": [
            {
                "tier": item.tier,
                "phase": item.phase,
                "inside_writer_lease": item.inside_writer_lease,
                "succeeded": item.succeeded,
                "count": item.count,
                "elapsed_ns": item.elapsed_ns,
            }
            for item in io_phase_snapshot()
        ],
        "unavailable": list(UNAVAILABLE_SQLITE_INTERNAL_PHASES),
    }


UNAVAILABLE_SQLITE_INTERNAL_PHASES = (
    "sqlite_implicit_begin",
    "sqlite_busy_wait",
    "sqlite_file_fsync",
    "sqlite_directory_fsync",
)


@contextmanager
def timed_io_phase(tier: Tier | None, phase: Phase) -> Iterator[None]:
    if tier is None:
        yield
        return
    owned = _inside_writer_lease()
    started = time.perf_counter_ns()
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        record_io_phase(
            tier,
            phase,
            time.perf_counter_ns() - started,
            succeeded=succeeded,
            inside_writer_lease=owned,
        )


def _transaction_phase(sql: str) -> Phase | None:
    token = sql.lstrip().split(None, 1)[0].upper() if sql.strip() else ""
    if token in {"BEGIN", "COMMIT", "END", "ROLLBACK"}:
        return "commit" if token == "END" else cast(Phase, token.lower())
    return None


class _MeasuredCursor(sqlite3.Cursor):
    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        cast(_MeasuredConnection, connection)._register_cursor(self)

    def close(self) -> None:
        super().close()
        cursors = getattr(self.connection, "_native_cursors", {})
        cursors.pop(id(self), None)

    def execute(self, sql: str, parameters: Any = (), /) -> _MeasuredCursor:
        phase = _transaction_phase(sql)
        connection = cast(_MeasuredConnection, self.connection)
        tier = getattr(connection, "_metric_tier", None)
        # One epoch per statement compile: an authorizer may verify facts
        # that cannot vary across one compile once per epoch instead of once
        # per callback. Nested execution gets its own epoch.
        connection._compile_epoch_counter += 1
        enclosing = connection._compile_epoch
        connection._compile_epoch = connection._compile_epoch_counter
        try:
            if phase is None:
                return super().execute(sql, parameters)
            connection._metric_statement_phase = phase
            try:
                with timed_io_phase(tier, phase):
                    return super().execute(sql, parameters)
            finally:
                connection._metric_statement_phase = None
        finally:
            connection._compile_epoch = enclosing


class _MeasuredConnection(sqlite3.Connection):
    _native_creator: tuple[int, threading.Thread] | None = None
    _metric_tier: Tier | None = None
    _metric_context_exit = False
    _metric_statement_phase: Phase | None = None
    #: The statement compile in progress on this connection, or ``None``
    #: outside a measured ``execute`` (executemany, executescript, Blob opens).
    _compile_epoch: int | None = None
    _compile_epoch_counter = 0
    _native_closed = False
    _incremental_blobs_readonly = False
    _incremental_blob_register: Callable[[sqlite3.Blob], None] | None = None
    _incremental_blob_admit: Callable[[], object] | None = None

    def blobopen(
        self, table: str, column: str, row: int, /, *, readonly: bool = False, name: str = "main"
    ) -> sqlite3.Blob:
        if getattr(self, "_incremental_blobs_readonly", False) and not readonly:
            raise sqlite3.OperationalError("guarded tier connections prohibit writable incremental blobs")
        if self._incremental_blob_admit is not None:
            self._incremental_blob_admit()
        blob = super().blobopen(table, column, row, readonly=readonly, name=name)
        register = self._incremental_blob_register
        if register is not None:
            register(blob)
        return blob

    @overload
    def cursor(self, factory: None = None) -> sqlite3.Cursor: ...

    @overload
    def cursor(self, factory: Callable[[sqlite3.Connection], _CursorT]) -> _CursorT: ...

    def cursor(self, factory: Callable[[sqlite3.Connection], sqlite3.Cursor] | None = None) -> sqlite3.Cursor:
        selected = _MeasuredCursor if factory is None else factory
        if (
            isinstance(selected, type)
            and type(selected) is type
            and issubclass(selected, sqlite3.Cursor)
            and selected.__new__ is sqlite3.Cursor.__new__
        ):
            # Preserve the exact plain subclass and its ordinary class-call
            # behavior, registering before its initializer can issue SQL.
            constructor: Any = selected

            def construct(connection: sqlite3.Connection) -> sqlite3.Cursor:
                cursor = cast(sqlite3.Cursor, constructor.__new__(constructor, connection))
                self._register_cursor(cursor)
                # Bind the actual initializer descriptor as the class call
                # does, including native, static and class method shapes.
                try:
                    descriptor: Any = next(
                        base.__dict__["__init__"] for base in constructor.__mro__ if "__init__" in base.__dict__
                    )
                    initializer = (
                        descriptor.__get__(cursor, constructor) if hasattr(descriptor, "__get__") else descriptor
                    )
                    returned = initializer(connection)
                    if returned is not None:
                        raise TypeError(f"__init__() should return None, not '{type(returned).__name__}'")
                except BaseException:
                    # A failing initializer that never called native __init__
                    # created no statement or handle. Do not retain a fiction.
                    if sqlite3.Cursor.connection.__get__(cursor) is None:
                        self._native_cursors.pop(id(cursor), None)
                    raise
                return cursor

            # Keep SQLite's native closed-handle and thread checks ahead of
            # factory invocation, as in the original Connection.cursor call.
            return super().cursor(construct)
        # Arbitrary callbacks/metaclasses retain their invocation semantics.
        # A cursor never exposed by a raising callback cannot be captured;
        # constructor physical proof covers our owned/plain-class factories.
        cursor = super().cursor(selected)
        self._register_cursor(cursor)
        return cursor

    def _register_cursor(self, cursor: sqlite3.Cursor) -> None:
        cursors = getattr(self, "_native_cursors", None)
        if cursors is None:
            self._native_cursors: dict[int, weakref.ReferenceType[sqlite3.Cursor]] = {}
        cursor_id = id(cursor)
        connection_ref = weakref.ref(self)

        def discard(reference: weakref.ReferenceType[sqlite3.Cursor]) -> None:
            connection = connection_ref()
            if connection is not None and connection._native_cursors.get(cursor_id) is reference:
                connection._native_cursors.pop(cursor_id)

        # Custom cursor equality/hash methods cannot merge two physical
        # statements. Identity keys also support unhashable custom cursors.
        self._native_cursors[cursor_id] = weakref.ref(cursor, discard)

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        # The stdlib shortcuts bypass the Python cursor factory. Create the
        # actual cursor first so even an execute-error traceback is owned.
        return self.cursor().execute(sql, parameters)

    def executemany(self, sql: str, parameters: Any, /) -> sqlite3.Cursor:
        return self.cursor().executemany(sql, parameters)

    def executescript(self, sql_script: str, /) -> sqlite3.Cursor:
        return self.cursor().executescript(sql_script)

    def live_cursors(self) -> tuple[sqlite3.Cursor, ...]:
        live = {
            id(cursor): cursor
            for reference in getattr(self, "_native_cursors", {}).values()
            if (cursor := reference()) is not None
        }
        live.update(getattr(self, "_unsettled_native_cursors", {}))
        return tuple(live.values())

    def close_cursor(self, cursor: sqlite3.Cursor) -> None:
        try:
            cursor.close()
        except BaseException:
            # Healthy cursors remain weakly inventoried. A failed physical
            # close must retain this exact actual statement independently of
            # an exception traceback or the producer's last local reference.
            pending = getattr(self, "_unsettled_native_cursors", None)
            if pending is None:
                self._unsettled_native_cursors: dict[int, sqlite3.Cursor] = {}
            self._unsettled_native_cursors[id(cursor)] = cursor
            raise
        getattr(self, "_native_cursors", {}).pop(id(cursor), None)
        getattr(self, "_unsettled_native_cursors", {}).pop(id(cursor), None)

    def settle_cursors(self) -> None:
        failures: list[BaseException] = []
        for cursor in self.live_cursors():
            try:
                self.close_cursor(cursor)
            except BaseException as failure:
                failures.append(failure)
        if failures:
            # sqlite3_close_v2 alone can leave a zombie connection held by a
            # partially consumed statement. Keep the actual connection open
            # for its existing native owner's creator-thread cleanup retry.
            raise BaseExceptionGroup("native SQLite cursors remain unsettled", failures)

    def close(self) -> None:
        if self._native_closed:
            return super().close()
        self.settle_cursors()
        super().close()
        self._native_closed = True
        if hasattr(self, "_native_cursors"):
            self._native_cursors.clear()

    def commit(self) -> None:
        if not self.in_transaction or self._metric_context_exit or self._metric_statement_phase is not None:
            return super().commit()
        with timed_io_phase(self._metric_tier, "commit"):
            return super().commit()

    def rollback(self) -> None:
        if not self.in_transaction or self._metric_context_exit or self._metric_statement_phase is not None:
            return super().rollback()
        with timed_io_phase(self._metric_tier, "rollback"):
            return super().rollback()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if not self.in_transaction:
            return super().__exit__(exc_type, exc_value, traceback)
        phase: Phase = "rollback" if exc_type is not None else "commit"
        self._metric_context_exit = True
        try:
            with timed_io_phase(self._metric_tier, phase):
                return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self._metric_context_exit = False


def bind_readonly_incremental_blob_custody(
    connection: sqlite3.Connection, register: Callable[[sqlite3.Blob], None], admit: Callable[[], object]
) -> None:
    """Keep guarded incremental handles with the existing actual SQL owner."""
    measured = cast(_MeasuredConnection, connection)
    measured._incremental_blobs_readonly = True
    measured._incremental_blob_register = register
    measured._incremental_blob_admit = admit


def native_connection_physically_closed(connection: sqlite3.Connection) -> bool:
    """Use only this factory's successful native-close and statement proof."""
    return isinstance(connection, _MeasuredConnection) and connection._native_closed and not connection.live_cursors()


def settle_connection_cursors(connection: sqlite3.Connection) -> None:
    """Settle statements through the same measured connection's creator owner."""
    cast(_MeasuredConnection, connection).settle_cursors()


def live_connection_cursors(connection: sqlite3.Connection) -> tuple[sqlite3.Cursor, ...]:
    return cast(_MeasuredConnection, connection).live_cursors()


def close_connection_cursor(connection: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    if isinstance(connection, _MeasuredConnection):
        connection.close_cursor(cursor)
    else:
        # Plain SQLite callers own their explicit connection context.
        cursor.close()


@contextmanager
def connection_cursor(
    connection: sqlite3.Connection, sql: str, parameters: Sequence[object] | Mapping[str, object] = ()
) -> Iterator[sqlite3.Cursor]:
    """Retain a statement before execution and settle its actual native cursor."""
    cursor = connection.cursor()
    primary: BaseException | None = None
    try:
        cursor.execute(sql, parameters)
        yield cursor
    except BaseException as failure:
        primary = failure
        raise
    finally:
        try:
            close_connection_cursor(connection, cursor)
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup("Statement and native cursor close failed", [primary, cleanup]) from primary
            raise


def native_connection_created_on_current_thread(connection: sqlite3.Connection) -> bool:
    """Prove the original measured factory's creator without adopting a handle."""
    return (
        isinstance(connection, _MeasuredConnection)
        and getattr(connection, "_native_creator", None) == (os.getpid(), threading.current_thread())
        and not connection._native_closed
    )


def connect_measured(database: str | Path, /, *args: Any, **kwargs: Any) -> sqlite3.Connection:
    """Time an actual returned SQLite handle without altering its PRAGMA policy."""
    from polylogue.storage.sqlite.population_admission import assert_population_admitted

    assert_population_admitted(database)
    tier = tier_for_path(database)
    owned = _inside_writer_lease()
    started = time.perf_counter_ns()
    succeeded = False
    try:
        conn = sqlite3.connect(str(database), *args, factory=_MeasuredConnection, **kwargs)
        conn._metric_tier = tier
        conn._native_creator = (os.getpid(), threading.current_thread())
        succeeded = True
        return conn
    finally:
        record_io_phase(
            tier,
            "connection_create",
            time.perf_counter_ns() - started,
            succeeded=succeeded,
            inside_writer_lease=owned,
        )


__all__ = [
    "IoPhaseSample",
    "UNAVAILABLE_SQLITE_INTERNAL_PHASES",
    "close_connection_cursor",
    "connect_measured",
    "live_connection_cursors",
    "native_connection_physically_closed",
    "native_connection_created_on_current_thread",
    "settle_connection_cursors",
    "io_phase_process_snapshot",
    "io_phase_snapshot",
    "record_io_phase",
    "tier_for_path",
    "timed_io_phase",
]
