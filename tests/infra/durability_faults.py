"""Reusable crash-point injection for storage durability tests.

The fixture in this module is deliberately test-only.  It instruments the
existing filesystem and SQLite seams at runtime; production modules do not
import it and do not grow fault-specific branches.

Typical use::

    def test_recovery_survives_every_commit_point(tmp_path, durability_faults):
        archive = make_archive(tmp_path)
        for point in durability_faults.points():
            durability_faults.run(
                point,
                lambda: production_mutation(archive),
                recover=lambda: production_startup_recovery(archive),
                assert_invariants=lambda: assert_archive_is_sound(archive),
            )

The operation is run once per point.  A raised fault is the portable
equivalent of a process death: all in-process cleanup is abandoned by the
operation, then the supplied recovery callable is invoked in a clean fault
state.  ``action="kill"`` raises :class:`InjectedCrash`, a
``BaseException`` subclass, so accidental broad ``except Exception`` cleanup
cannot turn a simulated kill into a successful operation.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Any
from unittest.mock import patch

import pytest


class DurabilityFaultPoint(StrEnum):
    """Named durability boundaries shared by all archive subsystems."""

    FSYNC = "fsync"
    COMMIT = "commit"
    REPLACE = "replace"
    UNLINK = "unlink"
    STATEMENT = "statement"


class InjectedFaultError(RuntimeError):
    """An ordinary raised failure at a selected durability boundary."""

    def __init__(self, point: DurabilityFaultPoint, occurrence: int) -> None:
        self.point = point
        self.occurrence = occurrence
        super().__init__(f"injected {point.value} fault at occurrence {occurrence}")


class InjectedCrash(BaseException):
    """A process-death simulation that is not caught by ``except Exception``."""

    def __init__(self, point: DurabilityFaultPoint, occurrence: int) -> None:
        self.point = point
        self.occurrence = occurrence
        super().__init__(f"injected {point.value} crash at occurrence {occurrence}")


@dataclass(frozen=True, slots=True)
class FaultEvent:
    point: DurabilityFaultPoint
    occurrence: int


@dataclass(frozen=True, slots=True)
class RecoveryRun:
    """Evidence from one injected operation and its recovery pass."""

    point: DurabilityFaultPoint
    occurrence: int
    interrupted: bool
    events: tuple[FaultEvent, ...]


@contextmanager
def _instrument_sqlite_connections(observer: Callable[[DurabilityFaultPoint], None]) -> Iterator[None]:
    """Release interrupted handles before recovery; retain caller ownership on success."""
    original_connect = sqlite3.connect
    connections: list[sqlite3.Connection] = []
    classes: dict[type[sqlite3.Connection], type[sqlite3.Connection]] = {}
    active = True

    def observe(point: DurabilityFaultPoint) -> None:
        if active:
            observer(point)

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        base = args[5] if len(args) > 5 else kwargs.get("factory", sqlite3.Connection)
        if base not in classes:
            classes[base] = _instrumented_connection_class(base, observe)
        if len(args) > 5:
            args = (*args[:5], classes[base], *args[6:])
        else:
            kwargs["factory"] = classes[base]
        connection: sqlite3.Connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    with patch.object(sqlite3, "connect", connect):
        try:
            yield
        except BaseException:
            active = False
            # A simulated process exit must release externally retained handles
            # too, before the caller opens a connection for recovery.
            with ExitStack() as cleanup:
                for connection in connections:
                    cleanup.callback(connection.close)
            raise
        finally:
            active = False


class DurabilityFaultRegistry:
    """Install scoped faults at filesystem and SQLite durability seams.

    ``modules`` is accepted as an explicit documentation/coverage contract.
    The filesystem functions are patched process-wide for the short operation
    scope because ``pathlib.Path.unlink`` resolves ``os.unlink`` internally;
    the operation must therefore be isolated in a temporary archive.  The
    registry records every hit, making a point that was never reached visible
    rather than silently vacuous.
    """

    _POINTS = tuple(DurabilityFaultPoint)

    def __init__(self, *, modules: tuple[str, ...] = ()) -> None:
        self.modules = modules
        self._target: DurabilityFaultPoint | None = None
        self._target_occurrence = 1
        self._action = "raise"
        self._counts: dict[DurabilityFaultPoint, int] = dict.fromkeys(self._POINTS, 0)
        self.events: list[FaultEvent] = []

    @classmethod
    def points(cls) -> tuple[DurabilityFaultPoint, ...]:
        return cls._POINTS

    def arm(
        self,
        point: DurabilityFaultPoint | str,
        *,
        occurrence: int = 1,
        action: str = "raise",
    ) -> None:
        point = DurabilityFaultPoint(point)
        if occurrence < 1:
            raise ValueError("occurrence must be positive")
        if action not in {"raise", "kill"}:
            raise ValueError("action must be 'raise' or 'kill'")
        self._target = point
        self._target_occurrence = occurrence
        self._action = action

    def disarm(self) -> None:
        self._target = None

    def reset(self) -> None:
        self._counts = dict.fromkeys(self._POINTS, 0)
        self.events.clear()

    def count(self, point: DurabilityFaultPoint | str) -> int:
        return self._counts[DurabilityFaultPoint(point)]

    def _hit(self, point: DurabilityFaultPoint) -> None:
        self._counts[point] += 1
        event = FaultEvent(point, self._counts[point])
        self.events.append(event)
        if self._target == point and event.occurrence == self._target_occurrence:
            if self._action == "kill":
                raise InjectedCrash(point, event.occurrence)
            raise InjectedFault(point, event.occurrence)

    @contextmanager
    def installed(self) -> Iterator[DurabilityFaultRegistry]:
        """Patch the seams for one operation and restore them afterwards."""

        def fsync(fd: int) -> None:
            self._hit(DurabilityFaultPoint.FSYNC)
            return _real_fsync(fd)

        def replace(src: Any, dst: Any) -> None:
            self._hit(DurabilityFaultPoint.REPLACE)
            return _real_replace(src, dst)

        def unlink(path: Any, *args: Any, **kwargs: Any) -> None:
            self._hit(DurabilityFaultPoint.UNLINK)
            return _real_unlink(path, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(os, "fsync", fsync))
            stack.enter_context(patch.object(os, "replace", replace))
            stack.enter_context(patch.object(os, "unlink", unlink))
            stack.enter_context(_instrument_sqlite_connections(self._hit))
            yield self

    def run(
        self,
        point: DurabilityFaultPoint | str,
        operation: Callable[[], Any],
        *,
        recover: Callable[[], Any],
        assert_invariants: Callable[[], Any],
        occurrence: int = 1,
        action: str = "kill",
    ) -> RecoveryRun:
        """Crash one selected point, run production recovery, assert laws."""
        self.reset()
        self.arm(point, occurrence=occurrence, action=action)
        interrupted = False
        try:
            with self.installed():
                operation()
        except (InjectedFaultError, InjectedCrash):
            interrupted = True
        finally:
            self.disarm()
        if not interrupted:
            raise AssertionError(
                f"fault point {DurabilityFaultPoint(point).value!r} occurrence {occurrence} was not reached"
            )
        recover()
        assert_invariants()
        return RecoveryRun(DurabilityFaultPoint(point), occurrence, interrupted, tuple(self.events))


_real_fsync = os.fsync
_real_replace = os.replace
_real_unlink = os.unlink

# Short compatibility spelling for tests that describe the event as a fault
# rather than an error.  The concrete class keeps the repository's N818 error
# naming convention while the public fixture API stays readable.
InjectedFault = InjectedFaultError


@pytest.fixture
def durability_faults() -> Iterator[DurabilityFaultRegistry]:
    """Provide a fresh scoped registry for durability-law tests."""
    yield DurabilityFaultRegistry(
        modules=(
            "polylogue.storage.blob_store",
            "polylogue.security.excision",
            "polylogue.storage.blob_gc",
            "polylogue.sources.hooks",
            "polylogue.operations.mutation_transaction",
        )
    )


_MUTATING_STATEMENT = re.compile(r"^\s*(?:insert|update|delete|replace)\b", re.IGNORECASE)


def _instrumented_connection_class(
    base: type[sqlite3.Connection], observer: Callable[[DurabilityFaultPoint], None]
) -> type[sqlite3.Connection]:
    """Keep the requested factory's methods and native cursor ownership."""
    from polylogue.storage.io_phase_metrics import _MeasuredConnection, _MeasuredCursor

    default_cursor: type[sqlite3.Cursor] = _MeasuredCursor if issubclass(base, _MeasuredConnection) else sqlite3.Cursor

    def execute(self: sqlite3.Cursor, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        if _MUTATING_STATEMENT.match(sql):
            observer(DurabilityFaultPoint.STATEMENT)
        return default_cursor.execute(self, sql, parameters)

    def executemany(self: sqlite3.Cursor, sql: str, parameters: Any, /) -> sqlite3.Cursor:
        if _MUTATING_STATEMENT.match(sql):
            observer(DurabilityFaultPoint.STATEMENT)
        return default_cursor.executemany(self, sql, parameters)

    crashing_cursor = type("CrashingCursor", (default_cursor,), {"execute": execute, "executemany": executemany})

    def cursor(self: sqlite3.Connection, factory: Any = None) -> sqlite3.Cursor:
        selected: sqlite3.Cursor = base.cursor(self, crashing_cursor if factory is None else factory)
        return selected

    def commit(self: sqlite3.Connection) -> None:
        observer(DurabilityFaultPoint.COMMIT)
        base.commit(self)

    def exit_connection(
        self: sqlite3.Connection,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Any:
        # Native context-managed commits bypass the public commit method.
        if exc_type is None:
            observer(DurabilityFaultPoint.COMMIT)
        return base.__exit__(self, exc_type, exc, traceback)

    namespace: dict[str, Any] = {"cursor": cursor, "commit": commit, "__exit__": exit_connection}
    # The native Connection.execute/executemany create their cursor without
    # calling the Python-level cursor(); route those through it, once. A base
    # that already implements them in Python reaches cursor() itself.
    if base.execute is sqlite3.Connection.execute:

        def connection_execute(self: sqlite3.Connection, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
            return cursor(self).execute(sql, parameters)

        namespace["execute"] = connection_execute
    if base.executemany is sqlite3.Connection.executemany:

        def connection_executemany(self: sqlite3.Connection, sql: str, parameters: Any, /) -> sqlite3.Cursor:
            return cursor(self).executemany(sql, parameters)

        namespace["executemany"] = connection_executemany
    return type(f"Instrumented{base.__name__}", (base,), namespace)


@contextmanager
def crash_after_mutating_statements(count: int) -> Iterator[list[int]]:
    """Raise :class:`InjectedCrash` on the ``count``-th mutating statement.

    The crash lands inside whatever transaction is open at that point, which is
    the interesting case for resume: unlike a commit-boundary fault it leaves a
    partially applied statement sequence for SQLite to roll back. The yielded
    list carries ``[observed, target]`` so a caller can assert the fault fired.
    The fault is injected at the connection-factory seam: each connection is a
    real subclass of the factory its caller requested.
    """
    if count < 1:
        raise ValueError("crash count must be positive")
    budget = [0, count]

    def observe(point: DurabilityFaultPoint) -> None:
        if point is DurabilityFaultPoint.STATEMENT:
            budget[0] += 1
            if budget[0] == budget[1]:
                raise InjectedCrash(point, budget[1])

    with _instrument_sqlite_connections(observe):
        yield budget


__all__ = [
    "DurabilityFaultPoint",
    "DurabilityFaultRegistry",
    "FaultEvent",
    "InjectedCrash",
    "InjectedFault",
    "InjectedFaultError",
    "RecoveryRun",
    "crash_after_mutating_statements",
]
