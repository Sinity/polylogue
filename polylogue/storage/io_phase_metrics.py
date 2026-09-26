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
from collections.abc import Callable, Iterator
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
    if current_write_lease() is None:
        return False
    try:
        return require_write_lease("I/O phase ownership sample") is not None
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
    def execute(self, sql: str, parameters: Any = (), /) -> _MeasuredCursor:
        phase = _transaction_phase(sql)
        connection = cast(_MeasuredConnection, self.connection)
        tier = getattr(connection, "_metric_tier", None)
        if phase is None:
            return super().execute(sql, parameters)
        connection._metric_statement_phase = phase
        try:
            with timed_io_phase(tier, phase):
                return super().execute(sql, parameters)
        finally:
            connection._metric_statement_phase = None


class _MeasuredConnection(sqlite3.Connection):
    _metric_tier: Tier | None = None
    _metric_context_exit = False
    _metric_statement_phase: Phase | None = None

    @overload
    def cursor(self, factory: None = None) -> sqlite3.Cursor: ...

    @overload
    def cursor(self, factory: Callable[[sqlite3.Connection], _CursorT]) -> _CursorT: ...

    def cursor(self, factory: Callable[[sqlite3.Connection], sqlite3.Cursor] | None = None) -> sqlite3.Cursor:
        return super().cursor(factory or _MeasuredCursor)

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        phase = _transaction_phase(sql)
        if phase is None:
            return super().execute(sql, parameters)
        self._metric_statement_phase = phase
        try:
            with timed_io_phase(self._metric_tier, phase):
                return super().execute(sql, parameters)
        finally:
            self._metric_statement_phase = None

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


def connect_measured(database: str | Path, /, **kwargs: Any) -> sqlite3.Connection:
    """Time an actual returned SQLite handle without altering its PRAGMA policy."""
    tier = tier_for_path(database)
    owned = _inside_writer_lease()
    started = time.perf_counter_ns()
    succeeded = False
    try:
        conn = sqlite3.connect(str(database), factory=_MeasuredConnection, **kwargs)
        conn._metric_tier = tier
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
    "connect_measured",
    "io_phase_process_snapshot",
    "io_phase_snapshot",
    "record_io_phase",
    "tier_for_path",
    "timed_io_phase",
]
