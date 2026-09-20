"""Opt-in SQLite work-unit counters for complexity-law tests."""

from __future__ import annotations

import os
import re
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

_DERIVED_SURFACES = (
    "messages_fts",
    "messages_fts_identity",
    "session_work_events_fts",
    "action_pairs",
    "delegation_facts",
    "delegation_refresh_scope",
)
_SQL_SPACE = re.compile(r"\s+")


def _database_name(database: object) -> str:
    """Classify a sqlite connection without retaining private path content."""
    value = os.fspath(database) if isinstance(database, os.PathLike) else str(database)
    if "index.db" in value:
        return "index"
    if "source.db" in value:
        return "source"
    if "ops.db" in value:
        return "ops"
    if "user.db" in value:
        return "user"
    if "embeddings.db" in value:
        return "embeddings"
    return "other"


def _normalize_sql(sql: str) -> str:
    return _SQL_SPACE.sub(" ", sql).strip().lower()


def _mentions_derived_surface(sql: str) -> bool:
    return any(surface in sql for surface in _DERIVED_SURFACES)


def _is_archive_wide_derived_statement(sql: str) -> bool:
    """Recognize the deleted qsagp shape, without naming a private call site."""
    if not _mentions_derived_surface(sql):
        return False
    if " values " in sql:
        return False
    if sql.startswith("delete from "):
        return " where " not in sql
    if sql.startswith("insert into messages_fts"):
        return "target.session_id = b.session_id" not in sql
    if sql.startswith("insert into action_pairs"):
        return "where u.session_id =" not in sql
    if sql.startswith("insert or replace into delegation_refresh_scope"):
        return "select session_id from sessions" in sql
    return False


@dataclass(slots=True)
class SQLiteWorkCounter:
    """Count SQLite VM work and derived-surface statements by tier.

    The counter is deliberately attached at ``sqlite3.connect`` rather than
    to a test double. Production code opens the connections, installs the
    normal schema/indexes, and executes the real SQL. Progress callbacks are
    sampled every ``step_interval`` VM instructions, which is sufficient for
    comparing shape while keeping sparse CI fixtures fast.
    """

    step_interval: int = 32
    statements_by_database: Counter[str] = field(default_factory=Counter)
    vm_steps_by_database: Counter[str] = field(default_factory=Counter)
    derived_vm_steps_by_database: Counter[str] = field(default_factory=Counter)
    archive_wide_derived_statements_by_database: Counter[str] = field(default_factory=Counter)
    connections_by_database: Counter[str] = field(default_factory=Counter)
    _current_sql_by_connection: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    def attach(
        self,
        real_connect: Callable[..., sqlite3.Connection],
        *args: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        """Open one real connection and attach trace/progress callbacks."""
        connection = real_connect(*args, **kwargs)
        database = _database_name(args[0] if args else kwargs.get("database", ""))
        connection_id = id(connection)
        self.connections_by_database[database] += 1

        def trace(sql: str) -> None:
            normalized = _normalize_sql(sql)
            self.statements_by_database[database] += 1
            self._current_sql_by_connection[connection_id] = normalized
            if _is_archive_wide_derived_statement(normalized):
                self.archive_wide_derived_statements_by_database[database] += 1

        def progress() -> int:
            self.vm_steps_by_database[database] += self.step_interval
            current_sql = self._current_sql_by_connection.get(connection_id, "")
            if _mentions_derived_surface(current_sql):
                self.derived_vm_steps_by_database[database] += self.step_interval
            return 0

        connection.set_trace_callback(trace)
        connection.set_progress_handler(progress, self.step_interval)
        return connection

    def metric(self, name: str, database: str = "index") -> int:
        """Read one named counter for a database tier."""
        counters = {
            "statements": self.statements_by_database,
            "vm_steps": self.vm_steps_by_database,
            "derived_vm_steps": self.derived_vm_steps_by_database,
            "archive_wide_derived_statements": self.archive_wide_derived_statements_by_database,
            "connections": self.connections_by_database,
        }
        try:
            return int(counters[name][database])
        except KeyError as exc:
            raise KeyError(f"unknown SQLite work metric {name!r}") from exc

    def summary(self) -> str:
        return (
            "SQLiteWorkCounter("
            f"statements={dict(self.statements_by_database)}, "
            f"vm_steps={dict(self.vm_steps_by_database)}, "
            f"derived_vm_steps={dict(self.derived_vm_steps_by_database)}, "
            f"archive_wide_derived_statements={dict(self.archive_wide_derived_statements_by_database)}"
            ")"
        )


@contextmanager
def sqlite_work_counter(*, step_interval: int = 32) -> Iterator[SQLiteWorkCounter]:
    """Count work on all SQLite connections opened inside the context."""
    if step_interval < 1:
        raise ValueError("step_interval must be positive")
    counter = SQLiteWorkCounter(step_interval=step_interval)
    real_connect = sqlite3.connect

    def counted_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        return counter.attach(real_connect, *args, **kwargs)

    with patch.object(sqlite3, "connect", counted_connect):
        yield counter


_MUTATION = re.compile(r"^\s*(?:insert|update|delete|replace)\b", re.IGNORECASE)


def _tier_name(database: object) -> str:
    """Name the archive tier a connection belongs to, for write attribution."""
    value = os.fspath(database) if isinstance(database, os.PathLike) else str(database)
    for tier in ("source", "index", "ops", "user", "audit", "embeddings"):
        if f"{tier}.db" in value:
            return tier
    return "other"


@contextmanager
def mutating_statements() -> Iterator[list[tuple[str, str]]]:
    """Record every mutating SQL statement executed inside the context.

    The recorder attaches at ``sqlite3.connect`` so production code opens its
    own connections and runs its own SQL; nothing is emulated. Each entry is
    ``(tier, normalized_sql)``. This is the direct instrument for the
    derivation law that a second pass over unchanged inputs writes nothing.
    """
    recorded: list[tuple[str, str]] = []
    real_connect: Callable[..., sqlite3.Connection] = sqlite3.connect

    def recording_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection: sqlite3.Connection = real_connect(*args, **kwargs)
        tier = _tier_name(args[0] if args else kwargs.get("database", ""))

        def trace(sql: str) -> None:
            normalized = _normalize_sql(sql)
            if _MUTATION.match(normalized):
                recorded.append((tier, normalized))

        connection.set_trace_callback(trace)
        return connection

    with patch.object(sqlite3, "connect", recording_connect):
        yield recorded


__all__ = ["SQLiteWorkCounter", "mutating_statements", "sqlite_work_counter"]
