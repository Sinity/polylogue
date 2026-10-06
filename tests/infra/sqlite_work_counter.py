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
    "action_pairs",
    "delegation_facts",
    "delegation_refresh_scope",
)
_SQL_SPACE = re.compile(r"\s+")
_SQL_LITERAL = re.compile(r"'(?:[^']|'')*'")
_IDENTIFIER = r"""(?:\w+|"(?:[^"]|"")+"|`(?:[^`]|``)+`|\[[^\]]+\]|'(?:[^']|'')+')"""
_QUALIFIED_IDENTIFIER = rf"(?:{_IDENTIFIER}\s*\.\s*)?{_IDENTIFIER}"
_CONFLICT = r"(?: or (?:rollback|abort|replace|fail|ignore))?"
_WRITE_TARGET = re.compile(
    rf"\b(delete from|update{_CONFLICT}|insert{_CONFLICT} into|replace into) "
    rf"({_QUALIFIED_IDENTIFIER})(?=\s|\(|;|$)"
)
_QUOTED_TOKEN = re.compile(r"""'(?:[^']|'')*'|"(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]*\]""")
_SQL_TOKEN_OR_COMMENT = re.compile(rf"{_QUOTED_TOKEN.pattern}|/\*[\s\S]*?(?:\*/|$)|--[^\r\n]*")
_BOUND_VALUE = r"(?:\?|\d+|(?:new|old)\.\w+)"
_BOUND_VALUES = rf"{_BOUND_VALUE}(?:\s*,\s*{_BOUND_VALUE})*"


def _scoped_content_mutation(sql: str, table: str) -> bool:
    """Recognize the concrete DELETE/UPDATE scopes used by the SQL owners."""
    where = sql.partition(" where ")[2].rstrip(";")
    if table in {"action_pairs", "delegation_facts"}:
        column = "session_id" if table == "action_pairs" else "(?:parent_session_id|child_session_id)"
        predicate = rf"{column} (?:= {_BOUND_VALUE}|in \(\s*{_BOUND_VALUES}\s*\))"
        return re.fullmatch(rf"{predicate}(?: or {predicate})?", where) is not None
    if re.fullmatch(r"rowid (?:= (?:\d+|(?:new|old)\.rowid)|in \(\s*\d+(?:\s*,\s*\d+)*\s*\))", where):
        return True
    return (
        re.fullmatch(
            rf"rowid in \( select blocks.rowid from blocks where blocks.session_id in \(\s*{_BOUND_VALUES}\s*\) \)"
            r"(?: and rowid in \(select id from messages_fts_docsize\))?",
            where,
        )
        is not None
    )


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
    # SQLite treats genuine comments as whitespace, including between header
    # keywords and qualified target slots. Quoted tokens own their contents:
    # comment-looking bytes there must not eat a later operation or predicate.
    uncommented = _SQL_TOKEN_OR_COMMENT.sub(lambda token: " " if token[0].startswith(("/*", "--")) else token[0], sql)
    return _SQL_SPACE.sub(" ", uncommented).strip().lower()


def _mentions_derived_surface(sql: str) -> bool:
    return any(surface in sql for surface in _DERIVED_SURFACES)


def _target_table(identifier: str) -> str:
    """Lower only the table slot, including SQLite's target-context quotes."""
    return list(re.finditer(_IDENTIFIER, identifier))[-1].group(0).strip("\"`[]'")


def _canonical_write_target(sql: str) -> tuple[str, str, str] | None:
    # A quoted value can contain an apparent UPDATE header. Only an operation
    # outside a quoted token owns a target; its target itself may be quoted.
    quoted = iter(_QUOTED_TOKEN.finditer(sql))
    token = next(quoted, None)
    for target in _WRITE_TARGET.finditer(sql):
        while token is not None and token.end() <= target.start():
            token = next(quoted, None)
        if token is not None and token.start() <= target.start() < token.end():
            continue
        operation = target[1]
        operation = "update" if operation.startswith("update") else operation
        operation = "insert into" if operation.startswith("insert") else operation
        table = _target_table(target[2])
        tail = sql[target.end() :]
        canonical = sql[: target.start()] + operation + " " + table + tail
        return canonical, table, tail
    return None


def _is_archive_wide_derived_statement(sql: str) -> bool:
    """Recognize global writes to derived content, excluding scoped work."""
    dropped = re.fullmatch(rf"drop table(?: if exists)? ({_QUALIFIED_IDENTIFIER});?", sql)
    if dropped:
        return _target_table(dropped[1]) in _DERIVED_SURFACES[:-1]
    if not sql.startswith(("delete ", "update ", "insert ", "replace ", "with ")):
        return False
    target = _canonical_write_target(sql)
    if target is None or target[1] not in _DERIVED_SURFACES:
        return False
    sql, table, tail = target
    # Parse target slots before neutralizing expanded values: SQLite accepts
    # single-quoted identifiers in these slots as well as string literals.
    sql = _SQL_LITERAL.sub("?", sql)
    operation = _WRITE_TARGET.search(sql)
    assert operation is not None
    operation_name = operation[1]
    if table == "delegation_refresh_scope":
        # Clearing the working allow-list does not rewrite archive content.
        # Populating it with every session does initiate a global refresh.
        if operation_name == "delete from":
            return False
        if re.match(r"\s*\([^)]*\) values\b", tail):
            return False
        return not (
            "where child_session_id = new.session_id" in sql
            or "select old.resolved_dst_session_id as parent_session_id union select new.resolved_dst_session_id" in sql
        )
    if operation_name in {"delete from", "update"}:
        return not _scoped_content_mutation(sql, table)
    values = re.match(r"\s*(?:\(([^)]*)\))?\s*values\b", tail)
    if table == "messages_fts" and values:
        # Ordinary row writes declare rowid/text. Every other VALUES shape,
        # including quoted FTS5 control columns, cannot claim this exemption.
        columns = values[1]
        return columns is None or [_target_table(column.strip()) for column in columns.split(",")] != ["rowid", "text"]
    if values:
        return False
    if table in {"messages_fts", "messages_fts_identity"}:
        return not (
            re.search(rf"from blocks as b where b.session_id = {_BOUND_VALUE} and b.search_text != \?$", sql)
            or (
                re.match(
                    rf"^with raw_target_sessions\(session_id\) as \( values \({_BOUND_VALUE}\)"
                    rf"(?:, \({_BOUND_VALUE}\))* \), target_sessions as \( select distinct session_id "
                    r"from raw_target_sessions \) insert(?: or replace)? into messages_fts(?:_identity)? ",
                    sql,
                )
                and sql.endswith(
                    "join target_sessions as target on target.session_id = b.session_id where b.search_text != ?"
                )
            )
            or (sql.partition(" select ")[2].startswith("new.rowid, ") and " from " not in sql)
            or re.search(
                rf"where d.id is null and b.search_text != \? and b.rowid > {_BOUND_VALUE} "
                rf"and b.rowid <= {_BOUND_VALUE} \) insert into messages_fts \(rowid, text\) "
                r"select rowid, [^;]+ from missing$",
                sql,
            )
            or re.search(
                rf"where b.search_text != \? and b.rowid > {_BOUND_VALUE} and b.rowid <= {_BOUND_VALUE} "
                r"on conflict\(rowid\) do update set block_id = excluded.block_id, "
                r"source_hash = excluded.source_hash, recipe_id = excluded.recipe_id "
                r"where messages_fts_identity.block_id != excluded.block_id "
                r"or messages_fts_identity.source_hash is not excluded.source_hash "
                r"or messages_fts_identity.recipe_id != excluded.recipe_id "
                r"on conflict\(block_id\) do update set rowid = excluded.rowid, "
                r"source_hash = excluded.source_hash, recipe_id = excluded.recipe_id$",
                sql,
            )
        )
    if table == "action_pairs":
        # The shared association owner's tool-use and tool-result block
        # sources and the unpaired tool-use branch must all be session-bound.
        blocks = r"from blocks (?:u|r)(?: indexed by idx_blocks_session_position)? where"
        return not all(
            re.search(predicate, sql)
            for predicate in (
                rf"{blocks} u.block_type=\? and u.session_id = {_BOUND_VALUE} union all ",
                rf"{blocks} r.block_type=\? and r.session_id = {_BOUND_VALUE} \), association_events as ",
                rf"{blocks} u.block_type = \? and \(u.tool_id is null or u.tool_id = \?\) "
                rf"and u.session_id = {_BOUND_VALUE}$",
            )
        )
    # The declared view reads only the current delegation_refresh_scope. Its
    # archive-wide population is counted above. Other INSERT/SELECT shapes
    # cannot claim this scope implicitly.
    return not sql.endswith("from delegation_facts_source")


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
            # SQLite annotates real FTS5 shadow operations with a leading
            # "--" in its trace. Preserve that execution evidence for VM
            # accounting; SQL comment removal owns write classification only.
            self._current_sql_by_connection[connection_id] = sql.lower()
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
