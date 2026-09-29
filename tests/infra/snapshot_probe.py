"""Interleave a concurrent commit into a reader between two of its statements.

A reader that measures several relations with separate autocommit statements
sees each statement's own commit. ``CommitBetweenStatements`` wraps the
reader's connection, and right after the first statement matching ``trigger_sql``
has produced its rows it runs ``commit`` (a different connection's write and
commit). A reader that holds one snapshot keeps reading the state it started
from; an autocommit reader sees the commit in every later statement.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Sequence
from typing import Any


def _normalized(sql: str) -> str:
    return " ".join(sql.split())


class _MaterializedCursor:
    def __init__(self, rows: Sequence[Any]) -> None:
        self._rows = list(rows)

    def fetchone(self) -> Any:
        return self._rows.pop(0) if self._rows else None

    def fetchall(self) -> list[Any]:
        rows, self._rows = self._rows, []
        return rows

    def __iter__(self) -> Iterator[Any]:
        return iter(self.fetchall())


class CommitBetweenStatements:
    """Connection proxy that commits a concurrent write after one statement."""

    def __init__(self, conn: sqlite3.Connection, *, trigger_sql: str, commit: Callable[[], None]) -> None:
        self._conn = conn
        self._trigger_sql = _normalized(trigger_sql)
        self._commit = commit
        self.fired = False

    def execute(self, sql: str, *args: Any) -> Any:
        cursor = self._conn.execute(sql, *args)
        if self.fired or _normalized(sql) != self._trigger_sql:
            return cursor
        # Finish the statement before the other connection commits, so its
        # result belongs to the state this reader observed first.
        rows = cursor.fetchall()
        self.fired = True
        self._commit()
        return _MaterializedCursor(rows)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


__all__ = ["CommitBetweenStatements"]
