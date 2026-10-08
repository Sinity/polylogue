"""Async sqlite3 adapter shared by accepted-marker and excision storage laws.

The ingest acceptance/outbox wiring laws that lived here drove the deleted
batch ingest route; their surviving laws are in tests/unit/sinex/test_obligations.py.
"""

from __future__ import annotations

import sqlite3
from typing import cast


class _AsyncCursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    async def fetchone(self) -> object | None:
        return cast(object | None, self._cursor.fetchone())

    async def fetchall(self) -> list[object]:
        return cast(list[object], self._cursor.fetchall())


class _AsyncConnection:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> _AsyncCursor:
        return _AsyncCursor(self._conn.execute(sql, parameters))
