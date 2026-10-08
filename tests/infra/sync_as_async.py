"""Await-shaped view of a synchronous SQLite connection for async write helpers.

Fixture seeding sometimes needs an ``async`` writer helper that only calls
``execute`` and the cursor fetches; this adapter runs those calls on the
caller's own synchronous connection and transaction.
"""

from __future__ import annotations

import sqlite3
from typing import cast


class AsyncCursorView:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    async def fetchone(self) -> object | None:
        return cast(object | None, self._cursor.fetchone())

    async def fetchall(self) -> list[object]:
        return cast(list[object], self._cursor.fetchall())


class AsyncConnectionView:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> AsyncCursorView:
        return AsyncCursorView(self._conn.execute(sql, parameters))
