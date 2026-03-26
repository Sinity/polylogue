"""Shared support helpers for durable session-product async queries."""

from __future__ import annotations

import aiosqlite

_ASYNC_COLUMN_CACHE: dict[tuple[int, str], bool] = {}


async def table_has_column(conn: aiosqlite.Connection, table: str, column: str) -> bool:
    key = (id(conn), f"{table}.{column}")
    cached = _ASYNC_COLUMN_CACHE.get(key)
    if cached is not None:
        return cached
    cursor = await conn.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    found = any(str(row["name"] if "name" in row else row[1]) == column for row in rows)
    _ASYNC_COLUMN_CACHE[key] = found
    return found


__all__ = ["table_has_column"]
