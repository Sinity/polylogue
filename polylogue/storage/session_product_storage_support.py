"""Shared storage helpers for session-product writes."""

from __future__ import annotations

import sqlite3

_SYNC_COLUMN_CACHE: dict[tuple[int, str], bool] = {}


def table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    key = (id(conn), f"{table}.{column}")
    cached = _SYNC_COLUMN_CACHE.get(key)
    if cached is not None:
        return cached
    found = any(
        str(row[1]) == column for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    )
    _SYNC_COLUMN_CACHE[key] = found
    return found


__all__ = ["table_has_column"]
