"""Runtime index extensions that are safe to ensure on existing archives."""

from __future__ import annotations

import sqlite3

import aiosqlite


def ensure_runtime_indexes_sync(conn: sqlite3.Connection) -> None:
    del conn


async def ensure_runtime_indexes_async(conn: aiosqlite.Connection) -> None:
    del conn


__all__ = [
    "ensure_runtime_indexes_async",
    "ensure_runtime_indexes_sync",
]
