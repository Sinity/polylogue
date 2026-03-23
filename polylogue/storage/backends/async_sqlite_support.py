"""Shared connection helpers for the async SQLite backend."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

import polylogue.paths as _paths
from polylogue.storage.backends.connection import DB_TIMEOUT


def default_db_path() -> Path:
    """Return the default database path (same as sync backend)."""
    return _paths.data_home() / "polylogue.db"


async def configure_connection(conn: aiosqlite.Connection) -> None:
    """Apply canonical connection settings."""
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute(f"PRAGMA busy_timeout = {DB_TIMEOUT * 1000}")


__all__ = ["configure_connection", "default_db_path"]
