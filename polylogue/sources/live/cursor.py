"""Per-file processing cursor for live ingestion.

Records the file size we've already routed through the parse pipeline so the
watcher can skip files that haven't grown since we last saw them. Stored in a
small auxiliary table inside the main archive SQLite — no schema-version bump,
created lazily via ``CREATE TABLE IF NOT EXISTS`` on first use.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

_DDL = """
CREATE TABLE IF NOT EXISTS live_cursor (
    source_path TEXT PRIMARY KEY,
    byte_size INTEGER NOT NULL,
    record_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
)
"""


class CursorStore:
    """SQLite-backed map of ``source_path -> byte_size`` last parsed."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(_DDL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get(self, path: Path) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT byte_size FROM live_cursor WHERE source_path = ?",
                (str(path),),
            ).fetchone()
        return int(row[0]) if row else 0

    def set(self, path: Path, byte_size: int, *, record_count: int = 0) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_cursor (source_path, byte_size, record_count, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (source_path) DO UPDATE SET
                    byte_size = excluded.byte_size,
                    record_count = excluded.record_count,
                    updated_at = excluded.updated_at
                """,
                (str(path), byte_size, record_count, now),
            )
            conn.commit()


__all__ = ["CursorStore"]
