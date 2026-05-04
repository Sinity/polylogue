"""Per-file processing cursor for live ingestion.

Cursor state enables content-aware skip decisions: same-size rewrites,
truncation, and parser version changes are detected via content fingerprint
comparison rather than relying on file size alone.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from polylogue.storage.sqlite.connection_profile import open_connection

_DDL = """
CREATE TABLE IF NOT EXISTS live_cursor (
    source_path TEXT PRIMARY KEY,
    byte_size INTEGER NOT NULL,
    byte_offset INTEGER NOT NULL DEFAULT 0,
    last_complete_newline INTEGER NOT NULL DEFAULT 0,
    record_count INTEGER NOT NULL DEFAULT 0,
    last_record_ts TEXT,
    parser_fingerprint TEXT,
    content_fingerprint TEXT,
    source_name TEXT,
    -- Stat metadata for fast-path skip (Phase 2 K: #620)
    st_dev INTEGER,
    st_ino INTEGER,
    mtime_ns INTEGER,
    -- Quarantine fields (Phase 2 K: #620)
    source_generation INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    next_retry_at TEXT,
    excluded INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class CursorRecord:
    """Stored live cursor state for one source file."""

    source_path: str
    byte_size: int
    byte_offset: int
    last_complete_newline: int
    record_count: int
    updated_at: str
    last_record_ts: str | None = None
    parser_fingerprint: str | None = None
    content_fingerprint: str | None = None
    source_name: str | None = None
    st_dev: int | None = None
    st_ino: int | None = None
    mtime_ns: int | None = None
    source_generation: int = 0
    failure_count: int = 0
    next_retry_at: str | None = None
    excluded: bool | int = False


class CursorStore:
    """SQLite-backed live cursor store keyed by source path."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(_DDL)
            self._ensure_columns(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = open_connection(self._db_path, timeout=10.0)
        return conn

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(live_cursor)")}
        columns = {
            "byte_offset": "INTEGER NOT NULL DEFAULT 0",
            "last_complete_newline": "INTEGER NOT NULL DEFAULT 0",
            "last_record_ts": "TEXT",
            "parser_fingerprint": "TEXT",
            "content_fingerprint": "TEXT",
            "source_name": "TEXT",
            "st_dev": "INTEGER",
            "st_ino": "INTEGER",
            "mtime_ns": "INTEGER",
            "source_generation": "INTEGER NOT NULL DEFAULT 0",
            "failure_count": "INTEGER NOT NULL DEFAULT 0",
            "next_retry_at": "TEXT",
            "excluded": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE live_cursor ADD COLUMN {name} {definition}")
        conn.commit()

    def get(self, path: Path) -> int:
        record = self.get_record(path)
        return record.byte_offset if record is not None else 0

    def get_record(self, path: Path) -> CursorRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    source_path,
                    byte_size,
                    byte_offset,
                    last_complete_newline,
                    record_count,
                    updated_at,
                    last_record_ts,
                    parser_fingerprint,
                    content_fingerprint,
                    source_name,
                    st_dev,
                    st_ino,
                    mtime_ns,
                    source_generation,
                    failure_count,
                    next_retry_at,
                    excluded
                FROM live_cursor
                WHERE source_path = ?
                """,
                (str(path),),
            ).fetchone()
        if row is None:
            return None
        return CursorRecord(
            source_path=str(row[0]),
            byte_size=int(row[1]),
            byte_offset=int(row[2]),
            last_complete_newline=int(row[3]),
            record_count=int(row[4]),
            updated_at=str(row[5]),
            last_record_ts=row[6],
            parser_fingerprint=row[7],
            content_fingerprint=row[8],
            source_name=row[9],
            st_dev=row[10],
            st_ino=row[11],
            mtime_ns=row[12],
            source_generation=int(row[13] or 0) if row[13] is not None else 0,
            failure_count=int(row[14] or 0) if row[14] is not None else 0,
            next_retry_at=row[15],
            excluded=bool(row[16]) if row[16] is not None else False,
        )

    def set(
        self,
        path: Path,
        byte_size: int,
        *,
        byte_offset: int | None = None,
        last_complete_newline: int | None = None,
        record_count: int = 0,
        last_record_ts: str | None = None,
        parser_fingerprint: str | None = None,
        content_fingerprint: str | None = None,
        source_name: str | None = None,
        st_dev: int | None = None,
        st_ino: int | None = None,
        mtime_ns: int | None = None,
        source_generation: int | None = None,
        failure_count: int | None = None,
        next_retry_at: str | None = None,
        excluded: bool | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        offset = byte_size if byte_offset is None else byte_offset
        newline_offset = offset if last_complete_newline is None else last_complete_newline
        excluded_int = 1 if excluded else 0
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_cursor (
                    source_path,
                    byte_size,
                    byte_offset,
                    last_complete_newline,
                    record_count,
                    last_record_ts,
                    parser_fingerprint,
                    content_fingerprint,
                    source_name,
                    st_dev,
                    st_ino,
                    mtime_ns,
                    source_generation,
                    failure_count,
                    next_retry_at,
                    excluded,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (source_path) DO UPDATE SET
                    byte_size = excluded.byte_size,
                    byte_offset = excluded.byte_offset,
                    last_complete_newline = excluded.last_complete_newline,
                    record_count = excluded.record_count,
                    last_record_ts = excluded.last_record_ts,
                    parser_fingerprint = excluded.parser_fingerprint,
                    content_fingerprint = excluded.content_fingerprint,
                    source_name = excluded.source_name,
                    st_dev = excluded.st_dev,
                    st_ino = excluded.st_ino,
                    mtime_ns = excluded.mtime_ns,
                    source_generation = excluded.source_generation,
                    failure_count = excluded.failure_count,
                    next_retry_at = excluded.next_retry_at,
                    excluded = excluded.excluded,
                    updated_at = excluded.updated_at
                """,
                (
                    str(path),
                    byte_size,
                    offset,
                    newline_offset,
                    record_count,
                    last_record_ts,
                    parser_fingerprint,
                    content_fingerprint,
                    source_name,
                    st_dev,
                    st_ino,
                    mtime_ns,
                    source_generation or 0,
                    failure_count or 0,
                    next_retry_at,
                    excluded_int,
                    now,
                ),
            )
            conn.commit()

    def mark_failed(self, path: Path) -> None:
        """Increment failure count and set exponential backoff."""
        record = self.get_record(path)
        if record is None:
            return
        failures = record.failure_count + 1
        delay_s = min(60 * (2 ** (failures - 1)), 3600)  # cap at 1 hour
        retry_at = datetime.now(UTC).timestamp() + delay_s
        with self._connect() as conn:
            conn.execute(
                "UPDATE live_cursor SET failure_count = ?, next_retry_at = ? WHERE source_path = ?",
                (failures, datetime.fromtimestamp(retry_at, tz=UTC).isoformat(), str(path)),
            )
            conn.commit()

    def mark_excluded(self, path: Path) -> None:
        """Quarantine a source file (poison pill)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE live_cursor SET excluded = 1 WHERE source_path = ?",
                (str(path),),
            )
            conn.commit()

    def reset_failures(self, path: Path) -> None:
        """Clear failure count and backoff after a successful parse."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE live_cursor SET failure_count = 0, next_retry_at = NULL WHERE source_path = ?",
                (str(path),),
            )
            conn.commit()

    def list_excluded(self) -> list[str]:
        """Return quarantined source paths."""
        with self._connect() as conn:
            rows = conn.execute("SELECT source_path FROM live_cursor WHERE excluded = 1").fetchall()
        return [str(row[0]) for row in rows]

    def list_failed_with_retry(self) -> list[str]:
        """Return sources that have failed and are NOT currently in backoff."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_path FROM live_cursor WHERE failure_count > 0 AND excluded = 0 AND (next_retry_at IS NULL OR next_retry_at <= ?)",
                (now,),
            ).fetchall()
        return [str(row[0]) for row in rows]


__all__ = ["CursorRecord", "CursorStore"]
