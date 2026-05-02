"""Per-file processing cursor for live ingestion.

Stores enough file state for the watcher to avoid metadata-only freshness
decisions. The current watcher still reparses complete files, but skip
decisions are based on content fingerprint plus byte position so same-size
rewrites and truncation are detected.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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


class CursorStore:
    """SQLite-backed live cursor store keyed by source path."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(_DDL)
            self._ensure_columns(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
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
                    source_name
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
    ) -> None:
        now = datetime.now(UTC).isoformat()
        offset = byte_size if byte_offset is None else byte_offset
        newline_offset = offset if last_complete_newline is None else last_complete_newline
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
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (source_path) DO UPDATE SET
                    byte_size = excluded.byte_size,
                    byte_offset = excluded.byte_offset,
                    last_complete_newline = excluded.last_complete_newline,
                    record_count = excluded.record_count,
                    last_record_ts = excluded.last_record_ts,
                    parser_fingerprint = excluded.parser_fingerprint,
                    content_fingerprint = excluded.content_fingerprint,
                    source_name = excluded.source_name,
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
                    now,
                ),
            )
            conn.commit()


__all__ = ["CursorRecord", "CursorStore"]
