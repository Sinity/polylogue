"""Cursor/source-file state DDL for the source_file_cursor table.

-- Slice B: cursor expansion columns added. SCHEMA_VERSION bump deferred to slice E (storage v2 + write gateway).
"""

from __future__ import annotations

SOURCE_FILE_CURSOR_DDL = """
    CREATE TABLE IF NOT EXISTS source_file_cursor (
        source_path TEXT PRIMARY KEY,
        st_dev INTEGER,
        st_ino INTEGER,
        st_size INTEGER,
        mtime_ns INTEGER,
        byte_offset INTEGER,
        last_complete_newline INTEGER,
        content_fingerprint TEXT,
        parser_fingerprint TEXT,
        source_generation INTEGER DEFAULT 0,
        failure_count INTEGER DEFAULT 0,
        next_retry_at REAL,
        excluded INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_source_file_cursor_excluded
    ON source_file_cursor(excluded)
    WHERE excluded != 0;
"""

__all__ = ["SOURCE_FILE_CURSOR_DDL"]
