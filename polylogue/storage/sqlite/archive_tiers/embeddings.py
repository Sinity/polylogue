"""Archive embedding DDL fragment for archive."""

from __future__ import annotations

EMBEDDINGS_SCHEMA_VERSION = 1
EMBEDDING_DIMENSION = 1024

EMBEDDINGS_DDL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
    message_id TEXT PRIMARY KEY,
    embedding float[{EMBEDDING_DIMENSION}],
    +session_id TEXT,
    +origin TEXT
);

CREATE TABLE IF NOT EXISTS message_embeddings_meta (
    message_id      TEXT PRIMARY KEY,
    model           TEXT NOT NULL,
    dimension       INTEGER NOT NULL CHECK(dimension = {EMBEDDING_DIMENSION}),
    content_hash    BLOB NOT NULL CHECK(length(content_hash) = 32),
    embedded_at_ms  INTEGER,
    needs_reindex   INTEGER NOT NULL DEFAULT 0 CHECK(needs_reindex IN (0, 1))
) STRICT;

CREATE TABLE IF NOT EXISTS embedding_status (
    session_id                 TEXT PRIMARY KEY,
    origin                     TEXT NOT NULL DEFAULT '',
    message_count_embedded     INTEGER NOT NULL DEFAULT 0 CHECK(message_count_embedded >= 0),
    last_embedded_at_ms        INTEGER,
    needs_reindex              INTEGER NOT NULL DEFAULT 0 CHECK(needs_reindex IN (0, 1)),
    error_message              TEXT
) STRICT;

"""

__all__ = ["EMBEDDING_DIMENSION", "EMBEDDINGS_DDL", "EMBEDDINGS_SCHEMA_VERSION"]
