"""Archive embedding DDL fragment for archive."""

from __future__ import annotations

from polylogue.core.enums import Origin
from polylogue.storage.sqlite.archive_tiers.common import check

EMBEDDINGS_SCHEMA_VERSION = 1
EMBEDDING_DIMENSION = 1024

EMBEDDINGS_DDL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
    message_id TEXT PRIMARY KEY,
    embedding float[{EMBEDDING_DIMENSION}],
    +session_id TEXT,
    +origin TEXT
);

CREATE TABLE IF NOT EXISTS embeddings_meta (
    target_id       TEXT PRIMARY KEY,
    target_type     TEXT NOT NULL CHECK(target_type IN ('message', 'session')),
    model           TEXT NOT NULL,
    dimension       INTEGER NOT NULL CHECK(dimension > 0),
    embedded_at_ms  INTEGER NOT NULL,
    content_hash    BLOB CHECK(content_hash IS NULL OR length(content_hash) = 32),
    origin          TEXT CHECK ({check("origin", Origin)})
) STRICT;

CREATE INDEX IF NOT EXISTS idx_embeddings_meta_type
ON embeddings_meta(target_type);

CREATE INDEX IF NOT EXISTS idx_embeddings_meta_origin
ON embeddings_meta(origin)
WHERE origin IS NOT NULL;

CREATE TABLE IF NOT EXISTS embedding_status (
    session_id              TEXT PRIMARY KEY,
    origin                  TEXT NOT NULL CHECK ({check("origin", Origin)}),
    message_count_embedded  INTEGER NOT NULL DEFAULT 0 CHECK(message_count_embedded >= 0),
    last_embedded_at_ms     INTEGER,
    needs_reindex           INTEGER NOT NULL DEFAULT 0 CHECK(needs_reindex IN (0, 1)),
    error_message           TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS idx_embedding_status_needs
ON embedding_status(needs_reindex)
WHERE needs_reindex = 1;

"""

__all__ = ["EMBEDDING_DIMENSION", "EMBEDDINGS_DDL", "EMBEDDINGS_SCHEMA_VERSION"]
