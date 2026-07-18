"""Archive embedding DDL fragment for archive."""

from __future__ import annotations

EMBEDDINGS_SCHEMA_VERSION = 3
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
    needs_reindex   INTEGER NOT NULL DEFAULT 0 CHECK(needs_reindex IN (0, 1)),
    recipe_hash     BLOB CHECK(recipe_hash IS NULL OR length(recipe_hash) = 32),
    derivation_key  BLOB CHECK(derivation_key IS NULL OR length(derivation_key) = 32),
    generation      INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0)
) STRICT;

CREATE TABLE IF NOT EXISTS embedding_status (
    session_id                 TEXT PRIMARY KEY,
    origin                     TEXT NOT NULL DEFAULT '',
    message_count_embedded     INTEGER NOT NULL DEFAULT 0 CHECK(message_count_embedded >= 0),
    last_embedded_at_ms        INTEGER,
    needs_reindex              INTEGER NOT NULL DEFAULT 0 CHECK(needs_reindex IN (0, 1)),
    error_message              TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS embedding_derivation_state (
    session_id             TEXT PRIMARY KEY,
    origin                 TEXT NOT NULL DEFAULT '',
    generation             INTEGER NOT NULL CHECK(generation >= 1),
    derivation_key         BLOB NOT NULL CHECK(length(derivation_key) = 32),
    source_hash            BLOB NOT NULL CHECK(length(source_hash) = 32),
    recipe_hash            BLOB NOT NULL CHECK(length(recipe_hash) = 32),
    output_contract_hash   BLOB NOT NULL CHECK(length(output_contract_hash) = 32),
    attempt_state          TEXT NOT NULL CHECK(attempt_state IN (
        'pending', 'succeeded', 'failed_retryable', 'failed_terminal'
    )),
    message_count          INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0),
    updated_at_ms          INTEGER NOT NULL CHECK(updated_at_ms >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_embedding_derivation_pending
ON embedding_derivation_state(attempt_state, recipe_hash, session_id);

CREATE TABLE IF NOT EXISTS embedding_failures (
    failure_id          TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    origin              TEXT NOT NULL,
    message_refs_json   TEXT NOT NULL DEFAULT '[]',
    provider            TEXT NOT NULL,
    model               TEXT NOT NULL,
    error_class         TEXT NOT NULL,
    error_message       TEXT NOT NULL,
    retryable           INTEGER NOT NULL CHECK(retryable IN (0, 1)),
    lifecycle_state     TEXT NOT NULL CHECK(lifecycle_state IN (
        'retryable', 'terminal', 'acknowledged', 'superseded', 'resolved'
    )),
    created_at_ms       INTEGER NOT NULL,
    updated_at_ms       INTEGER NOT NULL,
    resolved_at_ms      INTEGER,
    resolution_action   TEXT,
    resolution_note     TEXT,
    superseded_by       TEXT,
    generation          INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
    derivation_key      BLOB CHECK(derivation_key IS NULL OR length(derivation_key) = 32),
    source_hash         BLOB CHECK(source_hash IS NULL OR length(source_hash) = 32),
    recipe_hash         BLOB CHECK(recipe_hash IS NULL OR length(recipe_hash) = 32)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_embedding_failures_active
ON embedding_failures(lifecycle_state, updated_at_ms DESC, failure_id);

"""

__all__ = ["EMBEDDING_DIMENSION", "EMBEDDINGS_DDL", "EMBEDDINGS_SCHEMA_VERSION"]
