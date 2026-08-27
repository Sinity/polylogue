"""Archive embedding DDL fragment for archive."""

from __future__ import annotations

EMBEDDINGS_SCHEMA_VERSION = 5
EMBEDDING_DIMENSION = 1024

# v5: vectors are addressed by the complete typed provider request and output
# contract, including exact normalized input. Metadata is complete by schema,
# so a legacy row with unknown output identity cannot authorize reuse.
#
# embeddings.db is a rebuildable derived tier (no migration chain): a schema
# mismatch blue-green-replaces the tier from source
# (`polylogue ops reset --index && polylogued run`), so this is an in-place DDL
# edit, not an additive migration.
EMBEDDINGS_DDL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
    vector_derivation_hash TEXT PRIMARY KEY,
    embedding float[{EMBEDDING_DIMENSION}],
    model TEXT
);

CREATE TABLE IF NOT EXISTS message_embeddings_meta (
    vector_derivation_hash   BLOB PRIMARY KEY CHECK(length(vector_derivation_hash) = 32),
    model                  TEXT NOT NULL,
    dimension              INTEGER NOT NULL CHECK(dimension = {EMBEDDING_DIMENSION}),
    embedded_at_ms         INTEGER,
    recipe_hash            BLOB NOT NULL CHECK(length(recipe_hash) = 32),
    output_contract_hash   BLOB NOT NULL CHECK(length(output_contract_hash) = 32)
) STRICT;

-- Rebuildable message_id -> vector_derivation_hash mapping. Lives in the
-- embeddings tier (not index.db) so this rekey never bumps INDEX_SCHEMA_VERSION:
-- the mapping is derived purely from a message's current embedder input text
-- and this tier's own vectors, both already scoped to embeddings.db.
-- One message has exactly one *current* vector_derivation_hash; many messages
-- (fork/resume/auto-compaction replays, or genuinely identical prose) may
-- point at the same hash -- that convergence is the dedup win.
CREATE TABLE IF NOT EXISTS message_embedding_refs (
    message_id            TEXT PRIMARY KEY,
    session_id            TEXT NOT NULL,
    origin                TEXT NOT NULL,
    vector_derivation_hash  BLOB NOT NULL CHECK(length(vector_derivation_hash) = 32),
    embedded_at_ms        INTEGER
) STRICT;

CREATE INDEX IF NOT EXISTS idx_message_embedding_refs_hash
ON message_embedding_refs(vector_derivation_hash);

CREATE INDEX IF NOT EXISTS idx_message_embedding_refs_session
ON message_embedding_refs(session_id);

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
