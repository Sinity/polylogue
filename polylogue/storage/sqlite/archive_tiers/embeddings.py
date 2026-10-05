"""Archive embedding DDL fragment for archive."""

from __future__ import annotations

from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import EMBEDDINGS_TABLE_SPECS

EMBEDDINGS_SCHEMA_VERSION = 1
EMBEDDING_DIMENSION = 1024

# Purchased vectors are not replayable from Source. This is the fresh-v1
# declaration, not an instruction to replace an existing paid tier. Completion
# facts certify a paid transaction and never authorize a durable mutation.
#
# polylogue-a7xr.27: each CREATE TABLE in this module renders from a TableColumnSpec
# in archive_tiers_specs.py, matching the index tier. `message_embeddings` is
# the one exception and states its reason: `USING vec0(...)` is an
# extension-defined virtual-table declaration, not a column list a
# TableColumnSpec can render.
EMBEDDINGS_DDL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
    vector_derivation_hash TEXT PRIMARY KEY,
    embedding float[{EMBEDDING_DIMENSION}],
    model TEXT
);

CREATE TABLE IF NOT EXISTS message_embeddings_meta (
    {EMBEDDINGS_TABLE_SPECS["message_embeddings_meta"].ddl_body}
) STRICT;

-- Rebuildable message_id -> vector_derivation_hash mapping. Lives in the
-- embeddings tier (not index.db) so this rekey never bumps INDEX_SCHEMA_VERSION:
-- the mapping is derived purely from a message's current embedder input text
-- and this tier's own vectors, both already scoped to embeddings.db.
-- One message has exactly one *current* vector_derivation_hash; many messages
-- (fork/resume/auto-compaction replays, or genuinely identical prose) may
-- point at the same hash -- that convergence is the dedup win.
CREATE TABLE IF NOT EXISTS message_embedding_refs (
    {EMBEDDINGS_TABLE_SPECS["message_embedding_refs"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_message_embedding_refs_hash
ON message_embedding_refs(vector_derivation_hash);

CREATE INDEX IF NOT EXISTS idx_message_embedding_refs_session
ON message_embedding_refs(session_id);

CREATE TABLE IF NOT EXISTS embedding_status (
    {EMBEDDINGS_TABLE_SPECS["embedding_status"].ddl_body}
) STRICT;

CREATE TABLE IF NOT EXISTS embedding_derivation_state (
    {EMBEDDINGS_TABLE_SPECS["embedding_derivation_state"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_embedding_derivation_pending
ON embedding_derivation_state(attempt_state, recipe_hash, session_id);

CREATE TABLE IF NOT EXISTS embedding_failures (
    {EMBEDDINGS_TABLE_SPECS["embedding_failures"].ddl_body}
) STRICT;

CREATE TABLE IF NOT EXISTS excision_embedding_completions (
    {EMBEDDINGS_TABLE_SPECS["excision_embedding_completions"].ddl_body}
) STRICT;

CREATE INDEX IF NOT EXISTS idx_embedding_failures_active
ON embedding_failures(lifecycle_state, updated_at_ms DESC, failure_id);

"""

__all__ = ["EMBEDDING_DIMENSION", "EMBEDDINGS_DDL", "EMBEDDINGS_SCHEMA_VERSION"]
