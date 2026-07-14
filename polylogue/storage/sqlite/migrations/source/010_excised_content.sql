-- Durable removed-content ledger for standalone/off-mode excision
-- (polylogue-27m). Purely additive: one new table, no existing column or
-- constraint changes.
--
-- A row here is the authoritative "this content is forgotten on purpose"
-- marker. The acquire-time write chokepoint (``write_source_raw_session``)
-- refuses to re-store a raw payload whose ``blob_hash`` matches
-- ``removed_hash``, so an ordinary re-ingest of unmodified source files
-- cannot resurrect excised content even after an index.db rebuild.
CREATE TABLE excised_content (
    removed_hash    BLOB NOT NULL CHECK(length(removed_hash) = 32),
    hash_kind       TEXT NOT NULL DEFAULT 'blob_hash' CHECK(hash_kind IN ('blob_hash')),
    reason          TEXT NOT NULL,
    actor           TEXT NOT NULL,
    prior_revision  TEXT,
    span_start      INTEGER CHECK(span_start IS NULL OR span_start >= 0),
    span_end        INTEGER CHECK(span_end IS NULL OR span_end > span_start),
    excised_at_ms   INTEGER NOT NULL,
    PRIMARY KEY(removed_hash, hash_kind)
) STRICT;
