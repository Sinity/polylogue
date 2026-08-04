-- polylogue-s8s54: durable receipt for the operator-authorized repair of
-- browser-capture raw_sessions rows stamped unknown-export before the fixed
-- full-envelope provider probe landed.
--
-- This receipt proves the source-tier mutation only. The generated
-- sessions.session_id in index.db is intentionally left untouched and must be
-- repaired by the normal reparse/materialization route afterward.
CREATE TABLE IF NOT EXISTS raw_unknown_export_reclassification_receipts (
    raw_id                  TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    previous_origin         TEXT NOT NULL CHECK(previous_origin = 'unknown-export'),
    new_origin              TEXT NOT NULL CHECK(new_origin = 'chatgpt-export'),
    previous_capture_mode   TEXT,
    new_capture_mode        TEXT NOT NULL CHECK(new_capture_mode = 'chatgpt'),
    embedded_provider       TEXT NOT NULL CHECK(embedded_provider = 'chatgpt'),
    source_path             TEXT NOT NULL,
    blob_hash               BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size               INTEGER NOT NULL CHECK(blob_size >= 0),
    reclassified_at_ms      INTEGER NOT NULL CHECK(reclassified_at_ms >= 0),
    tool_version            TEXT NOT NULL,
    backup_manifest_path    TEXT NOT NULL,
    index_reparse_required  INTEGER NOT NULL CHECK(index_reparse_required = 1),
    detail                  TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_unknown_export_reclassification_receipts_reclassified_at
ON raw_unknown_export_reclassification_receipts(reclassified_at_ms);
