-- migration-safety: additive-no-backup
CREATE TABLE IF NOT EXISTS raw_legacy_append_resynthesis_receipts (
    raw_id                          TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key              TEXT NOT NULL,
    source_path                     TEXT NOT NULL,
    blob_hash                       BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                       INTEGER NOT NULL CHECK(blob_size >= 0),
    append_start_offset             INTEGER NOT NULL CHECK(append_start_offset >= 0),
    append_end_offset               INTEGER NOT NULL CHECK(append_end_offset > append_start_offset),
    matched_after_codex_header_strip INTEGER NOT NULL CHECK(matched_after_codex_header_strip IN (0, 1)),
    previous_revision_authority     TEXT NOT NULL CHECK(previous_revision_authority = 'quarantined'),
    source_prefix_sha256            TEXT NOT NULL CHECK(length(source_prefix_sha256) = 64),
    source_size                     INTEGER NOT NULL CHECK(source_size >= append_end_offset),
    source_mtime_ns                 INTEGER NOT NULL,
    source_ctime_ns                 INTEGER NOT NULL,
    observed_at_ms                  INTEGER NOT NULL CHECK(observed_at_ms >= 0),
    detail                          TEXT NOT NULL DEFAULT ''
) STRICT;
CREATE INDEX IF NOT EXISTS idx_raw_legacy_append_resynthesis_receipts_observed_at
ON raw_legacy_append_resynthesis_receipts(observed_at_ms);
