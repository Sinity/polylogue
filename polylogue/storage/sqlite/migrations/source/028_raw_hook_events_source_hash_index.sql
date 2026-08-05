-- polylogue-fix-3806: candidate-relevant hook reconciliation needs indexed
-- source_path/blob_hash lookups instead of repeated unrelated-row scans.
CREATE INDEX IF NOT EXISTS idx_raw_hook_events_source_hash
ON raw_hook_events(source_path, blob_hash);
