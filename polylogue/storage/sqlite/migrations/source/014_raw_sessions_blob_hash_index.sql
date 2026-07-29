-- migration-safety: additive-no-backup
-- Index-only performance fix for the blob-GC reference check
-- (storage/blob_gc.py:_archive_reference_surfaces / blob_publication.py:
-- _is_referenced both run "SELECT 1 FROM raw_sessions WHERE blob_hash = ?"
-- once per candidate blob file on disk). raw_sessions had no index whose
-- leading column is blob_hash, so this was a full table scan per candidate;
-- over a store of ~100K blobs the daemon's periodic blob-GC pass (every
-- 900s) took minutes per tick while holding the process-wide write
-- coordinator gate for the whole call (blob-store audit finding, no bead
-- filed separately -- see the audit PR).
--
-- Pure index add: cannot lose or corrupt data, trivially reversible
-- (DROP INDEX), so no backup manifest is required.
CREATE INDEX IF NOT EXISTS idx_raw_sessions_blob_hash
ON raw_sessions(blob_hash);
