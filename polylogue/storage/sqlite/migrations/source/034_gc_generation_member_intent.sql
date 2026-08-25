-- migration-safety: additive-no-backup
-- Durable exact member intent closes the GC unlink-before-evidence window.
CREATE TABLE gc_generation_members (
    generation_id            TEXT NOT NULL REFERENCES gc_generations(generation_id) ON DELETE CASCADE,
    blob_hash                BLOB NOT NULL CHECK(length(blob_hash) = 32),
    candidate_liveness       TEXT NOT NULL CHECK(candidate_liveness = 'unreferenced'),
    candidate_mtime_ns       INTEGER NOT NULL CHECK(candidate_mtime_ns >= 0),
    candidate_size_bytes     INTEGER NOT NULL CHECK(candidate_size_bytes >= 0),
    source_schema_version    INTEGER NOT NULL CHECK(source_schema_version >= 0),
    index_schema_version     INTEGER NOT NULL CHECK(index_schema_version >= 0),
    index_generation         TEXT NOT NULL,
    archive_identity_digest  TEXT NOT NULL CHECK(length(archive_identity_digest) = 64),
    code_identity            TEXT NOT NULL,
    intent_committed_at_ms   INTEGER NOT NULL CHECK(intent_committed_at_ms >= 0),
    outcome                  TEXT NOT NULL DEFAULT 'pending'
        CHECK(outcome IN ('pending', 'removed', 'reconciled_removed', 'skipped_still_live', 'failed')),
    outcome_at_ms            INTEGER CHECK(outcome_at_ms >= 0),
    outcome_detail           TEXT,
    PRIMARY KEY(generation_id, blob_hash),
    CHECK((outcome = 'pending') = (outcome_at_ms IS NULL))
) STRICT;

CREATE INDEX idx_gc_generation_members_pending
ON gc_generation_members(generation_id, blob_hash)
WHERE outcome = 'pending';
