-- migration-safety: additive-no-backup
-- Durable exact member intent closes the GC unlink-before-evidence window.
CREATE TABLE gc_generation_members (
    generation_id            TEXT NOT NULL REFERENCES gc_generations(generation_id) ON DELETE CASCADE,
    blob_hash                BLOB NOT NULL CHECK(length(blob_hash) = 32),
    candidate_size_bytes     INTEGER NOT NULL CHECK(candidate_size_bytes >= 0),
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
