-- migration-safety: additive-no-backup
CREATE TABLE IF NOT EXISTS source_attachments (
    source_generation_id TEXT NOT NULL REFERENCES source_generations(source_generation_id) ON DELETE CASCADE,
    reference_id         TEXT NOT NULL,
    origin               TEXT NOT NULL CHECK (length(trim(origin)) > 0),
    source_class         TEXT NOT NULL CHECK (length(trim(source_class)) > 0),
    reachability         TEXT NOT NULL CHECK(reachability IN ('current', 'unavailable')),
    reference_count      INTEGER NOT NULL CHECK(reference_count > 0),
    payload_identity     TEXT,
    blob_hash            BLOB CHECK(blob_hash IS NULL OR length(blob_hash) = 32),
    byte_count           INTEGER CHECK(byte_count IS NULL OR byte_count >= 0),
    disposition          TEXT NOT NULL CHECK(disposition IN (
        'pending', 'acquired', 'duplicate', 'expired', 'access_denied',
        'source_missing', 'malformed', 'policy_rejected', 'partial', 'interrupted'
    )),
    reason               TEXT,
    evidence_ref         TEXT,
    observed_at_ms       INTEGER NOT NULL CHECK(observed_at_ms >= 0),
    updated_at_ms        INTEGER NOT NULL CHECK(updated_at_ms >= 0),
    PRIMARY KEY(source_generation_id, reference_id),
    CHECK((disposition = 'acquired' AND reachability = 'current' AND blob_hash IS NOT NULL
           AND byte_count IS NOT NULL AND payload_identity IS NOT NULL AND reason IS NULL)
          OR (disposition != 'acquired' AND reason IS NOT NULL)),
    CHECK((blob_hash IS NULL) = (byte_count IS NULL))
) STRICT;
CREATE INDEX IF NOT EXISTS idx_source_attachments_census
ON source_attachments(source_generation_id, origin, source_class, reachability, disposition);
