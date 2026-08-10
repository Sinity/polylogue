CREATE TABLE raw_authority_artifact_census_receipts (
    receipt_id              TEXT PRIMARY KEY,
    receipt_sha256          TEXT NOT NULL UNIQUE CHECK(length(receipt_sha256) = 64),
    receipt_json            TEXT NOT NULL,
    backup_manifest_path    TEXT NOT NULL,
    applied_at_ms           INTEGER NOT NULL CHECK(applied_at_ms >= 0),
    tool_version            TEXT NOT NULL
) STRICT;

CREATE INDEX idx_raw_authority_artifact_census_receipts_applied_at
ON raw_authority_artifact_census_receipts(applied_at_ms);

CREATE TABLE raw_authority_artifact_census_checkpoints (
    census_id               TEXT PRIMARY KEY,
    universe_sha256         TEXT NOT NULL CHECK(length(universe_sha256) = 64),
    candidate_count         INTEGER NOT NULL CHECK(candidate_count >= 0),
    next_after_raw_id       TEXT,
    last_receipt_id         TEXT REFERENCES raw_authority_artifact_census_receipts(receipt_id),
    completed_at_ms         INTEGER,
    created_at_ms           INTEGER NOT NULL CHECK(created_at_ms >= 0)
) STRICT;

CREATE TABLE raw_authority_artifact_census_checkpoint_members (
    census_id               TEXT NOT NULL REFERENCES raw_authority_artifact_census_checkpoints(census_id) ON DELETE CASCADE,
    ordinal                 INTEGER NOT NULL CHECK(ordinal >= 0),
    raw_id                  TEXT NOT NULL REFERENCES raw_sessions(raw_id),
    PRIMARY KEY (census_id, raw_id),
    UNIQUE (census_id, ordinal)
) STRICT;

CREATE INDEX idx_raw_authority_artifact_census_checkpoint_members_page
ON raw_authority_artifact_census_checkpoint_members(census_id, ordinal);
