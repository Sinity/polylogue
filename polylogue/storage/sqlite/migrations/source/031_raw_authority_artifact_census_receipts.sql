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
    universe_complete       INTEGER NOT NULL DEFAULT 0 CHECK(universe_complete IN (0, 1)),
    snapshot_max_raw_rowid  INTEGER NOT NULL CHECK(snapshot_max_raw_rowid >= 0),
    materialized_after_rowid INTEGER NOT NULL DEFAULT 0 CHECK(materialized_after_rowid >= 0),
    index_generation        TEXT NOT NULL,
    index_identity_sha256   TEXT NOT NULL CHECK(length(index_identity_sha256) = 64),
    next_after_raw_id       TEXT,
    last_receipt_id         TEXT REFERENCES raw_authority_artifact_census_receipts(receipt_id),
    completed_at_ms         INTEGER,
    created_at_ms           INTEGER NOT NULL CHECK(created_at_ms >= 0)
) STRICT;

CREATE TABLE raw_authority_artifact_census_checkpoint_members (
    census_id               TEXT NOT NULL REFERENCES raw_authority_artifact_census_checkpoints(census_id) ON DELETE CASCADE,
    ordinal                 INTEGER NOT NULL CHECK(ordinal >= 0),
    raw_id                  TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    PRIMARY KEY (census_id, raw_id),
    UNIQUE (census_id, ordinal)
) STRICT;

CREATE INDEX idx_raw_authority_artifact_census_checkpoint_members_page
ON raw_authority_artifact_census_checkpoint_members(census_id, ordinal);

CREATE TRIGGER invalidate_pending_raw_authority_artifact_census_checkpoint_on_raw_delete
BEFORE DELETE ON raw_sessions
WHEN EXISTS (
    SELECT 1
    FROM raw_authority_artifact_census_checkpoint_members AS member
    JOIN raw_authority_artifact_census_checkpoints AS checkpoint
      ON checkpoint.census_id = member.census_id
    WHERE member.raw_id = OLD.raw_id
      AND checkpoint.completed_at_ms IS NULL
)
BEGIN
    DELETE FROM raw_authority_artifact_census_checkpoints
    WHERE census_id IN (
        SELECT member.census_id
        FROM raw_authority_artifact_census_checkpoint_members AS member
        JOIN raw_authority_artifact_census_checkpoints AS checkpoint
          ON checkpoint.census_id = member.census_id
        WHERE member.raw_id = OLD.raw_id
          AND checkpoint.completed_at_ms IS NULL
    );
END;

CREATE INDEX idx_raw_sessions_raw_authority_census_candidates
ON raw_sessions(revision_authority, parse_error);
