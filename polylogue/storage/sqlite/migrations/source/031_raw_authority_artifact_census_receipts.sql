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
