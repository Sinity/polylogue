CREATE TABLE IF NOT EXISTS raw_failure_disposition_receipts (
    raw_id                     TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    artifact_id                TEXT NOT NULL UNIQUE REFERENCES raw_artifacts(artifact_id),
    origin                     TEXT NOT NULL CHECK(origin IN (
        'claude-code-session', 'codex-session', 'gemini-cli-session',
        'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export',
        'chatgpt-export', 'claude-ai-export', 'claude-design-session',
        'aistudio-drive', 'unknown-export'
    )),
    source_path                TEXT NOT NULL,
    source_index               INTEGER NOT NULL,
    blob_hash                  BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                  INTEGER NOT NULL CHECK(blob_size >= 0),
    previous_parse_error       TEXT NOT NULL,
    previous_validation_status TEXT,
    disposition_kind           TEXT NOT NULL CHECK(disposition_kind IN (
        'terminal_corrupt_input',
        'terminal_unsupported_shape'
    )),
    manifest_sha256            TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
    disposed_at_ms             INTEGER NOT NULL CHECK(disposed_at_ms >= 0),
    tool_version               TEXT NOT NULL,
    backup_manifest_path       TEXT NOT NULL,
    detail                     TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_failure_disposition_receipts_disposed_at
ON raw_failure_disposition_receipts(disposed_at_ms);
