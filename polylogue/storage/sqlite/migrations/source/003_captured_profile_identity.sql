-- Receipt creation and complementary failure partitions require the normal verified backup.
CREATE TABLE raw_profile_identity_receipts (
    raw_id TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    profile_key TEXT NOT NULL CHECK(length(profile_key) = 12)
) STRICT;

DROP INDEX idx_raw_artifacts_source_identity;
CREATE UNIQUE INDEX idx_raw_artifacts_source_identity
ON raw_artifacts(origin, source_path, source_index)
WHERE artifact_kind NOT IN (
    'deferred_hot_jsonl_capture',
    'deferred_claude_code_partial_jsonl',
    'deferred_cas_frontier',
    'deferred_codex_cas_frontier',
    'terminal_corrupt_input',
    'terminal_superseded_deferred_cas_frontier',
    'terminal_unknown_json_decode',
    'terminal_unknown_export_no_session',
    'terminal_unsupported_shape',
    'terminal_missing_source_coordinates',
    'terminal_missing_profile_identity'
);

DROP INDEX idx_raw_artifacts_failure_identity;
CREATE UNIQUE INDEX idx_raw_artifacts_failure_identity
ON raw_artifacts(raw_id, origin, source_path, source_index)
WHERE artifact_kind IN (
    'deferred_hot_jsonl_capture',
    'deferred_claude_code_partial_jsonl',
    'deferred_cas_frontier',
    'deferred_codex_cas_frontier',
    'terminal_corrupt_input',
    'terminal_superseded_deferred_cas_frontier',
    'terminal_unknown_json_decode',
    'terminal_unknown_export_no_session',
    'terminal_unsupported_shape',
    'terminal_missing_source_coordinates',
    'terminal_missing_profile_identity'
);
