-- Replace the single raw-artifact coordinate carrier with two explicit
-- uniqueness domains. Ordinary artifact observations remain unique by source
-- coordinate. Typed raw-failure evidence is unique per retained raw and
-- exact coordinate, so a later acquisition cannot move authority away from
-- an older failed raw.
DROP INDEX IF EXISTS idx_raw_artifacts_source_identity;

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
    'terminal_unsupported_shape'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_artifacts_failure_identity
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
    'terminal_unsupported_shape'
);
