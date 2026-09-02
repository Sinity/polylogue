-- Rebuild raw-acquisition tables to the vocabulary-neutral canonical shape.

CREATE TABLE raw_sessions__040 (
    raw_id TEXT PRIMARY KEY,
    origin TEXT NOT NULL,
    capture_mode TEXT,
    native_id TEXT,
    source_path TEXT NOT NULL,
    source_index INTEGER NOT NULL DEFAULT 0,
    blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size INTEGER NOT NULL CHECK(blob_size >= 0),
    acquired_at_ms INTEGER NOT NULL,
    file_mtime_ms INTEGER,
    parsed_at_ms INTEGER,
    parse_error TEXT,
    validated_at_ms INTEGER,
    validation_status TEXT,
    validation_error TEXT,
    validation_drift_count INTEGER NOT NULL DEFAULT 0 CHECK(validation_drift_count >= 0),
    validation_mode TEXT,
    detection_warnings_json TEXT NOT NULL DEFAULT '[]',
    logical_source_key TEXT,
    revision_kind TEXT NOT NULL DEFAULT 'unknown' CHECK(revision_kind IN ('full', 'append', 'unknown')),
    source_revision TEXT,
    predecessor_source_revision TEXT,
    predecessor_raw_id TEXT,
    baseline_raw_id TEXT,
    append_start_offset INTEGER CHECK(append_start_offset >= 0),
    append_end_offset INTEGER CHECK(append_end_offset > append_start_offset),
    acquisition_generation INTEGER CHECK(acquisition_generation >= 0),
    revision_authority TEXT NOT NULL DEFAULT 'quarantined',
    revision_authority_evidence TEXT CHECK(revision_authority_evidence IS NULL OR revision_authority_evidence IN ('live_source_verification_v1')),
    detected_provider TEXT
) STRICT;
INSERT INTO raw_sessions__040 SELECT raw_id, origin, capture_mode, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms, file_mtime_ms, parsed_at_ms, parse_error, validated_at_ms, validation_status, validation_error, validation_drift_count, validation_mode, detection_warnings_json, logical_source_key, revision_kind, source_revision, predecessor_source_revision, predecessor_raw_id, baseline_raw_id, append_start_offset, append_end_offset, acquisition_generation, revision_authority, revision_authority_evidence, detected_provider FROM raw_sessions;
DROP TABLE raw_sessions;
ALTER TABLE raw_sessions__040 RENAME TO raw_sessions;

CREATE TABLE raw_artifacts__040 (
    artifact_id TEXT PRIMARY KEY,
    raw_id TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    origin TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_index INTEGER NOT NULL DEFAULT 0,
    artifact_kind TEXT NOT NULL,
    support_status TEXT NOT NULL,
    classification_reason TEXT NOT NULL,
    parse_as_session INTEGER NOT NULL DEFAULT 0 CHECK(parse_as_session IN (0, 1)),
    schema_eligible INTEGER NOT NULL DEFAULT 0 CHECK(schema_eligible IN (0, 1)),
    malformed_jsonl_lines INTEGER NOT NULL DEFAULT 0 CHECK(malformed_jsonl_lines >= 0),
    decode_error TEXT,
    cohort_id TEXT,
    link_group_key TEXT,
    sidecar_agent_type TEXT,
    first_observed_at_ms INTEGER NOT NULL,
    last_observed_at_ms INTEGER NOT NULL
) STRICT;
INSERT INTO raw_artifacts__040 SELECT artifact_id, raw_id, origin, source_path, source_index, artifact_kind, support_status, classification_reason, parse_as_session, schema_eligible, malformed_jsonl_lines, decode_error, cohort_id, link_group_key, sidecar_agent_type, first_observed_at_ms, last_observed_at_ms FROM raw_artifacts;
DROP TABLE raw_artifacts;
ALTER TABLE raw_artifacts__040 RENAME TO raw_artifacts;

CREATE TABLE raw_hook_events__040 (hook_event_id TEXT PRIMARY KEY, origin TEXT NOT NULL, native_id TEXT, session_native_id TEXT, source_path TEXT NOT NULL, event_type TEXT NOT NULL, payload_json TEXT NOT NULL, observed_at_ms INTEGER NOT NULL, blob_hash BLOB CHECK(blob_hash IS NULL OR length(blob_hash) = 32)) STRICT;
INSERT INTO raw_hook_events__040 SELECT hook_event_id, origin, native_id, session_native_id, source_path, event_type, payload_json, observed_at_ms, blob_hash FROM raw_hook_events;
DROP TABLE raw_hook_events;
ALTER TABLE raw_hook_events__040 RENAME TO raw_hook_events;

CREATE TABLE history_sidecars__040 (sidecar_id TEXT PRIMARY KEY, origin TEXT NOT NULL, source_path TEXT NOT NULL, payload_json TEXT NOT NULL, observed_at_ms INTEGER NOT NULL, content_hash BLOB NOT NULL CHECK(length(content_hash) = 32)) STRICT;
INSERT INTO history_sidecars__040 SELECT sidecar_id, origin, source_path, payload_json, observed_at_ms, content_hash FROM history_sidecars;
DROP TABLE history_sidecars;
ALTER TABLE history_sidecars__040 RENAME TO history_sidecars;

CREATE TABLE otlp_spans__040 (span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, parent_span_id TEXT, origin TEXT, session_native_id TEXT, name TEXT NOT NULL, kind TEXT, attributes_json TEXT NOT NULL DEFAULT '{}', events_json TEXT NOT NULL DEFAULT '[]', started_at_ms INTEGER, ended_at_ms INTEGER, received_at_ms INTEGER NOT NULL) STRICT;
INSERT INTO otlp_spans__040 SELECT span_id, trace_id, parent_span_id, origin, session_native_id, name, kind, attributes_json, events_json, started_at_ms, ended_at_ms, received_at_ms FROM otlp_spans;
DROP TABLE otlp_spans;
ALTER TABLE otlp_spans__040 RENAME TO otlp_spans;

CREATE TABLE raw_authority_verdicts__040 (raw_id TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE, logical_source_key TEXT NOT NULL, verdict TEXT NOT NULL, cohort_member_count INTEGER NOT NULL CHECK(cohort_member_count >= 1), cohort_fingerprint BLOB NOT NULL CHECK(length(cohort_fingerprint) = 32), computed_at_ms INTEGER NOT NULL CHECK(computed_at_ms >= 0)) STRICT;
INSERT INTO raw_authority_verdicts__040 SELECT raw_id, logical_source_key, verdict, cohort_member_count, cohort_fingerprint, computed_at_ms FROM raw_authority_verdicts;
DROP TABLE raw_authority_verdicts;
ALTER TABLE raw_authority_verdicts__040 RENAME TO raw_authority_verdicts;

CREATE TABLE raw_capture_observations__040 (raw_id TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE, capture_mode TEXT NOT NULL, first_observed_at_ms INTEGER NOT NULL CHECK(first_observed_at_ms >= 0), PRIMARY KEY(raw_id, capture_mode)) STRICT;
INSERT INTO raw_capture_observations__040 SELECT raw_id, capture_mode, first_observed_at_ms FROM raw_capture_observations;
DROP TABLE raw_capture_observations;
ALTER TABLE raw_capture_observations__040 RENAME TO raw_capture_observations;

CREATE TABLE raw_live_source_reconciliation_receipts__040 (raw_id TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE, verdict TEXT NOT NULL CHECK(verdict IN ('exact_match', 'codex_header_strip_match')), previous_revision_authority TEXT NOT NULL, source_path TEXT NOT NULL, blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32), blob_size INTEGER NOT NULL CHECK(blob_size >= 0), compared_at_ms INTEGER NOT NULL CHECK(compared_at_ms >= 0), tool_version TEXT NOT NULL, backup_manifest_path TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '') STRICT;
INSERT INTO raw_live_source_reconciliation_receipts__040 SELECT raw_id, verdict, previous_revision_authority, source_path, blob_hash, blob_size, compared_at_ms, tool_version, backup_manifest_path, detail FROM raw_live_source_reconciliation_receipts;
DROP TABLE raw_live_source_reconciliation_receipts;
ALTER TABLE raw_live_source_reconciliation_receipts__040 RENAME TO raw_live_source_reconciliation_receipts;

CREATE TABLE raw_membership_writeback_receipts__040 (raw_id TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE, logical_source_key TEXT NOT NULL, provider_session_id TEXT NOT NULL, membership_decision TEXT NOT NULL, previous_revision_authority TEXT NOT NULL, promoted_at_ms INTEGER NOT NULL CHECK(promoted_at_ms >= 0), tool_version TEXT NOT NULL, backup_manifest_path TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '') STRICT;
INSERT INTO raw_membership_writeback_receipts__040 SELECT raw_id, logical_source_key, provider_session_id, membership_decision, previous_revision_authority, promoted_at_ms, tool_version, backup_manifest_path, detail FROM raw_membership_writeback_receipts;
DROP TABLE raw_membership_writeback_receipts;
ALTER TABLE raw_membership_writeback_receipts__040 RENAME TO raw_membership_writeback_receipts;

CREATE TABLE raw_append_chain_backfill_receipts__040 (raw_id TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE, logical_source_key TEXT, source_path TEXT NOT NULL, blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32), blob_size INTEGER NOT NULL CHECK(blob_size >= 0), append_start_offset INTEGER NOT NULL CHECK(append_start_offset >= 0), append_end_offset INTEGER NOT NULL CHECK(append_end_offset > append_start_offset), matched_after_codex_header_strip INTEGER NOT NULL CHECK(matched_after_codex_header_strip IN (0, 1)), previous_revision_authority TEXT NOT NULL, compared_at_ms INTEGER NOT NULL CHECK(compared_at_ms >= 0), tool_version TEXT NOT NULL, backup_manifest_path TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '') STRICT;
INSERT INTO raw_append_chain_backfill_receipts__040 SELECT raw_id, logical_source_key, source_path, blob_hash, blob_size, append_start_offset, append_end_offset, matched_after_codex_header_strip, previous_revision_authority, compared_at_ms, tool_version, backup_manifest_path, detail FROM raw_append_chain_backfill_receipts;
DROP TABLE raw_append_chain_backfill_receipts;
ALTER TABLE raw_append_chain_backfill_receipts__040 RENAME TO raw_append_chain_backfill_receipts;

CREATE TABLE raw_session_memberships__040 (raw_id TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE, logical_source_key TEXT NOT NULL, provider_session_id TEXT NOT NULL, source_revision TEXT NOT NULL, normalized_content_hash BLOB NOT NULL CHECK(length(normalized_content_hash) = 32), message_count INTEGER NOT NULL CHECK(message_count >= 0), predecessor_raw_id TEXT, acquisition_generation INTEGER NOT NULL DEFAULT 0 CHECK(acquisition_generation >= 0), revision_authority TEXT NOT NULL DEFAULT 'quarantined', decision TEXT, decided_at_ms INTEGER CHECK(decided_at_ms >= 0), PRIMARY KEY(raw_id, logical_source_key), CHECK((decision IS NULL) = (decided_at_ms IS NULL))) STRICT;
INSERT INTO raw_session_memberships__040 SELECT raw_id, logical_source_key, provider_session_id, source_revision, normalized_content_hash, message_count, predecessor_raw_id, acquisition_generation, revision_authority, decision, decided_at_ms FROM raw_session_memberships;
DROP TABLE raw_session_memberships;
ALTER TABLE raw_session_memberships__040 RENAME TO raw_session_memberships;

CREATE INDEX idx_history_sidecars_path_hash ON history_sidecars(origin, source_path, content_hash);
CREATE INDEX idx_otlp_spans_session ON otlp_spans(origin, session_native_id, started_at_ms DESC) WHERE session_native_id IS NOT NULL;
CREATE INDEX idx_otlp_spans_trace ON otlp_spans(trace_id, started_at_ms DESC);
CREATE INDEX idx_raw_append_chain_backfill_receipts_compared_at ON raw_append_chain_backfill_receipts(compared_at_ms);
CREATE UNIQUE INDEX idx_raw_artifacts_failure_identity ON raw_artifacts(raw_id, origin, source_path, source_index) WHERE artifact_kind IN ('deferred_hot_jsonl_capture', 'deferred_claude_code_partial_jsonl', 'deferred_cas_frontier', 'deferred_codex_cas_frontier', 'terminal_corrupt_input', 'terminal_superseded_deferred_cas_frontier', 'terminal_unknown_json_decode', 'terminal_unknown_export_no_session', 'terminal_unsupported_shape');
CREATE INDEX idx_raw_artifacts_raw_id ON raw_artifacts(raw_id);
CREATE UNIQUE INDEX idx_raw_artifacts_source_identity ON raw_artifacts(origin, source_path, source_index) WHERE artifact_kind NOT IN ('deferred_hot_jsonl_capture', 'deferred_claude_code_partial_jsonl', 'deferred_cas_frontier', 'deferred_codex_cas_frontier', 'terminal_corrupt_input', 'terminal_superseded_deferred_cas_frontier', 'terminal_unknown_json_decode', 'terminal_unknown_export_no_session', 'terminal_unsupported_shape');
CREATE INDEX idx_raw_authority_verdicts_logical_source ON raw_authority_verdicts(logical_source_key);
CREATE INDEX idx_raw_capture_observations_raw_id ON raw_capture_observations(raw_id);
CREATE INDEX idx_raw_hook_events_session ON raw_hook_events(origin, session_native_id, observed_at_ms);
CREATE INDEX idx_raw_hook_events_source_hash ON raw_hook_events(source_path, blob_hash);
CREATE INDEX idx_raw_live_source_reconciliation_receipts_compared_at ON raw_live_source_reconciliation_receipts(compared_at_ms);
CREATE INDEX idx_raw_membership_writeback_receipts_promoted_at ON raw_membership_writeback_receipts(promoted_at_ms);
CREATE INDEX idx_raw_session_memberships_logical ON raw_session_memberships(logical_source_key, acquisition_generation, raw_id);
CREATE INDEX idx_raw_session_memberships_pending ON raw_session_memberships(raw_id) WHERE decision IS NULL OR decision IN ('ambiguous', 'deferred');
CREATE INDEX idx_raw_sessions_blob_hash ON raw_sessions(blob_hash);
CREATE INDEX idx_raw_sessions_blob_hash_raw_id ON raw_sessions(blob_hash, raw_id);
CREATE INDEX idx_raw_sessions_logical_revision ON raw_sessions(logical_source_key, acquisition_generation, raw_id) WHERE logical_source_key IS NOT NULL;
CREATE INDEX idx_raw_sessions_origin ON raw_sessions(origin);
CREATE INDEX idx_raw_sessions_origin_native ON raw_sessions(origin, native_id) WHERE native_id IS NOT NULL;
CREATE INDEX idx_raw_sessions_parse_ready ON raw_sessions(raw_id) WHERE parsed_at_ms IS NULL AND validated_at_ms IS NOT NULL AND (validation_status IS NULL OR validation_status != 'failed');
CREATE INDEX idx_raw_sessions_raw_authority_census_candidates ON raw_sessions(revision_authority, parse_error);
CREATE INDEX idx_raw_sessions_source_path ON raw_sessions(source_path, source_index);

CREATE TRIGGER invalidate_pending_raw_authority_artifact_census_checkpoint_on_raw_delete
BEFORE DELETE ON raw_sessions
WHEN EXISTS (SELECT 1 FROM raw_authority_artifact_census_checkpoint_members AS member JOIN raw_authority_artifact_census_checkpoints AS checkpoint ON checkpoint.census_id = member.census_id WHERE member.raw_id = OLD.raw_id AND checkpoint.completed_at_ms IS NULL)
BEGIN
    DELETE FROM raw_authority_artifact_census_checkpoints WHERE census_id IN (SELECT member.census_id FROM raw_authority_artifact_census_checkpoint_members AS member JOIN raw_authority_artifact_census_checkpoints AS checkpoint ON checkpoint.census_id = member.census_id WHERE member.raw_id = OLD.raw_id AND checkpoint.completed_at_ms IS NULL);
END;

CREATE TABLE IF NOT EXISTS raw_legacy_append_resynthesis_receipts (
    raw_id                          TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key              TEXT NOT NULL,
    source_path                     TEXT NOT NULL,
    blob_hash                       BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                       INTEGER NOT NULL CHECK(blob_size >= 0),
    append_start_offset             INTEGER NOT NULL CHECK(append_start_offset >= 0),
    append_end_offset               INTEGER NOT NULL CHECK(append_end_offset > append_start_offset),
    matched_after_codex_header_strip INTEGER NOT NULL CHECK(matched_after_codex_header_strip IN (0, 1)),
    previous_revision_authority     TEXT NOT NULL CHECK(previous_revision_authority = 'quarantined'),
    source_prefix_sha256            TEXT NOT NULL CHECK(length(source_prefix_sha256) = 64),
    source_size                     INTEGER NOT NULL CHECK(source_size >= append_end_offset),
    source_mtime_ns                 INTEGER NOT NULL,
    source_ctime_ns                 INTEGER NOT NULL,
    observed_at_ms                  INTEGER NOT NULL CHECK(observed_at_ms >= 0),
    detail                          TEXT NOT NULL DEFAULT ''
) STRICT;
CREATE INDEX IF NOT EXISTS idx_raw_legacy_append_resynthesis_receipts_observed_at
ON raw_legacy_append_resynthesis_receipts(observed_at_ms);
