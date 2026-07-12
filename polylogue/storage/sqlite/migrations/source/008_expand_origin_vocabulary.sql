-- Copy-forward the source-tier tables whose Origin CHECK constraints are
-- generated from core.enums.Origin. SQLite cannot ALTER a CHECK constraint;
-- rebuild the affected durable tables transactionally after the migration
-- runner has verified a source-tier backup receipt.

-- These adjunct tables were introduced in the fresh bootstrap DDL before the
-- durable migration chain started tracking them.  Older valid source tiers can
-- therefore lack them.  Create empty v7-shaped tables so every supported
-- predecessor (not only a freshly bootstrapped v7 archive) follows the same
-- copy-forward path below.
CREATE TABLE IF NOT EXISTS raw_artifacts (
    artifact_id              TEXT PRIMARY KEY,
    raw_id                   TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    origin                   TEXT NOT NULL,
    source_path              TEXT NOT NULL,
    source_index             INTEGER NOT NULL DEFAULT 0,
    artifact_kind            TEXT NOT NULL,
    support_status           TEXT NOT NULL,
    classification_reason    TEXT NOT NULL,
    parse_as_session         INTEGER NOT NULL DEFAULT 0,
    schema_eligible          INTEGER NOT NULL DEFAULT 0,
    malformed_jsonl_lines    INTEGER NOT NULL DEFAULT 0,
    decode_error             TEXT,
    cohort_id                TEXT,
    link_group_key           TEXT,
    sidecar_agent_type       TEXT,
    first_observed_at_ms     INTEGER NOT NULL,
    last_observed_at_ms      INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS raw_hook_events (
    hook_event_id      TEXT PRIMARY KEY,
    origin             TEXT NOT NULL,
    native_id          TEXT,
    session_native_id  TEXT,
    source_path        TEXT NOT NULL,
    event_type         TEXT NOT NULL,
    payload_json       TEXT NOT NULL,
    observed_at_ms     INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS otlp_spans (
    span_id            TEXT PRIMARY KEY,
    trace_id           TEXT NOT NULL,
    parent_span_id     TEXT,
    origin             TEXT,
    session_native_id  TEXT,
    name               TEXT NOT NULL,
    kind               TEXT,
    attributes_json    TEXT NOT NULL DEFAULT '{}',
    events_json        TEXT NOT NULL DEFAULT '[]',
    started_at_ms      INTEGER,
    ended_at_ms        INTEGER,
    received_at_ms     INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS history_sidecars (
    sidecar_id      TEXT PRIMARY KEY,
    origin          TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    observed_at_ms  INTEGER NOT NULL,
    content_hash    BLOB NOT NULL CHECK(length(content_hash) = 32)
) STRICT;

ALTER TABLE raw_artifacts RENAME TO raw_artifacts_v7;
ALTER TABLE raw_session_memberships RENAME TO raw_session_memberships_v7;
ALTER TABLE raw_membership_census RENAME TO raw_membership_census_v7;
ALTER TABLE raw_sessions RENAME TO raw_sessions_v7;
ALTER TABLE raw_hook_events RENAME TO raw_hook_events_v7;
ALTER TABLE otlp_spans RENAME TO otlp_spans_v7;
ALTER TABLE history_sidecars RENAME TO history_sidecars_v7;

CREATE TABLE raw_sessions (
    raw_id                  TEXT PRIMARY KEY,
    origin                  TEXT NOT NULL CHECK (origin IN ('claude-code-session', 'codex-session', 'gemini-cli-session', 'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export', 'chatgpt-export', 'claude-ai-export', 'aistudio-drive', 'unknown-export')),
    native_id               TEXT,
    source_path             TEXT NOT NULL,
    source_index            INTEGER NOT NULL DEFAULT 0,
    blob_hash               BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size               INTEGER NOT NULL CHECK(blob_size >= 0),
    acquired_at_ms          INTEGER NOT NULL,
    file_mtime_ms           INTEGER,
    parsed_at_ms            INTEGER,
    parse_error             TEXT,
    validated_at_ms         INTEGER,
    validation_status       TEXT CHECK (validation_status IN ('passed', 'failed', 'skipped') OR validation_status IS NULL),
    validation_error        TEXT,
    validation_drift_count  INTEGER NOT NULL DEFAULT 0 CHECK(validation_drift_count >= 0),
    validation_mode         TEXT CHECK (validation_mode IN ('off', 'advisory', 'strict') OR validation_mode IS NULL),
    detection_warnings_json TEXT NOT NULL DEFAULT '[]',
    logical_source_key      TEXT,
    revision_kind           TEXT NOT NULL DEFAULT 'unknown' CHECK(revision_kind IN ('full', 'append', 'unknown')),
    source_revision         TEXT,
    predecessor_source_revision TEXT,
    predecessor_raw_id      TEXT,
    baseline_raw_id         TEXT,
    append_start_offset     INTEGER CHECK(append_start_offset >= 0),
    append_end_offset       INTEGER CHECK(append_end_offset > append_start_offset),
    acquisition_generation  INTEGER CHECK(acquisition_generation >= 0),
    revision_authority      TEXT NOT NULL DEFAULT 'quarantined' CHECK(revision_authority IN ('asserted', 'byte_proven', 'quarantined'))
) STRICT;

CREATE TABLE raw_session_memberships (
    raw_id                  TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key      TEXT NOT NULL,
    provider_session_id     TEXT NOT NULL,
    source_revision         TEXT NOT NULL,
    normalized_content_hash BLOB NOT NULL CHECK(length(normalized_content_hash) = 32),
    message_count           INTEGER NOT NULL CHECK(message_count >= 0),
    predecessor_raw_id      TEXT,
    acquisition_generation  INTEGER NOT NULL DEFAULT 0 CHECK(acquisition_generation >= 0),
    revision_authority      TEXT NOT NULL DEFAULT 'quarantined' CHECK(revision_authority IN ('byte_proven', 'quarantined')),
    decision                TEXT CHECK(decision IN ('applied', 'superseded_equivalent', 'superseded_prefix', 'ambiguous', 'deferred')),
    decided_at_ms           INTEGER CHECK(decided_at_ms >= 0),
    PRIMARY KEY(raw_id, logical_source_key),
    CHECK((decision IS NULL) = (decided_at_ms IS NULL))
) STRICT;

CREATE TABLE raw_membership_census (
    raw_id             TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    parser_fingerprint TEXT NOT NULL,
    status             TEXT NOT NULL CHECK(status IN ('complete', 'failed', 'non_session')),
    member_count       INTEGER NOT NULL CHECK(member_count >= 0),
    censused_at_ms     INTEGER NOT NULL CHECK(censused_at_ms >= 0),
    detail             TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE TABLE raw_artifacts (
    artifact_id              TEXT PRIMARY KEY,
    raw_id                   TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    origin                   TEXT NOT NULL CHECK (origin IN ('claude-code-session', 'codex-session', 'gemini-cli-session', 'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export', 'chatgpt-export', 'claude-ai-export', 'aistudio-drive', 'unknown-export')),
    source_path              TEXT NOT NULL,
    source_index             INTEGER NOT NULL DEFAULT 0,
    artifact_kind            TEXT NOT NULL,
    support_status           TEXT NOT NULL CHECK (support_status IN ('supported_parseable', 'recognized_unparsed', 'unsupported_parseable', 'decode_failed', 'partial_decode', 'unknown')),
    classification_reason    TEXT NOT NULL,
    parse_as_session         INTEGER NOT NULL DEFAULT 0 CHECK(parse_as_session IN (0, 1)),
    schema_eligible          INTEGER NOT NULL DEFAULT 0 CHECK(schema_eligible IN (0, 1)),
    malformed_jsonl_lines    INTEGER NOT NULL DEFAULT 0 CHECK(malformed_jsonl_lines >= 0),
    decode_error             TEXT,
    cohort_id                TEXT,
    link_group_key           TEXT,
    sidecar_agent_type       TEXT,
    first_observed_at_ms     INTEGER NOT NULL,
    last_observed_at_ms      INTEGER NOT NULL
) STRICT;

CREATE TABLE raw_hook_events (
    hook_event_id   TEXT PRIMARY KEY,
    origin          TEXT NOT NULL CHECK (origin IN ('claude-code-session', 'codex-session', 'gemini-cli-session', 'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export', 'chatgpt-export', 'claude-ai-export', 'aistudio-drive', 'unknown-export')),
    native_id       TEXT,
    session_native_id TEXT,
    source_path     TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    observed_at_ms  INTEGER NOT NULL
) STRICT;

CREATE TABLE otlp_spans (
    span_id           TEXT PRIMARY KEY,
    trace_id          TEXT NOT NULL,
    parent_span_id    TEXT,
    origin            TEXT CHECK ((origin IN ('claude-code-session', 'codex-session', 'gemini-cli-session', 'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export', 'chatgpt-export', 'claude-ai-export', 'aistudio-drive', 'unknown-export') OR origin IS NULL)),
    session_native_id TEXT,
    name              TEXT NOT NULL,
    kind              TEXT,
    attributes_json   TEXT NOT NULL DEFAULT '{}',
    events_json       TEXT NOT NULL DEFAULT '[]',
    started_at_ms     INTEGER,
    ended_at_ms       INTEGER,
    received_at_ms    INTEGER NOT NULL
) STRICT;

CREATE TABLE history_sidecars (
    sidecar_id      TEXT PRIMARY KEY,
    origin          TEXT NOT NULL CHECK (origin IN ('claude-code-session', 'codex-session', 'gemini-cli-session', 'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export', 'chatgpt-export', 'claude-ai-export', 'aistudio-drive', 'unknown-export')),
    source_path     TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    observed_at_ms  INTEGER NOT NULL,
    content_hash    BLOB NOT NULL CHECK(length(content_hash) = 32)
) STRICT;

INSERT INTO raw_sessions SELECT * FROM raw_sessions_v7;
INSERT INTO raw_session_memberships SELECT * FROM raw_session_memberships_v7;
INSERT INTO raw_membership_census SELECT * FROM raw_membership_census_v7;
INSERT INTO raw_artifacts SELECT * FROM raw_artifacts_v7;
INSERT INTO raw_hook_events SELECT * FROM raw_hook_events_v7;
INSERT INTO otlp_spans SELECT * FROM otlp_spans_v7;
INSERT INTO history_sidecars SELECT * FROM history_sidecars_v7;

DROP TABLE raw_artifacts_v7;
DROP TABLE raw_membership_census_v7;
DROP TABLE raw_session_memberships_v7;
DROP TABLE raw_sessions_v7;
DROP TABLE raw_hook_events_v7;
DROP TABLE otlp_spans_v7;
DROP TABLE history_sidecars_v7;

CREATE INDEX idx_raw_sessions_origin ON raw_sessions(origin);
CREATE INDEX idx_raw_sessions_origin_native ON raw_sessions(origin, native_id) WHERE native_id IS NOT NULL;
CREATE INDEX idx_raw_sessions_source_path ON raw_sessions(source_path, source_index);
CREATE INDEX idx_raw_sessions_parse_ready ON raw_sessions(raw_id) WHERE parsed_at_ms IS NULL AND validated_at_ms IS NOT NULL AND (validation_status IS NULL OR validation_status != 'failed');
CREATE INDEX idx_raw_sessions_logical_revision ON raw_sessions(logical_source_key, acquisition_generation, raw_id) WHERE logical_source_key IS NOT NULL;
CREATE INDEX idx_raw_session_memberships_logical ON raw_session_memberships(logical_source_key, acquisition_generation, raw_id);
CREATE INDEX idx_raw_session_memberships_pending ON raw_session_memberships(raw_id) WHERE decision IS NULL OR decision IN ('ambiguous', 'deferred');
CREATE UNIQUE INDEX idx_raw_artifacts_source_identity ON raw_artifacts(origin, source_path, source_index);
CREATE INDEX idx_raw_artifacts_raw_id ON raw_artifacts(raw_id);
CREATE INDEX idx_raw_hook_events_session ON raw_hook_events(origin, session_native_id, observed_at_ms);
CREATE INDEX idx_otlp_spans_trace ON otlp_spans(trace_id, started_at_ms DESC);
CREATE INDEX idx_otlp_spans_session ON otlp_spans(origin, session_native_id, started_at_ms DESC) WHERE session_native_id IS NOT NULL;
CREATE UNIQUE INDEX idx_history_sidecars_path_hash ON history_sidecars(origin, source_path, content_hash);
