-- Widen the durable ``origin`` CHECK on the five source-tier tables that
-- carry it (raw_sessions, raw_artifacts, raw_hook_events, otlp_spans,
-- history_sidecars) to the current twelve-value core.enums.Origin
-- vocabulary.
--
-- Migration 009 baked an eleven-value literal list into these tables' CHECK
-- constraints the last time it rebuilt them. core.enums.Origin later gained
-- a twelfth token, 'claude-design-session' (bd polylogue-tbun -- Claude
-- Design is a distinct product with its own wire format, not claude.ai with
-- a flag; see sources/parsers/claude/ai_parser.py's design-chat parser), and
-- the CHECK constraint SQLite bakes into a table's schema at CREATE time is
-- never re-derived at runtime. archive_tiers/source.py's canonical DDL for
-- all five tables already generates this CHECK from the enum via
-- check()/nullable_check() -- so a *fresh* bootstrap has always accepted the
-- full vocabulary -- but any already-migrated source.db still carries the
-- stale eleven-value literal and raises sqlite3.IntegrityError the instant a
-- claude-design-session raw payload is acquired (bd polylogue-052vs).
--
-- SQLite cannot ALTER a CHECK constraint, so each table is rebuilt: create a
-- differently-named table with the widened CHECK, copy every row across by
-- explicit column name (never positionally -- migration 009's own comment
-- on raw_sessions documents why: ad hoc ALTER TABLE ADD COLUMN steps append
-- physically, so a column's position in an existing database does not
-- necessarily match its position in the current canonical DDL text), drop
-- the original, then rename the new table into the original's name.
--
-- Because this never renames the *original* table away (only a freshly
-- created, never-referenced temporary table is renamed into the final
-- slot), SQLite's automatic foreign-key-reference rewrite on
-- ``ALTER TABLE ... RENAME TO`` -- which rewrites *other* tables'
-- ``REFERENCES`` clauses to track a table that gets renamed away from its
-- original name -- never fires here. The seven other durable tables that
-- hold a ``REFERENCES raw_sessions(raw_id)`` foreign key
-- (raw_capture_observations, raw_session_memberships, raw_membership_census,
-- raw_live_source_reconciliation_receipts, raw_membership_writeback_receipts,
-- raw_append_chain_backfill_receipts, raw_authority_parser_census) never
-- need to be touched: their FK text still says ``REFERENCES raw_sessions``,
-- and a table with that exact name exists again the instant the final
-- rename statement below completes. Migration-runner connections
-- (migration_runner.py, cli/commands/maintenance/_migrate_tier.py,
-- archive_tiers/bootstrap.py) never set ``PRAGMA foreign_keys = ON`` before
-- applying a migration, so the intervening ``DROP TABLE`` steps below also
-- never trigger SQLite's drop-time cascading-delete behavior against rows in
-- those other tables.
--
-- raw_artifacts/raw_hook_events/otlp_spans/history_sidecars are each guarded
-- with a preceding ``CREATE TABLE IF NOT EXISTS`` in their stale (pre-021)
-- shape, mirroring migration 009's own precedent (its header comment: "Older
-- valid source tiers can therefore lack them. Create empty v7-shaped tables
-- so every supported predecessor ... follows the same copy-forward path
-- below") -- some already-migrated archives, and test fixtures that jump
-- straight to a mid-chain ``PRAGMA user_version`` without replaying every
-- earlier migration, may never have created one of these four adjunct
-- tables at all.

CREATE TABLE IF NOT EXISTS raw_artifacts (
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

CREATE TABLE IF NOT EXISTS raw_hook_events (
    hook_event_id   TEXT PRIMARY KEY,
    origin          TEXT NOT NULL CHECK (origin IN ('claude-code-session', 'codex-session', 'gemini-cli-session', 'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export', 'chatgpt-export', 'claude-ai-export', 'aistudio-drive', 'unknown-export')),
    native_id       TEXT,
    session_native_id TEXT,
    source_path     TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    observed_at_ms  INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS otlp_spans (
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

CREATE TABLE IF NOT EXISTS history_sidecars (
    sidecar_id      TEXT PRIMARY KEY,
    origin          TEXT NOT NULL CHECK (origin IN ('claude-code-session', 'codex-session', 'gemini-cli-session', 'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export', 'chatgpt-export', 'claude-ai-export', 'aistudio-drive', 'unknown-export')),
    source_path     TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    observed_at_ms  INTEGER NOT NULL,
    content_hash    BLOB NOT NULL CHECK(length(content_hash) = 32)
) STRICT;

CREATE TABLE raw_sessions__021ff (
    raw_id                  TEXT PRIMARY KEY,
    origin                  TEXT NOT NULL CHECK (origin IN ('claude-code-session', 'codex-session', 'gemini-cli-session', 'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export', 'chatgpt-export', 'claude-ai-export', 'claude-design-session', 'aistudio-drive', 'unknown-export')),
    capture_mode            TEXT CHECK ((capture_mode IN ('chatgpt', 'claude-ai', 'claude-design', 'claude-code', 'codex', 'gemini', 'gemini-cli', 'hermes', 'antigravity', 'beads', 'grok', 'drive', 'unknown') OR capture_mode IS NULL)),
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
    validation_status       TEXT CHECK ((validation_status IN ('passed', 'failed', 'skipped') OR validation_status IS NULL)),
    validation_error        TEXT,
    validation_drift_count  INTEGER NOT NULL DEFAULT 0 CHECK(validation_drift_count >= 0),
    validation_mode         TEXT CHECK ((validation_mode IN ('off', 'advisory', 'strict') OR validation_mode IS NULL)),
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
    revision_authority      TEXT NOT NULL DEFAULT 'quarantined' CHECK(revision_authority IN ('asserted', 'byte_proven', 'quarantined')),
    revision_authority_evidence TEXT
        CHECK(revision_authority_evidence IS NULL OR revision_authority_evidence IN ('live_source_verification_v1'))
) STRICT;

INSERT INTO raw_sessions__021ff (
    raw_id, origin, capture_mode, native_id, source_path, source_index, blob_hash, blob_size,
    acquired_at_ms, file_mtime_ms, parsed_at_ms, parse_error, validated_at_ms,
    validation_status, validation_error, validation_drift_count, validation_mode,
    detection_warnings_json, logical_source_key, revision_kind, source_revision,
    predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
    append_start_offset, append_end_offset, acquisition_generation,
    revision_authority, revision_authority_evidence
)
SELECT
    raw_id, origin, capture_mode, native_id, source_path, source_index, blob_hash, blob_size,
    acquired_at_ms, file_mtime_ms, parsed_at_ms, parse_error, validated_at_ms,
    validation_status, validation_error, validation_drift_count, validation_mode,
    detection_warnings_json, logical_source_key, revision_kind, source_revision,
    predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
    append_start_offset, append_end_offset, acquisition_generation,
    revision_authority, revision_authority_evidence
FROM raw_sessions;

DROP TABLE raw_sessions;
ALTER TABLE raw_sessions__021ff RENAME TO raw_sessions;

CREATE INDEX idx_raw_sessions_origin ON raw_sessions(origin);
CREATE INDEX idx_raw_sessions_origin_native ON raw_sessions(origin, native_id) WHERE native_id IS NOT NULL;
CREATE INDEX idx_raw_sessions_source_path ON raw_sessions(source_path, source_index);
CREATE INDEX idx_raw_sessions_parse_ready ON raw_sessions(raw_id) WHERE parsed_at_ms IS NULL AND validated_at_ms IS NOT NULL AND (validation_status IS NULL OR validation_status != 'failed');
CREATE INDEX idx_raw_sessions_logical_revision ON raw_sessions(logical_source_key, acquisition_generation, raw_id) WHERE logical_source_key IS NOT NULL;
CREATE INDEX idx_raw_sessions_blob_hash ON raw_sessions(blob_hash);
CREATE INDEX idx_raw_sessions_blob_hash_raw_id ON raw_sessions(blob_hash, raw_id);


CREATE TABLE raw_artifacts__021ff (
    artifact_id              TEXT PRIMARY KEY,
    raw_id                   TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    origin                   TEXT NOT NULL CHECK (origin IN ('claude-code-session', 'codex-session', 'gemini-cli-session', 'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export', 'chatgpt-export', 'claude-ai-export', 'claude-design-session', 'aistudio-drive', 'unknown-export')),
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

INSERT INTO raw_artifacts__021ff (
    artifact_id, raw_id, origin, source_path, source_index, artifact_kind, support_status,
    classification_reason, parse_as_session, schema_eligible, malformed_jsonl_lines,
    decode_error, cohort_id, link_group_key, sidecar_agent_type,
    first_observed_at_ms, last_observed_at_ms
)
SELECT
    artifact_id, raw_id, origin, source_path, source_index, artifact_kind, support_status,
    classification_reason, parse_as_session, schema_eligible, malformed_jsonl_lines,
    decode_error, cohort_id, link_group_key, sidecar_agent_type,
    first_observed_at_ms, last_observed_at_ms
FROM raw_artifacts;

DROP TABLE raw_artifacts;
ALTER TABLE raw_artifacts__021ff RENAME TO raw_artifacts;

CREATE UNIQUE INDEX idx_raw_artifacts_source_identity ON raw_artifacts(origin, source_path, source_index);
CREATE INDEX idx_raw_artifacts_raw_id ON raw_artifacts(raw_id);


CREATE TABLE raw_hook_events__021ff (
    hook_event_id   TEXT PRIMARY KEY,
    origin          TEXT NOT NULL CHECK (origin IN ('claude-code-session', 'codex-session', 'gemini-cli-session', 'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export', 'chatgpt-export', 'claude-ai-export', 'claude-design-session', 'aistudio-drive', 'unknown-export')),
    native_id       TEXT,
    session_native_id TEXT,
    source_path     TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    observed_at_ms  INTEGER NOT NULL
) STRICT;

INSERT INTO raw_hook_events__021ff (
    hook_event_id, origin, native_id, session_native_id, source_path, event_type,
    payload_json, observed_at_ms
)
SELECT
    hook_event_id, origin, native_id, session_native_id, source_path, event_type,
    payload_json, observed_at_ms
FROM raw_hook_events;

DROP TABLE raw_hook_events;
ALTER TABLE raw_hook_events__021ff RENAME TO raw_hook_events;

CREATE INDEX idx_raw_hook_events_session ON raw_hook_events(origin, session_native_id, observed_at_ms);


CREATE TABLE otlp_spans__021ff (
    span_id           TEXT PRIMARY KEY,
    trace_id          TEXT NOT NULL,
    parent_span_id    TEXT,
    origin            TEXT CHECK ((origin IN ('claude-code-session', 'codex-session', 'gemini-cli-session', 'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export', 'chatgpt-export', 'claude-ai-export', 'claude-design-session', 'aistudio-drive', 'unknown-export') OR origin IS NULL)),
    session_native_id TEXT,
    name              TEXT NOT NULL,
    kind              TEXT,
    attributes_json   TEXT NOT NULL DEFAULT '{}',
    events_json       TEXT NOT NULL DEFAULT '[]',
    started_at_ms     INTEGER,
    ended_at_ms       INTEGER,
    received_at_ms    INTEGER NOT NULL
) STRICT;

INSERT INTO otlp_spans__021ff (
    span_id, trace_id, parent_span_id, origin, session_native_id, name, kind,
    attributes_json, events_json, started_at_ms, ended_at_ms, received_at_ms
)
SELECT
    span_id, trace_id, parent_span_id, origin, session_native_id, name, kind,
    attributes_json, events_json, started_at_ms, ended_at_ms, received_at_ms
FROM otlp_spans;

DROP TABLE otlp_spans;
ALTER TABLE otlp_spans__021ff RENAME TO otlp_spans;

CREATE INDEX idx_otlp_spans_trace ON otlp_spans(trace_id, started_at_ms DESC);
CREATE INDEX idx_otlp_spans_session ON otlp_spans(origin, session_native_id, started_at_ms DESC) WHERE session_native_id IS NOT NULL;


CREATE TABLE history_sidecars__021ff (
    sidecar_id      TEXT PRIMARY KEY,
    origin          TEXT NOT NULL CHECK (origin IN ('claude-code-session', 'codex-session', 'gemini-cli-session', 'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export', 'chatgpt-export', 'claude-ai-export', 'claude-design-session', 'aistudio-drive', 'unknown-export')),
    source_path     TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    observed_at_ms  INTEGER NOT NULL,
    content_hash    BLOB NOT NULL CHECK(length(content_hash) = 32)
) STRICT;

INSERT INTO history_sidecars__021ff (
    sidecar_id, origin, source_path, payload_json, observed_at_ms, content_hash
)
SELECT
    sidecar_id, origin, source_path, payload_json, observed_at_ms, content_hash
FROM history_sidecars;

DROP TABLE history_sidecars;
ALTER TABLE history_sidecars__021ff RENAME TO history_sidecars;

CREATE UNIQUE INDEX idx_history_sidecars_path_hash ON history_sidecars(origin, source_path, content_hash);
