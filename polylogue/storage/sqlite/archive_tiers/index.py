"""Archive parsed/search DDL fragment for archive."""

from __future__ import annotations

from polylogue.core.enums import BlockType, BranchType, LinkType, MessageType, Origin, PasteBoundary, Role
from polylogue.storage.sqlite.archive_tiers.common import CONTENT_HASH_CHECK, check, nullable_check

INDEX_SCHEMA_VERSION = 2

INDEX_DDL = f"""
CREATE TABLE IF NOT EXISTS sessions (
    session_id              TEXT GENERATED ALWAYS AS (origin || ':' || native_id) STORED UNIQUE,
    native_id               TEXT NOT NULL,
    origin                  TEXT NOT NULL CHECK ({check("origin", Origin)}),
    parent_session_id       TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
    root_session_id         TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
    raw_id                  TEXT,
    branch_type             TEXT CHECK ({nullable_check("branch_type", BranchType)}),
    active_leaf_message_id  TEXT,
    title                   TEXT,
    title_source            TEXT CHECK(title_source IN ('origin', 'path', 'heuristic', 'user', 'unknown') OR title_source IS NULL),
    git_branch              TEXT,
    git_repository_url      TEXT,
    commit_hash             TEXT,
    instructions_text       TEXT,
    reported_duration_ms    INTEGER CHECK(reported_duration_ms IS NULL OR reported_duration_ms >= 0),
    message_count           INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0),
    word_count              INTEGER NOT NULL DEFAULT 0 CHECK(word_count >= 0),
    tool_use_count          INTEGER NOT NULL DEFAULT 0 CHECK(tool_use_count >= 0),
    thinking_count          INTEGER NOT NULL DEFAULT 0 CHECK(thinking_count >= 0),
    paste_count             INTEGER NOT NULL DEFAULT 0 CHECK(paste_count >= 0),
    user_message_count      INTEGER NOT NULL DEFAULT 0 CHECK(user_message_count >= 0),
    assistant_message_count INTEGER NOT NULL DEFAULT 0 CHECK(assistant_message_count >= 0),
    system_message_count    INTEGER NOT NULL DEFAULT 0 CHECK(system_message_count >= 0),
    tool_message_count      INTEGER NOT NULL DEFAULT 0 CHECK(tool_message_count >= 0),
    user_word_count         INTEGER NOT NULL DEFAULT 0 CHECK(user_word_count >= 0),
    assistant_word_count    INTEGER NOT NULL DEFAULT 0 CHECK(assistant_word_count >= 0),
    content_hash            BLOB NOT NULL {CONTENT_HASH_CHECK},
    created_at_ms           INTEGER,
    updated_at_ms           INTEGER,
    sort_key_ms             INTEGER GENERATED ALWAYS AS (COALESCE(updated_at_ms, created_at_ms)) STORED,
    PRIMARY KEY(origin, native_id)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_sessions_origin_sort
ON sessions(origin, sort_key_ms DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_parent
ON sessions(parent_session_id)
WHERE parent_session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_root
ON sessions(root_session_id)
WHERE root_session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_raw_id
ON sessions(raw_id)
WHERE raw_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS messages (
    message_id          TEXT GENERATED ALWAYS AS (
                            session_id || ':' || COALESCE(native_id, position || '.' || variant_index)
                        ) STORED UNIQUE,
    session_id          TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    native_id           TEXT,
    parent_message_id   TEXT REFERENCES messages(message_id) ON DELETE SET NULL,
    position            INTEGER NOT NULL CHECK(position >= 0),
    role                TEXT NOT NULL CHECK ({check("role", Role)}),
    message_type        TEXT NOT NULL DEFAULT 'message' CHECK ({check("message_type", MessageType)}),
    model_name          TEXT,
    model_effort        TEXT,
    has_tool_use        INTEGER NOT NULL DEFAULT 0 CHECK(has_tool_use IN (0, 1)),
    has_thinking        INTEGER NOT NULL DEFAULT 0 CHECK(has_thinking IN (0, 1)),
    has_paste           INTEGER NOT NULL DEFAULT 0 CHECK(has_paste IN (0, 1)),
    paste_boundary      TEXT CHECK ({nullable_check("paste_boundary", PasteBoundary)}),
    variant_index       INTEGER NOT NULL DEFAULT 0 CHECK(variant_index >= 0),
    is_active_path      INTEGER NOT NULL DEFAULT 1 CHECK(is_active_path IN (0, 1)),
    is_active_leaf      INTEGER NOT NULL DEFAULT 0 CHECK(is_active_leaf IN (0, 1)),
    word_count          INTEGER NOT NULL DEFAULT 0 CHECK(word_count >= 0),
    input_tokens        INTEGER NOT NULL DEFAULT 0 CHECK(input_tokens >= 0),
    output_tokens       INTEGER NOT NULL DEFAULT 0 CHECK(output_tokens >= 0),
    cache_read_tokens   INTEGER NOT NULL DEFAULT 0 CHECK(cache_read_tokens >= 0),
    cache_write_tokens  INTEGER NOT NULL DEFAULT 0 CHECK(cache_write_tokens >= 0),
    duration_ms         INTEGER CHECK(duration_ms IS NULL OR duration_ms >= 0),
    content_hash        BLOB NOT NULL {CONTENT_HASH_CHECK},
    occurred_at_ms      INTEGER,
    PRIMARY KEY(session_id, position, variant_index)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_messages_session_position
ON messages(session_id, position, variant_index);

CREATE INDEX IF NOT EXISTS idx_messages_parent
ON messages(parent_message_id)
WHERE parent_message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_messages_session_role
ON messages(session_id, role);

CREATE INDEX IF NOT EXISTS idx_messages_active_path
ON messages(session_id, is_active_path, position)
WHERE is_active_path = 1;

CREATE TABLE IF NOT EXISTS blocks (
    block_id        TEXT GENERATED ALWAYS AS (message_id || ':' || position) STORED UNIQUE,
    message_id      TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    position        INTEGER NOT NULL CHECK(position >= 0),
    block_type      TEXT NOT NULL CHECK ({check("block_type", BlockType)}),
    text            TEXT,
    tool_name       TEXT,
    tool_id         TEXT,
    tool_input      TEXT,
    semantic_type   TEXT,
    media_type      TEXT,
    language        TEXT,
    tool_command    TEXT GENERATED ALWAYS AS (json_extract(tool_input, '$.command')) VIRTUAL,
    tool_path       TEXT GENERATED ALWAYS AS (
                        COALESCE(json_extract(tool_input, '$.file_path'), json_extract(tool_input, '$.path'))
                    ) VIRTUAL,
    search_text     TEXT GENERATED ALWAYS AS (
                        trim(COALESCE(text, '') || ' ' ||
                             COALESCE(tool_name, '') || ' ' ||
                             COALESCE(json_extract(tool_input, '$.command'), '') || ' ' ||
                             COALESCE(json_extract(tool_input, '$.file_path'), '') || ' ' ||
                             COALESCE(json_extract(tool_input, '$.path'), ''))
                    ) VIRTUAL,
    PRIMARY KEY(message_id, position)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_blocks_session_position
ON blocks(session_id, message_id, position);

CREATE INDEX IF NOT EXISTS idx_blocks_type
ON blocks(block_type);

CREATE INDEX IF NOT EXISTS idx_blocks_tool_id
ON blocks(tool_id)
WHERE tool_id IS NOT NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    block_id UNINDEXED,
    message_id UNINDEXED,
    session_id UNINDEXED,
    block_type UNINDEXED,
    text,
    content='',
    contentless_delete=1,
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_ai
AFTER INSERT ON blocks WHEN new.search_text != '' BEGIN
    INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
    VALUES (new.rowid, new.block_id, new.message_id, new.session_id, new.block_type, new.search_text);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_ad
AFTER DELETE ON blocks WHEN old.search_text != '' BEGIN
    DELETE FROM messages_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_au
AFTER UPDATE ON blocks BEGIN
    DELETE FROM messages_fts WHERE rowid = old.rowid;
    INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
    SELECT new.rowid, new.block_id, new.message_id, new.session_id, new.block_type, new.search_text
    WHERE new.search_text != '';
END;

CREATE VIEW IF NOT EXISTS actions AS
SELECT
    u.session_id,
    u.message_id,
    u.block_id AS tool_use_block_id,
    u.tool_name,
    u.semantic_type,
    u.tool_command,
    u.tool_path,
    u.tool_input,
    r.text AS output_text,
    r.block_id AS tool_result_block_id
FROM blocks u
LEFT JOIN blocks r
    ON r.tool_id = u.tool_id
   AND r.block_type = 'tool_result'
WHERE u.block_type = 'tool_use';

CREATE TABLE IF NOT EXISTS session_events (
    event_id          TEXT GENERATED ALWAYS AS (session_id || ':' || position) STORED UNIQUE,
    session_id        TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    source_message_id TEXT REFERENCES messages(message_id) ON DELETE SET NULL,
    position          INTEGER NOT NULL CHECK(position >= 0),
    event_type        TEXT NOT NULL CHECK(event_type IN ('compaction')),
    summary           TEXT NOT NULL,
    occurred_at_ms    INTEGER,
    PRIMARY KEY(session_id, position)
) STRICT;

CREATE TABLE IF NOT EXISTS session_agent_policies (
    policy_id         TEXT GENERATED ALWAYS AS (session_id || ':' || position) STORED UNIQUE,
    session_id        TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    source_message_id TEXT REFERENCES messages(message_id) ON DELETE SET NULL,
    position          INTEGER NOT NULL CHECK(position >= 0),
    approval_policy   TEXT,
    sandbox_policy    TEXT,
    network_policy    TEXT,
    observed_at_ms    INTEGER,
    PRIMARY KEY(session_id, position)
) STRICT;

CREATE TABLE IF NOT EXISTS session_links (
    src_session_id          TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    dst_origin              TEXT NOT NULL CHECK ({check("dst_origin", Origin)}),
    dst_native_id           TEXT NOT NULL,
    link_type               TEXT NOT NULL CHECK ({check("link_type", LinkType)}),
    resolved_dst_session_id TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
    status                  TEXT CHECK(status IN ('repaired', 'quarantined') OR status IS NULL),
    method                  TEXT,
    confidence              REAL NOT NULL DEFAULT 1.0 CHECK(confidence BETWEEN 0 AND 1),
    evidence_json           TEXT NOT NULL DEFAULT '[]',
    observed_at_ms          INTEGER NOT NULL,
    resolved_at_ms          INTEGER,
    PRIMARY KEY(src_session_id, dst_origin, dst_native_id, link_type)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_links_dst
ON session_links(resolved_dst_session_id)
WHERE resolved_dst_session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_session_links_dst_native
ON session_links(dst_origin, dst_native_id);

CREATE TABLE IF NOT EXISTS threads (
    thread_id                    TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    dominant_repo_id             TEXT REFERENCES repos(repo_id) ON DELETE SET NULL,
    materializer_version         INTEGER NOT NULL DEFAULT 5,
    materialized_at              TEXT NOT NULL DEFAULT '',
    source_updated_at            TEXT,
    input_high_water_mark        TEXT,
    input_high_water_mark_source TEXT,
    input_row_count              INTEGER NOT NULL DEFAULT 0 CHECK(input_row_count >= 0),
    start_time                   TEXT,
    end_time                     TEXT,
    dominant_repo                TEXT,
    session_ids_json             TEXT NOT NULL DEFAULT '[]',
    session_count                INTEGER NOT NULL DEFAULT 0 CHECK(session_count >= 0),
    depth                        INTEGER NOT NULL DEFAULT 0 CHECK(depth >= 0),
    branch_count                 INTEGER NOT NULL DEFAULT 0 CHECK(branch_count >= 0),
    total_messages               INTEGER NOT NULL DEFAULT 0 CHECK(total_messages >= 0),
    total_cost_usd               REAL NOT NULL DEFAULT 0.0,
    wall_duration_ms             INTEGER NOT NULL DEFAULT 0 CHECK(wall_duration_ms >= 0),
    work_event_breakdown_json    TEXT NOT NULL DEFAULT '{{}}',
    payload_json                 TEXT NOT NULL DEFAULT '{{}}',
    search_text                  TEXT NOT NULL DEFAULT '',
    created_at_ms                INTEGER NOT NULL DEFAULT 0
) STRICT;

CREATE INDEX IF NOT EXISTS idx_threads_time
ON threads(end_time DESC, start_time DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS threads_fts USING fts5(
    thread_id UNINDEXED,
    root_id UNINDEXED,
    text,
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS threads_fts_ai
AFTER INSERT ON threads BEGIN
    INSERT INTO threads_fts (thread_id, root_id, text)
    VALUES (new.thread_id, new.thread_id, new.search_text);
END;

CREATE TRIGGER IF NOT EXISTS threads_fts_ad
AFTER DELETE ON threads BEGIN
    DELETE FROM threads_fts WHERE thread_id = old.thread_id;
END;

CREATE TRIGGER IF NOT EXISTS threads_fts_au
AFTER UPDATE ON threads BEGIN
    DELETE FROM threads_fts WHERE thread_id = old.thread_id;
    INSERT INTO threads_fts (thread_id, root_id, text)
    VALUES (new.thread_id, new.thread_id, new.search_text);
END;

CREATE TABLE IF NOT EXISTS thread_sessions (
    thread_id    TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    position     INTEGER NOT NULL CHECK(position >= 0),
    PRIMARY KEY(thread_id, session_id)
) STRICT;

CREATE TABLE IF NOT EXISTS session_working_dirs (
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    position    INTEGER NOT NULL CHECK(position >= 0),
    PRIMARY KEY(session_id, path)
) STRICT;

CREATE TABLE IF NOT EXISTS repos (
    origin_url        TEXT NOT NULL DEFAULT '',
    root_path         TEXT NOT NULL,
    repo_id           TEXT GENERATED ALWAYS AS (origin_url || char(31) || root_path) STORED UNIQUE,
    repo_name         TEXT NOT NULL DEFAULT '',
    first_seen_at_ms  INTEGER NOT NULL,
    last_seen_at_ms   INTEGER NOT NULL,
    PRIMARY KEY(origin_url, root_path)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_repos_root_path
ON repos(root_path)
WHERE root_path != '';

CREATE TABLE IF NOT EXISTS session_repos (
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    repo_id         TEXT NOT NULL REFERENCES repos(repo_id) ON DELETE CASCADE,
    branch_name     TEXT NOT NULL DEFAULT '',
    observed_at_ms  INTEGER NOT NULL,
    PRIMARY KEY(session_id, repo_id)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_repos_repo
ON session_repos(repo_id);

CREATE TABLE IF NOT EXISTS session_commits (
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    commit_sha      TEXT NOT NULL,
    repo_id         TEXT REFERENCES repos(repo_id) ON DELETE CASCADE,
    detection_type  TEXT NOT NULL CHECK(detection_type IN ('time_window', 'file_overlap', 'explicit_ref', 'origin_reported')),
    method          TEXT,
    confidence      REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    evidence_json   TEXT NOT NULL DEFAULT '{{}}',
    created_at_ms   INTEGER NOT NULL,
    PRIMARY KEY(session_id, commit_sha)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_commits_hash
ON session_commits(commit_sha);

CREATE INDEX IF NOT EXISTS idx_session_commits_repo
ON session_commits(repo_id)
WHERE repo_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS attachments (
    attachment_id          TEXT PRIMARY KEY,
    display_name           TEXT,
    media_type             TEXT,
    byte_count             INTEGER NOT NULL DEFAULT 0 CHECK(byte_count >= 0),
    blob_hash              BLOB NOT NULL CHECK(length(blob_hash) = 32),
    ref_count              INTEGER NOT NULL DEFAULT 0 CHECK(ref_count >= 0)
) STRICT;

CREATE TABLE IF NOT EXISTS attachment_refs (
    ref_id                 TEXT GENERATED ALWAYS AS (message_id || ':attachment:' || position) STORED UNIQUE,
    attachment_id          TEXT NOT NULL REFERENCES attachments(attachment_id) ON DELETE CASCADE,
    session_id             TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    message_id             TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    position               INTEGER NOT NULL CHECK(position >= 0),
    upload_origin          TEXT CHECK(upload_origin IN ('drive', 'paste', 'url', 'oauth') OR upload_origin IS NULL),
    source_url             TEXT,
    caption                TEXT,
    PRIMARY KEY(message_id, position)
) STRICT;

CREATE TABLE IF NOT EXISTS attachment_native_ids (
    ref_id     TEXT NOT NULL REFERENCES attachment_refs(ref_id) ON DELETE CASCADE,
    id_kind    TEXT NOT NULL CHECK(id_kind IN ('attachment', 'file', 'drive', 'source', 'url')),
    native_id  TEXT NOT NULL,
    PRIMARY KEY(ref_id, id_kind, native_id)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_attachment_refs_session
ON attachment_refs(session_id);

CREATE INDEX IF NOT EXISTS idx_attachment_refs_message
ON attachment_refs(message_id)
WHERE message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_attachment_refs_upload_origin
ON attachment_refs(upload_origin, session_id)
WHERE upload_origin IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_attachment_native_ids_native
ON attachment_native_ids(id_kind, native_id);

CREATE TABLE IF NOT EXISTS paste_spans (
    paste_id        TEXT GENERATED ALWAYS AS (message_id || ':' || position) STORED UNIQUE,
    message_id      TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    position        INTEGER NOT NULL CHECK(position >= 0),
    start_offset    INTEGER CHECK(start_offset IS NULL OR start_offset >= 0),
    end_offset      INTEGER CHECK(end_offset IS NULL OR end_offset >= start_offset),
    boundary_state  TEXT NOT NULL CHECK ({check("boundary_state", PasteBoundary)}),
    source_event_id TEXT,
    source_marker   TEXT,
    content_hash    BLOB NOT NULL CHECK(length(content_hash) = 32),
    observed_at_ms  INTEGER,
    PRIMARY KEY(message_id, position)
) STRICT;

CREATE TABLE IF NOT EXISTS price_catalogs (
    catalog_id       TEXT PRIMARY KEY,
    catalog_hash     TEXT NOT NULL,
    source_name      TEXT NOT NULL,
    effective_at_ms  INTEGER,
    loaded_at_ms     INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS model_prices (
    catalog_id                   TEXT NOT NULL REFERENCES price_catalogs(catalog_id) ON DELETE CASCADE,
    model_name                   TEXT NOT NULL,
    price_unit                   TEXT NOT NULL CHECK(price_unit IN ('tokens', 'credits', 'flat')),
    input_cost_per_million       REAL,
    output_cost_per_million      REAL,
    cache_read_cost_per_million  REAL,
    cache_write_cost_per_million REAL,
    credit_cost_per_unit         REAL,
    effective_from_ms            INTEGER NOT NULL DEFAULT 0,
    effective_to_ms              INTEGER,
    PRIMARY KEY(catalog_id, model_name, price_unit, effective_from_ms)
) STRICT;

CREATE TABLE IF NOT EXISTS session_reported_costs (
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    cost_kind       TEXT NOT NULL CHECK(cost_kind IN ('usd', 'credits')),
    amount          REAL NOT NULL,
    source          TEXT NOT NULL CHECK(source IN ('origin_reported', 'priced', 'estimated')),
    observed_at_ms  INTEGER,
    PRIMARY KEY(session_id, cost_kind, source)
) STRICT;

CREATE TABLE IF NOT EXISTS session_model_usage (
    session_id              TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    model_name              TEXT NOT NULL,
    input_tokens            INTEGER NOT NULL DEFAULT 0 CHECK(input_tokens >= 0),
    output_tokens           INTEGER NOT NULL DEFAULT 0 CHECK(output_tokens >= 0),
    cache_read_tokens       INTEGER NOT NULL DEFAULT 0 CHECK(cache_read_tokens >= 0),
    cache_write_tokens      INTEGER NOT NULL DEFAULT 0 CHECK(cache_write_tokens >= 0),
    message_count           INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0),
    priced_with             TEXT REFERENCES price_catalogs(catalog_id) ON DELETE SET NULL,
    priced_at_ms            INTEGER,
    cost_usd                REAL,
    cost_credits            REAL,
    cost_provenance         TEXT CHECK(cost_provenance IN ('origin_reported', 'priced', 'estimated') OR cost_provenance IS NULL),
    PRIMARY KEY(session_id, model_name)
) STRICT;

CREATE TABLE IF NOT EXISTS session_tags (
    session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    tag           TEXT NOT NULL,
    tag_source    TEXT NOT NULL CHECK(tag_source IN ('user', 'auto')),
    method        TEXT,
    confidence    REAL CHECK(confidence IS NULL OR confidence BETWEEN 0 AND 1),
    evidence_json TEXT,
    PRIMARY KEY(session_id, tag, tag_source)
) STRICT;

CREATE TABLE IF NOT EXISTS insight_materialization (
    insight_type                 TEXT NOT NULL CHECK(insight_type IN (
                                    'session_profile', 'work_events', 'phases', 'latency', 'thread')),
    session_id                   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    materializer_version         INTEGER NOT NULL,
    materialized_at_ms           INTEGER NOT NULL,
    source_updated_at_ms         INTEGER,
    source_sort_key_ms           INTEGER,
    input_high_water_mark_ms     INTEGER,
    input_row_count              INTEGER NOT NULL DEFAULT 0 CHECK(input_row_count >= 0),
    PRIMARY KEY(insight_type, session_id)
) STRICT;

CREATE TABLE IF NOT EXISTS session_work_events (
    event_id           TEXT GENERATED ALWAYS AS (session_id || ':work_event:' || position) STORED UNIQUE,
    session_id         TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    position           INTEGER NOT NULL CHECK(position >= 0),
    work_event_type    TEXT NOT NULL,
    summary            TEXT NOT NULL,
    confidence         REAL NOT NULL DEFAULT 0.0 CHECK(confidence BETWEEN 0 AND 1),
    start_index        INTEGER NOT NULL DEFAULT 0 CHECK(start_index >= 0),
    end_index          INTEGER NOT NULL DEFAULT 0 CHECK(end_index >= start_index),
    started_at_ms      INTEGER,
    ended_at_ms        INTEGER,
    duration_ms        INTEGER NOT NULL DEFAULT 0 CHECK(duration_ms >= 0),
    file_paths_json    TEXT NOT NULL DEFAULT '[]',
    tools_used_json    TEXT NOT NULL DEFAULT '[]',
    evidence_json      TEXT NOT NULL DEFAULT '{{}}',
    inference_json     TEXT NOT NULL DEFAULT '{{}}',
    search_text        TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(session_id, position)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_work_events_session
ON session_work_events(session_id, position);

CREATE INDEX IF NOT EXISTS idx_session_work_events_type
ON session_work_events(work_event_type, session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS session_work_events_fts USING fts5(
    event_id UNINDEXED,
    session_id UNINDEXED,
    work_event_type UNINDEXED,
    text,
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS session_work_events_fts_ai
AFTER INSERT ON session_work_events BEGIN
    INSERT INTO session_work_events_fts (event_id, session_id, work_event_type, text)
    VALUES (new.event_id, new.session_id, new.work_event_type, new.search_text);
END;

CREATE TABLE IF NOT EXISTS session_phases (
    phase_id        TEXT GENERATED ALWAYS AS (session_id || ':phase:' || position) STORED UNIQUE,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    position        INTEGER NOT NULL CHECK(position >= 0),
    phase_type      TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.0 CHECK(confidence BETWEEN 0 AND 1),
    start_index     INTEGER NOT NULL DEFAULT 0 CHECK(start_index >= 0),
    end_index       INTEGER NOT NULL DEFAULT 0 CHECK(end_index >= start_index),
    started_at_ms   INTEGER,
    ended_at_ms     INTEGER,
    duration_ms     INTEGER NOT NULL DEFAULT 0 CHECK(duration_ms >= 0),
    tool_counts_json TEXT NOT NULL DEFAULT '{{}}',
    word_count      INTEGER NOT NULL DEFAULT 0 CHECK(word_count >= 0),
    evidence_json   TEXT NOT NULL DEFAULT '{{}}',
    inference_json  TEXT NOT NULL DEFAULT '{{}}',
    search_text     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(session_id, position)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_phases_session
ON session_phases(session_id, position);

CREATE INDEX IF NOT EXISTS idx_session_phases_type
ON session_phases(phase_type, session_id);

CREATE TRIGGER IF NOT EXISTS session_work_events_fts_ad
AFTER DELETE ON session_work_events BEGIN
    DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
END;

CREATE TRIGGER IF NOT EXISTS session_work_events_fts_au
AFTER UPDATE ON session_work_events BEGIN
    DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
    INSERT INTO session_work_events_fts (event_id, session_id, work_event_type, text)
    VALUES (new.event_id, new.session_id, new.work_event_type, new.search_text);
END;

CREATE TABLE IF NOT EXISTS session_latency_profiles (
    session_id                       TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    materializer_version             INTEGER NOT NULL DEFAULT 5,
    materialized_at                  TEXT NOT NULL,
    source_updated_at                TEXT,
    source_sort_key                  REAL,
    input_high_water_mark            TEXT,
    input_high_water_mark_source     TEXT,
    input_row_count                  INTEGER NOT NULL DEFAULT 0 CHECK(input_row_count >= 0),
    source_name                      TEXT NOT NULL,
    title                            TEXT,
    first_message_at                 TEXT,
    last_message_at                  TEXT,
    canonical_session_date           TEXT,
    median_tool_call_ms              INTEGER NOT NULL DEFAULT 0 CHECK(median_tool_call_ms >= 0),
    p90_tool_call_ms                 INTEGER NOT NULL DEFAULT 0 CHECK(p90_tool_call_ms >= 0),
    max_tool_call_ms                 INTEGER NOT NULL DEFAULT 0 CHECK(max_tool_call_ms >= 0),
    stuck_tool_count                 INTEGER NOT NULL DEFAULT 0 CHECK(stuck_tool_count >= 0),
    median_agent_response_ms         INTEGER NOT NULL DEFAULT 0 CHECK(median_agent_response_ms >= 0),
    median_user_response_ms          INTEGER NOT NULL DEFAULT 0 CHECK(median_user_response_ms >= 0),
    tool_call_count_by_category_json TEXT NOT NULL DEFAULT '{{}}',
    evidence_payload_json            TEXT NOT NULL DEFAULT '{{}}',
    search_text                      TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_latency_profiles_provider
ON session_latency_profiles(source_name);

CREATE INDEX IF NOT EXISTS idx_session_latency_profiles_date
ON session_latency_profiles(canonical_session_date DESC);

CREATE INDEX IF NOT EXISTS idx_session_latency_profiles_stuck
ON session_latency_profiles(stuck_tool_count DESC, canonical_session_date DESC);

CREATE TABLE IF NOT EXISTS session_profiles (
    session_id                      TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    logical_session_id              TEXT,
    materializer_version            INTEGER NOT NULL DEFAULT 5,
    materialized_at                 TEXT NOT NULL DEFAULT '',
    source_updated_at               TEXT,
    source_sort_key                 REAL,
    input_high_water_mark           TEXT,
    input_high_water_mark_source    TEXT,
    input_row_count                 INTEGER NOT NULL DEFAULT 0 CHECK(input_row_count >= 0),
    source_name                     TEXT NOT NULL DEFAULT '',
    title                           TEXT,
    first_message_at                TEXT,
    last_message_at                 TEXT,
    canonical_session_date          TEXT,
    repo_paths_json                 TEXT,
    repo_names_json                 TEXT,
    tags_json                       TEXT,
    auto_tags_json                  TEXT,
    message_count                   INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0),
    substantive_count               INTEGER NOT NULL DEFAULT 0 CHECK(substantive_count >= 0),
    attachment_count                INTEGER NOT NULL DEFAULT 0 CHECK(attachment_count >= 0),
    work_event_count                INTEGER NOT NULL DEFAULT 0 CHECK(work_event_count >= 0),
    phase_count                     INTEGER NOT NULL DEFAULT 0 CHECK(phase_count >= 0),
    word_count                      INTEGER NOT NULL DEFAULT 0 CHECK(word_count >= 0),
    tool_use_count                  INTEGER NOT NULL DEFAULT 0 CHECK(tool_use_count >= 0),
    thinking_count                  INTEGER NOT NULL DEFAULT 0 CHECK(thinking_count >= 0),
    total_cost_usd                  REAL NOT NULL DEFAULT 0,
    total_duration_ms               INTEGER NOT NULL DEFAULT 0 CHECK(total_duration_ms >= 0),
    engaged_duration_ms             INTEGER NOT NULL DEFAULT 0 CHECK(engaged_duration_ms >= 0),
    tool_active_duration_ms         INTEGER NOT NULL DEFAULT 0 CHECK(tool_active_duration_ms >= 0),
    wall_duration_ms                INTEGER NOT NULL DEFAULT 0 CHECK(wall_duration_ms >= 0),
    workflow_shape                  TEXT,
    workflow_shape_method           TEXT,
    workflow_shape_confidence       REAL CHECK(workflow_shape_confidence BETWEEN 0 AND 1 OR workflow_shape_confidence IS NULL),
    workflow_shape_features_json    TEXT NOT NULL DEFAULT '{{}}',
    terminal_state                  TEXT,
    terminal_state_method           TEXT,
    terminal_state_confidence       REAL CHECK(terminal_state_confidence BETWEEN 0 AND 1 OR terminal_state_confidence IS NULL),
    terminal_state_evidence_json    TEXT NOT NULL DEFAULT '{{}}',
    cost_is_estimated               INTEGER NOT NULL DEFAULT 0 CHECK(cost_is_estimated IN (0, 1)),
    thinking_duration_ms            INTEGER NOT NULL DEFAULT 0 CHECK(thinking_duration_ms >= 0),
    output_duration_ms              INTEGER NOT NULL DEFAULT 0 CHECK(output_duration_ms >= 0),
    tool_duration_ms                INTEGER NOT NULL DEFAULT 0 CHECK(tool_duration_ms >= 0),
    latency_percentiles_ms_json     TEXT NOT NULL DEFAULT '{{}}',
    tool_calls_per_minute           REAL,
    timing_provenance               TEXT NOT NULL DEFAULT 'sort_key_estimated',
    total_input_tokens              INTEGER NOT NULL DEFAULT 0 CHECK(total_input_tokens >= 0),
    total_output_tokens             INTEGER NOT NULL DEFAULT 0 CHECK(total_output_tokens >= 0),
    total_cache_read_tokens         INTEGER NOT NULL DEFAULT 0 CHECK(total_cache_read_tokens >= 0),
    total_cache_write_tokens        INTEGER NOT NULL DEFAULT 0 CHECK(total_cache_write_tokens >= 0),
    total_credit_cost               REAL NOT NULL DEFAULT 0.0,
    cost_provenance                 TEXT NOT NULL DEFAULT 'unknown',
    per_model_cost_json             TEXT NOT NULL DEFAULT '{{}}',
    evidence_payload_json           TEXT NOT NULL DEFAULT '{{}}',
    inference_payload_json          TEXT NOT NULL DEFAULT '{{}}',
    enrichment_payload_json         TEXT NOT NULL DEFAULT '{{}}',
    evidence_search_text            TEXT NOT NULL DEFAULT '',
    inference_search_text           TEXT NOT NULL DEFAULT '',
    enrichment_search_text          TEXT NOT NULL DEFAULT '',
    enrichment_version              INTEGER NOT NULL DEFAULT 1,
    enrichment_family               TEXT NOT NULL DEFAULT 'scored_session_enrichment',
    inference_version               INTEGER NOT NULL DEFAULT 1,
    inference_family                TEXT NOT NULL DEFAULT 'heuristic_session_semantics',
    search_text                     TEXT NOT NULL DEFAULT '',
    duration_ms                     INTEGER CHECK(duration_ms IS NULL OR duration_ms >= 0),
    cost_credits                    REAL,
    cost_usd                        REAL,
    priced_with                     TEXT,
    priced_at_ms                    INTEGER
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_profiles_provider
ON session_profiles(source_name);

CREATE INDEX IF NOT EXISTS idx_session_profiles_logical_session
ON session_profiles(logical_session_id);

CREATE INDEX IF NOT EXISTS idx_session_profiles_sort
ON session_profiles(source_sort_key DESC);

CREATE INDEX IF NOT EXISTS idx_session_profiles_first_message
ON session_profiles(first_message_at DESC);

CREATE INDEX IF NOT EXISTS idx_session_profiles_canonical_date
ON session_profiles(canonical_session_date DESC);

CREATE TABLE IF NOT EXISTS session_tag_rollups (
    tag                          TEXT NOT NULL,
    bucket_day                   TEXT NOT NULL,
    source_name                  TEXT NOT NULL,
    materializer_version         INTEGER NOT NULL DEFAULT 5,
    materialized_at              TEXT NOT NULL,
    source_updated_at            TEXT,
    source_sort_key              REAL,
    input_high_water_mark        TEXT,
    input_high_water_mark_source TEXT,
    input_row_count              INTEGER NOT NULL DEFAULT 0 CHECK(input_row_count >= 0),
    session_count                INTEGER NOT NULL DEFAULT 0 CHECK(session_count >= 0),
    logical_session_count        INTEGER NOT NULL DEFAULT 0 CHECK(logical_session_count >= 0),
    logical_session_ids_json     TEXT NOT NULL DEFAULT '[]',
    explicit_count               INTEGER NOT NULL DEFAULT 0 CHECK(explicit_count >= 0),
    auto_count                   INTEGER NOT NULL DEFAULT 0 CHECK(auto_count >= 0),
    repo_breakdown_json          TEXT NOT NULL DEFAULT '{{}}',
    search_text                  TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(tag, bucket_day, source_name)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_tag_rollups_day
ON session_tag_rollups(bucket_day DESC, source_name, tag);

CREATE INDEX IF NOT EXISTS idx_session_tag_rollups_provider
ON session_tag_rollups(source_name, tag);
"""

__all__ = ["INDEX_DDL", "INDEX_SCHEMA_VERSION"]
