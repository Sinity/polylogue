"""Archive parsed/search DDL fragment for archive."""

from __future__ import annotations

from polylogue.core.enums import BlockType, BranchType, LinkType, MessageType, Origin, PasteBoundary, Role
from polylogue.storage.sqlite.archive_tiers.common import CONTENT_HASH_CHECK, check, nullable_check

INDEX_SCHEMA_VERSION = 1

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
    origin_meta             TEXT NOT NULL DEFAULT '{{}}',
    git_branch              TEXT,
    git_repository_url      TEXT,
    commit_hash             TEXT,
    instructions_text       TEXT,
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
    metadata        TEXT NOT NULL DEFAULT '{{}}',
    tool_command    TEXT GENERATED ALWAYS AS (json_extract(tool_input, '$.command')) VIRTUAL,
    tool_path       TEXT GENERATED ALWAYS AS (
                        COALESCE(json_extract(tool_input, '$.file_path'), json_extract(tool_input, '$.path'))
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

CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts USING fts5(
    block_id UNINDEXED,
    message_id UNINDEXED,
    session_id UNINDEXED,
    block_type UNINDEXED,
    text,
    content='blocks',
    content_rowid='rowid',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS blocks_fts_ai
AFTER INSERT ON blocks WHEN new.text IS NOT NULL BEGIN
    INSERT INTO blocks_fts(rowid, block_id, message_id, session_id, block_type, text)
    VALUES (new.rowid, new.block_id, new.message_id, new.session_id, new.block_type, new.text);
END;

CREATE TRIGGER IF NOT EXISTS blocks_fts_ad
AFTER DELETE ON blocks WHEN old.text IS NOT NULL BEGIN
    INSERT INTO blocks_fts(blocks_fts, rowid, block_id, message_id, session_id, block_type, text)
    VALUES ('delete', old.rowid, old.block_id, old.message_id, old.session_id, old.block_type, old.text);
END;

CREATE TRIGGER IF NOT EXISTS blocks_fts_au
AFTER UPDATE ON blocks BEGIN
    INSERT INTO blocks_fts(blocks_fts, rowid, block_id, message_id, session_id, block_type, text)
    VALUES ('delete', old.rowid, old.block_id, old.message_id, old.session_id, old.block_type, old.text);
    INSERT INTO blocks_fts(rowid, block_id, message_id, session_id, block_type, text)
    SELECT new.rowid, new.block_id, new.message_id, new.session_id, new.block_type, new.text
    WHERE new.text IS NOT NULL;
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
    event_type        TEXT NOT NULL CHECK(event_type IN ('compaction', 'ghost_commit', 'agent_policy')),
    summary           TEXT,
    payload           TEXT NOT NULL DEFAULT '{{}}',
    occurred_at_ms    INTEGER,
    PRIMARY KEY(session_id, position)
) STRICT;

CREATE TABLE IF NOT EXISTS session_links (
    src_session_id        TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    dst_session_native_id TEXT NOT NULL,
    dst_session_id        TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
    link_type             TEXT NOT NULL CHECK ({check("link_type", LinkType)}),
    status                TEXT NOT NULL DEFAULT 'unresolved'
                              CHECK(status IN ('unresolved', 'resolved', 'repaired', 'quarantined')),
    method                TEXT,
    confidence            REAL CHECK(confidence IS NULL OR confidence BETWEEN 0 AND 1),
    evidence_json         TEXT NOT NULL DEFAULT '[]',
    observed_at_ms        INTEGER,
    PRIMARY KEY(src_session_id, dst_session_native_id, link_type)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_links_dst
ON session_links(dst_session_id)
WHERE dst_session_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS threads (
    thread_id        TEXT PRIMARY KEY,
    root_session_id  TEXT NOT NULL UNIQUE REFERENCES sessions(session_id) ON DELETE CASCADE,
    origin           TEXT NOT NULL CHECK ({check("origin", Origin)}),
    created_at_ms    INTEGER,
    updated_at_ms    INTEGER,
    session_count    INTEGER NOT NULL DEFAULT 0 CHECK(session_count >= 0)
) STRICT;

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
    repository_url    TEXT NOT NULL DEFAULT '',
    root_path         TEXT NOT NULL DEFAULT '',
    repo_name         TEXT,
    first_seen_at_ms  INTEGER NOT NULL,
    last_seen_at_ms   INTEGER NOT NULL,
    PRIMARY KEY(repository_url, root_path)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_repos_root_path
ON repos(root_path)
WHERE root_path != '';

CREATE TABLE IF NOT EXISTS session_repos (
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    repository_url  TEXT NOT NULL DEFAULT '',
    root_path       TEXT NOT NULL DEFAULT '',
    branch_name     TEXT,
    observed_at_ms  INTEGER,
    PRIMARY KEY(session_id, repository_url, root_path),
    FOREIGN KEY(repository_url, root_path) REFERENCES repos(repository_url, root_path) ON DELETE CASCADE
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_repos_repo
ON session_repos(repository_url, root_path, session_id);

CREATE TABLE IF NOT EXISTS session_commits (
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    repository_url  TEXT NOT NULL DEFAULT '',
    root_path       TEXT NOT NULL DEFAULT '',
    commit_hash     TEXT NOT NULL,
    detection_method TEXT NOT NULL DEFAULT 'parser-git-meta',
    confidence      REAL CHECK(confidence IS NULL OR confidence BETWEEN 0 AND 1),
    evidence_json   TEXT NOT NULL DEFAULT '{{}}',
    observed_at_ms  INTEGER,
    PRIMARY KEY(session_id, repository_url, root_path, commit_hash),
    FOREIGN KEY(repository_url, root_path) REFERENCES repos(repository_url, root_path) ON DELETE CASCADE
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_commits_hash
ON session_commits(commit_hash);

CREATE TABLE IF NOT EXISTS attachments (
    attachment_id          TEXT PRIMARY KEY,
    mime_type              TEXT,
    size_bytes             INTEGER CHECK(size_bytes IS NULL OR size_bytes >= 0),
    path                   TEXT,
    provider_meta          TEXT NOT NULL DEFAULT '{{}}',
    provider_attachment_id TEXT,
    provider_file_id       TEXT,
    provider_drive_id      TEXT,
    upload_origin          TEXT CHECK(upload_origin IN ('drive', 'paste', 'url', 'oauth') OR upload_origin IS NULL),
    ref_count              INTEGER NOT NULL DEFAULT 0 CHECK(ref_count >= 0)
) STRICT;

CREATE TABLE IF NOT EXISTS attachment_refs (
    ref_id                 TEXT PRIMARY KEY,
    attachment_id          TEXT NOT NULL REFERENCES attachments(attachment_id) ON DELETE CASCADE,
    session_id             TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    message_id             TEXT REFERENCES messages(message_id) ON DELETE SET NULL,
    provider_meta          TEXT NOT NULL DEFAULT '{{}}',
    provider_attachment_id TEXT,
    provider_file_id       TEXT,
    provider_drive_id      TEXT,
    upload_origin          TEXT CHECK(upload_origin IN ('drive', 'paste', 'url', 'oauth') OR upload_origin IS NULL)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_attachment_refs_session
ON attachment_refs(session_id);

CREATE INDEX IF NOT EXISTS idx_attachment_refs_message
ON attachment_refs(message_id)
WHERE message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_attachment_refs_upload_origin
ON attachment_refs(upload_origin, session_id)
WHERE upload_origin IS NOT NULL;

CREATE TABLE IF NOT EXISTS paste_spans (
    paste_span_id  TEXT GENERATED ALWAYS AS (message_id || ':' || start_offset || ':' || end_offset) STORED UNIQUE,
    message_id     TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    session_id     TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    start_offset   INTEGER NOT NULL CHECK(start_offset >= 0),
    end_offset     INTEGER NOT NULL CHECK(end_offset >= start_offset),
    content_hash   BLOB NOT NULL CHECK(length(content_hash) = 32),
    boundary       TEXT NOT NULL CHECK ({check("boundary", PasteBoundary)}),
    PRIMARY KEY(message_id, start_offset, end_offset)
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
    start_index        INTEGER NOT NULL CHECK(start_index >= 0),
    end_index          INTEGER NOT NULL CHECK(end_index >= start_index),
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

CREATE INDEX IF NOT EXISTS idx_session_work_events_type
ON session_work_events(work_event_type, session_id);

CREATE TABLE IF NOT EXISTS session_phases (
    phase_id        TEXT GENERATED ALWAYS AS (session_id || ':phase:' || position) STORED UNIQUE,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    position        INTEGER NOT NULL CHECK(position >= 0),
    phase_type      TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.0 CHECK(confidence BETWEEN 0 AND 1),
    start_index     INTEGER NOT NULL CHECK(start_index >= 0),
    end_index       INTEGER NOT NULL CHECK(end_index >= start_index),
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

CREATE INDEX IF NOT EXISTS idx_session_phases_type
ON session_phases(phase_type, session_id);

CREATE TABLE IF NOT EXISTS session_profiles (
    session_id               TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    workflow_shape           TEXT,
    workflow_shape_method    TEXT,
    workflow_shape_confidence REAL CHECK(workflow_shape_confidence IS NULL OR workflow_shape_confidence BETWEEN 0 AND 1),
    terminal_state           TEXT,
    terminal_state_method    TEXT,
    terminal_state_confidence REAL CHECK(terminal_state_confidence IS NULL OR terminal_state_confidence BETWEEN 0 AND 1),
    duration_ms              INTEGER CHECK(duration_ms IS NULL OR duration_ms >= 0),
    substantive_count        INTEGER NOT NULL DEFAULT 0 CHECK(substantive_count >= 0),
    attachment_count         INTEGER NOT NULL DEFAULT 0 CHECK(attachment_count >= 0),
    work_event_count         INTEGER NOT NULL DEFAULT 0 CHECK(work_event_count >= 0),
    phase_count              INTEGER NOT NULL DEFAULT 0 CHECK(phase_count >= 0),
    tool_calls_per_minute    REAL,
    cost_credits             REAL,
    cost_usd                 REAL,
    cost_is_estimated        INTEGER NOT NULL DEFAULT 0 CHECK(cost_is_estimated IN (0, 1)),
    cost_provenance          TEXT CHECK(cost_provenance IN ('exact', 'priced', 'estimated') OR cost_provenance IS NULL),
    priced_with              TEXT,
    priced_at_ms             INTEGER,
    search_text              TEXT NOT NULL DEFAULT '',
    provenance_json          TEXT NOT NULL DEFAULT '{{}}'
) STRICT;
"""

__all__ = ["INDEX_DDL", "INDEX_SCHEMA_VERSION"]
