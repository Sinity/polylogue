"""Archive parsed/search DDL fragment for archive."""

from __future__ import annotations

from polylogue.core.enums import (
    BlockType,
    BranchType,
    LinkType,
    MaterialOrigin,
    MessageType,
    Origin,
    PasteBoundary,
    Role,
    SessionKind,
    WebConstructType,
)
from polylogue.storage.fts.sql import FTS_TRIGGER_DDL
from polylogue.storage.sqlite.action_relation import action_relation_select_sql
from polylogue.storage.sqlite.archive_tiers.common import (
    CONTENT_HASH_CHECK,
    check,
    json_object_check,
    nullable_check,
)

INDEX_SCHEMA_VERSION = 37

FTS_FRESHNESS_STATE_DDL = """
CREATE TABLE IF NOT EXISTS fts_freshness_state (
    surface TEXT PRIMARY KEY,
    state TEXT NOT NULL CHECK (state IN ('ready', 'stale', 'unknown')),
    checked_at TEXT NOT NULL,
    source_rows INTEGER NOT NULL DEFAULT 0,
    indexed_rows INTEGER NOT NULL DEFAULT 0,
    missing_rows INTEGER NOT NULL DEFAULT 0,
    excess_rows INTEGER NOT NULL DEFAULT 0,
    duplicate_rows INTEGER NOT NULL DEFAULT 0,
    detail TEXT
) STRICT;
"""

INDEX_DDL = f"""
CREATE TABLE IF NOT EXISTS raw_revision_applications (
    decision_id              TEXT PRIMARY KEY,
    raw_id                   TEXT NOT NULL,
    session_id               TEXT NOT NULL,
    logical_source_key       TEXT NOT NULL,
    source_revision          TEXT NOT NULL,
    acquisition_generation  INTEGER NOT NULL CHECK(acquisition_generation >= 0),
    decision                 TEXT NOT NULL CHECK(decision IN (
                                 'selected_baseline', 'applied_append', 'superseded',
                                 'ambiguous', 'deferred'
                             )),
    accepted_raw_id          TEXT,
    accepted_source_revision TEXT,
    accepted_content_hash    BLOB CHECK(
                                 accepted_content_hash IS NULL OR length(accepted_content_hash) = 32
                             ),
    baseline_raw_id          TEXT,
    predecessor_raw_id       TEXT,
    append_end_offset        INTEGER CHECK(append_end_offset IS NULL OR append_end_offset >= 0),
    detail                   TEXT NOT NULL,
    decided_at_ms            INTEGER NOT NULL CHECK(decided_at_ms >= 0),
    CHECK(
        (accepted_raw_id IS NULL AND accepted_source_revision IS NULL AND accepted_content_hash IS NULL)
        OR
        (accepted_raw_id IS NOT NULL AND accepted_source_revision IS NOT NULL AND accepted_content_hash IS NOT NULL)
    )
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_revision_applications_identity
ON raw_revision_applications(
    raw_id, session_id, decision, source_revision,
    COALESCE(accepted_source_revision, '')
);

CREATE INDEX IF NOT EXISTS idx_raw_revision_applications_logical
ON raw_revision_applications(logical_source_key, acquisition_generation, raw_id);

CREATE TABLE IF NOT EXISTS raw_revision_heads (
    logical_source_key       TEXT PRIMARY KEY,
    session_id               TEXT NOT NULL,
    accepted_raw_id          TEXT NOT NULL,
    accepted_source_revision TEXT NOT NULL,
    accepted_content_hash    BLOB NOT NULL CHECK(length(accepted_content_hash) = 32),
    accepted_frontier_kind   TEXT NOT NULL CHECK(accepted_frontier_kind IN ('byte', 'semantic')),
    accepted_frontier        INTEGER NOT NULL CHECK(accepted_frontier >= 0),
    acquisition_generation  INTEGER NOT NULL CHECK(acquisition_generation >= 0),
    append_end_offset        INTEGER CHECK(append_end_offset IS NULL OR append_end_offset >= 0),
    decided_at_ms            INTEGER NOT NULL CHECK(decided_at_ms >= 0)
) STRICT;

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
    session_kind            TEXT NOT NULL DEFAULT 'standard' CHECK ({check("session_kind", SessionKind)}),
    title_source            TEXT CHECK(title_source IN ('origin', 'path', 'heuristic', 'user', 'unknown') OR title_source IS NULL),
    git_branch              TEXT,
    git_repository_url      TEXT,
    provider_project_ref    TEXT,
    commit_hash             TEXT,
    instructions_text       TEXT,
    reported_duration_ms    INTEGER CHECK(reported_duration_ms IS NULL OR reported_duration_ms >= 0),
    message_count           INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0),
    word_count              INTEGER NOT NULL DEFAULT 0 CHECK(word_count >= 0),
    tool_use_count          INTEGER NOT NULL DEFAULT 0 CHECK(tool_use_count >= 0),
    thinking_count          INTEGER NOT NULL DEFAULT 0 CHECK(thinking_count >= 0),
    paste_count             INTEGER NOT NULL DEFAULT 0 CHECK(paste_count >= 0),
    user_message_count      INTEGER NOT NULL DEFAULT 0 CHECK(user_message_count >= 0),
    authored_user_message_count INTEGER NOT NULL DEFAULT 0 CHECK(authored_user_message_count >= 0),
    assistant_message_count INTEGER NOT NULL DEFAULT 0 CHECK(assistant_message_count >= 0),
    system_message_count    INTEGER NOT NULL DEFAULT 0 CHECK(system_message_count >= 0),
    tool_message_count      INTEGER NOT NULL DEFAULT 0 CHECK(tool_message_count >= 0),
    user_word_count         INTEGER NOT NULL DEFAULT 0 CHECK(user_word_count >= 0),
    authored_user_word_count INTEGER NOT NULL DEFAULT 0 CHECK(authored_user_word_count >= 0),
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
    material_origin     TEXT NOT NULL DEFAULT 'unknown' CHECK ({check("material_origin", MaterialOrigin)}),
    model_name          TEXT,
    model_effort        TEXT,
    sender_name         TEXT,
    recipient           TEXT,
    delivery_status     TEXT,
    end_turn            INTEGER CHECK(end_turn IN (0, 1) OR end_turn IS NULL),
    user_context_text   TEXT,
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

-- Serves the sort-key ordering used by iter_messages keyset pagination,
-- get_messages_batch, and get_messages_paginated. Those reads order by
-- `(occurred_at_ms IS NULL), occurred_at_ms, message_id` (NULL-timestamp rows
-- sort last). The leading IS NULL expression must itself be an indexed column or
-- the planner ignores the index and sorts the whole session in a temp B-tree on
-- every chunk — verified via EXPLAIN QUERY PLAN: a plain
-- (session_id, occurred_at_ms, message_id) index still triggers
-- `USE TEMP B-TREE FOR ORDER BY`, while the expression index below plans as a
-- covering-index scan with no sort (#2467 / #2475 perf audit).
CREATE INDEX IF NOT EXISTS idx_messages_session_sortkey
ON messages(session_id, (occurred_at_ms IS NULL), occurred_at_ms, message_id);

CREATE INDEX IF NOT EXISTS idx_messages_parent
ON messages(parent_message_id)
WHERE parent_message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_messages_session_role
ON messages(session_id, role);

CREATE INDEX IF NOT EXISTS idx_messages_role
ON messages(role);

CREATE INDEX IF NOT EXISTS idx_messages_session_material_origin
ON messages(session_id, material_origin);

CREATE INDEX IF NOT EXISTS idx_messages_message_type
ON messages(message_type);

CREATE INDEX IF NOT EXISTS idx_messages_material_origin
ON messages(material_origin);

-- Serves the paid embedding selector and status/preflight counts. The predicate
-- matches authored prose only: user/assistant message rows with human- or
-- assistant-authored material and positive word count. Keeping this as a
-- partial index avoids scanning tool, protocol, context-pack, and generated
-- runtime rows when computing cost windows over large archives.
CREATE INDEX IF NOT EXISTS idx_messages_embedding_prose
ON messages(session_id, position, variant_index, message_id, content_hash)
WHERE message_type = 'message'
  AND role IN ('user', 'assistant')
  AND material_origin IN ('human_authored', 'assistant_authored')
  AND word_count > 0;

CREATE INDEX IF NOT EXISTS idx_messages_active_path
ON messages(session_id, is_active_path, position)
WHERE is_active_path = 1;

CREATE INDEX IF NOT EXISTS idx_messages_active_leaf
ON messages(session_id, is_active_leaf)
WHERE is_active_leaf = 1;

CREATE TABLE IF NOT EXISTS blocks (
    block_id        TEXT GENERATED ALWAYS AS (message_id || ':' || position) STORED UNIQUE,
    message_id      TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    position        INTEGER NOT NULL CHECK(position >= 0),
    block_type      TEXT NOT NULL CHECK ({check("block_type", BlockType)}),
    text            TEXT,
    tool_name       TEXT,
    tool_id         TEXT,
    tool_input      TEXT CHECK ({json_object_check("tool_input", nullable=True)}),
    semantic_type   TEXT,
    media_type      TEXT,
    language        TEXT,
    tool_result_is_error  INTEGER CHECK (tool_result_is_error IN (0, 1)),
    tool_result_exit_code INTEGER,
    -- svfj: the citation anchor atom. Hashes canonical block EVIDENCE only
    -- (type, text, tool_name, canonical tool_input, semantic/media/language,
    -- is_error, exit_code) -- deliberately EXCLUDING session_id/message_id/
    -- position/tool_id so the hash survives fork-position shift, re-ingest,
    -- and provider tool-id regeneration. Nullable (not every low-level test
    -- fixture / raw SQL writer populates it) -- the real write path always
    -- sets it; the resolver treats a NULL row as its own state, never a crash.
    content_hash    BLOB CHECK(content_hash IS NULL OR length(content_hash) = 32),
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
    -- ohbx: lowercased command+path text for substring lookups (e.g. "did
    -- this bash/exec_command block invoke `polylogue`?") that need LIKE
    -- '%x%' semantics, not FTS token matching. Backs blocks_command_trigram
    -- below -- a plain LIKE scan here recomputes tool_command/tool_path's
    -- json_extract for every generic-tool row in the archive (measured
    -- ~70s over 915K rows on a 26GB archive).
    tool_detail_text TEXT GENERATED ALWAYS AS (
                        lower(COALESCE(tool_command, '') || ' ' || COALESCE(tool_path, ''))
                    ) VIRTUAL,
    PRIMARY KEY(message_id, position)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_blocks_session_position
ON blocks(session_id, message_id, position);

CREATE INDEX IF NOT EXISTS idx_blocks_content_hash
ON blocks(content_hash);

CREATE INDEX IF NOT EXISTS idx_blocks_type
ON blocks(block_type);

-- Serves structured failure/outcome reports and action predicates that anchor
-- on provider-reported tool-result failure rather than prose. The predicate
-- remains on the tool_result block because the public actions relation is a
-- view over paired tool_use/tool_result blocks; starting from failed result
-- rows avoids scanning every tool invocation in large archives.
CREATE INDEX IF NOT EXISTS idx_blocks_tool_result_outcome
ON blocks(block_type, tool_result_is_error, tool_result_exit_code, session_id, tool_id, message_id)
WHERE block_type = 'tool_result';

CREATE INDEX IF NOT EXISTS idx_blocks_type_tool
ON blocks(
    block_type,
    COALESCE(NULLIF(LOWER(tool_name), ''), 'unknown')
);

CREATE INDEX IF NOT EXISTS idx_blocks_tool_id
ON blocks(tool_id)
WHERE tool_id IS NOT NULL;

-- Serves message FTS readiness and repair. Search readiness compares the
-- text-bearing block set with messages_fts_docsize; without this partial index
-- the source-side count scans every block in large archives before the user can
-- run even a simple MATCH query.
CREATE INDEX IF NOT EXISTS idx_blocks_search_text_populated
ON blocks(message_id, position)
WHERE search_text != '';

CREATE TABLE IF NOT EXISTS web_content_constructs (
    construct_id    TEXT GENERATED ALWAYS AS (block_id || ':' || position) STORED UNIQUE,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    message_id      TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    block_id        TEXT NOT NULL REFERENCES blocks(block_id) ON DELETE CASCADE,
    position        INTEGER NOT NULL CHECK(position >= 0),
    provider        TEXT NOT NULL,
    construct_type  TEXT NOT NULL CHECK ({check("construct_type", WebConstructType)}),
    provider_key    TEXT,
    title           TEXT,
    url             TEXT,
    text            TEXT,
    source_id       TEXT,
    group_id        TEXT,
    group_title     TEXT,
    query           TEXT,
    asset_pointer   TEXT,
    mime_type       TEXT,
    status          TEXT,
    task_id         TEXT,
    task_type       TEXT,
    rank            INTEGER,
    start_index     INTEGER,
    end_index       INTEGER,
    PRIMARY KEY(block_id, position)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_web_constructs_session_type
ON web_content_constructs(session_id, construct_type);

-- polylogue-rgbj: the only messages(message_id) FK child table that lacked a
-- leading index on its foreign key column. Every other referencing table
-- (blocks, attachment_refs, paste_spans, session_events,
-- session_agent_policies, session_provider_usage_events) already has one
-- via its PRIMARY KEY or a dedicated index. Without this index, SQLite's FK
-- `ON DELETE CASCADE` enforcement falls back to a full table scan of
-- web_content_constructs for EVERY deleted message row (SQLite does not
-- require a child-key index for FK enforcement, it just scans if one is
-- missing). `_replace_full_session_messages_and_blocks`'s bare
-- `DELETE FROM messages WHERE session_id = ?` therefore cost
-- O(deleted_messages x web_content_constructs_rows): confirmed live at
-- 132,796 web_content_constructs rows, `EXPLAIN QUERY PLAN` showed
-- `SCAN web_content_constructs` before this index, `SEARCH ... USING INDEX`
-- after. See tests/unit/storage/test_schema_safety.py and
-- tests/benchmarks/test_full_session_replace.py for the query-plan and
-- timing proof.
CREATE INDEX IF NOT EXISTS idx_web_constructs_message
ON web_content_constructs(message_id);

CREATE INDEX IF NOT EXISTS idx_web_constructs_url
ON web_content_constructs(url)
WHERE url IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_web_constructs_query
ON web_content_constructs(query)
WHERE query IS NOT NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    block_id UNINDEXED,
    message_id UNINDEXED,
    session_id UNINDEXED,
    block_type UNINDEXED,
    text,
    content='',
    contentless_delete=1,
    tokenize='unicode61 remove_diacritics 2'
);

-- FTS triggers for messages_fts table are now dynamically composed from sql.py
-- (polylogue-a7xr.5: consolidate FTS trigger DDL to single source)

-- ohbx: trigram-tokenized (not unicode61) so `detail LIKE '%pattern%'` gets
-- SQLite's built-in trigram LIKE-acceleration for arbitrary substrings, not
-- just whole tokens -- unicode61/messages_fts would miss a match like
-- "notpolyloguefile.txt" (no token boundary around the substring), and
-- searches a much larger candidate set (the word can appear in ordinary
-- prose, not just tool invocations). External content (not contentless,
-- unlike messages_fts) is required: contentless trigram tables silently
-- return zero rows for LIKE queries because SQLite needs to re-read the
-- real text from the content table to verify a candidate trigram match
-- (verified locally against the SQLite forum's own explanation of this
-- exact mechanism before relying on it).
--
-- Query-shape note for callers: SQLite's planner does NOT automatically
-- drive a `blocks_command_trigram JOIN blocks` from the trigram table's
-- LIKE index -- a plain join lets it choose to scan `blocks` as the outer
-- loop and probe the trigram table per row, which is *slower* than the old
-- raw scan (measured: 26s vs 0.15s at 300K rows). The trigram index must
-- drive the query explicitly via `blocks.rowid IN (SELECT rowid FROM
-- blocks_command_trigram WHERE tool_detail_text LIKE ...)`, which forces
-- the trigram LIKE-optimization to run first and reduces the outer table
-- to an indexed rowid lookup (measured: 900x+ faster than the raw scan at
-- 915K rows, using this exact shape).
CREATE VIRTUAL TABLE IF NOT EXISTS blocks_command_trigram USING fts5(
    tool_detail_text,
    tokenize='trigram',
    content='blocks',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS blocks_command_trigram_ai
AFTER INSERT ON blocks WHEN new.block_type = 'tool_use' AND new.tool_detail_text != ' ' BEGIN
    INSERT INTO blocks_command_trigram(rowid, tool_detail_text)
    VALUES (new.rowid, new.tool_detail_text);
END;

-- External-content FTS5 tables require the special 'delete' command form
-- with the OLD column value supplied (not a plain DELETE by rowid) --
-- verified locally: a bare `DELETE FROM blocks_command_trigram WHERE rowid
-- = old.rowid` leaves stale trigram postings that later raise "fts5:
-- missing row N from content table" once the real row is gone from
-- `blocks`, because FTS5 needs the old text to locate the exact postings
-- to remove rather than re-reading it from the (already-deleted) content
-- row.
CREATE TRIGGER IF NOT EXISTS blocks_command_trigram_ad
AFTER DELETE ON blocks WHEN old.block_type = 'tool_use' AND old.tool_detail_text != ' ' BEGIN
    INSERT INTO blocks_command_trigram(blocks_command_trigram, rowid, tool_detail_text)
    VALUES ('delete', old.rowid, old.tool_detail_text);
END;

CREATE TRIGGER IF NOT EXISTS blocks_command_trigram_au
AFTER UPDATE ON blocks BEGIN
    INSERT INTO blocks_command_trigram(blocks_command_trigram, rowid, tool_detail_text)
    SELECT 'delete', old.rowid, old.tool_detail_text
    WHERE old.block_type = 'tool_use' AND old.tool_detail_text != ' ';
    INSERT INTO blocks_command_trigram(rowid, tool_detail_text)
    SELECT new.rowid, new.tool_detail_text
    WHERE new.block_type = 'tool_use' AND new.tool_detail_text != ' ';
END;

{FTS_FRESHNESS_STATE_DDL}

-- xnkf: a plain equality join on tool_id fans out when a provider re-emits
-- the same tool_id on distinct messages (verified live: identical toolu_
-- ids as 2 tool_use + 2 tool_result blocks at different positions, NOT
-- variants). Rank each side by transcript order (message position, THEN
-- variant_index, then block position -- messages are only unique on
-- (position, variant_index), so omitting variant_index would leave ties
-- between regenerated variant messages, letting SQLite assign ranks
-- independently/arbitrarily across the two CTEs and re-introduce
-- cross-pairing) within (session_id, tool_id), and pair same-rank rows --
-- the Nth use in the transcript gets the Nth result, never a cross
-- product. Uses with no tool_id (NULL or '') are never rank-paired -- SQL
-- equality never matches NULL, so a NULL tool_id use was already always
-- unpaired under the old plain-equality join; the second branch below
-- preserves that (still surfaced, just with NULL result columns) while the
-- empty-string guard stops '' specifically from cross-joining as if it
-- were a real shared id (parsers currently only ever emit NULL, not '').
CREATE VIEW IF NOT EXISTS actions AS
{action_relation_select_sql()};

CREATE TABLE IF NOT EXISTS session_events (
    event_id                   TEXT GENERATED ALWAYS AS (session_id || ':' || position) STORED UNIQUE,
    session_id                 TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    source_message_id          TEXT REFERENCES messages(message_id) ON DELETE SET NULL,
    source_message_provider_id TEXT,
    position                   INTEGER NOT NULL CHECK(position >= 0),
    event_type                 TEXT NOT NULL CHECK(length(trim(event_type)) > 0),
    summary                    TEXT NOT NULL,
    payload_json               TEXT NOT NULL DEFAULT '{{}}' CHECK ({json_object_check("payload_json")}),
    occurred_at_ms             INTEGER,
    PRIMARY KEY(session_id, position)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_events_source_message
ON session_events(source_message_id)
WHERE source_message_id IS NOT NULL;

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

CREATE INDEX IF NOT EXISTS idx_session_agent_policies_source_message
ON session_agent_policies(source_message_id)
WHERE source_message_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS session_links (
    src_session_id          TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    dst_origin              TEXT NOT NULL CHECK ({check("dst_origin", Origin)}),
    dst_native_id           TEXT NOT NULL,
    link_type               TEXT NOT NULL CHECK ({check("link_type", LinkType)}),
    resolved_dst_session_id TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
    -- Lineage normalization (#2467): for a prefix-sharing child (fork / resume /
    -- spawned subagent / auto-compaction copy) the child stores only its own
    -- divergent tail; `branch_point_message_id` is the last parent message the
    -- child inherited, and `inheritance` records whether the child shares the
    -- parent's leading prefix ('prefix-sharing') or is a fresh spawn that merely
    -- references the parent ('spawned-fresh'). NULL until the parent is resolved.
    -- Deliberately NOT a FK: message_id is deterministic, so a parent full-replace
    -- re-ingest re-creates the same id. An `ON DELETE SET NULL` FK would instead
    -- null this during the parent's DELETE step and permanently break the child's
    -- composition (the cascade fires before the re-INSERT) — see #2467 audit.
    branch_point_message_id TEXT,
    inheritance             TEXT CHECK(inheritance IN ('prefix-sharing', 'spawned-fresh') OR inheritance IS NULL),
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
    tokenize='unicode61 remove_diacritics 2'
);

-- FTS triggers for threads_fts table are now dynamically composed from sql.py
-- (polylogue-a7xr.5: consolidate FTS trigger DDL to single source)

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
    -- #2468: real SHA-256 of the stored bytes when acquired, else NULL. Previously
    -- a synthetic hash of attachment metadata was written here, falsely implying a
    -- blob existed (0 blobs were ever stored). `acquisition_status` records whether
    -- the bytes were fetched ('acquired'), are known unrecoverable from the source
    -- polylogue holds ('unavailable'), or have not yet been fetched ('unfetched').
    blob_hash              BLOB CHECK(blob_hash IS NULL OR length(blob_hash) = 32),
    acquisition_status     TEXT NOT NULL DEFAULT 'unfetched'
                               CHECK(acquisition_status IN ('acquired', 'unavailable', 'unfetched')),
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

CREATE TABLE IF NOT EXISTS session_provider_usage_events (
    usage_event_id                 TEXT GENERATED ALWAYS AS (session_id || ':usage:' || position) STORED UNIQUE,
    session_id                     TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    source_message_id              TEXT REFERENCES messages(message_id) ON DELETE SET NULL,
    position                       INTEGER NOT NULL CHECK(position >= 0),
    provider_event_type            TEXT NOT NULL CHECK(provider_event_type IN ('token_count', 'message_usage')),
    model_name                     TEXT,
    last_input_tokens              INTEGER NOT NULL DEFAULT 0 CHECK(last_input_tokens >= 0),
    last_output_tokens             INTEGER NOT NULL DEFAULT 0 CHECK(last_output_tokens >= 0),
    last_cached_input_tokens       INTEGER NOT NULL DEFAULT 0 CHECK(last_cached_input_tokens >= 0),
    last_cache_write_tokens        INTEGER NOT NULL DEFAULT 0 CHECK(last_cache_write_tokens >= 0),
    last_reasoning_output_tokens   INTEGER NOT NULL DEFAULT 0 CHECK(last_reasoning_output_tokens >= 0),
    last_total_tokens              INTEGER NOT NULL DEFAULT 0 CHECK(last_total_tokens >= 0),
    total_input_tokens             INTEGER NOT NULL DEFAULT 0 CHECK(total_input_tokens >= 0),
    total_output_tokens            INTEGER NOT NULL DEFAULT 0 CHECK(total_output_tokens >= 0),
    total_cached_input_tokens      INTEGER NOT NULL DEFAULT 0 CHECK(total_cached_input_tokens >= 0),
    total_cache_write_tokens       INTEGER NOT NULL DEFAULT 0 CHECK(total_cache_write_tokens >= 0),
    total_reasoning_output_tokens  INTEGER NOT NULL DEFAULT 0 CHECK(total_reasoning_output_tokens >= 0),
    total_tokens                   INTEGER NOT NULL DEFAULT 0 CHECK(total_tokens >= 0),
    model_context_window           INTEGER CHECK(model_context_window IS NULL OR model_context_window >= 0),
    payload_json                   TEXT NOT NULL DEFAULT '{{}}' CHECK ({json_object_check("payload_json")}),
    occurred_at_ms                 INTEGER,
    PRIMARY KEY(session_id, position)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_provider_usage_events_session
ON session_provider_usage_events(session_id, position);

CREATE INDEX IF NOT EXISTS idx_session_provider_usage_events_source_message
ON session_provider_usage_events(source_message_id)
WHERE source_message_id IS NOT NULL;

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
                                    'session_profile', 'work_events', 'phases', 'latency', 'thread',
                                    'runs', 'observed_events', 'context_snapshots', 'provider_usage')),
    session_id                   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    materializer_version         INTEGER NOT NULL,
    materialized_at_ms           INTEGER NOT NULL,
    source_updated_at_ms         INTEGER,
    source_sort_key_ms           INTEGER,
    input_high_water_mark_ms     INTEGER,
    input_high_water_mark_source TEXT,
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
    input_high_water_mark        TEXT,
    input_high_water_mark_source TEXT,
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
    tokenize='unicode61 remove_diacritics 2'
);

-- FTS triggers for session_work_events_fts table are now dynamically composed from sql.py
-- (polylogue-a7xr.5: consolidate FTS trigger DDL to single source)

CREATE TABLE IF NOT EXISTS session_phases (
    phase_id        TEXT GENERATED ALWAYS AS (session_id || ':phase:' || position) STORED UNIQUE,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    position        INTEGER NOT NULL CHECK(position >= 0),
    start_index     INTEGER NOT NULL DEFAULT 0 CHECK(start_index >= 0),
    end_index       INTEGER NOT NULL DEFAULT 0 CHECK(end_index >= start_index),
    started_at_ms   INTEGER,
    ended_at_ms     INTEGER,
    duration_ms     INTEGER NOT NULL DEFAULT 0 CHECK(duration_ms >= 0),
    tool_counts_json TEXT NOT NULL DEFAULT '{{}}',
    word_count      INTEGER NOT NULL DEFAULT 0 CHECK(word_count >= 0),
    input_high_water_mark        TEXT,
    input_high_water_mark_source TEXT,
    evidence_json   TEXT NOT NULL DEFAULT '{{}}',
    inference_json  TEXT NOT NULL DEFAULT '{{}}',
    search_text     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(session_id, position)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_phases_session
ON session_phases(session_id, position);

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
    priced_at_ms                    INTEGER,
    -- 1vpm.1: dominant model by assistant output-token share + its canonical
    -- family (anthropic/openai/deepseek/...) -- the enabling primitive for
    -- the `delegations` view's orchestrator/subagent model identity.
    primary_model_name              TEXT,
    primary_model_family            TEXT
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

-- polylogue-y964: delegations is a versioned, recomputable read model spined
-- on the PARENT's own dispatch actions, not on `session_links` plumbing. Two
-- upstream facts drove a full rebuild rather than a column-alias fix:
--
--   1. `session_links` stores the CHILD in `src_session_id` and the PARENT in
--      `resolved_dst_session_id` (see resolve_session_links_for_session,
--      storage/sqlite/queries/session_links.py: `child_id =
--      row["src_session_id"]`, and the resolved session is written into
--      `sessions.parent_session_id`) -- never the reverse. The prior shipped
--      view aliased these backwards, so every `orchestrator_*` column
--      actually described the CHILD session and vice versa.
--   2. `branch_point_message_id` is the last message the CHILD inherited from
--      the PARENT's prefix for lineage composition (see the `session_links`
--      DDL comment above) -- it is not the message that issued the Task
--      dispatch, and for a spawned-fresh child it is NULL entirely. The
--      prior view joined it against the (mislabeled) parent session as a
--      dispatch pointer, which does not hold in general.
--
-- The spine is therefore every parent-side dispatch action (an `actions` row
-- with semantic_type='subagent' -- a Task tool_use, optionally paired with
-- its tool_result). Stable action-observed identity is
-- (parent_session_id, instruction_tool_use_block_id): two Task calls in one
-- assistant message keep two distinct rows, never a fanout. Every dispatch
-- action gets exactly one row even when no child ever resolves
-- (mapping_state='unresolved' -- a dispatch error, or a resolution still
-- pending). A resolved session_links(subagent) edge with no discoverable
-- parent-side dispatch action (e.g. a Codex async subagent) surfaces as
-- mapping_state='edge_only', counted but never given a fabricated
-- instruction. Dispatch actions are corroborated against resolved children
-- by rank-pairing IN TRANSCRIPT ORDER within one parent session (the same
-- "Nth query gets Nth result" idiom `actions` itself uses for tool_use/
-- tool_result) -- but ONLY when a parent's dispatch count and resolved-child
-- count agree; a mismatched count would mean guessing a winner, so those
-- rows surface as mapping_state='ambiguous' with a real instruction but no
-- fabricated child. Quarantined edges (cycle-break, #866/#1260) surface
-- explicitly as mapping_state='quarantined' rather than silently vanishing.
--
-- Model identity is deliberately NOT collapsed into one "orchestrator model"
-- column (polylogue-4c27): `dispatch_turn_model` is the model that authored
-- the specific message containing the dispatch action (messages.model_name);
-- `requested_model` is an explicit routing override read from the dispatch
-- tool_input, honestly NULL when the provider recorded no such field;
-- `parent_session_dominant_model`/`child_session_dominant_model` are the
-- session-level dominant-model fallback (by assistant output-token share,
-- see archive/session/runtime.py:_primary_model) -- explicitly named as a
-- session-wide aggregate, excluded from turn-level dispatch claims. Use
-- `archive.semantic.pricing.resolve_model_identity()` to derive
-- vendor/model-line/pricing-source/confidence from any of these raw columns.
--
-- 100% derivable from existing tables -- VIEW, not a table, matching the
-- `actions` precedent (derived tier, no convergence stage needed).
CREATE VIEW IF NOT EXISTS delegations AS
WITH dispatch_actions AS (
    SELECT
        a.session_id                           AS parent_session_id,
        a.message_id                           AS instruction_message_id,
        a.tool_use_block_id                    AS instruction_tool_use_block_id,
        a.tool_input                           AS instruction_payload,
        a.tool_result_block_id                 AS artifact_block_id,
        a.output_text                          AS artifact_text,
        a.is_error                             AS result_is_error,
        a.exit_code                            AS result_exit_code,
        m.model_name                           AS dispatch_turn_model,
        json_extract(a.tool_input, '$.model')  AS requested_model,
        ROW_NUMBER() OVER (
            PARTITION BY a.session_id
            ORDER BY a.message_id, a.tool_use_block_id
        )                                       AS dispatch_rank
    FROM actions a
    JOIN messages m ON m.message_id = a.message_id
    WHERE a.semantic_type = 'subagent'
),
resolved_children AS (
    SELECT
        l.resolved_dst_session_id              AS parent_session_id,
        l.src_session_id                       AS child_session_id,
        l.branch_point_message_id              AS branch_point_message_id,
        l.confidence                           AS link_confidence,
        l.method                               AS link_method,
        l.inheritance                          AS inheritance,
        ROW_NUMBER() OVER (
            PARTITION BY l.resolved_dst_session_id
            ORDER BY l.observed_at_ms, l.src_session_id
        )                                       AS child_rank
    FROM session_links l
    WHERE l.link_type = 'subagent'
      AND l.resolved_dst_session_id IS NOT NULL
      AND (l.status IS NULL OR l.status != 'quarantined')
),
dispatch_counts AS (
    SELECT parent_session_id, COUNT(*) AS n FROM dispatch_actions GROUP BY parent_session_id
),
child_counts AS (
    SELECT parent_session_id, COUNT(*) AS n FROM resolved_children GROUP BY parent_session_id
),
pairable AS (
    SELECT dc.parent_session_id
    FROM dispatch_counts dc
    JOIN child_counts cc ON cc.parent_session_id = dc.parent_session_id
    WHERE dc.n = cc.n
),
resolved_rows AS (
    SELECT
        d.parent_session_id                    AS parent_session_id,
        c.child_session_id                     AS child_session_id,
        'resolved'                              AS mapping_state,
        c.link_confidence                      AS link_confidence,
        c.link_method                          AS link_method,
        c.inheritance                          AS inheritance,
        c.branch_point_message_id              AS branch_point_message_id,
        d.instruction_message_id               AS instruction_message_id,
        d.instruction_tool_use_block_id        AS instruction_tool_use_block_id,
        d.instruction_payload                  AS instruction_payload,
        d.dispatch_turn_model                  AS dispatch_turn_model,
        d.requested_model                      AS requested_model,
        d.artifact_block_id                    AS artifact_block_id,
        d.artifact_text                        AS artifact_text,
        d.result_is_error                      AS result_is_error,
        d.result_exit_code                     AS result_exit_code
    FROM dispatch_actions d
    JOIN pairable p ON p.parent_session_id = d.parent_session_id
    JOIN resolved_children c
      ON c.parent_session_id = d.parent_session_id
     AND c.child_rank = d.dispatch_rank
),
unresolved_rows AS (
    SELECT
        d.parent_session_id                    AS parent_session_id,
        NULL                                    AS child_session_id,
        'unresolved'                             AS mapping_state,
        NULL AS link_confidence, NULL AS link_method, NULL AS inheritance, NULL AS branch_point_message_id,
        d.instruction_message_id               AS instruction_message_id,
        d.instruction_tool_use_block_id        AS instruction_tool_use_block_id,
        d.instruction_payload                  AS instruction_payload,
        d.dispatch_turn_model                  AS dispatch_turn_model,
        d.requested_model                      AS requested_model,
        d.artifact_block_id                    AS artifact_block_id,
        d.artifact_text                        AS artifact_text,
        d.result_is_error                      AS result_is_error,
        d.result_exit_code                     AS result_exit_code
    FROM dispatch_actions d
    LEFT JOIN child_counts cc ON cc.parent_session_id = d.parent_session_id
    WHERE cc.n IS NULL
),
ambiguous_rows AS (
    SELECT
        d.parent_session_id                    AS parent_session_id,
        NULL                                    AS child_session_id,
        'ambiguous'                              AS mapping_state,
        NULL AS link_confidence, NULL AS link_method, NULL AS inheritance, NULL AS branch_point_message_id,
        d.instruction_message_id               AS instruction_message_id,
        d.instruction_tool_use_block_id        AS instruction_tool_use_block_id,
        d.instruction_payload                  AS instruction_payload,
        d.dispatch_turn_model                  AS dispatch_turn_model,
        d.requested_model                      AS requested_model,
        d.artifact_block_id                    AS artifact_block_id,
        d.artifact_text                        AS artifact_text,
        d.result_is_error                      AS result_is_error,
        d.result_exit_code                     AS result_exit_code
    FROM dispatch_actions d
    JOIN dispatch_counts dc ON dc.parent_session_id = d.parent_session_id
    JOIN child_counts cc ON cc.parent_session_id = d.parent_session_id
    WHERE dc.n != cc.n
),
edge_only_rows AS (
    SELECT
        c.parent_session_id                    AS parent_session_id,
        c.child_session_id                     AS child_session_id,
        'edge_only'                              AS mapping_state,
        c.link_confidence                      AS link_confidence,
        c.link_method                          AS link_method,
        c.inheritance                          AS inheritance,
        c.branch_point_message_id              AS branch_point_message_id,
        NULL AS instruction_message_id, NULL AS instruction_tool_use_block_id, NULL AS instruction_payload,
        NULL AS dispatch_turn_model, NULL AS requested_model,
        NULL AS artifact_block_id, NULL AS artifact_text, NULL AS result_is_error, NULL AS result_exit_code
    FROM resolved_children c
    LEFT JOIN dispatch_counts dc ON dc.parent_session_id = c.parent_session_id
    WHERE dc.n IS NULL
),
quarantined_rows AS (
    SELECT
        l.resolved_dst_session_id              AS parent_session_id,
        l.src_session_id                       AS child_session_id,
        'quarantined'                            AS mapping_state,
        l.confidence                           AS link_confidence,
        l.method                               AS link_method,
        l.inheritance                          AS inheritance,
        l.branch_point_message_id              AS branch_point_message_id,
        NULL AS instruction_message_id, NULL AS instruction_tool_use_block_id, NULL AS instruction_payload,
        NULL AS dispatch_turn_model, NULL AS requested_model,
        NULL AS artifact_block_id, NULL AS artifact_text, NULL AS result_is_error, NULL AS result_exit_code
    FROM session_links l
    WHERE l.link_type = 'subagent'
      AND l.status = 'quarantined'
      AND l.resolved_dst_session_id IS NOT NULL
),
attempts AS (
    SELECT * FROM resolved_rows
    UNION ALL SELECT * FROM unresolved_rows
    UNION ALL SELECT * FROM ambiguous_rows
    UNION ALL SELECT * FROM edge_only_rows
    UNION ALL SELECT * FROM quarantined_rows
)
SELECT
    att.parent_session_id                      AS parent_session_id,
    att.child_session_id                       AS child_session_id,
    att.mapping_state                          AS mapping_state,
    att.link_confidence                        AS link_confidence,
    att.link_method                            AS link_method,
    att.inheritance                            AS inheritance,
    att.branch_point_message_id                AS branch_point_message_id,
    att.instruction_message_id                 AS instruction_message_id,
    att.instruction_tool_use_block_id          AS instruction_tool_use_block_id,
    att.instruction_payload                    AS instruction_payload,
    att.dispatch_turn_model                    AS dispatch_turn_model,
    att.requested_model                        AS requested_model,
    att.artifact_block_id                      AS artifact_block_id,
    att.artifact_text                          AS artifact_text,
    att.result_is_error                        AS result_is_error,
    att.result_exit_code                       AS result_exit_code,
    CASE
        WHEN att.instruction_tool_use_block_id IS NULL THEN 'unknown'
        WHEN att.result_is_error IS NULL               THEN 'unknown'
        WHEN att.result_is_error = 1                   THEN 'error'
        ELSE 'ok'
    END                                          AS result_status,
    p.origin                                    AS parent_origin,
    pp.primary_model_name                       AS parent_session_dominant_model,
    pp.primary_model_family                     AS parent_session_dominant_model_family,
    pp.terminal_state                           AS parent_terminal_state,
    cp.primary_model_name                       AS child_session_dominant_model,
    cp.primary_model_family                     AS child_session_dominant_model_family,
    cp.total_cost_usd                           AS child_cost_usd,
    cp.cost_is_estimated                        AS child_cost_is_estimated,
    (COALESCE(cp.total_input_tokens, 0) + COALESCE(cp.total_output_tokens, 0)
       + COALESCE(cp.total_cache_read_tokens, 0) + COALESCE(cp.total_cache_write_tokens, 0)) AS child_tokens,
    cp.wall_duration_ms                         AS child_wall_ms,
    cp.terminal_state                           AS child_terminal_state
FROM attempts att
JOIN sessions p ON p.session_id = att.parent_session_id
LEFT JOIN session_profiles pp ON pp.session_id = att.parent_session_id
LEFT JOIN session_profiles cp ON cp.session_id = att.child_session_id;

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

# polylogue-a7xr.5 consolidated the FTS trigger CREATE statements into
# storage/fts/sql.py as the single canonical source (also used by the
# repair-path lifecycle in fts_lifecycle.py), but a fresh-database bootstrap
# still needs them appended to the script it executescript()s -- without
# this, a freshly created index.db has the messages_fts/threads_fts/
# session_work_events_fts virtual tables but no triggers populating them,
# so every insert silently produces an empty search index. Regression:
# devtools render demo-corpus-datasheet started failing with "Search index
# is incomplete" right after #2893 landed.
INDEX_DDL = INDEX_DDL + "\n\n" + ";\n\n".join(FTS_TRIGGER_DDL) + ";\n"

__all__ = ["FTS_FRESHNESS_STATE_DDL", "INDEX_DDL", "INDEX_SCHEMA_VERSION"]
