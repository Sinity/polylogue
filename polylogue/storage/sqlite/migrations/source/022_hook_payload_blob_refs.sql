-- polylogue-tfzw0: hook-event blob refs are born orphaned.
--
-- write_source_hook_event (storage/sqlite/archive_tiers/source_write.py)
-- deliberately never mints a raw_sessions row for a hook event
-- (polylogue-31r1 -- doing so inflated the archive with ~65K empty session
-- shells). Its durable blob ref was nonetheless written as
-- ref_type='raw_payload' keyed by a *synthetic* raw_id that never
-- corresponds to any raw_sessions row. Blob GC's liveness check
-- (storage/blob_gc.py:_still_referenced) treated ANY row in blob_refs as
-- proof of liveness -- a tautology, since the very row being asked about is
-- itself the evidence -- so these refs were retained forever, uncounted, and
-- invisible to GC (73,427 rows / ~1.94 GiB measured on the live archive).
--
-- This migration gives hook-event blob refs their own ref_type
-- ('hook_payload') and their own real referent (raw_hook_events.hook_event_id,
-- backed by a new blob_hash column on that table) so a corrected liveness
-- check (this bead's blob_gc.py change) can join ref_id against the table the
-- ref_type actually names, instead of trusting blob_refs' own existence.
--
-- SQLite cannot ALTER a CHECK constraint, and a bare `ALTER TABLE
-- raw_hook_events ADD COLUMN blob_hash` would collide with any archive/test
-- fixture that already carries a blob_hash column from a fresher bootstrap
-- masquerading at an older PRAGMA user_version (this repo's own migration
-- test fixtures do exactly that). Both tables are therefore rebuilt via the
-- same create-differently-named-table / copy-forward-by-explicit-column /
-- drop-original / rename-into-place dance migration 009 and (pending, PR
-- #3598) 021 use -- safe regardless of whether the source table already
-- happens to have a blob_hash column, because the copy only ever selects the
-- columns guaranteed to exist pre-v22.
--
-- Numbering note: at the time this migration was authored, migration 021
-- (widening the origin CHECK for claude-design-session, PR #3598) had not yet
-- merged. This file is deliberately numbered 022 (the next free slot after
-- 021) rather than reusing 021's number; the durable migration chain will have
-- a real gap (020 -> 022, skipping 021) until PR #3598 merges. An archive at
-- v20 cannot migrate to v22 until 021 exists on master -- this is expected,
-- not a bug in this file, and is called out in this PR's body.

CREATE TABLE IF NOT EXISTS raw_hook_events (
    hook_event_id      TEXT PRIMARY KEY,
    origin             TEXT NOT NULL CHECK(origin IN (
        'claude-code-session', 'codex-session', 'gemini-cli-session',
        'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export',
        'chatgpt-export', 'claude-ai-export', 'claude-design-session',
        'aistudio-drive', 'unknown-export'
    )),
    native_id          TEXT,
    session_native_id  TEXT,
    source_path        TEXT NOT NULL,
    event_type         TEXT NOT NULL,
    payload_json       TEXT NOT NULL,
    observed_at_ms     INTEGER NOT NULL
) STRICT;

ALTER TABLE raw_hook_events RENAME TO raw_hook_events_v21;

CREATE TABLE raw_hook_events (
    hook_event_id   TEXT PRIMARY KEY,
    origin          TEXT NOT NULL CHECK(origin IN (
        'claude-code-session', 'codex-session', 'gemini-cli-session',
        'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export',
        'chatgpt-export', 'claude-ai-export', 'claude-design-session',
        'aistudio-drive', 'unknown-export'
    )),
    native_id       TEXT,
    session_native_id TEXT,
    source_path     TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    observed_at_ms  INTEGER NOT NULL,
    blob_hash       BLOB CHECK(blob_hash IS NULL OR length(blob_hash) = 32)
) STRICT;

INSERT INTO raw_hook_events (
    hook_event_id, origin, native_id, session_native_id, source_path, event_type,
    payload_json, observed_at_ms
)
SELECT
    hook_event_id, origin, native_id, session_native_id, source_path, event_type,
    payload_json, observed_at_ms
FROM raw_hook_events_v21;

DROP TABLE raw_hook_events_v21;

CREATE INDEX idx_raw_hook_events_session
ON raw_hook_events(origin, session_native_id, observed_at_ms);

CREATE TABLE IF NOT EXISTS blob_refs (
    blob_hash       BLOB NOT NULL CHECK(length(blob_hash) = 32),
    ref_id          TEXT NOT NULL,
    ref_type        TEXT NOT NULL CHECK(ref_type IN ('raw_payload', 'attachment', 'sidecar')),
    source_path     TEXT,
    size_bytes      INTEGER NOT NULL CHECK(size_bytes >= 0),
    acquired_at_ms  INTEGER NOT NULL,
    PRIMARY KEY(blob_hash, ref_type, ref_id)
) STRICT;

ALTER TABLE blob_refs RENAME TO blob_refs_v21;

CREATE TABLE blob_refs (
    blob_hash       BLOB NOT NULL CHECK(length(blob_hash) = 32),
    ref_id          TEXT NOT NULL,
    ref_type        TEXT NOT NULL CHECK(ref_type IN ('raw_payload', 'attachment', 'sidecar', 'hook_payload')),
    source_path     TEXT,
    size_bytes      INTEGER NOT NULL CHECK(size_bytes >= 0),
    acquired_at_ms  INTEGER NOT NULL,
    PRIMARY KEY(blob_hash, ref_type, ref_id)
) STRICT;

INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
SELECT blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms
FROM blob_refs_v21;

DROP TABLE blob_refs_v21;

CREATE INDEX idx_blob_refs_ref_id
ON blob_refs(ref_id);
