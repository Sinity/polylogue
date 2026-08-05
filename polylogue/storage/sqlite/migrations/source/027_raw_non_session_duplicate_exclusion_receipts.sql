-- polylogue-r9xsj: evidence for a duplicate excluded as non-conversation must
-- bind its source blob to the indexed twin. raw_membership_census.status alone
-- records a parser classification, not a durable duplicate disposition.
-- Legacy source fixtures may predate the original GC generation table even
-- though the durable source contract has always required it. Recreate the
-- additive table here so the current migration repairs that omission without
-- weakening fresh-DDL parity.
CREATE TABLE IF NOT EXISTS gc_generations (
    generation_id    TEXT PRIMARY KEY,
    started_at_ms    INTEGER NOT NULL,
    completed_at_ms  INTEGER,
    reclaimed_count  INTEGER NOT NULL DEFAULT 0 CHECK(reclaimed_count >= 0),
    reclaimed_bytes  INTEGER NOT NULL DEFAULT 0 CHECK(reclaimed_bytes >= 0)
) STRICT;

-- Migration 022 rebuilt raw_hook_events for hook-payload blob references but
-- omitted its canonical Origin CHECK. Archives that already advanced beyond
-- v22 need this current-train copy-forward repair as well.
ALTER TABLE raw_hook_events RENAME TO raw_hook_events_v26;

CREATE TABLE raw_hook_events (
    hook_event_id     TEXT PRIMARY KEY,
    origin            TEXT NOT NULL CHECK(origin IN (
        'claude-code-session', 'codex-session', 'gemini-cli-session',
        'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export',
        'chatgpt-export', 'claude-ai-export', 'claude-design-session',
        'aistudio-drive', 'unknown-export'
    )),
    native_id         TEXT,
    session_native_id TEXT,
    source_path       TEXT NOT NULL,
    event_type        TEXT NOT NULL,
    payload_json      TEXT NOT NULL,
    observed_at_ms    INTEGER NOT NULL,
    blob_hash         BLOB CHECK(blob_hash IS NULL OR length(blob_hash) = 32)
) STRICT;

INSERT INTO raw_hook_events (
    hook_event_id, origin, native_id, session_native_id, source_path, event_type,
    payload_json, observed_at_ms, blob_hash
)
SELECT
    hook_event_id, origin, native_id, session_native_id, source_path, event_type,
    payload_json, observed_at_ms, blob_hash
FROM raw_hook_events_v26;

DROP TABLE raw_hook_events_v26;

CREATE INDEX idx_raw_hook_events_session
ON raw_hook_events(origin, session_native_id, observed_at_ms);

CREATE TABLE IF NOT EXISTS raw_non_session_duplicate_exclusion_receipts (
    raw_id                     TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    blob_hash                  BLOB NOT NULL CHECK(length(blob_hash) = 32),
    blob_size                  INTEGER NOT NULL CHECK(blob_size >= 0),
    indexed_twin_raw_id        TEXT NOT NULL,
    indexed_twin_session_id    TEXT NOT NULL,
    parser_fingerprint         TEXT NOT NULL,
    excluded_at_ms             INTEGER NOT NULL CHECK(excluded_at_ms >= 0),
    tool_version               TEXT NOT NULL,
    detail                     TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_non_session_duplicate_exclusion_receipts_twin
ON raw_non_session_duplicate_exclusion_receipts(indexed_twin_raw_id);
