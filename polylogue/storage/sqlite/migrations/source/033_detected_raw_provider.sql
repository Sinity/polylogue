-- Parser classification is not acquisition identity. Retain the exact
-- provider-wire result separately so replay can reuse it without rewriting
-- raw_sessions.origin and breaking deterministic reacquisition.
ALTER TABLE raw_sessions ADD COLUMN detected_provider TEXT CHECK (
    (detected_provider IN (
        'chatgpt', 'claude-ai', 'claude-design', 'claude-code', 'codex',
        'gemini', 'gemini-cli', 'hermes', 'antigravity', 'beads', 'grok',
        'drive', 'unknown'
    ) OR detected_provider IS NULL)
);

-- ZIP member raw ids bind the content hash while source_index stores a paired
-- central-directory ordinal and within-member split index. Source replacement
-- may legitimately change the row's blob hash without changing raw_id, so the
-- coordinate format must remain independently durable for later recovery.
CREATE TABLE raw_container_coordinates (
    raw_id             TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    coordinate_format  TEXT NOT NULL CHECK(coordinate_format = 'zip-v2'),
    entry_ordinal      INTEGER NOT NULL CHECK(entry_ordinal >= 0),
    split_index        INTEGER NOT NULL CHECK(split_index >= 0)
) STRICT;
