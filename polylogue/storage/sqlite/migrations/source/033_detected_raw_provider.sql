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
