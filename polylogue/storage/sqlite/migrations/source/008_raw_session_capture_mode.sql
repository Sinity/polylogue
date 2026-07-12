ALTER TABLE raw_sessions ADD COLUMN capture_mode TEXT CHECK(capture_mode IN (
    'chatgpt', 'claude-ai', 'claude-code', 'codex', 'gemini', 'gemini-cli',
    'hermes', 'antigravity', 'grok', 'drive', 'unknown'
));
