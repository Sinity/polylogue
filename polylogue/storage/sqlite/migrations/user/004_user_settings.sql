CREATE TABLE IF NOT EXISTS user_settings (
    setting_key    TEXT PRIMARY KEY,
    value_json     TEXT NOT NULL,
    updated_at_ms  INTEGER NOT NULL,
    author_ref     TEXT NOT NULL DEFAULT 'user:local'
) STRICT;
