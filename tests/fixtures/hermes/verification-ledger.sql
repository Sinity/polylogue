CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE verification_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    session_id TEXT NOT NULL,
    cwd TEXT NOT NULL,
    root TEXT NOT NULL,
    command TEXT NOT NULL,
    canonical_command TEXT NOT NULL,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    status TEXT NOT NULL,
    exit_code INTEGER NOT NULL,
    output_summary TEXT NOT NULL
);
CREATE TABLE verification_state (
    session_id TEXT NOT NULL,
    root TEXT NOT NULL,
    last_event_id INTEGER,
    last_edit_at TEXT,
    changed_paths_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (session_id, root)
);
INSERT INTO meta(key, value) VALUES ('schema_version', '1');
