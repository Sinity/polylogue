"""Schema and storage coverage for first-class provider events."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.schema import SCHEMA_VERSION, _ensure_schema


def test_ensure_schema_upgrades_v6_provider_meta_compactions_to_provider_events(tmp_path: Path) -> None:
    db_path = tmp_path / "v6-provider-events.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            payload_provider TEXT,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            blob_size INTEGER NOT NULL,
            acquired_at TEXT NOT NULL
        );
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            sort_key REAL,
            content_hash TEXT NOT NULL DEFAULT '',
            provider_meta TEXT,
            metadata TEXT DEFAULT '{}',
            source_name TEXT GENERATED ALWAYS AS (json_extract(provider_meta, '$.source')) STORED,
            version INTEGER NOT NULL,
            raw_id TEXT REFERENCES raw_conversations(raw_id) ON DELETE SET NULL
        );
        INSERT INTO raw_conversations (
            raw_id,
            provider_name,
            source_name,
            source_path,
            blob_size,
            acquired_at
        ) VALUES (
            'raw-1',
            'codex',
            'codex',
            '/tmp/session.jsonl',
            10,
            '2026-04-01T10:00:00+00:00'
        );
        INSERT INTO conversations (
            conversation_id,
            provider_name,
            provider_conversation_id,
            title,
            content_hash,
            provider_meta,
            metadata,
            version,
            raw_id
        ) VALUES (
            'codex:session-1',
            'codex',
            'session-1',
            'Session',
            'hash-1',
            '{"source":"codex","context_compactions":[{"summary":"Earlier context","timestamp":"2026-04-01T10:05:00+00:00"}],"git":{"branch":"master"}}',
            '{}',
            1,
            'raw-1'
        );
        PRAGMA user_version = 6;
        """
    )
    conn.commit()

    _ensure_schema(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    row = conn.execute(
        """
        SELECT event_id, conversation_id, provider_name, event_index, event_type,
               timestamp, payload_json, raw_id, materializer_version
        FROM provider_events
        WHERE conversation_id = ?
        """,
        ("codex:session-1",),
    ).fetchone()
    assert row is not None
    assert row["event_id"] == "codex:session-1:provider-event:000000"
    assert row["provider_name"] == "codex"
    assert row["event_index"] == 0
    assert row["event_type"] == "compaction"
    assert row["timestamp"] == "2026-04-01T10:05:00+00:00"
    assert row["raw_id"] == "raw-1"
    assert row["materializer_version"] == 1
    assert json.loads(row["payload_json"])["summary"] == "Earlier context"

    provider_meta = conn.execute(
        "SELECT provider_meta FROM conversations WHERE conversation_id = ?",
        ("codex:session-1",),
    ).fetchone()[0]
    assert "context_compactions" not in provider_meta
    assert '"source":"codex"' in provider_meta.replace(" ", "")
    assert '"git":{"branch":"master"}' in provider_meta.replace(" ", "")
    conn.close()
