from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.schema import SCHEMA_VERSION, _ensure_schema
from polylogue.storage.sqlite.schema_bootstrap import (
    SchemaSnapshot,
    build_v8_to_v9_upgrade_plan,
    build_v12_to_v13_upgrade_plan,
)


def test_ensure_schema_upgrades_v8_by_indexing_message_foreign_keys(tmp_path: Path) -> None:
    """The v8->v9 path adds indexes needed for fast message replacement."""
    db_path = tmp_path / "v8-fk-indexes.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            blob_size INTEGER NOT NULL,
            acquired_at TEXT NOT NULL
        );
        CREATE TABLE attachment_refs (
            ref_id TEXT PRIMARY KEY,
            attachment_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            message_id TEXT
        );
        CREATE TABLE provider_events (
            event_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            source_message_id TEXT
        );
        PRAGMA user_version = 8;
        """
    )
    conn.commit()

    _ensure_schema(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    indexes = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%_message'"
        ).fetchall()
    }
    assert "idx_attachment_refs_message" in indexes
    assert "idx_provider_events_source_message" in indexes
    conn.close()


def test_v8_to_v9_upgrade_plan_indexes_message_foreign_keys() -> None:
    """Deleting messages must not scan provider-events or attachment refs."""
    snapshot = SchemaSnapshot(
        current_version=8,
        table_columns={
            "raw_conversations": frozenset(),
            "attachment_refs": frozenset(),
            "provider_events": frozenset(),
        },
        index_sql={},
    )

    plan = build_v8_to_v9_upgrade_plan(snapshot)

    assert plan.scripts == ()
    assert any("idx_attachment_refs_message" in statement for statement in plan.statements)
    assert any("idx_provider_events_source_message" in statement for statement in plan.statements)


def test_ensure_schema_upgrades_v12_marks_to_target_aware_user_state(tmp_path: Path) -> None:
    """The v12->v13 path preserves existing conversation marks and adds annotations."""
    db_path = tmp_path / "v12-user-state.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT,
            title TEXT,
            content_hash TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT,
            text TEXT,
            content_hash TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE user_marks (
            conversation_id TEXT NOT NULL,
            mark_type TEXT NOT NULL CHECK (mark_type IN ('star', 'pin', 'archive')),
            created_at TEXT NOT NULL,
            PRIMARY KEY (conversation_id, mark_type)
        );
        INSERT INTO conversations(conversation_id, provider_name, provider_conversation_id, title, content_hash)
            VALUES ('conv-1', 'chatgpt', 'provider-1', 'Legacy', 'hash-1');
        INSERT INTO messages(message_id, conversation_id, role, text, content_hash)
            VALUES ('msg-1', 'conv-1', 'user', 'Hello', 'msg-hash-1');
        INSERT INTO user_marks(conversation_id, mark_type, created_at)
            VALUES ('conv-1', 'star', '2026-05-15T00:00:00+00:00');
        PRAGMA user_version = 12;
        """
    )
    conn.commit()

    _ensure_schema(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    columns = {row[1] for row in conn.execute("PRAGMA table_info(user_marks)").fetchall()}
    assert {"target_type", "target_id", "conversation_id", "message_id", "mark_type", "created_at"} <= columns
    row = conn.execute(
        "SELECT target_type, target_id, conversation_id, message_id, mark_type FROM user_marks"
    ).fetchone()
    assert row == ("conversation", "conv-1", "conv-1", None, "star")
    assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_annotations'").fetchone()
    conn.close()


def test_v12_to_v13_upgrade_plan_migrates_legacy_mark_table() -> None:
    snapshot = SchemaSnapshot(
        current_version=12,
        table_columns={"user_marks": frozenset({"conversation_id", "mark_type", "created_at"})},
        index_sql={},
    )

    plan = build_v12_to_v13_upgrade_plan(snapshot)

    assert plan.scripts == ()
    assert any("ALTER TABLE user_marks RENAME TO user_marks_legacy" in statement for statement in plan.statements)
    assert any("CREATE TABLE IF NOT EXISTS user_annotations" in statement for statement in plan.statements)
