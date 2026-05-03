"""Schema upgrade coverage for action-event FTS rowid alignment."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.backends.schema import SCHEMA_VERSION, _ensure_schema
from polylogue.storage.backends.schema_bootstrap import SchemaSnapshot, build_v3_to_v4_upgrade_plan


def _create_v3_action_fts_fixture(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE action_events (
            event_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            materializer_version INTEGER NOT NULL DEFAULT 1,
            sequence_index INTEGER NOT NULL,
            action_kind TEXT NOT NULL,
            normalized_tool_name TEXT NOT NULL,
            search_text TEXT NOT NULL
        );
        CREATE INDEX idx_action_events_conversation
        ON action_events(conversation_id);
        CREATE VIRTUAL TABLE action_events_fts USING fts5(
            event_id UNINDEXED,
            message_id UNINDEXED,
            conversation_id UNINDEXED,
            action_kind UNINDEXED,
            tool_name UNINDEXED,
            text,
            tokenize='unicode61'
        );
        CREATE TRIGGER action_events_fts_ai
        AFTER INSERT ON action_events BEGIN
            INSERT INTO action_events_fts (event_id, message_id, conversation_id, action_kind, tool_name, text)
            VALUES (
                new.event_id,
                new.message_id,
                new.conversation_id,
                new.action_kind,
                new.normalized_tool_name,
                new.search_text
            );
        END;
        CREATE TRIGGER action_events_fts_ad
        AFTER DELETE ON action_events BEGIN
            DELETE FROM action_events_fts WHERE rowid = old.rowid;
        END;
        INSERT INTO action_events_fts (event_id, message_id, conversation_id, action_kind, tool_name, text)
        VALUES ('sentinel', 'msg-sentinel', 'conv-sentinel', 'shell', 'bash', 'old rowid offset');
        INSERT INTO action_events (
            event_id, conversation_id, message_id, sequence_index,
            action_kind, normalized_tool_name, search_text
        ) VALUES
            ('event-a', 'conv-a', 'msg-a', 0, 'shell', 'bash', 'rowid migration needle'),
            ('event-b', 'conv-a', 'msg-b', 1, 'search', 'rg', 'second action');
        PRAGMA user_version = 3;
        """
    )
    conn.commit()


def test_ensure_schema_upgrades_v3_action_fts_rowids_without_reimport(tmp_path: Path) -> None:
    """The v3->v4 path rebuilds action FTS rows with base-table rowids."""
    db_path = tmp_path / "v3-action-fts.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _create_v3_action_fts_fixture(conn)

    before_count = conn.execute("SELECT COUNT(*) FROM action_events").fetchone()[0]
    mismatch_before = conn.execute(
        """
        SELECT COUNT(*)
        FROM action_events ae
        LEFT JOIN action_events_fts f ON f.rowid = ae.rowid
        WHERE f.event_id IS NULL OR f.event_id != ae.event_id
        """
    ).fetchone()[0]
    assert mismatch_before == before_count

    _ensure_schema(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM action_events").fetchone()[0] == before_count
    mismatch_after = conn.execute(
        """
        SELECT COUNT(*)
        FROM action_events ae
        LEFT JOIN action_events_fts f ON f.rowid = ae.rowid
        WHERE f.event_id IS NULL OR f.event_id != ae.event_id
        """
    ).fetchone()[0]
    assert mismatch_after == 0
    trigger_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='action_events_fts_ai'"
    ).fetchone()[0]
    assert "rowid, event_id" in trigger_sql
    conn.close()


def test_v3_to_v4_upgrade_plan_rebuilds_action_fts_with_rowids() -> None:
    snapshot = SchemaSnapshot(
        current_version=3,
        table_columns={
            "action_events": frozenset({"event_id", "conversation_id"}),
            "action_events_fts": frozenset({"event_id", "conversation_id", "text"}),
        },
        index_sql={},
    )

    plan = build_v3_to_v4_upgrade_plan(snapshot)

    assert "DELETE FROM action_events_fts" in plan.statements
    assert any("INSERT INTO action_events_fts (rowid," in " ".join(statement.split()) for statement in plan.statements)
