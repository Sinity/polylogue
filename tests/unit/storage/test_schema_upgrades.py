from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.schema import SCHEMA_VERSION, _ensure_schema
from polylogue.storage.sqlite.schema_bootstrap import SchemaSnapshot, build_v8_to_v9_upgrade_plan


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
