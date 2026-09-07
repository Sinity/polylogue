from __future__ import annotations

import sqlite3

import pytest

from devtools.schema_fast_forward import (
    SchemaFastForwardEngine,
    SchemaFastForwardError,
    SchemaFastForwardStep,
)


def _seeded_connection(version: int = 1) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE records (value TEXT NOT NULL)")
    conn.execute("INSERT INTO records VALUES ('kept')")
    conn.execute(f"PRAGMA user_version = {version}")
    return conn


def _assert_index(db: sqlite3.Connection) -> None:
    row = db.execute("SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'records_source_idx'").fetchone()
    assert row is not None, "index was not created"


def test_engine_handles_seeded_version_gap_with_one_declaration_set() -> None:
    conn = _seeded_connection()
    engine = SchemaFastForwardEngine(
        (
            SchemaFastForwardStep(2, "ALTER TABLE records ADD COLUMN source TEXT"),
            SchemaFastForwardStep(
                3,
                "CREATE INDEX records_source_idx ON records(source)",
                verify=_assert_index,
            ),
        ),
        tier="fixture",
    )

    result = engine.execute(conn)

    assert result.source_version == 1
    assert result.target_version == 3
    assert result.applied_versions == (2, 3)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
    assert conn.execute("SELECT value FROM records").fetchone()[0] == "kept"
    assert "source" in {row[1] for row in conn.execute("PRAGMA table_info(records)")}


def test_engine_rejects_a_missing_version_before_mutating() -> None:
    conn = _seeded_connection()
    engine = SchemaFastForwardEngine((SchemaFastForwardStep(3, "ALTER TABLE records ADD COLUMN source TEXT"),))

    with pytest.raises(SchemaFastForwardError, match="incomplete at v2"):
        engine.execute(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert "source" not in {row[1] for row in conn.execute("PRAGMA table_info(records)")}


def test_failed_step_rolls_back_schema_and_version_and_can_retry() -> None:
    conn = _seeded_connection()
    fail = {"enabled": True}

    def apply(db: sqlite3.Connection) -> None:
        db.execute("ALTER TABLE records ADD COLUMN source TEXT")
        if fail["enabled"]:
            raise RuntimeError("injected failure")

    engine = SchemaFastForwardEngine((SchemaFastForwardStep(2, apply),), tier="fixture")

    with pytest.raises(SchemaFastForwardError, match="injected failure"):
        engine.execute(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert "source" not in {row[1] for row in conn.execute("PRAGMA table_info(records)")}

    fail["enabled"] = False
    result = engine.execute(conn)
    assert result.applied_versions == (2,)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2


def test_already_current_is_an_idempotent_noop() -> None:
    conn = _seeded_connection(version=2)
    engine = SchemaFastForwardEngine((SchemaFastForwardStep(2, "ALTER TABLE records ADD COLUMN source TEXT"),))

    result = engine.execute(conn)

    assert result.applied_versions == ()
    assert result.changed is False
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
