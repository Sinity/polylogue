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


def test_two_statements_on_one_line_both_apply() -> None:
    """Statement boundaries are SQLite's, not the author's line breaks.

    ``sqlite3.complete_statement`` reports the whole accumulated line as
    complete, so a line-at-a-time scan handed ``A; B;`` to ``execute`` in one
    call and SQLite refused it: whether a valid transition applied depended
    only on where its author pressed Enter.

    Anti-vacuity: restore the line-at-a-time accumulation and this raises
    ``SchemaFastForwardError`` wrapping "You can only execute one statement at
    a time". ``test_a_newline_separated_transition_still_applies`` pins the
    opposite direction so splitting on every semicolon -- which would break a
    trigger body or a semicolon inside a string literal -- cannot pass.
    """
    conn = _seeded_connection()
    engine = SchemaFastForwardEngine(
        (
            SchemaFastForwardStep(
                2,
                "ALTER TABLE records ADD COLUMN source TEXT; ALTER TABLE records ADD COLUMN note TEXT;",
            ),
        ),
        tier="fixture",
    )

    result = engine.execute(conn)

    assert result.applied_versions == (2,)
    assert {"source", "note"} <= {row[1] for row in conn.execute("PRAGMA table_info(records)")}


def test_a_newline_separated_transition_still_applies() -> None:
    """A semicolon inside a string literal or a trigger body is not a boundary."""
    conn = _seeded_connection()
    engine = SchemaFastForwardEngine(
        (
            SchemaFastForwardStep(
                2,
                "ALTER TABLE records ADD COLUMN note TEXT DEFAULT 'a;b';\n"
                "CREATE TRIGGER records_guard AFTER INSERT ON records\n"
                "BEGIN\n"
                "  UPDATE records SET note = 'seen;' WHERE rowid = NEW.rowid;\n"
                "END;\n",
            ),
        ),
        tier="fixture",
    )

    engine.execute(conn)

    conn.execute("INSERT INTO records (value) VALUES ('new')")
    assert conn.execute("SELECT note FROM records WHERE value = 'new'").fetchone()[0] == "seen;"
    assert conn.execute("SELECT note FROM records WHERE value = 'kept'").fetchone()[0] == "a;b"
