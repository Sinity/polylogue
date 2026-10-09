"""``additive-no-backup`` is a proven classification, not a self-declaration.

The marker waives the verified-backup requirement for a numbered migration on
an irreplaceable durable tier. That is the right classification for a migration
whose whole statement set creates schema objects and writes no row -- the
runner has carried the class since #2905 and the shipped source slot 002 is
exactly that shape -- but nothing checked the claim against the file's own SQL,
so any statement could sit under the header and skip the backup.

Anti-vacuity: drop the ``_assert_additive_migration_sql`` call from
``_requires_migration_backup`` and every ``_refused`` case below goes green
(the claim is accepted and ``requires_backup`` is ``False``). The
``_still_waives`` cases pin the opposite direction so a guard that refused
every marked migration -- or that quietly started demanding a backup for
slot 002 -- cannot pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.sqlite.migration_runner import (
    MigrationError,
    _requires_migration_backup,
)

_MARKER = "-- migration-safety: additive-no-backup"

_ADDITIVE = (
    f"{_MARKER}\n"
    "-- a comment between statements\n"
    "CREATE TABLE IF NOT EXISTS t (a TEXT NOT NULL) STRICT;\n"
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_t_a ON t(a);\n"
    "CREATE VIEW IF NOT EXISTS v AS SELECT a FROM t;\n"
)

_NOT_ADDITIVE = {
    "row_delete": "DELETE FROM raw_sessions;",
    "row_update": "UPDATE raw_sessions SET blob_hash = NULL;",
    "row_insert": "INSERT INTO raw_sessions (raw_id) VALUES ('x');",
    "table_drop": "DROP TABLE raw_sessions;",
    "column_add": "ALTER TABLE raw_sessions ADD COLUMN extra TEXT;",
    "create_as_select": "CREATE TABLE copied AS SELECT * FROM raw_sessions;",
    "create_as_block_comment_select": "CREATE TABLE copied AS/*gap*/SELECT * FROM raw_sessions;",
    "create_as_line_comment_select": "CREATE TABLE copied AS\n-- gap\nSELECT * FROM raw_sessions;",
    "trigger": "CREATE TRIGGER trg AFTER INSERT ON t BEGIN DELETE FROM t; END;",
}


class TestAdditiveClaimIsProven:
    @pytest.mark.parametrize("case", sorted(_NOT_ADDITIVE))
    @pytest.mark.parametrize("separator", ["\n", " ", " /* comment; */ "])
    def test_a_false_additive_claim_is_refused(self, case: str, separator: str) -> None:
        sql = f"{_MARKER}\nCREATE TABLE IF NOT EXISTS t (a TEXT) STRICT;{separator}{_NOT_ADDITIVE[case]}\n"
        with pytest.raises(MigrationError, match="not additive-only"):
            _requires_migration_backup(Path("099_claimed.sql"), sql)

    def test_an_honest_additive_migration_still_waives_the_backup(self) -> None:
        """Opposite direction: refusing every marked migration would fail here."""
        assert _requires_migration_backup(Path("099_honest.sql"), _ADDITIVE) is False

    def test_an_unmarked_migration_still_requires_a_backup(self) -> None:
        """Opposite direction: the guard must not waive anything on its own."""
        assert _requires_migration_backup(Path("099_plain.sql"), "ALTER TABLE t ADD COLUMN b TEXT;\n") is True


@pytest.mark.parametrize("terminator", [";", ""])
def test_claim_discovery_rejects_same_line_destructive_suffix(terminator: str) -> None:
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.migration_runner import durable_migration_claim_for_sql

    sql = f"{_MARKER}\nCREATE TABLE harmless (id INTEGER); DELETE FROM raw_sessions{terminator}"
    with pytest.raises(MigrationError, match="not additive-only"):
        durable_migration_claim_for_sql(ArchiveTier.SOURCE, "099_claimed.sql", sql)


def test_sqlite_statement_boundaries_preserve_quoted_semicolons_and_comments() -> None:
    sql = (
        f"{_MARKER}\n"
        "CREATE TABLE [semi;colon] (value TEXT DEFAULT 'it''s; safe'); "
        "/* a comment; not a statement */ CREATE VIEW v AS SELECT ';' AS value; "
        "-- trailing comment;\n"
    )
    assert _requires_migration_backup(Path("099_honest.sql"), sql) is False


def test_trigger_body_is_one_statement_not_several() -> None:
    from polylogue.storage.sqlite.migration_runner import _iter_migration_statements

    trigger = "CREATE TRIGGER t AFTER INSERT ON items BEGIN UPDATE items SET x = ';'; DELETE FROM other; END;"
    assert list(_iter_migration_statements(trigger + " CREATE TABLE harmless (id INTEGER);")) == [
        trigger,
        "CREATE TABLE harmless (id INTEGER);",
    ]


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TABLE copied AS/*gap*/SELECT * FROM raw_sessions;",
        "CREATE TABLE copied AS\n-- gap\nSELECT * FROM raw_sessions;",
    ],
)
def test_commented_ctas_is_row_copying_sqlite_and_not_additive(statement: str) -> None:
    """Exercise the exact SQLite row effect and the production safety classifier."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE raw_sessions (raw_id TEXT)")
        conn.execute("INSERT INTO raw_sessions VALUES ('source-row')")
        conn.execute(statement)
        assert conn.execute("SELECT raw_id FROM copied").fetchall() == [("source-row",)]
    finally:
        conn.close()

    with pytest.raises(MigrationError, match="not additive-only"):
        _requires_migration_backup(Path("099_commented_ctas.sql"), f"{_MARKER}\n{statement}")


def test_ctas_words_inside_quoted_names_and_literals_remain_additive() -> None:
    import sqlite3

    statement = "CREATE TABLE \"name AS SELECT\" (value TEXT DEFAULT 'AS SELECT');"
    assert _requires_migration_backup(Path("099_quoted.sql"), f"{_MARKER}\n{statement}") is False
    with sqlite3.connect(":memory:") as conn:
        conn.execute(statement)
        conn.execute('INSERT INTO "name AS SELECT" DEFAULT VALUES')
        assert conn.execute('SELECT value FROM "name AS SELECT"').fetchone() == ("AS SELECT",)
