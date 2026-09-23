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

from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import (
    MigrationError,
    _load_migrations,
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
    "trigger": "CREATE TRIGGER trg AFTER INSERT ON t BEGIN DELETE FROM t; END;",
}


class TestAdditiveClaimIsProven:
    @pytest.mark.parametrize("case", sorted(_NOT_ADDITIVE))
    def test_a_false_additive_claim_is_refused(self, case: str) -> None:
        sql = f"{_MARKER}\nCREATE TABLE IF NOT EXISTS t (a TEXT) STRICT;\n{_NOT_ADDITIVE[case]}\n"
        with pytest.raises(MigrationError, match="not additive-only"):
            _requires_migration_backup(Path("099_claimed.sql"), sql)

    def test_an_honest_additive_migration_still_waives_the_backup(self) -> None:
        """Opposite direction: refusing every marked migration would fail here."""
        assert _requires_migration_backup(Path("099_honest.sql"), _ADDITIVE) is False

    def test_an_unmarked_migration_still_requires_a_backup(self) -> None:
        """Opposite direction: the guard must not waive anything on its own."""
        assert _requires_migration_backup(Path("099_plain.sql"), "ALTER TABLE t ADD COLUMN b TEXT;\n") is True

    def test_the_shipped_source_slot_still_waives_the_backup(self) -> None:
        """The shipped durable migration's classification is unchanged and now proven."""
        steps = _load_migrations(ArchiveTier.SOURCE)
        assert steps, "the source tier declares no numbered migration"
        assert [(step.name, step.requires_backup) for step in steps] == [
            ("002_excision_policy_projections.sql", False),
            ("003_raw_member_identity.sql", True),
        ]
