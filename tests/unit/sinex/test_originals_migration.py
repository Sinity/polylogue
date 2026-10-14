"""The original Source floor upgrades additively to publication-original custody."""

from __future__ import annotations

import sqlite3
from importlib.resources import files

from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, SOURCE_TIER_VERSION
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_BASELINE_DDL
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_existing_source001_upgrade_repeat_and_fresh_ddl_have_the_same_objects() -> None:
    sql = (
        files("polylogue.storage.sqlite.migrations.source").joinpath("002_sinex_publication_originals.sql").read_text()
    )
    steps = migration_runner._load_migrations(ArchiveTier.SOURCE)
    assert len(steps) == 1 and steps[0].version == SOURCE_TIER_VERSION == 2
    with sqlite3.connect(":memory:") as upgraded, sqlite3.connect(":memory:") as fresh:
        upgraded.executescript(SOURCE_BASELINE_DDL)
        upgraded.execute("PRAGMA user_version=1")
        upgraded.execute(
            "INSERT INTO sinex_publication_obligations "
            "(object_id,protocol_version,revision_id,manifest_digest,mode,status,attempt_count,created_at_ms,updated_at_ms) "
            "VALUES ('neutral','v1','revision','manifest','mirror','pending',0,1,1)"
        )
        upgraded.commit()
        original_rows = tuple(upgraded.execute("SELECT * FROM sinex_publication_obligations"))
        upgraded.execute("BEGIN IMMEDIATE")
        migration_runner._execute_migration_sql(upgraded, sql)
        upgraded.execute("PRAGMA user_version=2")
        upgraded.commit()
        upgraded.execute("BEGIN IMMEDIATE")
        migration_runner._execute_migration_sql(upgraded, sql)
        upgraded.commit()
        assert tuple(upgraded.execute("SELECT * FROM sinex_publication_obligations")) == original_rows
        assert upgraded.execute("PRAGMA user_version").fetchone()[0] == 2
        fresh.executescript(ARCHIVE_DDL_BY_TIER[ArchiveTier.SOURCE])

        def inventory(conn: sqlite3.Connection) -> tuple[tuple[object, ...], ...]:
            return tuple(
                conn.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
                )
            )

        assert inventory(upgraded) == inventory(fresh)
        assert upgraded.execute("PRAGMA foreign_key_list(sinex_publication_originals)").fetchall()
