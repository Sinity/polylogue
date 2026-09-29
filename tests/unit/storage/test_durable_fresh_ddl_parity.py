"""Fresh-vs-migrated parity reports every extra object and binds applied bytes."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import (
    DurableFreshDDLParityProof,
    capture_durable_database_evidence,
    prove_durable_fresh_ddl_parity,
)

SOURCE_TIER_VERSION = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]


@contextmanager
def _fresh_source() -> Iterator[sqlite3.Connection]:
    """A source tier exactly as current canonical DDL builds it."""
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript(SOURCE_DDL)
        conn.execute(f"PRAGMA user_version = {SOURCE_TIER_VERSION}")
        conn.commit()
        yield conn


def _prove(migrated: sqlite3.Connection, fresh: sqlite3.Connection) -> DurableFreshDDLParityProof:
    return prove_durable_fresh_ddl_parity(
        ArchiveTier.SOURCE,
        SOURCE_TIER_VERSION,
        migrated_connection=migrated,
        fresh_connection=fresh,
        evidence_ref="proof:fresh-ddl-parity-test",
    )


def test_identical_canonical_tiers_have_parity() -> None:
    with _fresh_source() as fresh, _fresh_source() as migrated:
        proof = _prove(migrated, fresh)

    assert proof.matches is True
    assert proof.migrated_inventory_sha256 == proof.fresh_inventory_sha256


def test_an_extra_table_is_unexpected() -> None:
    """No object outside fresh DDL is exempt from parity."""
    with _fresh_source() as fresh, _fresh_source() as migrated:
        migrated.execute("CREATE TABLE raw_not_in_fresh_ddl (raw_id TEXT PRIMARY KEY) STRICT")
        migrated.commit()
        proof = _prove(migrated, fresh)

    assert proof.unexpected_objects == ("table:raw_not_in_fresh_ddl",)
    assert proof.matches is False


def test_a_parity_proof_over_different_bytes_fails_the_applied_binding(tmp_path: Path) -> None:
    """The parity digest is the same capture the apply evidence records.

    A parity proof captured from a database whose schema is not the one the
    apply recorded must be unequal to that ``post`` evidence.
    """
    source_path = tmp_path / "source.db"
    # A real file, because ``capture_durable_database_evidence`` binds its
    # evidence to the tier's own archive identity on disk.
    initialize_archive_database(source_path, ArchiveTier.SOURCE)
    with closing(sqlite3.connect(source_path)) as migrated:
        post = capture_durable_database_evidence(migrated, ArchiveTier.SOURCE)
        with _fresh_source() as fresh:
            clean = _prove(migrated, fresh)
        migrated.execute("CREATE TABLE raw_undeclared_drift (id TEXT PRIMARY KEY) STRICT")
        migrated.commit()
        with _fresh_source() as fresh:
            drifted = _prove(migrated, fresh)

    assert clean.matches is True
    assert clean.migrated_inventory_sha256 == post.schema_inventory_sha256
    assert drifted.unexpected_objects == ("table:raw_undeclared_drift",)
    assert drifted.matches is False
    assert drifted.migrated_inventory_sha256 != post.schema_inventory_sha256
