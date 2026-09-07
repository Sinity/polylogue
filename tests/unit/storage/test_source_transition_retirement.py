"""The retired source-transition receipt family is gone from fresh archives.

Eight one-shot transition receipt tables recorded what a historical actuator
did to the pre-restart archive. Their actuators are deleted, so a fresh source
generation must not declare them, and a source tier that carries them anyway --
a migrated historical tier, or a tampered one -- must not be able to change
what the archive builds.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.schema_inventory import canonical_schema_objects
from polylogue.storage.sqlite.archive_tiers.source import RETIRED_SOURCE_SCHEMA_OBJECTS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_canonical_snapshot import (
    assert_canonical_snapshots_equal,
    capture_canonical_snapshot,
)
from tests.infra.convergence_harness import (
    ConvergenceArchive,
    converge_convergence_archive,
    ingest_composed_sources,
    initialize_active_archive,
    rich_convergence_sources,
)

#: The transition receipt tables whose actuators are deleted. Each recorded a
#: one-shot promotion out of quarantine, an exclusion, or a reclassification
#: applied to the pre-restart archive.
RETIRED_TRANSITION_TABLES = (
    "raw_live_source_reconciliation_receipts",
    "raw_membership_writeback_receipts",
    "raw_append_chain_backfill_receipts",
    "raw_byte_duplicate_supersession_receipts",
    "raw_failure_disposition_receipts",
    "raw_non_session_duplicate_exclusion_receipts",
    "raw_quarantine_group_dedup_receipts",
    "raw_unknown_export_reclassification_receipts",
)


def _ingest(root: Path) -> ConvergenceArchive:
    """Ingest the shared source set without converging derived models yet."""
    composed = rich_convergence_sources()
    initialize_active_archive(root)
    return ingest_composed_sources(
        root,
        composed,
        session_indexes=tuple(range(len(composed.sessions))),
        converge_after_each=False,
    )


def _seed_retired_transition_rows(source_path: Path) -> dict[str, int]:
    """Create every retired table and fill it with arbitrary rows.

    The rows name real ``raw_sessions`` rows so a reader that still consulted
    them would find a usable, and deliberately wrong, answer.
    """
    counts: dict[str, int] = {}
    with sqlite3.connect(source_path) as conn:
        raws = conn.execute("SELECT raw_id, source_path, blob_hash, blob_size FROM raw_sessions").fetchall()
        assert raws, "the ingest produced no raw rows to attach transition evidence to"
        conn.executescript(
            """
            CREATE TABLE raw_live_source_reconciliation_receipts (
                raw_id TEXT PRIMARY KEY, verdict TEXT NOT NULL, source_path TEXT NOT NULL
            ) STRICT;
            CREATE TABLE raw_membership_writeback_receipts (
                raw_id TEXT PRIMARY KEY, membership_decision TEXT NOT NULL
            ) STRICT;
            CREATE TABLE raw_append_chain_backfill_receipts (
                raw_id TEXT PRIMARY KEY, source_path TEXT NOT NULL
            ) STRICT;
            CREATE TABLE raw_byte_duplicate_supersession_receipts (
                raw_id TEXT PRIMARY KEY, blob_hash BLOB NOT NULL, blob_size INTEGER NOT NULL,
                duplicate_of_raw_id TEXT NOT NULL, duplicate_of_session_id TEXT NOT NULL
            ) STRICT;
            CREATE TABLE raw_failure_disposition_receipts (
                raw_id TEXT PRIMARY KEY, disposition_kind TEXT NOT NULL
            ) STRICT;
            CREATE TABLE raw_non_session_duplicate_exclusion_receipts (
                raw_id TEXT PRIMARY KEY, indexed_twin_raw_id TEXT NOT NULL
            ) STRICT;
            CREATE TABLE raw_quarantine_group_dedup_receipts (
                raw_id TEXT PRIMARY KEY, representative_raw_id TEXT NOT NULL
            ) STRICT;
            CREATE TABLE raw_unknown_export_reclassification_receipts (
                raw_id TEXT PRIMARY KEY, previous_origin TEXT NOT NULL, new_origin TEXT NOT NULL
            ) STRICT;
            """
        )
        for raw_id, path, blob_hash, blob_size in raws:
            conn.execute(
                "INSERT INTO raw_live_source_reconciliation_receipts VALUES (?, 'exact_match', ?)",
                (raw_id, path),
            )
            conn.execute("INSERT INTO raw_membership_writeback_receipts VALUES (?, 'accepted')", (raw_id,))
            conn.execute("INSERT INTO raw_append_chain_backfill_receipts VALUES (?, ?)", (raw_id, path))
            conn.execute(
                "INSERT INTO raw_byte_duplicate_supersession_receipts VALUES (?, ?, ?, ?, ?)",
                (raw_id, blob_hash, blob_size, "seeded-twin-raw", "seeded-twin-session"),
            )
            conn.execute("INSERT INTO raw_failure_disposition_receipts VALUES (?, 'terminal_corrupt_input')", (raw_id,))
            conn.execute(
                "INSERT INTO raw_non_session_duplicate_exclusion_receipts VALUES (?, 'seeded-twin-raw')", (raw_id,)
            )
            conn.execute(
                "INSERT INTO raw_quarantine_group_dedup_receipts VALUES (?, 'seeded-representative-raw')", (raw_id,)
            )
            conn.execute(
                "INSERT INTO raw_unknown_export_reclassification_receipts VALUES (?, 'unknown-export', 'chatgpt-export')",
                (raw_id,),
            )
        conn.commit()
        for table in RETIRED_TRANSITION_TABLES:
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return counts


@pytest.mark.parametrize("table", RETIRED_TRANSITION_TABLES)
def test_fresh_source_ddl_omits_every_retired_transition_table(table: str) -> None:
    """A fresh source generation declares none of the retired receipt objects."""
    declared = {obj.table_name for obj in canonical_schema_objects(ArchiveTier.SOURCE)}
    assert table not in declared
    assert f"table:{table}" in RETIRED_SOURCE_SCHEMA_OBJECTS


def test_retired_objects_are_declared_rather_than_merely_absent() -> None:
    """Parity excludes the retired objects explicitly, so migrated tiers stay readable."""
    declared = {obj.object_ref.split(":", 1)[1] for obj in canonical_schema_objects(ArchiveTier.SOURCE)}
    assert not declared & RETIRED_SOURCE_SCHEMA_OBJECTS


def test_seeded_transition_rows_cannot_alter_what_the_fresh_archive_builds(tmp_path: Path) -> None:
    """Convergence ignores the retired receipt family even when it is populated.

    Anti-vacuity: the seeded tables are asserted non-empty, and
    ``test_the_comparator_sees_a_real_authority_mutation`` proves the same
    comparator is sensitive to a source-tier change that does matter.
    """
    clean = _ingest(tmp_path / "clean")
    converge_convergence_archive(clean)

    seeded = _ingest(tmp_path / "seeded")
    counts = _seed_retired_transition_rows(seeded.root / "source.db")
    assert all(count > 0 for count in counts.values()), counts
    converge_convergence_archive(seeded)

    assert_canonical_snapshots_equal(capture_canonical_snapshot(clean.root), capture_canonical_snapshot(seeded.root))


def test_the_comparator_sees_a_real_authority_mutation(tmp_path: Path) -> None:
    """The sensitivity check the seeded-rows test relies on."""
    clean = _ingest(tmp_path / "clean")
    converge_convergence_archive(clean)

    mutated = _ingest(tmp_path / "mutated")
    with sqlite3.connect(mutated.root / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET blob_hash = zeroblob(32) "
            "WHERE raw_id = (SELECT raw_id FROM raw_sessions ORDER BY raw_id LIMIT 1)"
        )
        conn.commit()
    converge_convergence_archive(mutated)

    with pytest.raises(AssertionError):
        assert_canonical_snapshots_equal(
            capture_canonical_snapshot(clean.root), capture_canonical_snapshot(mutated.root)
        )
