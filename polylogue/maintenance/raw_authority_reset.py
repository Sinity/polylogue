"""Reset the raw-authority census planning ledger so the daemon rebuilds it.

The raw-authority census tables (``raw_authority_censuses`` and its
``plans``/``blockers``/``census_plans``/``census_post_plans`` children) are
DERIVED convergence bookkeeping: each census chains to its predecessor
(``sequence_no + 1``) and carries unresolved plans forward. The ACCEPTED
materialization state — ``raw_sessions.revision_authority`` and the index's
``raw_revision_heads`` / ``raw_revision_applications`` — lives OUTSIDE these
tables and is untouched here.

When the ledger accumulates inconsistent carried-forward state (e.g. a stale-plan
blocker marks sibling plans ``CARRIED_FORWARD``, and later raw deletions drop
them out of the recomputed frontier so the finalize postflight
``persistent ⊄ post_ids`` throws — live incident 2026-07-22 after hook
de-inflation), no new census can finalize to become a clean baseline, and the
daemon defers every pass. Emptying the ledger removes the poisoned predecessor:
the next daemon pass builds census #1 fresh over the current raw set
(``predecessor = None``), with no carried-forward history and no stale blocker.

``raw_authority_parser_census`` is intentionally KEPT — it holds resource-blocked
parser fingerprints keyed to raws (FK-cascaded from ``raw_sessions``), not
census-cycle bookkeeping, and the whale pass consumes it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.archive_identity import ArchiveLocation

# Children before parents: blockers/census_plans/census_post_plans reference
# raw_authority_plans via NO ACTION (RESTRICT) FKs; census_plans/post_plans also
# CASCADE from raw_authority_censuses. Deleting in this order with foreign_keys
# ON leaves no dangling reference.
_LEDGER_TABLES_CHILD_FIRST = (
    "raw_authority_blockers",
    "raw_authority_census_plans",
    "raw_authority_census_post_plans",
    "raw_authority_plans",
    "raw_authority_censuses",
)


@dataclass(frozen=True, slots=True)
class RawAuthorityResetReport:
    """Row counts removed from the census ledger (dry-run or applied)."""

    censuses: int
    plans: int
    blockers: int
    census_plans: int
    census_post_plans: int
    applied: bool


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in _LEDGER_TABLES_CHILD_FIRST
    }


def reset_raw_authority_census(archive_root: Path, *, dry_run: bool = True) -> RawAuthorityResetReport:
    """Empty the census planning ledger. ``dry_run`` reports counts only."""
    source_db = archive_root / "source.db"
    if not source_db.is_file():
        raise FileNotFoundError(source_db)

    with sqlite3.connect(source_db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        before = _counts(conn)

        if not dry_run:
            active_config = Config(
                archive_root=archive_root,
                render_root=render_root(),
                sources=[],
                db_path=ArchiveLocation.resolve(archive_root).active_index_path,
            )
            if reason := offline_maintenance_block_reason(active_config, active=True, dry_run=False):
                raise RuntimeError(reason)
            for table in _LEDGER_TABLES_CHILD_FIRST:
                conn.execute(f"DELETE FROM {table}")

    return RawAuthorityResetReport(
        censuses=before["raw_authority_censuses"],
        plans=before["raw_authority_plans"],
        blockers=before["raw_authority_blockers"],
        census_plans=before["raw_authority_census_plans"],
        census_post_plans=before["raw_authority_census_post_plans"],
        applied=not dry_run,
    )


@dataclass(frozen=True, slots=True)
class IndexSeedPruneReport:
    """Index revision-authority read-model rows removed (dry-run or applied)."""

    revision_heads: int
    revision_applications: int
    applied: bool


def prune_orphaned_index_revision_seeds(archive_root: Path, *, dry_run: bool = True) -> IndexSeedPruneReport:
    """Delete index raw-frontier seeds whose raw is gone from the source tier.

    ``raw_revision_heads`` / ``raw_revision_applications`` are the index's
    (rebuildable) revision-authority read model. After a source raw is deleted
    (hook de-inflation), the seeds referencing it become broken predecessor
    chains — the daemon's raw-frontier integrity check reports them as violated
    and cannot converge past them. Deleting the seeds whose ``accepted_raw_id`` /
    ``raw_id`` no longer exists in ``source.raw_sessions`` restores a clean
    frontier; seeds for present raws are untouched.
    """
    index_db = archive_root / "index.db"
    source_db = archive_root / "source.db"
    if not index_db.exists() or not source_db.is_file():
        raise FileNotFoundError(index_db if not index_db.exists() else source_db)

    with sqlite3.connect(index_db) as conn:
        conn.execute("ATTACH DATABASE ? AS src", (str(source_db),))
        heads = int(
            conn.execute(
                "SELECT COUNT(*) FROM raw_revision_heads WHERE accepted_raw_id NOT IN (SELECT raw_id FROM src.raw_sessions)"
            ).fetchone()[0]
        )
        apps = int(
            conn.execute(
                "SELECT COUNT(*) FROM raw_revision_applications WHERE raw_id NOT IN (SELECT raw_id FROM src.raw_sessions)"
            ).fetchone()[0]
        )
        if not dry_run:
            active_config = Config(
                archive_root=archive_root,
                render_root=render_root(),
                sources=[],
                db_path=ArchiveLocation.resolve(archive_root).active_index_path,
            )
            if reason := offline_maintenance_block_reason(active_config, active=True, dry_run=False):
                raise RuntimeError(reason)
            conn.execute(
                "DELETE FROM raw_revision_heads WHERE accepted_raw_id NOT IN (SELECT raw_id FROM src.raw_sessions)"
            )
            conn.execute(
                "DELETE FROM raw_revision_applications WHERE raw_id NOT IN (SELECT raw_id FROM src.raw_sessions)"
            )

    return IndexSeedPruneReport(revision_heads=heads, revision_applications=apps, applied=not dry_run)
