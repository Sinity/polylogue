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

from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.raw_authority import prune_orphaned_index_revision_seeds as _prune_orphaned_index_revision_seeds
from polylogue.storage.raw_authority import reset_raw_authority_census_ledger


@dataclass(frozen=True, slots=True)
class RawAuthorityResetReport:
    """Row counts removed from the census ledger (dry-run or applied)."""

    censuses: int
    plans: int
    blockers: int
    census_plans: int
    census_post_plans: int
    applied: bool


def _offline_config(archive_root: Path) -> Config:
    return Config(archive_root=archive_root, render_root=render_root(), sources=[])


def reset_raw_authority_census(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    dry_run: bool = True,
) -> RawAuthorityResetReport:
    """Empty the census planning ledger. ``dry_run`` reports counts only."""
    if not dry_run and (
        reason := offline_maintenance_block_reason(_offline_config(archive_root), active=True, dry_run=False)
    ):
        raise RuntimeError(reason)
    before = reset_raw_authority_census_ledger(
        archive_root,
        backup_manifest=backup_manifest,
        dry_run=dry_run,
    )

    return RawAuthorityResetReport(
        censuses=before.censuses,
        plans=before.plans,
        blockers=before.blockers,
        census_plans=before.census_plans,
        census_post_plans=before.census_post_plans,
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
    if not dry_run and (
        reason := offline_maintenance_block_reason(_offline_config(archive_root), active=True, dry_run=False)
    ):
        raise RuntimeError(reason)
    counts = _prune_orphaned_index_revision_seeds(archive_root, dry_run=dry_run)
    return IndexSeedPruneReport(
        revision_heads=counts.revision_heads,
        revision_applications=counts.revision_applications,
        applied=not dry_run,
    )
