"""Reset the raw-authority census planning ledger so the daemon rebuilds it.

The raw-authority census tables (``raw_authority_censuses`` and its
``plans``/``blockers``/``census_plans``/``census_post_plans`` children) are
DERIVED convergence bookkeeping: each census normally chains to its predecessor
(``sequence_no + 1``), while bounded header retention may set the oldest
retained header's predecessor to NULL as the explicit compaction boundary.
The ledger also carries unresolved plans forward. The ACCEPTED
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

from polylogue.maintenance.raw_authority_recovery import (
    RecoveryOperation,
    apply_raw_authority_recovery,
    inspect_raw_authority_recovery,
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


def reset_raw_authority_census(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    dry_run: bool = True,
) -> RawAuthorityResetReport:
    """Inspect or apply the guarded census-ledger recovery route."""
    plan = inspect_raw_authority_recovery(
        archive_root,
        RecoveryOperation.RESET_CENSUS,
        backup_manifest=backup_manifest,
    )
    report = apply_raw_authority_recovery(plan, backup_manifest=backup_manifest) if not dry_run else None
    counts = plan.before_counts

    return RawAuthorityResetReport(
        censuses=counts["raw_authority_censuses"],
        plans=counts["raw_authority_plans"],
        blockers=counts["raw_authority_blockers"],
        census_plans=counts["raw_authority_census_plans"],
        census_post_plans=counts["raw_authority_census_post_plans"],
        applied=report is not None and report.status == "applied",
    )


@dataclass(frozen=True, slots=True)
class IndexSeedPruneReport:
    """Index revision-authority read-model rows removed (dry-run or applied)."""

    revision_heads: int
    revision_applications: int
    applied: bool


def prune_orphaned_index_revision_seeds(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    dry_run: bool = True,
) -> IndexSeedPruneReport:
    """Delete index raw-frontier seeds whose raw is gone from the source tier.

    ``raw_revision_heads`` / ``raw_revision_applications`` are the index's
    (rebuildable) revision-authority read model. After a source raw is deleted
    (hook de-inflation), the seeds referencing it become broken predecessor
    chains — the daemon's raw-frontier integrity check reports them as violated
    and cannot converge past them. Deleting the seeds whose ``accepted_raw_id`` /
    ``raw_id`` no longer exists in ``source.raw_sessions`` restores a clean
    frontier; seeds for present raws are untouched.
    """
    plan = inspect_raw_authority_recovery(
        archive_root,
        RecoveryOperation.PRUNE_INDEX_SEEDS,
        backup_manifest=backup_manifest,
    )
    report = apply_raw_authority_recovery(plan, backup_manifest=backup_manifest) if not dry_run else None
    return IndexSeedPruneReport(
        revision_heads=len(plan.candidate_keys["raw_revision_heads"]),
        revision_applications=len(plan.candidate_keys["raw_revision_applications"]),
        applied=report is not None and report.status == "applied",
    )
