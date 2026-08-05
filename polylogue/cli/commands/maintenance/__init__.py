"""Maintenance command group: preview and run backfills.

Each subcommand lives in its own submodule and is attached lazily (the
``_LazyCommand`` pattern already used for the root CLI's own dispatch,
:mod:`polylogue.cli.click_command_registration`), so importing this
package -- which happens on *any* ``ops maintenance ...`` invocation,
including a bare ``--help`` listing -- never imports a specific
subcommand's own heavy runtime dependencies (``ArchiveStore``,
``blob_gc``, ``migration_runner``, ``embeddings.reconcile``, ...). Those
load only when that specific subcommand is actually resolved (dispatched
or its own ``--help`` requested). polylogue-sod7.
"""

from __future__ import annotations

import click

from polylogue.cli.click_command_registration import _LazyCommand

# (cli name, submodule, attribute, short_help)
_COMMANDS: tuple[tuple[str, str, str, str], ...] = (
    ("plan", "_plan", "plan_command", "Dry-run summary: show what would be rebuilt without executing."),
    ("archive-plan", "_archive_plan", "archive_plan_command", "Inspect readiness for the archive file set."),
    (
        "backup-plan",
        "_backup_plan",
        "backup_plan_command",
        "Inspect archive backup boundaries without copying data.",
    ),
    (
        "assertion-export",
        "_assertion_export",
        "assertion_export_command",
        "Export the durable assertion substrate from user.db.",
    ),
    ("archive-read", "_archive_read", "archive_read_command", "Read index sessions from the archive."),
    (
        "archive-init",
        "_archive_plan",
        "archive_init_command",
        "Initialize the archive file set after explicit confirmation.",
    ),
    (
        "migrate-tier",
        "_migrate_tier",
        "migrate_tier_command",
        "Apply additive migrations for one durable archive tier.",
    ),
    (
        "run-preview",
        "_run_preview",
        "run_preview_command",
        "Preview a maintenance run (dry, resumable) without executing. Read-only.",
    ),
    ("run", "_run", "run_command", "Execute maintenance backfill operations."),
    (
        "rebuild-index",
        "_rebuild_index",
        "rebuild_index_command",
        "Inspect or execute an authority-safe source-to-index rebuild.",
    ),
    (
        "rebuild-index-status",
        "_rebuild_index_status",
        "rebuild_index_status_command",
        "Report consolidated raw-replay rebuild status (lease/generation/cursor/delta/recovery). Read-only.",
    ),
    (
        "reindex-canary",
        "_reindex_canary",
        "reindex_canary_command",
        "Replay a representative inactive canary and persist its reviewed diff report.",
    ),
    (
        "raw-authority-frontier",
        "_raw_identity",
        "raw_authority_frontier_command",
        "Inspect and record the raw-authority frontier; plan application is daemon-owned.",
    ),
    (
        "raw-authority-census",
        "_raw_identity",
        "raw_authority_census_command",
        "Read a bounded page from a durable raw-authority census ledger.",
    ),
    (
        "raw-authority-detail",
        "_raw_identity",
        "raw_authority_detail_command",
        "Read a bounded chunk of a complete raw-authority ledger record.",
    ),
    (
        "raw-authority-blockers",
        "_raw_identity",
        "raw_authority_blockers_command",
        "List unresolved raw-authority blockers (frontier-judgment vs stale-plan). Read-only.",
    ),
    (
        "raw-authority-blocker-resolve",
        "_raw_identity",
        "raw_authority_blocker_resolve_command",
        "Resolve one stale-plan blocker against current source evidence.",
    ),
    ("preview", "_preview", "preview_command", "Staleness inventory by model and scope. Read-only."),
    ("blob-gc", "_blob_gc", "blob_gc_command", "Preview lease-safe blob garbage collection. Read-only."),
    (
        "blob-publications",
        "_blob_publications",
        "blob_publications_command",
        "Inspect publication receipts or explicitly abandon selected debt.",
    ),
    (
        "blob-reference-debt",
        "_blob_integrity",
        "blob_reference_debt_command",
        "Classify missing referenced blobs without mutating the archive.",
    ),
    (
        "blob-reference-liveness",
        "_blob_integrity",
        "blob_reference_liveness_command",
        "Classify source-tier orphan refs; apply only with backup and receipt.",
    ),
    (
        "hook-payload-ref-reconcile",
        "_hook_payload_ref_reconciliation",
        "hook_payload_ref_reconcile_command",
        "Classify legacy hook refs; apply exact matches with backup and receipt.",
    ),
    (
        "blob-namespace-quarantine",
        "_blob_namespace_quarantine",
        "blob_namespace_quarantine_command",
        "Dry-run-first offline quarantine for invalid blob namespace entries.",
    ),
    (
        "attachment-acquisition-debt",
        "_blob_integrity",
        "attachment_acquisition_debt_command",
        "Classify index-tier attachment acquisition state without mutating the archive.",
    ),
    (
        "blob-reference-recovery-plan",
        "_blob_integrity",
        "blob_reference_recovery_plan_command",
        "Plan recovery for raw-backed missing blobs without mutating archive state.",
    ),
    (
        "blob-reference-replace-from-source-preview",
        "_blob_integrity",
        "blob_reference_replace_from_source_preview_command",
        "Preview raw-backed blob-reference replacement without mutating the archive.",
    ),
    (
        "blob-reference-replace-from-source",
        "_blob_integrity",
        "blob_reference_replace_from_source_command",
        "Replace raw-backed missing blob refs with current source-derived bytes.",
    ),
    (
        "blob-reference-prune-orphans-preview",
        "_blob_integrity",
        "blob_reference_prune_orphans_preview_command",
        "Preview orphan blob_refs pruning without mutating the archive.",
    ),
    (
        "blob-reference-prune-orphans",
        "_blob_integrity",
        "blob_reference_prune_orphans_command",
        "Quarantine and prune missing blob_refs that no longer have raw rows.",
    ),
    (
        "embedding-orphan-reconcile",
        "_embeddings",
        "embedding_orphan_reconcile_command",
        "Inspect (default) or reconcile embeddings.db rows orphaned by an index rebuild.",
    ),
    ("gc-history", "_blob_gc", "gc_history_command", "Show recent blob-GC passes recorded in ``gc_generations``."),
    ("status", "_status", "status_command", "Inspect persisted maintenance operations (#1197)."),
    (
        "verify-archive",
        "_verify_archive",
        "verify_archive_command",
        "Prove the archive is coherent after a rebuild, restore, or promotion. Read-only.",
    ),
)


@click.group("maintenance")
@click.pass_context
def maintenance_group(ctx: click.Context) -> None:
    """Preview and run maintenance backfill operations.

    Prints the resolved archive root and its provenance (env override /
    config file / default) before dispatching to any subcommand -- see
    ``_shared.print_archive_root_provenance`` (polylogue-l1qg). This runs
    for every real subcommand invocation, including ``--help`` on a
    subcommand, but not for ``maintenance``/``maintenance --help`` on their
    own (no subcommand resolves, so there is nothing to warn about yet).
    """
    from polylogue.cli.commands.maintenance._shared import print_archive_root_provenance
    from polylogue.cli.shared.types import AppEnv

    env = ctx.obj
    if isinstance(env, AppEnv):
        print_archive_root_provenance(env)


for _cli_name, _submodule, _attr, _short_help in _COMMANDS:
    maintenance_group.add_command(
        _LazyCommand(
            _cli_name,
            f"polylogue.cli.commands.maintenance.{_submodule}",
            _attr,
            short_help=_short_help,
        )
    )

del _cli_name, _submodule, _attr, _short_help

__all__ = ["maintenance_group"]
