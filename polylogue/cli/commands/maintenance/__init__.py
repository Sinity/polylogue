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
    ("run", "_run", "run_command", "Run (or dry-run) maintenance backfill operations."),
    (
        "rebuild-index",
        "_rebuild_index",
        "rebuild_index_command",
        "Inspect or execute an authority-safe source-to-index rebuild.",
    ),
    (
        "missing-raw-blob-cursors",
        "_raw_identity",
        "missing_raw_blob_cursors_command",
        "Invalidate cursors hiding missing raw-blob re-acquisition debt.",
    ),
    (
        "quarantined-accepted-raws",
        "_raw_identity",
        "quarantined_accepted_raws_command",
        "Repair a typed accepted full raw whose byte authority stayed quarantined.",
    ),
    (
        "browser-capture-origin-mismatches",
        "_raw_identity",
        "browser_capture_origin_mismatches_command",
        "Copy mismatched browser captures forward under parsed origin authority.",
    ),
    (
        "legacy-browser-capture-missing-native-id",
        "_raw_identity",
        "legacy_browser_capture_missing_native_id_command",
        "Copy forward the narrow legacy browser shape whose native ID is NULL.",
    ),
    (
        "browser-canonical-authority-conflicts",
        "_raw_identity",
        "browser_canonical_authority_conflicts_command",
        "Show why a byte-proven-rekey actuator refuses these browser-capture raws.",
    ),
    (
        "duplicate-raw-identity",
        "_raw_identity",
        "duplicate_raw_identity_command",
        "Reconcile a pre-#2729 duplicate raw pair onto its post-fix accepted head.",
    ),
    ("preview", "_preview", "preview_command", "Staleness inventory by model and scope. Read-only."),
    ("blob-gc", "_blob_gc", "blob_gc_command", "Preview or run lease-safe blob garbage collection."),
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
        "blob-reference-replace-from-source",
        "_blob_integrity",
        "blob_reference_replace_from_source_command",
        "Replace raw-backed missing blob refs with current source-derived bytes.",
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
)


@click.group("maintenance")
def maintenance_group() -> None:
    """Preview and run maintenance backfill operations."""


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
