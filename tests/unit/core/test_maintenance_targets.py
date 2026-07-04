from __future__ import annotations

from polylogue.maintenance.targets import (
    CLEANUP_TARGETS,
    MAINTENANCE_TARGET_NAMES,
    SAFE_REPAIR_TARGETS,
    MaintenanceTargetMode,
    build_maintenance_target_catalog,
)


def test_maintenance_target_catalog_groups_targets_by_mode() -> None:
    catalog = build_maintenance_target_catalog()

    assert catalog.names() == MAINTENANCE_TARGET_NAMES
    # SAFE_REPAIR_TARGETS is the doctor's ``--repair`` umbrella set: every
    # REPAIR-mode target in the catalog.
    repair_mode_targets = catalog.names_for_mode(MaintenanceTargetMode.REPAIR)
    assert repair_mode_targets == SAFE_REPAIR_TARGETS
    assert catalog.names_for_mode(MaintenanceTargetMode.CLEANUP) == CLEANUP_TARGETS


def test_maintenance_target_catalog_resolves_aliases_to_canonical_targets() -> None:
    catalog = build_maintenance_target_catalog()

    spec = catalog.resolve_name("raw_snapshots")

    assert spec is not None
    assert spec.name == "superseded_raw_snapshots"


def test_maintenance_target_catalog_reports_preview_and_help_semantics() -> None:
    catalog = build_maintenance_target_catalog()

    assert catalog.preview_target_names() == (
        "session_insights",
        "message_type_backfill",
    )
    assert catalog.help_text() == (
        "Limit maintenance to named targets such as session_insights, "
        "message_type_backfill, orphaned_messages, empty_sessions, "
        "orphaned_attachments, orphaned_blobs, or superseded_raw_snapshots"
    )


def test_maintenance_target_catalog_exposes_archive_readiness_specs() -> None:
    catalog = build_maintenance_target_catalog()

    assert tuple(spec.name for spec in catalog.archive_readiness_specs(deep=False)) == (
        "orphaned_messages",
        "empty_sessions",
        "orphaned_attachments",
    )
    assert tuple(spec.name for spec in catalog.archive_readiness_specs(deep=True)) == (
        "orphaned_messages",
        "empty_sessions",
        "orphaned_attachments",
        "orphaned_blobs",
        "superseded_raw_snapshots",
    )


def test_maintenance_target_catalog_renders_repair_hints_from_canonical_targets() -> None:
    catalog = build_maintenance_target_catalog()

    assert catalog.repair_hint(("session_insights",), include_run_all=True) == (
        "Run `polylogue ops doctor --repair --target session_insights`, or `polylogued run`."
    )
    assert catalog.resolve_name("dangling_fts") is None
    assert catalog.resolve_name("raw_materialization") is None
