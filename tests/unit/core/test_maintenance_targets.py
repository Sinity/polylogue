from __future__ import annotations

from polylogue.maintenance.models import MaintenanceCategory
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
    assert catalog.names_for_mode(MaintenanceTargetMode.REPAIR) == SAFE_REPAIR_TARGETS
    assert catalog.names_for_mode(MaintenanceTargetMode.CLEANUP) == CLEANUP_TARGETS


def test_maintenance_target_catalog_resolves_aliases_to_canonical_targets() -> None:
    catalog = build_maintenance_target_catalog()

    spec = catalog.resolve_name("action_events")

    assert spec is not None
    assert spec.name == "action_event_read_model"
    assert spec.category is MaintenanceCategory.DERIVED_REPAIR
    assert spec.doctor_repair_operation == "materialize-action-events"
    assert spec.doctor_readiness_operation == "project-action-event-readiness"


def test_maintenance_target_catalog_reports_preview_and_help_semantics() -> None:
    catalog = build_maintenance_target_catalog()

    assert catalog.preview_target_names() == (
        "session_insights",
        "action_event_read_model",
        "dangling_fts",
    )
    assert catalog.help_text() == (
        "Limit maintenance to named targets such as session_insights, action_event_read_model, "
        "dangling_fts, wal_checkpoint, orphaned_messages, orphaned_content_blocks, "
        "empty_conversations, or orphaned_attachments"
    )


def test_maintenance_target_catalog_exposes_archive_readiness_specs() -> None:
    catalog = build_maintenance_target_catalog()

    assert tuple(spec.name for spec in catalog.archive_readiness_specs(deep=False)) == (
        "orphaned_messages",
        "empty_conversations",
        "orphaned_attachments",
    )
    assert tuple(spec.name for spec in catalog.archive_readiness_specs(deep=True)) == (
        "orphaned_messages",
        "orphaned_content_blocks",
        "empty_conversations",
        "orphaned_attachments",
    )


def test_maintenance_target_catalog_renders_repair_hints_from_canonical_targets() -> None:
    catalog = build_maintenance_target_catalog()

    assert catalog.repair_hint(("session_insights",), include_run_all=True) == (
        "Run `polylogue doctor --repair --target session_insights`, or `polylogue run all`."
    )
    assert catalog.repair_hint(("action_events",), include_run_all=False) == (
        "Run `polylogue doctor --repair --target action_event_read_model`."
    )
