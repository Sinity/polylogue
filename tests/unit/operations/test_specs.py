from __future__ import annotations

from polylogue.operations import (
    OperationKind,
    build_declared_operation_catalog,
    build_runtime_operation_catalog,
)


def test_runtime_operation_catalog_covers_the_current_runtime_paths() -> None:
    specs = build_runtime_operation_catalog().by_name()

    assert set(specs) == {
        "plan-validation-backlog",
        "plan-parse-backlog",
        "ingest-archive-runtime",
        "index-message-fts",
        "materialize-action-events",
        "query-conversations",
        "project-action-event-health",
        "materialize-session-products",
        "project-session-product-health",
        "project-archive-health",
        "query-session-profiles",
        "query-session-work-events",
        "query-session-phases",
        "query-work-threads",
        "query-session-tag-rollups",
        "query-day-session-summaries",
        "query-week-session-summaries",
        "query-provider-analytics",
        "query-session-enrichments",
        "query-session-product-status",
        "query-archive-debt",
    }
    assert specs["plan-validation-backlog"].kind is OperationKind.PLANNING
    assert specs["plan-validation-backlog"].path_targets == ("raw-reparse-loop", "raw-archive-ingest-loop")
    assert specs["ingest-archive-runtime"].kind is OperationKind.MATERIALIZATION
    assert specs["ingest-archive-runtime"].mutates_state is True
    assert specs["ingest-archive-runtime"].produces == ("raw_validation_state", "archive_conversation_rows")
    assert specs["ingest-archive-runtime"].path_targets == ("raw-archive-ingest-loop",)
    assert specs["materialize-action-events"].kind is OperationKind.MATERIALIZATION
    assert specs["materialize-action-events"].mutates_state is True
    assert specs["materialize-action-events"].produces == ("action_event_rows", "action_event_fts")
    assert specs["materialize-action-events"].path_targets == ("action-event-repair-loop",)
    assert specs["materialize-session-products"].kind is OperationKind.MATERIALIZATION
    assert specs["materialize-session-products"].mutates_state is True
    assert "session_product_rows" in specs["materialize-session-products"].produces
    assert "session_product_fts" in specs["materialize-session-products"].produces
    assert "session_profile_rows" in specs["materialize-session-products"].produces
    assert "work_thread_fts" in specs["materialize-session-products"].produces
    assert specs["materialize-session-products"].path_targets == ("session-product-repair-loop",)
    assert specs["project-action-event-health"].previewable is True
    assert specs["project-session-product-health"].previewable is True
    assert specs["query-session-profiles"].path_targets == ("session-profile-query-loop",)
    assert specs["query-session-enrichments"].path_targets == ("session-enrichment-query-loop",)
    assert specs["query-session-work-events"].path_targets == ("session-work-event-query-loop",)
    assert specs["query-session-product-status"].path_targets == ("session-product-status-query-loop",)
    assert specs["query-archive-debt"].path_targets == ("archive-debt-query-loop",)
    assert specs["query-provider-analytics"].path_targets == ("provider-analytics-query-loop",)


def test_runtime_operation_catalog_has_declared_surfaces_and_code_refs() -> None:
    for spec in build_runtime_operation_catalog().specs:
        assert spec.surfaces
        assert spec.code_refs


def test_declared_operation_catalog_contains_runtime_and_control_plane_operations() -> None:
    catalog = build_declared_operation_catalog()

    assert "project-action-event-health" in catalog.names()
    assert "benchmark.storage.crud" in catalog.names()
    assert "cli.json-contract" in catalog.names()


def test_operation_catalog_resolve_filters_unknown_names() -> None:
    catalog = build_declared_operation_catalog()

    assert tuple(spec.name for spec in catalog.resolve(("project-action-event-health", "missing"))) == (
        "project-action-event-health",
    )
