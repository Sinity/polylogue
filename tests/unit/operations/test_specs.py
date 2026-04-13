from __future__ import annotations

from polylogue.operations import (
    OperationKind,
    build_declared_operation_catalog,
    build_runtime_operation_catalog,
)


def test_runtime_operation_catalog_covers_the_two_proven_paths() -> None:
    specs = build_runtime_operation_catalog().by_name()

    assert set(specs) == {
        "plan-validation-backlog",
        "plan-parse-backlog",
        "materialize-action-events",
        "project-action-event-health",
        "materialize-session-products",
        "project-session-product-health",
    }
    assert specs["plan-validation-backlog"].kind is OperationKind.PLANNING
    assert specs["plan-validation-backlog"].path_targets == ("raw-reparse-loop",)
    assert specs["materialize-action-events"].kind is OperationKind.MATERIALIZATION
    assert specs["materialize-action-events"].mutates_state is True
    assert specs["materialize-action-events"].produces == ("action_event_rows", "action_event_fts")
    assert specs["materialize-action-events"].path_targets == ("action-event-repair-loop",)
    assert specs["materialize-session-products"].kind is OperationKind.MATERIALIZATION
    assert specs["materialize-session-products"].mutates_state is True
    assert specs["materialize-session-products"].produces == ("session_product_rows", "session_product_fts")
    assert specs["materialize-session-products"].path_targets == ("session-product-repair-loop",)
    assert specs["project-action-event-health"].previewable is True
    assert specs["project-session-product-health"].previewable is True


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
