from __future__ import annotations

from polylogue.operations import OperationKind, build_runtime_operation_specs


def test_runtime_operation_specs_cover_the_two_proven_paths() -> None:
    specs = {spec.name: spec for spec in build_runtime_operation_specs()}

    assert set(specs) == {
        "plan-validation-backlog",
        "plan-parse-backlog",
        "materialize-action-events",
        "project-action-event-health",
        "materialize-session-products",
        "project-session-product-health",
    }
    assert specs["plan-validation-backlog"].kind is OperationKind.PLANNING
    assert specs["materialize-action-events"].kind is OperationKind.MATERIALIZATION
    assert specs["materialize-action-events"].mutates_state is True
    assert specs["materialize-action-events"].produces == ("action_event_rows", "action_event_fts")
    assert specs["materialize-session-products"].kind is OperationKind.MATERIALIZATION
    assert specs["materialize-session-products"].mutates_state is True
    assert specs["materialize-session-products"].produces == ("session_product_rows", "session_product_fts")
    assert specs["project-action-event-health"].previewable is True
    assert specs["project-session-product-health"].previewable is True


def test_runtime_operation_specs_have_declared_surfaces_and_code_refs() -> None:
    for spec in build_runtime_operation_specs():
        assert spec.surfaces
        assert spec.code_refs
