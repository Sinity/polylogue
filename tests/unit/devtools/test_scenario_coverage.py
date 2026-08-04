from __future__ import annotations

from dataclasses import replace

import pytest

from devtools import artifact_graph
from devtools.scenario_coverage import build_runtime_scenario_coverage
from devtools.scenario_projection_catalog import build_scenario_projection_entries


def test_runtime_scenario_coverage_is_closed_over_production_declarations() -> None:
    coverage = build_runtime_scenario_coverage()

    assert not coverage.uncovered_artifacts
    assert not coverage.uncovered_operations
    assert not coverage.uncovered_maintenance_targets
    assert not coverage.uncovered_declared_operations
    assert all(path.complete for path in coverage.paths.values())
    assert {ref.name for ref in coverage.operations["mutate-add-tag"]} >= {"mutation-routes"}
    assert {ref.name for ref in coverage.operations["query-threads"]} >= {"insight-query-routes"}
    assert {ref.name for ref in coverage.operations["query-tool-usage"]} >= {"insight-query-routes"}
    assert {ref.name for ref in coverage.maintenance_targets["empty_sessions"]} >= {"maintenance-target-routes"}
    assert {ref.name for ref in coverage.maintenance_targets["message_type_backfill"]} >= {"maintenance-target-routes"}
    assert {ref.name for ref in coverage.maintenance_targets["superseded_raw_snapshots"]} >= {
        "maintenance-target-routes"
    }


def test_strict_artifact_graph_is_a_closed_publish_contract() -> None:
    assert artifact_graph.main(["--strict"]) == 0


def test_runtime_coverage_opens_when_a_declared_route_scenario_is_removed() -> None:
    projections = build_scenario_projection_entries()
    coverage = build_runtime_scenario_coverage(
        projections=tuple(projection for projection in projections if projection.name != "mutation-routes")
    )

    assert "mutate-add-tag" in coverage.uncovered_operations
    assert not coverage.paths["tag-mutation-loop"].complete


def test_runtime_coverage_rejects_a_missing_declared_runtime_target() -> None:
    projections = build_scenario_projection_entries()
    mutation_routes = next(projection for projection in projections if projection.name == "mutation-routes")
    broken_routes = replace(mutation_routes, path_targets=("missing-mutation-loop",))

    with pytest.raises(KeyError, match="missing-mutation-loop"):
        build_runtime_scenario_coverage(
            projections=tuple(
                broken_routes if projection.name == "mutation-routes" else projection for projection in projections
            )
        )


def test_runtime_coverage_opens_when_a_declared_route_path_is_removed() -> None:
    projections = build_scenario_projection_entries()
    mutation_routes = next(projection for projection in projections if projection.name == "mutation-routes")
    paths_removed = replace(mutation_routes, path_targets=())
    coverage = build_runtime_scenario_coverage(
        projections=tuple(
            paths_removed if projection.name == "mutation-routes" else projection for projection in projections
        )
    )

    assert coverage.paths["tag-mutation-loop"].missing_route_declaration
    assert not coverage.paths["tag-mutation-loop"].complete


def test_runtime_coverage_requires_each_operation_to_share_its_declared_path_route() -> None:
    projections = build_scenario_projection_entries()
    mutation_routes = next(projection for projection in projections if projection.name == "mutation-routes")
    operation_removed = replace(
        mutation_routes,
        operation_targets=tuple(
            operation for operation in mutation_routes.operation_targets if operation != "mutate-add-tag"
        ),
    )
    coverage = build_runtime_scenario_coverage(
        projections=tuple(
            operation_removed if projection.name == "mutation-routes" else projection for projection in projections
        )
    )

    assert coverage.paths["tag-mutation-loop"].uncovered_route_operations == ("mutate-add-tag",)
    assert not coverage.paths["tag-mutation-loop"].complete
