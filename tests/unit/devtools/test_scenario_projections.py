from __future__ import annotations

import json

from devtools import scenario_projections
from devtools.scenario_projection_catalog import build_scenario_projection_entries
from polylogue.scenarios import declared_operation_target_names


def test_render_scenario_projections_text_lists_authored_sources() -> None:
    rendered = scenario_projections.render_scenario_projections(as_json=False)

    assert "Scenario Projections (" in rendered
    assert "exercise:json-doctor-action-event-preview" in rendered
    assert "exercise:gen-schema-list" in rendered
    assert "validation-lane:machine-contract" in rendered
    assert "mutation-campaign:filters" in rendered
    assert "benchmark-campaign:search-filters" in rendered
    assert "synthetic-benchmark:action-event-materialization" in rendered


def test_render_scenario_projections_json_is_machine_readable() -> None:
    payload = json.loads(scenario_projections.render_scenario_projections(as_json=True))

    assert any(
        entry["source_kind"] == "exercise" and entry["name"] == "json-doctor-action-event-preview" for entry in payload
    )
    assert any(entry["source_kind"] == "validation-lane" and entry["name"] == "machine-contract" for entry in payload)
    assert any(entry["source_kind"] == "mutation-campaign" and entry["name"] == "filters" for entry in payload)
    assert any(entry["source_kind"] == "exercise" and entry["name"] == "gen-fmt-json-latest" for entry in payload)
    assert any(
        entry["source_kind"] == "synthetic-benchmark" and entry["name"] == "session-product-materialization"
        for entry in payload
    )


def test_render_scenario_projections_supports_targeted_filters() -> None:
    rendered = scenario_projections.render_scenario_projections(
        as_json=False,
        source_kinds=("exercise",),
        path_target="action-event-repair-loop",
        artifact_target="action_event_rows",
        operation_target="project-action-event-health",
        tag="maintenance",
    )

    assert "exercise:json-doctor-action-event-preview" in rendered
    assert "synthetic-benchmark:action-event-materialization" not in rendered
    assert "exercise:json-doctor-session-products-preview" not in rendered
    assert "path targets: action-event-repair-loop" in rendered


def test_all_projection_operation_targets_are_declared() -> None:
    declared = set(declared_operation_target_names())

    for entry in build_scenario_projection_entries():
        assert set(entry.operation_targets).issubset(declared)


def test_validation_lane_projection_entries_include_composite_metadata_unions() -> None:
    frontier_local = next(
        entry
        for entry in build_scenario_projection_entries()
        if entry.source_kind.value == "validation-lane" and entry.name == "frontier-local"
    )

    assert "cli.json-contract" in frontier_local.operation_targets
    assert "cli.help" in frontier_local.operation_targets
    assert "contract" in frontier_local.tags
