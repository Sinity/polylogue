from __future__ import annotations

import json

from devtools import scenario_projections
from devtools.scenario_projection_catalog import build_scenario_projection_entries
from polylogue.scenarios import (
    ScenarioProjectionSource,
    ScenarioProjectionSourceKind,
    declared_operation_target_names,
)


class _PresentationScenario(ScenarioProjectionSource):
    docs_role = "tour"
    caption = "Query recall demo"
    narrative_order = 1
    audience = ("operator",)
    demonstrates = ("recall", "json-output")
    privacy_level = "synthetic"
    media = ("terminal",)
    visual_style = "plain"
    origin = "test"
    path_targets = ()
    artifact_targets = ()
    operation_targets = ()
    tags = ()

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.EXERCISE

    @property
    def projection_name(self) -> str:
        return "presentation"

    @property
    def projection_description(self) -> str:
        return "Presentation metadata probe"


def test_render_scenario_projections_text_lists_authored_sources() -> None:
    rendered = scenario_projections.render_scenario_projections(as_json=False)

    assert "Scenario Projections (" in rendered
    assert "exercise:json-doctor-action-event-preview" in rendered
    assert "exercise:gen-schema-list" in rendered
    assert "validation-lane:machine-contract" in rendered
    assert "mutation-campaign:filters" in rendered
    assert "benchmark-campaign:search-filters" in rendered
    assert "synthetic-benchmark:action-event-materialization" in rendered
    assert "inferred-corpus-scenario:chatgpt:v1" in rendered


def test_render_scenario_projections_json_is_machine_readable() -> None:
    payload = json.loads(scenario_projections.render_scenario_projections(as_json=True))

    assert any(
        entry["source_kind"] == "exercise" and entry["name"] == "json-doctor-action-event-preview" for entry in payload
    )
    assert any(entry["source_kind"] == "validation-lane" and entry["name"] == "machine-contract" for entry in payload)
    assert any(entry["source_kind"] == "mutation-campaign" and entry["name"] == "filters" for entry in payload)
    assert any(entry["source_kind"] == "exercise" and entry["name"] == "gen-fmt-json-latest" for entry in payload)
    assert any(
        entry["source_kind"] == "synthetic-benchmark" and entry["name"] == "session-insight-materialization"
        for entry in payload
    )
    assert any(
        entry["source_kind"] == "inferred-corpus-scenario" and entry["name"] == "chatgpt:v1" for entry in payload
    )
    inferred = next(
        entry
        for entry in payload
        if entry["source_kind"] == "inferred-corpus-scenario" and entry["name"] == "chatgpt:v1"
    )
    assert inferred["source_payload"]["provider"] == "chatgpt"
    assert inferred["source_payload"]["package_version"] == "v1"


def test_render_scenario_projections_supports_targeted_filters() -> None:
    rendered = scenario_projections.render_scenario_projections(
        as_json=False,
        source_kinds=("exercise",),
        path_target="action-event-repair-loop",
        artifact_target="action_event_rows",
        operation_target="project-action-event-readiness",
        tag="maintenance",
    )

    assert "exercise:json-doctor-action-event-preview" in rendered
    assert "synthetic-benchmark:action-event-materialization" not in rendered
    assert "exercise:json-doctor-session-insights-preview" not in rendered
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


def test_projection_entries_preserve_presentation_metadata() -> None:
    entry = _PresentationScenario().to_projection_entry()

    assert entry.to_payload()["docs_role"] == "tour"
    assert entry.to_payload()["caption"] == "Query recall demo"
    assert entry.to_payload()["demonstrates"] == ["recall", "json-output"]
    assert entry.to_dict()["narrative_order"] == 1
