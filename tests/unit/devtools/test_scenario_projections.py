from __future__ import annotations

import json

from devtools import scenario_projections


def test_render_scenario_projections_text_lists_authored_sources() -> None:
    rendered = scenario_projections.render_scenario_projections(as_json=False)

    assert "Scenario Projections (" in rendered
    assert "exercise:json-doctor-action-event-preview" in rendered
    assert "benchmark-campaign:search-filters" in rendered
    assert "synthetic-benchmark:action-event-materialization" in rendered


def test_render_scenario_projections_json_is_machine_readable() -> None:
    payload = json.loads(scenario_projections.render_scenario_projections(as_json=True))

    assert any(
        entry["source_kind"] == "exercise" and entry["name"] == "json-doctor-action-event-preview"
        for entry in payload
    )
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
