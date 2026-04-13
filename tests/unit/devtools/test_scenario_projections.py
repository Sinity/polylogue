from __future__ import annotations

import json

from devtools import scenario_projections


def test_render_scenario_projections_text_lists_authored_sources() -> None:
    rendered = scenario_projections.render_scenario_projections(as_json=False)

    assert "Scenario Projections:" in rendered
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
