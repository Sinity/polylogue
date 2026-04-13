from __future__ import annotations

import json

from devtools import artifact_graph


def test_render_artifact_graph_text_mentions_both_vertical_paths() -> None:
    rendered = artifact_graph.render_artifact_graph(as_json=False)

    assert "Artifact Paths:" in rendered
    assert "Artifact Operations:" in rendered
    assert "Runtime Scenario Coverage:" in rendered
    assert "raw-reparse-loop" in rendered
    assert "action-event-repair-loop" in rendered
    assert "action_event_fts [index] <- action_event_rows" in rendered
    assert "plan-validation-backlog [planning]" in rendered
    assert "project-action-event-health" in rendered
    assert "json-doctor-action-event-preview" in rendered


def test_render_artifact_graph_json_is_machine_readable() -> None:
    payload = json.loads(artifact_graph.render_artifact_graph(as_json=True))

    assert {path["name"] for path in payload["paths"]} == {
        "raw-reparse-loop",
        "action-event-repair-loop",
    }
    assert any(node["name"] == "raw_validation_state" for node in payload["nodes"])
    assert any(operation["name"] == "plan-parse-backlog" for operation in payload["operations"])
    assert any(operation["kind"] == "projection" for operation in payload["operations"])
    assert payload["scenario_coverage"]["artifacts"]["action_event_rows"] == [
        {
            "source": "exercise",
            "name": "json-doctor-action-event-preview",
            "origin": "generated.json-contract",
        }
    ]
    assert payload["scenario_coverage"]["operations"]["project-action-event-health"] == [
        {
            "source": "exercise",
            "name": "json-doctor-action-event-preview",
            "origin": "generated.json-contract",
        }
    ]
