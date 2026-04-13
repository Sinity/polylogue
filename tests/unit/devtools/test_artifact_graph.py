from __future__ import annotations

import json

from devtools import artifact_graph


def test_render_artifact_graph_text_mentions_both_vertical_paths() -> None:
    rendered = artifact_graph.render_artifact_graph(as_json=False)

    assert "Artifact Paths:" in rendered
    assert "Artifact Operations:" in rendered
    assert "Runtime Path Coverage:" in rendered
    assert "Runtime Scenario Coverage:" in rendered
    assert "raw-reparse-loop" in rendered
    assert "action-event-repair-loop" in rendered
    assert "session-product-repair-loop" in rendered
    assert "action_event_fts [index] <- action_event_rows" in rendered
    assert "session_product_fts [index] <- session_product_rows" in rendered
    assert "plan-validation-backlog [planning]" in rendered
    assert "project-action-event-health" in rendered
    assert "project-session-product-health" in rendered
    assert "json-doctor-action-event-preview" in rendered
    assert "json-doctor-session-products-preview" in rendered
    assert "run-preview-reparse" in rendered
    assert "synthetic-benchmark:action-event-materialization" in rendered
    assert "synthetic-benchmark:session-product-materialization" in rendered
    assert "uncovered artifacts:" not in rendered
    assert "uncovered operations:" not in rendered


def test_render_artifact_graph_json_is_machine_readable() -> None:
    payload = json.loads(artifact_graph.render_artifact_graph(as_json=True))

    assert {path["name"] for path in payload["paths"]} == {
        "raw-reparse-loop",
        "action-event-repair-loop",
        "session-product-repair-loop",
    }
    assert any(node["name"] == "raw_validation_state" for node in payload["nodes"])
    assert any(operation["name"] == "plan-parse-backlog" for operation in payload["operations"])
    assert any(operation["kind"] == "projection" for operation in payload["operations"])
    assert payload["scenario_coverage"]["artifacts"]["action_event_rows"] == [
        {
            "source": "exercise",
            "name": "json-doctor-action-event-preview",
            "origin": "generated.json-contract",
        },
        {
            "source": "synthetic-benchmark",
            "name": "action-event-materialization",
            "origin": "authored.synthetic-benchmark",
        }
    ]
    assert payload["scenario_coverage"]["operations"]["project-action-event-health"] == [
        {
            "source": "exercise",
            "name": "json-doctor-action-event-preview",
            "origin": "generated.json-contract",
        }
    ]
    assert payload["scenario_coverage"]["artifacts"]["session_product_rows"] == [
        {
            "source": "exercise",
            "name": "json-doctor-session-products-preview",
            "origin": "generated.json-contract",
        },
        {
            "source": "synthetic-benchmark",
            "name": "session-product-materialization",
            "origin": "authored.synthetic-benchmark",
        }
    ]
    assert payload["scenario_coverage"]["operations"]["project-session-product-health"] == [
        {
            "source": "exercise",
            "name": "json-doctor-session-products-preview",
            "origin": "generated.json-contract",
        }
    ]
    assert payload["scenario_coverage"]["artifacts"]["raw_validation_state"] == [
        {
            "source": "exercise",
            "name": "run-preview-reparse",
            "origin": "authored.showcase-catalog",
        }
    ]
    assert payload["scenario_coverage"]["operations"]["plan-validation-backlog"] == [
        {
            "source": "exercise",
            "name": "run-preview-reparse",
            "origin": "authored.showcase-catalog",
        }
    ]
    assert payload["scenario_coverage"]["artifacts"]["tool_use_source_blocks"] == [
        {
            "source": "synthetic-benchmark",
            "name": "action-event-materialization",
            "origin": "authored.synthetic-benchmark",
        }
    ]
    assert payload["scenario_coverage"]["operations"]["materialize-action-events"] == [
        {
            "source": "synthetic-benchmark",
            "name": "action-event-materialization",
            "origin": "authored.synthetic-benchmark",
        }
    ]
    assert payload["scenario_coverage"]["artifacts"]["session_product_source_conversations"] == [
        {
            "source": "synthetic-benchmark",
            "name": "session-product-materialization",
            "origin": "authored.synthetic-benchmark",
        }
    ]
    assert payload["scenario_coverage"]["operations"]["materialize-session-products"] == [
        {
            "source": "synthetic-benchmark",
            "name": "session-product-materialization",
            "origin": "authored.synthetic-benchmark",
        }
    ]
    assert payload["scenario_coverage"]["paths"]["action-event-repair-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["raw-reparse-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["session-product-repair-loop"]["complete"] is True
    assert payload["scenario_coverage"]["uncovered_artifacts"] == []
    assert payload["scenario_coverage"]["uncovered_operations"] == []
