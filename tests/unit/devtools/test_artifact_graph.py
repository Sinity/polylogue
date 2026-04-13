from __future__ import annotations

import json

from devtools import artifact_graph


def test_render_artifact_graph_text_mentions_the_current_runtime_paths() -> None:
    rendered = artifact_graph.render_artifact_graph(as_json=False)

    assert "Artifact Paths:" in rendered
    assert "Artifact Operations:" in rendered
    assert "Runtime Path Coverage:" in rendered
    assert "Runtime Scenario Coverage:" in rendered
    assert "raw-reparse-loop" in rendered
    assert "message-fts-health-loop" in rendered
    assert "conversation-query-loop" in rendered
    assert "action-event-repair-loop" in rendered
    assert "session-product-repair-loop" in rendered
    assert "message_fts [index] <- message_source_rows" in rendered
    assert "action_event_fts [index] <- action_event_rows" in rendered
    assert "session_product_fts [index] <- session_product_rows" in rendered
    assert "plan-validation-backlog [planning]" in rendered
    assert "index-message-fts" in rendered
    assert "query-conversations" in rendered
    assert "project-action-event-health" in rendered
    assert "project-session-product-health" in rendered
    assert "project-archive-health" in rendered
    assert "json-doctor-action-event-preview" in rendered
    assert "json-doctor-session-products-preview" in rendered
    assert "run-preview-reparse" in rendered
    assert "startup-health" in rendered
    assert "retrieval-checks" in rendered
    assert "synthetic-benchmark:action-event-materialization" in rendered
    assert "synthetic-benchmark:session-product-materialization" in rendered
    assert "uncovered artifacts:" not in rendered
    assert "uncovered operations:" not in rendered


def test_render_artifact_graph_json_is_machine_readable() -> None:
    payload = json.loads(artifact_graph.render_artifact_graph(as_json=True))

    assert {path["name"] for path in payload["paths"]} == {
        "raw-reparse-loop",
        "message-fts-health-loop",
        "conversation-query-loop",
        "action-event-repair-loop",
        "session-product-repair-loop",
    }
    assert any(node["name"] == "raw_validation_state" for node in payload["nodes"])
    assert any(node["name"] == "message_fts" for node in payload["nodes"])
    assert any(operation["name"] == "plan-parse-backlog" for operation in payload["operations"])
    assert any(operation["name"] == "index-message-fts" for operation in payload["operations"])
    assert any(operation["kind"] == "projection" for operation in payload["operations"])
    assert {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["artifacts"]["action_event_rows"]
    } >= {
        ("exercise", "json-doctor-action-event-preview", "generated.json-contract"),
        ("synthetic-benchmark", "action-event-materialization", "authored.synthetic-benchmark"),
    }
    assert (
        "exercise",
        "json-doctor-action-event-preview",
        "generated.json-contract",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["project-action-event-health"]
    }
    assert {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["artifacts"]["session_product_rows"]
    } >= {
        ("exercise", "json-doctor-session-products-preview", "generated.json-contract"),
        ("synthetic-benchmark", "session-product-materialization", "authored.synthetic-benchmark"),
    }
    assert (
        "exercise",
        "json-doctor-session-products-preview",
        "generated.json-contract",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["project-session-product-health"]
    }
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
    assert {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["artifacts"]["message_source_rows"]
    } >= {
        ("synthetic-benchmark", "fts-rebuild", "authored.synthetic-benchmark"),
        ("synthetic-benchmark", "incremental-index", "authored.synthetic-benchmark"),
    }
    assert {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["index-message-fts"]
    } >= {
        ("synthetic-benchmark", "fts-rebuild", "authored.synthetic-benchmark"),
        ("synthetic-benchmark", "incremental-index", "authored.synthetic-benchmark"),
    }
    assert any(
        ref["name"] == "retrieval-checks"
        for ref in payload["scenario_coverage"]["operations"]["query-conversations"]
    )
    assert any(
        ref["name"] == "startup-health"
        for ref in payload["scenario_coverage"]["artifacts"]["archive_health"]
    )
    assert any(
        ref["name"] == "live-health-json"
        for ref in payload["scenario_coverage"]["operations"]["project-archive-health"]
    )
    assert (
        "synthetic-benchmark",
        "action-event-materialization",
        "authored.synthetic-benchmark",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["artifacts"]["tool_use_source_blocks"]
    }
    assert (
        "synthetic-benchmark",
        "action-event-materialization",
        "authored.synthetic-benchmark",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["materialize-action-events"]
    }
    assert (
        "synthetic-benchmark",
        "session-product-materialization",
        "authored.synthetic-benchmark",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["artifacts"]["session_product_source_conversations"]
    }
    assert (
        "synthetic-benchmark",
        "session-product-materialization",
        "authored.synthetic-benchmark",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["materialize-session-products"]
    }
    assert payload["scenario_coverage"]["paths"]["action-event-repair-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["message-fts-health-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["conversation-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["raw-reparse-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["session-product-repair-loop"]["complete"] is True
    assert payload["scenario_coverage"]["uncovered_artifacts"] == []
    assert payload["scenario_coverage"]["uncovered_operations"] == []
