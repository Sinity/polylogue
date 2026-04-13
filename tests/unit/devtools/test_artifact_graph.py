from __future__ import annotations

import json

from devtools import artifact_graph


def test_render_artifact_graph_text_mentions_the_current_runtime_paths() -> None:
    rendered = artifact_graph.render_artifact_graph(as_json=False)

    assert "Artifact Paths:" in rendered
    assert "Artifact Operations:" in rendered
    assert "Maintenance Targets:" in rendered
    assert "Runtime Path Coverage:" in rendered
    assert "Runtime Scenario Coverage:" in rendered
    assert "raw-reparse-loop" in rendered
    assert "raw-archive-ingest-loop" in rendered
    assert "message-fts-health-loop" in rendered
    assert "conversation-query-loop" in rendered
    assert "action-event-repair-loop" in rendered
    assert "session-product-repair-loop" in rendered
    assert "session-profile-query-loop" in rendered
    assert "session-work-event-query-loop" in rendered
    assert "work-thread-query-loop" in rendered
    assert "archive_conversation_rows [durable] <- raw_validation_state" in rendered
    assert "message_source_rows [source] <- archive_conversation_rows" in rendered
    assert "message_fts [index] <- message_source_rows" in rendered
    assert "action_event_fts [index] <- action_event_rows" in rendered
    assert "session_profile_merged_fts [index] <- session_profile_rows" in rendered
    assert "session_work_event_fts [index] <- session_work_event_rows" in rendered
    assert "work_thread_fts [index] <- work_thread_rows" in rendered
    assert "plan-validation-backlog [planning]" in rendered
    assert "ingest-archive-runtime [materialization]" in rendered
    assert "index-message-fts" in rendered
    assert "query-conversations" in rendered
    assert "query-session-profiles" in rendered
    assert "query-session-enrichments" in rendered
    assert "query-session-work-events" in rendered
    assert "query-work-threads" in rendered
    assert "query-session-product-status" in rendered
    assert "query-archive-debt" in rendered
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
    assert "exercise:json-products-profiles" in rendered
    assert "exercise:json-products-work-events" in rendered
    assert "validation-lane:live-products-status" in rendered
    assert "validation-lane:live-products-debt" in rendered
    assert "maintenance action_event_read_model:" in rendered
    assert "maintenance dangling_fts:" in rendered
    assert "maintenance session_products:" in rendered
    assert "uncovered maintenance targets: empty_conversations, orphaned_attachments, orphaned_content_blocks, orphaned_messages, wal_checkpoint" in rendered
    assert "uncovered artifacts:" not in rendered
    assert "uncovered operations:" not in rendered


def test_render_artifact_graph_json_is_machine_readable() -> None:
    payload = json.loads(artifact_graph.render_artifact_graph(as_json=True))

    assert {path["name"] for path in payload["paths"]} >= {
        "raw-reparse-loop",
        "raw-archive-ingest-loop",
        "message-fts-health-loop",
        "conversation-query-loop",
        "action-event-repair-loop",
        "session-product-repair-loop",
        "session-profile-query-loop",
        "session-enrichment-query-loop",
        "session-work-event-query-loop",
        "session-phase-query-loop",
        "work-thread-query-loop",
        "session-tag-rollup-query-loop",
        "day-summary-query-loop",
        "week-summary-query-loop",
        "provider-analytics-query-loop",
        "session-product-status-query-loop",
        "archive-debt-query-loop",
    }
    assert any(node["name"] == "raw_validation_state" for node in payload["nodes"])
    assert any(node["name"] == "archive_conversation_rows" for node in payload["nodes"])
    assert any(node["name"] == "message_fts" for node in payload["nodes"])
    assert {
        target["name"] for target in payload["maintenance_targets"]
    } >= {"session_products", "action_event_read_model", "dangling_fts"}
    assert any(operation["name"] == "plan-parse-backlog" for operation in payload["operations"])
    assert any(operation["name"] == "ingest-archive-runtime" for operation in payload["operations"])
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
    assert (
        "exercise",
        "json-doctor-action-event-preview",
        "generated.json-contract",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["maintenance_targets"]["action_event_read_model"]
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
    assert (
        "exercise",
        "json-doctor-session-products-preview",
        "generated.json-contract",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["maintenance_targets"]["session_products"]
    }
    assert (
        "exercise",
        "run-preview-reparse",
        "authored.showcase-catalog",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["artifacts"]["raw_validation_state"]
    }
    assert (
        "exercise",
        "run-preview-reparse",
        "authored.showcase-catalog",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["plan-validation-backlog"]
    }
    assert {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["artifacts"]["archive_conversation_rows"]
    } >= {
        ("validation-lane", "pipeline-probe-chatgpt", "authored.validation-lane"),
        ("validation-lane", "live-archive-subset-parse-probe", "authored.validation-lane"),
    }
    assert {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["ingest-archive-runtime"]
    } >= {
        ("validation-lane", "pipeline-probe-chatgpt", "authored.validation-lane"),
        ("validation-lane", "live-archive-subset-parse-probe", "authored.validation-lane"),
    }
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
    assert {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["maintenance_targets"]["dangling_fts"]
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
    assert (
        "exercise",
        "json-products-profiles",
        "generated.json-contract",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["query-session-profiles"]
    }
    assert (
        "validation-lane",
        "live-products-enrichments",
        "authored.validation-lane",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["query-session-enrichments"]
    }
    assert (
        "exercise",
        "json-products-work-events",
        "generated.json-contract",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["query-session-work-events"]
    }
    assert (
        "exercise",
        "json-products-threads",
        "generated.json-contract",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["query-work-threads"]
    }
    assert (
        "validation-lane",
        "live-products-status",
        "authored.validation-lane",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["query-session-product-status"]
    }
    assert (
        "validation-lane",
        "live-products-debt",
        "authored.validation-lane",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["query-archive-debt"]
    }
    assert payload["scenario_coverage"]["paths"]["session-profile-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["session-enrichment-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["session-work-event-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["session-phase-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["work-thread-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["session-tag-rollup-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["day-summary-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["uncovered_maintenance_targets"] == [
        "empty_conversations",
        "orphaned_attachments",
        "orphaned_content_blocks",
        "orphaned_messages",
        "wal_checkpoint",
    ]
    assert payload["scenario_coverage"]["paths"]["week-summary-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["provider-analytics-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["session-product-status-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["archive-debt-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["action-event-repair-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["message-fts-health-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["conversation-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["raw-reparse-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["raw-archive-ingest-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["session-product-repair-loop"]["complete"] is True
    assert payload["scenario_coverage"]["uncovered_artifacts"] == []
    assert payload["scenario_coverage"]["uncovered_operations"] == []
