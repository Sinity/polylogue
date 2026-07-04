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
    assert "message-fts-readiness-loop" in rendered
    assert "session-query-loop" in rendered
    assert "session-insight-repair-loop" in rendered
    assert "session-profile-query-loop" in rendered
    assert "session-work-event-query-loop" in rendered
    assert "thread-query-loop" in rendered
    assert "archive_session_rows [durable] <- raw_validation_state" in rendered
    assert "message_source_rows [source] <- archive_session_rows" in rendered
    assert "message_fts [index] <- message_source_rows" in rendered
    # session_profile_*_fts tables were removed; merged search now flows
    # through session_work_event_fts.
    assert "session_work_event_fts [index] <- session_work_event_rows" in rendered
    assert "thread_fts [index] <- thread_rows" in rendered
    assert "plan-validation-backlog [planning]" in rendered
    assert "ingest-archive-runtime [materialization]" in rendered
    assert "index-message-fts" in rendered
    assert "query-sessions" in rendered
    assert "query-session-profiles" in rendered
    assert "query-session-work-events" in rendered
    assert "query-threads" in rendered
    assert "query-session-insight-status" in rendered
    assert "query-archive-debt" in rendered
    assert "project-session-insight-readiness" in rendered
    assert "project-archive-readiness" in rendered
    assert "startup-readiness" in rendered
    assert "retrieval-checks" in rendered
    assert "synthetic-benchmark:session-insight-materialization" in rendered
    assert "validation-lane:live-insights-profiles-evidence" in rendered
    assert "validation-lane:live-insights-work-events" in rendered
    assert "validation-lane:live-insights-status" in rendered
    assert "validation-lane:live-insights-debt" in rendered
    assert "maintenance session_insights:" in rendered
    assert (
        "uncovered maintenance targets: empty_sessions, message_type_backfill, orphaned_attachments, orphaned_messages, superseded_raw_snapshots"
        in rendered
    )
    assert "uncovered artifacts: thread_results, tool_usage_results" in rendered


def test_render_artifact_graph_json_is_machine_readable() -> None:
    payload = json.loads(artifact_graph.render_artifact_graph(as_json=True))

    assert {path["name"] for path in payload["paths"]} >= {
        "raw-reparse-loop",
        "raw-archive-ingest-loop",
        "message-fts-readiness-loop",
        "session-query-loop",
        "session-insight-repair-loop",
        "session-profile-query-loop",
        "session-work-event-query-loop",
        "session-phase-query-loop",
        "thread-query-loop",
        "session-tag-rollup-query-loop",
        "archive-coverage-query-loop",
        "session-insight-status-query-loop",
        "archive-debt-query-loop",
    }
    assert any(node["name"] == "raw_validation_state" for node in payload["nodes"])
    assert any(node["name"] == "archive_session_rows" for node in payload["nodes"])
    assert any(node["name"] == "message_fts" for node in payload["nodes"])
    assert {target["name"] for target in payload["maintenance_targets"]} >= {
        "session_insights",
    }
    assert any(operation["name"] == "plan-parse-backlog" for operation in payload["operations"])
    assert any(operation["name"] == "ingest-archive-runtime" for operation in payload["operations"])
    assert any(operation["name"] == "index-message-fts" for operation in payload["operations"])
    assert any(operation["kind"] == "projection" for operation in payload["operations"])
    assert {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["artifacts"]["session_insight_rows"]
    } >= {
        ("synthetic-benchmark", "session-insight-materialization", "authored.synthetic-benchmark"),
        ("validation-lane", "live-session-insight-repair", "authored.validation-lane"),
    }
    assert ("validation-lane", "live-session-insight-repair", "authored.validation-lane") in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["project-session-insight-readiness"]
    }
    assert ("validation-lane", "live-session-insight-repair", "authored.validation-lane") in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["maintenance_targets"]["session_insights"]
    }
    assert {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["artifacts"]["archive_session_rows"]
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
    assert any(
        ref["name"] == "retrieval-checks" for ref in payload["scenario_coverage"]["operations"]["query-sessions"]
    )
    assert any(
        ref["name"] == "startup-readiness" for ref in payload["scenario_coverage"]["artifacts"]["archive_readiness"]
    )
    assert any(
        ref["name"] == "live-readiness-json"
        for ref in payload["scenario_coverage"]["operations"]["project-archive-readiness"]
    )
    assert (
        "synthetic-benchmark",
        "session-insight-materialization",
        "authored.synthetic-benchmark",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["artifacts"]["session_insight_source_sessions"]
    }
    assert (
        "synthetic-benchmark",
        "session-insight-materialization",
        "authored.synthetic-benchmark",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["materialize-session-insights"]
    }
    assert (
        "validation-lane",
        "live-insights-profiles-evidence",
        "authored.validation-lane",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["query-session-profiles"]
    }
    assert (
        "validation-lane",
        "live-insights-work-events",
        "authored.validation-lane",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["query-session-work-events"]
    }
    assert (
        "validation-lane",
        "live-insights-status",
        "authored.validation-lane",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["query-session-insight-status"]
    }
    assert (
        "validation-lane",
        "live-insights-debt",
        "authored.validation-lane",
    ) in {
        (ref["source"], ref["name"], ref["origin"])
        for ref in payload["scenario_coverage"]["operations"]["query-archive-debt"]
    }
    assert payload["scenario_coverage"]["paths"]["session-profile-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["session-work-event-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["session-phase-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["thread-query-loop"]["complete"] is False
    assert payload["scenario_coverage"]["paths"]["session-tag-rollup-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["archive-coverage-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["uncovered_maintenance_targets"] == [
        "empty_sessions",
        "message_type_backfill",
        "orphaned_attachments",
        "orphaned_messages",
        "superseded_raw_snapshots",
    ]
    assert payload["scenario_coverage"]["paths"]["session-insight-status-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["archive-debt-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["message-fts-readiness-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["session-query-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["raw-reparse-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["raw-archive-ingest-loop"]["complete"] is True
    assert payload["scenario_coverage"]["paths"]["session-insight-repair-loop"]["complete"] is True
    assert payload["scenario_coverage"]["uncovered_artifacts"] == ["thread_results", "tool_usage_results"]
    assert payload["scenario_coverage"]["uncovered_operations"] == [
        "mutate-add-tag",
        "mutate-bulk-tag-sessions",
        "mutate-delete-metadata",
        "mutate-delete-session",
        "mutate-remove-tag",
        "mutate-set-metadata",
        "query-threads",
        "query-tool-usage",
    ]
