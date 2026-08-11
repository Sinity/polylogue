from __future__ import annotations

import json

from devtools import artifact_graph


def test_render_artifact_graph_text_mentions_the_current_runtime_paths() -> None:
    rendered = artifact_graph.render_artifact_graph(as_json=False)

    assert "Artifact Paths:" in rendered
    assert "Artifact Operations:" in rendered
    assert "Maintenance Targets:" in rendered
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
    # Validation lanes and benchmark campaigns have their own executable
    # catalogs. The artifact graph reports product data paths and operations,
    # not a second copy of those control-plane registries.


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
        "tag-mutation-loop",
        "metadata-mutation-loop",
        "session-excision-loop",
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
    assert "scenario_coverage" not in payload
