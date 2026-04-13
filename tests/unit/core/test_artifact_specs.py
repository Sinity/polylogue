from __future__ import annotations

from polylogue.artifacts import build_runtime_artifact_nodes, build_runtime_artifact_paths


def test_runtime_artifact_specs_expose_the_curated_vertical_paths() -> None:
    nodes = build_runtime_artifact_nodes()
    paths = build_runtime_artifact_paths()

    assert {node.name for node in nodes} >= {
        "raw_validation_state",
        "validation_backlog",
        "parse_backlog",
        "parse_quarantine",
        "archive_conversation_rows",
        "tool_use_source_blocks",
        "action_event_rows",
        "action_event_fts",
        "action_event_health",
        "session_product_source_conversations",
        "session_profile_rows",
        "session_profile_merged_fts",
        "session_work_event_rows",
        "session_work_event_fts",
        "session_phase_rows",
        "work_thread_rows",
        "work_thread_fts",
        "session_tag_rollup_rows",
        "day_session_summary_rows",
        "session_product_rows",
        "session_product_fts",
        "session_product_health",
        "session_profile_results",
        "session_work_event_results",
        "session_phase_results",
        "work_thread_results",
        "session_tag_rollup_results",
        "day_session_summary_results",
        "week_session_summary_results",
        "provider_analytics_results",
        "conversation_render_projection",
        "rendered_conversation_artifacts",
        "site_conversation_pages",
        "site_publication_manifest",
        "publication_records",
        "schema_packages",
        "schema_cluster_manifests",
        "inferred_corpus_specs",
        "inferred_corpus_scenarios",
        "schema_list_results",
        "schema_explanation_results",
    }
    assert {path.name for path in paths} == {
        "raw-reparse-loop",
        "raw-archive-ingest-loop",
        "action-event-repair-loop",
        "session-product-repair-loop",
        "message-fts-health-loop",
        "conversation-query-loop",
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
        "conversation-render-loop",
        "site-publication-loop",
        "inferred-corpus-compilation-loop",
        "schema-list-query-loop",
        "schema-explain-query-loop",
    }


def test_runtime_artifact_paths_reference_only_declared_nodes() -> None:
    nodes = build_runtime_artifact_nodes()
    paths = build_runtime_artifact_paths()
    node_names = {node.name for node in nodes}

    for path in paths:
        assert path.nodes
        assert set(path.nodes).issubset(node_names)
