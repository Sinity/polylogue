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
        "archive_session_rows",
        "session_insight_source_sessions",
        "session_profile_rows",
        "session_work_event_rows",
        "session_work_event_fts",
        "session_phase_rows",
        "thread_rows",
        "thread_fts",
        "session_tag_rollup_rows",
        "session_insight_rows",
        "session_insight_fts",
        "session_insight_readiness",
        "session_profile_results",
        "session_work_event_results",
        "session_phase_results",
        "thread_results",
        "session_tag_rollup_results",
        "archive_coverage_results",
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
        "session-insight-repair-loop",
        "raw-session-insight-repair-loop",
        "session-digest-transform-loop",
        "message-fts-readiness-loop",
        "session-query-loop",
        "session-profile-query-loop",
        "session-work-event-query-loop",
        "session-phase-query-loop",
        "thread-query-loop",
        "session-tag-rollup-query-loop",
        "archive-coverage-query-loop",
        "tool-usage-query-loop",
        "session-insight-status-query-loop",
        "archive-debt-query-loop",
        "inferred-corpus-compilation-loop",
        "schema-list-query-loop",
        "schema-explain-query-loop",
        "embedding-materialization-loop",
        "embedding-status-query-loop",
        "retrieval-band-readiness-loop",
        "source-acquisition-loop",
    }


def test_runtime_artifact_paths_reference_only_declared_nodes() -> None:
    nodes = build_runtime_artifact_nodes()
    paths = build_runtime_artifact_paths()
    node_names = {node.name for node in nodes}

    for path in paths:
        assert path.nodes
        assert set(path.nodes).issubset(node_names)
