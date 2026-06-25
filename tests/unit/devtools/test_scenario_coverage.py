from __future__ import annotations

from devtools.scenario_coverage import build_runtime_scenario_coverage


def test_build_runtime_scenario_coverage_tracks_the_current_authored_map() -> None:
    coverage = build_runtime_scenario_coverage()

    assert set(coverage.paths) == {
        "source-acquisition-loop",
        "raw-reparse-loop",
        "raw-archive-ingest-loop",
        "message-fts-readiness-loop",
        "embedding-materialization-loop",
        "retrieval-band-readiness-loop",
        "embedding-status-query-loop",
        "session-query-loop",
        "session-insight-repair-loop",
        "raw-session-insight-repair-loop",
        "recovery-digest-transform-loop",
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
    }
    incomplete_paths = {name: path for name, path in coverage.paths.items() if not path.complete}
    assert set(incomplete_paths) == {"thread-query-loop", "tool-usage-query-loop"}
    assert "raw_validation_state" in coverage.artifacts
    assert "configured_sources" in coverage.artifacts
    assert "source_payload_stream" in coverage.artifacts
    assert "artifact_observation_rows" in coverage.artifacts
    assert "archive_session_rows" in coverage.artifacts
    assert "message_source_rows" in coverage.artifacts
    assert "message_fts" in coverage.artifacts
    assert "embedding_metadata_rows" in coverage.artifacts
    assert "embedding_status_rows" in coverage.artifacts
    assert "message_embedding_vectors" in coverage.artifacts
    assert "retrieval_band_readiness" in coverage.artifacts
    assert "embedding_status_results" in coverage.artifacts
    assert "session_query_results" in coverage.artifacts
    assert "archive_readiness" in coverage.artifacts
    assert "session_insight_source_sessions" in coverage.artifacts
    assert "inferred_corpus_specs" in coverage.artifacts
    assert "inferred_corpus_scenarios" in coverage.artifacts
    assert "schema_list_results" in coverage.artifacts
    assert "schema_explanation_results" in coverage.artifacts
    assert "acquire-raw-sessions" in coverage.operations
    assert "plan-validation-backlog" in coverage.operations
    assert "ingest-archive-runtime" in coverage.operations
    assert "index-message-fts" in coverage.operations
    assert "materialize-transcript-embeddings" in coverage.operations
    assert "project-retrieval-band-readiness" in coverage.operations
    assert "query-embedding-status" in coverage.operations
    assert "query-sessions" in coverage.operations
    assert "materialize-session-insights" in coverage.operations
    assert "project-archive-readiness" in coverage.operations
    assert "session_insights" in coverage.maintenance_targets
    assert "dangling_fts" in coverage.maintenance_targets
    assert "compile-inferred-corpus-specs" in coverage.operations
    assert "compile-inferred-corpus-scenarios" in coverage.operations
    assert "query-schema-catalog" in coverage.operations
    assert "query-schema-explanations" in coverage.operations
    assert "cli.help" in coverage.declared_operations
    assert "benchmark.query.search-filters" in coverage.declared_operations
    assert coverage.uncovered_artifacts == ("thread_results", "tool_usage_results")
    assert coverage.uncovered_operations == (
        "mutate-add-tag",
        "mutate-bulk-tag-sessions",
        "mutate-delete-metadata",
        "mutate-delete-session",
        "mutate-remove-tag",
        "mutate-set-metadata",
        "query-threads",
        "query-tool-usage",
    )
    assert coverage.uncovered_maintenance_targets == (
        "empty_sessions",
        "message_embeddings",
        "message_type_backfill",
        "orphaned_attachments",
        "orphaned_messages",
        "raw_materialization",
        "superseded_raw_snapshots",
        "wal_checkpoint",
    )
    assert coverage.uncovered_declared_operations == coverage.uncovered_operations
