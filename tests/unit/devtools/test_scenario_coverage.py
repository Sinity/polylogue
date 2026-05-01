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
        "conversation-query-loop",
        "action-event-repair-loop",
        "session-product-repair-loop",
        "raw-session-product-repair-loop",
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
    assert all(path.complete for path in coverage.paths.values())
    assert "raw_validation_state" in coverage.artifacts
    assert "configured_sources" in coverage.artifacts
    assert "source_payload_stream" in coverage.artifacts
    assert "artifact_observation_rows" in coverage.artifacts
    assert "archive_conversation_rows" in coverage.artifacts
    assert "message_source_rows" in coverage.artifacts
    assert "message_fts" in coverage.artifacts
    assert "embedding_metadata_rows" in coverage.artifacts
    assert "embedding_status_rows" in coverage.artifacts
    assert "message_embedding_vectors" in coverage.artifacts
    assert "tool_use_source_blocks" in coverage.artifacts
    assert "retrieval_band_readiness" in coverage.artifacts
    assert "embedding_status_results" in coverage.artifacts
    assert "conversation_query_results" in coverage.artifacts
    assert "archive_readiness" in coverage.artifacts
    assert "conversation_render_projection" in coverage.artifacts
    assert "rendered_conversation_artifacts" in coverage.artifacts
    assert "site_conversation_pages" in coverage.artifacts
    assert "site_publication_manifest" in coverage.artifacts
    assert "publication_records" in coverage.artifacts
    assert "session_product_source_conversations" in coverage.artifacts
    assert "inferred_corpus_specs" in coverage.artifacts
    assert "inferred_corpus_scenarios" in coverage.artifacts
    assert "schema_list_results" in coverage.artifacts
    assert "schema_explanation_results" in coverage.artifacts
    assert "acquire-raw-conversations" in coverage.operations
    assert "plan-validation-backlog" in coverage.operations
    assert "ingest-archive-runtime" in coverage.operations
    assert "index-message-fts" in coverage.operations
    assert "materialize-transcript-embeddings" in coverage.operations
    assert "materialize-action-events" in coverage.operations
    assert "project-retrieval-band-readiness" in coverage.operations
    assert "query-embedding-status" in coverage.operations
    assert "query-conversations" in coverage.operations
    assert "render-conversations" in coverage.operations
    assert "publish-site" in coverage.operations
    assert "materialize-session-insights" in coverage.operations
    assert "project-archive-readiness" in coverage.operations
    assert "session_products" in coverage.maintenance_targets
    assert "action_event_read_model" in coverage.maintenance_targets
    assert "dangling_fts" in coverage.maintenance_targets
    assert "compile-inferred-corpus-specs" in coverage.operations
    assert "compile-inferred-corpus-scenarios" in coverage.operations
    assert "query-schema-catalog" in coverage.operations
    assert "query-schema-explanations" in coverage.operations
    assert "cli.help" in coverage.declared_operations
    assert "benchmark.query.search-filters" in coverage.declared_operations
    assert coverage.uncovered_artifacts == ()
    assert coverage.uncovered_operations == ()
    assert coverage.uncovered_maintenance_targets == (
        "empty_conversations",
        "orphaned_attachments",
        "orphaned_content_blocks",
        "orphaned_messages",
        "wal_checkpoint",
    )
    assert coverage.uncovered_declared_operations == ()
