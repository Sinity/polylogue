from __future__ import annotations

from devtools.scenario_coverage import build_runtime_scenario_coverage


def test_build_runtime_scenario_coverage_tracks_the_current_authored_map() -> None:
    coverage = build_runtime_scenario_coverage()

    assert set(coverage.paths) == {
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
        "inferred-corpus-compilation-loop",
        "schema-list-query-loop",
        "schema-explain-query-loop",
    }
    assert all(path.complete for path in coverage.paths.values())
    assert "raw_validation_state" in coverage.artifacts
    assert "archive_conversation_rows" in coverage.artifacts
    assert "message_source_rows" in coverage.artifacts
    assert "message_fts" in coverage.artifacts
    assert "tool_use_source_blocks" in coverage.artifacts
    assert "conversation_query_results" in coverage.artifacts
    assert "archive_health" in coverage.artifacts
    assert "session_product_source_conversations" in coverage.artifacts
    assert "inferred_corpus_specs" in coverage.artifacts
    assert "inferred_corpus_scenarios" in coverage.artifacts
    assert "schema_list_results" in coverage.artifacts
    assert "schema_explanation_results" in coverage.artifacts
    assert "plan-validation-backlog" in coverage.operations
    assert "ingest-archive-runtime" in coverage.operations
    assert "index-message-fts" in coverage.operations
    assert "materialize-action-events" in coverage.operations
    assert "query-conversations" in coverage.operations
    assert "materialize-session-products" in coverage.operations
    assert "project-archive-health" in coverage.operations
    assert "compile-inferred-corpus-specs" in coverage.operations
    assert "compile-inferred-corpus-scenarios" in coverage.operations
    assert "query-schema-catalog" in coverage.operations
    assert "query-schema-explanations" in coverage.operations
    assert "cli.help" in coverage.declared_operations
    assert "benchmark.query.search-filters" in coverage.declared_operations
    assert coverage.uncovered_artifacts == ()
    assert coverage.uncovered_operations == ()
    assert coverage.uncovered_declared_operations == ()
