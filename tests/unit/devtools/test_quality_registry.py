from __future__ import annotations

from pathlib import Path

from devtools.quality_registry import build_quality_registry

ROOT = Path(__file__).resolve().parents[3]


def test_build_quality_registry_exposes_live_catalogs() -> None:
    registry = build_quality_registry()

    assert any(entry.name == "schema-list-contract" for entry in registry.contract_lanes)
    assert any(entry.name == "schema-explain-contract" for entry in registry.contract_lanes)
    assert any(entry.name == "frontier-local" for entry in registry.composite_lanes)
    assert any(entry.name == "machine-contract" for entry in registry.contract_lanes)
    assert any(entry.name == "live-exercises" for entry in registry.live_lanes)
    assert any(entry.name == "filters" for entry in registry.mutation_campaigns)
    assert any(entry.name == "search-filters" for entry in registry.benchmark_campaigns)
    assert any(entry.name == "pipeline" for entry in registry.benchmark_campaigns)
    assert any(entry.name == "recovery-digest" for entry in registry.benchmark_campaigns)
    assert any(entry.name == "session-insight-materialization" for entry in registry.synthetic_benchmark_campaigns)
    assert any(entry.name == "startup-readiness" for entry in registry.synthetic_benchmark_campaigns)
    assert any(
        scenario.provider == "chatgpt" and scenario.package_version == "v1"
        for scenario in registry.inferred_corpus_scenarios
    )
    assert any(entry.name == "machine-contract" for entry in registry.scenario_projections)
    assert any(
        entry.name == "filters" and entry.source_kind.value == "mutation-campaign"
        for entry in registry.scenario_projections
    )
    assert any(entry.name == "search-filters" for entry in registry.scenario_projections)
    assert any(entry.name == "session-insight-materialization" for entry in registry.scenario_projections)
    assert any(entry.source_kind.value == "inferred-corpus-scenario" for entry in registry.scenario_projections)
    assert registry.scenario_projections == registry.catalog.compile_projection_entries()
    inferred_chatgpt = next(
        entry
        for entry in registry.scenario_projections
        if entry.source_kind.value == "inferred-corpus-scenario" and entry.name == "chatgpt:v1"
    )
    assert inferred_chatgpt.source_payload["provider"] == "chatgpt"
    assert inferred_chatgpt.source_payload["package_version"] == "v1"
    assert inferred_chatgpt.source_payload["variant_count"] == 1
    assert inferred_chatgpt.path_targets == ("inferred-corpus-compilation-loop",)
    assert inferred_chatgpt.operation_targets == (
        "compile-inferred-corpus-specs",
        "compile-inferred-corpus-scenarios",
    )
    machine_contract = next(entry for entry in registry.contract_lanes if entry.name == "machine-contract")
    assert machine_contract.origin == "authored.validation-lane"
    assert machine_contract.operation_targets == ("cli.json-contract",)
    assert machine_contract.tags == ("contract", "json", "cli")
    live_session_insight_repair = next(
        entry for entry in registry.live_lanes if entry.name == "live-session-insight-repair"
    )
    assert live_session_insight_repair.path_targets == ("session-insight-repair-loop",)
    assert live_session_insight_repair.operation_targets == (
        "cli.json-contract",
        "materialize-session-insights",
        "project-session-insight-readiness",
    )
    search_filters = next(entry for entry in registry.benchmark_campaigns if entry.name == "search-filters")
    assert search_filters.origin == "authored.benchmark-domain"
    assert search_filters.artifact_targets == ("session_query_results", "message_fts")
    assert set(search_filters.operation_targets) == {"query-sessions", "benchmark.query.search-filters"}
    assert search_filters.tags == ("benchmark", "search", "filters")
    pipeline = next(entry for entry in registry.benchmark_campaigns if entry.name == "pipeline")
    assert pipeline.origin == "authored.benchmark-domain"
    assert pipeline.operation_targets == ("benchmark.pipeline.index-and-helpers",)
    assert pipeline.tags == ("benchmark", "pipeline")
    recovery_digest = next(entry for entry in registry.benchmark_campaigns if entry.name == "recovery-digest")
    assert recovery_digest.origin == "authored.benchmark-domain"
    assert recovery_digest.artifact_targets == (
        "recovery_digest",
        "forensic_index",
        "resume_bundle",
        "recovery_report_markdown",
    )
    assert recovery_digest.operation_targets == (
        "compile-recovery-digest",
        "render-recovery-report",
        "benchmark.transform.recovery-digest",
        "benchmark.transform.recovery-report",
    )
    assert recovery_digest.tags == ("benchmark", "transform", "recovery")
    startup_health = next(
        entry for entry in registry.synthetic_benchmark_campaigns if entry.name == "startup-readiness"
    )
    assert startup_health.origin == "authored.synthetic-benchmark"
    assert startup_health.summary_metric == "total_readiness_s"
    assert startup_health.summary_label == "s"
    assert startup_health.artifact_targets == ("message_fts", "archive_readiness")
    assert startup_health.operation_targets == ("project-archive-readiness", "readiness.startup.synthetic")
    assert startup_health.tags == ("benchmark", "synthetic", "readiness")
    retrieval_checks = next(entry for entry in registry.contract_lanes if entry.name == "retrieval-checks")
    assert retrieval_checks.path_targets == ("session-query-loop", "message-fts-readiness-loop")
    assert retrieval_checks.artifact_targets == ("message_fts", "session_query_results", "archive_readiness")
    assert retrieval_checks.operation_targets == ("query-sessions", "project-archive-readiness")
    assert retrieval_checks.tags == ("contract", "retrieval", "readiness")
    session_insights = next(
        entry for entry in registry.synthetic_benchmark_campaigns if entry.name == "session-insight-materialization"
    )
    assert session_insights.origin == "authored.synthetic-benchmark"
    assert session_insights.artifact_targets == (
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
    )
    assert session_insights.operation_targets == ("materialize-session-insights",)
    assert session_insights.tags == ("benchmark", "synthetic", "session-insights")
    daemon_live = next(
        entry for entry in registry.synthetic_benchmark_campaigns if entry.name == "daemon-live-convergence"
    )
    assert daemon_live.origin == "authored.synthetic-benchmark"
    assert daemon_live.summary_metric == "total_wall_s"
    assert daemon_live.artifact_targets == (
        "configured_sources",
        "source_payload_stream",
        "archive_session_rows",
        "message_source_rows",
        "message_fts",
    )
    assert daemon_live.operation_targets == (
        "ingest-archive-runtime",
        "index-message-fts",
        "materialize-session-insights",
    )
    assert daemon_live.tags == ("benchmark", "synthetic", "daemon", "live", "convergence")
    insight_profiles = next(
        entry for entry in registry.catalog.exercise_scenarios if entry.name == "json-insights-profiles"
    )
    assert insight_profiles.path_targets == ("session-profile-query-loop",)
    assert insight_profiles.artifact_targets == (
        "session_profile_rows",
        "session_profile_results",
    )
    assert insight_profiles.operation_targets == ("cli.json-contract", "query-session-profiles")
    insight_threads = next(
        entry for entry in registry.catalog.exercise_scenarios if entry.name == "json-insights-threads"
    )
    assert insight_threads.path_targets == ("thread-query-loop",)
    assert insight_threads.artifact_targets == ("thread_rows", "thread_fts", "thread_results")
    assert insight_threads.operation_targets == ("cli.json-contract", "query-threads")
    assert not any(entry.name == "json-doctor-action-preview" for entry in registry.scenario_projections)
    frontier_local = next(entry for entry in registry.composite_lanes if entry.name == "frontier-local")
    archive_intelligence = next(entry for entry in registry.composite_lanes if entry.name == "archive-intelligence")
    runtime_substrate = next(entry for entry in registry.composite_lanes if entry.name == "runtime-substrate-hardening")
    assert "cli.json-contract" in frontier_local.operation_targets
    assert "cli.help" in frontier_local.operation_targets
    assert "query-sessions" in archive_intelligence.operation_targets
    assert "project-archive-readiness" in archive_intelligence.operation_targets
    assert runtime_substrate.family == "runtime-substrate"
    runtime_projection = next(
        entry for entry in registry.scenario_projections if entry.name == "runtime-substrate-hardening"
    )
    assert runtime_projection.source_payload["family"] == "runtime-substrate"


def test_quality_registry_references_existing_files() -> None:
    registry = build_quality_registry()

    for campaign in registry.mutation_campaigns:
        for path in (*campaign.paths_to_mutate, *campaign.tests):
            assert (ROOT / path).exists(), path

    for benchmark_campaign in registry.benchmark_campaigns:
        for path in benchmark_campaign.tests:
            assert (ROOT / path).exists(), path
