from __future__ import annotations

from pathlib import Path

from devtools.quality_registry import build_quality_registry
from polylogue.scenarios import ScenarioProjectionSourceKind

ROOT = Path(__file__).resolve().parents[3]


def test_build_quality_registry_exposes_live_catalogs() -> None:
    registry = build_quality_registry()

    assert any(entry.name == "json-doctor-action-event-preview" for entry in registry.catalog.exercise_scenarios)
    assert any(entry.name == "gen-schema-list" for entry in registry.catalog.qa_extra_scenarios)
    assert any(entry.name == "frontier-local" for entry in registry.composite_lanes)
    assert any(entry.name == "machine-contract" for entry in registry.contract_lanes)
    assert any(entry.name == "live-exercises" for entry in registry.live_lanes)
    assert any(entry.name == "runtime-substrate" for entry in registry.validation_families)
    assert any(entry.name == "filters" for entry in registry.mutation_campaigns)
    assert any(entry.name == "search-filters" for entry in registry.benchmark_campaigns)
    assert any(entry.name == "pipeline" for entry in registry.benchmark_campaigns)
    assert any(entry.name == "action-event-materialization" for entry in registry.synthetic_benchmark_campaigns)
    assert any(entry.name == "session-product-materialization" for entry in registry.synthetic_benchmark_campaigns)
    assert any(entry.name == "startup-readiness" for entry in registry.synthetic_benchmark_campaigns)
    assert any(
        scenario.provider == "chatgpt" and scenario.package_version == "v1"
        for scenario in registry.inferred_corpus_scenarios
    )
    assert any(entry.name == "json-doctor-action-event-preview" for entry in registry.scenario_projections)
    assert any(
        entry.name == "runtime-substrate" and entry.source_kind.value == "validation-family"
        for entry in registry.scenario_projections
    )
    assert any(entry.name == "machine-contract" for entry in registry.scenario_projections)
    assert any(
        entry.name == "filters" and entry.source_kind.value == "mutation-campaign"
        for entry in registry.scenario_projections
    )
    assert any(entry.name == "search-filters" for entry in registry.scenario_projections)
    assert any(entry.name == "session-product-materialization" for entry in registry.scenario_projections)
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
    live_session_product_repair = next(
        entry for entry in registry.live_lanes if entry.name == "live-session-product-repair"
    )
    assert live_session_product_repair.path_targets == ("session-product-repair-loop",)
    assert live_session_product_repair.operation_targets == (
        "cli.json-contract",
        "materialize-session-products",
        "project-session-product-readiness",
    )
    search_filters = next(entry for entry in registry.benchmark_campaigns if entry.name == "search-filters")
    assert search_filters.origin == "authored.benchmark-domain"
    assert search_filters.artifact_targets == ("conversation_query_results", "message_fts")
    assert set(search_filters.operation_targets) == {"query-conversations", "benchmark.query.search-filters"}
    assert search_filters.tags == ("benchmark", "search", "filters")
    pipeline = next(entry for entry in registry.benchmark_campaigns if entry.name == "pipeline")
    assert pipeline.origin == "authored.benchmark-domain"
    assert pipeline.operation_targets == (
        "benchmark.pipeline.index-and-helpers",
        "benchmark.repair.action-events",
    )
    assert pipeline.tags == ("benchmark", "pipeline")
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
    assert retrieval_checks.path_targets == ("conversation-query-loop", "message-fts-readiness-loop")
    assert retrieval_checks.artifact_targets == ("message_fts", "conversation_query_results", "archive_readiness")
    assert retrieval_checks.operation_targets == ("query-conversations", "project-archive-readiness")
    assert retrieval_checks.tags == ("contract", "retrieval", "readiness")
    action_events = next(
        entry for entry in registry.synthetic_benchmark_campaigns if entry.name == "action-event-materialization"
    )
    assert action_events.origin == "authored.synthetic-benchmark"
    assert action_events.artifact_targets == ("tool_use_source_blocks", "action_event_rows", "action_event_fts")
    assert action_events.operation_targets == ("materialize-action-events",)
    assert action_events.tags == ("benchmark", "synthetic", "action-events")
    session_products = next(
        entry for entry in registry.synthetic_benchmark_campaigns if entry.name == "session-product-materialization"
    )
    assert session_products.origin == "authored.synthetic-benchmark"
    assert "session_product_source_conversations" in session_products.artifact_targets
    assert "session_profile_rows" in session_products.artifact_targets
    assert "session_profile_merged_fts" in session_products.artifact_targets
    assert "session_work_event_rows" in session_products.artifact_targets
    assert "work_thread_fts" in session_products.artifact_targets
    assert "session_product_rows" in session_products.artifact_targets
    assert "session_product_fts" in session_products.artifact_targets
    assert session_products.operation_targets == ("materialize-session-products",)
    assert session_products.tags == ("benchmark", "synthetic", "session-products")
    product_profiles = next(
        entry for entry in registry.catalog.exercise_scenarios if entry.name == "json-products-profiles"
    )
    assert product_profiles.path_targets == ("session-profile-query-loop",)
    assert product_profiles.artifact_targets == (
        "session_profile_rows",
        "session_profile_merged_fts",
        "session_profile_results",
    )
    assert product_profiles.operation_targets == ("cli.json-contract", "query-session-profiles")
    product_threads = next(
        entry for entry in registry.catalog.exercise_scenarios if entry.name == "json-products-threads"
    )
    assert product_threads.path_targets == ("work-thread-query-loop",)
    assert product_threads.artifact_targets == ("work_thread_rows", "work_thread_fts", "work_thread_results")
    assert product_threads.operation_targets == ("cli.json-contract", "query-work-threads")
    action_event_preview = next(
        entry for entry in registry.scenario_projections if entry.name == "json-doctor-action-event-preview"
    )
    assert action_event_preview.source_kind is ScenarioProjectionSourceKind.EXERCISE
    assert action_event_preview.path_targets == ("action-event-repair-loop",)
    assert action_event_preview.artifact_targets == (
        "action_event_rows",
        "action_event_fts",
        "action_event_readiness",
    )
    assert "cli.json-contract" in action_event_preview.operation_targets
    assert "project-action-event-readiness" in action_event_preview.operation_targets
    assert action_event_preview.tags == ("generated", "json-contract", "maintenance", "action-events")
    frontier_local = next(entry for entry in registry.composite_lanes if entry.name == "frontier-local")
    archive_intelligence = next(entry for entry in registry.composite_lanes if entry.name == "archive-intelligence")
    runtime_substrate = next(entry for entry in registry.composite_lanes if entry.name == "runtime-substrate-hardening")
    assert "cli.json-contract" in frontier_local.operation_targets
    assert "cli.help" in frontier_local.operation_targets
    assert "query-conversations" in archive_intelligence.operation_targets
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
