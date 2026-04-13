from __future__ import annotations

from pathlib import Path

from devtools.quality_registry import build_quality_registry

ROOT = Path(__file__).resolve().parents[3]


def test_build_quality_registry_exposes_live_catalogs() -> None:
    registry = build_quality_registry()

    assert any(entry.name == "frontier-local" for entry in registry.composite_lanes)
    assert any(entry.name == "machine-contract" for entry in registry.contract_lanes)
    assert any(entry.name == "live-exercises" for entry in registry.live_lanes)
    assert any(entry.name == "filters" for entry in registry.mutation_campaigns)
    assert any(entry.name == "search-filters" for entry in registry.benchmark_campaigns)
    assert any(entry.name == "action-event-materialization" for entry in registry.synthetic_benchmark_campaigns)
    assert any(entry.name == "session-product-materialization" for entry in registry.synthetic_benchmark_campaigns)
    assert any(entry.name == "startup-health" for entry in registry.synthetic_benchmark_campaigns)
    search_filters = next(entry for entry in registry.benchmark_campaigns if entry.name == "search-filters")
    assert search_filters.origin == "authored.benchmark-domain"
    assert search_filters.artifact_targets == ("conversation_query_results", "message_fts")
    assert search_filters.operation_targets == ("benchmark.query.search-filters",)
    assert search_filters.tags == ("benchmark", "search", "filters")
    startup_health = next(entry for entry in registry.synthetic_benchmark_campaigns if entry.name == "startup-health")
    assert startup_health.origin == "authored.synthetic-benchmark"
    assert startup_health.artifact_targets == ("archive_health",)
    assert startup_health.operation_targets == ("health.startup.synthetic",)
    assert startup_health.tags == ("benchmark", "synthetic", "health")
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
    assert session_products.artifact_targets == (
        "session_product_source_conversations",
        "session_product_rows",
        "session_product_fts",
    )
    assert session_products.operation_targets == ("materialize-session-products",)
    assert session_products.tags == ("benchmark", "synthetic", "session-products")


def test_quality_registry_references_existing_files() -> None:
    registry = build_quality_registry()

    for campaign in registry.mutation_campaigns:
        for path in (*campaign.paths_to_mutate, *campaign.tests):
            assert (ROOT / path).exists(), path

    for campaign in registry.benchmark_campaigns:
        for path in campaign.tests:
            assert (ROOT / path).exists(), path
