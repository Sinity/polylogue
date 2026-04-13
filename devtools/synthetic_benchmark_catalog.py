"""Authored synthetic benchmark scenarios shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.scenarios import ScenarioMetadata


@dataclass(frozen=True)
class SyntheticBenchmarkScenario(ScenarioMetadata):
    """Authored scenario metadata for synthetic long-haul benchmark campaigns."""

    scenario_id: str
    description: str
    summary_metric: str
    summary_label: str


SYNTHETIC_BENCHMARK_SCENARIOS: tuple[SyntheticBenchmarkScenario, ...] = (
    SyntheticBenchmarkScenario(
        scenario_id="fts-rebuild",
        description="Benchmark full FTS5 index rebuild",
        summary_metric="rebuild_wall_s",
        summary_label="s",
        origin="authored.synthetic-benchmark",
        artifact_targets=("message_fts",),
        operation_targets=("index.message-fts-rebuild",),
        tags=("benchmark", "synthetic", "fts"),
    ),
    SyntheticBenchmarkScenario(
        scenario_id="incremental-index",
        description="Benchmark incremental FTS index updates",
        summary_metric="total_wall_s",
        summary_label="s",
        origin="authored.synthetic-benchmark",
        artifact_targets=("message_fts",),
        operation_targets=("index.message-fts-incremental",),
        tags=("benchmark", "synthetic", "fts"),
    ),
    SyntheticBenchmarkScenario(
        scenario_id="filter-scan",
        description="Benchmark common filter query patterns",
        summary_metric="list_50_wall_s",
        summary_label="s",
        origin="authored.synthetic-benchmark",
        artifact_targets=("conversation_query_results",),
        operation_targets=("query.filters.synthetic-scan",),
        tags=("benchmark", "synthetic", "filters"),
    ),
    SyntheticBenchmarkScenario(
        scenario_id="startup-health",
        description="Benchmark check --runtime startup speed",
        summary_metric="total_health_s",
        summary_label="s",
        origin="authored.synthetic-benchmark",
        artifact_targets=("archive_health",),
        operation_targets=("health.startup.synthetic",),
        tags=("benchmark", "synthetic", "health"),
    ),
    SyntheticBenchmarkScenario(
        scenario_id="action-event-materialization",
        description="Benchmark action-event read-model rebuild over synthetic tool-use transcripts",
        summary_metric="rebuild_wall_s",
        summary_label="s",
        origin="authored.synthetic-benchmark",
        artifact_targets=("tool_use_source_blocks", "action_event_rows", "action_event_fts"),
        operation_targets=("materialize-action-events",),
        tags=("benchmark", "synthetic", "action-events"),
    ),
    SyntheticBenchmarkScenario(
        scenario_id="session-product-materialization",
        description="Benchmark durable session-product rebuild over synthetic archive conversations",
        summary_metric="rebuild_wall_s",
        summary_label="s",
        origin="authored.synthetic-benchmark",
        artifact_targets=("session_product_source_conversations", "session_product_rows", "session_product_fts"),
        operation_targets=("materialize-session-products",),
        tags=("benchmark", "synthetic", "session-products"),
    ),
)

SYNTHETIC_BENCHMARK_REGISTRY: dict[str, SyntheticBenchmarkScenario] = {
    scenario.scenario_id: scenario for scenario in SYNTHETIC_BENCHMARK_SCENARIOS
}


__all__ = [
    "SYNTHETIC_BENCHMARK_REGISTRY",
    "SYNTHETIC_BENCHMARK_SCENARIOS",
    "SyntheticBenchmarkScenario",
]
