"""Authored synthetic benchmark scenarios shared across control-plane surfaces."""

from __future__ import annotations

from devtools.benchmark_models import BenchmarkCampaignEntry, compile_benchmark_campaigns
from devtools.execution_specs import runner_execution
from polylogue.scenarios import ScenarioProjectionSourceKind

SYNTHETIC_BENCHMARK_SCENARIOS: tuple[BenchmarkCampaignEntry, ...] = (
    BenchmarkCampaignEntry(
        name="fts-rebuild",
        description="Benchmark full FTS5 index rebuild",
        execution=runner_execution("fts-rebuild"),
        summary_metric="rebuild_wall_s",
        summary_label="s",
        scale_targets=("small", "medium", "large", "stretch"),
        origin="authored.synthetic-benchmark",
        artifact_targets=("message_fts",),
        operation_targets=("index.message-fts-rebuild",),
        tags=("benchmark", "synthetic", "fts"),
        projection_kind=ScenarioProjectionSourceKind.SYNTHETIC_BENCHMARK,
    ),
    BenchmarkCampaignEntry(
        name="incremental-index",
        description="Benchmark incremental FTS index updates",
        execution=runner_execution("incremental-index"),
        summary_metric="total_wall_s",
        summary_label="s",
        scale_targets=("small", "medium", "large", "stretch"),
        origin="authored.synthetic-benchmark",
        artifact_targets=("message_fts",),
        operation_targets=("index.message-fts-incremental",),
        tags=("benchmark", "synthetic", "fts"),
        projection_kind=ScenarioProjectionSourceKind.SYNTHETIC_BENCHMARK,
    ),
    BenchmarkCampaignEntry(
        name="filter-scan",
        description="Benchmark common filter query patterns",
        execution=runner_execution("filter-scan"),
        summary_metric="list_50_wall_s",
        summary_label="s",
        scale_targets=("small", "medium", "large", "stretch"),
        origin="authored.synthetic-benchmark",
        artifact_targets=("conversation_query_results",),
        operation_targets=("query.filters.synthetic-scan",),
        tags=("benchmark", "synthetic", "filters"),
        projection_kind=ScenarioProjectionSourceKind.SYNTHETIC_BENCHMARK,
    ),
    BenchmarkCampaignEntry(
        name="startup-health",
        description="Benchmark check --runtime startup speed",
        execution=runner_execution("startup-health"),
        summary_metric="total_health_s",
        summary_label="s",
        scale_targets=("small", "medium", "large", "stretch"),
        origin="authored.synthetic-benchmark",
        artifact_targets=("archive_health",),
        operation_targets=("health.startup.synthetic",),
        tags=("benchmark", "synthetic", "health"),
        projection_kind=ScenarioProjectionSourceKind.SYNTHETIC_BENCHMARK,
    ),
    BenchmarkCampaignEntry(
        name="action-event-materialization",
        description="Benchmark action-event read-model rebuild over synthetic tool-use transcripts",
        execution=runner_execution("action-event-materialization"),
        summary_metric="rebuild_wall_s",
        summary_label="s",
        scale_targets=("small", "medium", "large", "stretch"),
        origin="authored.synthetic-benchmark",
        artifact_targets=("tool_use_source_blocks", "action_event_rows", "action_event_fts"),
        operation_targets=("materialize-action-events",),
        tags=("benchmark", "synthetic", "action-events"),
        projection_kind=ScenarioProjectionSourceKind.SYNTHETIC_BENCHMARK,
    ),
    BenchmarkCampaignEntry(
        name="session-product-materialization",
        description="Benchmark durable session-product rebuild over synthetic archive conversations",
        execution=runner_execution("session-product-materialization"),
        summary_metric="rebuild_wall_s",
        summary_label="s",
        scale_targets=("small", "medium", "large", "stretch"),
        origin="authored.synthetic-benchmark",
        artifact_targets=("session_product_source_conversations", "session_product_rows", "session_product_fts"),
        operation_targets=("materialize-session-products",),
        tags=("benchmark", "synthetic", "session-products"),
        projection_kind=ScenarioProjectionSourceKind.SYNTHETIC_BENCHMARK,
    ),
)

SYNTHETIC_BENCHMARK_REGISTRY: dict[str, BenchmarkCampaignEntry] = compile_benchmark_campaigns(
    SYNTHETIC_BENCHMARK_SCENARIOS
)


__all__ = [
    "SYNTHETIC_BENCHMARK_REGISTRY",
    "SYNTHETIC_BENCHMARK_SCENARIOS",
    "BenchmarkCampaignEntry",
]
