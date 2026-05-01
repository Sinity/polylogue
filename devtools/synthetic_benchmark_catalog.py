"""Authored synthetic benchmark scenarios shared across control-plane surfaces."""

from __future__ import annotations

from devtools.benchmark_models import BenchmarkCampaignEntry, compile_benchmark_campaigns
from polylogue.scenarios import ScenarioProjectionSourceKind, runner_execution

SYNTHETIC_BENCHMARK_SCENARIOS: tuple[BenchmarkCampaignEntry, ...] = (
    BenchmarkCampaignEntry(
        name="fts-rebuild",
        description="Benchmark full FTS5 index rebuild",
        execution=runner_execution("fts-rebuild"),
        summary_metric="rebuild_wall_s",
        summary_label="s",
        scale_targets=("small", "medium", "large", "stretch"),
        origin="authored.synthetic-benchmark",
        artifact_targets=("message_source_rows", "message_fts"),
        operation_targets=("index-message-fts", "index.message-fts-rebuild"),
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
        artifact_targets=("message_source_rows", "message_fts"),
        operation_targets=("index-message-fts", "index.message-fts-incremental"),
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
        artifact_targets=("message_fts", "conversation_query_results"),
        operation_targets=("query-conversations", "query.filters.synthetic-scan"),
        tags=("benchmark", "synthetic", "filters"),
        projection_kind=ScenarioProjectionSourceKind.SYNTHETIC_BENCHMARK,
    ),
    BenchmarkCampaignEntry(
        name="startup-readiness",
        description="Benchmark check --runtime startup speed",
        execution=runner_execution("startup-readiness"),
        summary_metric="total_readiness_s",
        summary_label="s",
        scale_targets=("small", "medium", "large", "stretch"),
        origin="authored.synthetic-benchmark",
        artifact_targets=("message_fts", "archive_readiness"),
        operation_targets=("project-archive-readiness", "readiness.startup.synthetic"),
        tags=("benchmark", "synthetic", "readiness"),
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
        artifact_targets=(
            "session_product_source_conversations",
            "session_profile_rows",
            "session_profile_merged_fts",
            "session_profile_evidence_fts",
            "session_profile_inference_fts",
            "session_profile_enrichment_fts",
            "session_work_event_rows",
            "session_work_event_fts",
            "session_phase_rows",
            "work_thread_rows",
            "work_thread_fts",
            "session_tag_rollup_rows",
            "day_session_summary_rows",
            "session_product_rows",
            "session_product_fts",
        ),
        operation_targets=("materialize-session-insights",),
        tags=("benchmark", "synthetic", "session-insights"),
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
