"""Authored durable benchmark campaigns shared across control-plane surfaces."""

from __future__ import annotations

from devtools.benchmark_models import BenchmarkCampaignEntry, compile_benchmark_campaigns
from devtools.execution_specs import pytest_execution

BENCHMARK_SCENARIOS: tuple[BenchmarkCampaignEntry, ...] = (
    BenchmarkCampaignEntry(
        name="search-filters",
        description="FTS and ConversationFilter benchmark domain",
        execution=pytest_execution("tests/benchmarks/test_search_filters.py"),
        notes=(
            "Canonical search/filter latency domain.",
            "Keep on session-seeded DB fixtures for comparability.",
        ),
        origin="authored.benchmark-domain",
        artifact_targets=("conversation_query_results", "message_fts"),
        operation_targets=("query-conversations", "benchmark.query.search-filters"),
        tags=("benchmark", "search", "filters"),
    ),
    BenchmarkCampaignEntry(
        name="storage",
        description="Repository/backend list/get-many/save benchmark domain",
        execution=pytest_execution("tests/benchmarks/test_storage.py"),
        notes=("Canonical storage CRUD and batch-write latency domain.",),
        origin="authored.benchmark-domain",
        artifact_targets=("conversation_rows", "message_rows", "raw_rows"),
        operation_targets=("benchmark.storage.crud",),
        tags=("benchmark", "storage"),
    ),
    BenchmarkCampaignEntry(
        name="pipeline",
        description="Index rebuild/update, action-event repair, plus hashing/semantic helper benchmark domain",
        execution=pytest_execution("tests/benchmarks/test_pipeline.py"),
        notes=("Covers indexing, repair, and hot helper throughput.",),
        origin="authored.benchmark-domain",
        artifact_targets=("index_state", "pipeline_helpers"),
        operation_targets=(
            "benchmark.pipeline.index-and-helpers",
            "benchmark.repair.action-events",
        ),
        tags=("benchmark", "pipeline"),
    ),
)

BENCHMARK_SCENARIO_INDEX: dict[str, BenchmarkCampaignEntry] = compile_benchmark_campaigns(BENCHMARK_SCENARIOS)


__all__ = [
    "BENCHMARK_SCENARIO_INDEX",
    "BENCHMARK_SCENARIOS",
    "BenchmarkCampaignEntry",
    "compile_benchmark_campaigns",
]
