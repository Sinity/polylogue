"""Authored durable benchmark scenarios shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.scenarios import ScenarioMetadata


@dataclass(frozen=True)
class Campaign(ScenarioMetadata):
    name: str
    description: str
    tests: tuple[str, ...]
    notes: tuple[str, ...] = ()
    warn_pct: float = 10.0
    fail_pct: float = 20.0


@dataclass(frozen=True)
class BenchmarkScenario(ScenarioMetadata):
    """Authored scenario metadata for one durable benchmark campaign."""

    scenario_id: str
    description: str
    tests: tuple[str, ...]
    notes: tuple[str, ...] = ()
    warn_pct: float = 10.0
    fail_pct: float = 20.0

    def compile(self) -> Campaign:
        return Campaign(
            name=self.scenario_id,
            description=self.description,
            tests=self.tests,
            notes=self.notes,
            warn_pct=self.warn_pct,
            fail_pct=self.fail_pct,
            origin=self.origin,
            path_targets=self.path_targets,
            artifact_targets=self.artifact_targets,
            operation_targets=self.operation_targets,
            tags=self.tags,
        )


def compile_benchmark_scenarios(scenarios: tuple[BenchmarkScenario, ...]) -> dict[str, Campaign]:
    return {scenario.scenario_id: scenario.compile() for scenario in scenarios}


BENCHMARK_SCENARIOS: tuple[BenchmarkScenario, ...] = (
    BenchmarkScenario(
        scenario_id="search-filters",
        description="FTS and ConversationFilter benchmark domain",
        tests=("tests/benchmarks/test_search_filters.py",),
        notes=(
            "Canonical search/filter latency domain.",
            "Keep on session-seeded DB fixtures for comparability.",
        ),
        origin="authored.benchmark-domain",
        artifact_targets=("conversation_query_results", "message_fts"),
        operation_targets=("benchmark.query.search-filters",),
        tags=("benchmark", "search", "filters"),
    ),
    BenchmarkScenario(
        scenario_id="storage",
        description="Repository/backend list/get-many/save benchmark domain",
        tests=("tests/benchmarks/test_storage.py",),
        notes=("Canonical storage CRUD and batch-write latency domain.",),
        origin="authored.benchmark-domain",
        artifact_targets=("conversation_rows", "message_rows", "raw_rows"),
        operation_targets=("benchmark.storage.crud",),
        tags=("benchmark", "storage"),
    ),
    BenchmarkScenario(
        scenario_id="pipeline",
        description="Index rebuild/update plus hashing/semantic helper benchmark domain",
        tests=("tests/benchmarks/test_pipeline.py",),
        notes=("Covers indexing plus hot helper throughput.",),
        origin="authored.benchmark-domain",
        artifact_targets=("index_state", "pipeline_helpers"),
        operation_targets=("benchmark.pipeline.index-and-helpers",),
        tags=("benchmark", "pipeline"),
    ),
)

BENCHMARK_CAMPAIGNS: dict[str, Campaign] = compile_benchmark_scenarios(BENCHMARK_SCENARIOS)


__all__ = [
    "BENCHMARK_CAMPAIGNS",
    "BENCHMARK_SCENARIOS",
    "BenchmarkScenario",
    "Campaign",
    "compile_benchmark_scenarios",
]
