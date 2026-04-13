"""Shared benchmark campaign metadata for control-plane catalogs."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.execution_specs import ExecutionSpec
from polylogue.scenarios import ScenarioMetadata, ScenarioProjectionSource, ScenarioProjectionSourceKind


@dataclass(frozen=True)
class BenchmarkCampaignEntry(ScenarioProjectionSource, ScenarioMetadata):
    name: str
    description: str
    execution: ExecutionSpec | None = None
    notes: tuple[str, ...] = ()
    warn_pct: float = 0.0
    fail_pct: float = 0.0
    runner_name: str = ""
    summary_metric: str = ""
    summary_label: str = ""
    scale_targets: tuple[str, ...] = ()
    projection_kind: ScenarioProjectionSourceKind = ScenarioProjectionSourceKind.BENCHMARK_CAMPAIGN

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return self.projection_kind

    @property
    def projection_name(self) -> str:
        return self.name

    @property
    def projection_description(self) -> str:
        return self.description

    @property
    def tests(self) -> tuple[str, ...]:
        if self.execution is None:
            return ()
        return self.execution.pytest_targets


def compile_benchmark_campaigns(
    campaigns: tuple[BenchmarkCampaignEntry, ...],
) -> dict[str, BenchmarkCampaignEntry]:
    return {campaign.name: campaign for campaign in campaigns}


__all__ = [
    "BenchmarkCampaignEntry",
    "compile_benchmark_campaigns",
]
