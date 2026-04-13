"""Shared benchmark campaign metadata for control-plane catalogs."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.scenarios import ExecutableScenario, ScenarioProjectionSourceKind


@dataclass(frozen=True, kw_only=True)
class BenchmarkCampaignEntry(ExecutableScenario):
    notes: tuple[str, ...] = ()
    warn_pct: float = 0.0
    fail_pct: float = 0.0
    summary_metric: str = ""
    summary_label: str = ""
    scale_targets: tuple[str, ...] = ()
    projection_kind: ScenarioProjectionSourceKind = ScenarioProjectionSourceKind.BENCHMARK_CAMPAIGN

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return self.projection_kind


def compile_benchmark_campaigns(
    campaigns: tuple[BenchmarkCampaignEntry, ...],
) -> dict[str, BenchmarkCampaignEntry]:
    return {campaign.name: campaign for campaign in campaigns}


__all__ = [
    "BenchmarkCampaignEntry",
    "compile_benchmark_campaigns",
]
