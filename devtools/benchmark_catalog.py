"""Typed benchmark campaign catalog shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.scenarios import ScenarioMetadata

from .benchmark_campaign import CAMPAIGNS as BENCHMARK_CAMPAIGNS
from .benchmark_campaigns import SYNTHETIC_BENCHMARK_SCENARIOS


@dataclass(frozen=True)
class BenchmarkCampaignEntry(ScenarioMetadata):
    name: str
    description: str
    tests: tuple[str, ...]
    notes: tuple[str, ...] = ()
    warn_pct: float = 0.0
    fail_pct: float = 0.0


def build_benchmark_entries() -> tuple[BenchmarkCampaignEntry, ...]:
    entries = [
        BenchmarkCampaignEntry(
            name=campaign.name,
            description=campaign.description,
            tests=tuple(campaign.tests),
            notes=tuple(campaign.notes),
            warn_pct=campaign.warn_pct,
            fail_pct=campaign.fail_pct,
            origin=campaign.origin,
            path_targets=tuple(campaign.path_targets),
            artifact_targets=tuple(campaign.artifact_targets),
            operation_targets=tuple(campaign.operation_targets),
            tags=tuple(campaign.tags),
        )
        for campaign in BENCHMARK_CAMPAIGNS.values()
    ]
    return tuple(sorted(entries, key=lambda item: item.name))


def build_synthetic_benchmark_entries() -> tuple[BenchmarkCampaignEntry, ...]:
    entries = [
        BenchmarkCampaignEntry(
            name=scenario.scenario_id,
            description=scenario.description,
            tests=(),
            notes=(),
            origin=scenario.origin,
            path_targets=scenario.path_targets,
            artifact_targets=scenario.artifact_targets,
            operation_targets=scenario.operation_targets,
            tags=scenario.tags,
        )
        for scenario in SYNTHETIC_BENCHMARK_SCENARIOS
    ]
    return tuple(sorted(entries, key=lambda item: item.name))


__all__ = [
    "BenchmarkCampaignEntry",
    "build_benchmark_entries",
    "build_synthetic_benchmark_entries",
]
