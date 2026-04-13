"""Typed benchmark campaign catalog shared across control-plane surfaces."""

from __future__ import annotations

from .benchmark_models import BenchmarkCampaignEntry
from .benchmark_scenario_catalog import BENCHMARK_SCENARIOS
from .synthetic_benchmark_catalog import SYNTHETIC_BENCHMARK_SCENARIOS


def build_benchmark_entries() -> tuple[BenchmarkCampaignEntry, ...]:
    return tuple(sorted(BENCHMARK_SCENARIOS, key=lambda item: item.name))


def build_synthetic_benchmark_entries() -> tuple[BenchmarkCampaignEntry, ...]:
    return tuple(sorted(SYNTHETIC_BENCHMARK_SCENARIOS, key=lambda item: item.name))


__all__ = [
    "BenchmarkCampaignEntry",
    "build_benchmark_entries",
    "build_synthetic_benchmark_entries",
]
