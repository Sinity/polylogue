from __future__ import annotations

from pathlib import Path

import pytest

from devtools.benchmark_campaigns import (
    CAMPAIGN_REGISTRY,
    SYNTHETIC_BENCHMARK_REGISTRY,
    SYNTHETIC_BENCHMARK_SCENARIOS,
    run_synthetic_benchmark_campaign,
)


def test_synthetic_benchmark_registry_is_compiled_from_authored_scenarios() -> None:
    assert set(CAMPAIGN_REGISTRY) == {scenario.scenario_id for scenario in SYNTHETIC_BENCHMARK_SCENARIOS}
    assert set(SYNTHETIC_BENCHMARK_REGISTRY) == set(CAMPAIGN_REGISTRY)
    assert SYNTHETIC_BENCHMARK_REGISTRY["fts-rebuild"].description == "Benchmark full FTS5 index rebuild"


@pytest.mark.asyncio
async def test_run_synthetic_benchmark_campaign_preserves_scenario_metadata(monkeypatch, tmp_path: Path) -> None:
    async def fake_incremental(_db_path: Path):
        from devtools.benchmark_campaigns import CampaignResult

        return CampaignResult(
            campaign_name="incremental-index",
            scale_level="",
            metrics={"total_wall_s": 1.5},
            db_stats={},
            timestamp="2026-04-13T00:00:00+00:00",
        )

    monkeypatch.setattr("devtools.benchmark_campaigns.run_incremental_index_campaign", fake_incremental)

    result = await run_synthetic_benchmark_campaign("incremental-index", tmp_path / "benchmark.db")

    assert result.origin == "authored.synthetic-benchmark"
    assert result.artifact_targets == ["message_fts"]
    assert result.operation_targets == ["index.message-fts-incremental"]
    assert result.tags == ["benchmark", "synthetic", "fts"]
