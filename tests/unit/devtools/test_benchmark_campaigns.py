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
    assert (
        SYNTHETIC_BENCHMARK_REGISTRY["action-event-materialization"].description
        == "Benchmark action-event read-model rebuild over synthetic tool-use transcripts"
    )


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


def test_action_event_materialization_campaign_reports_action_row_counts(monkeypatch, tmp_path: Path) -> None:
    from devtools.benchmark_campaigns import run_action_event_materialization_campaign

    before = {
        "action_events_count": 2,
        "action_fts_rows": 2,
        "db_size_bytes": 128,
    }
    after = {
        "action_events_count": 7,
        "action_fts_rows": 7,
        "db_size_bytes": 256,
    }
    row_counts = iter((before, after))
    executed: list[str] = []
    committed: list[str] = []

    class FakeConn:
        def execute(self, sql: str) -> None:
            executed.append(sql)

        def commit(self) -> None:
            committed.append("commit")

    class FakeContext:
        def __enter__(self) -> FakeConn:
            return FakeConn()

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr("devtools.benchmark_campaigns._db_row_counts", lambda _db_path: next(row_counts))
    monkeypatch.setattr("polylogue.storage.backends.connection.open_connection", lambda _db_path: FakeContext())
    monkeypatch.setattr("polylogue.storage.action_event_rebuild_runtime.rebuild_action_event_read_model_sync", lambda conn: 7)

    result = run_action_event_materialization_campaign(tmp_path / "benchmark.db")

    assert executed == ["DELETE FROM action_events"]
    assert committed == ["commit", "commit"]
    assert result.campaign_name == "action-event-materialization"
    assert result.metrics["action_event_rows_rebuilt"] == 7
    assert result.db_stats == {
        "action_events_before": 2,
        "action_events_after": 7,
        "action_fts_rows_after": 7,
        "db_size_bytes": 256,
    }
