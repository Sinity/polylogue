from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Literal

import pytest

from devtools import synthetic_benchmark_runtime as synthetic_runtime
from devtools.benchmark_campaigns import (
    SYNTHETIC_CAMPAIGNS,
    run_full_campaign,
    run_synthetic_benchmark_campaign,
)
from devtools.large_archive_generator import ArchiveMetrics
from devtools.synthetic_benchmark_catalog import (
    SYNTHETIC_BENCHMARK_REGISTRY,
    SYNTHETIC_BENCHMARK_SCENARIOS,
)
from devtools.synthetic_benchmark_runtime import (
    CampaignResult,
    resolve_synthetic_benchmark_runner,
    run_action_event_materialization_campaign,
    run_session_product_materialization_campaign,
)
from polylogue.scenarios import ExecutionKind
from polylogue.storage.session_product_runtime import SessionProductCounts


def test_synthetic_benchmark_registry_is_compiled_from_authored_scenarios() -> None:
    assert set(SYNTHETIC_CAMPAIGNS) == {scenario.name for scenario in SYNTHETIC_BENCHMARK_SCENARIOS}
    assert set(SYNTHETIC_BENCHMARK_REGISTRY) == set(SYNTHETIC_CAMPAIGNS)
    assert SYNTHETIC_CAMPAIGNS["incremental-index"].execution is not None
    assert SYNTHETIC_CAMPAIGNS["incremental-index"].execution.kind is ExecutionKind.RUNNER
    assert SYNTHETIC_CAMPAIGNS["incremental-index"].execution.runner == "incremental-index"
    assert SYNTHETIC_CAMPAIGNS["incremental-index"].scale_targets == ("small", "medium", "large", "stretch")
    assert SYNTHETIC_BENCHMARK_REGISTRY["fts-rebuild"].description == "Benchmark full FTS5 index rebuild"
    assert (
        SYNTHETIC_BENCHMARK_REGISTRY["action-event-materialization"].description
        == "Benchmark action-event read-model rebuild over synthetic tool-use transcripts"
    )
    assert (
        SYNTHETIC_BENCHMARK_REGISTRY["session-product-materialization"].description
        == "Benchmark durable session-product rebuild over synthetic archive conversations"
    )


def test_all_authored_synthetic_benchmark_runners_resolve() -> None:
    for campaign in SYNTHETIC_CAMPAIGNS.values():
        assert campaign.execution is not None
        assert campaign.execution.runner
        assert callable(resolve_synthetic_benchmark_runner(campaign.execution.runner))


@pytest.mark.asyncio
async def test_run_synthetic_benchmark_campaign_preserves_scenario_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_incremental(_db_path: Path) -> CampaignResult:
        return CampaignResult(
            campaign_name="incremental-index",
            scale_level="",
            metrics={"total_wall_s": 1.5},
            db_stats={},
            timestamp="2026-04-13T00:00:00+00:00",
        )

    monkeypatch.setitem(
        synthetic_runtime.SYNTHETIC_BENCHMARK_RUNNERS,
        "incremental-index",
        fake_incremental,
    )

    result = await run_synthetic_benchmark_campaign("incremental-index", tmp_path / "benchmark.db")

    assert result.origin == "authored.synthetic-benchmark"
    assert result.path_targets == []
    assert result.artifact_targets == ["message_source_rows", "message_fts"]
    assert result.operation_targets == ["index-message-fts", "index.message-fts-incremental"]
    assert result.tags == ["benchmark", "synthetic", "fts"]


@pytest.mark.asyncio
async def test_run_full_campaign_skips_scenarios_outside_scale_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_generate_archive(
        _spec: object, archive_dir: Path, *, corpus_source: object = None
    ) -> ArchiveMetrics:
        from devtools.large_archive_generator import ArchiveMetrics

        archive_dir.mkdir(parents=True, exist_ok=True)
        (archive_dir / "benchmark.db").write_bytes(b"")
        return ArchiveMetrics(
            wall_time_s=0.5,
            db_size_bytes=0,
            message_count=10,
            conversation_count=2,
        )

    async def fake_run_campaign(name: str, _db_path: Path) -> CampaignResult:
        return CampaignResult(
            campaign_name=name,
            scale_level="",
            metrics={"rebuild_wall_s": 1.0, "total_wall_s": 1.0, "list_50_wall_s": 1.0, "total_readiness_s": 1.0},
            db_stats={},
        )

    skipped = SYNTHETIC_CAMPAIGNS["startup-readiness"]
    limited = SYNTHETIC_CAMPAIGNS["incremental-index"]
    monkeypatch.setitem(
        SYNTHETIC_CAMPAIGNS,
        skipped.name,
        type(skipped)(**{**skipped.__dict__, "scale_targets": ("large", "stretch")}),
    )
    monkeypatch.setitem(
        SYNTHETIC_CAMPAIGNS,
        limited.name,
        type(limited)(**{**limited.__dict__, "scale_targets": ("small",)}),
    )
    monkeypatch.setattr("devtools.large_archive_generator.generate_archive", fake_generate_archive)
    monkeypatch.setattr("devtools.benchmark_campaigns.run_synthetic_benchmark_campaign", fake_run_campaign)

    results = await run_full_campaign("small", tmp_path)

    assert "incremental-index" in {result.campaign_name for result in results}
    assert "startup-readiness" not in {result.campaign_name for result in results}


def test_action_event_materialization_campaign_reports_action_row_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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

        def __exit__(
            self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
        ) -> Literal[False]:
            return False

    monkeypatch.setattr("devtools.synthetic_benchmark_runtime._db_row_counts", lambda _db_path: next(row_counts))
    monkeypatch.setattr("polylogue.storage.backends.connection.open_connection", lambda _db_path: FakeContext())
    monkeypatch.setattr(
        "polylogue.storage.action_event_rebuild_runtime.rebuild_action_event_read_model_sync", lambda conn: 7
    )

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


def test_session_product_materialization_campaign_reports_rebuild_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    before = {
        "session_profiles_count": 1,
        "session_profiles_fts_count": 1,
    }
    after = {
        "session_profiles_count": 5,
        "session_profiles_fts_count": 5,
        "session_work_events_count": 8,
        "session_work_events_fts_count": 8,
        "session_phases_count": 3,
        "work_threads_count": 2,
        "work_threads_fts_count": 2,
        "session_tag_rollups_count": 4,
        "day_session_summaries_count": 2,
        "week_session_summaries_count": 1,
    }
    table_counts = iter((before, after))
    committed: list[str] = []

    class FakeConn:
        def commit(self) -> None:
            committed.append("commit")

    class FakeContext:
        def __enter__(self) -> FakeConn:
            return FakeConn()

        def __exit__(
            self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
        ) -> Literal[False]:
            return False

    monkeypatch.setattr(
        "devtools.synthetic_benchmark_runtime._session_product_table_counts", lambda _db_path: next(table_counts)
    )
    monkeypatch.setattr("polylogue.storage.backends.connection.open_connection", lambda _db_path: FakeContext())
    monkeypatch.setattr(
        "polylogue.storage.session_product_rebuild.rebuild_session_products_sync",
        lambda conn: SessionProductCounts(
            profiles=5,
            work_events=8,
            phases=3,
            threads=2,
            tag_rollups=4,
            day_summaries=2,
        ),
    )

    result = run_session_product_materialization_campaign(tmp_path / "benchmark.db")

    assert committed == ["commit"]
    assert result.campaign_name == "session-product-materialization"
    assert result.metrics["profiles_rebuilt"] == 5
    assert result.metrics["threads_rebuilt"] == 2
    assert result.db_stats == {
        "session_profiles_before": 1,
        "session_profiles_after": 5,
        "session_profiles_fts_after": 5,
        "session_work_events_after": 8,
        "session_work_events_fts_after": 8,
        "session_phases_after": 3,
        "work_threads_after": 2,
        "work_threads_fts_after": 2,
        "session_tag_rollups_after": 4,
        "day_session_summaries_after": 2,
        "week_session_summaries_after": 1,
    }
