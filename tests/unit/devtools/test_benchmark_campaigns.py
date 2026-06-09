from __future__ import annotations

import json
from argparse import Namespace
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
from devtools.daemon_live_benchmark import append_daemon_live_workload, generate_daemon_live_workload
from devtools.large_archive_generator import ArchiveMetrics
from devtools.synthetic_benchmark_catalog import (
    SYNTHETIC_BENCHMARK_REGISTRY,
    SYNTHETIC_BENCHMARK_SCENARIOS,
)
from devtools.synthetic_benchmark_runtime import (
    CampaignResult,
    resolve_synthetic_benchmark_runner,
    run_daemon_live_convergence_campaign,
    run_session_insight_materialization_campaign,
)
from polylogue.scenarios import ExecutionKind
from polylogue.sources.dispatch import detect_provider
from polylogue.storage.insights.session.runtime import SessionInsightCounts
from polylogue.types import Provider


def test_synthetic_benchmark_registry_is_compiled_from_authored_scenarios() -> None:
    assert set(SYNTHETIC_CAMPAIGNS) == {scenario.name for scenario in SYNTHETIC_BENCHMARK_SCENARIOS}
    assert set(SYNTHETIC_BENCHMARK_REGISTRY) == set(SYNTHETIC_CAMPAIGNS)
    assert SYNTHETIC_CAMPAIGNS["incremental-index"].execution is not None
    assert SYNTHETIC_CAMPAIGNS["incremental-index"].execution.kind is ExecutionKind.RUNNER
    assert SYNTHETIC_CAMPAIGNS["incremental-index"].execution.runner == "incremental-index"
    assert SYNTHETIC_CAMPAIGNS["incremental-index"].scale_targets == ("small", "medium", "large", "stretch")
    assert SYNTHETIC_BENCHMARK_REGISTRY["fts-rebuild"].description == "Benchmark full FTS5 index rebuild"
    assert (
        SYNTHETIC_BENCHMARK_REGISTRY["session-insight-materialization"].description
        == "Benchmark durable session-insight rebuild over synthetic archive sessions"
    )
    assert (
        SYNTHETIC_BENCHMARK_REGISTRY["daemon-live-convergence"].description
        == "Benchmark daemon live batch convergence over generated JSONL source workloads"
    )


def test_all_authored_synthetic_benchmark_runners_resolve() -> None:
    for campaign in SYNTHETIC_CAMPAIGNS.values():
        assert campaign.execution is not None
        assert campaign.execution.runner
        assert callable(resolve_synthetic_benchmark_runner(campaign.execution.runner))


def test_daemon_live_workload_uses_synthetic_provider_wire_formats(tmp_path: Path) -> None:
    workload = generate_daemon_live_workload(tmp_path, scale="small")
    claude_file = workload.files_by_provider["claude-code"][0]
    codex_file = workload.files_by_provider["codex"][0]
    claude_records = [json.loads(line) for line in claude_file.read_text().splitlines()]
    codex_records = [json.loads(line) for line in codex_file.read_text().splitlines()]

    assert workload.message_count == 100
    assert len(workload.files_by_provider["claude-code"]) == 5
    assert len(workload.files_by_provider["codex"]) == 5
    assert claude_file.read_bytes().endswith(b"\n")
    assert codex_file.read_bytes().endswith(b"\n")
    assert detect_provider(claude_records, claude_file) is Provider.CLAUDE_CODE
    assert detect_provider(codex_records, codex_file) is Provider.CODEX
    assert all(isinstance(record.get("message"), dict) for record in claude_records)
    assert all(
        isinstance(record["message"].get("content"), list)
        for record in claude_records
        if isinstance(record.get("message"), dict)
    )
    assert _count_tool_use_blocks(claude_records, provider="claude-code") > 0
    assert all(record.get("type") == "message" for record in codex_records)
    assert _count_tool_use_blocks(codex_records, provider="codex") > 0


def _count_tool_use_blocks(records: list[dict[str, object]], *, provider: str) -> int:
    count = 0
    for record in records:
        if provider == "claude-code":
            message = record.get("message")
            content = message.get("content") if isinstance(message, dict) else None
        else:
            content = record.get("content")
        if not isinstance(content, list):
            continue
        count += sum(1 for block in content if isinstance(block, dict) and block.get("type") == "tool_use")
    return count


def test_daemon_live_append_preserves_generated_session_identity(tmp_path: Path) -> None:
    workload = generate_daemon_live_workload(tmp_path, scale="small")
    claude_file = workload.files_by_provider["claude-code"][0]
    codex_file = workload.files_by_provider["codex"][0]
    claude_before = [json.loads(line) for line in claude_file.read_text().splitlines()]
    codex_before = [json.loads(line) for line in codex_file.read_text().splitlines()]

    updated = append_daemon_live_workload(workload, message_index=10)
    claude_after = [json.loads(line) for line in claude_file.read_text().splitlines()]
    codex_after = [json.loads(line) for line in codex_file.read_text().splitlines()]

    assert updated.append_delta_bytes > 0
    assert len(claude_after) == len(claude_before) + 1
    assert claude_after[-1]["sessionId"] == claude_before[0]["sessionId"]
    assert claude_after[-1]["parentUuid"] == claude_before[-1]["uuid"]
    assert isinstance(claude_after[-1].get("message"), dict)
    assert isinstance(claude_after[-1]["message"].get("content"), list)
    assert len(codex_after) == len(codex_before) + 1
    assert codex_after[-1]["type"] == "message"
    assert codex_after[-1]["id"] == f"{codex_file.stem}-append-0010"
    assert detect_provider(codex_after, codex_file) is Provider.CODEX


def test_daemon_live_workload_generation_replaces_stale_jsonl(tmp_path: Path) -> None:
    workload = generate_daemon_live_workload(tmp_path, scale="small")
    stale_file = workload.files[0]
    stale_file.write_text("{}\n", encoding="utf-8")

    regenerated = generate_daemon_live_workload(tmp_path, scale="small")

    assert regenerated.files[0] == stale_file
    assert len(stale_file.read_text(encoding="utf-8").splitlines()) == (
        regenerated.message_count // len(regenerated.files)
    )


@pytest.mark.asyncio
async def test_run_benchmark_campaigns_skips_seed_archive_for_daemon_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from devtools import run_campaign

    stale_file = tmp_path / "archive-large" / "stale.db"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old benchmark state", encoding="utf-8")

    async def fail_generate_archive(*_args: object, **_kwargs: object) -> ArchiveMetrics:
        raise AssertionError("daemon-live-convergence should generate its own live workload")

    async def fake_run_campaign(name: str, db_path: Path) -> CampaignResult:
        assert name == "daemon-live-convergence"
        assert db_path == tmp_path / "archive-large" / "benchmark.db"
        assert not stale_file.exists()
        return CampaignResult(campaign_name=name, scale_level="", metrics={"total_wall_s": 1.0}, db_stats={})

    saved_results: list[CampaignResult] = []

    def fake_save_campaign_reports(results: list[CampaignResult], output_dir: Path) -> list[Path]:
        saved_results.extend(results)
        return [output_dir / "report.json"]

    monkeypatch.setattr("devtools.large_archive_generator.generate_archive", fail_generate_archive)
    monkeypatch.setattr("devtools.benchmark_campaigns.run_synthetic_benchmark_campaign", fake_run_campaign)
    monkeypatch.setattr("devtools.campaign_report.save_campaign_reports", fake_save_campaign_reports)

    result = await run_campaign._run(
        Namespace(
            list_campaigns=False,
            campaign="daemon-live-convergence",
            scale="large",
            output=tmp_path,
            seed=42,
            corpus_source="default",
        )
    )

    assert result == 0
    assert [item.campaign_name for item in saved_results] == ["daemon-live-convergence"]


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
async def test_run_daemon_live_convergence_campaign_reports_workload_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_workload(_db_path: Path) -> tuple[dict[str, float], dict[str, int]]:
        return (
            {
                "total_wall_s": 0.25,
                "files_ingested": 2.0,
                "messages_generated": 8.0,
                "messages_per_s": 32.0,
                "converged_files": 1.0,
                "failed_files": 0.0,
            },
            {
                "messages_count": 8,
                "fts_rows": 8,
                "live_cursor_count": 2,
                "claude_code_files": 1,
                "codex_files": 1,
            },
        )

    class FakeConn:
        def commit(self) -> None:
            pass

    class FakeContext:
        def __enter__(self) -> FakeConn:
            return FakeConn()

        def __exit__(
            self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
        ) -> Literal[False]:
            return False

    monkeypatch.setattr(
        "devtools.daemon_live_benchmark.run_daemon_live_convergence_workload",
        fake_workload,
    )
    monkeypatch.setattr("polylogue.storage.sqlite.connection.open_connection", lambda _db_path: FakeContext())
    monkeypatch.setattr(
        "polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync",
        lambda conn: SessionInsightCounts(
            profiles=2,
            work_events=3,
            phases=1,
            threads=1,
            tag_rollups=2,
        ),
    )
    monkeypatch.setattr("devtools.synthetic_benchmark_runtime._db_row_counts", lambda _db_path: {"messages_count": 8})
    monkeypatch.setattr(
        "devtools.synthetic_benchmark_runtime._session_insight_table_counts",
        lambda _db_path: {"session_profiles_count": 2},
    )

    result = await run_daemon_live_convergence_campaign(tmp_path / "benchmark.db")

    assert result.campaign_name == "daemon-live-convergence"
    assert result.metrics["total_wall_s"] == 0.25
    assert result.metrics["messages_per_s"] == 32.0
    assert result.metrics["profiles_rebuilt"] == 2.0
    assert result.db_stats == {
        "messages_count": 8,
        "fts_rows": 8,
        "live_cursor_count": 2,
        "claude_code_files": 1,
        "codex_files": 1,
        "session_profiles_count": 2,
    }


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
            session_count=2,
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


def test_session_insight_materialization_campaign_reports_rebuild_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    before = {
        "session_profiles_count": 1,
    }
    after = {
        "session_profiles_count": 5,
        "session_work_events_count": 8,
        "session_work_events_fts_count": 8,
        "session_phases_count": 3,
        "threads_count": 2,
        "threads_fts_count": 2,
        "session_tag_rollups_count": 4,
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
        "devtools.synthetic_benchmark_runtime._session_insight_table_counts", lambda _db_path: next(table_counts)
    )
    monkeypatch.setattr("polylogue.storage.sqlite.connection.open_connection", lambda _db_path: FakeContext())
    monkeypatch.setattr(
        "polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync",
        lambda conn: SessionInsightCounts(
            profiles=5,
            work_events=8,
            phases=3,
            threads=2,
            tag_rollups=4,
        ),
    )

    result = run_session_insight_materialization_campaign(tmp_path / "benchmark.db")

    assert committed == ["commit"]
    assert result.campaign_name == "session-insight-materialization"
    assert result.metrics["profiles_rebuilt"] == 5
    assert result.metrics["threads_rebuilt"] == 2
    assert result.db_stats == {
        "session_profiles_before": 1,
        "session_profiles_after": 5,
        "session_work_events_after": 8,
        "session_work_events_fts_after": 8,
        "session_phases_after": 3,
        "threads_after": 2,
        "threads_fts_after": 2,
        "session_tag_rollups_after": 4,
    }
