from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from polylogue.daemon import embedding_backlog, embedding_owner
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.status import format_daemon_status_lines
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge


class _EmbeddingConfig:
    embedding_enabled = True
    voyage_api_key = "pa-test"
    embedding_model = "voyage-4"
    embedding_dimension = 1024

    def get(self, key: str, default: object = None) -> object:
        return {"voyage_api_key": self.voyage_api_key, "embedding_max_cost_usd": 5.0}.get(key, default)


def test_periodic_embedding_backlog_waits_for_catch_up_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """The periodic production route invokes its retained composition only after catch-up.

    Anti-vacuity: removing the catch-up wait starts the callback before its
    gate is set, concurrently with ingest's writer-owned rebuild work.
    """

    calls: list[tuple[str, ...] | None] = []

    async def converge(scope: tuple[str, ...] | None) -> embedding_owner.EmbeddingConvergenceResult:
        calls.append(scope)
        raise asyncio.CancelledError

    async def exercise() -> None:
        gate = asyncio.Event()
        monkeypatch.setattr(embedding_backlog, "EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS", 0)
        task = asyncio.create_task(
            embedding_backlog.periodic_embedding_backlog_check(catch_up_complete=gate, converge=converge)
        )
        await asyncio.sleep(0)
        assert calls == []
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert calls == [None]


def test_embedding_composition_defers_disabled_without_constructing_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabled configuration is policy, so it cannot spend or create a provider client.

    Anti-vacuity: constructing the provider before the disabled check increments
    ``provider_calls`` despite a configuration that expressly disables paid work.
    """

    class _DisabledConfig(_EmbeddingConfig):
        embedding_enabled = False

    provider_calls: list[object] = []

    def create_provider(**kwargs: object) -> object:
        provider_calls.append(kwargs)
        raise AssertionError("disabled embedding convergence must not build a provider")

    async def exercise() -> embedding_owner.EmbeddingConvergenceResult:
        coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
        bridge = DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop())
        monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda: _DisabledConfig())
        monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", create_provider)
        composed = embedding_owner.compose_embedding_convergence(
            tmp_path / "index.db",
            compute_adapter=BoundedComputeAdapter(max_workers=1),
            write_bridge=bridge,
        )
        return await composed(None)

    result = asyncio.run(exercise())
    assert result.deferred_reason == "disabled"
    assert provider_calls == []


def test_embedding_startup_marks_running_catchup_receipts_interrupted(tmp_path: Path) -> None:
    from polylogue.core.enums import OperationStatus
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.ops_write import (
        list_embedding_catchup_runs,
        upsert_embedding_catchup_run,
    )
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    ops_db = tmp_path / "ops.db"
    with sqlite3.connect(ops_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.OPS)
        upsert_embedding_catchup_run(
            conn,
            run_id="unfinished",
            status=OperationStatus.RUNNING,
            started_at_ms=1,
        )

    assert embedding_backlog.recover_embedding_catchup_receipts(tmp_path) == 1
    with sqlite3.connect(ops_db) as conn:
        (run,) = list_embedding_catchup_runs(conn)
    assert run.status == "interrupted"
    assert run.finished_at_ms is not None


def test_daemon_status_lines_include_latest_embedding_catchup() -> None:
    lines = format_daemon_status_lines(
        {
            "embedding_readiness": {
                "embedding_enabled": True,
                "embedding_coverage_percent": 12.5,
                "embedding_pending_count": 10,
                "embedding_pending_message_count": 200,
                "embedding_stale_count": 0,
                "embedding_failure_count": 0,
                "embedding_estimated_cost_usd": 0.02,
                "embedding_model": "voyage-4",
                "embedding_dimension": 1024,
                "embedding_latest_catchup_run": {
                    "status": "running",
                    "processed_sessions": 3,
                    "planned_sessions": 10,
                    "embedded_messages": 42,
                },
            }
        }
    )
    assert "  latest catch-up: running, 3/10 convs, 42 msgs embedded" in lines
