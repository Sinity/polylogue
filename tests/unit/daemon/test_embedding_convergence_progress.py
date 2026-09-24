from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

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


def test_periodic_embedding_backlog_waits_for_watcher_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """The periodic production route invokes its retained composition only after catch-up.

    Anti-vacuity: removing the catch-up wait starts the callback before its
    gate is set, concurrently with ingest's writer-owned rebuild work.
    """

    calls: list[tuple[str, ...] | None] = []

    async def converge(scope: Sequence[str] | None) -> embedding_owner.EmbeddingConvergenceResult:
        calls.append(None if scope is None else tuple(scope))
        raise asyncio.CancelledError

    async def exercise() -> None:
        gate = asyncio.Event()
        monkeypatch.setattr(embedding_backlog, "EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS", 0)
        task = asyncio.create_task(
            embedding_backlog.periodic_embedding_backlog_check(watcher_registered=gate, converge=converge)
        )
        await asyncio.sleep(0)
        assert calls == []
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert calls == [None]


def test_embedding_admission_bounds_no_timeout_wait_on_owner_loop_liveness() -> None:
    """The surviving embedding bridge call cannot strand its caller forever.

    The embedding owner deliberately passes ``None`` as its wait budget: an
    admitted publication must settle rather than be cancelled by a caller
    timeout.  The owner loop stopping is the only bounded failure, and must
    report an indeterminate typed outcome while the admitted body still
    settles on its own thread.

    Anti-vacuity: changing ``DaemonEmbeddingAdmission`` back to a raw
    ``future.result(timeout=None)`` leaves ``caller`` alive after the loop
    stops and makes this test fail.
    """
    from polylogue.daemon.write_coordinator import DaemonWriterOwnerLoopStopped

    loop = asyncio.new_event_loop()
    ready = threading.Event()
    admission_holder: list[embedding_owner.DaemonEmbeddingAdmission] = []

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        coordinator = DaemonWriteCoordinator()
        bridge = DaemonWriteThreadBridge(coordinator, loop, timeout=0.05)

        async def compose() -> None:
            admission_holder.append(embedding_owner.DaemonEmbeddingAdmission(bridge, loop))
            ready.set()

        loop.run_until_complete(compose())
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    assert ready.wait(timeout=5.0)

    started = threading.Event()
    allow_settle = threading.Event()
    settled: list[str] = []
    raised: list[BaseException] = []

    def publish() -> str:
        started.set()
        assert allow_settle.wait(timeout=5.0)
        settled.append("published")
        return "receipt"

    def caller_body() -> None:
        try:
            admission_holder[0]("embedding.publish", publish)
        except BaseException as exc:
            raised.append(exc)

    caller = threading.Thread(target=caller_body, daemon=True)
    caller.start()
    try:
        assert started.wait(timeout=5.0)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5.0)
        assert not loop_thread.is_alive()

        caller.join(timeout=5.0)
        assert not caller.is_alive(), "embedding admission outlived its owner loop"
        assert len(raised) == 1
        assert isinstance(raised[0], DaemonWriterOwnerLoopStopped)
        assert "may still be in flight" in str(raised[0])

        allow_settle.set()
        for _ in range(500):
            if settled:
                break
            time.sleep(0.01)
        assert settled == ["published"]
    finally:
        allow_settle.set()
        caller.join(timeout=5.0)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5.0)


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


def test_scoped_foreground_convergence_honors_the_monthly_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The scoped (ingest-foreground) call obeys the same cap as the periodic pass.

    Foreground ingest embedding runs through the very composition the periodic
    backlog uses (``compose_embedding_convergence``; see
    ``converge_ingest_embeddings`` in ``polylogue/daemon/cli.py`` and
    ``periodic_embedding_backlog_check``), so a scope argument cannot buy work
    the monthly cap has already spent (polylogue-liwst).

    Anti-vacuity: a foreground path that skips the cap check goes on to build a
    frame from the stub adapter below and raises instead of returning the
    ``monthly_cost_cap`` deferral.
    """

    class _StubAdapter:
        domain = "embedding"

    async def exercise() -> embedding_owner.EmbeddingConvergenceResult:
        coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
        bridge = DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop())
        monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda: _EmbeddingConfig())
        monkeypatch.setattr(
            "polylogue.operations.embedding_derivation.make_embedding_derivation",
            lambda *_args, **_kwargs: _StubAdapter(),
        )
        # The month's estimated spend already exceeds embedding_max_cost_usd.
        monkeypatch.setattr(
            embedding_backlog,
            "_archive_embedding_catchup_estimated_cost_this_month",
            lambda _ops_db: 999.0,
        )
        composed = embedding_owner.compose_embedding_convergence(
            tmp_path / "index.db",
            compute_adapter=BoundedComputeAdapter(max_workers=1),
            write_bridge=bridge,
        )
        # A scoped call is exactly what ingest foreground convergence makes.
        return await composed(["claude-code-session:s1"])

    result = asyncio.run(exercise())
    assert result.deferred_reason == "monthly_cost_cap"
    assert result.report is None


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


def test_catchup_receipt_recovery_sweeps_every_non_terminal_status() -> None:
    """Startup recovery is defined as the complement of the terminal statuses.

    A catch-up receipt only ever transitions out of ``running``, so a row
    parked in any other non-terminal state is permanent debt unless the
    startup sweep covers it (polylogue-f7bf9).

    Anti-vacuity: replacing the derived sweep set with a hardcoded IN-list --
    or classifying a new non-terminal ``OperationStatus`` member as terminal --
    leaves that member out of ``UNFINISHED_CATCHUP_RECEIPT_STATUSES`` and this
    comparison fails.
    """
    from polylogue.core.enums import OperationStatus

    expected = {
        status.value
        for status in OperationStatus
        if status.value not in embedding_backlog.TERMINAL_CATCHUP_RECEIPT_STATUSES
    }
    assert set(embedding_backlog.UNFINISHED_CATCHUP_RECEIPT_STATUSES) == expected
    assert OperationStatus.RUNNING.value in embedding_backlog.UNFINISHED_CATCHUP_RECEIPT_STATUSES
    assert OperationStatus.COMPLETED.value not in embedding_backlog.UNFINISHED_CATCHUP_RECEIPT_STATUSES


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


def test_embedding_derivation_emits_intermediate_progress_before_terminal_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The embedding seam exposes work while a pass is still computing.

    Anti-vacuity: removing the progress callback from ``make_embedding_derivation``
    leaves the adapter's quiet observation point silent, so the event list stays
    empty even though the derivation was constructed successfully.
    """

    class _Provider:
        model = "voyage-4"
        dimension = 1024

        def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
            del texts, input_type
            return []

    captured: dict[str, object] = {}

    class _Adapter:
        domain = "embeddings"
        recipe_version = "recipe"

    def fake_adapter(*args: object, **kwargs: object) -> _Adapter:
        del args
        captured.update(kwargs)
        return _Adapter()

    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda **_: _Provider())
    monkeypatch.setattr("polylogue.operations.embedding_derivation.EmbeddingDerivationAdapter", fake_adapter)
    monkeypatch.setattr(
        "polylogue.operations.embedding_derivation.resolve_active_index_path",
        lambda _root: tmp_path / "index.db",
    )
    events: list[dict[str, object]] = []
    from polylogue.operations.embedding_derivation import make_embedding_derivation

    adapter = make_embedding_derivation(
        tmp_path / "index.db",
        archive_root=tmp_path,
        voyage_api_key="key",
        model="voyage-4",
        dimension=1024,
        reserve=lambda _actor, fn: fn(),
        progress_callback=cast("Callable[[Mapping[str, object]], None]", events.append),
    )
    assert adapter is not None
    quiet = captured["quiet"]
    assert callable(quiet)
    assert quiet(None, "message:m1") is False
    assert events == [{"state": "started", "message_id": "m1", "session_id": None}]


def test_embedding_progress_ring_is_request_scoped_monotone_and_signals_overflow() -> None:
    """The operation-await buffer bounds observations without losing gap evidence."""
    from polylogue.daemon.operation_runtime import DaemonOperationRuntime, _Exchange
    from polylogue.operations.daemon_protocol import DaemonOperationRequest

    runtime = object.__new__(DaemonOperationRuntime)
    runtime._condition = threading.Condition()
    runtime._exchanges = {}
    first = DaemonOperationRequest("maintenance.embeddings.backfill", {}, request_id="embedding-first")
    second = DaemonOperationRequest("maintenance.embeddings.backfill", {}, request_id="embedding-second")
    runtime._exchanges[str(first.request_id)] = _Exchange(first, cast(Any, None), 0.0, 0)
    runtime._exchanges[str(second.request_id)] = _Exchange(second, cast(Any, None), 0.0, 0)

    for ordinal in range(70):
        runtime.emit_progress(first, {"state": "started", "ordinal": ordinal})
    runtime.emit_progress(second, {"state": "started", "ordinal": 0})

    first_state = runtime._progress_state(runtime._exchanges[str(first.request_id)], 0)
    second_state = runtime._progress_state(runtime._exchanges[str(second.request_id)], 0)
    assert first_state["progress_sequence"] == 70
    assert first_state["progress_gap"] == {"from_sequence": 1, "to_sequence": 6}
    first_events = cast(list[dict[str, object]], first_state["progress_events"])
    assert [frame["sequence"] for frame in first_events] == list(range(7, 71))
    assert second_state["progress_sequence"] == 1
    assert second_state["progress_gap"] is None
