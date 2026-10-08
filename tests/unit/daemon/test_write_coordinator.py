"""Deterministic proofs for intra-daemon archive write serialization."""

from __future__ import annotations

import asyncio
import io
import json
import os
import select
import signal
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from builtins import BaseExceptionGroup
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest

from polylogue import logging as plog
from polylogue.archive.write_gateway import ArchiveWriteGateway
from polylogue.core.write_lease import arm_write_lease_enforcement
from polylogue.daemon import write_coordinator as write_coordinator_module
from polylogue.daemon.write_coordinator import (
    _DETACHED_WRITER_FAILURE_OVERFLOW_ACTOR,
    _MAX_DETACHED_WRITER_FAILURE_ACTOR_LENGTH,
    _MAX_DETACHED_WRITER_FAILURE_ACTORS,
    DaemonWriteCoordinator,
    DaemonWriteEvent,
    DaemonWriterSettlementError,
    DaemonWriteThreadBridge,
    _actor_priority,
    _PriorityGate,
    daemon_write_telemetry_payload,
)
from polylogue.sources.live.cold_build import ColdBuildGeneration, active_index_generation_is_empty
from polylogue.sources.live.watcher import WatchSource
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.archive_custody_probe import archive_custody_available
from tests.infra.sqlite_cursor_settlement import (
    SettlementConnection,
    arm_settlement,
    native_settlement_connections,  # noqa: F401  # Pytest fixture discovery.
)


def test_actor_priority_classifies_bulk_ingest_below_everything_else() -> None:
    assert _actor_priority("watcher.catch_up.chunk") == 1
    assert _actor_priority("watcher.live_batch") == 1
    assert _actor_priority("maintenance.fts_merge") == 0
    assert _actor_priority("maintenance.wal_checkpoint") == 0
    assert _actor_priority("startup.fts_automerge") == 0
    assert _actor_priority("daemon.lifecycle.heartbeat") == 0
    # Exact "watcher" (no trailing segment) is not the bulk-ingest convention.
    assert _actor_priority("watcher") == 0


@pytest.mark.asyncio
async def test_cold_build_lifecycle_writable_opens_stay_under_one_coordinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daemon's generation lifecycle uses the same archive-bound gate."""
    root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    coordinator = DaemonWriteCoordinator(archive_root=root)
    generation: ColdBuildGeneration | None = None
    try:
        with arm_write_lease_enforcement(process_wide=True):
            assert (
                await coordinator.run_sync(
                    "daemon.cold_build.probe",
                    active_index_generation_is_empty,
                    root,
                )
                is True
            )
            generation = await coordinator.run_sync(
                "daemon.cold_build.begin",
                ColdBuildGeneration.begin,
                root,
                reason="test",
                observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", root / "absent-source"),)),
            )
            assert (
                await coordinator.run_sync(
                    "daemon.cold_build.session_count",
                    generation.session_count,
                )
                == 0
            )
            await coordinator.run_sync("daemon.cold_build.discard", generation.discard)
    finally:
        await coordinator.shutdown(timeout=1.0)
    assert generation is not None
    assert generation.settled


@pytest.mark.asyncio
async def test_priority_gate_wake_survives_cancellation_before_resume() -> None:
    """Reproduce the exact race a stdlib-Lock-shaped gate must survive: a
    waiter is woken (its future gets a result) but is cancelled before it
    resumes past ``await``. The grant must be forwarded, not dropped."""
    gate = _PriorityGate()
    await gate.acquire(0)  # first caller takes the gate synchronously

    async def waiter() -> None:
        await gate.acquire(0)

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)  # let it enqueue
    assert gate.locked

    gate.release()  # wakes the queued waiter's future (call_soon, not yet resumed)
    task.cancel()  # cancel before the event loop resumes the woken task
    with pytest.raises(asyncio.CancelledError):
        await task

    # The grant must not be stranded: a fresh acquirer still succeeds.
    successor = asyncio.create_task(gate.acquire(0))
    await asyncio.wait_for(successor, timeout=1.0)
    assert gate.locked


@pytest.mark.asyncio
async def test_real_sqlite_writer_collision_is_eliminated_without_sleep_timing(tmp_path: Path) -> None:
    """Reproduce the pre-fix lock, then prove the coordinator removes it."""
    db = tmp_path / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE writes (actor TEXT NOT NULL)")

    def hold_writer(entered: threading.Event, release: threading.Event, actor: str) -> None:
        with sqlite3.connect(db, timeout=0) as conn:
            conn.execute("BEGIN IMMEDIATE")
            entered.set()
            release.wait()
            conn.execute("INSERT INTO writes VALUES (?)", (actor,))
            conn.commit()

    def write_now(actor: str) -> None:
        with sqlite3.connect(db, timeout=0) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO writes VALUES (?)", (actor,))
            conn.commit()

    # Control: the two independent daemon-style connections deterministically
    # collide while the first actor owns SQLite's write transaction.
    direct_entered = threading.Event()
    release_direct = threading.Event()
    direct = asyncio.create_task(asyncio.to_thread(hold_writer, direct_entered, release_direct, "direct"))
    assert await asyncio.to_thread(direct_entered.wait)
    try:
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            await asyncio.to_thread(write_now, "colliding-watcher")
    finally:
        release_direct.set()
        await direct

    watcher_queued = asyncio.Event()

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "queued" and event.actor == "watcher.live_ingest":
            watcher_queued.set()

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path, observer=observe)
    coordinated_entered = threading.Event()
    release_coordinated = threading.Event()
    maintenance = asyncio.create_task(
        coordinator.run_sync(
            "maintenance.raw_materialization",
            hold_writer,
            coordinated_entered,
            release_coordinated,
            "maintenance",
        )
    )
    assert await asyncio.to_thread(coordinated_entered.wait)
    watcher = asyncio.create_task(coordinator.run_sync("watcher.live_ingest", write_now, "watcher"))
    await watcher_queued.wait()

    release_coordinated.set()
    await asyncio.gather(maintenance, watcher)

    with sqlite3.connect(db) as conn:
        actors = [str(row[0]) for row in conn.execute("SELECT actor FROM writes ORDER BY rowid")]
    assert actors == ["direct", "maintenance", "watcher"]


@pytest.mark.asyncio
async def test_coordinator_serializes_fifo_without_writer_overlap(tmp_path: Path) -> None:
    queued = {actor: asyncio.Event() for actor in ("watcher", "raw", "embedding")}
    events: list[DaemonWriteEvent] = []

    def observe(event: DaemonWriteEvent) -> None:
        events.append(event)
        if event.phase == "queued":
            queued[event.actor].set()

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path, observer=observe)
    release_watcher = asyncio.Event()
    watcher_entered = asyncio.Event()
    call_order: list[str] = []
    active = 0
    max_active = 0

    async def writer(actor: str, release: asyncio.Event | None = None) -> str:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        call_order.append(actor)
        if actor == "watcher":
            watcher_entered.set()
        if release is not None:
            await release.wait()
        active -= 1
        return actor

    watcher = asyncio.create_task(coordinator.run("watcher", lambda: writer("watcher", release_watcher)))
    await watcher_entered.wait()
    raw = asyncio.create_task(coordinator.run("raw", lambda: writer("raw")))
    embedding = asyncio.create_task(coordinator.run("embedding", lambda: writer("embedding")))
    await asyncio.gather(queued["raw"].wait(), queued["embedding"].wait())

    assert coordinator.snapshot().active_actor == "watcher"
    assert coordinator.snapshot().queued_actors == ("raw", "embedding")
    release_watcher.set()

    results = await asyncio.gather(watcher, raw, embedding)
    assert tuple(results) == ("watcher", "raw", "embedding")
    assert call_order == ["watcher", "raw", "embedding"]
    assert max_active == 1
    released = [event for event in events if event.phase == "released"]
    assert [event.actor for event in released] == call_order
    assert all(event.wait_seconds is not None and event.hold_seconds is not None for event in released)
    assert all(event.outcome == "success" for event in released)


@pytest.mark.asyncio
async def test_maintenance_actor_jumps_ahead_of_queued_bulk_ingest_actors(tmp_path: Path) -> None:
    """polylogue-de2a: a continuously-refilling watcher backlog must not starve
    periodic maintenance. A queued ``maintenance.*``/other actor is admitted
    before an earlier-queued ``watcher.*`` actor, though never before an
    already-admitted one (queue-admission fairness only, not preemption)."""
    queued = {
        actor: asyncio.Event()
        for actor in ("watcher.catch_up.chunk", "watcher.catch_up.chunk.2", "maintenance.fts_merge")
    }
    call_order: list[str] = []

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "queued" and event.actor in queued:
            queued[event.actor].set()

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path, observer=observe)
    release_owner = asyncio.Event()
    owner_entered = asyncio.Event()

    async def owner() -> None:
        owner_entered.set()
        await release_owner.wait()

    async def tracked(actor: str) -> str:
        call_order.append(actor)
        return actor

    owner_task = asyncio.create_task(coordinator.run("owner", owner))
    await owner_entered.wait()

    # Two watcher chunks queue first (as a continuously-refilling backlog
    # would), then a maintenance actor queues last.
    first_watcher = asyncio.create_task(
        coordinator.run("watcher.catch_up.chunk", lambda: tracked("watcher.catch_up.chunk"))
    )
    await queued["watcher.catch_up.chunk"].wait()
    second_watcher = asyncio.create_task(
        coordinator.run("watcher.catch_up.chunk.2", lambda: tracked("watcher.catch_up.chunk.2"))
    )
    await queued["watcher.catch_up.chunk.2"].wait()
    maintenance = asyncio.create_task(
        coordinator.run("maintenance.fts_merge", lambda: tracked("maintenance.fts_merge"))
    )
    await queued["maintenance.fts_merge"].wait()

    assert coordinator.snapshot().active_actor == "owner"
    release_owner.set()
    await asyncio.gather(owner_task, first_watcher, second_watcher, maintenance)

    # Maintenance queued last but is admitted before either watcher waiter;
    # among the two same-priority watcher waiters, arrival order still wins.
    assert call_order == ["maintenance.fts_merge", "watcher.catch_up.chunk", "watcher.catch_up.chunk.2"]


@pytest.mark.asyncio
async def test_cancelled_priority_waiter_does_not_strand_the_grant(tmp_path: Path) -> None:
    """Cancelling a still-queued higher-priority waiter must not drop the
    gate: the next (lower-priority) waiter still gets admitted afterward."""
    queued_maintenance = asyncio.Event()
    queued_watcher = asyncio.Event()

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "queued" and event.actor == "maintenance.wal_checkpoint":
            queued_maintenance.set()
        if event.phase == "queued" and event.actor == "watcher.catch_up.chunk":
            queued_watcher.set()

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path, observer=observe)
    release_owner = asyncio.Event()
    owner_entered = asyncio.Event()

    async def owner() -> None:
        owner_entered.set()
        await release_owner.wait()

    owner_task = asyncio.create_task(coordinator.run("owner", owner))
    await owner_entered.wait()

    maintenance_task = asyncio.create_task(coordinator.run("maintenance.wal_checkpoint", _unexpected_operation))
    await queued_maintenance.wait()
    watcher_task = asyncio.create_task(coordinator.run("watcher.catch_up.chunk", _return_ready))
    await queued_watcher.wait()

    # Cancel the higher-priority waiter while it is still queued (owner has
    # not released yet), then release: the lower-priority watcher waiter must
    # still be admitted, and the gate must not be left stuck.
    maintenance_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await maintenance_task
    release_owner.set()
    await owner_task
    assert await watcher_task == "ready"
    assert await coordinator.run("next", _return_ready) == "ready"


@pytest.mark.asyncio
async def test_waiting_cancellation_removes_actor_without_deadlock(tmp_path: Path) -> None:
    raw_queued = asyncio.Event()

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "queued" and event.actor == "raw":
            raw_queued.set()

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path, observer=observe)
    release_watcher = asyncio.Event()
    watcher_entered = asyncio.Event()

    async def watcher_operation() -> None:
        watcher_entered.set()
        await release_watcher.wait()

    watcher = asyncio.create_task(coordinator.run("watcher", watcher_operation))
    await watcher_entered.wait()
    raw = asyncio.create_task(coordinator.run("raw", _unexpected_operation))
    await raw_queued.wait()
    raw.cancel()
    with pytest.raises(asyncio.CancelledError):
        await raw

    assert coordinator.snapshot().queued_actors == ()
    release_watcher.set()
    await watcher
    assert await coordinator.run("next", _return_ready) == "ready"


@pytest.mark.asyncio
async def test_sync_writer_cancellation_holds_gate_until_thread_finishes(tmp_path: Path) -> None:
    released = asyncio.Event()
    events: list[DaemonWriteEvent] = []

    def observe(event: DaemonWriteEvent) -> None:
        events.append(event)
        if event.phase == "released" and event.actor == "raw":
            released.set()

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path, observer=observe)
    worker_started = threading.Event()
    allow_worker_finish = threading.Event()

    def raw_writer() -> None:
        worker_started.set()
        assert allow_worker_finish.wait(timeout=1.0)

    task = asyncio.create_task(coordinator.run_sync("raw", raw_writer))
    assert await asyncio.to_thread(worker_started.wait, 1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.1)

    assert coordinator.snapshot().active_actor == "raw"
    assert not released.is_set()
    successor_entered = asyncio.Event()

    async def successor_writer() -> None:
        successor_entered.set()

    successor = asyncio.create_task(coordinator.run("successor", successor_writer))
    while coordinator.snapshot().queued_actors != ("successor",):
        await asyncio.sleep(0)
    assert not successor_entered.is_set()
    allow_worker_finish.set()
    await successor
    assert await coordinator.shutdown(timeout=1.0)
    assert successor_entered.is_set()
    assert released.is_set()
    assert coordinator.snapshot().active_actor is None
    raw_release = next(event for event in events if event.phase == "released" and event.actor == "raw")
    # The caller disconnected, but the admitted worker returned normally. The
    # release receipt must preserve that terminal success rather than
    # reporting the caller's cancellation as the writer's outcome.
    assert raw_release.outcome == "success"


@pytest.mark.asyncio
async def test_transaction_receipt_distinguishes_commit_from_rollback_after_disconnect(tmp_path: Path) -> None:
    """Terminal evidence comes from the admitted transaction, not its caller.

    A disconnected caller must not relabel a transaction that later commits.
    A transaction that rolls back and raises keeps the error evidence instead.
    The SQLite rows make both outcomes deterministic and observable after the
    coordinator has released ownership.
    """
    db = tmp_path / "terminal-evidence.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE writes (value TEXT NOT NULL)")
        conn.commit()

    events: list[DaemonWriteEvent] = []
    commit_released = asyncio.Event()

    def observe(event: DaemonWriteEvent) -> None:
        events.append(event)
        if event.actor == "transaction.commit" and event.phase == "released":
            commit_released.set()

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path, observer=observe)
    worker_started = threading.Event()
    allow_commit = threading.Event()

    def commit_writer() -> None:
        with sqlite3.connect(db) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO writes VALUES ('committed')")
            worker_started.set()
            assert allow_commit.wait(timeout=1.0)
            conn.commit()

    caller = asyncio.create_task(coordinator.run_sync("transaction.commit", commit_writer))
    assert await asyncio.to_thread(worker_started.wait, 1.0)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, timeout=0.1)
    allow_commit.set()
    await commit_released.wait()

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT value FROM writes").fetchall() == [("committed",)]
    commit_release = next(
        event for event in events if event.actor == "transaction.commit" and event.phase == "released"
    )
    assert commit_release.outcome == "success"

    def rollback_writer() -> None:
        with sqlite3.connect(db) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO writes VALUES ('rolled-back')")
            conn.rollback()
        raise RuntimeError("transaction rolled back")

    with pytest.raises(RuntimeError, match="transaction rolled back"):
        await coordinator.run_sync("transaction.rollback", rollback_writer)

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT value FROM writes").fetchall() == [("committed",)]
    rollback_release = next(
        event for event in events if event.actor == "transaction.rollback" and event.phase == "released"
    )
    assert rollback_release.outcome == "error"
    assert await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_child_task_cannot_inherit_reentrant_write_lease(tmp_path: Path) -> None:
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)

    async def parent_writer() -> str:
        child = asyncio.create_task(coordinator.run("child", _return_ready))
        with pytest.raises(RuntimeError, match="inherited by a child task"):
            await child
        return await coordinator.run("same-task", _return_ready)

    assert await coordinator.run("parent", parent_writer) == "ready"
    released = coordinator.snapshot().last_event
    assert released is not None
    assert released.actor == "parent"
    assert released.outcome == "success"


@pytest.mark.asyncio
async def test_cancelled_queued_writer_never_runs(tmp_path: Path) -> None:
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    child_called = False

    async def owner() -> None:
        entered.set()
        await release.wait()

    async def queued() -> None:
        nonlocal child_called
        child_called = True

    owner_task = asyncio.create_task(coordinator.run("owner", owner))
    await entered.wait()
    queued_task = asyncio.create_task(coordinator.run("queued", queued))
    while coordinator.snapshot().queued_actors != ("queued",):
        await asyncio.sleep(0)
    queued_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued_task
    release.set()
    await owner_task
    assert not child_called


@pytest.mark.asyncio
async def test_shutdown_is_bounded_without_releasing_active_sync_writer(tmp_path: Path) -> None:
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    worker_started = threading.Event()
    worker_release = threading.Event()

    def writer() -> None:
        worker_started.set()
        worker_release.wait()

    task = asyncio.create_task(coordinator.run_sync("sync", writer))
    assert await asyncio.to_thread(worker_started.wait, 1.0)
    assert not await coordinator.shutdown(timeout=0.01)
    assert coordinator.snapshot().active_actor == "sync"
    with pytest.raises(RuntimeError, match="shutting down"):
        await coordinator.run("late", _return_ready)
    worker_release.set()
    await task
    assert await coordinator.shutdown(timeout=0.1)


def test_stuck_sync_writer_cannot_pin_process_exit(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import asyncio
        import sys
        from pathlib import Path
        import contextlib
        import threading

        from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

        async def main() -> None:
            coordinator = DaemonWriteCoordinator(archive_root=Path(sys.argv[1]))
            started = threading.Event()

            def writer() -> None:
                started.set()
                threading.Event().wait()

            caller = asyncio.create_task(coordinator.run_sync("stuck", writer))
            while not started.is_set():
                await asyncio.sleep(0.001)
            caller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await caller
            assert await coordinator.shutdown(timeout=0.01) is False

        asyncio.run(main())
        """
    )

    # polylogue-es7b: bumped from 5.0s -- this module's cold-import cost alone
    # can run several seconds under concurrent xdist workers (each spawning
    # its own subprocess simultaneously), which made this timeout marginal
    # once a third subprocess-spawning test landed in this file.
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        timeout=20.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("helper_name", "writer_name", "helper_args"),
    [
        (
            "_run_startup_embedding_lifecycle",
            "_ensure_embedding_lifecycle_startup_sync",
            "coordinator, Path(sys.argv[1])",
        ),
    ],
)
def test_real_startup_writer_routes_cannot_pin_process_exit(
    tmp_path: Path, helper_name: str, writer_name: str, helper_args: str
) -> None:
    script = textwrap.dedent(
        f"""
        import asyncio
        import sys
        from pathlib import Path
        import contextlib
        import threading

        from polylogue.daemon import cli
        from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

        async def main() -> None:
            coordinator = DaemonWriteCoordinator(archive_root=Path(sys.argv[1]))
            started = threading.Event()

            def writer(*args) -> None:
                started.set()
                threading.Event().wait()

            setattr(cli, {writer_name!r}, writer)
            helper = getattr(cli, {helper_name!r})
            caller = asyncio.create_task(helper({helper_args}))
            while not started.is_set():
                await asyncio.sleep(0.001)
            caller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await caller
            assert await coordinator.shutdown(timeout=0.01) is False

        asyncio.run(main())
        """
    )

    # polylogue-es7b: bumped from 5.0s -- this module's cold-import cost alone
    # can run several seconds under concurrent xdist workers (each spawning
    # its own subprocess simultaneously), which made this timeout marginal
    # once a third subprocess-spawning test landed in this file.
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        timeout=20.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.asyncio
async def test_operational_telemetry_reports_actor_queue_wait_and_hold(tmp_path: Path) -> None:
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def owner() -> None:
        entered.set()
        await release.wait()

    owner_task = asyncio.create_task(coordinator.run("maintenance", owner))
    await entered.wait()
    queued_task = asyncio.create_task(coordinator.run("watcher", _return_ready))
    while coordinator.snapshot().queued_actors != ("watcher",):
        await asyncio.sleep(0)
    payload = daemon_write_telemetry_payload()
    assert payload["active_actor"] == "maintenance"
    assert payload["queued_actors"] == ["watcher"]
    assert payload["queue_depth"] == 1
    release.set()
    await asyncio.gather(owner_task, queued_task)
    payload = daemon_write_telemetry_payload()
    assert payload["active_actor"] is None
    assert payload["queue_depth"] == 0
    event = payload["last_event"]
    assert isinstance(event, dict)
    assert event["actor"] == "watcher"
    assert isinstance(event["wait_seconds"], float)
    assert isinstance(event["hold_seconds"], float)


@pytest.mark.asyncio
async def test_detached_writer_failure_increments_lifetime_counter(tmp_path: Path) -> None:
    """polylogue-es7b: a failed writer task's exception previously surfaced only via a log line.

    ``completed()`` (the task's done-callback) must also increment a durable
    daemon-lifetime counter so the failure is observable through the
    coordinator's telemetry snapshot/payload, not only in logs.
    """
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    assert coordinator.snapshot().detached_writer_failures == 0

    async def boom() -> None:
        raise RuntimeError("writer blew up")

    with pytest.raises(RuntimeError, match="writer blew up"):
        await coordinator.run("actor", boom)

    # The done-callback that increments the counter is scheduled via
    # call_soon alongside the outer await's own resumption; give the loop one
    # more tick so ordering between the two doesn't make this test flaky.
    for _ in range(10):
        if coordinator.snapshot().detached_writer_failures:
            break
        await asyncio.sleep(0)

    assert coordinator.snapshot().detached_writer_failures == 1
    assert coordinator.snapshot().detached_writer_failures_by_actor == (("actor", 1),)
    payload = daemon_write_telemetry_payload()
    assert payload["detached_writer_failures"] == 1
    assert payload["detached_writer_failures_by_actor"] == {"actor": 1}

    # A second failure keeps accumulating -- this is a lifetime counter, not
    # a one-shot flag.
    with pytest.raises(RuntimeError, match="writer blew up"):
        await coordinator.run("actor", boom)
    for _ in range(10):
        if coordinator.snapshot().detached_writer_failures == 2:
            break
        await asyncio.sleep(0)
    assert coordinator.snapshot().detached_writer_failures == 2
    assert coordinator.snapshot().detached_writer_failures_by_actor == (("actor", 2),)

    with pytest.raises(RuntimeError, match="writer blew up"):
        await coordinator.run("other-actor", boom)
    for _ in range(10):
        if coordinator.snapshot().detached_writer_failures == 3:
            break
        await asyncio.sleep(0)
    assert coordinator.snapshot().detached_writer_failures_by_actor == (("actor", 2), ("other-actor", 1))


@pytest.mark.asyncio
async def test_sync_writer_failure_propagates_and_releases_gate(tmp_path: Path) -> None:
    """A live-loop worker exception must reach the caller, not become a hang."""
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)

    def boom() -> None:
        raise RuntimeError("sync writer blew up")

    with pytest.raises(RuntimeError, match="sync writer blew up"):
        await coordinator.run_sync("sync-failure", boom)

    assert coordinator.snapshot().active_actor is None
    assert await coordinator.run("successor", _return_ready) == "ready"


@pytest.mark.asyncio
async def test_sync_writer_immediate_result_does_not_poll(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed thread result wakes the loop without a timed poll."""
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    started = threading.Event()
    release = threading.Event()
    poll_delays: list[float] = []
    original_sleep = asyncio.sleep

    async def track_sleep(delay: float) -> None:
        if delay:
            poll_delays.append(delay)
        await original_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", track_sleep)

    def writer() -> str:
        started.set()
        assert release.wait(1.0)
        return "ready"

    task = asyncio.create_task(coordinator.run_sync("immediate", writer))
    while not started.is_set():
        await original_sleep(0)
    release.set()

    assert await task == "ready"
    assert poll_delays == []


@pytest.mark.asyncio
async def test_detached_writer_failure_attribution_is_bounded_and_coalesces_overflow(tmp_path: Path) -> None:
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)

    async def boom() -> None:
        raise RuntimeError("writer blew up")

    for index in range(_MAX_DETACHED_WRITER_FAILURE_ACTORS + 4):
        with pytest.raises(RuntimeError, match="writer blew up"):
            await coordinator.run(f"caller-{index}", boom)
        await asyncio.sleep(0)

    long_actor = "x" * (_MAX_DETACHED_WRITER_FAILURE_ACTOR_LENGTH + 1)
    with pytest.raises(RuntimeError, match="writer blew up"):
        await coordinator.run(long_actor, boom)
    await asyncio.sleep(0)

    attribution = dict(coordinator.snapshot().detached_writer_failures_by_actor)
    assert len(attribution) == _MAX_DETACHED_WRITER_FAILURE_ACTORS
    assert attribution["caller-0"] == 1
    assert attribution["caller-30"] == 1
    assert "caller-31" not in attribution
    assert attribution[_DETACHED_WRITER_FAILURE_OVERFLOW_ACTOR] == 6


@pytest.mark.asyncio
async def test_detached_writer_failure_reserved_labels_cannot_collide_with_overflow(tmp_path: Path) -> None:
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    operation_calls = 0

    async def boom() -> None:
        nonlocal operation_calls
        operation_calls += 1
        raise RuntimeError("writer blew up")

    for index in range(_MAX_DETACHED_WRITER_FAILURE_ACTORS):
        with pytest.raises(RuntimeError, match="writer blew up"):
            await coordinator.run(f"caller-{index}", boom)
        await asyncio.sleep(0)

    for reserved_actor in (_DETACHED_WRITER_FAILURE_OVERFLOW_ACTOR, "<other> (actor)"):
        with pytest.raises(ValueError, match="reserved telemetry label"):
            await coordinator.run(reserved_actor, boom)
    assert operation_calls == _MAX_DETACHED_WRITER_FAILURE_ACTORS

    with pytest.raises(RuntimeError, match="writer blew up"):
        await coordinator.run("caller-overflow", boom)
    await asyncio.sleep(0)

    attribution = dict(coordinator.snapshot().detached_writer_failures_by_actor)
    assert len(attribution) == _MAX_DETACHED_WRITER_FAILURE_ACTORS
    assert attribution["caller-0"] == 1
    assert attribution["caller-30"] == 1
    assert "caller-31" not in attribution
    assert attribution[_DETACHED_WRITER_FAILURE_OVERFLOW_ACTOR] == 2
    assert "<other> (actor)" not in attribution


@pytest.mark.asyncio
async def test_daemon_write_telemetry_payload_isolated_from_nested_map_mutation(tmp_path: Path) -> None:
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)

    async def boom() -> None:
        raise RuntimeError("writer blew up")

    with pytest.raises(RuntimeError, match="writer blew up"):
        await coordinator.run("stable-actor", boom)
    await asyncio.sleep(0)

    payload = daemon_write_telemetry_payload()
    actor_failures = payload["detached_writer_failures_by_actor"]
    assert isinstance(actor_failures, dict)
    actor_failures["forged-actor"] = 99

    refreshed = daemon_write_telemetry_payload()
    refreshed_failures = refreshed["detached_writer_failures_by_actor"]
    assert isinstance(refreshed_failures, dict)
    assert refreshed_failures == {"stable-actor": 1}


@pytest.mark.asyncio
async def test_daemon_write_telemetry_exposes_hold_budget_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The status envelope retains checkpoint/publication hold accounting.

    Anti-vacuity: removing the snapshot counter or the release-event budget
    fields from ``_publish_telemetry`` makes this assertion lose the evidence
    that a writer exceeded its declared bound.
    """
    from polylogue.daemon import write_coordinator as wc

    monkeypatch.setattr(wc, "WRITE_HOLD_BUDGETS_S", {"slow.": 0.0})
    coordinator = wc.DaemonWriteCoordinator(archive_root=tmp_path)

    async def operation() -> None:
        await asyncio.sleep(0.01)

    await coordinator.run("slow.actor", operation)
    payload = wc.daemon_write_telemetry_payload()
    assert payload["over_budget_holds"] == 1
    event = payload["last_event"]
    assert isinstance(event, dict)
    assert event["hold_budget_s"] == 0.0
    assert event["hold_over_budget"] is True


def test_run_in_daemon_thread_logs_instead_of_hanging_when_loop_already_closed(tmp_path: Path) -> None:
    """polylogue-es7b: a worker thread finishing after its loop closed must not hang silently.

    ``_run_in_daemon_thread`` spawns a plain (uncancellable) ``threading.Thread``.
    If the coordinator's event loop closes before that thread finishes --
    the real shape of the hazard is a stuck sync writer that outlives a
    process shutdown that gave up waiting on it (see
    ``test_stuck_sync_writer_cannot_pin_process_exit`` above) -- the worker's
    final ``loop.call_soon_threadsafe`` raises ``RuntimeError`` because the
    loop is closed. Previously this was silently swallowed, leaving the
    original awaiting future unresolved with zero forensic trace. The fix
    emits ``daemon.writer.result_abandoned`` instead of swallowing it
    silently; run in a subprocess because it requires actually closing a real
    event loop out from under a still-running background thread.
    """
    script = textwrap.dedent(
        """
        import asyncio
        import sys
        from pathlib import Path
        import threading

        from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
        from polylogue.logging import configure_events

        # Render events to stderr so the parent can assert on the real sink
        # path, not on an in-process capture the production route bypasses.
        configure_events(fmt="console", level="warning", stream=sys.stderr)

        finish = threading.Event()

        async def main() -> None:
            coordinator = DaemonWriteCoordinator(archive_root=Path(sys.argv[1]))
            started = threading.Event()

            def writer() -> None:
                started.set()
                finish.wait()
                raise RuntimeError("late failure after loop closed")

            asyncio.create_task(coordinator.run_sync("late", writer))
            while not started.is_set():
                await asyncio.sleep(0.001)
            # Return now: asyncio.run() cancels outstanding tasks and closes
            # the loop while ``writer`` is still blocked in its background
            # thread -- the real-world shape of the hazard.

        asyncio.run(main())
        # The loop asyncio.run() owned is now closed. Release the writer
        # thread so it raises and tries to publish onto the closed loop.
        finish.set()
        threading.Event().wait(0.5)
        """
    )

    # A generous timeout: importing ``polylogue.daemon.write_coordinator`` in
    # a cold subprocess dominates this test's wall time far more than the
    # writer/loop-close dance itself does.
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        timeout=20.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    # Anti-vacuity: remove the emit() in _run_in_daemon_thread and this is red.
    assert "daemon.writer.result_abandoned" in completed.stderr, completed.stderr
    assert "reason=event_loop_closed" in completed.stderr, completed.stderr
    assert "error_type=RuntimeError" in completed.stderr, completed.stderr


def test_thread_bridge_serializes_sync_request_bodies_without_overlap(tmp_path: Path) -> None:
    loop = asyncio.new_event_loop()
    loop_ready = threading.Event()
    second_queued = threading.Event()
    coordinator_holder: list[DaemonWriteCoordinator] = []

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "queued" and event.actor == "http.user.marks.post":
            second_queued.set()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        coordinator_holder.append(DaemonWriteCoordinator(archive_root=tmp_path, observer=observe))
        loop.call_soon(loop_ready.set)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    assert loop_ready.wait(timeout=1.0)
    coordinator = coordinator_holder[0]
    bridge = DaemonWriteThreadBridge(coordinator, loop, timeout=1.0)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    order: list[str] = []

    def first_request() -> None:
        with bridge.hold("http.reset"):
            order.append("first")
            first_entered.set()
            assert release_first.wait(timeout=1.0)

    def second_request() -> None:
        with bridge.hold("http.user.marks.post"):
            order.append("second")
            second_entered.set()

    first = threading.Thread(target=first_request)
    second = threading.Thread(target=second_request)
    first.start()
    assert first_entered.wait(timeout=1.0)
    second.start()
    assert second_queued.wait(timeout=1.0)
    assert not second_entered.is_set()
    release_first.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert order == ["first", "second"]

    future = asyncio.run_coroutine_threadsafe(coordinator.shutdown(timeout=1.0), loop)
    assert future.result(timeout=1.0)
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=1.0)


@pytest.mark.uses_real_clock("physical bridge acquisition outlives its client wait budget")
def test_thread_bridge_hold_waits_for_actual_acquisition(tmp_path: Path) -> None:
    loop = asyncio.new_event_loop()
    loop_ready = threading.Event()
    acquired = threading.Event()
    entered = threading.Event()
    unblock_observer = threading.Event()
    coordinator_holder: list[DaemonWriteCoordinator] = []
    errors: list[BaseException] = []

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "acquired" and event.actor == "http.slow-acquisition":
            acquired.set()
            assert unblock_observer.wait(timeout=5.0)

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        coordinator_holder.append(DaemonWriteCoordinator(archive_root=tmp_path, observer=observe))
        loop.call_soon(loop_ready.set)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    assert loop_ready.wait(timeout=5.0)
    coordinator = coordinator_holder[0]
    bridge = DaemonWriteThreadBridge(coordinator, loop, timeout=0.05)

    def request_body() -> None:
        try:
            with bridge.hold("http.slow-acquisition"):
                entered.set()
        except BaseException as exc:
            errors.append(exc)

    request = threading.Thread(target=request_body)
    request.start()
    try:
        assert acquired.wait(timeout=5.0)
        request.join(timeout=0.1)
        assert request.is_alive()
        assert not entered.is_set()
        assert errors == []
        unblock_observer.set()
        request.join(timeout=5.0)
        assert not request.is_alive()
        assert entered.is_set()
        assert errors == []
        successor = asyncio.run_coroutine_threadsafe(coordinator.run("successor", _return_ready), loop)
        assert successor.result(timeout=5.0) == "ready"
    finally:
        unblock_observer.set()
        request.join(timeout=5.0)
        shutdown = asyncio.run_coroutine_threadsafe(coordinator.shutdown(timeout=5.0), loop)
        assert shutdown.result(timeout=5.0)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5.0)


async def _unexpected_operation() -> None:
    raise AssertionError("cancelled queued writer must not enter")


async def _return_ready() -> str:
    return "ready"


def test_thread_bridge_run_sync_uses_the_bridge_default_timeout(tmp_path: Path) -> None:
    """polylogue-ogn1 (#2/#5): the bare ``run_sync`` waits at most the bridge's own timeout.

    A blocking function that outlives the bridge's constructor timeout must
    raise ``TimeoutError`` through ``run_sync`` -- proving the default path
    is genuinely bounded, not merely documented as such.
    """
    loop = asyncio.new_event_loop()
    loop_ready = threading.Event()
    coordinator_holder: list[DaemonWriteCoordinator] = []

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        coordinator_holder.append(DaemonWriteCoordinator(archive_root=tmp_path))
        loop.call_soon(loop_ready.set)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    try:
        assert loop_ready.wait(timeout=1.0)
        coordinator = coordinator_holder[0]
        bridge = DaemonWriteThreadBridge(coordinator, loop, timeout=0.05)

        def slow_write() -> str:
            import time

            time.sleep(0.2)
            return "too-late"

        with pytest.raises(TimeoutError):
            bridge.run_sync("http.slow", slow_write)

        # Let the still-running background write actually finish before
        # tearing down the loop, so the coordinator's task unwinds cleanly
        # instead of being destroyed mid-flight (cosmetic only -- the
        # TimeoutError above is the real assertion).
        shutdown = asyncio.run_coroutine_threadsafe(coordinator.shutdown(timeout=1.0), loop)
        assert shutdown.result(timeout=1.0)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1.0)


def test_thread_bridge_run_sync_with_timeout_overrides_the_bridge_default(tmp_path: Path) -> None:
    """polylogue-ogn1 (#2/#5): a per-call override lets a long operation finish.

    ``run_sync_with_timeout`` must wait up to its own ``timeout`` argument
    instead of the bridge's (shorter) constructor default -- this is the fix
    for rebuild-index's HTTP route, which needs up to 600s while the bridge's
    ordinary request timeout stays a much shorter 30s.
    """
    loop = asyncio.new_event_loop()
    loop_ready = threading.Event()
    coordinator_holder: list[DaemonWriteCoordinator] = []

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        coordinator_holder.append(DaemonWriteCoordinator(archive_root=tmp_path))
        loop.call_soon(loop_ready.set)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    try:
        assert loop_ready.wait(timeout=1.0)
        coordinator = coordinator_holder[0]
        # The bridge's own default timeout is far shorter than the override
        # below -- if the override were ignored, this would raise TimeoutError.
        bridge = DaemonWriteThreadBridge(coordinator, loop, timeout=0.05)

        def slow_write() -> str:
            import time

            time.sleep(0.2)
            return "done"

        result = bridge.run_sync_with_timeout("http.maintenance.rebuild-index", 2.0, slow_write)
        assert result == "done"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1.0)


@pytest.mark.asyncio
async def test_queued_cancellation_never_invokes_admitted_completion_callback(tmp_path: Path) -> None:
    """A queued cancellation has one pre-admission outcome, never a continuation."""
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    callback_tasks: list[asyncio.Task[object]] = []

    async def admitted_operation() -> str:
        entered.set()
        await release.wait()
        return "done"

    async def queued_operation() -> str:
        raise AssertionError("queued cancellation must not execute")

    first = asyncio.create_task(coordinator.run("maintenance.whale", admitted_operation))
    await entered.wait()
    second = asyncio.create_task(
        coordinator.run(
            "maintenance.whale",
            queued_operation,
            on_complete=callback_tasks.append,
        )
    )
    await asyncio.sleep(0)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert callback_tasks == []

    release.set()
    assert await first == "done"
    assert await coordinator.shutdown(timeout=1.0) is True


class TestDeclaredHoldBudgets:
    """A hold that starves other writers is surfaced, not silently absorbed.

    The coordinator cannot abort an operation already inside a SQLite
    transaction, so the budget does not preempt. What it does is make an
    over-long hold impossible to miss: rehearsal-11's Drive catch-up was
    measured at hold_max 18,623 s against a 30 s busy_timeout for every
    non-gated writer, and nothing counted or reported it (polylogue-8qm4k).
    """

    def test_longest_matching_actor_prefix_wins(self) -> None:
        from polylogue.daemon.write_coordinator import write_hold_budget_s

        assert write_hold_budget_s("watcher.catch_up.chunk") == 30.0
        assert write_hold_budget_s("maintenance.drive_catchup") == 120.0
        # An undeclared actor still has a budget rather than an exemption.
        assert write_hold_budget_s("something.undeclared") == 60.0

    def test_an_over_budget_hold_is_flagged_and_counted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anti-vacuity: drop the ``hold_seconds > budget_s`` comparison and the
        released event reports ``hold_over_budget`` False while the counter
        stays at zero -- the state the daemon was in for an 18,623 s hold."""
        from polylogue.daemon import write_coordinator as wc

        monkeypatch.setattr(wc, "WRITE_HOLD_BUDGETS_S", {"slow.": 0.0})

        async def scenario() -> tuple[wc.DaemonWriteEvent, int]:
            coordinator = wc.DaemonWriteCoordinator(archive_root=tmp_path)

            async def operation() -> None:
                await asyncio.sleep(0.01)

            await coordinator.run("slow.actor", operation)
            snapshot = coordinator.snapshot()
            assert snapshot.last_event is not None
            return snapshot.last_event, snapshot.over_budget_holds

        event, count = asyncio.run(scenario())

        assert event.phase == "released"
        assert event.hold_budget_s == 0.0
        assert event.hold_over_budget is True
        assert count == 1

    def test_a_hold_inside_its_budget_is_not_flagged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The budget must not fire on ordinary work, or it reports nothing."""
        from polylogue.daemon import write_coordinator as wc

        monkeypatch.setattr(wc, "WRITE_HOLD_BUDGETS_S", {"quick.": 300.0})

        async def scenario() -> tuple[wc.DaemonWriteEvent, int]:
            coordinator = wc.DaemonWriteCoordinator(archive_root=tmp_path)

            async def operation() -> None:
                return None

            await coordinator.run("quick.actor", operation)
            snapshot = coordinator.snapshot()
            assert snapshot.last_event is not None
            return snapshot.last_event, snapshot.over_budget_holds

        event, count = asyncio.run(scenario())

        assert event.hold_budget_s == 300.0
        assert event.hold_over_budget is False
        assert count == 0

    def test_over_budget_failed_admission_is_flagged_and_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anti-vacuity: hard-coding False hides a slow refusing admission hook."""
        from polylogue.daemon import write_coordinator as wc

        monkeypatch.setattr(wc, "WRITE_HOLD_BUDGETS_S", {"slow.": 0.0})

        async def scenario() -> tuple[wc.DaemonWriteEvent, int]:
            coordinator = wc.DaemonWriteCoordinator(archive_root=tmp_path)

            def refuse_slowly() -> None:
                time.sleep(0.01)
                raise RuntimeError("refused")

            with pytest.raises(RuntimeError, match="refused"):
                await coordinator.run("slow.admission", _return_ready, on_admit=refuse_slowly)
            event = coordinator.snapshot().last_event
            assert event is not None
            return event, coordinator.snapshot().over_budget_holds

        event, count = asyncio.run(scenario())
        assert event.hold_over_budget is True
        assert count == 1

    def test_admitted_work_finishes_past_its_diagnostic_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The declared bound reaches the work that has to respect it.

        Anti-vacuity: drop ``enter_write_hold`` from ``_execute`` and the
        checkpoint sees no hold, so the operation runs to completion and the
        only trace of an over-long hold is the release warning -- the state
        this closes.
        """
        from polylogue.core.write_hold import active_write_hold
        from polylogue.daemon import write_coordinator as wc

        monkeypatch.setattr(wc, "WRITE_HOLD_BUDGETS_S", {"slow.": 0.0})
        reached: list[str] = []

        async def scenario() -> tuple[BaseException | None, wc.DaemonWriteEvent]:
            coordinator = wc.DaemonWriteCoordinator(archive_root=tmp_path)

            def work_item(name: str) -> None:
                assert active_write_hold() is not None
                reached.append(name)

            async def operation() -> None:
                # ``run_sync`` is the production route: the checkpoint runs in
                # a worker thread and must still see its caller's hold.
                await coordinator.run_sync("slow.actor.inner", work_item, "first")
                await coordinator.run_sync("slow.actor.inner", work_item, "second")

            error: BaseException | None = None
            try:
                await coordinator.run("slow.actor", operation)
            except BaseException as exc:
                error = exc
            snapshot = coordinator.snapshot()
            assert snapshot.last_event is not None
            return error, snapshot.last_event

        error, event = asyncio.run(scenario())

        assert error is None
        assert reached == ["first", "second"]
        assert event.hold_over_budget is True

    def test_an_unadmitted_context_has_no_hold_telemetry(self) -> None:
        """Un-gated callers (CLI ingest, focused tests) keep their old shape."""
        from polylogue.core.write_hold import active_write_hold

        assert active_write_hold() is None


@pytest.mark.asyncio
async def test_priority_gate_never_grants_twice_when_a_waiter_cancels_during_handoff() -> None:
    """polylogue-tcear: an in-flight grant is not skipped past.

    ``release()`` completes the head waiter's future before that waiter has
    resumed. A sibling waiter cancelling in that window ran ``_wake_next``,
    which popped the already-done head and granted a SECOND waiter, so two
    ``acquire()`` calls completed against one gate. Anti-vacuity: restore
    pop-and-continue for done-but-not-cancelled futures and ``acquired``
    reaches 2.
    """
    from polylogue.daemon.write_coordinator import _PriorityGate

    gate = _PriorityGate()
    await gate.acquire(0)
    acquired: list[str] = []

    async def waiter(name: str, priority: int) -> None:
        await gate.acquire(priority)
        acquired.append(name)

    second = asyncio.create_task(waiter("second", 1))
    third = asyncio.create_task(waiter("third", 2))
    fourth = asyncio.create_task(waiter("fourth", 3))
    for _ in range(3):
        await asyncio.sleep(0)
    # Cancel the third waiter, then release, before yielding: the third
    # waiter's cancellation handler runs before the second waiter resumes.
    third.cancel()
    gate.release()
    for _ in range(5):
        await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await third
    assert acquired == ["second"]
    assert gate.locked is True
    assert not fourth.done()
    gate.release()
    await fourth
    assert acquired == ["second", "fourth"]
    gate.release()
    await second


def test_gate_release_waits_for_a_delegated_body_that_outlived_its_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller that stops waiting must not hand SQLite to a second writer.

    polylogue-8r4zq AC2. The mutating HTTP route holds the gate around
    ``_sync_run``; when its bounded wait expires the route raises
    ``DaemonMutationIndeterminate`` and unwinds ``hold()`` while the submitted
    body is still adopted and still writing. Releasing the gate on that unwind
    admits a second writer alongside a live one.

    The caller must still be bounded: it hands the hold off to the owner loop
    and returns, rather than waiting for the body it already gave up on.

    Anti-vacuity: delete ``_retain_until_delegation_settles`` from
    ``wait_for_release`` (or make ``retire()`` always answer ``False``) and the
    successor acquires while the delegated body is still inside its adoption,
    turning ``successor_entered.is_set()`` red before ``allow_body`` is set.
    """
    from polylogue.core.write_lease import adopt_write_lease

    monkeypatch.setattr(write_coordinator_module, "_DELEGATION_SETTLEMENT_WARN_S", 0.0)
    log_stream = io.StringIO()
    plog.reset_events()
    plog.configure_events(stream=log_stream, fmt="json", level="debug", bridge_stdlib=False)

    loop = asyncio.new_event_loop()
    loop_ready = threading.Event()
    coordinator_holder: list[DaemonWriteCoordinator] = []

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        coordinator_holder.append(DaemonWriteCoordinator(archive_root=tmp_path))
        loop.call_soon(loop_ready.set)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    try:
        assert loop_ready.wait(timeout=5.0)
        coordinator = coordinator_holder[0]
        # Deliberately shorter than the body below: the caller-side release
        # wait is a client-patience bound and must not become an early release.
        bridge = DaemonWriteThreadBridge(coordinator, loop, timeout=0.05)

        body_adopted = threading.Event()
        allow_body = threading.Event()
        body_left = threading.Event()
        caller_returned = threading.Event()

        def request() -> None:
            with bridge.hold("http.user.annotations.post") as delegation:

                def body() -> None:
                    with adopt_write_lease(delegation):
                        body_adopted.set()
                        assert allow_body.wait(timeout=5.0)
                    body_left.set()

                worker = threading.Thread(target=body, daemon=True)
                worker.start()
                assert body_adopted.wait(timeout=5.0)
                # The route's bounded mutation wait expired here.
            caller_returned.set()

        caller = threading.Thread(target=request, daemon=True)
        caller.start()
        # AC1 stays true: the caller is bounded by its own deadline, not by
        # however long the write it abandoned keeps running.
        assert caller_returned.wait(timeout=5.0)
        caller.join(timeout=5.0)
        assert not caller.is_alive()
        assert not body_left.is_set()

        successor_entered = threading.Event()

        async def successor() -> str:
            successor_entered.set()
            return "entered"

        successor_future = asyncio.run_coroutine_threadsafe(coordinator.run("maintenance.successor", successor), loop)
        # A queued successor stays queued while the abandoned body writes on.
        assert not successor_entered.wait(timeout=0.3)
        assert coordinator.snapshot().active_actor == "http.user.annotations.post"

        allow_body.set()
        assert successor_future.result(timeout=5.0) == "entered"
        assert body_left.is_set()
        records = [json.loads(line) for line in log_stream.getvalue().splitlines()]
        unsettled = [record for record in records if record.get("event") == "daemon.writer.delegation_unsettled"]
        assert len(unsettled) == 1
        assert isinstance(unsettled[0].get("wait_ms"), (int, float))
        assert not any(
            record.get("event") == "log.field_rejected" and record.get("field") == "waited_ms" for record in records
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5.0)
        plog.reset_events()


def test_unbounded_bridge_wait_ends_when_its_owner_loop_stops(tmp_path: Path) -> None:
    """The no-timeout bridge is bound on owner-loop liveness, not on a clock.

    polylogue-8r4zq AC5. ``run_sync_with_timeout(actor, None, fn)`` is the
    deliberate no-timeout bridge used by derivation publication, embedding
    phases and operation writes: cancelling the writer in a timeout handler is
    the bug the AC forbids. But if the owner loop stops without closing, the
    admitted operation can never deliver its receipt through that loop, and the
    caller waited forever.

    Anti-vacuity: restore ``future.result(timeout=None)`` and the calling
    thread never returns, so ``caller.is_alive()`` stays true and this is red.
    The second half is the other direction: ``committed`` proves the admitted
    writer was left alone rather than cancelled.
    """
    from polylogue.daemon.write_coordinator import DaemonWriterOwnerLoopStopped

    loop = asyncio.new_event_loop()
    loop_ready = threading.Event()
    coordinator_holder: list[DaemonWriteCoordinator] = []

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        coordinator_holder.append(DaemonWriteCoordinator(archive_root=tmp_path))
        loop.call_soon(loop_ready.set)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    assert loop_ready.wait(timeout=5.0)
    bridge = DaemonWriteThreadBridge(coordinator_holder[0], loop, timeout=0.05)

    writer_started = threading.Event()
    allow_writer = threading.Event()
    committed: list[str] = []
    raised: list[BaseException] = []

    def publish() -> str:
        writer_started.set()
        assert allow_writer.wait(timeout=5.0)
        committed.append("published")
        return "receipt"

    def caller_body() -> None:
        try:
            bridge.run_sync_with_timeout("derivation.session_profile", None, publish)
        except BaseException as exc:
            raised.append(exc)

    caller = threading.Thread(target=caller_body, daemon=True)
    caller.start()
    try:
        assert writer_started.wait(timeout=5.0)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5.0)
        assert not loop_thread.is_alive()

        caller.join(timeout=5.0)
        assert not caller.is_alive(), "the unbounded wait outlived its owner loop"
        assert len(raised) == 1
        assert isinstance(raised[0], DaemonWriterOwnerLoopStopped)
        assert "may still be in flight" in str(raised[0])

        # The writer was never withdrawn: it settles on its own thread.
        allow_writer.set()
        for _ in range(500):
            if committed:
                break
            time.sleep(0.01)
        assert committed == ["published"]
    finally:
        allow_writer.set()
        caller.join(timeout=5.0)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5.0)


def test_bridge_refuses_submission_to_an_already_stopped_owner_loop(tmp_path: Path) -> None:
    """A stopped loop yields a typed pre-admission result, not a raw runtime error."""
    from polylogue.daemon.write_coordinator import DaemonWriterOwnerLoopStopped

    loop = asyncio.new_event_loop()
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    loop.close()
    bridge = DaemonWriteThreadBridge(coordinator, loop, timeout=0.05)
    try:
        with pytest.raises(DaemonWriterOwnerLoopStopped, match="did not start"):
            bridge.run_sync_with_timeout("embedding.publish", None, lambda: "unreachable")
    finally:
        if not loop.is_closed():
            loop.close()


def test_bridge_refuses_submission_to_stopped_but_open_owner_loop(tmp_path: Path) -> None:
    """Anti-vacuity: is_closed alone submits to a loop that can never run it."""
    from polylogue.daemon.write_coordinator import DaemonWriterOwnerLoopStopped

    loop = asyncio.new_event_loop()
    stopped = threading.Event()

    def run_then_stop() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(loop.stop)
        loop.run_forever()
        stopped.set()

    thread = threading.Thread(target=run_then_stop, daemon=True)
    thread.start()
    try:
        assert stopped.wait(timeout=5.0)
        assert not loop.is_closed()
        assert not loop.is_running()
        bridge = DaemonWriteThreadBridge(DaemonWriteCoordinator(archive_root=tmp_path), loop, timeout=0.05)
        with pytest.raises(DaemonWriterOwnerLoopStopped, match="did not start"):
            bridge.run_sync_with_timeout("embedding.publish", None, lambda: "unreachable")
    finally:
        thread.join(timeout=5.0)
        loop.close()


@pytest.mark.asyncio
async def test_caller_cancelled_between_admission_and_start_keeps_the_admitted_write(tmp_path: Path) -> None:
    """Admitted-but-not-started is settlement's problem, not the caller's.

    polylogue-8r4zq AC3. ``on_admit`` fires inside the coordinator-owned
    execution after the gate is taken and before the operation body is awaited,
    so cancelling the caller there is the deterministic barrier for that race.
    The admitted write must still run to completion and the gate must stay held
    until it does.

    Anti-vacuity: make ``run`` cancel its execution task on caller
    cancellation regardless of ``request.acquired`` and ``receipts`` stays
    empty while the successor enters early.
    """
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    receipts: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()
    caller: list[asyncio.Task[str]] = []

    async def operation() -> str:
        entered.set()
        await release.wait()
        receipts.append("committed")
        return "receipt"

    def cancel_on_admission() -> None:
        caller[0].cancel()

    caller.append(
        asyncio.create_task(coordinator.run("control.mutation", operation, on_admit=cancel_on_admission)),
    )
    with pytest.raises(asyncio.CancelledError):
        await caller[0]

    while not entered.is_set():
        await asyncio.sleep(0)
    assert receipts == []
    assert coordinator.snapshot().active_actor == "control.mutation"

    successor_entered = asyncio.Event()

    async def successor() -> str:
        successor_entered.set()
        return "entered"

    successor_task = asyncio.create_task(coordinator.run("maintenance.successor", successor))
    while coordinator.snapshot().queued_actors != ("maintenance.successor",):
        await asyncio.sleep(0)
    assert not successor_entered.is_set()

    release.set()
    assert await successor_task == "entered"
    assert receipts == ["committed"]
    assert await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_failed_admission_hook_publishes_terminal_release(tmp_path: Path) -> None:
    """An admission refusal still settles the acquired gate in telemetry.

    Anti-vacuity: removing the admission-failure ``released`` event makes the
    final assertion fail even though the gate itself is released, leaving
    observers unable to distinguish a stuck writer from a refused one.
    """
    events: list[DaemonWriteEvent] = []
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path, observer=events.append)

    def refuse() -> None:
        raise RuntimeError("admission refused")

    async def operation() -> None:
        raise AssertionError("a refused admission must not run the body")

    with pytest.raises(RuntimeError, match="admission refused"):
        await coordinator.run("control.refused", operation, on_admit=refuse)

    assert [event.phase for event in events] == ["queued", "acquired", "released"]
    release = events[-1]
    assert release.outcome == "error"
    assert release.hold_budget_s is not None
    assert coordinator.snapshot().active_actor is None
    assert await coordinator.run("successor", lambda: _return_ready()) == "ready"
    assert await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_fresh_task_cannot_write_after_successful_shutdown(tmp_path: Path) -> None:
    """Anti-vacuity: adding a new task to the managed set would admit this write."""
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    assert await coordinator.shutdown(timeout=1.0)
    ran = False

    async def operation() -> str:
        nonlocal ran
        ran = True
        return "written"

    with pytest.raises(RuntimeError, match="shutting down"):
        await asyncio.create_task(coordinator.run("late.write", operation))
    assert ran is False


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("actual worker settlement and independent kernel exclusion")
@pytest.mark.parametrize("caught", [False, True])
async def test_terminal_worker_retains_all_sql_owners_until_successful_successor_admission(
    tmp_path: Path, caught: bool
) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore, ArchiveStoreSettlementError

    root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    handles: list[SettlementConnection] = []
    stores: list[ArchiveStore] = []
    original_threads: list[threading.Thread] = []

    def leave_unsettled() -> None:
        original_threads.append(threading.current_thread())
        stores.extend(ArchiveStore(root, initialize=False) for _ in range(2))
        for position, store in enumerate(stores):
            store._enter_mutation_lease()
            store._conn.execute("BEGIN IMMEDIATE" if position == 0 else "BEGIN")
            store._conn.execute("SELECT COUNT(*) FROM sessions")
            handle = arm_settlement(store._conn)
            handles.append(handle)
        for store in stores:
            try:
                store.close()
            except ArchiveStoreSettlementError:
                if not caught:
                    raise

    successor_called = False

    def successor() -> None:
        nonlocal successor_called
        successor_called = True
        assert not archive_custody_available(root)
        with ArchiveStore(root, initialize=False) as store:
            store._conn.execute("CREATE TABLE successor_probe (value INTEGER)")
            store.commit()

    try:
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.unsettled", leave_unsettled)
        assert original_threads[0].is_alive()
        assert not archive_custody_available(root)
        assert coordinator.snapshot().unsettled_writer_workers == 1
        assert coordinator.snapshot().sql_settlement_state == "required"
        assert daemon_write_telemetry_payload()["unsettled_writer_workers"] == 1
        assert not coordinator._idle.is_set()
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.refused_successor", successor)
        assert not successor_called
        assert len(handles) == 2
        assert all(any(name == "close" for name, _thread in handle.calls) for handle in handles)
        assert not archive_custody_available(root)
        for handle in handles:
            handle.allow_cleanup.set()
        await coordinator.run_sync("test.successor", successor)
        original_threads[0].join()
        assert not original_threads[0].is_alive()
        assert successor_called
        assert archive_custody_available(root)
        assert coordinator.snapshot().unsettled_writer_workers == 0
        assert coordinator.snapshot().sql_settlement_state == "idle"
        assert all(thread is handle.owner for handle in handles for _name, thread in handle.calls)
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
            if handle.continue_cleanup is not None:
                handle.continue_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_failed", [False, True])
@pytest.mark.uses_real_clock("temporary User SQL remains excluded until its original worker closes it")
async def test_terminal_worker_retains_failed_temporary_user_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commit_failed: bool,
) -> None:
    from polylogue.storage.sqlite.archive_tiers import archive as archive_module

    root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    handles: list[SettlementConnection] = []
    stores: list[archive_module.ArchiveStore] = []
    real_open = cast(Callable[..., sqlite3.Connection], vars(archive_module)["open_connection"])

    def controlled_open(path: Path, *args: object, **kwargs: object) -> sqlite3.Connection:
        connection = real_open(path, *args, **kwargs)
        if path.name == "user.db":
            handle = arm_settlement(connection)
            handles.append(handle)
            return handle
        return connection

    def failed_commit(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic User commit failure")

    monkeypatch.setattr(archive_module, "open_connection", controlled_open)
    if commit_failed:
        monkeypatch.setattr(ArchiveWriteGateway, "commit_write_sync", failed_commit)

    def leave_unsettled() -> None:
        store = archive_module.ArchiveStore(root, initialize=False)
        stores.append(store)
        try:
            store.add_user_tags((), ())
        except OSError:
            pass

    try:
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.user_cleanup", leave_unsettled)
        assert handles[0].in_transaction is commit_failed
        assert [id(conn) for conn in stores[0]._user_write_connections] == [id(handle) for handle in handles]
        assert not archive_custody_available(root)
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.user_cleanup_retry", lambda: None)
        assert handles[0].in_transaction is commit_failed
        handles[0].allow_cleanup.set()
        await coordinator.run_sync("test.user_successor", lambda: None)
        assert not stores[0]._user_write_connections
        handles[0].owner.join()
        assert not handles[0].owner.is_alive()
        assert archive_custody_available(root)
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("cancelled cleanup waiter leaves its accepted original worker alive")
async def test_terminal_worker_cleanup_survives_shutdown_waiter_cancellation(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    handles: list[SettlementConnection] = []

    def leave_unsettled() -> None:
        store = ArchiveStore(root, initialize=False)
        store._enter_mutation_lease()
        store._conn.execute("BEGIN IMMEDIATE")
        handle = arm_settlement(store._conn)
        handles.append(handle)
        store.close()

    try:
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.unsettled", leave_unsettled)
        handle = handles[0]
        assert not await coordinator.shutdown(timeout=30.0)
        assert coordinator.snapshot().sql_settlement_state == "required"
        assert not coordinator._idle.is_set()
        assert not archive_custody_available(root)
        handle.cleanup_started.clear()
        handle.allow_cleanup.set()
        handle.continue_cleanup = threading.Event()
        shutdown = asyncio.create_task(coordinator.shutdown(timeout=30.0))
        await asyncio.to_thread(handle.cleanup_started.wait)
        assert coordinator.snapshot().sql_settlement_state == "settling"
        shutdown.cancel()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        assert handle.owner.is_alive()
        assert not archive_custody_available(root)
        assert not coordinator._idle.is_set()
        handle.continue_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)
        handle.owner.join()
        assert not handle.owner.is_alive()
        assert archive_custody_available(root)
        assert coordinator.snapshot().unsettled_writer_workers == 0
        with pytest.raises(RuntimeError, match="shutting down"):
            await coordinator.run_sync("test.after_shutdown", lambda: None)
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
            if handle.continue_cleanup is not None:
                handle.continue_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("caller cancellation cannot discard terminal worker custody")
async def test_cancelled_run_sync_caller_leaves_terminal_settlement_owned(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    ready = threading.Event()
    release = threading.Event()
    handles: list[SettlementConnection] = []
    completions: asyncio.Queue[asyncio.Task[object]] = asyncio.Queue()

    def operation() -> None:
        store = ArchiveStore(root, initialize=False)
        store._enter_mutation_lease()
        store._conn.execute("BEGIN IMMEDIATE")
        handle = arm_settlement(store._conn)
        handles.append(handle)
        ready.set()
        release.wait()
        store.close()

    caller = asyncio.create_task(
        coordinator.run_sync_with_completion("test.cancelled", operation, completions.put_nowait)
    )
    try:
        await asyncio.to_thread(ready.wait)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert not archive_custody_available(root)
        release.set()
        completed = await completions.get()
        assert isinstance(completed.exception(), DaemonWriterSettlementError)
        assert coordinator.snapshot().unsettled_writer_workers == 1
        assert not coordinator._idle.is_set()
        handles[0].allow_cleanup.set()
        await coordinator.run_sync("test.successor", lambda: None)
        handles[0].owner.join()
        assert archive_custody_available(root)
    finally:
        release.set()
        for handle in handles:
            handle.allow_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("real nested async worker retains its original SQLite thread")
@pytest.mark.parametrize("nested", [False, True])
async def test_terminal_settlement_owns_direct_and_nested_async_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    from polylogue.core.write_lease import adopt_write_lease, delegate_write_lease
    from polylogue.storage.sqlite import async_sqlite

    root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    handles: list[SettlementConnection] = []
    backends: list[async_sqlite.SQLiteBackend] = []
    writer_threads: list[threading.Thread] = []
    real_connect = async_sqlite._connect_write_thread

    def controlled_connect(backend: async_sqlite.SQLiteBackend, grant: object) -> sqlite3.Connection:
        handle = arm_settlement(real_connect(backend, grant))  # type: ignore[arg-type]
        handle.allow_cleanup.set()
        handles.append(handle)
        return handle

    monkeypatch.setattr(async_sqlite, "_connect_write_thread", controlled_connect)

    async def leave_unsettled() -> None:
        writer_threads.append(threading.current_thread())
        backend = async_sqlite.SQLiteBackend(db_path=root / "index.db")
        backends.append(backend)
        await backend.begin()
        handle = handles[-1]
        handle.allow_cleanup.clear()
        try:
            await backend.close()
        except BaseExceptionGroup as refused:
            assert len(refused.exceptions) == 2
            assert all(isinstance(error, OSError) for error in refused.exceptions)
            # Discarding an outcome cannot discard its physical owner.

    def nested_operation() -> None:
        delegation = delegate_write_lease()

        async def adopted() -> None:
            with adopt_write_lease(delegation):
                await leave_unsettled()

        asyncio.run(adopted())

    try:
        with pytest.raises(DaemonWriterSettlementError):
            if nested:
                await coordinator.run_sync("test.nested_async", nested_operation)
            else:
                await coordinator.run("test.direct_async", leave_unsettled)
        backend = backends[0]
        connection = backend._txn_conn
        assert connection is not None
        assert connection._thread.is_alive()
        assert not archive_custody_available(root)
        assert coordinator.snapshot().unsettled_writer_workers == int(nested)
        assert coordinator.snapshot().unsettled_async_backends == int(not nested)
        assert coordinator.snapshot().sql_settlement_state == "required"
        assert not coordinator._idle.is_set()
        if hasattr(os, "fork"):
            child_pid = None
            read_fd, write_fd = os.pipe()
            try:
                with async_sqlite._BACKEND_CONNECTIONS_LOCK:
                    child_pid = os.fork()
                    if child_pid == 0:
                        try:
                            os.close(read_fd)
                            assert not async_sqlite.retained_write_backends_on_current_thread()
                            assert async_sqlite._FORK_ABANDONED_BACKEND_CONNECTIONS
                            closing = backend.close()
                            try:
                                closing.send(None)
                            except RuntimeError:
                                pass
                            else:
                                os._exit(2)
                            finally:
                                closing.close()
                            os.write(write_fd, b"refused")
                            os._exit(0)
                        except BaseException:
                            os._exit(3)
                os.close(write_fd)
                write_fd = -1
                ready, _, _ = select.select([read_fd], [], [], 60.0)
                assert ready, "child blocked on inherited async writer registry"
                assert os.read(read_fd, 7) == b"refused"
                _, status = os.waitpid(child_pid, 0)
                child_pid = None
                assert os.waitstatus_to_exitcode(status) == 0
                assert connection._thread.is_alive()
                assert not archive_custody_available(root)
            finally:
                if child_pid is not None and child_pid > 0:
                    os.kill(child_pid, signal.SIGKILL)
                    os.waitpid(child_pid, 0)
                os.close(read_fd)
                if write_fd >= 0:
                    os.close(write_fd)
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.failed_async_settlement", lambda: None)
        assert connection._thread.is_alive()
        assert not archive_custody_available(root)
        handle = handles[-1]
        handle.cleanup_started.clear()
        handle.allow_cleanup.set()
        handle.continue_cleanup = threading.Event()
        successor = asyncio.create_task(coordinator.run_sync("test.cancelled_async_settlement", lambda: None))
        await asyncio.to_thread(handle.cleanup_started.wait)
        assert coordinator.snapshot().sql_settlement_state == "settling"
        successor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await successor
        assert connection._thread.is_alive()
        assert not archive_custody_available(root)
        handle.continue_cleanup.set()
        await coordinator.run_sync("test.async_successor", lambda: None)
        connection._thread.join()
        assert not connection._thread.is_alive()
        assert backend._txn_conn is None
        assert archive_custody_available(root)
        assert coordinator.snapshot().unsettled_writer_workers == 0
        assert coordinator.snapshot().unsettled_async_backends == 0
        assert all(thread is handle.owner for handle in handles for _name, thread in handle.calls)
        if nested:
            writer_threads[0].join()
            assert not writer_threads[0].is_alive()
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
            if handle.continue_cleanup is not None:
                handle.continue_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("forked child must refuse before inherited terminal mutex and SQL")
@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable")
async def test_terminal_worker_fork_refuses_before_inherited_mutex_or_sql(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    handles: list[SettlementConnection] = []
    stores: list[ArchiveStore] = []

    def leave_unsettled() -> None:
        store = ArchiveStore(root, initialize=False)
        stores.append(store)
        store._enter_mutation_lease()
        store._conn.execute("BEGIN IMMEDIATE")
        handle = arm_settlement(store._conn)
        handles.append(handle)
        store.close()

    child_pid: int | None = None
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.fork_parent", leave_unsettled)
        with coordinator._terminal_guard:
            child_pid = os.fork()
            if child_pid == 0:
                try:
                    os.close(read_fd)
                    operation = coordinator.run("test.fork_child", _return_ready)
                    try:
                        operation.send(None)
                    except DaemonWriterSettlementError:
                        pass
                    else:
                        os._exit(2)
                    finally:
                        operation.close()
                    try:
                        stores[0].close()
                    except RuntimeError:
                        pass
                    else:
                        os._exit(3)
                    os.write(write_fd, b"refused")
                    os._exit(0)
                except BaseException:
                    os._exit(4)
        os.close(write_fd)
        write_fd = -1
        ready, _, _ = select.select([read_fd], [], [], 60.0)
        assert ready, "child blocked on inherited terminal mutex"
        assert os.read(read_fd, 7) == b"refused"
        _, status = os.waitpid(child_pid, 0)
        child_pid = None
        assert os.waitstatus_to_exitcode(status) == 0
        assert not archive_custody_available(root)
        assert handles[0].owner.is_alive()
        handles[0].allow_cleanup.set()
        await coordinator.run_sync("test.fork_successor", lambda: None)
        handles[0].owner.join()
        assert archive_custody_available(root)
    finally:
        if child_pid is not None and child_pid > 0:
            os.kill(child_pid, signal.SIGKILL)
            os.waitpid(child_pid, 0)
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
        for handle in handles:
            handle.allow_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory_name",
    [
        "open_connection",
        "open_daemon_connection",
        "open_source_tier_write_connection",
        "open_isolated_write_connection",
        "initialize_archive_database",
    ],
)
@pytest.mark.uses_real_clock("native configure and close failure retains actual original-thread SQL custody")
async def test_native_factory_failure_keeps_actual_connection_until_terminal_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
) -> None:
    from polylogue.storage.sqlite import connection_profile
    from polylogue.storage.sqlite.archive_tiers import bootstrap
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    handles: list[SettlementConnection] = []
    real_sqlite_connect = cast(Callable[..., sqlite3.Connection], sqlite3.connect)

    from typing import Any

    class FailedConfigurationHandle(SettlementConnection):
        def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
            raise OSError("synthetic profile setup failure")

    def controlled_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = real_sqlite_connect(*args, **{**kwargs, "factory": FailedConfigurationHandle})
        sqlite3.Connection.execute(
            connection, "BEGIN" if factory_name == "open_readonly_connection" else "BEGIN IMMEDIATE"
        )
        handle = arm_settlement(connection)
        handles.append(handle)
        return handle

    if factory_name == "initialize_archive_database":
        monkeypatch.setattr(sqlite3, "connect", controlled_connect)
    else:
        monkeypatch.setattr(connection_profile, "connect_measured", controlled_connect)

    def leave_unsettled() -> None:
        try:
            if factory_name == "initialize_archive_database":
                bootstrap.initialize_archive_database(root / "fresh-index.db", ArchiveTier.INDEX, page_size=4096)
                return
            factory = getattr(connection_profile, factory_name)
            if factory_name == "open_isolated_write_connection":
                factory(root / "source.db", archive_root=root, purpose="test.native_configuration")
            else:
                factory(root / "source.db", archive_root=root)
        except connection_profile.NativeConnectionSettlementError:
            # Even callers discarding the exception cannot discard the actual
            # connection or its original-thread physical custody.
            pass

    try:
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.native_configuration", leave_unsettled)
        assert handles[0].in_transaction
        assert not archive_custody_available(root)
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.native_configuration_retry", lambda: None)
        assert handles[0].in_transaction
        handles[0].allow_cleanup.set()
        await coordinator.run_sync("test.native_configuration_successor", lambda: None)
        handles[0].owner.join()
        assert not handles[0].owner.is_alive()
        assert archive_custody_available(root)
        assert {thread for _operation, thread in handles[0].calls} == {handles[0].owner}
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("begin_inside_context", [False, True])
@pytest.mark.uses_real_clock("cached commit settlement and failed close retain actual kernel custody")
async def test_cached_connection_settles_after_context_and_retains_failed_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    begin_inside_context: bool,
) -> None:
    from polylogue.storage.sqlite import connection as cached
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError

    root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    handles: list[SettlementConnection] = []
    real_connect = cast(Callable[..., sqlite3.Connection], vars(cached)["connect_measured"])

    def controlled_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        handle = real_connect(*args, **kwargs)
        assert isinstance(handle, SettlementConnection)
        handles.append(handle)
        return handle

    monkeypatch.setattr(cached, "connect_measured", controlled_connect)

    def committed_after_context() -> None:
        with cached.connection_context(root / "index.db") as connection:
            if begin_inside_context:
                connection.execute("BEGIN IMMEDIATE")
        if not begin_inside_context:
            connection.execute("BEGIN IMMEDIATE")
        # The cached context intentionally does not settle its caller's SQL.
        connection.commit()
        handles[-1].allow_cleanup.set()

    def failed_close() -> None:
        committed_after_context()
        arm_settlement(handles[-1])
        try:
            cached._clear_connection_cache()
        except NativeConnectionSettlementError:
            pass
        assert cached._connection_cache.conns[str(root / "index.db")].connection is handles[-1]

    try:
        await coordinator.run_sync("test.cached_commit", committed_after_context)
        assert archive_custody_available(root)
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.cached_close", failed_close)
        assert not handles[-1].in_transaction
        assert not archive_custody_available(root)
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.cached_retry", lambda: None)
        handles[-1].allow_cleanup.set()
        await coordinator.run_sync("test.cached_successor", lambda: None)
        handles[-1].owner.join()
        assert archive_custody_available(root)
        assert {thread for _operation, thread in handles[-1].calls} == {handles[-1].owner}
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context_name",
    ["open_verified_audit_connection", "open_verified_sqlite_write_connection", "open_verified_sqlite_read_connection"],
)
@pytest.mark.uses_real_clock("verified leaf remains pinned until actual native SQLite settlement")
async def test_verified_leaf_is_retained_with_failed_native_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    context_name: str,
) -> None:
    from polylogue.storage.sqlite import audit_leaf, connection_profile

    root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    handles: list[SettlementConnection] = []
    leaves: list[audit_leaf.VerifiedAuditLeaf] = []
    real_connect = cast(Callable[..., sqlite3.Connection], sqlite3.connect)
    real_enter = audit_leaf.VerifiedAuditLeaf.__enter__

    def controlled_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        handle = arm_settlement(real_connect(*args, **kwargs))
        handles.append(handle)
        return handle

    def remember_leaf(leaf: audit_leaf.VerifiedAuditLeaf) -> audit_leaf.VerifiedAuditLeaf:
        result = real_enter(leaf)
        leaves.append(result)
        return result

    monkeypatch.setattr(sqlite3, "connect", controlled_connect)
    monkeypatch.setattr(audit_leaf.VerifiedAuditLeaf, "__enter__", remember_leaf)

    def leave_unsettled() -> None:
        try:
            with getattr(audit_leaf, context_name)(root / "audit.db") as connection:
                connection.execute("SELECT 1").fetchone()
                raise RuntimeError("synthetic body failure")
        except connection_profile.NativeConnectionSettlementError:
            pass

    try:
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.verified_leaf", leave_unsettled)
        assert leaves[-1]._leaf_fd is not None
        os.fstat(leaves[-1]._leaf_fd)
        assert not archive_custody_available(root)
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.verified_leaf_retry", lambda: None)
        assert leaves[-1]._leaf_fd is not None
        for handle in handles:
            handle.allow_cleanup.set()
        await coordinator.run_sync("test.verified_leaf_successor", lambda: None)
        handles[-1].owner.join()
        assert leaves[-1]._leaf_fd is None
        assert leaves[-1]._directory_fd is None
        assert archive_custody_available(root)
        assert {thread for handle in handles for _operation, thread in handle.calls} == {handles[-1].owner}
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("cache cleanup attempts every real handle and retains only failed closes")
async def test_cached_cleanup_attempts_later_handle_after_first_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite import connection as cached
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError

    root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    handles: list[SettlementConnection] = []
    real_connect = cast(Callable[..., sqlite3.Connection], vars(cached)["connect_measured"])

    def controlled_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        handle = real_connect(*args, **kwargs)
        assert isinstance(handle, SettlementConnection)
        handles.append(handle)
        return handle

    monkeypatch.setattr(cached, "connect_measured", controlled_connect)

    def leave_first_unsettled() -> None:
        for path in (root / "index.db", root / "scratch-index.db"):
            with cached.connection_context(path) as connection:
                connection.execute("SELECT 1").fetchone()
        assert len(handles) == 2
        arm_settlement(handles[0])
        try:
            cached._clear_connection_cache()
        except NativeConnectionSettlementError:
            pass
        assert list(cached._connection_cache.conns) == [str(root / "index.db")]
        assert any(operation == "close" for operation, _thread in handles[1].calls)

    try:
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.cached_all_attempt", leave_first_unsettled)
        assert len(handles) == 2
        assert not archive_custody_available(root)
        handles[0].allow_cleanup.set()
        await coordinator.run_sync("test.cached_all_attempt_successor", lambda: None)
        handles[0].owner.join()
        assert archive_custody_available(root)
        assert {thread for handle in handles for _operation, thread in handle.calls} == {handles[0].owner}
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("admitted async readers retain actual SQL and physical custody")
async def test_admitted_reader_failed_close_remains_in_terminal_writer_census(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite import async_sqlite

    root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    handles: list[SettlementConnection] = []
    configured = 0
    actual_configure = async_sqlite.configure_read_connection

    async def configure(connection: object, *, archive_root: Path) -> None:
        nonlocal configured
        await actual_configure(connection, archive_root=archive_root)  # type: ignore[arg-type]
        configured += 1
        if configured == 2:

            def install() -> None:
                handle = arm_settlement(connection._connection)  # type: ignore[attr-defined]
                handles.append(handle)

            await connection._execute(install)  # type: ignore[attr-defined]

    monkeypatch.setattr(async_sqlite, "configure_read_connection", configure)

    async def operation() -> None:
        backend = async_sqlite.SQLiteBackend(root / "index.db")
        try:
            async with backend.read_connection() as connection:
                await connection.execute("BEGIN")
                await connection.execute("SELECT COUNT(*) FROM sessions")
        except BaseExceptionGroup as refused:
            assert len(refused.exceptions) == 2
            assert all(isinstance(error, OSError) for error in refused.exceptions)

    called = False

    async def successor() -> None:
        nonlocal called
        called = True

    try:
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run("test.reader_close", operation)
        assert len(handles) == 1
        assert not archive_custody_available(root)
        assert coordinator.snapshot().unsettled_async_backends == 1
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run("test.reader_refusal", successor)
        assert not called
        handles[0].allow_cleanup.set()
        await coordinator.run("test.reader_successor", successor)
        assert called
        assert archive_custody_available(root)
        assert coordinator.snapshot().unsettled_async_backends == 0
        assert all(thread is handles[0].owner for _name, thread in handles[0].calls)
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("last grant returns physical custody on actual creator worker")
async def test_last_worker_grant_failure_retains_terminal_worker_and_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import Future

    from polylogue.storage.sqlite.write_lease import async_write_lease

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    real_close = os.close
    entered, release = threading.Event(), threading.Event()
    physical: Future[None] = Future()
    threads: list[threading.Thread] = []
    attempts: list[threading.Thread] = []

    async with async_write_lease("test.last_worker_grant", archive_root=tmp_path) as lease:
        custody = lease.custody
        assert custody is not None
        lock = custody._fd

        def refuse_close(descriptor: int) -> None:
            if descriptor == lock:
                attempts.append(threading.current_thread())
                raise OSError("synthetic worker descriptor close before effect")
            real_close(descriptor)

        def operation() -> None:
            entered.set()
            release.wait()

        def submit(function: Callable[[], None]) -> Future[None]:
            def run() -> None:
                try:
                    function()
                except BaseException as error:
                    physical.set_exception(error)
                else:
                    physical.set_result(None)

            thread = threading.Thread(target=run, name="test-last-grant-creator")
            threads.append(thread)
            thread.start()
            assert entered.wait(5)
            # Actual owner retirement leaves the executing worker's grant as
            # the last reference. Its complete() must not erase new cleanup.
            lease.retire()
            custody.close_owner()
            monkeypatch.setattr(os, "close", refuse_close)
            release.set()
            return physical

        dispatch = write_coordinator_module._WorkerDispatch(submit, lambda: ())
        try:
            with pytest.raises(DaemonWriterSettlementError):
                await write_coordinator_module._run_writer_worker(coordinator, dispatch, operation, "test.last_grant")
            assert not physical.done()
            assert threads[0].is_alive()
            assert custody._descriptor_cleanup_thread is threads[0]
            assert coordinator._retained_workers()
            assert not archive_custody_available(tmp_path)
            assert attempts == [threads[0]]
        finally:
            monkeypatch.setattr(os, "close", real_close)
            release.set()
            for descriptor in tuple(custody._pending_descriptor_closes):
                real_close(descriptor)
            if coordinator._retained_workers():
                await coordinator._settle_terminal_workers()
            for thread in threads:
                await asyncio.to_thread(thread.join)
        assert physical.done()
        physical.result()
        assert not coordinator._retained_workers()
        assert custody._fd == -1
        assert attempts == [threads[0]]
    assert await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("explicit coordinator retry wakes original async cleanup task")
@pytest.mark.parametrize("cancelled_waiter", [False, True])
async def test_coordinator_second_settlement_retries_original_last_grant_cleanup(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, cancelled_waiter: bool
) -> None:
    from typing import Any

    from polylogue.storage.sqlite import async_sqlite
    from polylogue.storage.sqlite.write_lease import current_write_lease

    root = workspace_env["archive_root"]
    coordinator = DaemonWriteCoordinator(archive_root=root)
    backend = async_sqlite.SQLiteBackend(root / "index.db")
    real_close = os.close
    failed = asyncio.Event()
    custodies = []
    connections = []
    original_failure = OSError("synthetic first worker close refusal")
    descriptor_failure = OSError("synthetic later grant close before effect")
    attempts = []

    async def operation() -> None:
        lease = current_write_lease()
        assert lease is not None and lease.custody is not None
        custodies.append(lease.custody)
        conn = await async_sqlite._open_configured_backend_connection(backend, read_only=True)
        connections.append(conn)
        execute = conn._execute
        first = True

        async def refuse_first_close(function: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal first
            if first and getattr(function, "__name__", None) == "close_raw":
                first = False
                raise original_failure
            return await execute(function, *args, **kwargs)  # type: ignore[no-untyped-call]

        monkeypatch.setattr(conn, "_execute", refuse_first_close)
        with pytest.raises(OSError) as caught:
            await async_sqlite._close_backend_connection(conn)
        assert caught.value is original_failure

    with pytest.raises(DaemonWriterSettlementError):
        await coordinator.run("test.async_last_grant", operation)
    custody = custodies[0]
    lock = custody._fd

    def refuse_descriptor(descriptor: int) -> None:
        if descriptor == lock:
            attempts.append(threading.current_thread())
            failed.set()
            raise descriptor_failure
        real_close(descriptor)

    monkeypatch.setattr(os, "close", refuse_descriptor)
    first = asyncio.create_task(coordinator._settle_terminal_workers())
    second = None
    try:
        await asyncio.wait_for(failed.wait(), 5)
        original = coordinator._terminal_async_attempt
        assert original is not None and not original.done()
        entry = async_sqlite._BACKEND_CONNECTIONS[id(connections[0])]
        assert entry.cleanup_task is original
        assert entry.cleanup_attempt is not None and not entry.cleanup_attempt.done()
        if cancelled_waiter:
            second = asyncio.create_task(coordinator._settle_terminal_workers())
            await asyncio.sleep(0)
            second.cancel()
            await asyncio.sleep(0)
            assert not original.done()
            assert not entry.cleanup_attempt.done()
        monkeypatch.setattr(os, "close", real_close)
        real_close(lock)
        if second is None:
            second = asyncio.create_task(coordinator._settle_terminal_workers())
        else:
            backend.request_sql_settlement()
        outcomes = await asyncio.gather(first, second, return_exceptions=True)
        assert isinstance(outcomes[0], BaseException)
        assert isinstance(outcomes[1], asyncio.CancelledError if cancelled_waiter else BaseException)
        assert original.done()
        assert attempts == [threading.current_thread()]
        assert id(connections[0]) not in async_sqlite._BACKEND_CONNECTIONS
        assert not coordinator._retained_async_backends()
    finally:
        monkeypatch.setattr(os, "close", real_close)
        for descriptor in tuple(custody._pending_descriptor_closes):
            real_close(descriptor)
        backend.request_sql_settlement()
        await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)
        assert await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("actual async SQLite cleanup retains custody through canceled waiters")
@pytest.mark.parametrize("post_close_failure", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_async_terminal_failure_classification_tracks_actual_physical_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, post_close_failure: bool, cancelled: bool
) -> None:
    from polylogue.storage.sqlite import async_sqlite

    root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    backend = async_sqlite.SQLiteBackend(root / "index.db")
    handles: list[SettlementConnection] = []
    entered_cleanup = asyncio.Event()
    release_cleanup = asyncio.Event()
    successor_called = False

    async def leave_unsettled() -> None:
        await backend.begin()
        connection = backend._txn_conn
        assert connection is not None

        def arm_actual_admitted_connection() -> None:
            native = connection._connection
            assert native is not None
            handles.append(arm_settlement(native))

        execute_on_creator = cast(Callable[[Callable[[], None]], Awaitable[None]], connection._execute)
        await execute_on_creator(arm_actual_admitted_connection)
        with pytest.raises(BaseExceptionGroup) as original:
            await backend.close()
        assert len(original.value.exceptions) == 2
        assert all(isinstance(error, OSError) for error in original.value.exceptions)

    actual_close = backend.close

    async def held_cleanup() -> None:
        entered_cleanup.set()
        await release_cleanup.wait()
        await actual_close()
        raise ValueError("synthetic failure after physical SQL settlement")

    async def successor() -> None:
        nonlocal successor_called
        successor_called = True

    try:
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run("test.actual_async_initial_fault", leave_unsettled)
        assert len(handles) == 1
        assert handles[0].cleanup_started.is_set()
        assert not archive_custody_available(root)
        monkeypatch.setattr(backend, "close", held_cleanup)
        if post_close_failure:
            handles[0].allow_cleanup.set()
        waiting = asyncio.create_task(coordinator.run("test.actual_async_cleanup", successor))
        await asyncio.wait_for(entered_cleanup.wait(), timeout=5)
        executions = tuple(coordinator._executions)
        assert len(executions) == 1
        assert coordinator.snapshot().unsettled_async_backends == 1
        assert not successor_called
        assert not archive_custody_available(root)
        if cancelled:
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert coordinator.snapshot().unsettled_async_backends == 1
            assert not archive_custody_available(root)
            assert not executions[0].done()
        release_cleanup.set()
        await asyncio.wait((executions[0],))
        if post_close_failure:
            with pytest.raises(ValueError):
                executions[0].result()
            assert coordinator.snapshot().unsettled_async_backends == 0
            assert archive_custody_available(root)
        else:
            with pytest.raises(DaemonWriterSettlementError) as refused:
                executions[0].result()
            original_failure = refused.value.__cause__
            assert isinstance(original_failure, BaseExceptionGroup)
            assert len(original_failure.exceptions) == 2
            assert all(isinstance(error, OSError) for error in original_failure.exceptions)
            assert coordinator.snapshot().unsettled_async_backends == 1
            assert not archive_custody_available(root)
        if not cancelled:
            with pytest.raises(ValueError if post_close_failure else DaemonWriterSettlementError):
                await waiting
        assert not successor_called
    finally:
        release_cleanup.set()
        monkeypatch.setattr(backend, "close", actual_close)
        for handle in handles:
            handle.allow_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)
