"""Deterministic proofs for intra-daemon archive write serialization."""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from polylogue.daemon.write_coordinator import (
    DaemonWriteCoordinator,
    DaemonWriteEvent,
    DaemonWriteThreadBridge,
    daemon_write_telemetry_payload,
)


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

    coordinator = DaemonWriteCoordinator(observer=observe)
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
async def test_coordinator_serializes_fifo_without_writer_overlap() -> None:
    queued = {actor: asyncio.Event() for actor in ("watcher", "raw", "embedding")}
    events: list[DaemonWriteEvent] = []

    def observe(event: DaemonWriteEvent) -> None:
        events.append(event)
        if event.phase == "queued":
            queued[event.actor].set()

    coordinator = DaemonWriteCoordinator(observer=observe)
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
async def test_waiting_cancellation_removes_actor_without_deadlock() -> None:
    raw_queued = asyncio.Event()

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "queued" and event.actor == "raw":
            raw_queued.set()

    coordinator = DaemonWriteCoordinator(observer=observe)
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
async def test_sync_writer_cancellation_holds_gate_until_thread_finishes() -> None:
    released = asyncio.Event()
    events: list[DaemonWriteEvent] = []

    def observe(event: DaemonWriteEvent) -> None:
        events.append(event)
        if event.phase == "released" and event.actor == "raw":
            released.set()

    coordinator = DaemonWriteCoordinator(observer=observe)
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
    assert raw_release.outcome == "cancelled"


@pytest.mark.asyncio
async def test_child_task_cannot_inherit_reentrant_write_lease() -> None:
    coordinator = DaemonWriteCoordinator()

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
async def test_cancelled_queued_writer_never_runs() -> None:
    coordinator = DaemonWriteCoordinator()
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
async def test_shutdown_is_bounded_without_releasing_active_sync_writer() -> None:
    coordinator = DaemonWriteCoordinator()
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


def test_stuck_sync_writer_cannot_pin_process_exit() -> None:
    script = textwrap.dedent(
        """
        import asyncio
        import contextlib
        import threading

        from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

        async def main() -> None:
            coordinator = DaemonWriteCoordinator()
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

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("helper_name", "writer_name"),
    [
        ("_run_startup_fts_readiness", "_ensure_fts_startup_readiness_sync"),
        ("_run_startup_lineage_readiness", "_ensure_lineage_startup_readiness_sync"),
    ],
)
def test_real_startup_writer_routes_cannot_pin_process_exit(helper_name: str, writer_name: str) -> None:
    script = textwrap.dedent(
        f"""
        import asyncio
        import contextlib
        import threading

        from polylogue.daemon import cli
        from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

        async def main() -> None:
            coordinator = DaemonWriteCoordinator()
            started = threading.Event()

            def writer() -> None:
                started.set()
                threading.Event().wait()

            setattr(cli, {writer_name!r}, writer)
            helper = getattr(cli, {helper_name!r})
            caller = asyncio.create_task(helper(coordinator))
            while not started.is_set():
                await asyncio.sleep(0.001)
            caller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await caller
            assert await coordinator.shutdown(timeout=0.01) is False

        asyncio.run(main())
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.asyncio
async def test_operational_telemetry_reports_actor_queue_wait_and_hold() -> None:
    coordinator = DaemonWriteCoordinator()
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


def test_thread_bridge_serializes_sync_request_bodies_without_overlap() -> None:
    loop = asyncio.new_event_loop()
    loop_ready = threading.Event()
    second_queued = threading.Event()
    coordinator_holder: list[DaemonWriteCoordinator] = []

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "queued" and event.actor == "http.user.marks.post":
            second_queued.set()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        coordinator_holder.append(DaemonWriteCoordinator(observer=observe))
        loop_ready.set()
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


def test_thread_bridge_timeout_releases_a_racing_acquisition() -> None:
    loop = asyncio.new_event_loop()
    loop_ready = threading.Event()
    acquired = threading.Event()
    unblock_observer = threading.Event()
    coordinator_holder: list[DaemonWriteCoordinator] = []

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "acquired" and event.actor == "http.timeout-race":
            acquired.set()
            assert unblock_observer.wait(timeout=1.0)

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        coordinator_holder.append(DaemonWriteCoordinator(observer=observe))
        loop_ready.set()
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    assert loop_ready.wait(timeout=1.0)
    coordinator = coordinator_holder[0]
    bridge = DaemonWriteThreadBridge(coordinator, loop, timeout=0.05)
    errors: list[BaseException] = []

    def timed_request() -> None:
        try:
            with bridge.hold("http.timeout-race"):
                raise AssertionError("timed-out bridge must not enter the request body")
        except BaseException as exc:
            errors.append(exc)

    request = threading.Thread(target=timed_request)
    request.start()
    assert acquired.wait(timeout=1.0)
    request.join(timeout=1.0)
    assert not request.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], TimeoutError)

    unblock_observer.set()

    async def successor() -> str:
        return "entered"

    future = asyncio.run_coroutine_threadsafe(coordinator.run("successor", successor), loop)
    assert future.result(timeout=1.0) == "entered"
    shutdown = asyncio.run_coroutine_threadsafe(coordinator.shutdown(timeout=1.0), loop)
    assert shutdown.result(timeout=1.0)
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=1.0)


async def _unexpected_operation() -> None:
    raise AssertionError("cancelled queued writer must not enter")


async def _return_ready() -> str:
    return "ready"
