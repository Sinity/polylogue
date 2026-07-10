"""Deterministic proofs for intra-daemon archive write serialization."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path

import pytest

from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteEvent


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

    def observe(event: DaemonWriteEvent) -> None:
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
    loop_turn = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(loop_turn.set_result, None)
    await loop_turn

    assert coordinator.snapshot().active_actor == "raw"
    assert not released.is_set()
    allow_worker_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)
    assert released.is_set()
    assert coordinator.snapshot().active_actor is None
    last_event = coordinator.snapshot().last_event
    assert last_event is not None
    assert last_event.outcome == "cancelled"


async def _unexpected_operation() -> None:
    raise AssertionError("cancelled queued writer must not enter")


async def _return_ready() -> str:
    return "ready"
