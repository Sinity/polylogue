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
    _actor_priority,
    _PriorityGate,
    daemon_write_telemetry_payload,
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
async def test_maintenance_actor_jumps_ahead_of_queued_bulk_ingest_actors() -> None:
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

    coordinator = DaemonWriteCoordinator(observer=observe)
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
async def test_cancelled_priority_waiter_does_not_strand_the_grant() -> None:
    """Cancelling a still-queued higher-priority waiter must not drop the
    gate: the next (lower-priority) waiter still gets admitted afterward."""
    queued_maintenance = asyncio.Event()
    queued_watcher = asyncio.Event()

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "queued" and event.actor == "maintenance.wal_checkpoint":
            queued_maintenance.set()
        if event.phase == "queued" and event.actor == "watcher.catch_up.chunk":
            queued_watcher.set()

    coordinator = DaemonWriteCoordinator(observer=observe)
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

    # polylogue-es7b: bumped from 5.0s -- this module's cold-import cost alone
    # can run several seconds under concurrent xdist workers (each spawning
    # its own subprocess simultaneously), which made this timeout marginal
    # once a third subprocess-spawning test landed in this file.
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        timeout=20.0,
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

    # polylogue-es7b: bumped from 5.0s -- this module's cold-import cost alone
    # can run several seconds under concurrent xdist workers (each spawning
    # its own subprocess simultaneously), which made this timeout marginal
    # once a third subprocess-spawning test landed in this file.
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        timeout=20.0,
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


@pytest.mark.asyncio
async def test_detached_writer_failure_increments_lifetime_counter() -> None:
    """polylogue-es7b: a failed writer task's exception previously surfaced only via a log line.

    ``completed()`` (the task's done-callback) must also increment a durable
    daemon-lifetime counter so the failure is observable through the
    coordinator's telemetry snapshot/payload, not only in logs.
    """
    coordinator = DaemonWriteCoordinator()
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
    payload = daemon_write_telemetry_payload()
    assert payload["detached_writer_failures"] == 1

    # A second failure keeps accumulating -- this is a lifetime counter, not
    # a one-shot flag.
    with pytest.raises(RuntimeError, match="writer blew up"):
        await coordinator.run("actor", boom)
    for _ in range(10):
        if coordinator.snapshot().detached_writer_failures == 2:
            break
        await asyncio.sleep(0)
    assert coordinator.snapshot().detached_writer_failures == 2


def test_run_in_daemon_thread_logs_instead_of_hanging_when_loop_already_closed() -> None:
    """polylogue-es7b: a worker thread finishing after its loop closed must not hang silently.

    ``_run_in_daemon_thread`` spawns a plain (uncancellable) ``threading.Thread``.
    If the coordinator's event loop closes before that thread finishes --
    the real shape of the hazard is a stuck sync writer that outlives a
    process shutdown that gave up waiting on it (see
    ``test_stuck_sync_writer_cannot_pin_process_exit`` above) -- the worker's
    final ``loop.call_soon_threadsafe`` raises ``RuntimeError`` because the
    loop is closed. Previously this was silently swallowed, leaving the
    original awaiting future unresolved with zero forensic trace. The fix
    logs a warning instead of swallowing it silently; run in a subprocess
    because it requires actually closing a real event loop out from under a
    still-running background thread.
    """
    script = textwrap.dedent(
        """
        import asyncio
        import logging
        import sys
        import threading

        from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

        finish = threading.Event()

        async def main() -> None:
            coordinator = DaemonWriteCoordinator()
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
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        timeout=20.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "event loop already closed" in completed.stderr, completed.stderr


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


def test_thread_bridge_run_sync_uses_the_bridge_default_timeout() -> None:
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
        coordinator_holder.append(DaemonWriteCoordinator())
        loop_ready.set()
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


def test_thread_bridge_run_sync_with_timeout_overrides_the_bridge_default() -> None:
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
        coordinator_holder.append(DaemonWriteCoordinator())
        loop_ready.set()
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
