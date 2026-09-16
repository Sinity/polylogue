"""The daemon event bus has one real producer and one real consumer.

The typed bus landed with full unit coverage of its pub/sub core but nothing in
the daemon published to it or subscribed from it (polylogue-14t7). The producer
is now a declared post-commit write effect; the consumer is the embedding
backlog loop, which waits on the event instead of sleeping out its full poll
interval and keeps that interval as its reconciliation tick.
"""

from __future__ import annotations

import asyncio
import random
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.write_effects import commit_archive_write_effects
from polylogue.archive.write_gateway import WriteOperation
from polylogue.daemon.event_bus import EventBus, IngestCommitted, reset_daemon_event_bus
from polylogue.daemon.periodic import PeriodicRunner


@pytest.fixture(autouse=True)
def _fresh_bus() -> object:
    reset_daemon_event_bus()
    yield
    reset_daemon_event_bus()


def _archive(tmp_path: Path) -> sqlite3.Connection:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    conn = sqlite3.connect(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    return conn


def test_a_committed_write_announces_exactly_one_typed_event(tmp_path: Path) -> None:
    """The producer publishes after commit, with the committed session refs.

    Anti-vacuity: remove the ``announce_ingest_committed`` entry from
    ``WRITE_EFFECT_REGISTRY`` and ``seen`` stays empty.
    """
    from polylogue.daemon.event_bus import daemon_event_bus

    seen: list[IngestCommitted] = []
    daemon_event_bus().subscribe(IngestCommitted, seen.append)

    conn = _archive(tmp_path)
    try:
        commit_archive_write_effects(
            conn,
            WriteOperation.INGEST,
            {
                "changed_session_ids": ("origin:one", "origin:two"),
                "repair_message_fts": False,
                "_db_path": str(tmp_path / "index.db"),
            },
        )
    finally:
        conn.close()

    assert len(seen) == 1
    assert seen[0].session_refs == ("origin:one", "origin:two")


def test_a_write_that_changed_nothing_announces_nothing(tmp_path: Path) -> None:
    """An announcement with no session refs would wake every consumer for free.

    Anti-vacuity: drop ``should_run`` from the effect and ``seen`` gains an
    event with an empty ``session_refs``.
    """
    from polylogue.daemon.event_bus import daemon_event_bus

    seen: list[IngestCommitted] = []
    daemon_event_bus().subscribe(IngestCommitted, seen.append)

    conn = _archive(tmp_path)
    try:
        commit_archive_write_effects(conn, WriteOperation.INGEST, {"changed_session_ids": ()})
    finally:
        conn.close()

    assert seen == []


@pytest.mark.asyncio
async def test_the_event_wakes_the_consumer_without_waiting_out_the_poll() -> None:
    """A woken pass must not wait for the slow interval to elapse.

    Anti-vacuity: pass ``wakeup=None`` to ``PeriodicRunner.run`` and the second
    pass only arrives after the full 3600s sleep, so ``wait_for`` times out.
    """
    slept: list[float] = []
    released = asyncio.Event()

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        # A cadence this long never elapses inside the test: only the event can
        # produce the second pass.
        await released.wait()

    runner = PeriodicRunner(jitter_ratio=0.0, rng=random.Random(0), sleep=sleep)
    wakeup = asyncio.Event()
    bus = EventBus()
    bus.subscribe(IngestCommitted, lambda _event: wakeup.set())

    passes = asyncio.Queue[int]()
    count = 0

    async def work() -> None:
        nonlocal count
        count += 1
        await passes.put(count)

    task = asyncio.create_task(runner.run("embedding_backlog", work, interval_s=3600.0, wakeup=wakeup, run_first=True))
    try:
        assert await asyncio.wait_for(passes.get(), timeout=2) == 1
        bus.publish(IngestCommitted(cursor=None, session_refs=("origin:one",)))
        assert await asyncio.wait_for(passes.get(), timeout=2) == 2
    finally:
        released.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    state = runner.state("embedding_backlog")
    assert state is not None
    assert state.wakeups == 1
    assert slept and slept[0] == 3600.0, "the reconciliation tick must still be the declared interval"
    assert not wakeup.is_set(), "a consumed wakeup must be cleared so the next one is observable"


@pytest.mark.asyncio
async def test_a_dropped_event_still_converges_on_the_reconciliation_tick() -> None:
    """In-process delivery is an optimization; the cadence remains authority.

    Anti-vacuity: make the runner wait only on the event and the second pass
    never arrives, because nothing publishes here.
    """
    ticks = asyncio.Queue[float]()

    async def sleep(seconds: float) -> None:
        await ticks.put(seconds)
        await asyncio.sleep(0)

    runner = PeriodicRunner(jitter_ratio=0.0, rng=random.Random(0), sleep=sleep)
    passes = asyncio.Queue[int]()
    count = 0

    async def work() -> None:
        nonlocal count
        count += 1
        await passes.put(count)

    task = asyncio.create_task(
        runner.run("embedding_backlog", work, interval_s=60.0, wakeup=asyncio.Event(), run_first=True)
    )
    try:
        assert await asyncio.wait_for(passes.get(), timeout=2) == 1
        assert await asyncio.wait_for(passes.get(), timeout=2) == 2
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    state = runner.state("embedding_backlog")
    assert state is not None
    assert state.wakeups == 0
    assert await ticks.get() == 60.0
