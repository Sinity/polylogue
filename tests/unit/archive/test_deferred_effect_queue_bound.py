"""The deferred write-effect queue retains only in-flight work (polylogue-yooge)."""

from __future__ import annotations

import sqlite3
import threading

from polylogue.archive.write_effects import DeferredEffectQueue, WriteEffect, WriteEffectContext


def _ctx(staleness_key: str) -> WriteEffectContext:
    return WriteEffectContext(
        conn=sqlite3.connect(":memory:"),
        op="session_write",  # type: ignore[arg-type]
        payload={},
        changed_session_ids=(),
        staleness_key=staleness_key,
        run_archive_effects=False,
    )


def test_settled_deliveries_leave_nothing_behind() -> None:
    """Anti-vacuity: retain a record per settled key and pending_count grows with every batch."""
    queue = DeferredEffectQueue()
    delivered: list[str] = []
    done = threading.Event()

    def run(ctx: WriteEffectContext) -> None:
        delivered.append(ctx.staleness_key)
        if len(delivered) == 200:
            done.set()

    effect = WriteEffect(name="probe", phase="async-deferred", run=run)
    for index in range(200):
        queue.enqueue(effect, _ctx(f"batch-{index}"))
    assert done.wait(10)
    queue._executor.shutdown(wait=True)

    assert len(delivered) == 200
    assert queue.pending_count == 0


def test_in_flight_duplicate_is_not_delivered_twice() -> None:
    queue = DeferredEffectQueue()
    gate = threading.Event()
    calls: list[str] = []

    def run(ctx: WriteEffectContext) -> None:
        calls.append(ctx.staleness_key)
        gate.wait(5)

    effect = WriteEffect(name="probe", phase="async-deferred", run=run)
    queue.enqueue(effect, _ctx("same"))
    queue.enqueue(effect, _ctx("same"))
    gate.set()
    queue._executor.shutdown(wait=True)

    assert calls == ["same"]
    assert queue.pending_count == 0


def test_failed_delivery_is_retained_and_retried() -> None:
    """Anti-vacuity: drop the failed map and the committed invalidation is never retried."""
    queue = DeferredEffectQueue()
    attempts: list[str] = []
    fail_once = {"armed": True}

    def flaky(ctx: WriteEffectContext) -> None:
        attempts.append(ctx.staleness_key)
        if fail_once["armed"]:
            fail_once["armed"] = False
            raise RuntimeError("connection unavailable")

    flaky_effect = WriteEffect(name="invalidate", phase="async-deferred", run=flaky)
    queue.enqueue(flaky_effect, _ctx("batch-1"))
    queue._executor.submit(lambda: None).result(5)
    assert queue.failed_count == 1

    queue.enqueue(WriteEffect(name="other", phase="async-deferred", run=lambda _ctx: None), _ctx("batch-2"))
    queue._executor.shutdown(wait=True)

    assert attempts == ["batch-1", "batch-1"]
    assert queue.failed_count == 0
    assert queue.pending_count == 0
