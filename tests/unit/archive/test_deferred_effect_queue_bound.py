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


def _session_ctx(staleness_key: str, session_ids: tuple[str, ...]) -> WriteEffectContext:
    return WriteEffectContext(
        conn=sqlite3.connect(":memory:"),
        op="session_write",  # type: ignore[arg-type]
        payload={"_db_path": "/archive/index.db"},
        changed_session_ids=session_ids,
        staleness_key=staleness_key,
        run_archive_effects=False,
    )


def test_an_enqueue_during_a_failed_release_keeps_the_obligation() -> None:
    """A failure still marked pending is not cleared by a concurrent enqueue.

    Anti-vacuity: clear every failure at enqueue and resubmit it separately,
    and ``_submit`` rejects the retry because the key is still pending, so the
    obligation is gone once the worker releases the key.
    """
    queue = DeferredEffectQueue()
    effect = WriteEffect(name="invalidate", phase="async-deferred", run=lambda _ctx: None)
    ctx = _session_ctx("batch-1", ("s1",))
    key = queue._obligation_key(effect, ctx)
    with queue._lock:
        queue._failed[key] = (effect, ctx, frozenset({"s1"}))
        queue._pending.add(key)

    queue.enqueue(WriteEffect(name="other", phase="async-deferred", run=lambda _ctx: None), _ctx("batch-2"))
    queue._executor.shutdown(wait=True)

    assert queue.failed_count == 1


def test_a_sustained_outage_retains_one_coalesced_obligation() -> None:
    """Failures owed to one effect merge into one retry over the session union.

    Anti-vacuity: retain one entry per failed batch and ``failed_count`` grows
    with every write while each enqueue resubmits all of them.
    """
    queue = DeferredEffectQueue()
    seen: list[tuple[str, ...]] = []

    def unavailable(ctx: WriteEffectContext) -> None:
        seen.append(ctx.changed_session_ids)
        raise RuntimeError("index unavailable")

    effect = WriteEffect(name="invalidate", phase="async-deferred", run=unavailable)
    for index in range(40):
        queue.enqueue(effect, _session_ctx(f"batch-{index}", (f"s{index}",)))
        queue._executor.submit(lambda: None).result(5)
    queue._executor.shutdown(wait=True)

    assert queue.failed_count == 1
    [(_effect, _ctx, owed)] = queue._failed.values()
    assert owed == frozenset(f"s{index}" for index in range(40))
    # Each enqueue runs at most one retry beside its own batch.
    assert len(seen) <= 2 * 40
