"""One runner owns every daemon cadence loop's scheduling and observability.

Each daemon periodic service used to carry its own ``while True`` with its own
sleep order, existence guard, exception policy and gate wait, and none of them
recorded when they last ran or what last failed (polylogue-74wvj). These tests
pin the behaviours those seventeen loops collectively needed, so the deleted
bodies are covered by the runner rather than by nothing.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

from polylogue.daemon.periodic import PeriodicGate, PeriodicRunner
from polylogue.daemon.services import ServiceTrigger, service_spec, service_specs


class _StepClock:
    """Deterministic clock and sleep: no wall-clock read, no real delay."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


def _runner(clock: _StepClock, *, jitter_ratio: float = 0.0) -> PeriodicRunner:
    return PeriodicRunner(
        jitter_ratio=jitter_ratio,
        rng=random.Random(0),
        sleep=clock.sleep,
        clock=clock.time,
    )


async def _drive(runner: PeriodicRunner, name: str, *, passes: int, **kwargs: Any) -> None:
    """Run one loop until it has taken ``passes`` ticks, then cancel it."""
    done = asyncio.Event()
    count = 0
    inner = kwargs.pop("work")

    async def work() -> None:
        nonlocal count
        count += 1
        try:
            await inner()
        finally:
            if count >= passes:
                done.set()

    task = asyncio.create_task(runner.run(name, work, **kwargs))
    try:
        await asyncio.wait_for(done.wait(), timeout=2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_run_first_publishes_before_the_first_sleep() -> None:
    """The status-snapshot and convergence loops must not idle one cadence first.

    Anti-vacuity: drop the ``run_first`` branch from ``PeriodicRunner.run`` and
    the first recorded sleep moves before the first pass, so ``order`` starts
    with ``"sleep"``.
    """
    clock = _StepClock()
    runner = _runner(clock)
    order: list[str] = []

    async def work() -> None:
        order.append("work")

    async def sleep(seconds: float) -> None:
        order.append("sleep")
        await clock.sleep(seconds)

    runner._sleep = sleep
    await _drive(runner, "status_snapshot_refresh", passes=2, work=work, interval_s=10.0, run_first=True)

    assert order[:3] == ["work", "sleep", "work"]


@pytest.mark.asyncio
async def test_sleep_first_loops_wait_one_interval_before_their_first_pass() -> None:
    """WAL checkpoint and the daily optimize must not add a startup IO burst.

    Anti-vacuity: default ``run_first`` to True and the first element of
    ``order`` becomes ``"work"``.
    """
    clock = _StepClock()
    runner = _runner(clock)
    order: list[str] = []

    async def work() -> None:
        order.append("work")

    async def sleep(seconds: float) -> None:
        order.append("sleep")
        await clock.sleep(seconds)

    runner._sleep = sleep
    await _drive(runner, "wal_checkpoint", passes=1, work=work, interval_s=300.0)

    assert order[:2] == ["sleep", "work"]
    assert clock.slept[0] == 300.0


@pytest.mark.asyncio
async def test_a_false_precondition_is_a_recorded_skip_not_a_silent_continue() -> None:
    """``if not db.exists(): continue`` left no evidence at all in any loop.

    Anti-vacuity: make the runner run ``work`` regardless of the precondition
    and ``runs`` becomes non-zero while ``skips`` stays at zero.
    """
    clock = _StepClock()
    runner = _runner(clock)
    ran = 0
    ticks = 0

    async def work() -> None:
        nonlocal ran
        ran += 1

    def precondition() -> bool:
        nonlocal ticks
        ticks += 1
        return False

    task = asyncio.create_task(
        runner.run("fts_merge", work, interval_s=60.0, precondition=precondition, run_first=True)
    )
    for _ in range(50):
        await asyncio.sleep(0)
        if ticks >= 3:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    state = runner.state("fts_merge")
    assert state is not None
    assert ran == 0
    assert state.runs == 0
    assert state.skips >= 3
    assert state.next_run_at is not None


@pytest.mark.asyncio
async def test_a_recorded_failure_keeps_the_cadence_and_names_itself() -> None:
    """Every converted loop swallowed its exception; none of them recorded it.

    Anti-vacuity: let the runner propagate by default and the loop dies on the
    first failure, so the second pass never happens and ``runs`` stays at zero.
    """
    clock = _StepClock()
    runner = _runner(clock)
    passes = 0

    async def work() -> None:
        nonlocal passes
        passes += 1
        if passes == 1:
            raise RuntimeError("archive briefly unavailable")

    await _drive(runner, "heartbeat", passes=2, work=work, interval_s=900.0, run_first=True)

    state = runner.state("heartbeat")
    assert state is not None
    assert state.failures == 1
    assert state.runs == 1, "the loop must keep ticking after a recorded failure"
    assert state.last_error_type == "RuntimeError"
    assert "archive briefly unavailable" in str(state.last_error)


@pytest.mark.asyncio
async def test_propagate_lets_the_schema_recovery_signal_reach_the_supervisor() -> None:
    """Schema-preflight recovery ends its loop on purpose: the daemon restarts.

    Anti-vacuity: record instead of propagating and ``runner.run`` never
    returns the exception, so ``pytest.raises`` fails.
    """
    clock = _StepClock()
    runner = _runner(clock)

    async def work() -> None:
        raise RuntimeError("schema preflight recovered; restart required")

    with pytest.raises(RuntimeError, match="restart required"):
        await runner.run("schema_preflight_recheck", work, interval_s=60.0, run_first=True, on_error="propagate")

    state = runner.state("schema_preflight_recheck")
    assert state is not None and state.failures == 1


@pytest.mark.asyncio
async def test_a_blocked_loop_names_its_gate_and_an_idle_one_does_not() -> None:
    """Idle and stalled were indistinguishable behind a bare ``asyncio.Event``.

    Anti-vacuity: stop setting ``blocked_on``/``blocked_since`` and the blocked
    assertion below reads the same as the released one.
    """
    clock = _StepClock()
    runner = _runner(clock)
    event = asyncio.Event()
    gate = PeriodicGate(name="watcher_registered", event=event, timeout_s=30.0)
    started = asyncio.Event()

    async def work() -> None:
        started.set()

    task = asyncio.create_task(runner.run("convergence_check", work, interval_s=60.0, gate=gate, run_first=True))
    await asyncio.sleep(0)
    blocked = runner.state("convergence_check")
    assert blocked is not None
    assert blocked.blocked_on == "watcher_registered"
    assert blocked.blocked_since is not None
    assert gate.waiting == {"convergence_check"}
    assert blocked.runs == 0

    event.set()
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    released = runner.state("convergence_check")
    assert released is not None
    assert released.blocked_on is None and released.blocked_since is None
    assert gate.waiting == set()


@pytest.mark.asyncio
async def test_jitter_desynchronises_loops_started_in_one_composition_tick() -> None:
    """Seventeen loops registered together would otherwise stay phase-locked.

    Anti-vacuity: set the jitter term to zero in ``run`` and every recorded
    sleep equals the interval exactly, collapsing the assertion below.
    """
    clock = _StepClock()
    runner = _runner(clock, jitter_ratio=0.1)

    async def work() -> None:
        return None

    await _drive(runner, "blob_gc", passes=3, work=work, interval_s=100.0)

    assert all(100.0 <= slept <= 110.0 for slept in clock.slept)
    assert len(set(clock.slept)) > 1, "a jittered cadence must not repeat one exact delay"


@pytest.mark.asyncio
async def test_the_payload_surfaces_are_fed_from_one_snapshot() -> None:
    """Status and metrics read the same per-loop record, not two ledgers."""
    clock = _StepClock()
    runner = _runner(clock)

    async def work() -> None:
        return None

    await _drive(runner, "secret_scan_sweep", passes=1, work=work, interval_s=3600.0, run_first=True)

    loops = runner.payload()["loops"]
    assert isinstance(loops, list)
    row = next(entry for entry in loops if entry["name"] == "secret_scan_sweep")
    assert row["interval_s"] == 3600.0
    assert row["runs"] == 1
    assert row["last_run_completed_at"] is not None
    assert row["next_run_at"] is not None
    assert row["last_error"] is None


def test_every_declared_periodic_service_has_a_cadence() -> None:
    """The registry is where a loop's interval is declared, so it must be there.

    Anti-vacuity: drop ``cadence_s`` from any PERIODIC spec and this fails by
    name.
    """
    missing = [
        spec.name for spec in service_specs() if spec.trigger is ServiceTrigger.PERIODIC and spec.cadence_s is None
    ]
    assert missing == []


def test_declared_cadence_matches_the_interval_each_loop_registers() -> None:
    """A spec's cadence and the constant its loop registers are one fact.

    Eight of the sixteen periodic specs had drifted from their loop bodies while
    each loop owned its own literal. Anti-vacuity: change either side of any
    pair below without the other and this fails, naming the service.
    """
    from polylogue.daemon import blob_gc_periodic, embedding_backlog, secret_scan_sweep
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.lifecycle import DAEMON_HEARTBEAT_INTERVAL_SECONDS

    registered = {
        "fts_merge": daemon_cli._FTS_MERGE_INTERVAL_SECONDS,
        "heartbeat": daemon_cli._HEARTBEAT_INTERVAL_SECONDS,
        "db_optimize": daemon_cli._DB_OPTIMIZE_INTERVAL_SECONDS,
        "wal_checkpoint": daemon_cli._WAL_CHECKPOINT_INTERVAL_SECONDS,
        "status_snapshot_refresh": daemon_cli._STATUS_SNAPSHOT_REFRESH_INTERVAL_SECONDS,
        "convergence_check": daemon_cli._CONVERGENCE_DEBT_RETRY_INTERVAL_SECONDS,
        "raw_observation_convergence": daemon_cli._RAW_MATERIALIZATION_CONVERGENCE_INTERVAL_SECONDS,
        "schema_preflight_recheck": daemon_cli._SCHEMA_PREFLIGHT_RECHECK_INTERVAL_SECONDS,
        "lifecycle_heartbeat": DAEMON_HEARTBEAT_INTERVAL_SECONDS,
        "blob_gc": blob_gc_periodic.BLOB_GC_INTERVAL_SECONDS,
        "blob_publication_reconciliation": blob_gc_periodic.BLOB_PUBLICATION_RECONCILIATION_INTERVAL_SECONDS,
        "embedding_backlog": embedding_backlog.EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS,
        "embedding_orphan_reconcile": embedding_backlog.EMBEDDING_ORPHAN_RECONCILE_INTERVAL_SECONDS,
        "secret_scan_sweep": secret_scan_sweep.SECRET_SCAN_SWEEP_INTERVAL_SECONDS,
    }
    mismatched = {
        service: (service_spec(service).cadence_s, float(interval))
        for service, interval in registered.items()
        if service_spec(service).cadence_s != float(interval)
    }
    assert mismatched == {}


def test_every_periodic_service_either_registers_a_cadence_or_is_a_named_residual() -> None:
    """The one loop still hand-rolled must be named, not silently missing.

    ``judgment_automation`` reloads config before its sleep and writes an async
    failure receipt on a reload failure, which the runner's synchronous
    ``interval_s`` contract cannot host; it stays a hand-rolled loop until that
    fallback is restructured. Anti-vacuity: convert it and this fails, which is
    the prompt to delete the exemption.
    """
    import inspect

    from polylogue.daemon import judgment_automation

    assert "while True:" in inspect.getsource(judgment_automation.periodic_judgment_automation_sweep)


# -- drained backlogs publish one terminal transition, not one per wake ------


class _PassBarrier:
    """A cadence wait the test opens one pass at a time.

    Every wait parks here until :meth:`release` is called, so the number of
    passes is a value the test sets rather than a consequence of wall-clock
    time elapsing. That is what lets the spin regression below be observed
    directly instead of inferred from a timeout.
    """

    def __init__(self) -> None:
        self.waits = 0
        self._open = asyncio.Event()

    async def sleep(self, _seconds: float) -> None:
        self.waits += 1
        await self._open.wait()
        self._open.clear()

    def release(self) -> None:
        self._open.set()


@pytest.mark.asyncio
async def test_a_drained_backlog_publishes_one_terminal_transition_per_cycle() -> None:
    """Repeated empty passes are one drain, not one drain each.

    polylogue-09rn is the shape this prevents: an already-empty backlog woken
    again and again, announcing completion every time. The edge -- "there was
    work, now there is none" -- is the terminal transition; the flat stretch
    after it is not.

    Anti-vacuity: drop the ``state.drained is not True`` guard in
    ``_record_pass_outcome`` and the first assertion reads 4 instead of 1,
    because every empty pass counts again. Executed.
    """
    from polylogue.daemon.periodic import PassOutcome

    barrier = _PassBarrier()
    runner = PeriodicRunner(jitter_ratio=0.0, rng=random.Random(0), sleep=barrier.sleep, clock=lambda: 0.0)
    outcomes: list[PassOutcome] = [PassOutcome.DRAINED] * 4
    passes = asyncio.Queue[int]()
    index = 0

    async def work() -> PassOutcome:
        nonlocal index
        outcome = outcomes[index]
        index += 1
        await passes.put(index)
        return outcome

    task = asyncio.create_task(runner.run("embedding_backlog", work, interval_s=3600.0, run_first=True))
    try:
        assert await asyncio.wait_for(passes.get(), timeout=2) == 1
        for expected in (2, 3, 4):
            barrier.release()
            assert await asyncio.wait_for(passes.get(), timeout=2) == expected

        state = runner.state("embedding_backlog")
        assert state is not None
        assert state.runs == 4
        assert state.drained is True
        assert state.drain_transitions == 1, "an empty backlog announced completion more than once"

        # A pass that does work re-arms the cycle, and the next empty pass is a
        # second, genuine terminal transition.
        outcomes.extend([PassOutcome.PROGRESSED, PassOutcome.DRAINED])
        for expected in (5, 6):
            barrier.release()
            assert await asyncio.wait_for(passes.get(), timeout=2) == expected
        assert state.drain_transitions == 2
        assert state.last_drained_at == 0.0
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_repeated_wakeups_on_a_drained_backlog_cannot_spin() -> None:
    """One wake, one pass -- the loop always returns to its cadence wait.

    The daemon wakes this loop from every committed ingest. When the backlog
    is already drained each of those wakes must cost exactly one pass and
    then park again; a consumed wakeup that stays set turns the cadence wait
    into a no-op and the loop runs continuously.

    Anti-vacuity: delete ``wakeup.clear()`` from
    ``PeriodicRunner._sleep_until_woken`` and the final assertion sees the
    pass count run away instead of holding at 3. Executed.
    """
    from polylogue.daemon.periodic import PassOutcome
    from tests.infra.daemon_service_harness import record_private_lifecycle_probe

    barrier = _PassBarrier()
    runner = PeriodicRunner(jitter_ratio=0.0, rng=random.Random(0), sleep=barrier.sleep, clock=lambda: 0.0)
    wakeup = asyncio.Event()
    passes = 0

    async def work() -> PassOutcome:
        nonlocal passes
        passes += 1
        return PassOutcome.DRAINED

    task = asyncio.create_task(runner.run("embedding_backlog", work, interval_s=3600.0, wakeup=wakeup, run_first=True))
    try:
        # Pass 1 is the startup pass; passes 2 and 3 are the two wakes.
        for _ in range(2):
            while barrier.waits == 0 or passes == 0:
                await asyncio.sleep(0)
            before = passes
            wakeup.set()
            while passes == before:
                await asyncio.sleep(0)

        assert passes == 3
        state = runner.state("embedding_backlog")
        assert state is not None
        assert state.wakeups == 2
        assert state.drain_transitions == 1, "two idle wakes announced two drains"

        # The loop is parked in its cadence wait, not looping. Turning the
        # event loop over cannot produce another pass without another wake.
        for _ in range(50):
            await asyncio.sleep(0)
        assert passes == 3, "the loop kept running without a wakeup or an elapsed cadence"
        assert not wakeup.is_set()
        record_private_lifecycle_probe(
            "idle-wake",
            {
                "service": "embedding_backlog",
                "passes": passes,
                "wakeups": state.wakeups,
                "unexpected_idle_passes": passes - 1 - state.wakeups,
                "drain_transitions": state.drain_transitions,
                "event_loop_turns_after_last_wake": 50,
                "historical_before": "unmeasured",
            },
        )
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
