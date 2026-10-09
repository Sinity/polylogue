"""Repeated supervisor cancellation cannot retire a running convergence owner."""

from __future__ import annotations

import asyncio
import threading
from builtins import BaseExceptionGroup
from types import SimpleNamespace
from typing import cast

import pytest

from polylogue.core.compute import BoundedComputeAdapter
from polylogue.daemon import convergence
from polylogue.daemon.convergence import DaemonConverger, SelectedSessionTarget, SessionProfileConvergenceOwner
from polylogue.daemon.derivation import DerivationFrame, DerivationReport
from polylogue.daemon.services import ServiceCapability, ServiceState
from polylogue.daemon.supervisor import DaemonSupervisor
from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge


@pytest.mark.contract
@pytest.mark.uses_real_clock("worker settlement and supervisor cancellation use OS scheduling")
@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("cleanup_failed", [False, True])
async def test_repeated_supervisor_cancel_retains_convergence_lock_until_worker_settles(
    monkeypatch: pytest.MonkeyPatch, selected: bool, cleanup_failed: bool
) -> None:
    """A single suppressed drain await releases the lock on the second cancel."""
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    physical_calls: list[int] = []
    cleanup_calls: list[int] = []
    frame = DerivationFrame("synthetic", "generation", {"session_profile": "recipe"}, scope=("session",))
    converger = DaemonConverger(())

    def physical(*args: object, **kwargs: object) -> object:
        physical_calls.append(1)
        loop.call_soon_threadsafe(entered.set)
        release.wait()
        return () if selected else DerivationReport(frame)

    if selected:
        adapter = SimpleNamespace(
            recipe_version="recipe",
            selected_part_facts=lambda *args: None,
            selected_frame_is_current=lambda *args: True,
            quiet=lambda *args: False,
            compute=lambda *args: None,
            publish=lambda *args: True,
        )
        monkeypatch.setattr(converger, "_derivation_adapter", lambda domain: adapter)
        monkeypatch.setattr(convergence, "_converge_selected_session_parts_sync", physical)
    else:
        monkeypatch.setattr(converger, "converge_derivations", physical)
    pool = BoundedComputeAdapter(max_workers=2, queue_units=2)

    def settle(*args: object, **kwargs: object) -> BaseException | None:
        cleanup_calls.append(1)
        return RuntimeError("synthetic cleanup failed") if cleanup_failed and len(cleanup_calls) == 1 else None

    monkeypatch.setattr(pool, "_settle_native_sql", settle)
    owner = SessionProfileConvergenceOwner(
        converger, compute_adapter=pool, write_bridge=cast("DaemonWriteThreadBridge", SimpleNamespace())
    )

    async def run_owner() -> None:
        if selected:
            await owner.converge_selected(
                frame,
                targets=(SelectedSessionTarget("session", "required"),),
                expected_generation="generation",
                expected_recipe="recipe",
                stop_requested=lambda: None,
            )
        else:
            await owner.converge(frame)

    supervisor = DaemonSupervisor(capabilities=(ServiceCapability.DERIVED_WRITES,))
    child = supervisor.start("convergence_check", run_owner)
    assert child is not None
    waiting = asyncio.create_task(supervisor.wait())
    shutdown: asyncio.Task[object] | None = None
    competing: asyncio.Task[None] | None = None
    try:
        await entered.wait()
        waiting.cancel()
        await asyncio.sleep(0)
        shutdown = asyncio.create_task(supervisor.shutdown())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        competing = asyncio.create_task(run_owner())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not child.done()
        assert owner._converge_lock.locked()
        assert physical_calls == [1]
        assert pool.snapshot().active_units == 1
        release.set()
        await shutdown
        try:
            await waiting
        except asyncio.CancelledError:
            pass
        await competing
        assert physical_calls == [1, 1]
        assert not owner._converge_lock.locked()
        assert pool.snapshot().active_units == 0
        failure = supervisor.failure("convergence_check")
        if cleanup_failed:
            assert supervisor.state("convergence_check") is ServiceState.FAILED
            assert isinstance(failure, BaseExceptionGroup)
            assert failure.subgroup(asyncio.CancelledError) is not None
            assert failure.subgroup(RuntimeError) is not None
        else:
            assert supervisor.state("convergence_check") is ServiceState.STOPPED
            assert failure is None
    finally:
        release.set()
        await asyncio.gather(
            *(task for task in (waiting, shutdown, competing, child) if task is not None), return_exceptions=True
        )
        pool.shutdown(wait=True)
