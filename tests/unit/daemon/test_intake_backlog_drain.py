"""The intake loop promotes a cold build only after observed work drains."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from polylogue.operations.intake_adapters import DaemonIntakeService
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import ColdBuildGeneration, active_index_generation_is_empty


@dataclass
class _Pass:
    progressed: bool
    quiescent: bool = True


class _ScriptedDispatcher:
    def __init__(self, script: list[_Pass]) -> None:
        self._script = script
        self.calls = 0

    async def run_once(self, *, budget: int) -> _Pass:
        del budget
        result = self._script[self.calls] if self.calls < len(self._script) else _Pass(False)
        self.calls += 1
        return result


async def _run_passes(script: list[_Pass], fired: list[int]) -> _ScriptedDispatcher:
    dispatcher = _ScriptedDispatcher(script)

    def drained() -> None:
        fired.append(dispatcher.calls)

    service = DaemonIntakeService(
        cast(Any, dispatcher),
        idle_delay_s=0.05,
        on_backlog_drained=drained,
    )
    task = asyncio.create_task(service.run())
    for _ in range(40):
        await asyncio.sleep(0.01)
        if fired or dispatcher.calls > len(script) + 2:
            break
    task.cancel()
    return dispatcher


def test_the_drain_signal_waits_for_a_pass_that_did_something() -> None:
    """An idle startup is not a drained backlog.

    Anti-vacuity: firing the callback on any non-progressing pass makes the
    recorded pass index 1 instead of 3.
    """
    fired: list[int] = []
    asyncio.run(_run_passes([_Pass(False), _Pass(True), _Pass(False), _Pass(False)], fired))
    assert fired == [3]


def test_the_drain_signal_fires_once() -> None:
    """A later backlog is ordinary live ingest, not a second cold build.

    Anti-vacuity: leaving ``_on_backlog_drained`` set after the first call
    makes this list longer than one element.
    """
    fired: list[int] = []
    asyncio.run(_run_passes([_Pass(True), _Pass(False), _Pass(True), _Pass(False), _Pass(False)], fired))
    assert fired == [2]


def test_deferred_or_unmeasured_pass_does_not_promote_candidate() -> None:
    """A no-admission pass can still owe retry or further discovery work.

    Anti-vacuity: using only ``progressed`` promotes on pass 2, before the
    scripted deferred and full-page reports have drained.
    """
    fired: list[int] = []
    asyncio.run(_run_passes([_Pass(True), _Pass(False, False), _Pass(False, False), _Pass(False)], fired))
    assert fired == [4]


def test_deferred_only_build_never_claims_completed_backlog() -> None:
    """Without a successful admission, a deferred source cannot settle a build."""
    fired: list[int] = []
    asyncio.run(_run_passes([_Pass(False, False), _Pass(False), _Pass(False)], fired))
    assert fired == []


def test_empty_no_input_build_waits_for_future_files_then_discards_on_shutdown(tmp_path: Path) -> None:
    """An empty startup leaves the candidate available until daemon shutdown."""
    assert active_index_generation_is_empty(tmp_path)
    generation = ColdBuildGeneration.begin(
        tmp_path,
        reason="empty active index generation",
        sources=(WatchSource("fixture", tmp_path / "absent-source"),),
    )
    generation_root = generation.generation_root

    async def run_idle() -> None:
        dispatcher = _ScriptedDispatcher([_Pass(False), _Pass(False)])

        def premature_promotion() -> None:
            generation.promote()

        service = DaemonIntakeService(
            cast(Any, dispatcher),
            idle_delay_s=0.05,
            on_backlog_drained=premature_promotion,
        )
        task = asyncio.create_task(service.run())
        try:
            async with asyncio.timeout(1):
                while dispatcher.calls < 2:
                    await asyncio.sleep(0.01)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    try:
        asyncio.run(run_idle())
        assert generation.generation.state == "inactive"
        assert not generation.settled
    finally:
        # This is the daemon's shutdown action for an unpromoted candidate.
        generation.discard()
    assert not generation_root.exists()


def test_durable_pending_retry_blocks_promotion_after_quiescent_pass() -> None:
    """A retry with a future due time is invisible to the current page."""

    async def scenario() -> list[int]:
        dispatcher = _ScriptedDispatcher([_Pass(True), _Pass(False), _Pass(False)])
        fired: list[int] = []
        pending_checks = 0

        def pending() -> bool:
            nonlocal pending_checks
            pending_checks += 1
            return pending_checks == 1

        service = DaemonIntakeService(
            cast(Any, dispatcher),
            idle_delay_s=0.05,
            on_backlog_drained=lambda: fired.append(dispatcher.calls),
            has_pending_backlog=pending,
        )
        task = asyncio.create_task(service.run())
        try:
            async with asyncio.timeout(1):
                while not fired:
                    await asyncio.sleep(0.01)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return fired

    assert asyncio.run(scenario()) == [3]


def test_failed_promotion_keeps_callback_for_retry() -> None:
    """A failed readiness pass leaves the same candidate eligible to settle."""

    async def scenario() -> int:
        dispatcher = _ScriptedDispatcher([_Pass(True), _Pass(False), _Pass(False)])
        calls = 0

        def drained() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("readiness failed")

        service = DaemonIntakeService(cast(Any, dispatcher), idle_delay_s=0.05, on_backlog_drained=drained)
        try:
            await service.run()
        except RuntimeError:
            pass
        else:
            raise AssertionError("readiness failure was swallowed")
        assert service._on_backlog_drained is drained
        task = asyncio.create_task(service.run())
        try:
            async with asyncio.timeout(1):
                while calls < 2:
                    await asyncio.sleep(0.01)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return calls

    assert asyncio.run(scenario()) == 2


def test_candidate_progress_refresh_runs_only_after_admission() -> None:
    """An idle discovery pass does not rescan the candidate for ETA."""

    async def scenario() -> list[int]:
        dispatcher = _ScriptedDispatcher([_Pass(False), _Pass(True), _Pass(False), _Pass(True)])
        refreshed: list[int] = []
        service = DaemonIntakeService(
            cast(Any, dispatcher),
            idle_delay_s=0.05,
            on_pass_complete=lambda _result: refreshed.append(dispatcher.calls),
        )
        task = asyncio.create_task(service.run())
        try:
            async with asyncio.timeout(1):
                while dispatcher.calls < 4:
                    await asyncio.sleep(0.01)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return refreshed

    assert asyncio.run(scenario()) == [2, 4]
