"""The intake loop settles cold builds after complete quiescent discovery."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.core.enums import Provider, Role
from polylogue.daemon.intake import (
    AdmissionOutcome,
    AdmissionResult,
    FairIntakeDispatcher,
    IntakeClassSpec,
    IntakeItem,
)
from polylogue.operations.intake_adapters import ColdBuildSettlement, DaemonIntakeService
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import ColdBuildGeneration, active_index_generation_is_empty
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from tests.infra.index_writer import fixture_index_connection, write_fixture_index_session


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

    def schedulable_classes(self) -> tuple[()]:
        return ()


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
        if fired or dispatcher.calls >= len(script):
            break
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return dispatcher


def test_initial_quiescent_pass_settles_without_a_watched_admission() -> None:
    """An operation-written candidate does not need watched-file progress."""
    fired: list[int] = []
    asyncio.run(_run_passes([_Pass(False), _Pass(True), _Pass(False)], fired))
    assert fired == [1]


def test_external_candidate_write_settles_when_watched_input_is_excluded(tmp_path: Path) -> None:
    """External writes remain eligible when fair intake admits no sessions."""

    class ExcludedHistory:
        acknowledged = False
        admissions = 0

        async def discover(self, *, limit: int) -> tuple[IntakeItem, ...]:
            return () if self.acknowledged or limit < 1 else (IntakeItem("history", "local"),)

        async def admit(self, item: IntakeItem) -> AdmissionResult:
            self.admissions += 1
            return AdmissionResult(AdmissionOutcome.EXCLUDED, reason="no_sessions")

        async def acknowledge(self, item: IntakeItem) -> None:
            self.acknowledged = True

    with fixture_index_connection(tmp_path / "candidate" / "index.db") as conn:
        write_fixture_index_session(
            conn,
            ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id="external-fixture",
                messages=[ParsedMessage(provider_message_id="fixture-message", role=Role.USER, text="fixture message")],
            ),
        )
        conn.commit()
        counts: list[int] = []
        history = ExcludedHistory()

        async def scenario() -> None:
            dispatcher = FairIntakeDispatcher((IntakeClassSpec(name="local", adapter=history),))
            settled = asyncio.Event()

            def drained() -> None:
                counts.append(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
                settled.set()

            service = DaemonIntakeService(dispatcher, idle_delay_s=0.05, on_backlog_drained=drained)
            task = asyncio.create_task(service.run())
            try:
                async with asyncio.timeout(1):
                    await settled.wait()
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        asyncio.run(scenario())

    assert counts == [1]
    assert history.admissions == 1
    assert history.acknowledged


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
    """A pass with unresolved discovery cannot settle a build."""
    fired: list[int] = []
    asyncio.run(_run_passes([_Pass(False, False), _Pass(False, False), _Pass(False, False)], fired))
    assert fired == []


def test_empty_no_input_build_discards_after_quiescent_discovery(tmp_path: Path) -> None:
    """A settled empty candidate retires without replacing the active index."""
    assert active_index_generation_is_empty(tmp_path)
    generation = ColdBuildGeneration.begin(
        tmp_path,
        reason="empty active index generation",
        observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", tmp_path / "absent-source"),)),
    )
    generation_root = generation.generation_root

    async def run_idle() -> None:
        dispatcher = _ScriptedDispatcher([_Pass(False), _Pass(False)])

        def discard_empty_candidate() -> None:
            generation.discard()

        service = DaemonIntakeService(
            cast(Any, dispatcher),
            idle_delay_s=0.05,
            on_backlog_drained=discard_empty_candidate,
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
        assert generation.discarded
        assert active_index_generation_is_empty(tmp_path)
    finally:
        # Keep failed scenarios from leaving an owned inactive candidate.
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


@pytest.mark.uses_real_clock("the service retry is due on the asyncio monotonic clock")
def test_retryable_settlement_retries_in_the_same_service() -> None:
    """An escaped callback or old one-shot cleanup makes this test red."""

    async def scenario() -> int:
        dispatcher = _ScriptedDispatcher([_Pass(True), _Pass(False), _Pass(False)])
        calls = 0

        def drained() -> ColdBuildSettlement:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ColdBuildSettlement("retryable", "sqlite_busy", calls, time.monotonic() + 0.05)
            return ColdBuildSettlement("complete", attempts=calls)

        service = DaemonIntakeService(cast(Any, dispatcher), idle_delay_s=0.05, on_backlog_drained=drained)
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


def test_blocked_settlement_waits_for_new_source_revision() -> None:
    async def scenario() -> int:
        dispatcher = _ScriptedDispatcher([_Pass(True), _Pass(False)])
        revision = 0
        calls = 0

        def drained() -> ColdBuildSettlement:
            nonlocal calls
            calls += 1
            return ColdBuildSettlement("blocked", "source_integrity", calls)

        service = DaemonIntakeService(
            cast(Any, dispatcher),
            idle_delay_s=0.05,
            on_backlog_drained=drained,
            settlement_revision=lambda: (revision,),
        )
        task = asyncio.create_task(service.run())
        try:
            async with asyncio.timeout(1):
                while calls < 1:
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0.16)
            assert calls == 1
            revision += 1
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


def test_blocked_settlement_preserves_external_change_during_callback() -> None:
    async def scenario() -> int:
        dispatcher = _ScriptedDispatcher([_Pass(True), _Pass(False)])
        revision = 0
        calls = 0

        def drained() -> ColdBuildSettlement:
            nonlocal revision, calls
            calls += 1
            if calls == 1:
                # The source is repaired after the observation but before
                # the callback reports its blocked verdict.
                revision += 1
                return ColdBuildSettlement("blocked", "source_integrity", calls)
            return ColdBuildSettlement("complete", attempts=calls)

        service = DaemonIntakeService(
            cast(Any, dispatcher),
            idle_delay_s=0.05,
            on_backlog_drained=drained,
            settlement_revision=lambda: (revision,),
            settlement_external_revision=lambda: (revision,),
        )
        task = asyncio.create_task(service.run())
        try:
            async with asyncio.timeout(1):
                while calls < 2:
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0.12)
            assert calls == 2
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
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
