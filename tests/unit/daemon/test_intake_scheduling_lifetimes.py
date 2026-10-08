"""Polling deadlines, bounded scan progress and process-local halt lifetimes."""

from __future__ import annotations

import asyncio
import errno
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

# Import the frame's recipe bindings before replacing its derivation in a test.
import polylogue.storage.derived.raw as raw_inspection
from polylogue.core.degraded import DegradedReason
from polylogue.core.source_halts import clear_all_source_halts, set_source_halt
from polylogue.daemon.catchup_status import _halted_sources
from polylogue.daemon.intake import FairIntakeDispatcher, IntakeClassSpec
from polylogue.daemon.service_halt import HaltReason, HaltRegistry, UnitKind, unit_id
from polylogue.operations.drive_readiness import DriveCatchupReport, DriveCatchupState
from polylogue.operations.intake_adapters import (
    DaemonIntakeContext,
    DaemonIntakeService,
    RawMaterializationDiscovery,
    build_intake_adapters,
)
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop
from tests.infra.frozen_clock import FrozenClock


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [0, 1])
async def test_composed_remote_poll_waits_an_hour_after_completion(
    tmp_path: Path, frozen_clock: FrozenClock, changed: int
) -> None:
    calls = 0

    def poll() -> DriveCatchupReport:
        nonlocal calls
        calls += 1
        frozen_clock.advance(30)  # The interval starts after this work finishes.
        return DriveCatchupReport(DriveCatchupState.COMPLETE, changed_count=changed)

    pairs = build_intake_adapters(DaemonIntakeContext(tmp_path, cast(Any, None), ()), remote_callback=poll)
    dispatcher = FairIntakeDispatcher(
        tuple(IntakeClassSpec(name, adapter) for name, adapter in pairs), clock=frozen_clock.monotonic
    )
    await dispatcher.run_once()
    assert calls == 1
    for _ in range(3):
        frozen_clock.advance(5)
        await dispatcher.run_once()
    frozen_clock.advance(3584)
    await dispatcher.run_once()
    assert calls == 1
    frozen_clock.advance(1)
    await dispatcher.run_once()
    assert calls == 2


@pytest.mark.asyncio
async def test_failed_remote_poll_retries_before_the_normal_poll_deadline(
    tmp_path: Path, frozen_clock: FrozenClock
) -> None:
    calls = 0

    def poll() -> DriveCatchupReport:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise BlockingIOError(errno.EAGAIN, "synthetic temporary unavailability")
        return DriveCatchupReport(DriveCatchupState.COMPLETE)

    pairs = build_intake_adapters(DaemonIntakeContext(tmp_path, cast(Any, None), ()), remote_callback=poll)
    dispatcher = FairIntakeDispatcher(
        tuple(IntakeClassSpec(name, adapter) for name, adapter in pairs), clock=frozen_clock.monotonic
    )
    await dispatcher.run_once()
    frozen_clock.advance(60)
    await dispatcher.run_once()
    assert calls == 2
    frozen_clock.advance(5)
    await dispatcher.run_once()
    assert calls == 2


@pytest.mark.asyncio
async def test_valid_only_raw_pages_keep_the_service_moving_then_become_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: FrozenClock
) -> None:
    # Bootstrap takes a synchronous lease; run it off the event loop.
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    calls: list[tuple[str | None, int]] = []
    waits: list[float] = []

    class ValidPages:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def required_page(
            self, _frame: object, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            calls.append((cursor, limit))
            start = int(cursor or 0)
            end = min(start + limit, 96)
            return tuple(f"valid-{index}" for index in range(start, end)), str(end) if end < 96 else None

        def inspect(self, _frame: object, keys: Sequence[str]) -> dict[str, str]:
            return dict.fromkeys(keys, "valid")

        def terminal_decode_refusals(self, _keys: Sequence[str]) -> dict[str, Exception]:
            # No retained raw in this fixture refuses to decode.
            return {}

    monkeypatch.setattr(raw_inspection, "RawObservationInspection", ValidPages)
    discovery = RawMaterializationDiscovery(tmp_path)

    async def discover(limit: int) -> tuple[tuple[str, int], ...]:
        return discovery.discover_pending_raw_ids(limit)

    def unexpected_admission(_raw_id: str) -> int:
        raise AssertionError("valid evidence was selected for admission")

    pairs = build_intake_adapters(
        DaemonIntakeContext(tmp_path, cast(Any, None), ()),
        raw_discover=discover,
        raw_callback=unexpected_admission,
        raw_discovery_pending=lambda: discovery.discovery_pending,
    )

    class Wakeup:
        def clear(self) -> None:
            pass

        async def wait(self) -> None:
            raise TimeoutError

    service = DaemonIntakeService(
        FairIntakeDispatcher(
            tuple(IntakeClassSpec(name, adapter) for name, adapter in pairs), clock=frozen_clock.monotonic
        ),
        wakeup=cast(Any, Wakeup()),
        idle_delay_s=5,
    )

    @asynccontextmanager
    async def timeout(delay: float) -> AsyncIterator[None]:
        waits.append(delay)
        if len(waits) == 3:
            raise asyncio.CancelledError
        frozen_clock.advance(delay)
        yield

    # Replace only the service module's timeout boundary, not asyncio globally.
    # No real sleeping is needed to observe the scheduler's chosen deadlines.
    monkeypatch.setattr("polylogue.operations.intake_adapters.asyncio", SimpleNamespace(timeout=timeout))
    with pytest.raises(asyncio.CancelledError):
        await service.run()
    assert calls == [(None, 32), ("32", 32), ("64", 32)]
    assert waits == [0.05, 0.05, 5]
    assert not discovery.discovery_pending


def test_a_resweep_pages_promptly_only_after_resting_nine_sweep_durations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: FrozenClock
) -> None:
    """Prompt paging holds a bounded share of wall time on any archive size.

    A valid-only page keeps the first sweep prompt (07.F032), but a sweep that
    follows a completed one must not keep the service at its prompt cadence
    forever. Anti-vacuity: drop the rest window and ``discovery_pending`` is
    true for the whole second sweep.
    """
    bootstrap_archive_root(tmp_path)

    class ValidPages:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def required_page(
            self, _frame: object, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            frozen_clock.advance(1)  # Each page costs one second of sweep time.
            start = int(cursor or 0)
            end = min(start + limit, 96)
            return tuple(f"valid-{index}" for index in range(start, end)), str(end) if end < 96 else None

        def inspect(self, _frame: object, keys: Sequence[str]) -> dict[str, str]:
            return dict.fromkeys(keys, "valid")

        def terminal_decode_refusals(self, _keys: Sequence[str]) -> dict[str, Exception]:
            # No retained raw in this fixture refuses to decode.
            return {}

    monkeypatch.setattr(raw_inspection, "RawObservationInspection", ValidPages)
    discovery = RawMaterializationDiscovery(tmp_path)

    discovery.discover_pending_raw_ids(32)
    assert discovery.discovery_pending  # the first sweep is prompt
    discovery.discover_pending_raw_ids(32)
    discovery.discover_pending_raw_ids(32)
    assert not discovery.discovery_pending  # a three-second sweep completed

    discovery.discover_pending_raw_ids(32)
    assert not discovery.discovery_pending  # one second into a 27-second rest
    frozen_clock.advance(25)
    assert not discovery.discovery_pending
    frozen_clock.advance(1)
    assert discovery.discovery_pending


def test_process_halts_remain_visible_without_persisting_and_operator_halts_survive(tmp_path: Path) -> None:
    registry = HaltRegistry(tmp_path)
    registry.halt(
        unit_id(UnitKind.SOURCE, "operator-paused"),
        reason=HaltReason.OPERATOR_HALT,
        message="deliberately paused",
        frame="operator",
    )
    try:
        set_source_halt(
            "schema-source", DegradedReason(code="schema_version_mismatch", message="recheck after restart")
        )
        statuses = _halted_sources(tmp_path / "ops.db")
        assert [(status.source_name, status.code) for status in statuses] == [
            ("operator-paused", "operator_halt"),
            ("schema-source", "schema_version_mismatch"),
        ]
        assert not registry.is_halted(unit_id(UnitKind.SOURCE, "schema-source"))
    finally:
        clear_all_source_halts()
    assert [status.source_name for status in _halted_sources(tmp_path / "ops.db")] == ["operator-paused"]
    assert HaltRegistry(tmp_path).is_halted(unit_id(UnitKind.SOURCE, "operator-paused"))


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [DriveCatchupState.PENDING, DriveCatchupState.BLOCKED, DriveCatchupState.UNKNOWN])
async def test_remote_readiness_gap_is_not_a_failed_attempt_or_hourly_completion(
    tmp_path: Path, frozen_clock: FrozenClock, state: DriveCatchupState
) -> None:
    from polylogue.daemon.intake import AdmissionOutcome
    from polylogue.operations.intake_adapters import DriveIntakeAdapter

    calls = 0

    def callback() -> DriveCatchupReport:
        nonlocal calls
        calls += 1
        return DriveCatchupReport(state, materialization_pending=1 if state is DriveCatchupState.PENDING else None)

    adapter = DriveIntakeAdapter(callback)
    item = (await adapter.discover(limit=1))[0]
    result = await adapter.admit(item)
    assert result.outcome is AdmissionOutcome.DEFERRED
    await adapter.acknowledge(item)
    if state is DriveCatchupState.PENDING:
        assert await adapter.discover(limit=1)
    else:
        assert not await adapter.discover(limit=1)
        frozen_clock.advance(60)
        assert await adapter.discover(limit=1)
    assert calls == 1


@pytest.mark.asyncio
async def test_local_cold_settlement_completes_while_drive_readiness_stays_pending() -> None:
    import contextlib

    from polylogue.operations.intake_adapters import CallbackIntakeAdapter, ColdBuildSettlement, DriveIntakeAdapter

    settled = asyncio.Event()
    remote = DriveIntakeAdapter(lambda: DriveCatchupReport(DriveCatchupState.PENDING, materialization_pending=1))
    local = CallbackIntakeAdapter("configured_local", lambda: 0, persistent=False)
    dispatcher = FairIntakeDispatcher(
        (IntakeClassSpec("configured_local", local), IntakeClassSpec("configured_remote", remote))
    )

    def settle() -> ColdBuildSettlement:
        settled.set()
        return ColdBuildSettlement("complete")

    service = DaemonIntakeService(dispatcher, on_backlog_drained=settle, idle_delay_s=0.05)
    task = asyncio.create_task(service.run())
    try:
        await asyncio.wait_for(settled.wait(), 5)
        assert remote.discovery_pending
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
