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
import polylogue.operations.raw_observation_derivation as raw_derivation
from polylogue.core.degraded import DegradedReason
from polylogue.core.source_halts import clear_all_source_halts, set_source_halt
from polylogue.daemon.catchup_status import _halted_sources
from polylogue.daemon.intake import FairIntakeDispatcher, IntakeClassSpec
from polylogue.daemon.service_halt import HaltReason, HaltRegistry, UnitKind, unit_id
from polylogue.operations.intake_adapters import (
    DaemonIntakeContext,
    DaemonIntakeService,
    RawMaterializationDiscovery,
    build_intake_adapters,
)
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.frozen_clock import FrozenClock


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [0, 1])
async def test_composed_remote_poll_waits_an_hour_after_completion(
    tmp_path: Path, frozen_clock: FrozenClock, changed: int
) -> None:
    calls = 0

    def poll() -> int:
        nonlocal calls
        calls += 1
        frozen_clock.advance(30)  # The interval starts after this work finishes.
        return changed

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

    def poll() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise BlockingIOError(errno.EAGAIN, "synthetic temporary unavailability")
        return 0

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
    bootstrap_archive_root(tmp_path)
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

    monkeypatch.setattr(raw_derivation, "RawObservationDerivation", ValidPages)
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
