"""Contract tests for the sync→async bridge (``polylogue.api.sync.bridge``).

The bridge runs a coroutine synchronously, including when the caller is already
inside a running event loop (a sync CLI surface invoked from an async test). The
regression these tests guard is a hang: an earlier shared-worker-loop design
could be left pointing at a loop that no longer ran, so ``future.result()``
blocked forever (it hung the full suite under ``devtools verify`` because the
harness patches ``threading.Thread.run`` per test). The current design uses a
fresh per-call thread, so it cannot be poisoned by prior calls or test ordering.
Each test runs under a hard timeout so any regression fails loudly, not hangs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import pytest

from polylogue.api.sync.bridge import run_coroutine_sync

_R = TypeVar("_R")


async def _echo(value: int) -> int:
    await asyncio.sleep(0)
    return value * 2


async def _boom() -> int:
    await asyncio.sleep(0)
    raise ValueError("boom")


def _with_timeout(fn: Callable[[], _R], *, seconds: float = 15.0) -> _R:
    """Run ``fn`` on a side thread and fail (not hang) if it blocks too long."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=seconds)
        except TimeoutError:  # pragma: no cover - only on a real regression
            pytest.fail("run_coroutine_sync hung; the sync bridge deadlocked")


def test_run_coroutine_sync_without_running_loop() -> None:
    assert _with_timeout(lambda: run_coroutine_sync(_echo(3))) == 6


def test_run_coroutine_sync_from_within_running_loop() -> None:
    async def driver() -> int:
        # Called while a loop is already running: the bridge must run the coro on
        # a separate thread, not deadlock the current loop.
        return run_coroutine_sync(_echo(7))

    assert _with_timeout(lambda: asyncio.run(driver())) == 14


def test_run_coroutine_sync_repeated_calls_within_loop() -> None:
    # Repeated calls (the parity-test pattern) must each complete; the original
    # bug only surfaced after prior calls had run.
    async def driver() -> tuple[int, ...]:
        return tuple(run_coroutine_sync(_echo(n)) for n in range(5))

    assert _with_timeout(lambda: asyncio.run(driver())) == (0, 2, 4, 6, 8)


def test_run_coroutine_sync_propagates_exceptions_within_loop() -> None:
    async def driver() -> None:
        run_coroutine_sync(_boom())

    with pytest.raises(ValueError, match="boom"):
        _with_timeout(lambda: asyncio.run(driver()))
