"""Contracts for the shared cross-boundary wait guards."""

from __future__ import annotations

import asyncio
import subprocess

import pytest

from polylogue.core.bounded import BoundedTimeoutError, bounded, run_bounded


def test_run_bounded_checks_exit_status_by_default() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        run_bounded(["/bin/sh", "-c", "exit 7"], 2)


def test_run_bounded_can_report_nonzero_exit_without_swallowing_timeout() -> None:
    result = run_bounded(["/bin/sh", "-c", "exit 7"], 2, check_exit=False)
    assert result.returncode == 7


def test_run_bounded_translates_timeout_to_typed_error() -> None:
    with pytest.raises(BoundedTimeoutError) as caught:
        run_bounded(["/bin/sh", "-c", "sleep 10"], 0.01)
    assert caught.value.budget == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_bounded_translates_timeout_and_cancels_awaitable() -> None:
    cancelled = False

    async def stalled() -> None:
        nonlocal cancelled
        try:
            await asyncio.sleep(10)
        finally:
            cancelled = True

    with pytest.raises(BoundedTimeoutError):
        await bounded(stalled(), 0.01)
    assert cancelled


@pytest.mark.asyncio
async def test_bounded_accepts_a_typed_timeout_handler() -> None:
    class TypedTimeoutError(Exception):
        pass

    with pytest.raises(TypedTimeoutError):
        await bounded(asyncio.sleep(10), 0.01, on_timeout=TypedTimeoutError)


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan"), True])
def test_bounded_rejects_non_positive_or_non_finite_budgets(budget: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        run_bounded(["true"], budget)  # type: ignore[arg-type]
