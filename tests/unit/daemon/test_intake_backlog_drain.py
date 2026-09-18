"""The intake loop's one-shot backlog-drained signal (polylogue-b7dkb).

A cold build has exactly one moment it can be declared finished: a dispatcher
pass that found nothing to do, after a pass that did something. Anything
earlier promotes a half-built generation; anything later never promotes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

from polylogue.operations.intake_adapters import DaemonIntakeService


@dataclass
class _Pass:
    progressed: bool


class _ScriptedDispatcher:
    def __init__(self, script: list[bool]) -> None:
        self._script = script
        self.calls = 0

    async def run_once(self, *, budget: int) -> _Pass:
        del budget
        progressed = self._script[self.calls] if self.calls < len(self._script) else False
        self.calls += 1
        return _Pass(progressed=progressed)


async def _run_passes(script: list[bool], fired: list[int]) -> _ScriptedDispatcher:
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
    asyncio.run(_run_passes([False, True, False, False], fired))
    assert fired == [3]


def test_the_drain_signal_fires_once() -> None:
    """A later backlog is ordinary live ingest, not a second cold build.

    Anti-vacuity: leaving ``_on_backlog_drained`` set after the first call
    makes this list longer than one element.
    """
    fired: list[int] = []
    asyncio.run(_run_passes([True, False, True, False, False], fired))
    assert fired == [2]
