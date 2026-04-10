from __future__ import annotations

import asyncio

from polylogue.sync import _run


def test_private_run_executes_coroutine_from_sync_code() -> None:
    assert _run(asyncio.sleep(0, result="ok")) == "ok"


async def _nested_private_run() -> int:
    return _run(asyncio.sleep(0, result=7))


def test_private_run_executes_coroutine_while_event_loop_is_running() -> None:
    assert asyncio.run(_nested_private_run()) == 7
