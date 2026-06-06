from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Coroutine
from typing import TypeVar

T = TypeVar("T")


async def _await(awaitable: Awaitable[T]) -> T:
    return await awaitable


def run_coroutine_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine from sync code, even when already inside an event loop.

    When no event loop is running, drive it directly with ``asyncio.run``.
    When a loop is already running (sync CLI surfaces invoked from inside an
    async caller — e.g. a CliRunner test, or one async API calling a sync
    wrapper), the coroutine is executed on a dedicated short-lived thread with
    its own fresh event loop.

    A per-call thread is deliberate. An earlier implementation kept a persistent
    shared worker loop for performance, but a shared mutable loop is fragile
    under test isolation that patches ``threading.Thread.run`` per test: the
    global could be left pointing at a loop whose thread had stopped driving it,
    so ``run_coroutine_threadsafe`` scheduled work that never ran and
    ``future.result()`` blocked forever (observed hanging the full test suite).
    A fresh thread + ``asyncio.run`` per call has no shared state, so it cannot
    be poisoned by prior calls or test ordering; the cost (one thread spawn per
    sync-bridge call, ~once per CLI invocation) is negligible.
    """
    wrapper: Coroutine[object, object, T] = _await(coro)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(wrapper)

    result: list[T] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(asyncio.run(wrapper))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=_runner, name="polylogue-sync-bridge", daemon=True)
    thread.start()
    thread.join()

    if error:
        raise error[0]
    return result[0]


__all__ = ["run_coroutine_sync"]
