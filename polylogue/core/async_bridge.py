"""Synchronous-to-async coroutine bridge, deliberately dependency-free.

Historically this lived at ``polylogue.api.sync.bridge``. Importing anything
from a submodule of ``polylogue.api`` forces Python to execute
``polylogue/api/__init__.py`` first -- the whole ``Polylogue`` async facade
(pydantic viewport/surface models, storage repository mixins, ~1.7-2.8s of
import time) -- even though this helper itself only needs ``asyncio`` and
``threading``. CLI hot paths that only want to drive a coroutine (notably the
daemon-fast-path query dispatch, polylogue-g3jk) import this module directly
so they never pull in ``polylogue.api`` at all when the daemon serves the
request. ``polylogue.api.sync.bridge`` re-exports ``run_coroutine_sync`` from
here for existing callers that already pay the ``polylogue.api`` cost.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import sys
import threading
from collections.abc import Awaitable, Coroutine
from typing import TypeVar

T = TypeVar("T")


async def _await(awaitable: Awaitable[T]) -> T:
    return await awaitable


def complete_without_suspension(coroutine: Coroutine[object, object, T]) -> T:
    """Drive a coroutine whose awaits all resolve synchronously, on this thread.

    Shared builders are coroutines because their facade readers are
    asynchronous; over a pinned synchronous snapshot reader they never
    suspend. Admitted archive reads may run that work nested on a compute
    worker that is already driving an event loop, where ``asyncio.run`` and a
    hop to another thread (which would lose the snapshot's thread-bound
    connections) are both wrong. A coroutine that does suspend is refused.
    """
    try:
        coroutine.send(None)
    except StopIteration as finished:
        return finished.value  # type: ignore[no-any-return]
    coroutine.close()
    raise RuntimeError("a pinned-snapshot coroutine suspended; its reader must not await")


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
    The caller joins that physical thread. When it holds a compute reservation,
    the bridge borrows that exact reservation and cancellation mailbox, so a
    nested pure unit cannot submit-and-wait onto the saturated adapter. Native
    SQL created by the bridge settles in its owning coroutine Task before the
    loop or thread retires.
    """
    wrapper: Coroutine[object, object, T] = _await(coro)
    compute = sys.modules.get("polylogue.core.compute")
    borrow = compute.capture_compute_bridge() if compute is not None else contextlib.nullcontext

    async def joined() -> T:
        # Settle inside the physical coroutine Task, before asyncio.run retires
        # it. Only newly created owners belong to this nested bridge unit.
        with borrow():
            return await wrapper

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(joined())

    submitter_context = contextvars.copy_context()
    result: list[T] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(submitter_context.run(asyncio.run, joined()))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=_runner, name="polylogue-sync-bridge", daemon=True)
    thread.start()
    thread.join()

    if error:
        raise error[0]
    return result[0]


__all__ = ["run_coroutine_sync"]
