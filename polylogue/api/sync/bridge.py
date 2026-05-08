from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Awaitable, Coroutine
from concurrent.futures import Future
from typing import TypeVar

T = TypeVar("T")

_worker_loop: asyncio.AbstractEventLoop | None = None
_worker_thread: threading.Thread | None = None
_lock = threading.Lock()


def _ensure_worker() -> asyncio.AbstractEventLoop:
    global _worker_loop, _worker_thread

    if _worker_loop is not None and _worker_loop.is_running():
        return _worker_loop

    with _lock:
        if _worker_loop is not None and _worker_loop.is_running():
            return _worker_loop
        _worker_loop = asyncio.new_event_loop()
        _worker_thread = threading.Thread(target=_worker_loop.run_forever, daemon=True)
        _worker_thread.start()

    return _worker_loop


def _stop_worker() -> None:
    global _worker_loop, _worker_thread
    with _lock:
        if _worker_loop is not None:
            _worker_loop.call_soon_threadsafe(_worker_loop.stop)
            _worker_loop = None
            _worker_thread = None


atexit.register(_stop_worker)


async def _await(awaitable: Awaitable[T]) -> T:
    return await awaitable


def run_coroutine_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine from sync code, even when already inside an event loop.

    When no event loop is running, uses asyncio.run() directly.
    When inside an event loop, delegates to a persistent worker thread
    with its own event loop, avoiding per-call thread creation.
    """
    wrapper: Coroutine[object, object, T] = _await(coro)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(wrapper)

    loop = _ensure_worker()
    future: Future[T] = asyncio.run_coroutine_threadsafe(wrapper, loop)
    return future.result()


__all__ = ["run_coroutine_sync"]
