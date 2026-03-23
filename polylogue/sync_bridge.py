from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


def run_coroutine_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine from sync code, even when already inside an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[T] = []
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - re-raised on caller thread
            error.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    if not result:
        raise RuntimeError("Coroutine thread completed without returning a result")
    return result[0]


__all__ = ["run_coroutine_sync"]
