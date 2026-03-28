"""Watch loop abstraction for continuous sync.

Separates the polling/retry/event-dispatch logic from the CLI
so it can be reused by MCP servers, daemons, or batch jobs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from polylogue.lib.log import get_logger
from polylogue.pipeline.events import SyncEvent, SyncEventHandler
from polylogue.sources import DriveError

if TYPE_CHECKING:
    from polylogue.storage.store import RunResult

logger = get_logger(__name__)


class WatchRunner:
    """Continuous sync loop with configurable interval and event dispatch.

    Args:
        sync_fn: Callable that executes a single sync run and returns a
            ``RunResult``.  The CLI passes a closure over
            ``_run_sync_once``.
        handler: Event handler (or composite) to notify after each run.
        interval: Seconds between sync runs (default 60).
        on_idle: Optional callback when no new conversations are found.
            Receives the current ``RunResult``.
        on_error: Optional callback for sync errors.  Receives the
            exception instance.
    """

    __slots__ = ("_sync_fn", "_handler", "_interval", "_on_idle", "_on_error", "_running")

    def __init__(
        self,
        sync_fn: Callable[[], RunResult],
        handler: SyncEventHandler,
        interval: int = 60,
        on_idle: Callable[[RunResult], Any] | None = None,
        on_error: Callable[[Exception], Any] | None = None,
    ) -> None:
        self._sync_fn = sync_fn
        self._handler = handler
        self._interval = interval
        self._on_idle = on_idle
        self._on_error = on_error
        self._running = False

    def run(self) -> None:
        """Run the watch loop until ``stop()`` is called or KeyboardInterrupt."""
        self._running = True
        try:
            while self._running:
                try:
                    result = self._sync_fn()
                    new_count = result.counts.get("conversations", 0)
                    event = SyncEvent(new_conversations=new_count, run_result=result)
                    self._handler.on_sync(event)
                    if new_count <= 0 and self._on_idle:
                        self._on_idle(result)
                except DriveError as exc:
                    if self._on_error:
                        self._on_error(exc)
                    else:
                        logger.warning("Sync error: %s", exc)
                except Exception as exc:
                    if self._on_error:
                        self._on_error(exc)
                    else:
                        logger.error("Unexpected error during sync: %s", exc)
                time.sleep(self._interval)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the watch loop to stop after the current iteration."""
        self._running = False


__all__ = ["WatchRunner"]
