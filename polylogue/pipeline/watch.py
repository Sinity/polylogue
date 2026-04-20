"""Watch loop abstraction for continuous sync."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from polylogue.pipeline.observers import RunObserver

if TYPE_CHECKING:
    from polylogue.storage.run_state import RunResult


class WatchRunner:
    """Continuous sync loop with configurable interval and observer notifications."""

    __slots__ = ("_sync_fn", "_observer", "_interval", "_running")

    def __init__(
        self,
        sync_fn: Callable[[], RunResult],
        observer: RunObserver,
        interval: int = 60,
    ) -> None:
        self._sync_fn = sync_fn
        self._observer = observer
        self._interval = interval
        self._running = False

    def run(self) -> None:
        """Run the watch loop until ``stop()`` is called or KeyboardInterrupt."""
        self._running = True
        try:
            while self._running:
                try:
                    result = self._sync_fn()
                    new_count = _conversation_count(result)
                    self._observer.on_completed(result)
                    if new_count <= 0:
                        self._observer.on_idle(result)
                except Exception as exc:
                    self._observer.on_error(exc)
                time.sleep(self._interval)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the watch loop to stop after the current iteration."""
        self._running = False


__all__ = ["WatchRunner"]


def _conversation_count(result: object) -> int:
    counts = getattr(result, "counts", None)
    int_value = getattr(counts, "int_value", None)
    if callable(int_value):
        value = int_value("conversations")
        return value if isinstance(value, int) else 0
    if isinstance(counts, Mapping):
        value = counts.get("conversations", 0)
        return value if isinstance(value, int) else 0
    return 0
