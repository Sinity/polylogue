"""Serialize workflows that reload or reconfigure Polylogue in shared Chrome."""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_POLL_INTERVAL_S = 0.05


class SharedChromeLockTimeoutError(TimeoutError):
    """Another shared-Chrome workflow held the lock past the waiter's budget."""


@contextmanager
def shared_chrome_extension_lock(*, timeout_s: float) -> Iterator[None]:
    """Hold the shared-Chrome lock, waiting at most ``timeout_s`` to acquire it.

    Acquisition polls non-blockingly so a wedged holder cannot extend a
    workflow past its declared budget before that workflow's own timeout starts.
    """
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    lock_path = runtime_dir / "polylogue-shared-chrome-extension.lock"
    deadline = time.monotonic() + timeout_s
    with lock_path.open("a", encoding="utf-8") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SharedChromeLockTimeoutError(
                        f"shared Chrome extension lock still held after {timeout_s:g}s: {lock_path}"
                    ) from None
                time.sleep(_POLL_INTERVAL_S)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
