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


def _holder_deadline(lock_path: Path) -> float | None:
    """Wall-clock deadline the current holder declared, or ``None`` if unreadable."""
    try:
        return float(lock_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


@contextmanager
def shared_chrome_extension_lock(*, timeout_s: float) -> Iterator[None]:
    """Hold the shared-Chrome lock for a workflow whose budget is ``timeout_s``.

    The holder records its own wall-clock deadline in the lock file. A waiter
    keeps waiting while that holder is still inside its declared budget, so a
    shorter-budget workflow does not fail behind a longer, still-valid one; it
    gives up once both its own budget and the holder's have passed, so a wedged
    holder cannot hold every waiter indefinitely.
    """
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    lock_path = runtime_dir / "polylogue-shared-chrome-extension.lock"
    own_deadline = time.time() + timeout_s
    with lock_path.open("a", encoding="utf-8") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                holder_deadline = _holder_deadline(lock_path)
                deadline = own_deadline if holder_deadline is None else max(own_deadline, holder_deadline)
                if time.time() >= deadline:
                    raise SharedChromeLockTimeoutError(
                        f"shared Chrome extension lock still held past its holder's declared budget: {lock_path}"
                    ) from None
                time.sleep(_POLL_INTERVAL_S)
        try:
            lock_file.truncate(0)
            lock_file.write(f"{time.time() + timeout_s}\n")
            lock_file.flush()
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
