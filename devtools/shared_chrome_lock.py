"""Serialize workflows that reload or reconfigure Polylogue in shared Chrome."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def shared_chrome_extension_lock() -> Iterator[None]:
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    lock_path = runtime_dir / "polylogue-shared-chrome-extension.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
