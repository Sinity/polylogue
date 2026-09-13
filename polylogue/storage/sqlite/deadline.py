"""Bound a query without replacing its connection owner's progress handler."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Timer


@contextmanager
def query_deadline(conn: sqlite3.Connection, *, seconds: float | None) -> Iterator[None]:
    """Interrupt this scope while preserving an outer cancellation handler.

    SQLite cannot retrieve or compose progress handlers. Its thread-safe
    interrupt hook lets nested readers bound their own queries without clearing
    their operation's cancellation guard. Joining the timer prevents an expired
    callback from interrupting a later query after this scope returns.
    """
    if seconds is None:
        yield
        return
    if seconds <= 0:
        raise sqlite3.OperationalError("interrupted: query deadline exceeded")
    timer = Timer(seconds, conn.interrupt)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        timer.join()
