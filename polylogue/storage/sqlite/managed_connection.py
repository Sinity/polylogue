"""A ``sqlite3`` context manager that also closes the connection.

The builtin ``with sqlite3.connect(path) as conn`` form manages a
*transaction*, not the connection: the connection object -- and its file
descriptor -- outlives the block until the collector reaches it. Accumulated
across a test session that deferral exhausts a worker's descriptor limit, and
the process that fails is whichever one runs next, not the one that leaked.

:func:`sqlite_connection` keeps the transaction semantics of the builtin form
(commit on clean exit, rollback on exception) and closes the connection.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["sqlite_connection"]


@contextmanager
def sqlite_connection(*args: Any, **kwargs: Any) -> Iterator[sqlite3.Connection]:
    """Open a ``sqlite3`` connection, commit or roll back, then always close."""
    connection = sqlite3.connect(*args, **kwargs)
    try:
        with connection:
            yield connection
    finally:
        connection.close()
