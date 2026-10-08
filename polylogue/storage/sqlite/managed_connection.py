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

from polylogue.core.sql_settlement import current_native_sql_lifetimes
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner, _close_failed_native_construction

__all__ = ["sqlite_connection"]


@contextmanager
def sqlite_connection(*args: Any, **kwargs: Any) -> Iterator[sqlite3.Connection]:
    """Open a ``sqlite3`` connection, commit or roll back, then always close."""
    from polylogue.storage.sqlite.population_admission import assert_population_admitted

    database = args[0] if args else kwargs.pop("database")
    assert_population_admitted(database)
    connection = connect_measured(database, *args[1:], **kwargs)
    owner = NativeSQLCustodyOwner(connection, lifetime_dependencies=current_native_sql_lifetimes())
    try:
        with connection:
            yield connection
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()
