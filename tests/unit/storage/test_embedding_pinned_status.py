from __future__ import annotations

import sqlite3

import pytest

from polylogue.storage.embeddings.status_payload import _scalar_int_with_timeout


def test_pinned_status_scalar_preserves_operation_cancellation_handler() -> None:
    """A local status timeout must not replace or clear operation cancellation.

    Anti-vacuity: restoring the former local progress-handler install makes
    the first query complete and clears the sentinel before the second query.
    """

    conn = sqlite3.connect(":memory:")
    conn.set_progress_handler(lambda: 1, 1)
    canceled_query = """
        WITH RECURSIVE counter(value) AS (
            VALUES(1)
            UNION ALL
            SELECT value + 1 FROM counter WHERE value < 1_000
        )
        SELECT SUM(value) FROM counter
    """
    try:
        with pytest.raises(sqlite3.OperationalError, match="interrupted"):
            _scalar_int_with_timeout(
                conn,
                canceled_query,
                timeout_ms=None,
            )

        with pytest.raises(sqlite3.OperationalError, match="interrupted"):
            conn.execute(canceled_query).fetchone()
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()
