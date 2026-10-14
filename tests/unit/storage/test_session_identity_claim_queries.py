"""Claimant lookups and cleanup must not scan all provider identity claims."""

from pathlib import Path

import pytest

from polylogue.storage.io_phase_metrics import close_connection_cursor
from polylogue.storage.sqlite.connection import open_connection


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT origin, provider_value FROM session_identity_claims "
        "WHERE claimant_session_id=? AND identity_namespace='provider-session'",
        "DELETE FROM session_identity_claims WHERE claimant_session_id=?",
    ],
)
def test_claimant_queries_seek_the_selected_session(test_db: Path, statement: str) -> None:
    with open_connection(test_db) as connection:
        cursor = connection.execute("EXPLAIN QUERY PLAN " + statement, ("fixture-session",))
        try:
            details = [str(row[3]) for row in cursor]
        finally:
            close_connection_cursor(connection, cursor)
    assert any(detail.startswith("SEARCH session_identity_claims ") for detail in details), details
