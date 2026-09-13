"""Census operation-layer maintenance reader authority."""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest

from polylogue.operations import attachment_convergence, durable_reference_transition


def test_operation_reader_census_records_migration_writer_exception() -> None:
    """Mutation: a direct reader or an unclassified durable writer fails the census."""
    assert "sqlite3.connect" not in inspect.getsource(attachment_convergence)
    transition_source = inspect.getsource(durable_reference_transition)
    assert transition_source.count("sqlite3.connect") == 2
    assert "Explicit non-read classification" in transition_source


def test_durable_transition_reader_cannot_mutate(tmp_path: Path) -> None:
    """Mutation: replacing the profile-backed helper with sqlite3.connect permits INSERT."""
    database = tmp_path / "maintenance.db"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE entries (value TEXT NOT NULL)")

    with durable_reference_transition._readonly(database) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO entries VALUES ('forbidden')")
