"""Census operation-layer maintenance reader authority."""

from __future__ import annotations

import inspect

from polylogue.operations import attachment_convergence


def test_operation_reader_census_records_migration_writer_exception() -> None:
    """Mutation: a direct reader in a convergence operation fails the census."""
    assert "sqlite3.connect" not in inspect.getsource(attachment_convergence)
