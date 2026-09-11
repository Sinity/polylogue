"""Read authority must name the connections that produced the rows."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.query.execution_control import QueryCancelledError, QueryExecutionContext
from polylogue.operations.authority import authority_for_reader
from polylogue.operations.operation_context import open_operation_read
from tests.infra.archive_templates import bootstrap_archive_root

pytestmark = pytest.mark.uses_real_clock("exercises the canonical query deadline/cancellation controller")


def test_operation_snapshot_pins_source_and_attached_tiers(tmp_path: Path) -> None:
    """A mutation after pinning cannot change either of the two reader handles."""

    bootstrap_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute("CREATE TABLE synthetic_snapshot_value(value INTEGER NOT NULL) STRICT")
        source.execute("INSERT INTO synthetic_snapshot_value VALUES (1)")
    with open_operation_read(tmp_path) as pinned:
        source = pinned.archive.source_connection
        index = pinned.archive.index_connection
        assert index is not None
        assert source.execute("SELECT value FROM synthetic_snapshot_value").fetchone()[0] == 1
        with sqlite3.connect(tmp_path / "source.db") as writer:
            writer.execute("UPDATE synthetic_snapshot_value SET value = 2")
        assert source.execute("SELECT value FROM synthetic_snapshot_value").fetchone()[0] == 1
        assert index.execute("SELECT value FROM source_tier.synthetic_snapshot_value").fetchone()[0] == 1
        observed = authority_for_reader(pinned.archive, server_identity="direct").to_dict()
        assert observed["archive_epoch"] == pinned.identity.authority_identity_digest
        assert observed["tier_schema_versions"] == pinned.schema_versions
        assert set(pinned.schema_versions) == {"source", "index", "user", "audit", "ops", "embeddings"}
    with sqlite3.connect(tmp_path / "source.db") as after:
        assert after.execute("SELECT value FROM synthetic_snapshot_value").fetchone()[0] == 2


@pytest.mark.parametrize("tier", ["index", "source"])
def test_operation_snapshot_installs_existing_cancellation_control_on_executing_handles(
    tmp_path: Path,
    tier: str,
) -> None:
    """Bypassing the source sibling progress guard lets this recursive query finish."""
    bootstrap_archive_root(tmp_path)
    context = QueryExecutionContext(call_id="cancelled-read", query_ref="synthetic-recursion")
    with pytest.raises(QueryCancelledError):
        with open_operation_read(tmp_path, execution_context=context) as pinned:
            connection = pinned.archive.index_connection if tier == "index" else pinned.archive.source_connection
            assert connection is not None
            context.cancel()
            connection.execute(
                "WITH RECURSIVE work(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM work WHERE n<100000) "
                "SELECT SUM(n) FROM work"
            ).fetchone()
    assert context.receipt.interrupted
    assert context.receipt.cleanup_complete
