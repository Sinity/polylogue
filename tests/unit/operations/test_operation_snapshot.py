"""Read authority must name the connections that produced the rows."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.archive.query.execution_control import QueryCancelledError, QueryExecutionContext
from polylogue.operations.authority import authority_for_reader
from polylogue.operations.operation_context import (
    observe_control_authority,
    open_operation_read,
    prepare_operation_journals,
)
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from tests.infra.archive_templates import bootstrap_archive_root

pytestmark = pytest.mark.uses_real_clock("exercises the canonical query deadline/cancellation controller")


def test_control_authority_refuses_generation_change_during_audit_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generation switch between identity and receipt observation cannot be merged into one authority."""
    bootstrap_archive_root(tmp_path)
    original = ArchiveIdentity.resolve_location
    calls = 0

    def changed(location: ArchiveLocation) -> ArchiveIdentity:
        nonlocal calls
        calls += 1
        identity = original(location)
        return replace(identity, active_generation="synthetic-new-generation") if calls > 1 else identity

    monkeypatch.setattr(ArchiveIdentity, "resolve_location", changed)
    with pytest.raises(ValueError, match="archive changed"):
        observe_control_authority(tmp_path)


def test_operation_snapshot_pins_source_and_attached_tiers(tmp_path: Path) -> None:
    """A mutation after pinning cannot change either of the two reader handles."""

    bootstrap_archive_root(tmp_path)
    prepare_operation_journals(tmp_path)
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


def test_reader_does_not_activate_journals_but_writer_startup_does(tmp_path: Path) -> None:
    """Mutation: move journal negotiation into the readonly pin and the mode changes early."""
    bootstrap_archive_root(tmp_path)
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        source.execute("PRAGMA journal_mode=DELETE")
    with open_operation_read(tmp_path) as pinned:
        assert pinned.archive.source_connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    prepare_operation_journals(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        index = pinned.archive.index_connection
        assert index is not None
        for tier, alias in (
            ("index", "main"),
            ("source", "source_tier"),
            ("audit", "audit_tier"),
            ("user", "user_tier"),
        ):
            assert index.execute(f"PRAGMA {alias}.journal_mode").fetchone()[0] == "wal", tier


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
