"""A fresh tier's schema is created in one transaction, not one commit per statement."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers import bootstrap
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_TIERS_WITHOUT_EXTENSIONS = (
    ArchiveTier.SOURCE,
    ArchiveTier.INDEX,
    ArchiveTier.USER,
    ArchiveTier.OPS,
    ArchiveTier.AUDIT,
)


@pytest.mark.parametrize("tier", _TIERS_WITHOUT_EXTENSIONS, ids=lambda tier: tier.value)
def test_fresh_tier_schema_runs_inside_one_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: ArchiveTier
) -> None:
    """No schema statement of a fresh tier's DDL route runs in autocommit.

    In autocommit every CREATE is its own synced commit: hundreds per fresh
    index tier. Anti-vacuity: drop the ``BEGIN`` from the DDL script (or let
    the identity stamp ``executescript`` its table, which commits the
    caller's transaction) and the CREATE statements are traced outside a
    transaction.
    """
    # Force the canonical DDL route instead of a page-copy prototype.
    monkeypatch.setattr(bootstrap, "_TIER_PROTOTYPES", {})
    monkeypatch.setattr(bootstrap, "_record_tier_prototype", lambda *_args: None)
    with closing(sqlite3.connect(tmp_path / f"{tier.value}.db")) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        autocommitted: list[str] = []
        created = 0

        def trace(statement: str) -> None:
            nonlocal created
            if statement.lstrip().upper().startswith("CREATE"):
                created += 1
                if not conn.in_transaction:
                    autocommitted.append(statement.strip()[:80])

        conn.set_trace_callback(trace)
        bootstrap.initialize_archive_tier(conn, tier)
        conn.set_trace_callback(None)

        assert created > 0
        assert autocommitted == []
        assert not conn.in_transaction
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == bootstrap.archive_tier_spec(tier).version

    # The committed schema is what a new connection sees.
    with closing(sqlite3.connect(tmp_path / f"{tier.value}.db")) as reader:
        assert int(reader.execute("PRAGMA user_version").fetchone()[0]) == bootstrap.archive_tier_spec(tier).version
        assert int(reader.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0]) > 0
