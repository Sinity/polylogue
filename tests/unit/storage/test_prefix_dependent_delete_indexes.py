"""Plan-assertion coverage for polylogue-crd8.

Live evidence (2026-07-19): a whale prefix-sharing lineage session's
``_reextract_prefix_tail_db`` -> ``_delete_prefix_message_dependents`` held the
writer for hours doing full scans of ``session_events``/``session_agent_policies``
because those tables' write-mode connections were never guaranteed to carry
the ``idx_session_events_source_message`` / ``idx_session_agent_policies_source_message``
partial indexes. The indexes themselves were already present in the canonical
DDL and in ``runtime_indexes.py`` (added 2026-06-24, ddf4f3efc7) -- since that
DDL addition was not accompanied by an ``ArchiveTier.INDEX`` version bump
(correct, additive-derived policy), any archive/generation whose ``index.db``
reached the current version *before* that DDL addition landed never replays
the DDL again (``initialize_archive_database`` only re-applies DDL for
OPS/USER tiers on a same-version reopen) and only reads the index in via one
of the "ensure" call sites. The read-only path
(``ArchiveStore._ensure_read_runtime_indexes``) already had that ensure call;
the write-mode path did not -- meaning every ``ArchiveStore`` write open,
including ``open_owned_inactive_generation`` (used by bulk rebuilds/revision
backfill), could run its entire lifetime against an index.db missing these
indexes. This module asserts both the DDL-level index shape and the
write-open retrofit fix.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    clear_messages_parent_sql,
    clear_session_agent_policies_source_message_sql,
    clear_session_events_source_message_sql,
)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _plan(conn: sqlite3.Connection, sql: str, placeholders: str) -> list[str]:
    params = tuple(f"m{i}" for i in range(placeholders.count("?")))
    rows = conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
    return [str(row["detail"]) for row in rows]


def test_prefix_dependent_clear_statements_use_leading_indexes(tmp_path: Path) -> None:
    """Fresh-bootstrapped DB: the three prefix-delete UPDATEs used by both
    ``_delete_prefix_message_dependents`` and
    ``_delete_all_session_message_dependents`` must each probe a leading
    index, not scan their whole table."""
    conn = _connect(tmp_path / "index.db")
    try:
        placeholders = ",".join("?" for _ in range(3))

        messages_plan = _plan(conn, clear_messages_parent_sql(placeholders), placeholders)
        assert any("SEARCH" in step and "idx_messages_parent" in step for step in messages_plan), messages_plan

        events_plan = _plan(conn, clear_session_events_source_message_sql(placeholders), placeholders)
        assert any("SEARCH" in step and "idx_session_events_source_message" in step for step in events_plan), (
            events_plan
        )

        policies_plan = _plan(conn, clear_session_agent_policies_source_message_sql(placeholders), placeholders)
        assert any(
            "SEARCH" in step and "idx_session_agent_policies_source_message" in step for step in policies_plan
        ), policies_plan
    finally:
        conn.close()


def test_archive_store_write_open_retrofits_dropped_prefix_indexes(tmp_path: Path) -> None:
    """Anti-vacuity target for the ``ArchiveStore`` write-open fix.

    Simulates a same-version ``index.db`` that predates the
    ``idx_session_events_source_message`` / ``idx_session_agent_policies_source_message``
    DDL addition (or was otherwise stripped of them) by dropping both indexes
    after a fresh bootstrap, then reopening the archive root through
    ``ArchiveStore`` in write mode -- the exact code path
    ``open_owned_inactive_generation`` (bulk rebuild / revision backfill) and
    every other write-mode caller use. Before the fix in
    ``ArchiveStore._initialize_store``, the write branch never called
    ``ensure_runtime_indexes_sync`` on its own connection (only the read-only
    branch did), so a dropped/never-created index stayed missing for the
    entire lifetime of that write connection. Revert the ``else:
    ensure_runtime_indexes_sync(self._conn)`` branch in
    ``polylogue/storage/sqlite/archive_tiers/archive.py`` to see this test
    fail (SCAN instead of SEARCH, and the index missing from
    ``PRAGMA index_list``).
    """
    root = tmp_path / "archive"
    with ArchiveStore(root):
        pass  # fresh bootstrap only; index.db now exists at current version.

    index_db_path = root / "index.db"
    conn = sqlite3.connect(index_db_path)
    try:
        conn.execute("DROP INDEX idx_session_events_source_message")
        conn.execute("DROP INDEX idx_session_agent_policies_source_message")
        conn.commit()
        remaining = {row[1] for row in conn.execute("PRAGMA index_list('session_events')")} | {
            row[1] for row in conn.execute("PRAGMA index_list('session_agent_policies')")
        }
        assert "idx_session_events_source_message" not in remaining
        assert "idx_session_agent_policies_source_message" not in remaining
    finally:
        conn.close()

    with ArchiveStore.open_existing(root, read_only=False) as facade:
        events_indexes = {row["name"] for row in facade._conn.execute("PRAGMA index_list('session_events')")}
        policies_indexes = {row["name"] for row in facade._conn.execute("PRAGMA index_list('session_agent_policies')")}
        assert "idx_session_events_source_message" in events_indexes
        assert "idx_session_agent_policies_source_message" in policies_indexes

        placeholders = ",".join("?" for _ in range(3))
        events_plan = _plan(facade._conn, clear_session_events_source_message_sql(placeholders), placeholders)
        assert any("SEARCH" in step and "idx_session_events_source_message" in step for step in events_plan), (
            events_plan
        )
        policies_plan = _plan(facade._conn, clear_session_agent_policies_source_message_sql(placeholders), placeholders)
        assert any(
            "SEARCH" in step and "idx_session_agent_policies_source_message" in step for step in policies_plan
        ), policies_plan
