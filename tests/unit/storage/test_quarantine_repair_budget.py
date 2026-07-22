"""Quarantined-accepted-raw repair blob-budget classification.

Live incident 2026-07-22: a single 298 MB codex whale raw over the 256 MiB
per-target limit made the budget check (then a hard ``RuntimeError``) abort
the daemon's entire raw-authority frontier convergence pass every cycle --
one oversized target poisoned the whole batch. The budget is now a
per-target *partition*: oversized targets come back as typed ineligible
items, aggregate overflow defers the tail, and every other target stays
inspectable.
"""

from __future__ import annotations

import sqlite3

import pytest

from polylogue.storage.repair import (
    _QUARANTINED_ACCEPTED_RAW_REPAIR_BLOB_LIMIT_BYTES,
    _QUARANTINED_ACCEPTED_RAW_REPAIR_TOTAL_BLOB_LIMIT_BYTES,
    _partition_quarantined_raw_repair_blob_budget,
)


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    # Only the two columns the partition reads; the real raw_sessions schema
    # is exercised by the archive-tier test suite.
    connection.execute("CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY, blob_size INTEGER NOT NULL)")
    return connection


def _seed(connection: sqlite3.Connection, sizes: dict[str, int]) -> None:
    connection.executemany(
        "INSERT INTO raw_sessions (raw_id, blob_size) VALUES (?, ?)",
        list(sizes.items()),
    )


def test_oversized_whale_is_excluded_not_batch_fatal(conn: sqlite3.Connection) -> None:
    whale = "a" * 64
    small = "b" * 64
    _seed(conn, {whale: _QUARANTINED_ACCEPTED_RAW_REPAIR_BLOB_LIMIT_BYTES + 1, small: 1024})

    inspectable, excluded = _partition_quarantined_raw_repair_blob_budget(conn, [whale, small])

    assert inspectable == [small]
    assert [item.raw_id for item in excluded] == [whale]
    assert excluded[0].status == "ineligible"
    assert "per-target repair limit" in excluded[0].reason


def test_aggregate_overflow_defers_tail_targets(conn: sqlite3.Connection) -> None:
    # Three targets each just under the per-target cap; the third pushes the
    # running total over the aggregate cap and must be deferred, not raise.
    size = _QUARANTINED_ACCEPTED_RAW_REPAIR_BLOB_LIMIT_BYTES - 1
    assert 3 * size > _QUARANTINED_ACCEPTED_RAW_REPAIR_TOTAL_BLOB_LIMIT_BYTES
    ids = ["1" * 64, "2" * 64, "3" * 64]
    _seed(conn, dict.fromkeys(ids, size))

    inspectable, excluded = _partition_quarantined_raw_repair_blob_budget(conn, ids)

    assert inspectable == ids[:2]
    assert [item.raw_id for item in excluded] == [ids[2]]
    assert "deferred" in excluded[0].reason


def test_within_budget_set_is_fully_inspectable(conn: sqlite3.Connection) -> None:
    ids = ["c" * 64, "d" * 64]
    _seed(conn, dict.fromkeys(ids, 4096))

    inspectable, excluded = _partition_quarantined_raw_repair_blob_budget(conn, ids)

    assert inspectable == ids
    assert excluded == []


def test_unknown_raw_id_stays_inspectable_for_downstream_typed_reporting(conn: sqlite3.Connection) -> None:
    """A raw id with no raw_sessions row is not a budget question -- the
    inspect path itself reports it as a typed 'raw row is missing' item."""
    missing = "e" * 64
    inspectable, excluded = _partition_quarantined_raw_repair_blob_budget(conn, [missing])

    assert inspectable == [missing]
    assert excluded == []
