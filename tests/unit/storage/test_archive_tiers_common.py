"""Unit tests for shared DDL-generation helpers (archive_tiers/common.py).

order_check() is prepared for polylogue-cuxz.10 (13,743 ended<started rows in
session_phases/session_work_events) but not yet wired into any table's DDL —
see the docstring on order_check for why. These tests pin the SQL shape it
generates so it is ready to drop into a CREATE TABLE the moment it's needed,
and verify the generated CHECK actually enforces (or deliberately permits)
the intended rows via a real SQLite connection.
"""

from __future__ import annotations

import sqlite3

import pytest

from polylogue.storage.sqlite.archive_tiers.common import order_check


def test_order_check_nullable_default_permits_either_side_null() -> None:
    expr = order_check("ended_at_ms", "started_at_ms")
    assert expr == "(ended_at_ms IS NULL OR started_at_ms IS NULL OR ended_at_ms >= started_at_ms)"


def test_order_check_non_nullable_omits_null_clause() -> None:
    expr = order_check("last_seen_at_ms", "first_seen_at_ms", nullable=False)
    assert expr == "(last_seen_at_ms >= first_seen_at_ms)"


@pytest.mark.parametrize(
    ("earlier", "later", "should_pass"),
    [
        (100, 200, True),
        (200, 200, True),
        (200, 100, False),
        (None, 200, True),
        (100, None, True),
        (None, None, True),
    ],
)
def test_order_check_nullable_enforced_by_sqlite(earlier: int | None, later: int | None, should_pass: bool) -> None:
    conn = sqlite3.connect(":memory:")
    try:
        expr = order_check("ended_at_ms", "started_at_ms")
        conn.execute(f"CREATE TABLE t (started_at_ms INTEGER, ended_at_ms INTEGER, CHECK{expr})")
        if should_pass:
            conn.execute("INSERT INTO t VALUES (?, ?)", (earlier, later))
        else:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO t VALUES (?, ?)", (earlier, later))
    finally:
        conn.close()


def test_order_check_non_nullable_rejects_null_on_either_side() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        expr = order_check("last_seen_at_ms", "first_seen_at_ms", nullable=False)
        conn.execute(
            f"CREATE TABLE repos (first_seen_at_ms INTEGER NOT NULL, last_seen_at_ms INTEGER NOT NULL, CHECK{expr})"
        )
        conn.execute("INSERT INTO repos VALUES (100, 200)")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO repos VALUES (200, 100)")
    finally:
        conn.close()
