"""Saved producer comparator obeys the same primitive ordering as SQLite."""

from __future__ import annotations

import sqlite3

from hypothesis import given
from hypothesis import strategies as st

from polylogue.archive.query.search_cursor import position


@given(
    st.one_of(st.none(), st.integers(min_value=-(2**63), max_value=2**63 - 1)),
    st.one_of(st.none(), st.integers(min_value=-(2**63), max_value=2**63 - 1)),
    st.booleans(),
)
def test_cursor_order_matches_sqlite_null_and_direction(a: int | None, b: int | None, descending: bool) -> None:
    if a == b:
        assert not position("session", "a", ((a, descending, False),)).after(
            position("session", "b", ((b, descending, False),))
        )
        return
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE keys (id TEXT, value INTEGER)")
        conn.executemany("INSERT INTO keys VALUES (?,?)", [("a", a), ("b", b)])
        expected = [
            row[0] for row in conn.execute("SELECT id FROM keys ORDER BY value " + ("DESC" if descending else "ASC"))
        ]
    assert position("session", "a", ((a, descending, False),)).after(
        position("session", "b", ((b, descending, False),))
    ) == (expected == ["b", "a"])
