"""Bounded discovery of transaction-owned session-profile obligations."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence


def profile_demand_page(
    conn: sqlite3.Connection,
    *,
    cursor: str | None,
    limit: int,
    scope: Sequence[str] | None,
) -> tuple[tuple[str, ...], str | None]:
    """Page present sessions with pending demand, optionally within a caller scope."""
    if limit < 1:
        raise ValueError("profile demand page limit must be positive")
    if scope is None:
        rows = conn.execute(
            """SELECT d.session_id FROM session_profile_demand AS d
               JOIN sessions AS s ON s.session_id = d.session_id
               WHERE d.session_id > COALESCE(?, '')
               ORDER BY d.session_id LIMIT ?""",
            (cursor, limit + 1),
        ).fetchall()
        keys = tuple(str(row[0]) for row in rows[:limit])
        return keys, (keys[-1] if len(rows) > limit else None)

    scoped = tuple(sorted(dict.fromkeys(str(key) for key in scope if cursor is None or str(key) > cursor)))
    if not scoped:
        return (), None
    found: set[str] = set()
    for start in range(0, len(scoped), 500):
        chunk = scoped[start : start + 500]
        rows = conn.execute(
            f"""SELECT d.session_id FROM session_profile_demand AS d
                 JOIN sessions AS s ON s.session_id = d.session_id
                 WHERE d.session_id IN ({",".join("?" * len(chunk))})
                 ORDER BY d.session_id LIMIT ?""",
            (*chunk, limit + 1),
        ).fetchall()
        found.update(str(row[0]) for row in rows)
    ordered = tuple(sorted(found))
    keys = ordered[:limit]
    return keys, (keys[-1] if len(ordered) > limit else None)
