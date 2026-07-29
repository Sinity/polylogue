"""Async reads for ``session_refs`` (polylogue-2qx.4).

``session_refs`` is a dedicated, tracker-agnostic table the writer
(``archive_tiers/write.py._write_session_refs``) populates from
``ParsedSession.session_refs`` -- the Claude Code pr-link (20,702
occurrences on the wire, previously discarded entirely) and any future
issue-tracker reference. Follows the same shape as
``queries/session_agent_policies.py``.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Sequence

import aiosqlite

from polylogue.storage.runtime import SessionRefRecord
from polylogue.storage.sqlite.queries.mappers import _row_to_session_ref

__all__ = [
    "get_session_refs",
    "get_session_refs_batch",
    "sync_get_session_refs",
]

_SELECT_COLUMNS = "ref_id, session_id, position, kind, repo, ref_number, url, observed_at_ms"


async def get_session_refs(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[SessionRefRecord]:
    """Return all reference rows for one session, ordered by position."""
    rows = await (
        await conn.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM session_refs
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        )
    ).fetchall()
    return [_row_to_session_ref(row) for row in rows]


async def get_session_refs_batch(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[SessionRefRecord]]:
    """Return reference rows for many sessions, grouped by session_id."""
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = await (
        await conn.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM session_refs
            WHERE session_id IN ({placeholders})
            ORDER BY session_id, position
            """,
            tuple(session_ids),
        )
    ).fetchall()
    result: dict[str, list[SessionRefRecord]] = defaultdict(list)
    for session_id in session_ids:
        result.setdefault(session_id, [])
    for row in rows:
        record = _row_to_session_ref(row)
        result[str(record.session_id)].append(record)
    return dict(result)


def sync_get_session_refs(conn: sqlite3.Connection, session_id: str) -> list[SessionRefRecord]:
    """Sync sibling of :func:`get_session_refs`."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT {_SELECT_COLUMNS}
        FROM session_refs
        WHERE session_id = ?
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    return [_row_to_session_ref(row) for row in rows]
