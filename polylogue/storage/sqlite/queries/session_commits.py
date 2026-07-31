"""Async reads for ``session_commits`` (polylogue-cijx.3).

``session_commits`` is populated at write time from the parser-reported
``ParsedSession.git_commit_hash`` (``archive_tiers/write.py``, the sole
writer, ``method='parser-git-meta'``) -- the repo checkout's HEAD commit at
the moment the session was captured. Until this module existed, the table
had zero readers anywhere on the read/API/CLI/MCP surface: a write-only
table (polylogue-cijx.3 AC3). This is deliberately a narrow, honest fact
("what commit was checked out when the session began"), distinct from the
broader on-demand git/GitHub correlation surface
(``insights.session_commit``, which scores commits by time-window/
file-overlap or a Claude-Session git trailer match against a live repo).
Follows the same shape as ``queries/session_refs.py``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import aiosqlite

from polylogue.storage.runtime import SessionCommitRecord
from polylogue.storage.sqlite.queries.mappers import _row_to_session_commit

__all__ = [
    "get_session_commits",
    "get_session_commits_batch",
    "sync_get_session_commits",
]

_SELECT_COLUMNS = "session_id, commit_sha, repo_id, detection_type, method, confidence, evidence_json, created_at_ms"


async def get_session_commits(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[SessionCommitRecord]:
    """Return the checkout-HEAD commit row(s) for one session."""
    rows = await (
        await conn.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM session_commits
            WHERE session_id = ?
            ORDER BY created_at_ms
            """,
            (session_id,),
        )
    ).fetchall()
    return [_row_to_session_commit(row) for row in rows]


async def get_session_commits_batch(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[SessionCommitRecord]]:
    """Return checkout-HEAD commit rows for many sessions, grouped by session_id."""
    result: dict[str, list[SessionCommitRecord]] = {session_id: [] for session_id in session_ids}
    if not session_ids:
        return result
    placeholders = ", ".join("?" for _ in session_ids)
    rows = await (
        await conn.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM session_commits
            WHERE session_id IN ({placeholders})
            ORDER BY session_id, created_at_ms
            """,
            tuple(session_ids),
        )
    ).fetchall()
    for row in rows:
        record = _row_to_session_commit(row)
        result[str(record.session_id)].append(record)
    return result


def sync_get_session_commits(conn: sqlite3.Connection, session_id: str) -> list[SessionCommitRecord]:
    """Sync sibling of :func:`get_session_commits`."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT {_SELECT_COLUMNS}
        FROM session_commits
        WHERE session_id = ?
        ORDER BY created_at_ms
        """,
        (session_id,),
    ).fetchall()
    return [_row_to_session_commit(row) for row in rows]
