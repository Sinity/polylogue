"""Async reads for ``session_agent_policies``.

``session_agent_policies`` is a dedicated typed table the writer
(``archive_tiers/write.py``) upserts Codex ``agent_policy`` events into
instead of duplicating them into ``session_events`` (see
``_SESSION_EVENTS_REDUNDANT_TYPES`` there) -- sandbox policy, approval
policy, and network policy facts. Until this module existed, the table had
a sync reader (``read_session_agent_policies``, exercised only by tests)
but no async query path, so nothing on the repository/API/MCP surface could
reach the evidence: real facts landed durably, with no reader. This module
is the async twin, following the same ``queries/session_events.py`` shape
(single-session + batch).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Sequence

import aiosqlite

from polylogue.storage.sqlite.archive_tiers.write import ArchiveAgentPolicy

__all__ = [
    "get_session_agent_policies",
    "get_session_agent_policies_batch",
]

_SELECT_COLUMNS = (
    "policy_id, session_id, position, approval_policy, "
    "sandbox_policy, network_policy, observed_at_ms, source_message_id"
)


def _row_to_agent_policy(row: sqlite3.Row) -> ArchiveAgentPolicy:
    return ArchiveAgentPolicy(
        policy_id=str(row["policy_id"]),
        session_id=str(row["session_id"]),
        position=int(row["position"]),
        approval_policy=row["approval_policy"],
        sandbox_policy=row["sandbox_policy"],
        network_policy=row["network_policy"],
        observed_at_ms=row["observed_at_ms"],
        source_message_id=row["source_message_id"],
    )


async def get_session_agent_policies(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[ArchiveAgentPolicy]:
    """Return all agent-policy rows for one session, ordered by position."""
    rows = await (
        await conn.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM session_agent_policies
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        )
    ).fetchall()
    return [_row_to_agent_policy(row) for row in rows]


async def get_session_agent_policies_batch(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[ArchiveAgentPolicy]]:
    """Return agent-policy rows for many sessions, grouped by session_id."""
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = await (
        await conn.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM session_agent_policies
            WHERE session_id IN ({placeholders})
            ORDER BY session_id, position
            """,
            tuple(session_ids),
        )
    ).fetchall()
    result: dict[str, list[ArchiveAgentPolicy]] = defaultdict(list)
    for session_id in session_ids:
        result.setdefault(session_id, [])
    for row in rows:
        policy = _row_to_agent_policy(row)
        result[policy.session_id].append(policy)
    return dict(result)
