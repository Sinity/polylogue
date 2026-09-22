"""Delete-side upkeep for durable session insights.

polylogue-foour.  This module used to own a *second* profile lifecycle beside
:mod:`polylogue.storage.derived.session.derivation`: an incremental
"refresh" that rebuilt ``session_profiles`` / ``session_latency_profiles`` and
stamped ``input_content_hash`` itself.  Stamping that binding is a
certification -- :func:`inspect_session_profiles` treats a matching binding as
proof the rows were computed from the current inputs -- and this route made it
without the checks the owning domain makes:
:func:`~polylogue.storage.derived.session.derivation.publish_session_profile`
refuses to certify a profile whose ``session_usage_rollup`` prerequisite is not
VALID, because the profile reads canonical ``session_model_usage`` values, and
the refresh route had no such refusal.  Two writers of one output, one of which
could certify rows built from a superseded rollup.

The update side is therefore gone, not renamed: profile convergence has exactly
one owner, the domain.  What remains here is the delete side, which is not a
lifecycle at all -- when a session row goes away its derived rows must go with
it inside the caller's own transaction
(:mod:`polylogue.storage.repository.archive.writes.sessions`).
"""

from __future__ import annotations

import aiosqlite

from polylogue.storage.derived.session.aggregates import profile_provider_day
from polylogue.storage.derived.session.runtime import SessionInsightCounts
from polylogue.storage.sqlite.queries.mappers import _row_to_session_profile_record

__all__ = [
    "delete_session_insights_for_session_async",
    "refresh_thread_after_session_delete_async",
]


async def refresh_thread_after_session_delete_async(
    conn: aiosqlite.Connection,
    root_id: str | None,
    *,
    transaction_depth: int = 0,
) -> int:
    """Count the thread rows still standing behind a deleted session's root."""
    del transaction_depth
    if root_id is None:
        return 0
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM threads WHERE thread_id = ?",
        (root_id,),
    )
    row = await cursor.fetchone()
    return int(row[0]) if row is not None else 0


async def delete_session_insights_for_session_async(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    transaction_depth: int = 0,
) -> SessionInsightCounts:
    """Drop one session's derived profile rows in the caller's transaction."""
    del transaction_depth
    cursor = await conn.execute(
        "SELECT * FROM session_profiles WHERE session_id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    old_group = profile_provider_day(_row_to_session_profile_record(row)) if row else None
    await conn.execute("DELETE FROM session_profiles WHERE session_id = ?", (session_id,))
    await conn.execute("DELETE FROM session_latency_profiles WHERE session_id = ?", (session_id,))
    counts = SessionInsightCounts()
    counts.add(
        profiles=1 if row is not None else 0,
        tag_rollups=1 if old_group is not None else 0,
    )
    return counts
