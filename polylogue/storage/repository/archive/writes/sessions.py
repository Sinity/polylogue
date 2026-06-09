"""Session delete helpers for the repository mixin."""

from __future__ import annotations

from polylogue.storage.insights.session.refresh import (
    delete_session_insights_for_session_async,
    refresh_thread_after_session_delete_async,
)
from polylogue.storage.insights.session.threads import thread_root_id_async
from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.search.cache import invalidate_search_cache


async def delete_session_via_backend(
    backend: RepositoryBackendProtocol,
    session_id: str,
) -> bool:
    from polylogue.storage.sqlite.queries import sessions as sessions_q

    async with backend.transaction(), backend.connection() as conn:
        parent_row = await (
            await conn.execute(
                "SELECT parent_session_id FROM sessions WHERE session_id = ?",
                (session_id,),
            )
        ).fetchone()
        child_rows = await (
            await conn.execute(
                "SELECT session_id FROM sessions WHERE parent_session_id = ?",
                (session_id,),
            )
        ).fetchall()
        existing_row = await (
            await conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,),
            )
        ).fetchone()
        deleted = False
        if existing_row is not None:
            await delete_session_insights_for_session_async(
                conn,
                session_id,
                transaction_depth=backend.transaction_depth,
            )
            deleted = await sessions_q.delete_session_sql(
                conn,
                session_id,
                backend.transaction_depth,
            )
        if deleted:
            affected_seeds = {str(row["session_id"]) for row in child_rows}
            if parent_row is not None and parent_row["parent_session_id"] is not None:
                affected_seeds.add(str(parent_row["parent_session_id"]))
            affected_roots: set[str] = set()
            for seed in affected_seeds:
                root_id = await thread_root_id_async(conn, seed)
                if root_id is not None:
                    affected_roots.add(root_id)
            for root_id in affected_roots:
                await refresh_thread_after_session_delete_async(
                    conn,
                    root_id,
                    transaction_depth=backend.transaction_depth,
                )
    if deleted:
        invalidate_search_cache()
    return deleted


__all__ = ["delete_session_via_backend"]
