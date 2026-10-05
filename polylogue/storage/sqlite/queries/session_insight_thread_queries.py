"""Thread durable session-insight queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.runtime import ThreadRecord

__all__ = [
    "get_thread",
]


async def get_thread(
    conn: aiosqlite.Connection,
    thread_id: str,
) -> ThreadRecord | None:
    from polylogue.storage.derived.session.threads import build_thread_records_for_roots_async

    records = await build_thread_records_for_roots_async(conn, [thread_id])
    return records.get(thread_id)
