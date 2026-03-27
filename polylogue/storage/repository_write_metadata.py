"""Metadata mutation helpers for the repository mixin."""

from __future__ import annotations

from collections.abc import Callable

from polylogue.storage.backends.queries import conversations as conversations_q


async def metadata_read_modify_write(
    backend,
    conversation_id: str,
    mutator: Callable[[dict[str, object]], bool],
) -> None:
    async with backend.transaction(), backend.connection() as conn:
        current = await conversations_q.get_metadata(conn, conversation_id)
        if mutator(current):
            await conversations_q.update_metadata_raw(
                conn,
                conversation_id,
                current,
            )
