"""Metadata mutation helpers for the repository mixin."""

from __future__ import annotations

from collections.abc import Callable

from polylogue.core.json import JSONDocument
from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.sqlite.queries import sessions as sessions_q


async def metadata_read_modify_write(
    backend: RepositoryBackendProtocol,
    session_id: str,
    mutator: Callable[[JSONDocument], bool],
) -> bool:
    async with backend.transaction(), backend.connection() as conn:
        current = await sessions_q.get_metadata(conn, session_id)
        if mutator(current):
            await sessions_q.update_metadata_raw(
                conn,
                session_id,
                current,
            )
            return True
        return False
