"""Write/admin method mixin for the session repository."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.repository.archive.writes.metadata import metadata_read_modify_write
from polylogue.storage.repository.archive.writes.sessions import (
    delete_session_via_backend,
)
from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.search.cache import invalidate_search_cache
from polylogue.storage.sqlite.queries import sessions as sessions_q


class RepositoryWriteMixin:
    if TYPE_CHECKING:
        _backend: RepositoryBackendProtocol

    async def save_parsed_session(self, session: ParsedSession, content_hash: str) -> dict[str, int]:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        db_path = Path(self._backend.db_path)
        with ArchiveStore(db_path.parent) as archive:
            archive.write_parsed(session, content_hash=content_hash)
        invalidate_search_cache()
        return {
            "sessions": 1,
            "messages": len(session.messages),
            "attachments": len(session.attachments),
            "session_events": len(session.session_events),
            "skipped_sessions": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
            "skipped_session_events": 0,
        }

    async def get_metadata(self, session_id: str) -> JSONDocument:
        async with self._backend.connection() as conn:
            return await sessions_q.get_metadata(conn, session_id)

    async def _upsert_normalized_tag(self, session_id: str, tag_name: str) -> None:
        """Upsert a tag into the normalized tags table and link to session."""
        async with self._backend.transaction(), self._backend.connection() as conn:
            await conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
            cursor = await conn.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
            row = await cursor.fetchone()
            if row is not None:
                tag_id = row["id"]
                await conn.execute(
                    "INSERT OR IGNORE INTO session_tags (session_id, tag_id) VALUES (?, ?)",
                    (session_id, tag_id),
                )

    async def _metadata_read_modify_write(
        self,
        session_id: str,
        mutator: Callable[[JSONDocument], bool],
    ) -> bool:
        return await metadata_read_modify_write(self._backend, session_id, mutator)

    async def update_metadata(self, session_id: str, key: str, value: JSONValue) -> bool:
        def _set(meta: JSONDocument) -> bool:
            if key in meta and meta[key] == value:
                return False
            meta[key] = value
            return True

        return await self._metadata_read_modify_write(session_id, _set)

    async def add_tag(self, session_id: str, tag: str) -> bool:
        # #1240: tags are stored only in the M2M tables (tags + session_tags).
        # The previous dual-write into ``sessions.metadata['tags']`` was
        # obsolete bookkeeping for the JSON read-fallback that has been removed.
        if not tag or not tag.strip():
            raise ValueError("tag must be a non-empty string")
        if len(tag) > 200:
            raise ValueError("tag must be at most 200 characters")

        async with self._backend.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT 1 FROM session_tags ct
                JOIN tags t ON t.id = ct.tag_id
                WHERE ct.session_id = ? AND t.name = ?
                """,
                (session_id, tag),
            )
            already_present = await cursor.fetchone() is not None

        await self._upsert_normalized_tag(session_id, tag)
        return not already_present

    async def delete_session(self, session_id: str) -> bool:
        return await delete_session_via_backend(self._backend, session_id)
