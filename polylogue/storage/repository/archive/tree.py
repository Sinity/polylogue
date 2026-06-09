"""Parent/child/root tree reads for the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.session.domain_models import Session
from polylogue.storage.query_models import SessionRecordQuery
from polylogue.storage.runtime import SessionRecord

if TYPE_CHECKING:
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class RepositoryArchiveTreeMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

        async def get(self, session_id: str) -> Session | None: ...

        async def _hydrate_sessions(
            self,
            session_records: list[SessionRecord],
            *,
            ordered_ids: list[str] | None = None,
        ) -> list[Session]: ...

    async def get_parent(self, session_id: str) -> Session | None:
        conv_record = await self.queries.get_session(session_id)
        if conv_record and conv_record.parent_session_id:
            return await self.get(str(conv_record.parent_session_id))
        return None

    async def get_children(self, session_id: str) -> list[Session]:
        child_records = await self.queries.list_sessions(SessionRecordQuery(parent_id=session_id))
        if not child_records:
            return []
        return await self._hydrate_sessions(child_records)

    async def _get_root_record(self, session_id: str) -> SessionRecord:
        current = await self.queries.get_session(session_id)
        if not current:
            raise ValueError(f"Session {session_id} not found")

        while current.parent_session_id:
            parent = await self.queries.get_session(str(current.parent_session_id))
            if not parent:
                break
            current = parent
        return current

    async def get_root(self, session_id: str) -> Session:
        root_record = await self._get_root_record(session_id)
        root = await self.get(root_record.session_id)
        if root is None:
            raise ValueError(f"Session {session_id} not found")
        return root

    async def get_session_tree(self, session_id: str) -> list[Session]:
        root_record = await self._get_root_record(session_id)

        tree_records: list[SessionRecord] = []
        queue: list[SessionRecord] = [root_record]

        while queue:
            current = queue.pop(0)
            tree_records.append(current)
            children = await self.queries.list_sessions(SessionRecordQuery(parent_id=current.session_id))
            queue.extend(children)

        return await self._hydrate_sessions(
            tree_records,
            ordered_ids=[record.session_id for record in tree_records],
        )


__all__ = ["RepositoryArchiveTreeMixin"]
