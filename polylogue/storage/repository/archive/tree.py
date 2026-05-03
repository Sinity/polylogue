"""Parent/child/root tree reads for the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.conversation.models import Conversation
from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.storage.runtime import ConversationRecord

if TYPE_CHECKING:
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class RepositoryArchiveTreeMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

        async def get(self, conversation_id: str) -> Conversation | None: ...

        async def _hydrate_conversations(
            self,
            conversation_records: list[ConversationRecord],
            *,
            ordered_ids: list[str] | None = None,
        ) -> list[Conversation]: ...

    async def get_parent(self, conversation_id: str) -> Conversation | None:
        conv_record = await self.queries.get_conversation(conversation_id)
        if conv_record and conv_record.parent_conversation_id:
            return await self.get(str(conv_record.parent_conversation_id))
        return None

    async def get_children(self, conversation_id: str) -> list[Conversation]:
        child_records = await self.queries.list_conversations(ConversationRecordQuery(parent_id=conversation_id))
        if not child_records:
            return []
        return await self._hydrate_conversations(child_records)

    async def _get_root_record(self, conversation_id: str) -> ConversationRecord:
        current = await self.queries.get_conversation(conversation_id)
        if not current:
            raise ValueError(f"Conversation {conversation_id} not found")

        while current.parent_conversation_id:
            parent = await self.queries.get_conversation(str(current.parent_conversation_id))
            if not parent:
                break
            current = parent
        return current

    async def get_root(self, conversation_id: str) -> Conversation:
        root_record = await self._get_root_record(conversation_id)
        root = await self.get(root_record.conversation_id)
        if root is None:
            raise ValueError(f"Conversation {conversation_id} not found")
        return root

    async def get_session_tree(self, conversation_id: str) -> list[Conversation]:
        root_record = await self._get_root_record(conversation_id)

        tree_records: list[ConversationRecord] = []
        queue: list[ConversationRecord] = [root_record]

        while queue:
            current = queue.pop(0)
            tree_records.append(current)
            children = await self.queries.list_conversations(ConversationRecordQuery(parent_id=current.conversation_id))
            queue.extend(children)

        return await self._hydrate_conversations(
            tree_records,
            ordered_ids=[record.conversation_id for record in tree_records],
        )


__all__ = ["RepositoryArchiveTreeMixin"]
