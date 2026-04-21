"""Conversation and message hydration reads for the repository."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from polylogue.lib.conversation_models import Conversation, ConversationSummary
from polylogue.lib.message_models import Message
from polylogue.lib.message_roles import MessageRoleFilter
from polylogue.storage.archive_views import ConversationRenderProjection
from polylogue.storage.hydrators import (
    conversation_from_records,
    conversation_summary_from_record,
    message_from_record,
)
from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.storage.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord

if TYPE_CHECKING:
    from polylogue.storage.backends.query_store import SQLiteQueryStore
    from polylogue.types import ConversationId


class RepositoryArchiveConversationMixin:
    if TYPE_CHECKING:
        _backend: RepositoryBackendProtocol
        queries: SQLiteQueryStore

    async def resolve_id(self, id_prefix: str) -> ConversationId | None:
        resolved = await self.queries.resolve_id(id_prefix)
        from polylogue.types import ConversationId

        return ConversationId(resolved) if resolved else None

    async def get(self, conversation_id: str) -> Conversation | None:
        conv_record = await self.queries.get_conversation(conversation_id)
        if not conv_record:
            return None

        msg_records, att_records = await asyncio.gather(
            self.queries.get_messages(conversation_id),
            self.queries.get_attachments(conversation_id),
        )
        return conversation_from_records(conv_record, msg_records, att_records)

    async def get_render_projection(self, conversation_id: str) -> ConversationRenderProjection | None:
        conv_record = await self.queries.get_conversation(conversation_id)
        if not conv_record:
            return None

        msg_records, att_records = await asyncio.gather(
            self.queries.get_messages(conversation_id),
            self.queries.get_attachments(conversation_id),
        )
        return ConversationRenderProjection(
            conversation=conv_record,
            messages=msg_records,
            attachments=att_records,
        )

    async def view(self, conversation_id: str) -> Conversation | None:
        full_id = await self.resolve_id(conversation_id) or conversation_id
        return await self.get(str(full_id))

    async def get_eager(self, conversation_id: str) -> Conversation | None:
        return await self.get(conversation_id)

    async def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        return await self.queries.get_messages(conversation_id)

    async def get_conversations_batch(self, ids: list[str]) -> list[ConversationRecord]:
        return await self.queries.get_conversations_batch(ids)

    async def get_messages_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[MessageRecord]]:
        return await self.queries.get_messages_batch(conversation_ids)

    async def get_attachments_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[AttachmentRecord]]:
        return await self.queries.get_attachments_batch(conversation_ids)

    async def _hydrate_conversations(
        self,
        conversation_records: list[ConversationRecord],
        *,
        ordered_ids: list[str] | None = None,
    ) -> list[Conversation]:
        if not conversation_records:
            return []

        by_id: dict[str, ConversationRecord] = {str(record.conversation_id): record for record in conversation_records}
        conversation_ids = ordered_ids or [record.conversation_id for record in conversation_records]
        present_ids = [conversation_id for conversation_id in conversation_ids if conversation_id in by_id]
        if not present_ids:
            return []

        async with self._backend.read_pool(size=2):
            msgs_by_id, atts_by_id = await asyncio.gather(
                self.queries.get_messages_batch(present_ids),
                self.queries.get_attachments_batch(present_ids),
            )
        return [
            conversation_from_records(
                by_id[conversation_id],
                msgs_by_id.get(conversation_id, []),
                atts_by_id.get(conversation_id, []),
            )
            for conversation_id in present_ids
        ]

    async def get_summary(self, conversation_id: str) -> ConversationSummary | None:
        conv_record = await self.queries.get_conversation(conversation_id)
        if not conv_record:
            return None
        return conversation_summary_from_record(conv_record)

    async def list_summaries_by_query(
        self,
        query: ConversationRecordQuery,
    ) -> list[ConversationSummary]:
        conv_records = await self.queries.list_conversation_summaries(query)
        return [conversation_summary_from_record(record) for record in conv_records]

    async def list_by_query(
        self,
        query: ConversationRecordQuery,
    ) -> list[Conversation]:
        conv_records = await self.queries.list_conversations(query)
        return await self._hydrate_conversations(conv_records)

    async def get_many(self, conversation_ids: list[str]) -> list[Conversation]:
        if not conversation_ids:
            return []
        records = await self.queries.get_conversations_batch(conversation_ids)
        return await self._hydrate_conversations(records, ordered_ids=conversation_ids)

    async def iter_messages(
        self,
        conversation_id: str,
        *,
        dialogue_only: bool = False,
        message_roles: MessageRoleFilter = (),
        limit: int | None = None,
    ) -> AsyncIterator[Message]:
        conv_record = await self.queries.get_conversation(conversation_id)
        provider_name = conv_record.provider_name if conv_record else None
        async for record in self.queries.iter_messages(
            conversation_id,
            dialogue_only=dialogue_only,
            message_roles=message_roles,
            limit=limit,
        ):
            yield message_from_record(record, attachments=[], provider=provider_name)


__all__ = ["RepositoryArchiveConversationMixin"]
