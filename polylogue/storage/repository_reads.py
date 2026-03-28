"""Read/query method mixin for the conversation repository."""

from __future__ import annotations

import asyncio
import builtins
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.storage.hydrators import (
    conversation_from_records,
    conversation_summary_from_record,
    message_from_record,
)
from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.storage.state_views import ConversationRenderProjection
from polylogue.storage.store import ConversationRecord
from polylogue.types import ConversationId

if TYPE_CHECKING:
    from polylogue.lib import filters


class RepositoryReadMixin:
    async def resolve_id(self, id_prefix: str) -> ConversationId | None:
        resolved = await self.queries.resolve_id(id_prefix)
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

    async def _hydrate_conversations(
        self,
        conversation_records: builtins.list[ConversationRecord],
        *,
        ordered_ids: builtins.list[str] | None = None,
    ) -> builtins.list[Conversation]:
        if not conversation_records:
            return []

        by_id = {record.conversation_id: record for record in conversation_records}
        conversation_ids = ordered_ids or [record.conversation_id for record in conversation_records]
        present_ids = [conversation_id for conversation_id in conversation_ids if conversation_id in by_id]
        if not present_ids:
            return []

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
    ) -> builtins.list[ConversationSummary]:
        conv_records = await self.queries.list_conversations(query)
        return [conversation_summary_from_record(rec) for rec in conv_records]

    async def list_summaries(
        self,
        limit: int | None = 50,
        offset: int = 0,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        source: str | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        path_terms: builtins.list[str] | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
    ) -> builtins.list[ConversationSummary]:
        return await self.list_summaries_by_query(
            ConversationRecordQuery(
                source=source,
                provider=provider,
                providers=tuple(providers or ()),
                since=since,
                until=until,
                title_contains=title_contains,
                path_terms=tuple(path_terms or ()),
                action_terms=tuple(action_terms or ()),
                excluded_action_terms=tuple(excluded_action_terms or ()),
                tool_terms=tuple(tool_terms or ()),
                excluded_tool_terms=tuple(excluded_tool_terms or ()),
                limit=limit,
                offset=offset,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
            )
        )

    async def iter_summary_pages(
        self,
        *,
        page_size: int = 50,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        source: str | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        path_terms: builtins.list[str] | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
    ) -> AsyncIterator[builtins.list[ConversationSummary]]:
        offset = 0
        while True:
            page = await self.list_summaries(
                limit=page_size,
                offset=offset,
                provider=provider,
                providers=providers,
                source=source,
                since=since,
                until=until,
                title_contains=title_contains,
                path_terms=path_terms,
                action_terms=action_terms,
                excluded_action_terms=excluded_action_terms,
                tool_terms=tool_terms,
                excluded_tool_terms=excluded_tool_terms,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
            )
            if not page:
                break
            yield page
            if len(page) < page_size:
                break
            offset += len(page)

    async def list_by_query(
        self,
        query: ConversationRecordQuery,
    ) -> builtins.list[Conversation]:
        conv_records = await self.queries.list_conversations(query)
        return await self._hydrate_conversations(conv_records)

    async def list(
        self,
        limit: int | None = 50,
        offset: int = 0,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        path_terms: builtins.list[str] | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
    ) -> builtins.list[Conversation]:
        return await self.list_by_query(
            ConversationRecordQuery(
                provider=provider,
                providers=tuple(providers or ()),
                since=since,
                until=until,
                title_contains=title_contains,
                path_terms=tuple(path_terms or ()),
                action_terms=tuple(action_terms or ()),
                excluded_action_terms=tuple(excluded_action_terms or ()),
                tool_terms=tuple(tool_terms or ()),
                excluded_tool_terms=tuple(excluded_tool_terms or ()),
                limit=limit,
                offset=offset,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
            )
        )

    async def count_by_query(self, query: ConversationRecordQuery) -> int:
        return await self.queries.count_conversations(query)

    async def count(
        self,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        path_terms: builtins.list[str] | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
    ) -> int:
        return await self.count_by_query(
            ConversationRecordQuery(
                provider=provider,
                providers=tuple(providers or ()),
                since=since,
                until=until,
                title_contains=title_contains,
                path_terms=tuple(path_terms or ()),
                action_terms=tuple(action_terms or ()),
                excluded_action_terms=tuple(excluded_action_terms or ()),
                tool_terms=tuple(tool_terms or ()),
                excluded_tool_terms=tuple(excluded_tool_terms or ()),
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
            )
        )

    async def get_parent(self, conversation_id: str) -> Conversation | None:
        conv_record = await self.queries.get_conversation(conversation_id)
        if conv_record and conv_record.parent_conversation_id:
            return await self.get(str(conv_record.parent_conversation_id))
        return None

    async def get_children(self, conversation_id: str) -> builtins.list[Conversation]:
        child_records = await self.queries.list_conversations(
            ConversationRecordQuery(parent_id=conversation_id)
        )
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

    async def get_session_tree(self, conversation_id: str) -> builtins.list[Conversation]:
        root_record = await self._get_root_record(conversation_id)

        tree_records: builtins.list[ConversationRecord] = []
        queue: builtins.list[ConversationRecord] = [root_record]

        while queue:
            current = queue.pop(0)
            tree_records.append(current)
            children = await self.queries.list_conversations(
                ConversationRecordQuery(parent_id=current.conversation_id)
            )
            queue.extend(children)

        return await self._hydrate_conversations(
            tree_records,
            ordered_ids=[record.conversation_id for record in tree_records],
        )

    async def search_summaries(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ) -> builtins.list[ConversationSummary]:
        ids, records = await self._search_records(query, limit=limit, providers=providers)
        if not ids:
            return []
        return [conversation_summary_from_record(rec) for rec in records]

    async def search(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ) -> builtins.list[Conversation]:
        ids, records = await self._search_records(query, limit=limit, providers=providers)
        return await self._hydrate_conversations(records, ordered_ids=ids)

    async def search_actions(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ) -> builtins.list[Conversation]:
        ids, records = await self._search_action_records(query, limit=limit, providers=providers)
        return await self._hydrate_conversations(records, ordered_ids=ids)

    async def _search_records(
        self,
        query: str,
        *,
        limit: int,
        providers: builtins.list[str] | None,
    ) -> tuple[builtins.list[str], builtins.list[ConversationRecord]]:
        ids = await self.queries.search_conversations(query, limit=limit, providers=providers)
        if not ids:
            return [], []
        records = await self.queries.get_conversations_batch(ids)
        by_id = {record.conversation_id: record for record in records}
        return ids, [by_id[conversation_id] for conversation_id in ids if conversation_id in by_id]

    async def _search_action_records(
        self,
        query: str,
        *,
        limit: int,
        providers: builtins.list[str] | None,
    ) -> tuple[builtins.list[str], builtins.list[ConversationRecord]]:
        ids = await self.queries.search_action_conversations(query, limit=limit, providers=providers)
        if not ids:
            return [], []
        records = await self.queries.get_conversations_batch(ids)
        by_id = {record.conversation_id: record for record in records}
        return ids, [by_id[conversation_id] for conversation_id in ids if conversation_id in by_id]

    async def get_many(self, conversation_ids: builtins.list[str]) -> builtins.list[Conversation]:
        if not conversation_ids:
            return []

        records = await self.queries.get_conversations_batch(conversation_ids)
        return await self._hydrate_conversations(records, ordered_ids=conversation_ids)

    async def iter_messages(
        self,
        conversation_id: str,
        *,
        dialogue_only: bool = False,
        limit: int | None = None,
    ) -> AsyncIterator[Message]:
        conv_record = await self.queries.get_conversation(conversation_id)
        provider_name = conv_record.provider_name if conv_record else None
        async for record in self.queries.iter_messages(
            conversation_id,
            dialogue_only=dialogue_only,
            limit=limit,
        ):
            yield message_from_record(record, attachments=[], provider=provider_name)

    def filter(self) -> filters.ConversationFilter:
        from polylogue.lib import filters

        return filters.ConversationFilter(self)
