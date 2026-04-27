"""Query-shaping archive reads for the repository."""

from __future__ import annotations

import builtins
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from polylogue.lib.conversation.models import Conversation, ConversationSummary
from polylogue.protocols import ConversationQueryRuntimeStore
from polylogue.storage.backends.queries.stats import AggregateMessageStats
from polylogue.storage.query_models import ConversationRecordQuery

if TYPE_CHECKING:
    from polylogue.lib.filter.filters import ConversationFilter
    from polylogue.storage.backends.query_store import SQLiteQueryStore


class RepositoryArchiveQueryMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

        async def list_summaries_by_query(
            self,
            query: ConversationRecordQuery,
        ) -> list[ConversationSummary]: ...

        async def list_by_query(
            self,
            query: ConversationRecordQuery,
        ) -> list[Conversation]: ...

    async def list_summaries(
        self,
        limit: int | None = 50,
        offset: int = 0,
        provider: str | None = None,
        providers: list[str] | None = None,
        source: str | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        path_terms: list[str] | None = None,
        action_terms: list[str] | None = None,
        excluded_action_terms: list[str] | None = None,
        tool_terms: list[str] | None = None,
        excluded_tool_terms: list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
    ) -> list[ConversationSummary]:
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
        providers: list[str] | None = None,
        source: str | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        path_terms: list[str] | None = None,
        action_terms: list[str] | None = None,
        excluded_action_terms: list[str] | None = None,
        tool_terms: list[str] | None = None,
        excluded_tool_terms: list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
    ) -> AsyncIterator[list[ConversationSummary]]:
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

    async def list(
        self,
        limit: int | None = 50,
        offset: int = 0,
        provider: str | None = None,
        providers: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        path_terms: list[str] | None = None,
        action_terms: list[str] | None = None,
        excluded_action_terms: list[str] | None = None,
        tool_terms: list[str] | None = None,
        excluded_tool_terms: list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
    ) -> list[Conversation]:
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

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, int]:
        return await self.queries.get_conversation_stats(conversation_id)

    async def get_message_counts_batch(self, conversation_ids: builtins.list[str]) -> dict[str, int]:
        return await self.queries.get_message_counts_batch(conversation_ids)

    async def aggregate_message_stats(
        self,
        conversation_ids: builtins.list[str] | None = None,
    ) -> AggregateMessageStats:
        return await self.queries.aggregate_message_stats(conversation_ids)

    async def get_stats_by(self, group_by: str = "provider") -> dict[str, int]:
        return await self.queries.get_stats_by(group_by)

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

    def filter(self: ConversationQueryRuntimeStore) -> ConversationFilter:
        from polylogue.lib.filter import filters

        return filters.ConversationFilter(self)


__all__ = ["RepositoryArchiveQueryMixin"]
