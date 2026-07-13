"""Query-shaping archive reads for the repository."""

from __future__ import annotations

import builtins
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.storage.query_models import SessionRecordQuery
from polylogue.storage.sqlite.queries.stats import AggregateMessageStats

if TYPE_CHECKING:
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class RepositoryArchiveQueryMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

        async def list_summaries_by_query(
            self,
            query: SessionRecordQuery,
        ) -> list[SessionSummary]: ...

        async def list_by_query(
            self,
            query: SessionRecordQuery,
        ) -> list[Session]: ...

    async def list_summaries(
        self,
        limit: int | None = 50,
        offset: int = 0,
        origin: str | None = None,
        origins: list[str] | None = None,
        source: str | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        referenced_path: list[str] | None = None,
        cwd_prefix: str | None = None,
        action_terms: list[str] | None = None,
        excluded_action_terms: list[str] | None = None,
        tool_terms: list[str] | None = None,
        excluded_tool_terms: list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        message_type: str | None = None,
    ) -> list[SessionSummary]:
        return await self.list_summaries_by_query(
            SessionRecordQuery(
                source=source,
                origin=origin,
                origins=tuple(origins or ()),
                since=since,
                until=until,
                title_contains=title_contains,
                referenced_path=tuple(referenced_path or ()),
                cwd_prefix=cwd_prefix,
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
                max_words=max_words,
                message_type=message_type,
            )
        )

    async def iter_summary_pages(
        self,
        *,
        page_size: int = 50,
        origin: str | None = None,
        origins: list[str] | None = None,
        source: str | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        referenced_path: list[str] | None = None,
        cwd_prefix: str | None = None,
        action_terms: list[str] | None = None,
        excluded_action_terms: list[str] | None = None,
        tool_terms: list[str] | None = None,
        excluded_tool_terms: list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        message_type: str | None = None,
    ) -> AsyncIterator[list[SessionSummary]]:
        offset = 0
        while True:
            page = await self.list_summaries(
                limit=page_size,
                offset=offset,
                origin=origin,
                origins=origins,
                source=source,
                since=since,
                until=until,
                title_contains=title_contains,
                referenced_path=referenced_path,
                cwd_prefix=cwd_prefix,
                action_terms=action_terms,
                excluded_action_terms=excluded_action_terms,
                tool_terms=tool_terms,
                excluded_tool_terms=excluded_tool_terms,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                max_words=max_words,
                message_type=message_type,
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
        origin: str | None = None,
        origins: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        referenced_path: list[str] | None = None,
        cwd_prefix: str | None = None,
        action_terms: list[str] | None = None,
        excluded_action_terms: list[str] | None = None,
        tool_terms: list[str] | None = None,
        excluded_tool_terms: list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        message_type: str | None = None,
    ) -> list[Session]:
        return await self.list_by_query(
            SessionRecordQuery(
                origin=origin,
                origins=tuple(origins or ()),
                since=since,
                until=until,
                title_contains=title_contains,
                referenced_path=tuple(referenced_path or ()),
                cwd_prefix=cwd_prefix,
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
                max_words=max_words,
                message_type=message_type,
            )
        )

    async def count_by_query(self, query: SessionRecordQuery) -> int:
        return await self.queries.count_sessions(query)

    async def get_session_stats(self, session_id: str) -> dict[str, int]:
        return await self.queries.get_session_stats(session_id)

    async def get_message_counts_batch(self, session_ids: builtins.list[str]) -> dict[str, int]:
        return await self.queries.get_message_counts_batch(session_ids)

    async def aggregate_message_stats(
        self,
        session_ids: builtins.list[str] | None = None,
    ) -> AggregateMessageStats:
        return await self.queries.aggregate_message_stats(session_ids)

    async def get_stats_by(self, group_by: str = "origin") -> dict[str, int]:
        return await self.queries.get_stats_by(group_by)

    async def count(
        self,
        origin: str | None = None,
        origins: builtins.list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        referenced_path: builtins.list[str] | None = None,
        cwd_prefix: str | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        message_type: str | None = None,
    ) -> int:
        return await self.count_by_query(
            SessionRecordQuery(
                origin=origin,
                origins=tuple(origins or ()),
                since=since,
                until=until,
                title_contains=title_contains,
                referenced_path=tuple(referenced_path or ()),
                cwd_prefix=cwd_prefix,
                action_terms=tuple(action_terms or ()),
                excluded_action_terms=tuple(excluded_action_terms or ()),
                tool_terms=tuple(tool_terms or ()),
                excluded_tool_terms=tuple(excluded_tool_terms or ()),
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                max_words=max_words,
                message_type=message_type,
            )
        )


__all__ = ["RepositoryArchiveQueryMixin"]
