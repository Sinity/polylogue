"""Conversation-oriented sync facade methods."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.api.sync.bridge import run_coroutine_sync

if TYPE_CHECKING:
    from polylogue.api import ArchiveStats, Polylogue
    from polylogue.archive.conversation.models import Conversation, ConversationSummary
    from polylogue.archive.conversation.neighbor_candidates import ConversationNeighborCandidate
    from polylogue.archive.message.models import Message
    from polylogue.archive.message.roles import MessageRoleFilter
    from polylogue.archive.semantic.content_projection import ContentProjectionSpec
    from polylogue.readiness import ReadinessReport
    from polylogue.storage.search import SearchResult
    from polylogue.storage.sqlite.queries.message_query_reads import MessageTypeName


class SyncConversationQueriesMixin:
    """Conversation and archive query helpers for ``SyncPolylogue``."""

    _facade: Polylogue

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        return run_coroutine_sync(self._facade.get_conversation(conversation_id))

    def get_conversations(self, conversation_ids: list[str]) -> list[Conversation]:
        return run_coroutine_sync(self._facade.get_conversations(conversation_ids))

    def get_messages_paginated(
        self,
        conversation_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        limit: int = 50,
        offset: int = 0,
        content_projection: ContentProjectionSpec | None = None,
    ) -> tuple[list[Message], int]:
        return run_coroutine_sync(
            self._facade.get_messages_paginated(
                conversation_id,
                message_role=message_role,
                message_type=message_type,
                limit=limit,
                offset=offset,
                content_projection=content_projection,
            )
        )

    def bulk_get_messages(
        self,
        conversation_ids: Sequence[str],
        *,
        since: str | None = None,
        until: str | None = None,
        message_role: MessageRoleFilter = (),
        content_projection: ContentProjectionSpec | None = None,
    ) -> dict[str, list[Message]]:
        return run_coroutine_sync(
            self._facade.bulk_get_messages(
                conversation_ids,
                since=since,
                until=until,
                message_role=message_role,
                content_projection=content_projection,
            )
        )

    def list_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
    ) -> list[Conversation]:
        return run_coroutine_sync(self._facade.list_conversations(provider=provider, limit=limit))

    def query_conversations(
        self,
        *,
        provider: str | None = None,
        tag: str | None = None,
        since: str | None = None,
        until: str | None = None,
        sort: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        typed_only: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        return run_coroutine_sync(
            self._facade.query_conversations(
                provider=provider,
                tag=tag,
                since=since,
                until=until,
                sort=sort,
                limit=limit,
                offset=offset,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                has_paste=has_paste,
                typed_only=typed_only,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                **kwargs,
            )
        )

    def count_conversations(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        **kwargs: object,
    ) -> int:
        return run_coroutine_sync(
            self._facade.count_conversations(
                provider=provider,
                since=since,
                until=until,
                **kwargs,
            )
        )

    def list_summaries(
        self,
        *,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        provider: str | None = None,
        limit: int | None = None,
    ) -> list[ConversationSummary]:
        filt = self._facade.filter()
        if provider:
            filt = filt.provider(provider)
        if since:
            filt = filt.since(since)
        if until:
            filt = filt.until(until)
        if limit:
            filt = filt.limit(limit)
        return run_coroutine_sync(filt.list_summaries())

    def get_conversation_summary(self, conversation_id: str) -> ConversationSummary | None:
        return run_coroutine_sync(self._facade.get_conversation_summary(conversation_id))

    def get_conversation_stats(self, conversation_id: str) -> dict[str, int]:
        return run_coroutine_sync(self._facade.get_conversation_stats(conversation_id))

    def get_session_tree(self, conversation_id: str) -> list[Conversation]:
        return run_coroutine_sync(self._facade.get_session_tree(conversation_id))

    def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        return run_coroutine_sync(self._facade.list_tags(provider=provider))

    def search(
        self,
        query: str,
        *,
        limit: int = 100,
        source: str | None = None,
        since: str | None = None,
    ) -> SearchResult:
        return run_coroutine_sync(self._facade.search(query, limit=limit, source=source, since=since))

    def stats(self) -> ArchiveStats:
        return run_coroutine_sync(self._facade.stats())

    def health_check(self) -> ReadinessReport:
        return run_coroutine_sync(self._facade.health_check())

    def neighbor_candidates(
        self,
        *,
        conversation_id: str | None = None,
        query: str | None = None,
        provider: str | None = None,
        limit: int = 10,
        window_hours: int = 24,
    ) -> list[ConversationNeighborCandidate]:
        return run_coroutine_sync(
            self._facade.neighbor_candidates(
                conversation_id=conversation_id,
                query=query,
                provider=provider,
                limit=limit,
                window_hours=window_hours,
            )
        )


__all__ = ["SyncConversationQueriesMixin"]
