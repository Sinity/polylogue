"""Conversation-oriented sync facade methods."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.api.sync.bridge import run_coroutine_sync

if TYPE_CHECKING:
    from polylogue.api import ArchiveStats, Polylogue
    from polylogue.lib.conversation.models import Conversation, ConversationSummary
    from polylogue.storage.search import SearchResult


class SyncConversationQueriesMixin:
    """Conversation and archive query helpers for ``SyncPolylogue``."""

    _facade: Polylogue

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        return run_coroutine_sync(self._facade.get_conversation(conversation_id))

    def get_conversations(self, conversation_ids: list[str]) -> list[Conversation]:
        return run_coroutine_sync(self._facade.get_conversations(conversation_ids))

    def list_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
    ) -> list[Conversation]:
        return run_coroutine_sync(self._facade.list_conversations(provider=provider, limit=limit))

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


__all__ = ["SyncConversationQueriesMixin"]
