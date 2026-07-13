"""Session-oriented sync facade methods."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.api.sync.bridge import run_coroutine_sync

if TYPE_CHECKING:
    from polylogue.api import ArchiveStats, Polylogue
    from polylogue.archive.message.models import Message
    from polylogue.archive.message.roles import MessageRoleFilter
    from polylogue.archive.semantic.content_projection import ContentProjectionSpec
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.archive.session.neighbor_candidates import SessionNeighborCandidate
    from polylogue.readiness import ReadinessReport
    from polylogue.storage.search import SearchResult
    from polylogue.storage.sqlite.queries.message_query_reads import MessageTypeName


class SyncSessionQueriesMixin:
    """Session and archive query helpers for ``SyncPolylogue``."""

    _facade: Polylogue

    def get_session(self, session_id: str) -> Session | None:
        return run_coroutine_sync(self._facade.get_session(session_id))

    def get_sessions(self, session_ids: list[str]) -> list[Session]:
        return run_coroutine_sync(self._facade.get_sessions(session_ids))

    def get_messages_paginated(
        self,
        session_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        limit: int = 50,
        offset: int = 0,
        content_projection: ContentProjectionSpec | None = None,
    ) -> tuple[list[Message], int]:
        return run_coroutine_sync(
            self._facade.get_messages_paginated(
                session_id,
                message_role=message_role,
                message_type=message_type,
                limit=limit,
                offset=offset,
                content_projection=content_projection,
            )
        )

    def bulk_get_messages(
        self,
        session_ids: Sequence[str],
        *,
        since: str | None = None,
        until: str | None = None,
        message_role: MessageRoleFilter = (),
        content_projection: ContentProjectionSpec | None = None,
    ) -> dict[str, list[Message]]:
        return run_coroutine_sync(
            self._facade.bulk_get_messages(
                session_ids,
                since=since,
                until=until,
                message_role=message_role,
                content_projection=content_projection,
            )
        )

    def list_sessions(
        self,
        origin: str | None = None,
        limit: int | None = None,
    ) -> list[Session]:
        return run_coroutine_sync(self._facade.list_sessions(origin=origin, limit=limit))

    def query_sessions(
        self,
        *,
        origin: str | None = None,
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
            self._facade.query_sessions(
                origin=origin,
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

    def count_sessions(
        self,
        *,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
        **kwargs: object,
    ) -> int:
        return run_coroutine_sync(
            self._facade.count_sessions(
                origin=origin,
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
        origin: str | None = None,
        limit: int | None = None,
    ) -> list[SessionSummary]:
        filt = self._facade.filter()
        if origin:
            filt = filt.origin(origin)
        if since:
            filt = filt.since(since)
        if until:
            filt = filt.until(until)
        if limit:
            filt = filt.limit(limit)
        return run_coroutine_sync(filt.list_summaries())

    def get_session_summary(self, session_id: str) -> SessionSummary | None:
        return run_coroutine_sync(self._facade.get_session_summary(session_id))

    def get_session_stats(self, session_id: str) -> dict[str, int]:
        return run_coroutine_sync(self._facade.get_session_stats(session_id))

    def get_session_tree(self, session_id: str) -> list[Session]:
        return run_coroutine_sync(self._facade.get_session_tree(session_id))

    def list_tags(self, *, origin: str | None = None) -> dict[str, int]:
        return run_coroutine_sync(self._facade.list_tags(origin=origin))

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
        session_id: str | None = None,
        query: str | None = None,
        origin: str | None = None,
        limit: int = 10,
        window_hours: int = 24,
    ) -> list[SessionNeighborCandidate]:
        return run_coroutine_sync(
            self._facade.neighbor_candidates(
                session_id=session_id,
                query=query,
                origin=origin,
                limit=limit,
                window_hours=window_hours,
            )
        )


__all__ = ["SyncSessionQueriesMixin"]
