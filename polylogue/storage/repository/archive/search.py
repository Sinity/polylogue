"""Search-oriented archive reads for the repository."""

from __future__ import annotations

import builtins
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.conversation.models import Conversation, ConversationSummary
    from polylogue.lib.search_hits import ConversationSearchHit
    from polylogue.storage.backends.query_store import SQLiteQueryStore
    from polylogue.storage.runtime import ConversationRecord
    from polylogue.storage.search.models import ConversationSearchEvidenceHit, ConversationSearchResult


def _rerank_evidence_hits(
    hits: builtins.list[ConversationSearchEvidenceHit],
) -> builtins.list[ConversationSearchEvidenceHit]:
    return [replace(hit, rank=rank) for rank, hit in enumerate(hits, start=1)]


def _merge_evidence_hits(
    *,
    attachment_hits: builtins.list[ConversationSearchEvidenceHit],
    message_hits: builtins.list[ConversationSearchEvidenceHit],
    limit: int,
) -> builtins.list[ConversationSearchEvidenceHit]:
    if limit <= 0:
        return []
    merged: builtins.list[ConversationSearchEvidenceHit] = []
    seen: set[str] = set()
    for hit in (*attachment_hits, *message_hits):
        if hit.conversation_id in seen:
            continue
        seen.add(hit.conversation_id)
        merged.append(hit)
        if len(merged) >= limit:
            break
    return _rerank_evidence_hits(merged)


class RepositoryArchiveSearchMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

        async def _hydrate_conversations(
            self,
            conversation_records: builtins.list[ConversationRecord],
            *,
            ordered_ids: builtins.list[str] | None = None,
        ) -> builtins.list[Conversation]: ...

    async def search_summaries(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ) -> builtins.list[ConversationSummary]:
        from polylogue.storage.hydrators import conversation_summary_from_record

        hits, records = await self._search_records(query, limit=limit, providers=providers)
        if not hits.hits:
            return []
        return [conversation_summary_from_record(record) for record in records]

    async def search_summary_hits(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
        since: str | None = None,
    ) -> builtins.list[ConversationSearchHit]:
        from polylogue.errors import DatabaseError
        from polylogue.lib.search_hits import conversation_search_hit_from_summary
        from polylogue.storage.hydrators import conversation_summary_from_record

        attachment_hits = await self.queries.search_attachment_identity_evidence_hits(
            query,
            limit=limit,
            providers=providers,
            since=since,
        )
        try:
            message_hits = await self.queries.search_conversation_evidence_hits(
                query,
                limit=limit,
                providers=providers,
                since=since,
            )
        except DatabaseError:
            if not attachment_hits:
                raise
            message_hits = []

        evidence_hits = _merge_evidence_hits(
            attachment_hits=attachment_hits,
            message_hits=message_hits,
            limit=limit,
        )
        if not evidence_hits:
            return []

        records = await self.queries.get_conversations_batch([hit.conversation_id for hit in evidence_hits])
        summaries_by_id = {str(record.conversation_id): conversation_summary_from_record(record) for record in records}
        return [
            conversation_search_hit_from_summary(
                summaries_by_id[hit.conversation_id],
                rank=hit.rank,
                retrieval_lane=hit.retrieval_lane,
                match_surface=hit.match_surface,
                message_id=hit.message_id,
                snippet=hit.snippet,
                score=hit.score,
            )
            for hit in evidence_hits
            if hit.conversation_id in summaries_by_id
        ]

    async def search(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ) -> builtins.list[Conversation]:
        hits, records = await self._search_records(query, limit=limit, providers=providers)
        return await self._hydrate_conversations(records, ordered_ids=hits.conversation_ids())

    async def search_actions(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ) -> builtins.list[Conversation]:
        hits, records = await self._search_action_records(query, limit=limit, providers=providers)
        return await self._hydrate_conversations(records, ordered_ids=hits.conversation_ids())

    async def _search_records(
        self,
        query: str,
        *,
        limit: int,
        providers: builtins.list[str] | None,
    ) -> tuple[ConversationSearchResult, builtins.list[ConversationRecord]]:
        hits = await self.queries.search_conversation_hits(query, limit=limit, providers=providers)
        if not hits.hits:
            return hits, []
        records = await self.queries.get_conversations_batch(hits.conversation_ids())
        return hits, records

    async def _search_action_records(
        self,
        query: str,
        *,
        limit: int,
        providers: builtins.list[str] | None,
    ) -> tuple[ConversationSearchResult, builtins.list[ConversationRecord]]:
        hits = await self.queries.search_action_conversation_hits(query, limit=limit, providers=providers)
        if not hits.hits:
            return hits, []
        records = await self.queries.get_conversations_batch(hits.conversation_ids())
        return hits, records


__all__ = ["RepositoryArchiveSearchMixin"]
