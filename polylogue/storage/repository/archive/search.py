"""Search-oriented archive reads for the repository."""

from __future__ import annotations

import builtins
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.archive.query.search_hits import SessionSearchHit
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.storage.runtime import SessionRecord
    from polylogue.storage.search.models import SessionSearchEvidenceRow, SessionSearchResult
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


def _rerank_evidence_hits(
    hits: builtins.list[SessionSearchEvidenceRow],
) -> builtins.list[SessionSearchEvidenceRow]:
    return [replace(hit, rank=rank) for rank, hit in enumerate(hits, start=1)]


def _merge_evidence_hits(
    *,
    attachment_hits: builtins.list[SessionSearchEvidenceRow],
    message_hits: builtins.list[SessionSearchEvidenceRow],
    limit: int,
) -> builtins.list[SessionSearchEvidenceRow]:
    if limit <= 0:
        return []
    merged: builtins.list[SessionSearchEvidenceRow] = []
    seen: set[str] = set()
    for hit in (*attachment_hits, *message_hits):
        if hit.session_id in seen:
            continue
        seen.add(hit.session_id)
        merged.append(hit)
        if len(merged) >= limit:
            break
    return _rerank_evidence_hits(merged)


class RepositoryArchiveSearchMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

        async def _hydrate_sessions(
            self,
            session_records: builtins.list[SessionRecord],
            *,
            ordered_ids: builtins.list[str] | None = None,
        ) -> builtins.list[Session]: ...

    async def search_summaries(
        self,
        query: str,
        limit: int = 20,
        origins: builtins.list[str] | None = None,
    ) -> builtins.list[SessionSummary]:
        from polylogue.storage.hydrators import session_summary_from_record

        hits, records = await self._search_records(query, limit=limit, origins=origins)
        if not hits.hits:
            return []
        # Hydrate message_count from the current sessions aggregate.
        ids = [str(record.session_id) for record in records]
        counts_by_id = await self.queries.get_message_counts_batch(ids)
        return [
            session_summary_from_record(
                record,
                message_count=counts_by_id.get(str(record.session_id)),
            )
            for record in records
        ]

    async def search_summary_hits(
        self,
        query: str,
        limit: int = 20,
        origins: builtins.list[str] | None = None,
        since: str | None = None,
    ) -> builtins.list[SessionSearchHit]:
        from polylogue.archive.query.search_hits import session_search_hit_from_summary
        from polylogue.errors import DatabaseError
        from polylogue.storage.hydrators import session_summary_from_record

        attachment_hits = await self.queries.search_attachment_identity_evidence_hits(
            query,
            limit=limit,
            origins=origins,
            since=since,
        )
        try:
            message_hits = await self.queries.search_session_evidence_hits(
                query,
                limit=limit,
                origins=origins,
                since=since,
            )
        except DatabaseError:
            message_hits = []

        evidence_hits = _merge_evidence_hits(
            attachment_hits=attachment_hits,
            message_hits=message_hits,
            limit=limit,
        )
        if not evidence_hits:
            return []

        records = await self.queries.get_sessions_batch([hit.session_id for hit in evidence_hits])
        # Hydrate message_count from the current sessions aggregate.
        ids = [str(record.session_id) for record in records]
        counts_by_id = await self.queries.get_message_counts_batch(ids) if ids else {}
        summaries_by_id = {
            str(record.session_id): session_summary_from_record(
                record, message_count=counts_by_id.get(str(record.session_id))
            )
            for record in records
        }
        return [
            session_search_hit_from_summary(
                summaries_by_id[hit.session_id],
                rank=hit.rank,
                retrieval_lane=hit.retrieval_lane,
                match_surface=hit.match_surface,
                message_id=hit.message_id,
                snippet=hit.snippet,
                score=hit.score,
                matched_terms=hit.matched_terms,
                score_components=hit.score_components,
                score_kind=hit.score_kind,
                lane_rank=hit.lane_rank,
                lane_contribution=hit.lane_contribution,
                raw_score=hit.raw_score,
            )
            for hit in evidence_hits
            if hit.session_id in summaries_by_id
        ]

    async def search(
        self,
        query: str,
        limit: int = 20,
        origins: builtins.list[str] | None = None,
    ) -> builtins.list[Session]:
        hits, records = await self._search_records(query, limit=limit, origins=origins)
        return await self._hydrate_sessions(records, ordered_ids=hits.session_ids())

    async def search_actions(
        self,
        query: str,
        limit: int = 20,
        origins: builtins.list[str] | None = None,
    ) -> builtins.list[Session]:
        hits, records = await self._search_action_records(query, limit=limit, origins=origins)
        return await self._hydrate_sessions(records, ordered_ids=hits.session_ids())

    async def _search_records(
        self,
        query: str,
        *,
        limit: int,
        origins: builtins.list[str] | None,
    ) -> tuple[SessionSearchResult, builtins.list[SessionRecord]]:
        hits = await self.queries.search_session_hits(query, limit=limit, origins=origins)
        if not hits.hits:
            return hits, []
        records = await self.queries.get_sessions_batch(hits.session_ids())
        return hits, records

    async def _search_action_records(
        self,
        query: str,
        *,
        limit: int,
        origins: builtins.list[str] | None,
    ) -> tuple[SessionSearchResult, builtins.list[SessionRecord]]:
        hits = await self.queries.search_action_session_hits(query, limit=limit, origins=origins)
        if not hits.hits:
            return hits, []
        records = await self.queries.get_sessions_batch(hits.session_ids())
        return hits, records


__all__ = ["RepositoryArchiveSearchMixin"]
