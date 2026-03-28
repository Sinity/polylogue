"""Search-oriented archive reads for the repository."""

from __future__ import annotations

import builtins


class RepositoryArchiveSearchMixin:
    async def search_summaries(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ):
        from polylogue.storage.hydrators import conversation_summary_from_record

        ids, records = await self._search_records(query, limit=limit, providers=providers)
        if not ids:
            return []
        return [conversation_summary_from_record(record) for record in records]

    async def search(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ):
        ids, records = await self._search_records(query, limit=limit, providers=providers)
        return await self._hydrate_conversations(records, ordered_ids=ids)

    async def search_actions(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ):
        ids, records = await self._search_action_records(query, limit=limit, providers=providers)
        return await self._hydrate_conversations(records, ordered_ids=ids)

    async def _search_records(
        self,
        query: str,
        *,
        limit: int,
        providers: builtins.list[str] | None,
    ):
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
    ):
        ids = await self.queries.search_action_conversations(query, limit=limit, providers=providers)
        if not ids:
            return [], []
        records = await self.queries.get_conversations_batch(ids)
        by_id = {record.conversation_id: record for record in records}
        return ids, [by_id[conversation_id] for conversation_id in ids if conversation_id in by_id]


__all__ = ["RepositoryArchiveSearchMixin"]
