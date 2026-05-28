"""Conversation and message hydration reads for the repository."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from polylogue.archive.conversation.models import Conversation, ConversationSummary
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import MessageRoleFilter
from polylogue.storage.archive_views import ConversationRenderProjection
from polylogue.storage.hydrators import (
    conversation_from_records,
    conversation_summary_from_record,
    message_from_record,
)
from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.runtime import AttachmentRecord, ConversationRecord, MessageRecord

if TYPE_CHECKING:
    from polylogue.storage.sqlite.queries.messages import MessageTypeName
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore
    from polylogue.types import ConversationId


class RepositoryArchiveConversationMixin:
    if TYPE_CHECKING:
        _backend: RepositoryBackendProtocol
        queries: SQLiteQueryStore

    async def _fetch_tags_by_conversation(self, conversation_ids: list[str]) -> dict[str, tuple[str, ...]]:
        """#1240: batch-fetch M2M tags for hydration of Conversation/ConversationSummary."""
        if not conversation_ids:
            return {}
        result: dict[str, list[str]] = {cid: [] for cid in conversation_ids}
        async with self._backend.connection() as conn:
            placeholders = ",".join("?" for _ in conversation_ids)
            cursor = await conn.execute(
                f"""
                SELECT ct.conversation_id AS cid, t.name AS name
                FROM conversation_tags ct
                JOIN tags t ON t.id = ct.tag_id
                WHERE ct.conversation_id IN ({placeholders})
                ORDER BY t.name
                """,
                conversation_ids,
            )
            rows = await cursor.fetchall()
            for row in rows:
                cid = row["cid"]
                name = row["name"]
                if cid in result:
                    result[cid].append(name)
        return {cid: tuple(names) for cid, names in result.items()}

    async def resolve_id(self, id_prefix: str, *, strict: bool = False) -> ConversationId | None:
        resolved = await self.queries.resolve_id(id_prefix, strict=strict)
        from polylogue.types import ConversationId

        return ConversationId(resolved) if resolved else None

    async def get(self, conversation_id: str) -> Conversation | None:
        conv_record = await self.queries.get_conversation(conversation_id)
        if not conv_record:
            return None

        msg_records, att_records, provider_event_records = await asyncio.gather(
            self.queries.get_messages(conversation_id),
            self.queries.get_attachments(conversation_id),
            self.queries.get_provider_events(conversation_id),
        )
        tags_by_id = await self._fetch_tags_by_conversation([conversation_id])
        return conversation_from_records(
            conv_record,
            msg_records,
            att_records,
            provider_event_records,
            tags=tags_by_id.get(conversation_id, ()),
        )

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

    async def get_messages_paginated(
        self,
        conversation_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Message], int]:
        conv_record = await self.queries.get_conversation(conversation_id)
        source_name = conv_record.source_name if conv_record else None
        records, total = await self.queries.get_messages_paginated(
            conversation_id,
            message_role=message_role,
            message_type=message_type,
            limit=limit,
            offset=offset,
        )
        messages = [message_from_record(r, attachments=[], provider=source_name) for r in records]
        return messages, total

    async def get_conversations_batch(self, ids: list[str]) -> list[ConversationRecord]:
        return await self.queries.get_conversations_batch(ids)

    async def get_messages_batch(
        self,
        conversation_ids: list[str],
        *,
        sort_key_since: float | None = None,
        sort_key_until: float | None = None,
        message_role: MessageRoleFilter = (),
    ) -> dict[str, list[MessageRecord]]:
        return await self.queries.get_messages_batch(
            conversation_ids,
            sort_key_since=sort_key_since,
            sort_key_until=sort_key_until,
            message_role=message_role,
        )

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
            msgs_by_id, atts_by_id, provider_events_by_id = await asyncio.gather(
                self.queries.get_messages_batch(present_ids),
                self.queries.get_attachments_batch(present_ids),
                self.queries.get_provider_events_batch(present_ids),
            )
        tags_by_id = await self._fetch_tags_by_conversation(present_ids)
        return [
            conversation_from_records(
                by_id[conversation_id],
                msgs_by_id.get(conversation_id, []),
                atts_by_id.get(conversation_id, []),
                provider_events_by_id.get(conversation_id, []),
                tags=tags_by_id.get(conversation_id, ()),
            )
            for conversation_id in present_ids
        ]

    async def get_summary(self, conversation_id: str) -> ConversationSummary | None:
        conv_record = await self.queries.get_conversation(conversation_id)
        if not conv_record:
            return None
        tags_by_id = await self._fetch_tags_by_conversation([conversation_id])
        # #1630: hydrate message_count from conversation_stats so the daemon
        # HTTP /api/conversations/{id} endpoint and any other ``get_summary``
        # caller see a real total instead of None. Same shape as
        # list_summaries_by_query below.
        counts_by_id = await self.queries.get_message_counts_batch([conversation_id])
        return conversation_summary_from_record(
            conv_record,
            tags=tags_by_id.get(conversation_id, ()),
            message_count=counts_by_id.get(conversation_id),
        )

    async def list_summaries_by_query(
        self,
        query: ConversationRecordQuery,
    ) -> list[ConversationSummary]:
        conv_records = await self.queries.list_conversation_summaries(query)
        ids = [str(record.conversation_id) for record in conv_records]
        tags_by_id = await self._fetch_tags_by_conversation(ids)
        # #1623: hydrate message_count from conversation_stats so callers
        # like compute_facets see a real total instead of summing zeros.
        counts_by_id = await self.queries.get_message_counts_batch(ids) if ids else {}
        return [
            conversation_summary_from_record(
                record,
                tags=tags_by_id.get(str(record.conversation_id), ()),
                message_count=counts_by_id.get(str(record.conversation_id)),
            )
            for record in conv_records
        ]

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
        source_name = conv_record.source_name if conv_record else None
        async for record in self.queries.iter_messages(
            conversation_id,
            dialogue_only=dialogue_only,
            message_roles=message_roles,
            limit=limit,
        ):
            yield message_from_record(record, attachments=[], provider=source_name)

    async def aggregate_facet_families(
        self,
        *,
        conversation_ids: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Run per-family SQL aggregators for facets that can't be computed
        from conversation summaries alone.

        Returns a dict keyed by family name (``repos``, ``message_types``,
        ``action_types``, ``has_flags``). When ``conversation_ids`` is
        provided, results are scoped to those conversations.

        #1672 (phase 2 of #1623).
        """
        result: dict[str, dict[str, int]] = {
            "repos": {},
            "message_types": {},
            "action_types": {},
            "has_flags": {},
        }
        # Empty list produces SQL ``IN ()`` which is a syntax error.
        if conversation_ids is not None and not conversation_ids:
            return result
        # SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 999. When the
        # scoped path passes >900 IDs we chunk and merge so the query
        # never exceeds the engine limit. The global path (None) has no
        # IN clause and does not need chunking.
        _max_in_vars = 900

        from typing import Any

        async def _scoped_rows(
            conn: Any,
            scoped_sql: str,
            global_sql: str,
            params: list[str] | None,
        ) -> list[Any]:
            """Run a possibly-chunked scoped query or a single global query."""
            if params is None:
                return list(await (await conn.execute(global_sql)).fetchall())
            rows: list[object] = []
            for i in range(0, len(params), _max_in_vars):
                chunk = params[i : i + _max_in_vars]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(await (await conn.execute(scoped_sql.format(placeholders), chunk)).fetchall())
            return rows

        # Accumulate keyed results from possibly-chunked rows.
        def _keyed(rows: list[Any]) -> dict[str, int]:
            return {row[0]: row[1] for row in rows if row[0]}

        async with self._backend.connection() as conn:
            ids = conversation_ids  # None → global, list → scoped

            result["repos"] = _keyed(
                await _scoped_rows(
                    conn,
                    """SELECT git_repository_url, count(*) AS n
                    FROM conversations
                    WHERE git_repository_url IS NOT NULL
                      AND conversation_id IN ({})
                    GROUP BY git_repository_url""",
                    """SELECT git_repository_url, count(*) AS n
                    FROM conversations
                    WHERE git_repository_url IS NOT NULL
                    GROUP BY git_repository_url""",
                    ids,
                )
            )

            result["message_types"] = _keyed(
                await _scoped_rows(
                    conn,
                    """SELECT message_type, count(*) AS n
                    FROM messages
                    WHERE conversation_id IN ({})
                    GROUP BY message_type""",
                    """SELECT message_type, count(*) AS n
                    FROM messages
                    GROUP BY message_type""",
                    ids,
                )
            )

            result["action_types"] = _keyed(
                await _scoped_rows(
                    conn,
                    """SELECT action_kind, count(*) AS n
                    FROM action_events
                    WHERE conversation_id IN ({})
                    GROUP BY action_kind""",
                    """SELECT action_kind, count(*) AS n
                    FROM action_events
                    GROUP BY action_kind""",
                    ids,
                )
            )

            flag_rows = await _scoped_rows(
                conn,
                """SELECT coalesce(sum(has_tool_use),0), coalesce(sum(has_thinking),0),
                coalesce(sum(has_paste),0) FROM messages WHERE conversation_id IN ({})""",
                """SELECT coalesce(sum(has_tool_use),0), coalesce(sum(has_thinking),0),
                coalesce(sum(has_paste),0) FROM messages""",
                ids,
            )
            # Merge chunked flag rows by summing corresponding columns.
            tool_use = sum(r[0] for r in flag_rows)
            thinking = sum(r[1] for r in flag_rows)
            paste = sum(r[2] for r in flag_rows)
            result["has_flags"] = {"has_tool_use": tool_use, "has_thinking": thinking, "has_paste": paste}

        return result


__all__ = ["RepositoryArchiveConversationMixin"]
