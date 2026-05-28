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
        async with self._backend.connection() as conn:
            # repos — grouped by git_repository_url on conversations
            if conversation_ids is not None:
                placeholders = ",".join("?" for _ in conversation_ids)
                repo_rows = await (
                    await conn.execute(
                        f"""
                        SELECT git_repository_url, count(*) AS n
                        FROM conversations
                        WHERE git_repository_url IS NOT NULL
                          AND conversation_id IN ({placeholders})
                        GROUP BY git_repository_url
                        """,
                        conversation_ids,
                    )
                ).fetchall()
            else:
                repo_rows = await (
                    await conn.execute(
                        """
                        SELECT git_repository_url, count(*) AS n
                        FROM conversations
                        WHERE git_repository_url IS NOT NULL
                        GROUP BY git_repository_url
                        """
                    )
                ).fetchall()
            result["repos"] = {row[0]: row[1] for row in repo_rows if row[0]}

            # message_types
            if conversation_ids is not None:
                placeholders = ",".join("?" for _ in conversation_ids)
                mt_rows = await (
                    await conn.execute(
                        f"""
                        SELECT message_type, count(*) AS n
                        FROM messages
                        WHERE conversation_id IN ({placeholders})
                        GROUP BY message_type
                        """,
                        conversation_ids,
                    )
                ).fetchall()
            else:
                mt_rows = await (
                    await conn.execute(
                        """
                        SELECT message_type, count(*) AS n
                        FROM messages
                        GROUP BY message_type
                        """
                    )
                ).fetchall()
            result["message_types"] = {row[0]: row[1] for row in mt_rows if row[0]}

            # action_types
            if conversation_ids is not None:
                placeholders = ",".join("?" for _ in conversation_ids)
                at_rows = await (
                    await conn.execute(
                        f"""
                        SELECT action_kind, count(*) AS n
                        FROM action_events
                        WHERE conversation_id IN ({placeholders})
                        GROUP BY action_kind
                        """,
                        conversation_ids,
                    )
                ).fetchall()
            else:
                at_rows = await (
                    await conn.execute(
                        """
                        SELECT action_kind, count(*) AS n
                        FROM action_events
                        GROUP BY action_kind
                        """
                    )
                ).fetchall()
            result["action_types"] = {row[0]: row[1] for row in at_rows if row[0]}

            # has_flags — booleans on messages
            if conversation_ids is not None:
                placeholders = ",".join("?" for _ in conversation_ids)
                flag_rows = await (
                    await conn.execute(
                        f"""
                        SELECT
                            coalesce(sum(has_tool_use), 0) AS tool_use,
                            coalesce(sum(has_thinking), 0) AS thinking,
                            coalesce(sum(has_paste), 0)   AS paste
                        FROM messages
                        WHERE conversation_id IN ({placeholders})
                        """,
                        conversation_ids,
                    )
                ).fetchall()
            else:
                flag_rows = await (
                    await conn.execute(
                        """
                        SELECT
                            coalesce(sum(has_tool_use), 0) AS tool_use,
                            coalesce(sum(has_thinking), 0) AS thinking,
                            coalesce(sum(has_paste), 0)   AS paste
                        FROM messages
                        """
                    )
                ).fetchall()
            row = flag_rows[0] if flag_rows else (0, 0, 0)
            result["has_flags"] = {
                "has_tool_use": row[0],
                "has_thinking": row[1],
                "has_paste": row[2],
            }

        return result


__all__ = ["RepositoryArchiveConversationMixin"]
