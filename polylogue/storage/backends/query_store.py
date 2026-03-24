"""Low-level read/query surface for the SQLite archive backend."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager

import aiosqlite

from polylogue.storage.backends.queries import (
    action_events as action_events_q,
)
from polylogue.storage.backends.queries import (
    attachments as attachments_q,
)
from polylogue.storage.backends.queries import (
    conversations as conversations_q,
)
from polylogue.storage.backends.queries import (
    maintenance_runs as maintenance_runs_q,
)
from polylogue.storage.backends.queries import (
    messages as messages_q,
)
from polylogue.storage.backends.queries import (
    publications as publications_q,
)
from polylogue.storage.backends.queries import (
    raw as raw_queries,
)
from polylogue.storage.backends.queries import (
    runs as runs_q,
)
from polylogue.storage.backends.queries import (
    session_products as session_products_q,
)
from polylogue.storage.backends.queries import (
    stats as stats_q,
)
from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.storage.store import (
    ActionEventRecord,
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    DaySessionSummaryRecord,
    MaintenanceRunRecord,
    MessageRecord,
    PublicationRecord,
    RunRecord,
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionTagRollupRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
)


class SQLiteQueryStore:
    """Canonical low-level read/query API for SQLite archive state."""

    def __init__(
        self,
        *,
        connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]],
    ) -> None:
        self._connection_factory = connection_factory

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Retrieve a conversation record by ID."""
        async with self._connection_factory() as conn:
            return await conversations_q.get_conversation(conn, conversation_id)

    async def get_conversations_batch(self, ids: list[str]) -> list[ConversationRecord]:
        """Retrieve multiple conversation records in input order."""
        async with self._connection_factory() as conn:
            return await conversations_q.get_conversations_batch(conn, ids)

    async def list_conversations(
        self,
        request: ConversationRecordQuery,
    ) -> list[ConversationRecord]:
        """List conversation records with filtering and pagination."""
        async with self._connection_factory() as conn:
            return await conversations_q.list_conversations(conn, **request.to_list_kwargs())

    async def count_conversations(
        self,
        request: ConversationRecordQuery,
    ) -> int:
        """Count conversation records matching the given filters."""
        async with self._connection_factory() as conn:
            return await conversations_q.count_conversations(conn, **request.to_count_kwargs())

    async def conversation_exists_by_hash(self, content_hash: str) -> bool:
        """Check whether a conversation with the given content hash exists."""
        async with self._connection_factory() as conn:
            return await conversations_q.conversation_exists_by_hash(conn, content_hash)

    async def resolve_id(self, id_prefix: str) -> str | None:
        """Resolve a partial conversation ID to a full ID."""
        async with self._connection_factory() as conn:
            return await conversations_q.resolve_id(conn, id_prefix)

    async def search_conversations(
        self, query: str, limit: int = 100, providers: list[str] | None = None
    ) -> list[str]:
        """Return ranked conversation IDs for the given search query."""
        async with self._connection_factory() as conn:
            return await conversations_q.search_conversations(conn, query, limit, providers)

    async def search_action_conversations(
        self, query: str, limit: int = 100, providers: list[str] | None = None
    ) -> list[str]:
        """Return ranked conversation IDs for persisted action-aware search."""
        async with self._connection_factory() as conn:
            return await conversations_q.search_action_conversations(conn, query, limit, providers)

    async def get_action_events(self, conversation_id: str) -> list[ActionEventRecord]:
        """Get durable action-event rows for one conversation."""
        async with self._connection_factory() as conn:
            return await action_events_q.get_action_events(conn, conversation_id)

    async def get_action_events_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[ActionEventRecord]]:
        """Get durable action-event rows for multiple conversations."""
        async with self._connection_factory() as conn:
            return await action_events_q.get_action_events_batch(conn, conversation_ids)

    async def get_action_event_read_model_status(self) -> dict[str, int | bool]:
        """Return readiness metadata for the durable action-event read model."""
        from polylogue.storage.action_event_lifecycle import action_event_read_model_status_async

        async with self._connection_factory() as conn:
            return await action_event_read_model_status_async(conn)

    async def get_session_product_status(self) -> dict[str, int | bool]:
        """Return readiness metadata for durable session-product read models."""
        from polylogue.storage.session_product_lifecycle import session_product_status_async

        async with self._connection_factory() as conn:
            return await session_product_status_async(conn)

    async def get_session_profile(self, conversation_id: str) -> SessionProfileRecord | None:
        async with self._connection_factory() as conn:
            return await session_products_q.get_session_profile(conn, conversation_id)

    async def get_session_profiles_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, SessionProfileRecord]:
        async with self._connection_factory() as conn:
            return await session_products_q.get_session_profiles_batch(conn, conversation_ids)

    async def list_session_profiles(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        first_message_since: str | None = None,
        first_message_until: str | None = None,
        session_date_since: str | None = None,
        session_date_until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[SessionProfileRecord]:
        async with self._connection_factory() as conn:
            return await session_products_q.list_session_profiles(
                conn,
                provider=provider,
                since=since,
                until=until,
                first_message_since=first_message_since,
                first_message_until=first_message_until,
                session_date_since=session_date_since,
                session_date_until=session_date_until,
                limit=limit,
                offset=offset,
                query=query,
            )

    async def get_session_work_events(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventRecord]:
        async with self._connection_factory() as conn:
            return await session_products_q.get_work_events(conn, conversation_id)

    async def get_session_phases(
        self,
        conversation_id: str,
    ) -> list[SessionPhaseRecord]:
        async with self._connection_factory() as conn:
            return await session_products_q.get_session_phases(conn, conversation_id)

    async def list_session_work_events(
        self,
        *,
        conversation_id: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[SessionWorkEventRecord]:
        async with self._connection_factory() as conn:
            return await session_products_q.list_work_events(
                conn,
                conversation_id=conversation_id,
                provider=provider,
                since=since,
                until=until,
                kind=kind,
                limit=limit,
                offset=offset,
                query=query,
            )

    async def list_session_phases(
        self,
        *,
        conversation_id: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionPhaseRecord]:
        async with self._connection_factory() as conn:
            return await session_products_q.list_session_phases(
                conn,
                conversation_id=conversation_id,
                provider=provider,
                since=since,
                until=until,
                kind=kind,
                limit=limit,
                offset=offset,
            )

    async def get_work_thread(self, thread_id: str) -> WorkThreadRecord | None:
        async with self._connection_factory() as conn:
            return await session_products_q.get_work_thread(conn, thread_id)

    async def list_work_threads(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[WorkThreadRecord]:
        async with self._connection_factory() as conn:
            return await session_products_q.list_work_threads(
                conn,
                since=since,
                until=until,
                limit=limit,
                offset=offset,
                query=query,
            )

    async def list_session_tag_rollup_rows(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
    ) -> list[SessionTagRollupRecord]:
        async with self._connection_factory() as conn:
            return await session_products_q.list_session_tag_rollup_rows(
                conn,
                provider=provider,
                since=since,
                until=until,
                query=query,
            )

    async def list_day_session_summaries(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[DaySessionSummaryRecord]:
        async with self._connection_factory() as conn:
            return await session_products_q.list_day_session_summaries(
                conn,
                provider=provider,
                since=since,
                until=until,
            )

    async def list_maintenance_runs(self, *, limit: int = 20) -> list[MaintenanceRunRecord]:
        async with self._connection_factory() as conn:
            return await maintenance_runs_q.list_maintenance_runs(conn, limit=limit)

    async def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        """Get message records for a conversation with content blocks attached."""
        async with self._connection_factory() as conn:
            messages = await messages_q.get_messages(conn, conversation_id)
        if not messages:
            return []
        blocks_by_message = await self.get_content_blocks([message.message_id for message in messages])
        return [
            message.model_copy(
                update={"content_blocks": blocks_by_message.get(message.message_id, [])}
            )
            for message in messages
        ]

    async def get_messages_batch(self, conversation_ids: list[str]) -> dict[str, list[MessageRecord]]:
        """Get message records for multiple conversations with content blocks attached."""
        if not conversation_ids:
            return {}
        async with self._connection_factory() as conn:
            result, all_messages = await messages_q.get_messages_batch(conn, conversation_ids)
        if not all_messages:
            return result
        blocks_by_message = await self.get_content_blocks([message.message_id for message in all_messages])
        return {
            conversation_id: [
                message.model_copy(
                    update={"content_blocks": blocks_by_message.get(message.message_id, [])}
                )
                for message in records
            ]
            for conversation_id, records in result.items()
        }

    async def get_content_blocks(
        self, message_ids: list[str]
    ) -> dict[str, list[ContentBlockRecord]]:
        """Get content block records keyed by message ID."""
        async with self._connection_factory() as conn:
            return await attachments_q.get_content_blocks(conn, message_ids)

    async def get_attachments(self, conversation_id: str) -> list[AttachmentRecord]:
        """Get attachment records for one conversation."""
        async with self._connection_factory() as conn:
            return await attachments_q.get_attachments(conn, conversation_id)

    async def get_attachments_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[AttachmentRecord]]:
        """Get attachment records for multiple conversations."""
        async with self._connection_factory() as conn:
            return await attachments_q.get_attachments_batch(conn, conversation_ids)

    async def iter_messages(
        self,
        conversation_id: str,
        *,
        dialogue_only: bool = False,
        limit: int | None = None,
    ) -> AsyncIterator[MessageRecord]:
        """Stream message records for a conversation."""
        async with self._connection_factory() as conn:
            async for record in messages_q.iter_messages(
                conn,
                conversation_id,
                dialogue_only=dialogue_only,
                limit=limit,
            ):
                yield record

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, int]:
        """Get lightweight message statistics for one conversation."""
        async with self._connection_factory() as conn:
            return await messages_q.get_conversation_stats(conn, conversation_id)

    async def get_message_counts_batch(self, conversation_ids: list[str]) -> dict[str, int]:
        """Get total message counts for multiple conversations."""
        async with self._connection_factory() as conn:
            return await messages_q.get_message_counts_batch(conn, conversation_ids)

    async def aggregate_message_stats(
        self, conversation_ids: list[str] | None = None
    ) -> dict[str, int]:
        """Compute archive-wide or scoped aggregate message statistics."""
        async with self._connection_factory() as conn:
            return await stats_q.aggregate_message_stats(conn, conversation_ids)

    async def get_stats_by(self, group_by: str = "provider") -> dict[str, int]:
        """Get conversation counts grouped by provider, month, or year."""
        async with self._connection_factory() as conn:
            return await stats_q.get_stats_by(conn, group_by)

    async def get_provider_conversation_counts(self) -> list[dict[str, object]]:
        """Return archive conversation counts grouped by provider."""
        async with self._connection_factory() as conn:
            return await stats_q.get_provider_conversation_counts(conn)

    async def get_provider_metrics_rows(self) -> list[dict[str, object]]:
        """Return raw per-provider metrics rows."""
        async with self._connection_factory() as conn:
            return await stats_q.get_provider_metrics_rows(conn)

    async def get_last_sync_timestamp(self) -> str | None:
        """Return the timestamp of the latest ingestion run, if any."""
        async with self._connection_factory() as conn:
            return await conversations_q.get_last_sync_timestamp(conn)

    def conversation_id_query(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Build the canonical scoped conversation-ID query."""
        return conversations_q.conversation_id_query(source_names=source_names)

    async def count_conversation_ids(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> int:
        """Count conversation IDs within the optional source scope."""
        async with self._connection_factory() as conn:
            return await conversations_q.count_conversation_ids(conn, source_names=source_names)

    async def iter_conversation_ids(
        self,
        *,
        source_names: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        """Iterate scoped conversation IDs in bounded batches."""
        async with self._connection_factory() as conn:
            async for conversation_id in conversations_q.iter_conversation_ids(
                conn, source_names=source_names, page_size=page_size
            ):
                yield conversation_id

    def raw_id_query(
        self,
        *,
        source_names: list[str] | None = None,
        provider_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Build the canonical scoped raw-ID query."""
        return raw_queries.raw_id_query(
            source_names=source_names,
            provider_name=provider_name,
            require_unparsed=require_unparsed,
            require_unvalidated=require_unvalidated,
            validation_statuses=validation_statuses,
        )

    async def iter_raw_ids(
        self,
        *,
        source_names: list[str] | None = None,
        provider_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        """Iterate raw conversation IDs for a processing slice."""
        async with self._connection_factory() as conn:
            async for raw_id in raw_queries.iter_raw_ids(
                conn,
                source_names=source_names,
                provider_name=provider_name,
                require_unparsed=require_unparsed,
                require_unvalidated=require_unvalidated,
                validation_statuses=validation_statuses,
                page_size=page_size,
            ):
                yield raw_id

    async def get_known_source_mtimes(self) -> dict[str, str]:
        """Return persisted source mtimes keyed by source path."""
        async with self._connection_factory() as conn:
            return await raw_queries.get_known_source_mtimes(conn)

    async def get_latest_run(self) -> RunRecord | None:
        """Fetch the most recent pipeline run record."""
        async with self._connection_factory() as conn:
            return await runs_q.get_latest_run(conn)

    async def get_latest_publication(
        self,
        publication_kind: str,
    ) -> PublicationRecord | None:
        """Fetch the most recent publication record for one publication kind."""
        async with self._connection_factory() as conn:
            return await publications_q.get_latest_publication(conn, publication_kind)


__all__ = ["SQLiteQueryStore"]
