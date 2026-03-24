"""Derived-model write/query methods for the async SQLite backend."""

from __future__ import annotations

from polylogue.storage.backends.queries import action_events as action_events_q
from polylogue.storage.backends.queries import maintenance_runs as maintenance_runs_q
from polylogue.storage.backends.queries import session_products as session_products_q
from polylogue.storage.backends.queries import stats as stats_q
from polylogue.storage.store import (
    ActionEventRecord,
    MaintenanceRunRecord,
    MessageRecord,
    SessionProfileRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
)


class SQLiteDerivedMixin:
    """Derived action/product/maintenance methods for ``SQLiteBackend``."""

    async def replace_action_events(
        self,
        conversation_id: str,
        records: list[ActionEventRecord],
    ) -> None:
        """Replace durable action-event rows for one conversation."""
        async with self._get_connection() as conn:
            await action_events_q.replace_action_events(
                conn,
                conversation_id,
                records,
                self._transaction_depth,
            )

    async def get_action_events(self, conversation_id: str) -> list[ActionEventRecord]:
        """Get durable action-event rows for one conversation."""
        return await self.queries.get_action_events(conversation_id)

    async def get_action_events_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[ActionEventRecord]]:
        """Get durable action-event rows for multiple conversations."""
        return await self.queries.get_action_events_batch(conversation_ids)

    async def replace_session_profile(
        self,
        record: SessionProfileRecord,
    ) -> None:
        """Replace one durable session-profile row."""
        async with self._get_connection() as conn:
            await session_products_q.replace_session_profile(
                conn,
                record,
                self._transaction_depth,
            )

    async def replace_session_work_events(
        self,
        conversation_id: str,
        records: list[SessionWorkEventRecord],
    ) -> None:
        """Replace durable work-event rows for one conversation."""
        async with self._get_connection() as conn:
            await session_products_q.replace_session_work_events(
                conn,
                conversation_id,
                records,
                self._transaction_depth,
            )

    async def replace_work_thread(
        self,
        thread_id: str,
        record: WorkThreadRecord | None,
    ) -> None:
        """Replace one durable work-thread row."""
        async with self._get_connection() as conn:
            await session_products_q.replace_work_thread(
                conn,
                thread_id,
                record,
                self._transaction_depth,
            )

    async def record_maintenance_run(
        self,
        record: MaintenanceRunRecord,
    ) -> None:
        """Persist one maintenance lineage record."""
        async with self._get_connection() as conn:
            await maintenance_runs_q.record_maintenance_run(
                conn,
                record,
                self._transaction_depth,
            )

    async def upsert_conversation_stats(
        self,
        conversation_id: str,
        provider_name: str,
        messages: list[MessageRecord],
    ) -> None:
        """Upsert precomputed per-conversation aggregate stats."""
        async with self._get_connection() as conn:
            await stats_q.upsert_conversation_stats(
                conn, conversation_id, provider_name, messages, self._transaction_depth
            )


__all__ = ["SQLiteDerivedMixin"]
