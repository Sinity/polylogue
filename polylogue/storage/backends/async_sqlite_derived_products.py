"""Derived session-product writes for the async SQLite backend."""

from __future__ import annotations

from polylogue.storage.backends.queries import (
    session_product_profile_writes as session_product_profiles_q,
)
from polylogue.storage.backends.queries import (
    session_product_thread_queries as session_product_threads_q,
)
from polylogue.storage.backends.queries import (
    session_product_timeline_writes as session_product_timelines_q,
)
from polylogue.storage.store import (
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
)


class SQLiteDerivedProductsMixin:
    """Derived durable session-product methods for ``SQLiteBackend``."""

    async def replace_session_profile(
        self,
        record: SessionProfileRecord,
    ) -> None:
        """Replace one durable session-profile row."""
        async with self._get_connection() as conn:
            await session_product_profiles_q.replace_session_profile(
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
            await session_product_timelines_q.replace_session_work_events(
                conn,
                conversation_id,
                records,
                self._transaction_depth,
            )

    async def replace_session_phases(
        self,
        conversation_id: str,
        records: list[SessionPhaseRecord],
    ) -> None:
        """Replace durable phase rows for one conversation."""
        async with self._get_connection() as conn:
            await session_product_timelines_q.replace_session_phases(
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
            await session_product_threads_q.replace_work_thread(
                conn,
                thread_id,
                record,
                self._transaction_depth,
            )


__all__ = ["SQLiteDerivedProductsMixin"]
