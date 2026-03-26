"""Synchronous bridge for the Polylogue async facade.

Wraps the async ``Polylogue`` facade so that synchronous callers
(like Lynchpin's trajectory pipeline) can consume session profiles,
summaries, and conversations without managing an event loop.

Example::

    from polylogue.sync import SyncPolylogue

    poly = SyncPolylogue()
    summaries = poly.list_summaries(since="2026-01-01")
    conv = poly.get_conversation("abc12345")
    poly.close()
"""

from __future__ import annotations

from collections.abc import Awaitable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from polylogue.archive_products import (
    ArchiveDebtProduct,
    ArchiveDebtProductQuery,
    DaySessionSummaryProduct,
    DaySessionSummaryProductQuery,
    MaintenanceRunProduct,
    MaintenanceRunProductQuery,
    ProviderAnalyticsProduct,
    ProviderAnalyticsProductQuery,
    SessionEnrichmentProduct,
    SessionEnrichmentProductQuery,
    SessionPhaseProduct,
    SessionPhaseProductQuery,
    SessionProfileProduct,
    SessionProfileProductQuery,
    SessionTagRollupProduct,
    SessionTagRollupQuery,
    SessionWorkEventProduct,
    SessionWorkEventProductQuery,
    WeekSessionSummaryProduct,
    WeekSessionSummaryProductQuery,
    WorkThreadProduct,
    WorkThreadProductQuery,
)
from polylogue.sync_bridge import run_coroutine_sync

if TYPE_CHECKING:
    from polylogue.facade import ArchiveStats
    from polylogue.lib.conversation_models import Conversation, ConversationSummary
    from polylogue.storage.search import SearchResult

T = TypeVar("T")


def _run(coro: Awaitable[T]) -> T:
    """Run a Polylogue coroutine from synchronous callers.

    This remains the canonical sync execution seam for module-local sync helpers
    and external consumers that build async filter pipelines before forcing the
    terminal awaitable.
    """

    return run_coroutine_sync(coro)


class SyncPolylogue:
    """Synchronous wrapper around the async ``Polylogue`` facade.

    All methods delegate to ``Polylogue`` through the shared sync bridge.
    """

    def __init__(
        self,
        archive_root: str | Path | None = None,
        db_path: str | Path | None = None,
    ):
        from polylogue.facade import Polylogue

        self._facade = Polylogue(archive_root=archive_root, db_path=db_path)

    def close(self) -> None:
        """Release database connections."""
        _run(self._facade.close())

    def __enter__(self) -> SyncPolylogue:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # --- Queries ---

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Fetch a single conversation by full or prefix ID."""
        return _run(self._facade.get_conversation(conversation_id))

    def get_conversations(self, conversation_ids: list[str]) -> list[Conversation]:
        """Batch fetch conversations."""
        return _run(self._facade.get_conversations(conversation_ids))

    def list_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
    ) -> list[Conversation]:
        """List conversations with optional filtering."""
        return _run(self._facade.list_conversations(provider=provider, limit=limit))

    def list_summaries(
        self,
        *,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        provider: str | None = None,
        limit: int | None = None,
    ) -> list[ConversationSummary]:
        """List lightweight conversation summaries."""
        filt = self._facade.filter()
        if provider:
            filt = filt.provider(provider)
        if since:
            filt = filt.since(since)
        if until:
            filt = filt.until(until)
        if limit:
            filt = filt.limit(limit)
        return _run(filt.list_summaries())

    def search(
        self,
        query: str,
        *,
        limit: int = 100,
        source: str | None = None,
        since: str | None = None,
    ) -> SearchResult:
        """Search conversations."""
        return _run(self._facade.search(query, limit=limit, source=source, since=since))

    def stats(self) -> ArchiveStats:
        """Get archive statistics."""
        return _run(self._facade.stats())

    def get_session_product_status(self) -> dict[str, int | bool]:
        """Get durable session-product readiness counters."""
        return _run(self._facade.get_session_product_status())

    def get_session_profile_product(
        self,
        conversation_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileProduct | None:
        """Get the versioned durable session-profile product for one conversation."""
        return _run(self._facade.get_session_profile_product(conversation_id, tier=tier))

    def list_session_profile_products(
        self,
        query: SessionProfileProductQuery | None = None,
    ) -> list[SessionProfileProduct]:
        """List versioned durable session-profile products."""
        return _run(self._facade.list_session_profile_products(query))

    def get_session_enrichment_product(
        self,
        conversation_id: str,
    ) -> SessionEnrichmentProduct | None:
        """Get the versioned durable session-enrichment product for one conversation."""
        return _run(self._facade.get_session_enrichment_product(conversation_id))

    def list_session_enrichment_products(
        self,
        query: SessionEnrichmentProductQuery | None = None,
    ) -> list[SessionEnrichmentProduct]:
        """List versioned durable session-enrichment products."""
        return _run(self._facade.list_session_enrichment_products(query))

    def list_session_tag_rollup_products(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupProduct]:
        """List versioned durable session-tag rollup products."""
        return _run(self._facade.list_session_tag_rollup_products(query))

    def get_session_work_event_products(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventProduct]:
        """Get versioned durable work-event products for one conversation."""
        return _run(self._facade.get_session_work_event_products(conversation_id))

    def list_session_work_event_products(
        self,
        query: SessionWorkEventProductQuery | None = None,
    ) -> list[SessionWorkEventProduct]:
        """List versioned durable work-event products."""
        return _run(self._facade.list_session_work_event_products(query))

    def get_session_phase_products(
        self,
        conversation_id: str,
    ) -> list[SessionPhaseProduct]:
        """Get versioned durable session-phase products for one conversation."""
        return _run(self._facade.get_session_phase_products(conversation_id))

    def list_session_phase_products(
        self,
        query: SessionPhaseProductQuery | None = None,
    ) -> list[SessionPhaseProduct]:
        """List versioned durable session-phase products."""
        return _run(self._facade.list_session_phase_products(query))

    def get_work_thread_product(self, thread_id: str) -> WorkThreadProduct | None:
        """Get the versioned durable work-thread product for one thread."""
        return _run(self._facade.get_work_thread_product(thread_id))

    def list_work_thread_products(
        self,
        query: WorkThreadProductQuery | None = None,
    ) -> list[WorkThreadProduct]:
        """List versioned durable work-thread products."""
        return _run(self._facade.list_work_thread_products(query))

    def list_day_session_summary_products(
        self,
        query: DaySessionSummaryProductQuery | None = None,
    ) -> list[DaySessionSummaryProduct]:
        """List durable day-level session summary products."""
        return _run(self._facade.list_day_session_summary_products(query))

    def list_week_session_summary_products(
        self,
        query: WeekSessionSummaryProductQuery | None = None,
    ) -> list[WeekSessionSummaryProduct]:
        """List durable week-level session summary products."""
        return _run(self._facade.list_week_session_summary_products(query))

    def list_maintenance_run_products(
        self,
        query: MaintenanceRunProductQuery | None = None,
    ) -> list[MaintenanceRunProduct]:
        """List versioned maintenance-lineage products."""
        return _run(self._facade.list_maintenance_run_products(query))

    def list_provider_analytics_products(
        self,
        query: ProviderAnalyticsProductQuery | None = None,
    ) -> list[ProviderAnalyticsProduct]:
        """List provider-level analytics products."""
        return _run(self._facade.list_provider_analytics_products(query))

    def list_archive_debt_products(
        self,
        query: ArchiveDebtProductQuery | None = None,
    ) -> list[ArchiveDebtProduct]:
        """List live archive-debt products."""
        return _run(self._facade.list_archive_debt_products(query))

    def filter(self):
        """Create a fluent filter builder (terminal methods are still async)."""
        return self._facade.filter()

    def __repr__(self) -> str:
        return f"SyncPolylogue(facade={self._facade!r})"


__all__ = ["SyncPolylogue", "_run"]
