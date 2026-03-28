"""Session/timeline archive-product mixins."""

from __future__ import annotations

from polylogue.archive_products import (
    SessionEnrichmentProduct,
    SessionEnrichmentProductQuery,
    SessionPhaseProduct,
    SessionPhaseProductQuery,
    SessionProfileProduct,
    SessionProfileProductQuery,
    SessionWorkEventProduct,
    SessionWorkEventProductQuery,
    WorkThreadProduct,
    WorkThreadProductQuery,
)


class ArchiveProductSessionMixin:
    async def get_session_profile_product(
        self,
        conversation_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileProduct | None:
        record = await self.repository.get_session_profile_record(conversation_id)
        return SessionProfileProduct.from_record(record, tier=tier) if record is not None else None

    async def list_session_profile_products(
        self,
        query: SessionProfileProductQuery | None = None,
    ) -> list[SessionProfileProduct]:
        request = query or SessionProfileProductQuery()
        records = await self.repository.list_session_profile_records(
            provider=request.provider,
            since=request.since,
            until=request.until,
            first_message_since=request.first_message_since,
            first_message_until=request.first_message_until,
            session_date_since=request.session_date_since,
            session_date_until=request.session_date_until,
            tier=request.tier,
            limit=request.limit,
            offset=request.offset,
            query=request.query,
        )
        return [SessionProfileProduct.from_record(record, tier=request.tier) for record in records]

    async def get_session_enrichment_product(
        self,
        conversation_id: str,
    ) -> SessionEnrichmentProduct | None:
        record = await self.repository.get_session_enrichment_record(conversation_id)
        return SessionEnrichmentProduct.from_record(record) if record is not None else None

    async def list_session_enrichment_products(
        self,
        query: SessionEnrichmentProductQuery | None = None,
    ) -> list[SessionEnrichmentProduct]:
        request = query or SessionEnrichmentProductQuery()
        records = await self.repository.list_session_enrichment_records(
            provider=request.provider,
            since=request.since,
            until=request.until,
            first_message_since=request.first_message_since,
            first_message_until=request.first_message_until,
            session_date_since=request.session_date_since,
            session_date_until=request.session_date_until,
            refined_work_kind=request.refined_work_kind,
            limit=request.limit,
            offset=request.offset,
            query=request.query,
        )
        return [SessionEnrichmentProduct.from_record(record) for record in records]

    async def get_session_work_event_products(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventProduct]:
        records = await self.repository.get_session_work_event_records(conversation_id)
        return [SessionWorkEventProduct.from_record(record) for record in records]

    async def list_session_work_event_products(
        self,
        query: SessionWorkEventProductQuery | None = None,
    ) -> list[SessionWorkEventProduct]:
        request = query or SessionWorkEventProductQuery()
        records = await self.repository.list_session_work_event_records(
            conversation_id=request.conversation_id,
            provider=request.provider,
            since=request.since,
            until=request.until,
            kind=request.kind,
            limit=request.limit,
            offset=request.offset,
            query=request.query,
        )
        return [SessionWorkEventProduct.from_record(record) for record in records]

    async def get_session_phase_products(
        self,
        conversation_id: str,
    ) -> list[SessionPhaseProduct]:
        records = await self.repository.get_session_phase_records(conversation_id)
        return [SessionPhaseProduct.from_record(record) for record in records]

    async def list_session_phase_products(
        self,
        query: SessionPhaseProductQuery | None = None,
    ) -> list[SessionPhaseProduct]:
        request = query or SessionPhaseProductQuery()
        records = await self.repository.list_session_phase_records(
            conversation_id=request.conversation_id,
            provider=request.provider,
            since=request.since,
            until=request.until,
            kind=request.kind,
            limit=request.limit,
            offset=request.offset,
        )
        return [SessionPhaseProduct.from_record(record) for record in records]

    async def get_work_thread_product(self, thread_id: str) -> WorkThreadProduct | None:
        record = await self.repository.get_work_thread_record(thread_id)
        return WorkThreadProduct.from_record(record) if record is not None else None

    async def list_work_thread_products(
        self,
        query: WorkThreadProductQuery | None = None,
    ) -> list[WorkThreadProduct]:
        request = query or WorkThreadProductQuery()
        records = await self.repository.list_work_thread_records(
            since=request.since,
            until=request.until,
            limit=request.limit,
            offset=request.offset,
            query=request.query,
        )
        return [WorkThreadProduct.from_record(record) for record in records]


__all__ = ["ArchiveProductSessionMixin"]
