"""Raw/evidence persistence and query mixin for the conversation repository."""

from __future__ import annotations

from collections.abc import AsyncIterator

from polylogue.storage.backends.queries import artifacts as artifacts_q
from polylogue.storage.backends.queries import raw as raw_queries
from polylogue.storage.state_views import RawConversationState
from polylogue.storage.store import (
    ArtifactObservationRecord,
    RawConversationRecord,
)
from polylogue.types import Provider, ValidationMode, ValidationStatus


class RepositoryRawMixin:
    async def save_raw_conversation(self, record: RawConversationRecord) -> bool:
        async with self._backend.connection() as conn:
            return await raw_queries.save_raw_conversation(
                conn,
                record,
                self._backend.transaction_depth,
            )

    async def save_artifact_observation(self, record: ArtifactObservationRecord) -> bool:
        async with self._backend.connection() as conn:
            return await artifacts_q.save_artifact_observation(
                conn,
                record,
                self._backend.transaction_depth,
            )

    async def get_raw_conversation(self, raw_id: str) -> RawConversationRecord | None:
        async with self._backend.connection() as conn:
            return await raw_queries.get_raw_conversation(conn, raw_id)

    async def mark_raw_parsed(
        self,
        raw_id: str,
        *,
        error: str | None = None,
        payload_provider: Provider | str | None = None,
    ) -> None:
        async with self._backend.connection() as conn:
            await raw_queries.mark_raw_parsed(
                conn,
                raw_id,
                error=error,
                payload_provider=payload_provider,
                transaction_depth=self._backend.transaction_depth,
            )

    async def mark_raw_validated(
        self,
        raw_id: str,
        *,
        status: ValidationStatus | str,
        error: str | None = None,
        drift_count: int = 0,
        provider: Provider | str | None = None,
        mode: ValidationMode | str | None = None,
        payload_provider: Provider | str | None = None,
    ) -> None:
        async with self._backend.connection() as conn:
            await raw_queries.mark_raw_validated(
                conn,
                raw_id,
                status=status,
                error=error,
                drift_count=drift_count,
                provider=provider,
                mode=mode,
                payload_provider=payload_provider,
                transaction_depth=self._backend.transaction_depth,
            )

    async def get_known_source_mtimes(self) -> dict[str, str]:
        return await self._backend.queries.get_known_source_mtimes()

    async def reset_parse_status(self, *, provider: str | None = None) -> int:
        async with self._backend.connection() as conn:
            return await raw_queries.reset_parse_status(
                conn,
                provider=provider,
                transaction_depth=self._backend.transaction_depth,
            )

    async def reset_validation_status(self, *, provider: str | None = None) -> int:
        async with self._backend.connection() as conn:
            return await raw_queries.reset_validation_status(
                conn,
                provider=provider,
                transaction_depth=self._backend.transaction_depth,
            )

    async def get_raw_conversations_batch(
        self,
        raw_ids: list[str],
    ) -> list[RawConversationRecord]:
        async with self._backend.connection() as conn:
            return await raw_queries.get_raw_conversations_batch(conn, raw_ids)

    async def get_raw_conversation_states(
        self,
        raw_ids: list[str],
    ) -> dict[str, RawConversationState]:
        async with self._backend.connection() as conn:
            return await raw_queries.get_raw_conversation_states(conn, raw_ids)

    async def iter_raw_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[RawConversationRecord]:
        async with self._backend.connection() as conn:
            async for record in raw_queries.iter_raw_conversations(
                conn,
                provider=provider,
                limit=limit,
            ):
                yield record

    async def get_raw_conversation_count(self, provider: str | None = None) -> int:
        async with self._backend.connection() as conn:
            return await raw_queries.get_raw_conversation_count(conn, provider=provider)


__all__ = ["RepositoryRawMixin"]
