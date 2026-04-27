"""Raw/evidence persistence and query mixin for the conversation repository."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from polylogue.storage.backends.queries import artifacts as artifacts_q
from polylogue.storage.backends.queries import raw as raw_queries
from polylogue.storage.raw.models import RawConversationState, RawConversationStateUpdate
from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.runtime import (
    ArtifactObservationRecord,
    RawConversationRecord,
)
from polylogue.types import Provider, ValidationMode, ValidationStatus


class RepositoryRawMixin:
    if TYPE_CHECKING:
        _backend: RepositoryBackendProtocol

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

    async def update_raw_state(
        self,
        raw_id: str,
        *,
        state: RawConversationStateUpdate,
    ) -> None:
        """Apply a typed raw-state mutation."""
        async with self._backend.connection() as conn:
            await raw_queries.apply_raw_state_update(
                conn,
                raw_id,
                state=state,
                transaction_depth=self._backend.transaction_depth,
            )

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
        async with self._backend.connection() as conn:
            return await raw_queries.get_known_source_mtimes(conn)

    async def reset_parse_status(
        self,
        *,
        provider: str | None = None,
        source_names: list[str] | None = None,
    ) -> int:
        async with self._backend.connection() as conn:
            return await raw_queries.reset_parse_status(
                conn,
                provider=provider,
                source_names=source_names,
                transaction_depth=self._backend.transaction_depth,
            )

    async def reset_validation_status(
        self,
        *,
        provider: str | None = None,
        source_names: list[str] | None = None,
    ) -> int:
        async with self._backend.connection() as conn:
            return await raw_queries.reset_validation_status(
                conn,
                provider=provider,
                source_names=source_names,
                transaction_depth=self._backend.transaction_depth,
            )

    async def get_raw_conversations_batch(
        self,
        raw_ids: list[str],
    ) -> list[RawConversationRecord]:
        async with self._backend.connection() as conn:
            return await raw_queries.get_raw_conversations_batch(conn, raw_ids)

    async def get_raw_blob_sizes(
        self,
        raw_ids: list[str],
    ) -> list[tuple[str, int]]:
        async with self._backend.connection() as conn:
            return await raw_queries.get_raw_blob_sizes(conn, raw_ids)

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

    async def iter_raw_headers(
        self,
        *,
        source_names: list[str] | None = None,
        provider_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[tuple[str, int]]:
        async with self._backend.connection() as conn:
            async for header in raw_queries.iter_raw_headers(
                conn,
                source_names=source_names,
                provider_name=provider_name,
                require_unparsed=require_unparsed,
                require_unvalidated=require_unvalidated,
                validation_statuses=validation_statuses,
                page_size=page_size,
            ):
                yield header

    async def get_raw_conversation_count(self, provider: str | None = None) -> int:
        async with self._backend.connection() as conn:
            return await raw_queries.get_raw_conversation_count(conn, provider=provider)


__all__ = ["RepositoryRawMixin"]
