"""Raw/provenance methods for the async SQLite backend."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

from polylogue.storage.backends.queries import artifacts as artifacts_q
from polylogue.storage.backends.queries import raw as raw_queries
from polylogue.storage.state_views import RawConversationState, RawConversationStateUpdate
from polylogue.storage.store import ArtifactObservationRecord, RawConversationRecord
from polylogue.types import Provider, ValidationMode, ValidationStatus

if TYPE_CHECKING:
    import aiosqlite

    from polylogue.storage.backends.query_store import SQLiteQueryStore


class SQLiteRawMixin:
    """Raw archive and provenance methods for ``SQLiteBackend``."""

    if TYPE_CHECKING:
        queries: SQLiteQueryStore
        _transaction_depth: int

        def _get_connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]: ...

    def _raw_id_query(
        self,
        *,
        source_names: list[str] | None = None,
        provider_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Build the canonical scoped raw-ID query."""
        return self.queries.raw_id_query(
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
        """Iterate raw conversation IDs for a pipeline state slice."""
        async for rid in self.queries.iter_raw_ids(
            source_names=source_names,
            provider_name=provider_name,
            require_unparsed=require_unparsed,
            require_unvalidated=require_unvalidated,
            validation_statuses=validation_statuses,
            page_size=page_size,
        ):
            yield rid

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
        """Iterate raw conversation IDs with blob sizes for lightweight batching."""
        async for raw_header in self.queries.iter_raw_headers(
            source_names=source_names,
            provider_name=provider_name,
            require_unparsed=require_unparsed,
            require_unvalidated=require_unvalidated,
            validation_statuses=validation_statuses,
            page_size=page_size,
        ):
            yield raw_header

    async def save_raw_conversation(self, record: RawConversationRecord) -> bool:
        """Save a raw conversation record. Returns True if inserted."""
        async with self._get_connection() as conn:
            return await raw_queries.save_raw_conversation(conn, record, self._transaction_depth)

    async def save_artifact_observation(self, record: ArtifactObservationRecord) -> bool:
        """Persist or refresh one durable artifact observation."""
        async with self._get_connection() as conn:
            return await artifacts_q.save_artifact_observation(conn, record, self._transaction_depth)

    async def get_raw_conversation(self, raw_id: str) -> RawConversationRecord | None:
        """Retrieve a raw conversation by ID."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_conversation(conn, raw_id)

    async def update_raw_state(
        self,
        raw_id: str,
        *,
        state: RawConversationStateUpdate,
    ) -> None:
        """Apply a typed raw-state mutation."""
        async with self._get_connection() as conn:
            await raw_queries.apply_raw_state_update(
                conn,
                raw_id,
                state=state,
                transaction_depth=self._transaction_depth,
            )

    async def mark_raw_parsed(
        self,
        raw_id: str,
        *,
        error: str | None = None,
        payload_provider: Provider | str | None = None,
    ) -> None:
        """Mark a raw conversation as parsed (or record a parse error)."""
        async with self._get_connection() as conn:
            await raw_queries.mark_raw_parsed(
                conn,
                raw_id,
                error=error,
                payload_provider=payload_provider,
                transaction_depth=self._transaction_depth,
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
        """Persist validation status for a raw conversation record."""
        async with self._get_connection() as conn:
            await raw_queries.mark_raw_validated(
                conn,
                raw_id,
                status=status,
                error=error,
                drift_count=drift_count,
                provider=provider,
                mode=mode,
                payload_provider=payload_provider,
                transaction_depth=self._transaction_depth,
            )

    async def get_known_source_mtimes(self) -> dict[str, str]:
        """Return {source_path: file_mtime} for all raw records with an mtime."""
        return await self.queries.get_known_source_mtimes()

    async def reset_parse_status(
        self,
        *,
        provider: str | None = None,
        source_names: list[str] | None = None,
    ) -> int:
        """Clear parsed_at/parse_error to force re-parsing on next run."""
        async with self._get_connection() as conn:
            return await raw_queries.reset_parse_status(
                conn,
                provider=provider,
                source_names=source_names,
                transaction_depth=self._transaction_depth,
            )

    async def reset_validation_status(
        self,
        *,
        provider: str | None = None,
        source_names: list[str] | None = None,
    ) -> int:
        """Clear validation tracking to force re-validation on next run."""
        async with self._get_connection() as conn:
            return await raw_queries.reset_validation_status(
                conn,
                provider=provider,
                source_names=source_names,
                transaction_depth=self._transaction_depth,
            )

    async def get_raw_conversations_batch(
        self,
        raw_ids: list[str],
    ) -> list[RawConversationRecord]:
        """Fetch multiple raw conversations in a single query."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_conversations_batch(conn, raw_ids)

    async def get_raw_blob_sizes(
        self,
        raw_ids: list[str],
    ) -> list[tuple[str, int]]:
        """Fetch raw conversation blob sizes without hydrating full records."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_blob_sizes(conn, raw_ids)

    async def get_raw_conversation_states(
        self,
        raw_ids: list[str],
    ) -> dict[str, RawConversationState]:
        """Fetch persisted processing state for raw conversation IDs."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_conversation_states(conn, raw_ids)

    async def iter_raw_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[RawConversationRecord]:
        """Iterate over raw conversation records."""
        async with self._get_connection() as conn:
            async for record in raw_queries.iter_raw_conversations(conn, provider, limit):
                yield record

    async def get_raw_conversation_count(self, provider: str | None = None) -> int:
        """Get count of raw conversations."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_conversation_count(conn, provider)


__all__ = ["SQLiteRawMixin"]
