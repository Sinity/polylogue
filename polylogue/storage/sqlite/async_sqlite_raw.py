"""Raw/provenance methods for the async SQLite backend."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.core.sources import origin_from_provider
from polylogue.storage.raw.models import RawSessionState, RawSessionStateUpdate
from polylogue.storage.runtime import ArtifactObservationRecord, RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.write import _timestamp_ms
from polylogue.storage.sqlite.queries import artifacts as artifacts_q
from polylogue.storage.sqlite.queries import raw as raw_queries
from polylogue.types import Provider, ValidationMode, ValidationStatus

if TYPE_CHECKING:
    from pathlib import Path

    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class SQLiteRawMixin:
    """Raw archive and provenance methods for ``SQLiteBackend``."""

    if TYPE_CHECKING:
        queries: SQLiteQueryStore
        _transaction_depth: int
        _source_db_path: Path

        def _get_connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]: ...

    def _raw_id_query(
        self,
        *,
        source_names: list[str] | None = None,
        source_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Build the canonical scoped raw-ID query."""
        return self.queries.raw_id_query(
            source_names=source_names,
            source_name=source_name,
            require_unparsed=require_unparsed,
            require_unvalidated=require_unvalidated,
            validation_statuses=validation_statuses,
        )

    async def iter_raw_ids(
        self,
        *,
        source_names: list[str] | None = None,
        source_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        """Iterate raw session IDs for a pipeline state slice."""
        async for rid in self.queries.iter_raw_ids(
            source_names=source_names,
            source_name=source_name,
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
        source_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[tuple[str, int]]:
        """Iterate raw session IDs with blob sizes for lightweight batching."""
        async for raw_header in self.queries.iter_raw_headers(
            source_names=source_names,
            source_name=source_name,
            require_unparsed=require_unparsed,
            require_unvalidated=require_unvalidated,
            validation_statuses=validation_statuses,
            page_size=page_size,
        ):
            yield raw_header

    async def save_raw_session(self, record: RawSessionRecord) -> bool:
        """Save a raw session record. Returns True if inserted."""
        # payload_provider wins when the payload has been classified; otherwise
        # fall back to the source_name token (#1743 collapses both onto origin).
        if record.payload_provider is not None:
            origin = origin_from_provider(record.payload_provider)
        else:
            origin = origin_from_provider(Provider.from_string(record.source_name or "unknown"))
        try:
            blob_hash = bytes.fromhex(record.raw_id)
        except ValueError:
            blob_hash = record.raw_id.encode("utf-8")
        if len(blob_hash) != 32:
            blob_hash = hashlib.sha256(blob_hash).digest()

        acquired_at_ms = _timestamp_ms(record.acquired_at) or 0
        async with aiosqlite.connect(self._source_db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON")
            cursor = await conn.execute("SELECT 1 FROM raw_sessions WHERE raw_id = ?", (record.raw_id,))
            existed = await cursor.fetchone() is not None
            await conn.execute(
                """
                INSERT OR REPLACE INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash,
                    blob_size, acquired_at_ms, file_mtime_ms, parsed_at_ms, parse_error,
                    validated_at_ms, validation_status, validation_error, validation_drift_count,
                    validation_mode, detection_warnings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.raw_id,
                    origin.value,
                    None,
                    record.source_path,
                    int(record.source_index or 0),
                    blob_hash,
                    int(record.blob_size),
                    acquired_at_ms,
                    _timestamp_ms(record.file_mtime),
                    _timestamp_ms(record.parsed_at),
                    record.parse_error,
                    _timestamp_ms(record.validated_at),
                    record.validation_status.value if record.validation_status is not None else None,
                    record.validation_error,
                    int(record.validation_drift_count or 0),
                    record.validation_mode.value if record.validation_mode is not None else None,
                    record.detection_warnings or "[]",
                ),
            )
            await conn.execute(
                """
                INSERT OR REPLACE INTO blob_refs (
                    blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms
                ) VALUES (?, ?, 'raw_payload', ?, ?, ?)
                """,
                (
                    blob_hash,
                    record.raw_id,
                    record.source_path,
                    int(record.blob_size),
                    acquired_at_ms,
                ),
            )
            await conn.commit()
            return not existed

    async def save_artifact_observation(self, record: ArtifactObservationRecord) -> bool:
        """Persist or refresh one durable artifact observation."""
        async with self._get_connection() as conn:
            return await artifacts_q.save_artifact_observation(conn, record, self._transaction_depth)

    async def get_raw_session(self, raw_id: str) -> RawSessionRecord | None:
        """Retrieve a raw session by ID."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_session(conn, raw_id)

    async def update_raw_state(
        self,
        raw_id: str,
        *,
        state: RawSessionStateUpdate,
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
        """Mark a raw session as parsed (or record a parse error)."""
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
        """Persist validation status for a raw session record."""
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

    async def get_raw_sessions_batch(
        self,
        raw_ids: list[str],
    ) -> list[RawSessionRecord]:
        """Fetch multiple raw sessions in a single query."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_sessions_batch(conn, raw_ids)

    async def get_raw_blob_sizes(
        self,
        raw_ids: list[str],
    ) -> list[tuple[str, int]]:
        """Fetch raw session blob sizes without hydrating full records."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_blob_sizes(conn, raw_ids)

    async def get_raw_session_states(
        self,
        raw_ids: list[str],
    ) -> dict[str, RawSessionState]:
        """Fetch persisted processing state for raw session IDs."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_session_states(conn, raw_ids)

    async def iter_raw_sessions(
        self,
        provider: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[RawSessionRecord]:
        """Iterate over raw session records."""
        async with self._get_connection() as conn:
            async for record in raw_queries.iter_raw_sessions(conn, provider, limit):
                yield record

    async def get_raw_session_count(self, provider: str | None = None) -> int:
        """Get count of raw sessions."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_session_count(conn, provider)

    async def get_raw_records_for_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[RawSessionRecord], int]:
        """Look up raw session records for a given session ID."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_records_for_session(
                conn,
                session_id,
                limit=limit,
                offset=offset,
            )


__all__ = ["SQLiteRawMixin"]
