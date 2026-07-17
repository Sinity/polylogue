"""Raw/evidence persistence and query mixin for the session repository."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.logging import get_logger
from polylogue.storage.raw.models import RawSessionState, RawSessionStateUpdate
from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.runtime import (
    ArtifactObservationRecord,
    RawSessionRecord,
)
from polylogue.storage.sqlite.queries import artifacts as artifacts_q
from polylogue.storage.sqlite.queries import cursor as cursor_queries
from polylogue.storage.sqlite.queries import raw as raw_queries

logger = get_logger(__name__)


class RepositoryRawMixin:
    if TYPE_CHECKING:
        _backend: RepositoryBackendProtocol

    async def save_raw_session(self, record: RawSessionRecord) -> bool:
        async with self._backend.connection() as conn:
            return await raw_queries.save_raw_session(
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

    async def get_raw_session(self, raw_id: str) -> RawSessionRecord | None:
        async with self._backend.connection() as conn:
            return await raw_queries.get_raw_session(conn, raw_id)

    async def update_raw_state(
        self,
        raw_id: str,
        *,
        state: RawSessionStateUpdate,
    ) -> None:
        """Apply a typed raw-state mutation."""
        source_backend = getattr(self, "_source_backend", None)
        backend = source_backend if source_backend is not None else self._backend
        async with backend.connection() as conn:
            await raw_queries.apply_raw_state_update(
                conn,
                raw_id,
                state=state,
                transaction_depth=backend.transaction_depth,
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

    async def get_known_source_cursors(self) -> dict[str, dict[str, object]]:
        """Return ingest_cursor stat fields for the stat-based fast path."""
        async with self._backend.connection() as conn:
            try:
                return await cursor_queries.get_known_source_cursors(conn)
            except Exception as exc:
                # An empty map silently disables the stat fast path (every
                # source re-hashes); make the degradation visible.
                logger.warning(
                    "source-cursor fast path unavailable (%s: %s); falling back to full source scan",
                    type(exc).__name__,
                    exc,
                )
                return {}

    async def upsert_source_file_cursor(
        self,
        source_path: str,
        *,
        st_dev: int | None = None,
        st_ino: int | None = None,
        st_size: int | None = None,
        mtime_ns: int | None = None,
    ) -> None:
        async with self._backend.connection() as conn:
            from polylogue.storage.sqlite.queries.cursor import upsert_source_file_cursor as _upsert

            await _upsert(
                conn,
                source_path,
                st_dev=st_dev,
                st_ino=st_ino,
                st_size=st_size,
                mtime_ns=mtime_ns,
                transaction_depth=self._backend.transaction_depth,
            )

    async def reset_parse_status(
        self,
        *,
        origin: str | None = None,
        source_names: list[str] | None = None,
    ) -> int:
        async with self._backend.connection() as conn:
            return await raw_queries.reset_parse_status(
                conn,
                origin=origin,
                source_names=source_names,
                transaction_depth=self._backend.transaction_depth,
            )

    async def reset_validation_status(
        self,
        *,
        origin: str | None = None,
        source_names: list[str] | None = None,
    ) -> int:
        async with self._backend.connection() as conn:
            return await raw_queries.reset_validation_status(
                conn,
                origin=origin,
                source_names=source_names,
                transaction_depth=self._backend.transaction_depth,
            )

    async def get_raw_sessions_batch(
        self,
        raw_ids: list[str],
    ) -> list[RawSessionRecord]:
        async with self._backend.connection() as conn:
            return await raw_queries.get_raw_sessions_batch(conn, raw_ids)

    async def get_raw_blob_sizes(
        self,
        raw_ids: list[str],
    ) -> list[tuple[str, int]]:
        async with self._backend.connection() as conn:
            return await raw_queries.get_raw_blob_sizes(conn, raw_ids)

    async def get_raw_session_states(
        self,
        raw_ids: list[str],
    ) -> dict[str, RawSessionState]:
        async with self._backend.connection() as conn:
            return await raw_queries.get_raw_session_states(conn, raw_ids)

    async def iter_raw_sessions(
        self,
        origin: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[RawSessionRecord]:
        async with self._backend.connection() as conn:
            async for record in raw_queries.iter_raw_sessions(
                conn,
                origin=origin,
                limit=limit,
            ):
                yield record

    async def iter_raw_headers(
        self,
        *,
        source_paths: list[str] | None = None,
        source_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[tuple[str, int]]:
        async with self._backend.connection() as conn:
            async for header in raw_queries.iter_raw_headers(
                conn,
                source_paths=source_paths,
                source_name=source_name,
                require_unparsed=require_unparsed,
                require_unvalidated=require_unvalidated,
                validation_statuses=validation_statuses,
                page_size=page_size,
            ):
                yield header

    async def get_raw_session_count(self, origin: str | None = None) -> int:
        async with self._backend.connection() as conn:
            return await raw_queries.get_raw_session_count(conn, origin=origin)

    async def get_raw_records_for_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[RawSessionRecord], int]:
        async with self._backend.connection() as conn:
            return await raw_queries.get_raw_records_for_session(
                conn,
                session_id,
                limit=limit,
                offset=offset,
            )


__all__ = ["RepositoryRawMixin"]
