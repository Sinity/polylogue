"""Raw/run/publication query band for SQLiteQueryStore (maintenance run queries removed)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.storage.backends.queries import publications as publications_q
from polylogue.storage.backends.queries import raw as raw_queries
from polylogue.storage.backends.queries import runs as runs_q
from polylogue.storage.runtime import PublicationRecord, RunRecord


class SQLiteQueryStoreMaintenanceMixin:
    if TYPE_CHECKING:
        _connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]]

    def raw_id_query(
        self,
        *,
        source_names: list[str] | None = None,
        provider_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
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
        async with self._connection_factory() as conn:
            async for raw_header in raw_queries.iter_raw_headers(
                conn,
                source_names=source_names,
                provider_name=provider_name,
                require_unparsed=require_unparsed,
                require_unvalidated=require_unvalidated,
                validation_statuses=validation_statuses,
                page_size=page_size,
            ):
                yield raw_header

    async def get_known_source_mtimes(self) -> dict[str, str]:
        async with self._connection_factory() as conn:
            return await raw_queries.get_known_source_mtimes(conn)

    async def get_latest_run(self) -> RunRecord | None:
        async with self._connection_factory() as conn:
            return await runs_q.get_latest_run(conn)

    async def get_latest_publication(
        self,
        publication_kind: str,
    ) -> PublicationRecord | None:
        async with self._connection_factory() as conn:
            return await publications_q.get_latest_publication(conn, publication_kind)
