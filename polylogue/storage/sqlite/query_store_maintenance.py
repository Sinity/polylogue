"""Raw query band for SQLiteQueryStore maintenance reads."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.queries import raw as raw_queries


class SQLiteQueryStoreMaintenanceMixin:
    if TYPE_CHECKING:
        _connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]]

    def raw_id_query(
        self,
        *,
        source_paths: list[str] | None = None,
        source_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        return raw_queries.raw_id_query(
            source_paths=source_paths,
            source_name=source_name,
            require_unparsed=require_unparsed,
            require_unvalidated=require_unvalidated,
            validation_statuses=validation_statuses,
        )

    async def iter_raw_ids(
        self,
        *,
        source_paths: list[str] | None = None,
        source_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        async with self._connection_factory() as conn:
            async for raw_id in raw_queries.iter_raw_ids(
                conn,
                source_paths=source_paths,
                source_name=source_name,
                require_unparsed=require_unparsed,
                require_unvalidated=require_unvalidated,
                validation_statuses=validation_statuses,
                page_size=page_size,
            ):
                yield raw_id

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
        async with self._connection_factory() as conn:
            async for raw_header in raw_queries.iter_raw_headers(
                conn,
                source_paths=source_paths,
                source_name=source_name,
                require_unparsed=require_unparsed,
                require_unvalidated=require_unvalidated,
                validation_statuses=validation_statuses,
                page_size=page_size,
            ):
                yield raw_header

    async def get_known_source_mtimes(self) -> dict[str, str]:
        async with self._connection_factory() as conn:
            return await raw_queries.get_known_source_mtimes(conn)

    async def get_raw_records_for_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[RawSessionRecord], int]:
        async with self._connection_factory() as conn:
            return await raw_queries.get_raw_records_for_session(
                conn,
                session_id,
                limit=limit,
                offset=offset,
            )
