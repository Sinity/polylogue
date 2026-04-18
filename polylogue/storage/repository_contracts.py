"""Typed internal contracts shared by repository mixins."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import aiosqlite

if TYPE_CHECKING:
    from polylogue.storage.backends.query_store import SQLiteQueryStore


class RepositoryBackendProtocol(Protocol):
    """Minimal backend surface required by repository helper mixins."""

    @property
    def transaction_depth(self) -> int: ...

    @property
    def db_path(self) -> Path: ...

    @property
    def queries(self) -> SQLiteQueryStore: ...

    def connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]: ...

    def read_connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]: ...

    def transaction(self) -> AbstractAsyncContextManager[object]: ...

    async def close(self) -> None: ...
