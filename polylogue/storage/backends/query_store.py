"""Low-level SQLite query store composed from explicit concern bands."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

import aiosqlite

from polylogue.storage.backends.query_store_archive import SQLiteQueryStoreArchiveMixin
from polylogue.storage.backends.query_store_maintenance import SQLiteQueryStoreMaintenanceMixin
from polylogue.storage.backends.query_store_products import SQLiteQueryStoreProductsMixin


class SQLiteQueryStore(
    SQLiteQueryStoreArchiveMixin,
    SQLiteQueryStoreProductsMixin,
    SQLiteQueryStoreMaintenanceMixin,
):
    """Canonical low-level read/query API for SQLite archive state."""

    def __init__(
        self,
        *,
        connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]],
    ) -> None:
        self._connection_factory = connection_factory


__all__ = ["SQLiteQueryStore"]
