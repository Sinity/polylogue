"""Low-level SQLite query store composed from explicit concern bands."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

import aiosqlite

from polylogue.storage.backends.query_store_archive import SQLiteQueryStoreArchiveMixin
from polylogue.storage.backends.query_store_maintenance import SQLiteQueryStoreMaintenanceMixin
from polylogue.storage.backends.query_store_product_actions import (
    SQLiteQueryStoreProductActionsMixin,
)
from polylogue.storage.backends.query_store_product_profiles import (
    SQLiteQueryStoreProductProfilesMixin,
)
from polylogue.storage.backends.query_store_product_status import (
    SQLiteQueryStoreProductStatusMixin,
)
from polylogue.storage.backends.query_store_product_summaries import (
    SQLiteQueryStoreProductSummariesMixin,
)
from polylogue.storage.backends.query_store_product_threads import (
    SQLiteQueryStoreProductThreadsMixin,
)
from polylogue.storage.backends.query_store_product_timelines import (
    SQLiteQueryStoreProductTimelinesMixin,
)


class SQLiteQueryStore(
    SQLiteQueryStoreArchiveMixin,
    SQLiteQueryStoreProductActionsMixin,
    SQLiteQueryStoreProductStatusMixin,
    SQLiteQueryStoreProductProfilesMixin,
    SQLiteQueryStoreProductTimelinesMixin,
    SQLiteQueryStoreProductThreadsMixin,
    SQLiteQueryStoreProductSummariesMixin,
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
