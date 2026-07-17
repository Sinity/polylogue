"""Read band for provider-neutral work-evidence traversal."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.insights.work_evidence import WorkEvidenceTraversal
from polylogue.storage.query_models import WorkEvidenceTraversalQuery
from polylogue.storage.sqlite.queries import work_evidence as work_evidence_q


class SQLiteQueryStoreWorkEvidenceMixin:
    if TYPE_CHECKING:
        _connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]]

    async def get_work_evidence_traversal(
        self,
        query: WorkEvidenceTraversalQuery,
    ) -> WorkEvidenceTraversal | None:
        async with self._connection_factory() as conn:
            return await work_evidence_q.get_work_evidence_traversal(conn, query)


__all__ = ["SQLiteQueryStoreWorkEvidenceMixin"]
