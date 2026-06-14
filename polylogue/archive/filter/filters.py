"""Fluent adapter over the canonical immutable session query plan.

Execution runs over the archive
(:class:`~polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore`); the filter
holds an archive root plus the runtime config used to resolve the optional
vector provider for semantic/hybrid retrieval.
"""

from __future__ import annotations

import builtins
from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.archive.filter.builder import SessionFilterBuilderMixin
from polylogue.archive.filter.types import SortField
from polylogue.archive.query.archive_execution import (
    count_archive,
    delete_archive,
    first_archive,
    list_archive,
    list_summaries_archive,
)
from polylogue.archive.query.fields import SqlPushdownParams
from polylogue.archive.query.plan import SessionQueryPlan

if TYPE_CHECKING:
    from pathlib import Path

    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.config import Config
    from polylogue.protocols import VectorProvider


class SessionFilter(SessionFilterBuilderMixin):
    """Fluent query shell backed directly by the canonical execution plan."""

    def __init__(
        self,
        *,
        archive_root: Path,
        config: Config | None = None,
        vector_provider: VectorProvider | None = None,
        query_plan: SessionQueryPlan | None = None,
    ) -> None:
        self._archive_root = archive_root
        self._config = config
        self._plan = query_plan or SessionQueryPlan(vector_provider=vector_provider)

    @classmethod
    def from_query_plan(
        cls,
        query_plan: SessionQueryPlan,
        *,
        archive_root: Path,
        config: Config | None = None,
    ) -> SessionFilter:
        return cls(archive_root=archive_root, config=config, query_plan=query_plan)

    @property
    def _since_date(self) -> datetime | None:
        return self._plan.since

    @property
    def _until_date(self) -> datetime | None:
        return self._plan.until

    @property
    def _continuation(self) -> bool | None:
        return self._plan.continuation

    @property
    def _sidechain(self) -> bool | None:
        return self._plan.sidechain

    @property
    def _has_branches(self) -> bool | None:
        return self._plan.has_branches

    def build_query_plan(self) -> SessionQueryPlan:
        return self._plan

    def _sql_pushdown_params(self) -> SqlPushdownParams:
        return self._plan.sql_pushdown_params()

    def _has_post_filters(self) -> bool:
        return self._plan.has_post_filters()

    def _needs_content_loading(self) -> bool:
        return self._plan.needs_content_loading()

    def can_use_summaries(self) -> bool:
        return self._plan.can_use_summaries()

    def describe(self) -> list[str]:
        return self._plan.describe()

    async def list(self) -> builtins.list[Session]:
        return await list_archive(self._plan, archive_root=self._archive_root, config=self._config)

    async def list_summaries(self) -> builtins.list[SessionSummary]:
        return await list_summaries_archive(self._plan, archive_root=self._archive_root, config=self._config)

    async def list_all_summaries(self) -> builtins.list[SessionSummary]:
        """Resolve every matching summary (unbounded), not a single page.

        ``list_summaries`` caps at the default page limit (50). Mutation and
        cardinality paths (delete/mark) must act on the complete matched set, so
        they resolve unbounded — mirroring ``count_archive`` (#1873).
        """
        return await list_summaries_archive(
            self._plan.with_limit(None),
            archive_root=self._archive_root,
            config=self._config,
            default_limit=1_000_000,
        )

    async def list_all(self) -> builtins.list[Session]:
        """Resolve every matching session (unbounded); see :meth:`list_all_summaries`."""
        return await list_archive(
            self._plan.with_limit(None),
            archive_root=self._archive_root,
            config=self._config,
            default_limit=1_000_000,
        )

    async def first(self) -> Session | None:
        return await first_archive(self._plan, archive_root=self._archive_root, config=self._config)

    async def count(self) -> int:
        return await count_archive(self._plan, archive_root=self._archive_root, config=self._config)

    async def delete(self) -> int:
        return await delete_archive(self._plan, archive_root=self._archive_root, config=self._config)


__all__ = ["SessionFilter", "SortField"]
