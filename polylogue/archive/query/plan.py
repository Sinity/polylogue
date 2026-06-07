"""Canonical immutable session-query plan model."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

from polylogue.archive.query.fields import (
    SqlPushdownParams,
    session_record_query_for_plan,
    sql_pushdown_params_for_plan,
)
from polylogue.archive.query.plan_description import describe_plan, effective_fetch_limit, plan_has_filters
from polylogue.archive.query.retrieval import (
    actions_ready,
    can_use_action_stats_with,
    candidate_record_query,
    candidate_record_query_for,
    fetch_record_query_for,
    search_limit,
    should_batch_post_filter_fetch,
    uses_actions,
)
from polylogue.archive.query.runtime import (
    apply_common_filters,
    apply_full_filters,
    matches_action_sequence,
    matches_action_terms,
    matches_action_text_terms,
    matches_referenced_path,
    matches_tool_terms,
    plan_can_count_in_sql,
    plan_can_use_action_stats,
    plan_has_post_filters,
    plan_needs_content_loading,
)
from polylogue.archive.query.sorting import SortKey, finalize_results, sort_generic, sort_sessions, sort_summaries
from polylogue.archive.query.support import session_has_branches
from polylogue.storage.query_models import SessionRecordQuery

if TYPE_CHECKING:
    from pathlib import Path

    from polylogue.archive.filter.types import SortField
    from polylogue.archive.models import Session, SessionSummary
    from polylogue.archive.query.runtime_filters import FilterableSessionLike
    from polylogue.config import Config
    from polylogue.protocols import SessionQueryRuntimeStore, VectorProvider

_T = TypeVar("_T")
_FilterableT = TypeVar("_FilterableT", bound="FilterableSessionLike")


# ---------------------------------------------------------------------------
# Record-query translation helpers
# ---------------------------------------------------------------------------


def plan_record_query(plan: SessionQueryPlan) -> SessionRecordQuery:
    return session_record_query_for_plan(plan)


def plan_sql_pushdown_params(plan: SessionQueryPlan) -> SqlPushdownParams:
    return sql_pushdown_params_for_plan(plan)


def _archive_root_for_config(config: Config) -> Path:
    from polylogue.paths import archive_file_set_root_for_paths

    return archive_file_set_root_for_paths(
        archive_root_path=config.archive_root,
        db_anchor=config.db_path,
    )


# ---------------------------------------------------------------------------
# Query plan dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionQueryPlan:
    """Canonical immutable execution state for session selection."""

    query_terms: tuple[str, ...] = ()
    contains_terms: tuple[str, ...] = ()
    negative_terms: tuple[str, ...] = ()
    retrieval_lane: str = "auto"
    referenced_path: tuple[str, ...] = ()
    cwd_prefix: str | None = None
    action_terms: tuple[str, ...] = ()
    excluded_action_terms: tuple[str, ...] = ()
    action_sequence: tuple[str, ...] = ()
    action_text_terms: tuple[str, ...] = ()
    tool_terms: tuple[str, ...] = ()
    excluded_tool_terms: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()
    excluded_origins: tuple[str, ...] = ()
    repo_names: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    excluded_tags: tuple[str, ...] = ()
    has_types: tuple[str, ...] = ()
    title: str | None = None
    session_id: str | None = None
    parent_id: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    sort: SortField | None = None
    reverse: bool = False
    limit: int | None = None
    sample: int | None = None
    similar_text: str | None = None
    predicates: tuple[Callable[[Session], bool], ...] = ()
    continuation: bool | None = None
    sidechain: bool | None = None
    root: bool | None = None
    has_branches: bool | None = None
    filter_has_tool_use: bool = False
    filter_has_thinking: bool = False
    filter_has_paste: bool = False
    typed_only: bool = False
    min_messages: int | None = None
    max_messages: int | None = None
    min_words: int | None = None
    since_session_id: str | None = None
    message_type: str | None = None
    offset: int = 0
    cursor: str | None = None
    ranking_policy: str = "default"
    ranking_policy_version: str = "1"
    vector_provider: VectorProvider | None = None

    # -- Description / record-query methods (was QueryPlanDescriptionMixin) --

    @property
    def fts_terms(self) -> tuple[str, ...]:
        return self.query_terms + self.contains_terms

    @property
    def sql_pushed(self) -> bool:
        return not self.fts_terms and self.session_id is None

    @property
    def record_query(self) -> SessionRecordQuery:
        return plan_record_query(self)

    def sql_pushdown_params(self) -> SqlPushdownParams:
        return plan_sql_pushdown_params(self)

    def describe(self) -> list[str]:
        return describe_plan(self)

    def has_filters(self) -> bool:
        return plan_has_filters(self)

    def effective_fetch_limit(self) -> int | None:
        return effective_fetch_limit(self)

    def with_limit(self, limit: int | None) -> SessionQueryPlan:
        return replace(self, limit=limit)

    # -- Runtime filtering and sorting methods (was QueryPlanRuntimeMixin) --

    def has_post_filters(self) -> bool:
        return plan_has_post_filters(self)

    def needs_content_loading(self) -> bool:
        return plan_needs_content_loading(self)

    def can_use_summaries(self) -> bool:
        return not self.needs_content_loading()

    def can_count_in_sql(self) -> bool:
        return plan_can_count_in_sql(self)

    def can_use_action_stats(self) -> bool:
        return plan_can_use_action_stats(self)

    def _matches_referenced_path(self, session: Session) -> bool:
        return matches_referenced_path(self, session)

    def _matches_action_terms(self, session: Session) -> bool:
        return matches_action_terms(self, session)

    def _matches_tool_terms(self, session: Session) -> bool:
        return matches_tool_terms(self, session)

    def _matches_action_sequence(self, session: Session) -> bool:
        return matches_action_sequence(self, session)

    def _matches_action_text_terms(self, session: Session) -> bool:
        return matches_action_text_terms(self, session)

    def _apply_common_filters(
        self,
        items: builtins.list[_FilterableT],
        *,
        sql_pushed: bool,
    ) -> builtins.list[_FilterableT]:
        return apply_common_filters(self, items, sql_pushed=sql_pushed)

    def _apply_full_filters(self, sessions: list[Session], *, sql_pushed: bool) -> list[Session]:
        return apply_full_filters(self, sessions, sql_pushed=sql_pushed)

    def _sort_generic(self, items: list[_T], key_fn: Callable[[_T], SortKey]) -> list[_T]:
        return sort_generic(self, items, key_fn)

    def _sort_sessions(self, sessions: list[Session]) -> list[Session]:
        return sort_sessions(self, sessions)

    def _sort_summaries(self, summaries: list[SessionSummary]) -> list[SessionSummary]:
        return sort_summaries(self, summaries)

    def _finalize(self, items: list[_T]) -> list[_T]:
        return finalize_results(self, items)

    # -- Retrieval and execution methods (was QueryPlanExecutionMixin) --

    def _candidate_record_query(self) -> tuple[SessionRecordQuery, bool]:
        return candidate_record_query(self)

    def fetch_record_query(self) -> SessionRecordQuery:
        record_query, _ = self._candidate_record_query()
        return record_query.with_limit(self.effective_fetch_limit())

    def _uses_action_read_model(self) -> bool:
        return uses_actions(self)

    async def _actions_ready(self, repository: SessionQueryRuntimeStore) -> bool:
        return await actions_ready(self, repository)

    async def can_use_action_stats_with(self, repository: SessionQueryRuntimeStore) -> bool:
        return await can_use_action_stats_with(self, repository)

    async def _candidate_record_query_for(
        self,
        repository: SessionQueryRuntimeStore,
    ) -> tuple[SessionRecordQuery, bool]:
        return await candidate_record_query_for(self, repository)

    async def fetch_record_query_for(self, repository: SessionQueryRuntimeStore) -> SessionRecordQuery:
        return await fetch_record_query_for(self, repository)

    def _should_batch_post_filter_fetch(self) -> bool:
        return should_batch_post_filter_fetch(self)

    def _search_limit(self) -> int:
        return search_limit(self)

    async def list(self, config: Config) -> list[Session]:
        from polylogue.archive.query.archive_execution import list_archive

        return await list_archive(self, archive_root=_archive_root_for_config(config), config=config)

    async def list_summaries(self, config: Config) -> builtins.list[SessionSummary]:
        from polylogue.archive.query.archive_execution import list_summaries_archive

        return await list_summaries_archive(self, archive_root=_archive_root_for_config(config), config=config)

    async def first(self, config: Config) -> Session | None:
        from polylogue.archive.query.archive_execution import first_archive

        return await first_archive(self, archive_root=_archive_root_for_config(config), config=config)

    async def count(self, config: Config) -> int:
        from polylogue.archive.query.archive_execution import count_archive

        return await count_archive(self, archive_root=_archive_root_for_config(config), config=config)

    async def delete(self, config: Config) -> int:
        from polylogue.archive.query.archive_execution import delete_archive

        return await delete_archive(self, archive_root=_archive_root_for_config(config), config=config)


__all__ = ["SessionQueryPlan", "session_has_branches", "plan_record_query", "plan_sql_pushdown_params"]
