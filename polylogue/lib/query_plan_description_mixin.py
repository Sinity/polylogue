"""Description and record-query mixin methods for conversation plans."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from polylogue.lib.query_plan_description import describe_plan, effective_fetch_limit, plan_has_filters
from polylogue.lib.query_plan_records import plan_record_query, plan_sql_pushdown_params

if TYPE_CHECKING:
    from polylogue.lib.query_plan import ConversationQueryPlan
    from polylogue.storage.query_models import ConversationRecordQuery


class QueryPlanDescriptionMixin:
    @property
    def fts_terms(self) -> tuple[str, ...]:
        return self.query_terms + self.contains_terms

    @property
    def sql_pushed(self) -> bool:
        return not self.fts_terms and self.conversation_id is None

    @property
    def record_query(self) -> ConversationRecordQuery:
        return plan_record_query(self)

    def sql_pushdown_params(self) -> dict[str, object]:
        return plan_sql_pushdown_params(self)

    def describe(self) -> list[str]:
        return describe_plan(self)

    def has_filters(self) -> bool:
        return plan_has_filters(self)

    def effective_fetch_limit(self) -> int | None:
        return effective_fetch_limit(self)

    def with_limit(self, limit: int | None) -> ConversationQueryPlan:
        return replace(self, limit=limit)


__all__ = ["QueryPlanDescriptionMixin"]
