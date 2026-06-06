"""Description and sizing helpers for immutable session query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.query.fields import describe_plan_fields, plan_has_selection_filters

if TYPE_CHECKING:
    from polylogue.archive.query.plan import SessionQueryPlan


def describe_plan(plan: SessionQueryPlan) -> list[str]:
    return describe_plan_fields(plan)


def plan_has_filters(plan: SessionQueryPlan) -> bool:
    return plan_has_selection_filters(plan)


def effective_fetch_limit(plan: SessionQueryPlan) -> int | None:
    if plan.limit is None:
        return None
    if plan.has_post_filters():
        return max(plan.limit * 10, 500)
    if plan.sample is not None:
        return max(plan.sample * 3, 200)
    return max(plan.limit * 2, 2)


__all__ = [
    "describe_plan",
    "effective_fetch_limit",
    "plan_has_filters",
]
