"""Sorting and finalization helpers for immutable session query plans."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar

if TYPE_CHECKING:
    from polylogue.archive.models import Session, SessionSummary

from polylogue.archive.filter.types import SortField

_T = TypeVar("_T")
SortKey: TypeAlias = datetime | float | int | str


class QuerySortPlan(Protocol):
    """Minimal plan surface required by result sorting/finalization."""

    @property
    def sort(self) -> SortField | None: ...

    @property
    def reverse(self) -> bool: ...

    @property
    def limit(self) -> int | None: ...

    @property
    def sample(self) -> int | None: ...


@dataclass(frozen=True, slots=True)
class ResultWindow:
    """Typed sampling and limit settings for finalized query results."""

    sample: int | None = None
    limit: int | None = None

    @classmethod
    def from_plan(cls, plan: QuerySortPlan) -> ResultWindow:
        return cls(sample=plan.sample, limit=plan.limit)


def sort_generic(
    plan: QuerySortPlan,
    items: list[_T],
    key_fn: Callable[[_T], SortKey],
) -> list[_T]:
    if plan.sort == "random":
        shuffled = list(items)
        random.shuffle(shuffled)
        return shuffled
    return sorted(items, key=key_fn, reverse=not plan.reverse)


def _session_measured_tokens(session: Session) -> tuple[bool, int]:
    """Mirrors SQL's ``tokens`` order key: ``(unmeasured, summed_tokens)``.

    A session none of whose messages carries any token counter has an
    UNKNOWN total, not a measured zero (``_summary_order_by`` in
    ``storage/sqlite/archive_tiers/archive.py``, polylogue-qgyuj); ranking it
    by an estimated text-length proxy instead can put it ahead of a
    genuinely-measured session on ``sort=tokens``. ``unmeasured`` sorts last
    in both directions, exactly as SQL's leading ``NOT EXISTS`` key does.
    """
    measured = any(
        message.input_tokens is not None
        or message.output_tokens is not None
        or message.cache_read_tokens is not None
        or message.cache_write_tokens is not None
        for message in session.messages
    )
    total = sum(
        (message.input_tokens or 0)
        + (message.output_tokens or 0)
        + (message.cache_read_tokens or 0)
        + (message.cache_write_tokens or 0)
        for message in session.messages
    )
    return not measured, total


def sort_sessions(
    plan: QuerySortPlan,
    sessions: list[Session],
) -> list[Session]:
    dt_min = datetime.min.replace(tzinfo=timezone.utc)

    if plan.sort == "tokens":
        # A composite key alone (unmeasured, total) can't share sort_generic's
        # single reversal: reversing would also flip "unmeasured" to sort
        # first. Sort unmeasured-last unconditionally, then by measured total
        # in the requested direction, exactly as the SQL ORDER BY does.
        scored = [(session, *_session_measured_tokens(session)) for session in sessions]
        measured = sorted(
            ((session, total) for session, unmeasured, total in scored if not unmeasured),
            key=lambda pair: pair[1],
            reverse=not plan.reverse,
        )
        unmeasured = [session for session, is_unmeasured, _total in scored if is_unmeasured]
        return [session for session, _total in measured] + unmeasured

    def _key(session: Session) -> SortKey:
        if plan.sort == "date":
            return session.updated_at or dt_min
        if plan.sort == "messages":
            return len(session.messages)
        if plan.sort == "words":
            return sum(message.word_count for message in session.messages)
        if plan.sort == "longest":
            return max((message.word_count for message in session.messages), default=0)
        return session.updated_at or dt_min

    return sort_generic(plan, sessions, _key)


def sort_summaries(
    plan: QuerySortPlan,
    summaries: list[SessionSummary],
) -> list[SessionSummary]:
    dt_min = datetime.min.replace(tzinfo=timezone.utc)
    # Mirrors SQL's ``sort_key_ms = COALESCE(updated_at_ms, created_at_ms)``.
    return sort_generic(plan, summaries, lambda summary: summary.updated_at or summary.created_at or dt_min)


def finalize_results(
    plan: QuerySortPlan,
    items: list[_T],
) -> list[_T]:
    return finalize_window(ResultWindow.from_plan(plan), items)


def finalize_window(
    window: ResultWindow,
    items: list[_T],
) -> list[_T]:
    """Apply a typed result window to already sorted query results."""
    results = list(items)
    if window.sample is not None and window.sample < len(results):
        results = random.sample(results, window.sample)
    if window.limit is not None:
        results = results[: window.limit]
    return results


__all__ = [
    "finalize_results",
    "finalize_window",
    "QuerySortPlan",
    "ResultWindow",
    "sort_sessions",
    "sort_generic",
    "sort_summaries",
]
