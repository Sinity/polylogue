"""Sorting and finalization helpers for immutable session query plans."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Generic, Protocol, TypeAlias, TypeVar

if TYPE_CHECKING:
    from polylogue.archive.models import Session, SessionSummary

from polylogue.archive.filter.types import SortField
from polylogue.core.timestamps import _aware_utc

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
    key_fn: Callable[[_T], SortKey | tuple[SortKey, ...]],
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


# SQLite stores these values without losing integer counters. The first key
# always places unmeasured tokens last; the remaining keys share direction.
SessionOrderValues: TypeAlias = tuple[bool, int | float, int, str]


def _time_value(value: datetime | None) -> int:
    reference = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = _aware_utc(value or datetime.min.replace(tzinfo=timezone.utc)) - reference
    return ((delta.days * 86400 + delta.seconds) * 1_000_000) + delta.microseconds


def compare_numeric_order_values(left: str, right: str) -> int:
    """Compare exact staged numbers without an integer bind limit or rounding."""
    lhs, rhs = Decimal(left), Decimal(right)
    return (lhs > rhs) - (lhs < rhs)


def session_order_values(plan: QuerySortPlan, session: Session) -> SessionOrderValues:
    """The exact comparison values shared by composed and staged SQL sorting."""
    ties = (_time_value(session.updated_at or session.created_at), str(session.id))
    if plan.sort == "tokens":
        unmeasured, total = _session_measured_tokens(session)
        return unmeasured, total, *ties
    if plan.sort == "messages":
        return False, len(session.messages), *ties
    if plan.sort == "words":
        return False, sum(message.word_count for message in session.messages), *ties
    if plan.sort == "longest":
        return False, max((message.word_count for message in session.messages), default=0), *ties
    if plan.sort == "random":
        return False, random.random(), 0, ""
    return False, _time_value(session.updated_at), 0, ""


def summary_order_values(
    plan: QuerySortPlan,
    summary: SessionSummary,
    *,
    metrics: tuple[int, int, int, bool, int] | None = None,
) -> SessionOrderValues:
    """Build summary ordering keys from declared values and optional transcript aggregates."""
    if plan.sort == "random":
        return False, random.random(), 0, ""
    if metrics is not None:
        message_count, word_count, longest, measured_tokens, tokens = metrics
        ties = (_time_value(summary.updated_at or summary.created_at), str(summary.id))
        if plan.sort == "messages":
            return False, message_count, *ties
        if plan.sort == "words":
            return False, word_count, *ties
        if plan.sort == "longest":
            return False, longest, *ties
        if plan.sort == "tokens":
            return not measured_tokens, tokens, *ties
    return False, _time_value(summary.updated_at or summary.created_at), 0, ""


def sort_sessions(plan: QuerySortPlan, sessions: list[Session]) -> list[Session]:
    if plan.sort == "random":
        return sort_generic(plan, sessions, lambda session: 0)
    if plan.sort != "tokens":
        return sorted(sessions, key=lambda session: session_order_values(plan, session)[1:], reverse=not plan.reverse)
    scored = [(session, session_order_values(plan, session)) for session in sessions]
    measured = sorted(
        (pair for pair in scored if not pair[1][0]), key=lambda pair: pair[1][1:], reverse=not plan.reverse
    )
    unmeasured = sorted((pair for pair in scored if pair[1][0]), key=lambda pair: pair[1][1:], reverse=not plan.reverse)
    return [session for session, _values in measured] + [session for session, _values in unmeasured]


def sort_summaries(plan: QuerySortPlan, summaries: list[SessionSummary]) -> list[SessionSummary]:
    return sort_generic(plan, summaries, lambda summary: summary_order_values(plan, summary)[1:])


class SessionReservoir(Generic[_T]):
    """A uniform random sample of fixed size over a stream (Algorithm R).

    Holds at most ``size`` items however many are offered, and draws the same
    distribution as ``random.sample`` over the whole stream.
    """

    def __init__(self, size: int) -> None:
        self._size = size
        self._seen = 0
        self._items: list[_T] = []

    def offer(self, items: Iterable[_T]) -> None:
        for item in items:
            self._seen += 1
            if len(self._items) < self._size:
                self._items.append(item)
            else:
                slot = random.randrange(self._seen)
                if slot < self._size:
                    self._items[slot] = item

    def items(self) -> list[_T]:
        sampled = list(self._items)
        random.shuffle(sampled)
        return sampled


class OffsetSampledPage(Generic[_T]):
    """The ``offset`` best items by sort, then a uniform sample of the rest.

    The shape a sampled page with an offset takes when every candidate is
    held: the offset drops the sort's head, and the sample draws from what
    remains. Holds ``offset + sample`` items however many are offered.
    """

    def __init__(self, *, offset: int, sample: int, sort: Callable[[list[_T]], list[_T]]) -> None:
        self._offset = offset
        self._sort = sort
        self._head: list[_T] = []
        self._rest: SessionReservoir[_T] = SessionReservoir(sample)

    def offer(self, items: Iterable[_T]) -> None:
        if not self._offset:
            self._rest.offer(items)
            return
        ranked = self._sort([*self._head, *items])
        # Each item enters the reservoir exactly once: when it is first seen
        # outside the head, or when a better item displaces it from the head.
        self._head, displaced = ranked[: self._offset], ranked[self._offset :]
        self._rest.offer(displaced)

    def items(self) -> list[_T]:
        return [*self._head, *self._rest.items()]


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
    "OffsetSampledPage",
    "SessionReservoir",
    "ResultWindow",
    "sort_sessions",
    "session_order_values",
    "summary_order_values",
    "SessionOrderValues",
    "compare_numeric_order_values",
    "sort_generic",
    "sort_summaries",
]
