"""Sorting and finalization helpers for immutable session query plans."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Generic, Protocol, TypeAlias, TypeVar

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
    key_fn: Callable[[_T], SortKey | tuple[SortKey, datetime, str]],
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

    def _ties(session: Session) -> tuple[datetime, str]:
        # SQL breaks every count order by ``sort_key_ms`` then ``session_id``,
        # in the same direction (``_summary_order_by``); a missing sort key
        # is SQLite's NULL, the smallest value.
        return session.updated_at or session.created_at or dt_min, str(session.id)

    if plan.sort == "tokens":
        # A composite key alone (unmeasured, total) can't share sort_generic's
        # single reversal: reversing would also flip "unmeasured" to sort
        # first. Sort unmeasured-last unconditionally, then by measured total
        # in the requested direction, exactly as the SQL ORDER BY does.
        scored = [(session, *_session_measured_tokens(session)) for session in sessions]
        measured = sorted(
            ((session, total) for session, unmeasured, total in scored if not unmeasured),
            key=lambda pair: (pair[1], *_ties(pair[0])),
            reverse=not plan.reverse,
        )
        unmeasured = sorted(
            (session for session, is_unmeasured, _total in scored if is_unmeasured),
            key=lambda session: (0, *_ties(session)),
            reverse=not plan.reverse,
        )
        return [session for session, _total in measured] + unmeasured

    def _key(session: Session) -> SortKey | tuple[SortKey, datetime, str]:
        if plan.sort == "date":
            return session.updated_at or dt_min
        if plan.sort == "messages":
            return (len(session.messages), *_ties(session))
        if plan.sort == "words":
            return (sum(message.word_count for message in session.messages), *_ties(session))
        if plan.sort == "longest":
            return (max((message.word_count for message in session.messages), default=0), *_ties(session))
        return session.updated_at or dt_min

    return sort_generic(plan, sessions, _key)


def sort_summaries(
    plan: QuerySortPlan,
    summaries: list[SessionSummary],
) -> list[SessionSummary]:
    dt_min = datetime.min.replace(tzinfo=timezone.utc)
    # Mirrors SQL's ``sort_key_ms = COALESCE(updated_at_ms, created_at_ms)``.
    return sort_generic(plan, summaries, lambda summary: summary.updated_at or summary.created_at or dt_min)


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
    "sort_generic",
    "sort_summaries",
]
