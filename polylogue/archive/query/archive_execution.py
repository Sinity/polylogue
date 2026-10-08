"""Execution of session-query plans over ``ArchiveStore``.

The fluent :class:`~polylogue.archive.filter.filters.SessionFilter` no
longer has a competing storage route. Its five terminal operations
(``list``/``list_summaries``/``first``/``count``/``delete``) translate the
canonical :class:`~polylogue.archive.query.plan.SessionQueryPlan` into the
``ArchiveStore`` filter kwarg set, fetch session summaries/envelopes from
the archive, and re-apply the plan's residual post-filters
(topology, predicates, negative terms, action sequences) that the SQL layer
does not push down.
"""

from __future__ import annotations

import builtins
import sqlite3
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import ExitStack, closing, contextmanager
from dataclasses import replace
from itertools import islice
from typing import TYPE_CHECKING, Literal, TypeVar, cast

from polylogue.archive.hydration import archive_envelope_to_session, archive_summary_to_domain
from polylogue.archive.query.filter_kwargs import (
    plan_filter_kwargs,
)
from polylogue.archive.query.search_contract import ArchiveSearchResult, LaneFailure, LaneName, SearchExecution
from polylogue.archive.query.sorting import OffsetSampledPage, session_order_values, summary_order_values
from polylogue.archive.query.spec import DEFAULT_SESSION_LIST_LIMIT
from polylogue.archive.query.transaction import archive_read_context, run_archive_read
from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.core.errors import EmbeddingRetrievalNotReadyError

_AttachableT = TypeVar("_AttachableT", Session, SessionSummary)

if TYPE_CHECKING:
    from pathlib import Path

    from polylogue.archive.query.expression import WithUnitWindow
    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.config import Config
    from polylogue.storage.sqlite.archive_tiers.archive import (
        ArchiveSessionSearchHit,
        ArchiveSessionSummary,
        ArchiveStore,
    )


def _plan_text_query(plan: SessionQueryPlan) -> str | None:
    terms = (*plan.query_terms, *plan.contains_terms)
    text = " ".join(term for term in terms if term).strip()
    return text or None


def _fetch_limit(plan: SessionQueryPlan, *, default: int) -> int:
    limit = plan.limit if plan.limit is not None else default
    if plan.has_post_filters():
        # Post-filters discard rows after fetch. The caller repeats this
        # bounded batch until the filtered page is full or candidates end.
        return max(limit * 10, 500) if limit < 1_000_000 else limit
    return limit


#: Sorts over per-session counters, which the index stores for a lineage
#: child's own tail only.
_COMPOSED_COUNT_SORTS = frozenset({"messages", "words", "longest", "tokens"})
#: Sessions hydrated at once while a complete composed sort selects its page.
_COMPOSED_SORT_CHUNK = 200


def _ranked_window(plan: SessionQueryPlan) -> bool:
    """Whether results come from the complete scoped ranked session relation."""
    return (
        plan.similar_text is not None
        or plan.similar_session_id is not None
        or plan.retrieval_lane in {"semantic", "hybrid"}
    )


def _archive_summaries(
    plan: SessionQueryPlan,
    archive: ArchiveStore,
    *,
    config: Config | None,
    archive_root: Path,
    default_limit: int,
    keep: Callable[[list[ArchiveSessionSummary]], list[ArchiveSessionSummary]] | None = None,
    complete: bool = False,
    on_batch: Callable[[list[ArchiveSessionSummary]], None] | None = None,
    full_sort: bool = False,
) -> list[ArchiveSessionSummary]:
    """Fetch the candidate rows for ``plan`` in the archive's SQL order.

    SQL is the one ordering authority for lexical and structured queries:
    callers do not re-sort these rows. That is what makes a post-filtered
    page exact without reading the whole archive -- ``keep`` applies the
    residual filters to each fetched batch, and fetching stops as soon as
    ``offset + limit`` rows have passed them, since no later row can precede
    one already kept (polylogue-6xrab, polylogue-ztm1t).

    Every returned row has passed ``keep`` exactly once, on every route, so a
    caller that supplies it does not filter again. ``complete`` pages through
    the whole candidate set for a caller that orders it itself.

    With ``on_batch``, each fetched batch that passed ``keep`` is handed to it
    as it arrives and nothing is accumulated (the result is empty): a
    complete scan feeding a bounded reducer holds one batch at a time.
    """

    def deliver(rows: list[ArchiveSessionSummary]) -> list[ArchiveSessionSummary]:
        if on_batch is None:
            return rows
        on_batch(rows)
        return []

    filter_kwargs = plan_filter_kwargs(plan)
    limit = _fetch_limit(plan, default=default_limit)
    # A post-filtered plan pages its candidates from offset zero and applies
    # the offset once, over survivors, in the caller. An unlimited plan takes
    # the same route: a SQL offset here would skip unfiltered candidates and
    # the caller would then skip survivors again.
    post_filter_fetch = plan.has_post_filters() or complete
    wanted = None if plan.limit is None or plan.sample is not None else plan.offset + plan.limit
    sort = plan.sort
    reverse = plan.reverse

    if _ranked_window(plan):
        try:
            with (
                _scoped_ranked_hits(
                    plan,
                    archive,
                    config=config,
                    archive_root=archive_root,
                    keep=keep,
                ) as (ranked_hits, _execution),
                _ordered_scoped_hits(
                    plan,
                    archive,
                    ranked_hits,
                    full=full_sort,
                ) as hits,
            ):
                wanted_ranked = None if complete or plan.sample or plan.limit is None else plan.offset + plan.limit
                selected = hits if wanted_ranked is None else islice(hits, wanted_ranked)
                result: list[ArchiveSessionSummary] = []
                while batch := list(islice(selected, _COMPOSED_SORT_CHUNK)):
                    archive.check_operation_read()
                    result.extend(deliver(_summaries_from_hits(archive, batch)))
                return result
        except EmbeddingRetrievalNotReadyError as exc:
            if plan.similar_session_id is not None:
                from polylogue.archive.query.expression import ExpressionCompileError

                raise ExpressionCompileError(str(exc), field="near") from exc
            raise

    query_text = _plan_text_query(plan)
    if query_text is not None:
        if post_filter_fetch and sort == "random":
            random_kept_summaries: list[ArchiveSessionSummary] = []
            random_kept_count = 0
            with closing(
                archive.iter_session_identities(
                    query=query_text,
                    actions_only=plan.retrieval_lane == "actions",
                    limit=None,
                    offset=0,
                    sort="random",
                    reverse=reverse,
                    **filter_kwargs,
                )
            ) as random_candidates:
                while identity_batch := list(islice(random_candidates, limit)):
                    rows: list[ArchiveSessionSummary] = []
                    for identity in identity_batch:
                        try:
                            rows.append(archive.read_summary(identity.session_id))
                        except KeyError:
                            continue
                    kept_rows = keep(rows) if keep is not None else rows
                    random_kept_count += len(kept_rows)
                    random_kept_summaries.extend(deliver(kept_rows))
                    if wanted is not None and random_kept_count >= wanted:
                        break
            return random_kept_summaries
        if not post_filter_fetch:
            return deliver(
                _kept(
                    keep,
                    _summaries_from_hits(
                        archive,
                        archive.search_summaries(
                            query_text, limit=limit, offset=plan.offset, sort=sort, reverse=reverse, **filter_kwargs
                        ),
                    ),
                )
            )
        kept_hits: list[ArchiveSessionSummary] = []
        kept_count = 0
        seen: set[str] = set()
        fetch_offset = 0
        while True:
            batch = archive.search_summaries(
                query_text,
                limit=limit,
                offset=fetch_offset,
                sort=sort,
                reverse=reverse,
                **filter_kwargs,
            )
            fresh = [hit for hit in batch if hit.session_id not in seen]
            seen.update(hit.session_id for hit in fresh)
            rows = _summaries_from_hits(archive, fresh)
            kept_rows = keep(rows) if keep is not None else rows
            kept_count += len(kept_rows)
            kept_hits.extend(deliver(kept_rows))
            if len(batch) < limit or (keep is not None and wanted is not None and kept_count >= wanted):
                break
            fetch_offset += len(batch)
        return kept_hits

    if not post_filter_fetch:
        return deliver(
            _kept(
                keep,
                archive.list_summaries(
                    limit=limit,
                    offset=plan.offset,
                    sort=sort,
                    reverse=reverse,
                    sample=plan.sample is not None,
                    **filter_kwargs,
                ),
            )
        )
    if sort == "random":
        random_summaries: list[ArchiveSessionSummary] = []
        random_summary_count = 0
        with closing(
            archive.iter_summaries(
                limit=None,
                offset=0,
                sort="random",
                reverse=reverse,
                sample=False,
                **filter_kwargs,
            )
        ) as random_summary_candidates:
            while summary_random_batch := list(islice(random_summary_candidates, limit)):
                kept_summary_random_batch = keep(summary_random_batch) if keep is not None else summary_random_batch
                random_summary_count += len(kept_summary_random_batch)
                random_summaries.extend(deliver(kept_summary_random_batch))
                if wanted is not None and random_summary_count >= wanted:
                    break
        return random_summaries
    summaries: list[ArchiveSessionSummary] = []
    summary_count = 0
    fetch_offset = 0
    while True:
        summary_batch = archive.list_summaries(
            limit=limit,
            offset=fetch_offset,
            sort=sort,
            reverse=reverse,
            sample=False,
            **filter_kwargs,
        )
        kept_batch = keep(summary_batch) if keep is not None else summary_batch
        summary_count += len(kept_batch)
        summaries.extend(deliver(kept_batch))
        if len(summary_batch) < limit or (keep is not None and wanted is not None and summary_count >= wanted):
            break
        fetch_offset += len(summary_batch)
    return summaries


def _kept(
    keep: Callable[[list[ArchiveSessionSummary]], list[ArchiveSessionSummary]] | None,
    rows: list[ArchiveSessionSummary],
) -> list[ArchiveSessionSummary]:
    return keep(rows) if keep is not None else rows


def _summaries_from_hits(archive: ArchiveStore, hits: list[ArchiveSessionSearchHit]) -> list[ArchiveSessionSummary]:
    summaries: list[ArchiveSessionSummary] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.session_id in seen:
            continue
        seen.add(hit.session_id)
        try:
            summaries.append(archive.read_summary(hit.session_id))
        except KeyError:
            continue
    return summaries


def _attach_units_to_domain(
    items: builtins.list[_AttachableT],
    archive: ArchiveStore,
    with_units: tuple[str, ...],
    with_unit_fields: dict[str, tuple[str, ...]] | None = None,
    with_unit_windows: Mapping[str, WithUnitWindow] | None = None,
    page_width: int | None = None,
) -> builtins.list[_AttachableT]:
    """Attach ``with <units>`` projection rows onto domain models (#2492).

    Returns updated copies carrying ``attached_units`` (pydantic models are
    treated as immutable for safety). A no-op when no units are requested.

    ``SessionSummary``/``Session`` carry rows, not an outcome envelope, so a
    bounded projection cannot degrade a terminal outcome on this route the way
    it can on the daemon operation. It is emitted as a named degraded event
    instead -- the same disposition ``archive.postmortem_bundle.truncated``
    uses -- rather than dropped, which is what made the cut invisible.
    """

    if not with_units or not items:
        return items
    from polylogue.archive.query.attached_units import fetch_attached_units
    from polylogue.logging import WARNING, emit

    session_ids = [item.id for item in items]
    attached = fetch_attached_units(
        archive,
        session_ids,
        with_units,
        unit_fields=with_unit_fields,
        unit_windows=with_unit_windows,
        page_width=page_width,
    )
    for gap in attached.gaps:
        emit(
            "archive.attached_units.truncated",
            level=WARNING,
            outcome="degraded",
            reason=gap,
            sessions=len(session_ids),
            domain=",".join(with_units),
        )
    updated: builtins.list[_AttachableT] = []
    for item in items:
        per_session = {unit: tuple(by_session.get(item.id, ())) for unit, by_session in attached.rows.items()}
        updated.append(item.model_copy(update={"attached_units": per_session}))
    return updated


def _list_summaries_in_archive(
    plan: SessionQueryPlan,
    archive: ArchiveStore,
    *,
    config: Config | None,
    archive_root: Path,
    default_limit: int,
    with_units: tuple[str, ...] = (),
    with_unit_fields: dict[str, tuple[str, ...]] | None = None,
    with_unit_windows: Mapping[str, WithUnitWindow] | None = None,
) -> builtins.list[SessionSummary]:
    """Execute the canonical summary plan on a caller's pinned archive read."""

    def keep_matching(rows: list[ArchiveSessionSummary]) -> list[ArchiveSessionSummary]:
        by_id = {row.session_id: row for row in rows}
        if plan.can_use_summaries():
            matching = plan._apply_common_filters([archive_summary_to_domain(row) for row in rows], sql_pushed=True)
            matching_ids = {str(summary.id) for summary in matching}
        else:
            sessions = [
                archive_envelope_to_session(
                    archive.read_session(row.session_id),
                    display_label=row.display_label,
                    display_label_source=row.display_label_source,
                )
                for row in rows
            ]
            matching_ids = {str(session.id) for session in plan._apply_full_filters(sessions, sql_pushed=True)}
        return [row for session_id, row in by_id.items() if session_id in matching_ids]

    reduce_order = _ranked_window(plan) and plan.sample is not None
    best: list[SessionSummary] = []
    bound = plan.offset + (plan.limit if plan.limit is not None else default_limit)
    reservoir = (
        OffsetSampledPage(offset=plan.offset, sample=plan.sample, sort=plan._sort_summaries) if plan.sample else None
    )

    def reduce_batch(rows: list[ArchiveSessionSummary]) -> None:
        nonlocal best
        summaries = [archive_summary_to_domain(row) for row in rows]
        if reservoir is not None:
            reservoir.offer(summaries)
            best = reservoir.items()
        else:
            best = plan._sort_summaries([*best, *summaries])[:bound]

    archive_rows = _archive_summaries(
        plan,
        archive,
        config=config,
        archive_root=archive_root,
        default_limit=default_limit,
        keep=keep_matching if plan.has_post_filters() else None,
        complete=reduce_order,
        on_batch=reduce_batch if reduce_order else None,
    )
    summaries = _attach_units_to_domain(
        best if reduce_order else [archive_summary_to_domain(row) for row in archive_rows],
        archive,
        with_units,
        with_unit_fields,
        with_unit_windows,
    )
    filtered = summaries if plan.has_post_filters() else plan._apply_common_filters(summaries, sql_pushed=True)
    ranked_window = _ranked_window(plan)
    ordered = filtered
    if (plan.has_post_filters() or ranked_window) and plan.offset:
        ordered = ordered[plan.offset :]
    return plan._finalize(ordered)


async def list_summaries_archive(
    plan: SessionQueryPlan,
    *,
    archive_root: Path,
    config: Config | None,
    default_limit: int = DEFAULT_SESSION_LIST_LIMIT,
    with_units: tuple[str, ...] = (),
    with_unit_fields: dict[str, tuple[str, ...]] | None = None,
    with_unit_windows: Mapping[str, WithUnitWindow] | None = None,
) -> builtins.list[SessionSummary]:
    def read(archive: ArchiveStore) -> list[SessionSummary]:
        return _list_summaries_in_archive(
            plan,
            archive,
            config=config,
            archive_root=archive_root,
            default_limit=default_limit,
            with_units=with_units,
            with_unit_fields=with_unit_fields,
            with_unit_windows=with_unit_windows,
        )

    return await run_archive_read(
        archive_root,
        operation="archive.query.list-summaries",
        arguments={"plan": plan, "default_limit": default_limit, "with_units": with_units},
        work=read,
        page_size=plan.limit,
        offset=plan.offset,
        projection="session-summaries",
        workload_class="scan" if _ranked_window(plan) or plan.limit is None or plan.limit > 1000 else "interactive",
    )


def _list_sessions_in_archive(
    plan: SessionQueryPlan,
    archive: ArchiveStore,
    *,
    archive_root: Path,
    config: Config | None,
    default_limit: int = DEFAULT_SESSION_LIST_LIMIT,
    with_units: tuple[str, ...] = (),
    with_unit_fields: dict[str, tuple[str, ...]] | None = None,
    with_unit_windows: Mapping[str, WithUnitWindow] | None = None,
) -> builtins.list[Session]:
    # Stored counters describe a lineage child's own divergent tail, while the
    # returned Session recomposes its inherited prefix. A count-ordered page of
    # full sessions is therefore ordered over the composed sessions, from the
    # unwindowed candidate set, instead of trusting the tail-only SQL keys.
    composed_order = plan.sort in _COMPOSED_COUNT_SORTS
    ranked_window = _ranked_window(plan)
    # SQL-backed routes page through the whole candidate set for a composed
    # sort. Ranked routes settle every qualifying session before reducing
    # an explicit sort or sample to the requested page.
    complete = (not ranked_window and composed_order) or (ranked_window and plan.sample is not None)
    fetch_plan = replace(plan, limit=None, offset=0) if complete else plan
    # Units are projected at most one result page at a time (a candidate
    # batch, ten pages wide under post-filters, exceeds the projector's row
    # budget), and always with the allowance of a full requested page. A
    # predicate therefore sees exactly the rows the served session carries,
    # however its candidate chunk or served page happens to be filled.
    served = (
        plan.limit
        if plan.limit is not None and plan.limit > 0
        # A sampled page with no limit serves its whole sample.
        else plan.sample
        if plan.sample
        # A complete composed sort serves the default page; its units get
        # that page's allowance, not a candidate chunk's.
        else (default_limit if complete else None)
    )
    # A sampled page serves at most the sample, whatever the limit.
    unit_page = min(served, plan.sample) if served is not None and plan.sample else served

    def attach(archive: ArchiveStore, sessions: list[Session]) -> list[Session]:
        width = unit_page or max(len(sessions), 1)
        attached: list[Session] = []
        for start in range(0, len(sessions), width):
            attached.extend(
                _attach_units_to_domain(
                    sessions[start : start + width],
                    archive,
                    with_units,
                    with_unit_fields,
                    with_unit_windows,
                    page_width=unit_page,
                )
            )
        return attached

    def hydrate(archive: ArchiveStore, rows: list[ArchiveSessionSummary]) -> list[Session]:
        return [
            archive_envelope_to_session(
                archive.read_session(summary.session_id),
                display_label=summary.display_label,
                display_label_source=summary.display_label_source,
            )
            for summary in rows
        ]

    def read(archive: ArchiveStore) -> list[Session]:
        # Each candidate is hydrated and filtered once; the survivors are
        # kept as hydrated, so no predicate runs twice for one session.
        kept_sessions: dict[str, Session] = {}
        # A complete composed sort keeps only the best ``offset + limit``
        # hydrated sessions seen so far: a one-row page over a large archive
        # must not hold every recomposed transcript at once.
        # A sampled page draws uniformly from every qualified candidate
        # through a reservoir of the sample's size, so neither kind of page
        # holds more sessions than it can serve. An omitted limit is the
        # default page.
        page_width = plan.limit if plan.limit is not None else default_limit
        bound = (plan.offset or 0) + page_width
        best: list[Session] = []
        reservoir: OffsetSampledPage[Session] | None = (
            OffsetSampledPage(offset=plan.offset or 0, sample=plan.sample, sort=plan._sort_sessions)
            if plan.sample
            else None
        )

        def retain(sessions: list[Session]) -> None:
            nonlocal best
            if reservoir is not None:
                reservoir.offer(sessions)
                best = reservoir.items()
                return
            best = plan._sort_sessions([*best, *sessions])[:bound]

        def keep(rows: list[ArchiveSessionSummary]) -> list[ArchiveSessionSummary]:
            # Predicates see the same fully hydrated Session the caller gets:
            # display label and requested units included.
            if complete and not ranked_window:
                # Hydrated a chunk at a time: a candidate page may be wider
                # than what the sort should hold.
                survivor_ids: set[str] = set()
                for start in range(0, len(rows), _COMPOSED_SORT_CHUNK):
                    chunk = rows[start : start + _COMPOSED_SORT_CHUNK]
                    chunk_survivors = plan._apply_full_filters(
                        attach(archive, hydrate(archive, chunk)), sql_pushed=True
                    )
                    retain(chunk_survivors)
                    survivor_ids.update(str(session.id) for session in chunk_survivors)
                return [row for row in rows if row.session_id in survivor_ids]
            survivors = plan._apply_full_filters(attach(archive, hydrate(archive, rows)), sql_pushed=True)
            survivor_ids = {str(session.id) for session in survivors}
            if not ranked_window:
                for session in survivors:
                    kept_sessions[str(session.id)] = session
            return [row for row in rows if row.session_id in survivor_ids]

        filtering = plan.has_post_filters()

        def reduce_batch(rows: list[ArchiveSessionSummary]) -> None:
            # A complete scan streams its candidates into the bounded
            # reducer; a filtered one already retained its survivors in
            # ``keep``.
            if filtering and not ranked_window:
                return
            for start in range(0, len(rows), _COMPOSED_SORT_CHUNK):
                chunk = rows[start : start + _COMPOSED_SORT_CHUNK]
                sessions = hydrate(archive, chunk)
                retain(sessions if ranked_window else plan._apply_full_filters(sessions, sql_pushed=True))

        archive_rows = _archive_summaries(
            fetch_plan,
            archive,
            config=config,
            archive_root=archive_root,
            default_limit=default_limit,
            keep=keep if filtering else None,
            complete=complete,
            on_batch=reduce_batch if complete else None,
            full_sort=True,
        )
        if complete:
            ordered = best
        else:
            if filtering:
                candidates = (
                    attach(archive, hydrate(archive, archive_rows))
                    if ranked_window
                    else [kept_sessions[row.session_id] for row in archive_rows]
                )
            else:
                candidates = plan._apply_full_filters(hydrate(archive, archive_rows), sql_pushed=True)
            ordered = plan._sort_sessions(candidates) if composed_order and not ranked_window else candidates
        if (complete or filtering or ranked_window) and plan.offset:
            ordered = ordered[plan.offset :]
        # Filtered survivors already carry the page-width projection their
        # predicate saw; unfiltered sessions are projected over the served
        # page only.
        if complete and plan.limit is None and not plan.sample:
            ordered = ordered[:default_limit]
        page = plan._finalize(ordered)
        return attach(archive, page) if complete and ranked_window else page if filtering else attach(archive, page)

    return read(archive)


async def list_archive(
    plan: SessionQueryPlan,
    *,
    archive_root: Path,
    config: Config | None,
    default_limit: int = DEFAULT_SESSION_LIST_LIMIT,
    with_units: tuple[str, ...] = (),
    with_unit_fields: dict[str, tuple[str, ...]] | None = None,
    with_unit_windows: Mapping[str, WithUnitWindow] | None = None,
) -> builtins.list[Session]:
    ranked_window = _ranked_window(plan)
    composed_order = plan.sort in _COMPOSED_COUNT_SORTS
    complete = (not ranked_window and composed_order) or (ranked_window and plan.sample is not None)
    return await run_archive_read(
        archive_root,
        operation="archive.query.list",
        arguments={"plan": plan, "default_limit": default_limit, "with_units": with_units},
        work=lambda archive: _list_sessions_in_archive(
            plan,
            archive,
            config=config,
            archive_root=archive_root,
            default_limit=default_limit,
            with_units=with_units,
            with_unit_fields=with_unit_fields,
            with_unit_windows=with_unit_windows,
        ),
        page_size=plan.limit,
        offset=plan.offset,
        projection="sessions",
        # A complete composed sort reads and hydrates every candidate,
        # whatever the requested page size.
        workload_class="scan"
        if ranked_window or complete or plan.limit is None or plan.limit > 1000
        else "interactive",
    )


async def first_archive(
    plan: SessionQueryPlan,
    *,
    archive_root: Path,
    config: Config | None,
) -> Session | None:
    results = await list_archive(plan.with_limit(1), archive_root=archive_root, config=config)
    return results[0] if results else None


async def count_archive(
    plan: SessionQueryPlan,
    *,
    archive_root: Path,
    config: Config | None,
) -> int:
    if (
        not plan.has_post_filters()
        and plan.similar_text is None
        and plan.similar_session_id is None
        and plan.retrieval_lane not in {"semantic", "hybrid"}
    ):
        filter_kwargs = plan_filter_kwargs(plan)
        query_text = _plan_text_query(plan)
        with archive_read_context(
            archive_root,
            operation="archive.query.count",
            arguments={"plan": plan},
            page_size=1,
            projection="count",
            workload_class="scan",
        ) as archive:
            if query_text is not None:
                return int(
                    archive.count_search_sessions(
                        query_text,
                        actions_only=plan.retrieval_lane == "actions",
                        **filter_kwargs,
                    )
                )
            return int(archive.count_sessions(**filter_kwargs))

    if _ranked_window(plan):
        unbounded = replace(plan, limit=None, offset=0)

        def read(archive: ArchiveStore) -> int:
            total = 0

            def count_batch(rows: list[ArchiveSessionSummary]) -> None:
                nonlocal total
                total += len(rows)

            _archive_summaries(
                unbounded,
                archive,
                config=config,
                archive_root=archive_root,
                default_limit=DEFAULT_SESSION_LIST_LIMIT,
                complete=True,
                on_batch=count_batch,
            )
            return min(total, plan.sample) if plan.sample is not None else total

        return await run_archive_read(
            archive_root,
            operation="archive.query.count",
            arguments={"plan": plan},
            work=read,
            page_size=1,
            projection="count",
            workload_class="scan",
        )

    # A count is the size of the whole result, not of one page: the SQL
    # count above ignores the window, so this route drops the offset too.
    unbounded = replace(plan, limit=None, offset=0)
    if unbounded.can_use_summaries():
        rows = await list_summaries_archive(
            unbounded,
            archive_root=archive_root,
            config=config,
            default_limit=1_000_000,
        )
        return len(rows)
    sessions = await list_archive(
        unbounded,
        archive_root=archive_root,
        config=config,
        default_limit=1_000_000,
    )
    return len(sessions)


def _count_in_archive(
    plan: SessionQueryPlan,
    archive: ArchiveStore,
    *,
    config: Config | None,
    archive_root: Path,
    default_limit: int = DEFAULT_SESSION_LIST_LIMIT,
) -> int:
    """Count the canonical query scope on a caller's pinned archive read."""
    if (
        not plan.has_post_filters()
        and plan.similar_text is None
        and plan.similar_session_id is None
        and plan.retrieval_lane not in {"semantic", "hybrid"}
    ):
        filter_kwargs = plan_filter_kwargs(plan)
        query_text = _plan_text_query(plan)
        if query_text is not None:
            return int(
                archive.count_search_sessions(
                    query_text,
                    actions_only=plan.retrieval_lane == "actions",
                    **filter_kwargs,
                )
            )
        return int(archive.count_sessions(**filter_kwargs))

    unbounded = replace(plan, limit=None, offset=0)
    if _ranked_window(plan):
        total = 0

        def count_ranked_batch(rows: list[ArchiveSessionSummary]) -> None:
            nonlocal total
            total += len(rows)

        _archive_summaries(
            unbounded,
            archive,
            config=config,
            archive_root=archive_root,
            default_limit=default_limit,
            complete=True,
            on_batch=count_ranked_batch,
        )
        return min(total, plan.sample) if plan.sample is not None else total

    total = 0

    def keep(rows: list[ArchiveSessionSummary]) -> list[ArchiveSessionSummary]:
        if unbounded.can_use_summaries():
            matching = unbounded._apply_common_filters(
                [archive_summary_to_domain(row) for row in rows], sql_pushed=True
            )
            survivor_ids = {str(session.id) for session in matching}
        else:
            sessions = [
                archive_envelope_to_session(
                    archive.read_session(row.session_id),
                    display_label=row.display_label,
                    display_label_source=row.display_label_source,
                )
                for row in rows
            ]
            survivor_ids = {str(session.id) for session in unbounded._apply_full_filters(sessions, sql_pushed=True)}
        return [row for row in rows if row.session_id in survivor_ids]

    def count_batch(rows: list[ArchiveSessionSummary]) -> None:
        nonlocal total
        total += len(rows)

    _archive_summaries(
        unbounded,
        archive,
        config=config,
        archive_root=archive_root,
        default_limit=default_limit,
        keep=keep if unbounded.has_post_filters() else None,
        complete=True,
        on_batch=count_batch,
    )
    return total


def _fts_lane_candidates(
    archive: ArchiveStore,
    plan: SessionQueryPlan,
    *,
    text: str,
    limit: int,
    actions_only: bool,
) -> list[ArchiveSessionSearchHit]:
    """Read block hits over one cursor until ``limit`` distinct sessions are seen.

    Block hits repeat their session, so a SQL ``LIMIT`` over blocks cannot
    bound distinct sessions; the stream stops as soon as the page is full.
    """
    candidates: dict[str, ArchiveSessionSearchHit] = {}
    if limit <= 0:
        return []
    with closing(
        archive.iter_search_summaries(
            text,
            limit=None,
            actions_only=actions_only,
            sort=plan.sort,
            reverse=plan.reverse,
            **plan_filter_kwargs(plan),
        )
    ) as hits:
        for hit in hits:
            candidates.setdefault(hit.session_id, hit)
            if len(candidates) == limit:
                break
    return list(candidates.values())


class _VectorTraversalError(Exception):
    """A provider iterator fault, distinct from an archive witness read fault."""


def _qualified_scope(
    plan: SessionQueryPlan,
    archive: ArchiveStore,
    keep: Callable[[list[ArchiveSessionSummary]], list[ArchiveSessionSummary]] | None,
) -> Generator[str, None, None]:
    with closing(archive.iter_summaries(limit=None, **plan_filter_kwargs(plan))) as source:
        while batch := list(islice(source, _COMPOSED_SORT_CHUNK)):
            archive.check_operation_read()
            if keep is not None:
                qualified = keep(batch)
            elif plan.has_post_filters():
                if plan.can_use_summaries():
                    summary_survivors = plan._apply_common_filters(
                        [archive_summary_to_domain(row) for row in batch],
                        sql_pushed=True,
                    )
                    surviving = {str(row.id) for row in summary_survivors}
                else:
                    session_survivors = plan._apply_full_filters(
                        [
                            archive_envelope_to_session(
                                archive.read_session(row.session_id),
                                display_label=row.display_label,
                                display_label_source=row.display_label_source,
                            )
                            for row in batch
                        ],
                        sql_pushed=True,
                    )
                    surviving = {str(row.id) for row in session_survivors}
                qualified = [row for row in batch if row.session_id in surviving]
            else:
                qualified = batch
            yield from (row.session_id for row in qualified)


def _vector_witnesses(archive: ArchiveStore, rows: Iterator[tuple[str, float]]) -> Iterator[ArchiveSessionSearchHit]:
    from polylogue.archive.query.execution_control import (
        QueryCancelledError,
        QueryTimeoutError,
        QueryWorkBudgetExceededError,
    )
    from polylogue.storage.sqlite.connection_profile import ReadFrameCancelledError, ReadFrameExpiredError

    while True:
        archive.check_operation_read()
        batch: list[tuple[str, float]] = []
        for _ in range(_COMPOSED_SORT_CHUNK):
            try:
                batch.append(next(rows))
            except StopIteration:
                break
            except (
                QueryCancelledError,
                QueryTimeoutError,
                QueryWorkBudgetExceededError,
                ReadFrameCancelledError,
                ReadFrameExpiredError,
            ):
                raise
            except Exception as exc:
                archive.check_operation_read()
                raise _VectorTraversalError(type(exc).__name__) from exc
        if not batch:
            return
        # Only provider traversal is optional. Failures in this canonical
        # evidence projection propagate as archive read failures.
        yield from archive.semantic_summaries(batch, limit=len(batch))


@contextmanager
def _scoped_ranked_hits(
    plan: SessionQueryPlan,
    archive: ArchiveStore,
    *,
    config: Config | None,
    archive_root: Path,
    keep: Callable[[list[ArchiveSessionSummary]], list[ArchiveSessionSummary]] | None = None,
    vector_failure: LaneFailure | None = None,
) -> Iterator[tuple[Iterator[ArchiveSessionSearchHit], SearchExecution]]:
    from polylogue.archive.query.execution_control import (
        QueryCancelledError,
        QueryTimeoutError,
        QueryWorkBudgetExceededError,
    )
    from polylogue.archive.query.search_contract import resolve_vector_provider
    from polylogue.core.errors import EmbeddingRetrievalNotReadyError
    from polylogue.storage.sqlite.connection_profile import ReadFrameCancelledError, ReadFrameExpiredError

    hybrid = plan.retrieval_lane == "hybrid" and plan.similar_session_id is None
    text = plan.similar_text or _plan_text_query(plan) or ""
    failure = vector_failure
    completed: list[LaneName] = []
    with closing(_qualified_scope(plan, archive, keep)) as session_ids, archive.scoped_search_population(session_ids):
        archive.check_operation_read()
        provider = plan.vector_provider
        if provider is not None and failure is not None:
            raise ValueError("vector setup cannot supply both a provider and a failure")
        if failure is None:
            provider, failure = resolve_vector_provider(
                config,
                archive_root=archive_root,
                provider=provider,
                require_credentials=plan.similar_session_id is None,
            )
        if provider is not None:
            archive.check_operation_read()
            with ExitStack() as cleanup:
                traversal = None
                try:
                    seed = archive.resolve_session_id(plan.similar_session_id) if plan.similar_session_id else None
                    scoped_ids = cleanup.enter_context(closing(archive.iter_scoped_search_session_ids()))
                    traversal = cleanup.enter_context(
                        provider.scoped_query(
                            scoped_ids,
                            index_connection=archive._conn,
                            configure_connection=archive.configure_operation_read_connection,
                            check_cancelled=archive.check_operation_read,
                            text=text if seed is None else None,
                            seed_session_id=seed,
                        )
                    )
                except (
                    QueryCancelledError,
                    QueryTimeoutError,
                    QueryWorkBudgetExceededError,
                    ReadFrameCancelledError,
                    ReadFrameExpiredError,
                ):
                    raise
                except EmbeddingRetrievalNotReadyError as exc:
                    if not hybrid:
                        raise
                    failure = LaneFailure("vector", "unavailable", exc.readiness_status, str(exc))
                except Exception as exc:
                    archive.check_operation_read()
                    failure = LaneFailure("vector", "execution_failed", type(exc).__name__, "vector retrieval failed")
                if traversal is not None:
                    if not traversal.exact or traversal.population != "eligible-sessions":
                        failure = LaneFailure(
                            "vector", "unavailable", "inexact_scope", "exact scoped vector retrieval is unavailable"
                        )
                    else:
                        try:
                            archive.settle_scoped_search_lane("vector", _vector_witnesses(archive, traversal.rows))
                        except _VectorTraversalError as exc:
                            archive.check_operation_read()
                            archive.discard_scoped_search_lane("vector")
                            failure = LaneFailure("vector", "execution_failed", str(exc), "vector retrieval failed")
                        else:
                            completed.append("vector")
        if failure is not None and not hybrid:
            raise EmbeddingRetrievalNotReadyError(
                failure.advisory,
                readiness_status="disabled" if failure.kind == "unavailable" else "failed",
            )
        if hybrid:
            lanes: tuple[tuple[LaneName, bool], ...] = (("text", False), ("action", True))
            for lane, actions_only in lanes:
                archive.check_operation_read()
                with closing(
                    archive.iter_search_summaries(
                        text,
                        limit=None,
                        actions_only=actions_only,
                        sort=plan.sort,
                        reverse=plan.reverse,
                        **plan_filter_kwargs(plan),
                    )
                ) as hits:
                    archive.settle_scoped_search_lane(lane, hits)
                completed.append(lane)
        successful = tuple(lane for lane in ("text", "action", "vector") if lane in completed)
        execution = SearchExecution(
            requested_lanes=("text", "action", "vector") if hybrid else ("vector",),
            executed_lanes=successful,
            completed_lanes=successful,
            exactness="exact",
            unavailable_lanes=("vector",) if failure is not None and failure.kind == "unavailable" else (),
            failed_lanes=(failure,) if failure is not None and failure.kind != "unavailable" else (),
        )
        with closing(archive.iter_scoped_search_hits(hybrid=hybrid)) as hits:
            yield hits, execution


@contextmanager
def _ordered_scoped_hits(
    plan: SessionQueryPlan,
    archive: ArchiveStore,
    hits: Iterator[ArchiveSessionSearchHit],
    *,
    full: bool,
) -> Iterator[Iterator[ArchiveSessionSearchHit]]:
    """Stage comparison keys, then stream the complete sorted relation."""
    if plan.sort is None:
        yield hits
        return

    def keys() -> Iterator[tuple[str, bool, int | float, int, str, int]]:
        for hit in hits:
            archive.check_operation_read()
            if full:
                values = session_order_values(plan, archive_envelope_to_session(archive.read_session(hit.session_id)))
            elif plan.sort in _COMPOSED_COUNT_SORTS:
                summary = archive_summary_to_domain(archive.read_summary(hit.session_id))
                values = summary_order_values(
                    plan,
                    summary,
                    metrics=archive.read_session_sort_metrics(
                        hit.session_id,
                        sort=cast(Literal["messages", "words", "longest", "tokens"], plan.sort),
                    ),
                )
            else:
                values = summary_order_values(plan, archive_summary_to_domain(archive.read_summary(hit.session_id)))
            yield hit.session_id, *values, hit.rank

    archive.settle_scoped_search_order(keys())
    with closing(
        archive.iter_scoped_search_hits(
            hybrid=plan.retrieval_lane == "hybrid" and plan.similar_session_id is None,
            explicit_sort=True,
            reverse=plan.reverse,
        )
    ) as ordered:
        yield ordered


def archive_search_hits(
    plan: SessionQueryPlan,
    *,
    archive_root: Path,
    config: Config | None,
    default_limit: int = DEFAULT_SESSION_LIST_LIMIT,
    archive: ArchiveStore | None = None,
    vector_failure: LaneFailure | None = None,
) -> ArchiveSearchResult:
    """Execute requested lanes once and return their actual outcome with hits.

    Hybrid keeps its request identity even when vector retrieval is unavailable
    or fails; text and action lanes still fuse. A supplied vector failure is
    setup evidence from an operation-scoped reader, not a request to retry
    construction outside that reader's snapshot.
    """
    from dataclasses import replace as _replace

    text = plan.similar_text or _plan_text_query(plan) or ""
    limit = plan.limit if plan.limit is not None else default_limit
    offset = plan.offset
    filter_kwargs = plan_filter_kwargs(plan)

    def read(archive: ArchiveStore) -> ArchiveSearchResult:
        if not _ranked_window(plan) and plan.retrieval_lane in {"auto", "dialogue", "actions"}:
            if plan.retrieval_lane == "actions":
                candidates = _fts_lane_candidates(archive, plan, text=text, limit=offset + limit, actions_only=True)
                hits = [
                    _replace(hit, rank=offset + rank)
                    for rank, hit in enumerate(candidates[offset : offset + limit], start=1)
                ]
                return ArchiveSearchResult(
                    _pair_hits(archive, hits), "actions", SearchExecution(("action",), ("action",))
                )
            hits = archive.search_summaries(
                text, limit=limit, offset=offset, sort=plan.sort, reverse=plan.reverse, **filter_kwargs
            )
            return ArchiveSearchResult(_pair_hits(archive, hits), "dialogue", SearchExecution(("text",), ("text",)))

        with (
            _scoped_ranked_hits(
                plan,
                archive,
                config=config,
                archive_root=archive_root,
                vector_failure=vector_failure,
            ) as (ranked_hits, execution),
            _ordered_scoped_hits(plan, archive, ranked_hits, full=False) as ordered_hits,
        ):
            page = list(islice(ordered_hits, offset, None if plan.limit is None else offset + plan.limit))
            return ArchiveSearchResult(
                _pair_hits(archive, page),
                "hybrid" if plan.retrieval_lane == "hybrid" else "semantic",
                execution,
            )

    from polylogue.storage.fts.fts_lifecycle import search_index_read_refusal

    try:
        if archive is not None:
            return read(archive)
        with archive_read_context(
            archive_root,
            operation="archive.query.search-hits",
            arguments={"plan": plan, "default_limit": default_limit},
            page_size=plan.limit,
            offset=plan.offset,
            projection="search-hits",
            workload_class="scan" if _ranked_window(plan) or plan.limit is None or plan.limit > 1000 else "interactive",
        ) as controlled_archive:
            return read(controlled_archive)
    except sqlite3.Error as exc:
        refusal = search_index_read_refusal(exc)
        if refusal is None:
            raise
        raise refusal from exc


def _pair_hits(
    archive: ArchiveStore,
    hits: list[ArchiveSessionSearchHit],
) -> list[tuple[ArchiveSessionSearchHit, ArchiveSessionSummary]]:
    paired: list[tuple[ArchiveSessionSearchHit, ArchiveSessionSummary]] = []
    for hit in hits:
        try:
            paired.append((hit, archive.read_summary(hit.session_id)))
        except KeyError:
            continue
    return paired


__all__ = [
    "count_archive",
    "first_archive",
    "list_archive",
    "list_summaries_archive",
]
