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
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, TypeVar

from polylogue.archive.hydration import archive_envelope_to_session, archive_summary_to_domain
from polylogue.archive.query.filter_kwargs import (
    plan_filter_kwargs,
)
from polylogue.archive.query.search_contract import ArchiveSearchResult, LaneFailure, SearchExecution
from polylogue.archive.query.sorting import OffsetSampledPage
from polylogue.archive.query.spec import DEFAULT_SESSION_LIST_LIMIT
from polylogue.archive.query.transaction import archive_read_context, run_archive_read
from polylogue.archive.session.domain_models import Session, SessionSummary

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


def _session_seed_scored(
    plan: SessionQueryPlan,
    *,
    config: Config | None,
    archive_root: Path,
    pool: int,
) -> list[tuple[str, float]]:
    """Resolve a ``near:id:<ref>`` plan to vector-ranked ``(message_id, distance)``.

    Unlike the text-semantic leg, a session seed is an explicit request to rank by
    a *stored* session's embeddings; there is no text query to fall back to. When
    the request cannot be honored — no vector backend is available, or the seed
    session has no stored embeddings — this raises a typed ``ExpressionCompileError``
    on the ``near`` field rather than degrading to an unfiltered or silently-empty
    listing. The returned hits already exclude the seed session's own messages
    (enforced by ``VectorProvider.query_by_session``).
    """
    from polylogue.archive.query.expression import ExpressionCompileError
    from polylogue.storage.search_providers import create_vector_provider
    from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError

    seed = plan.similar_session_id
    assert seed is not None
    vector_provider = plan.vector_provider
    if vector_provider is None and config is not None:
        vector_provider = create_vector_provider(config, db_path=archive_root / "embeddings.db")
    if vector_provider is None:
        raise ExpressionCompileError(
            "near:id: session-seeded similarity needs a configured vector backend "
            "(sqlite-vec plus an embedded archive); none is available for this archive.",
            field="near",
        )
    try:
        return vector_provider.query_by_session(seed, limit=pool)
    except SqliteVecError as exc:
        raise ExpressionCompileError(str(exc), field="near") from exc


def _session_seed_hits(
    plan: SessionQueryPlan,
    archive: ArchiveStore,
    *,
    config: Config | None,
    archive_root: Path,
) -> list[ArchiveSessionSearchHit]:
    """Resolve a session-seeded plan to filtered session-level hits."""
    limit = plan.limit if plan.limit is not None else 50
    pool = max(limit + plan.offset, limit) * 3
    scored = _session_seed_scored(plan, config=config, archive_root=archive_root, pool=pool)
    return archive.semantic_summaries(scored, limit=pool, offset=0, **plan_filter_kwargs(plan))


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


def _semantic_hits(
    plan: SessionQueryPlan,
    archive: ArchiveStore,
    *,
    config: Config | None,
    archive_root: Path,
) -> list[ArchiveSessionSearchHit]:
    """Resolve the vector leg of a semantic/hybrid plan.

    Graceful-degradation contract (#1743): when no vector provider can be
    constructed for the active archive — sqlite-vec/Voyage not configured, the
    archive holds no embeddings, or the configured embeddings are unusable — the
    semantic leg yields no hits instead of raising. The caller (``_archive_summaries``)
    therefore returns an empty result set for a pure-semantic request and falls
    back to the lexical leg for hybrid, never surfacing a hard error to read
    surfaces over an embeddings-less archive.
    """
    from polylogue.storage.search_providers import create_vector_provider

    text = plan.similar_text or _plan_text_query(plan) or ""
    if not text:
        return []
    vector_provider = plan.vector_provider
    if vector_provider is None and config is not None:
        vector_provider = create_vector_provider(config, db_path=archive_root / "embeddings.db")
    if vector_provider is None:
        from polylogue.core.errors import EmbeddingRetrievalNotReadyError

        raise EmbeddingRetrievalNotReadyError(
            "semantic retrieval is unavailable: no configured/constructible vector backend; "
            "configure Voyage/sqlite-vec and retry",
            readiness_status="disabled",
        )
    limit = plan.limit if plan.limit is not None else 50
    scored = vector_provider.query(text, limit=max(limit + plan.offset, limit) * 3)
    return archive.semantic_summaries(
        scored,
        limit=max(limit + plan.offset, limit) * 3,
        offset=0,
        **plan_filter_kwargs(plan),
    )


#: Sorts over per-session counters, which the index stores for a lineage
#: child's own tail only.
_COMPOSED_COUNT_SORTS = frozenset({"messages", "words", "longest", "tokens"})
#: Sessions hydrated at once while a complete composed sort selects its page.
_COMPOSED_SORT_CHUNK = 200


def _ranked_window(plan: SessionQueryPlan) -> bool:
    """Whether results are ordered by a vector rank SQL cannot express."""
    return (
        plan.similar_text is not None
        or plan.similar_session_id is not None
        or plan.retrieval_lane in {"semantic", "hybrid"}
    )


def _sort_owned_here(plan: SessionQueryPlan) -> bool:
    """Whether filtered candidates still need ordering after the fetch.

    SQL already ordered every lexical and structured result, and filtering
    preserves that order, so those are used as fetched. Only a vector-ranked
    route with an explicit sort is ordered in Python; one with no sort keeps
    its rank order.
    """
    return _ranked_window(plan) and plan.sort is not None


def order_query_summaries(plan: SessionQueryPlan, candidates: list[SessionSummary]) -> list[SessionSummary]:
    """Order filtered summaries fetched by :func:`_archive_summaries`."""
    return plan._sort_summaries(candidates) if _sort_owned_here(plan) else candidates


def order_query_sessions(plan: SessionQueryPlan, candidates: list[Session]) -> list[Session]:
    """Order filtered sessions fetched by :func:`_archive_summaries`."""
    return plan._sort_sessions(candidates) if _sort_owned_here(plan) else candidates


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

    if plan.similar_session_id is not None:
        search_hits = _session_seed_hits(plan, archive, config=config, archive_root=archive_root)
        return deliver(_kept(keep, _summaries_from_hits(archive, search_hits)))

    if plan.similar_text is not None or plan.retrieval_lane in {"semantic", "hybrid"}:
        try:
            search_hits = _semantic_hits(plan, archive, config=config, archive_root=archive_root)
        except Exception as exc:
            from polylogue.core.errors import EmbeddingRetrievalNotReadyError

            if plan.retrieval_lane != "hybrid" or not isinstance(exc, EmbeddingRetrievalNotReadyError):
                raise
            # Hybrid policy permits a partial lexical result; its search-hit
            # execution envelope records the unavailable vector lane.
            search_hits = archive.search_summaries(
                _plan_text_query(plan) or "",
                # The shared ranked window is applied after this fallback,
                # just as it is after the semantic leg. Fetch its prefix at
                # offset zero so a later page cannot be offset twice.
                limit=max(limit + plan.offset, limit),
                offset=0,
                sort=sort,
                reverse=reverse,
                **filter_kwargs,
            )
        return deliver(_kept(keep, _summaries_from_hits(archive, search_hits)))

    query_text = _plan_text_query(plan)
    if query_text is not None:
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
            units=list(with_units),
        )
    updated: builtins.list[_AttachableT] = []
    for item in items:
        per_session = {unit: tuple(by_session.get(item.id, ())) for unit, by_session in attached.rows.items()}
        updated.append(item.model_copy(update={"attached_units": per_session}))
    return updated


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
    def keep_matching(rows: list[ArchiveSessionSummary]) -> list[ArchiveSessionSummary]:
        by_id = {row.session_id: row for row in rows}
        matching = plan._apply_common_filters([archive_summary_to_domain(row) for row in rows], sql_pushed=True)
        return [by_id[str(summary.id)] for summary in matching]

    def read(archive: ArchiveStore) -> list[SessionSummary]:
        archive_rows = _archive_summaries(
            plan,
            archive,
            config=config,
            archive_root=archive_root,
            default_limit=default_limit,
            keep=keep_matching if plan.has_post_filters() else None,
        )
        summaries = _attach_units_to_domain(
            [archive_summary_to_domain(summary) for summary in archive_rows],
            archive,
            with_units,
            with_unit_fields,
            with_unit_windows,
        )
        return summaries

    summaries = await run_archive_read(
        archive_root,
        operation="archive.query.list-summaries",
        arguments={"plan": plan, "default_limit": default_limit, "with_units": with_units},
        work=read,
        page_size=plan.limit,
        offset=plan.offset,
        projection="session-summaries",
        workload_class="scan" if plan.limit is None or plan.limit > 1000 else "interactive",
    )
    # ``keep_matching`` already filtered every row once; a predicate is never
    # evaluated twice for one candidate.
    filtered = summaries if plan.has_post_filters() else plan._apply_common_filters(summaries, sql_pushed=True)
    # SQL orders every lexical and structured result. Only a vector-ranked
    # route with an explicit sort is ordered here. Ranked routes fetch an
    # unwindowed candidate prefix, so filters and rank-preserving
    # deduplication precede one final page cut. Hybrid's lexical fallback
    # follows that same rule.
    ranked_window = _ranked_window(plan)
    ordered = order_query_summaries(plan, filtered)
    if (plan.has_post_filters() or ranked_window) and plan.offset:
        ordered = ordered[plan.offset :]
    return plan._finalize(ordered)


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
    # Stored counters describe a lineage child's own divergent tail, while the
    # returned Session recomposes its inherited prefix. A count-ordered page of
    # full sessions is therefore ordered over the composed sessions, from the
    # unwindowed candidate set, instead of trusting the tail-only SQL keys.
    composed_order = plan.sort in _COMPOSED_COUNT_SORTS
    ranked_window = _ranked_window(plan)
    # SQL-backed routes page through the whole candidate set for a composed
    # sort. A ranked route already fetches an unwindowed candidate prefix
    # sized from the requested window, so it keeps the requested plan.
    complete = composed_order and not ranked_window
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
            if complete:
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
            for session in survivors:
                kept_sessions[str(session.id)] = session
            return [row for row in rows if row.session_id in kept_sessions]

        filtering = plan.has_post_filters()

        def reduce_batch(rows: list[ArchiveSessionSummary]) -> None:
            # A complete scan streams its candidates into the bounded
            # reducer; a filtered one already retained its survivors in
            # ``keep``.
            if filtering:
                return
            for start in range(0, len(rows), _COMPOSED_SORT_CHUNK):
                chunk = rows[start : start + _COMPOSED_SORT_CHUNK]
                retain(plan._apply_full_filters(hydrate(archive, chunk), sql_pushed=True))

        archive_rows = _archive_summaries(
            fetch_plan,
            archive,
            config=config,
            archive_root=archive_root,
            default_limit=default_limit,
            keep=keep if filtering else None,
            complete=complete,
            on_batch=reduce_batch if complete else None,
        )
        if complete:
            ordered = best
        else:
            if filtering:
                candidates = [kept_sessions[row.session_id] for row in archive_rows]
            else:
                candidates = plan._apply_full_filters(hydrate(archive, archive_rows), sql_pushed=True)
            ordered = plan._sort_sessions(candidates) if composed_order else order_query_sessions(plan, candidates)
        if (complete or filtering or ranked_window) and plan.offset:
            ordered = ordered[plan.offset :]
        # Filtered survivors already carry the page-width projection their
        # predicate saw; unfiltered sessions are projected over the served
        # page only.
        if complete and plan.limit is None and not plan.sample:
            ordered = ordered[:default_limit]
        page = plan._finalize(ordered)
        return page if filtering else attach(archive, page)

    return await run_archive_read(
        archive_root,
        operation="archive.query.list",
        arguments={"plan": plan, "default_limit": default_limit, "with_units": with_units},
        work=read,
        page_size=plan.limit,
        offset=plan.offset,
        projection="sessions",
        # A complete composed sort reads and hydrates every candidate,
        # whatever the requested page size.
        workload_class="scan" if complete or plan.limit is None or plan.limit > 1000 else "interactive",
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
                return int(archive.count_search_sessions(query_text, **filter_kwargs))
            return int(archive.count_sessions(**filter_kwargs))

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
    for hit in archive.iter_search_summaries(
        text,
        limit=None,
        actions_only=actions_only,
        sort=plan.sort,
        reverse=plan.reverse,
        **plan_filter_kwargs(plan),
    ):
        candidates.setdefault(hit.session_id, hit)
        if len(candidates) == limit:
            break
    return list(candidates.values())


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

    from polylogue.archive.query.execution_control import (
        QueryCancelledError,
        QueryTimeoutError,
        QueryWorkBudgetExceededError,
    )
    from polylogue.archive.query.search_contract import resolve_vector_provider
    from polylogue.core.errors import EmbeddingRetrievalNotReadyError
    from polylogue.storage.search_providers import reciprocal_rank_fusion
    from polylogue.storage.sqlite.connection_profile import ReadFrameCancelledError, ReadFrameExpiredError

    text = plan.similar_text or _plan_text_query(plan) or ""
    limit = plan.limit if plan.limit is not None else default_limit
    offset = plan.offset
    filter_kwargs = plan_filter_kwargs(plan)

    def read(archive: ArchiveStore) -> ArchiveSearchResult:
        if plan.similar_session_id is not None:
            # A session seed has no lexical leg to degrade to: an unusable
            # vector backend is a typed readiness refusal, as for text semantic.
            seed_plan = plan
            if plan.vector_provider is None:
                seed_provider, seed_failure = (
                    (None, vector_failure)
                    if vector_failure is not None
                    else resolve_vector_provider(config, archive_root=archive_root)
                )
                if seed_failure is not None:
                    raise EmbeddingRetrievalNotReadyError(
                        seed_failure.advisory,
                        readiness_status="disabled" if seed_failure.kind == "unavailable" else "failed",
                    )
                seed_plan = _replace(plan, vector_provider=seed_provider)
            pool = max(limit + offset, limit) * 3
            seed_scored = _session_seed_scored(seed_plan, config=config, archive_root=archive_root, pool=pool)
            seed_hits = archive.semantic_summaries(seed_scored, limit=pool, offset=0, **filter_kwargs)
            return ArchiveSearchResult(
                _pair_hits(archive, seed_hits[offset : offset + limit]),
                "semantic",
                SearchExecution(("vector",), ("vector",)),
            )

        if plan.similar_text is None and plan.retrieval_lane in {"auto", "dialogue", "actions"}:
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

        failure = vector_failure
        provider = plan.vector_provider
        if failure is not None and provider is not None:
            raise ValueError("vector setup cannot supply both a provider and a failure")
        if failure is None:
            provider, failure = resolve_vector_provider(config, archive_root=archive_root, provider=provider)
        pool = max(limit + offset, limit) * 3
        semantic_hits: list[ArchiveSessionSearchHit] = []
        if provider is not None:
            scored: list[tuple[str, float]] | None = None
            try:
                scored = provider.query(plan.similar_text or text, limit=pool)
            except (
                QueryCancelledError,
                QueryTimeoutError,
                QueryWorkBudgetExceededError,
                ReadFrameCancelledError,
                ReadFrameExpiredError,
            ):
                raise
            except EmbeddingRetrievalNotReadyError as exc:
                if plan.retrieval_lane != "hybrid":
                    raise
                failure = LaneFailure(
                    "vector",
                    "unavailable",
                    exc.readiness_status,
                    "vector retrieval has no current embedded evidence; run embedding status and backfill before retrying",
                )
            except Exception as exc:
                failure = LaneFailure(
                    "vector",
                    "execution_failed",
                    type(exc).__name__,
                    "vector retrieval failed; inspect the embedding backend and retry",
                )
            # The archive's own read is not the optional vector lane: its
            # failure must not be relabelled as a degraded vector leg.
            if scored is not None:
                semantic_hits = archive.semantic_summaries(scored, limit=pool, offset=0, **filter_kwargs)
        if plan.retrieval_lane != "hybrid":
            if failure is not None:
                raise EmbeddingRetrievalNotReadyError(
                    failure.advisory,
                    readiness_status="disabled" if failure.kind == "unavailable" else "failed",
                )
            return ArchiveSearchResult(
                _pair_hits(archive, semantic_hits[offset : offset + limit]),
                "semantic",
                SearchExecution(("vector",), ("vector",)),
            )

        lexical_hits = _fts_lane_candidates(archive, plan, text=text, limit=pool, actions_only=False)
        action_hits = _fts_lane_candidates(archive, plan, text=text, limit=pool, actions_only=True)
        lanes = {"text": lexical_hits, "action": action_hits, "vector": semantic_hits}
        hit_by_session: dict[str, ArchiveSessionSearchHit] = {}
        ranks: dict[str, dict[str, int]] = {}
        for lane, lane_hits in lanes.items():
            ranks[lane] = {}
            for rank, hit in enumerate(lane_hits, start=1):
                hit_by_session.setdefault(hit.session_id, hit)
                ranks[lane].setdefault(hit.session_id, rank)
        fused = reciprocal_rank_fusion(*[[(hit.session_id, 0.0) for hit in lane_hits] for lane_hits in lanes.values()])
        ranked = [
            _replace(
                hit_by_session[session_id],
                rank=offset + index,
                lane_ranks={lane: lane_ranks.get(session_id) for lane, lane_ranks in ranks.items()},
            )
            for index, (session_id, _score) in enumerate(fused[offset : offset + limit], start=1)
        ]
        return ArchiveSearchResult(
            _pair_hits(archive, ranked),
            "hybrid",
            SearchExecution(
                requested_lanes=("text", "action", "vector"),
                executed_lanes=("text", "action", "vector") if failure is None else ("text", "action"),
                unavailable_lanes=("vector",) if failure is not None and failure.kind == "unavailable" else (),
                failed_lanes=(failure,) if failure is not None and failure.kind != "unavailable" else (),
            ),
        )

    if archive is not None:
        return read(archive)
    with archive_read_context(
        archive_root,
        operation="archive.query.search-hits",
        arguments={"plan": plan, "default_limit": default_limit},
        page_size=plan.limit,
        offset=plan.offset,
        projection="search-hits",
        workload_class="scan" if plan.limit is None or plan.limit > 1000 else "interactive",
    ) as controlled_archive:
        return read(controlled_archive)


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
