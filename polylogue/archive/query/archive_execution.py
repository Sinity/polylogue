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
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeVar

from polylogue.archive.hydration import archive_envelope_to_session, archive_summary_to_domain
from polylogue.archive.query.filter_kwargs import (
    plan_filter_kwargs,
)
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


def _archive_summaries(
    plan: SessionQueryPlan,
    archive: ArchiveStore,
    *,
    config: Config | None,
    archive_root: Path,
    default_limit: int,
) -> list[ArchiveSessionSummary]:
    filter_kwargs = plan_filter_kwargs(plan)
    limit = _fetch_limit(plan, default=default_limit)
    post_filter_fetch = plan.has_post_filters() and plan.limit is not None
    sort = plan.sort
    reverse = plan.reverse

    if plan.similar_session_id is not None:
        search_hits = _session_seed_hits(plan, archive, config=config, archive_root=archive_root)
        return _summaries_from_hits(archive, search_hits)

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
        return _summaries_from_hits(archive, search_hits)

    query_text = _plan_text_query(plan)
    if query_text is not None:
        hits: list[ArchiveSessionSearchHit] = []
        fetch_offset = 0 if post_filter_fetch else plan.offset
        while True:
            batch = archive.search_summaries(
                query_text,
                limit=limit,
                offset=fetch_offset,
                sort=sort,
                reverse=reverse,
                **filter_kwargs,
            )
            hits.extend(batch)
            if not post_filter_fetch or len(batch) < limit:
                break
            fetch_offset += len(batch)
        return _summaries_from_hits(archive, hits)

    if not post_filter_fetch:
        return archive.list_summaries(
            limit=limit,
            offset=plan.offset,
            sort=sort,
            reverse=reverse,
            sample=plan.sample is not None,
            **filter_kwargs,
        )
    summaries: list[ArchiveSessionSummary] = []
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
        summaries.extend(summary_batch)
        if len(summary_batch) < limit:
            break
        fetch_offset += len(summary_batch)
    return summaries


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


def _open_archive_for_write(archive_root: Path) -> ArchiveStore:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    return ArchiveStore.open_existing(archive_root, read_only=False)


def _attach_units_to_domain(
    items: builtins.list[_AttachableT],
    archive: ArchiveStore,
    with_units: tuple[str, ...],
    with_unit_fields: dict[str, tuple[str, ...]] | None = None,
    with_unit_windows: Mapping[str, WithUnitWindow] | None = None,
) -> builtins.list[_AttachableT]:
    """Attach ``with <units>`` projection rows onto domain models (#2492).

    Returns updated copies carrying ``attached_units`` (pydantic models are
    treated as immutable for safety). A no-op when no units are requested.
    """

    if not with_units or not items:
        return items
    from polylogue.archive.query.attached_units import fetch_attached_units

    session_ids = [item.id for item in items]
    attached = fetch_attached_units(
        archive, session_ids, with_units, unit_fields=with_unit_fields, unit_windows=with_unit_windows
    )
    updated: builtins.list[_AttachableT] = []
    for item in items:
        per_session = {unit: tuple(by_session.get(item.id, ())) for unit, by_session in attached.items()}
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
    rank_first = bool(
        plan.sort is None
        and (
            plan.fts_terms
            or plan.similar_text is not None
            or plan.similar_session_id is not None
            or plan.retrieval_lane in {"semantic", "hybrid"}
        )
    )

    def read(archive: ArchiveStore) -> list[SessionSummary]:
        archive_rows = _archive_summaries(
            plan,
            archive,
            config=config,
            archive_root=archive_root,
            default_limit=default_limit,
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
    filtered = plan._apply_common_filters(summaries, sql_pushed=True)
    ordered = filtered if rank_first else plan._sort_summaries(filtered)
    # SQL owns ordinary lexical and structured windows. Ranked routes fetch
    # an unwindowed candidate prefix, so filters and rank-preserving
    # deduplication precede one final page cut. Hybrid's lexical fallback
    # follows that same rule.
    ranked_window = (
        plan.similar_text is not None
        or plan.similar_session_id is not None
        or plan.retrieval_lane in {"semantic", "hybrid"}
    )
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
    rank_first = bool(
        plan.sort is None
        and (
            plan.fts_terms
            or plan.similar_text is not None
            or plan.similar_session_id is not None
            or plan.retrieval_lane in {"semantic", "hybrid"}
        )
    )

    def read(archive: ArchiveStore) -> list[Session]:
        archive_rows = _archive_summaries(
            plan,
            archive,
            config=config,
            archive_root=archive_root,
            default_limit=default_limit,
        )
        sessions = _attach_units_to_domain(
            [
                archive_envelope_to_session(
                    archive.read_session(summary.session_id), display_label=summary.display_label
                )
                for summary in archive_rows
            ],
            archive,
            with_units,
            with_unit_fields,
            with_unit_windows,
        )
        return sessions

    sessions = await run_archive_read(
        archive_root,
        operation="archive.query.list",
        arguments={"plan": plan, "default_limit": default_limit, "with_units": with_units},
        work=read,
        page_size=plan.limit,
        offset=plan.offset,
        projection="sessions",
        workload_class="scan" if plan.limit is None or plan.limit > 1000 else "interactive",
    )
    filtered = plan._apply_full_filters(sessions, sql_pushed=True)
    ordered = filtered if rank_first else plan._sort_sessions(filtered)
    ranked_window = (
        plan.similar_text is not None
        or plan.similar_session_id is not None
        or plan.retrieval_lane in {"semantic", "hybrid"}
    )
    if (plan.has_post_filters() or ranked_window) and plan.offset:
        ordered = ordered[plan.offset :]
    return plan._finalize(ordered)


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

    unbounded = plan.with_limit(None)
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


def archive_search_hits(
    plan: SessionQueryPlan,
    *,
    archive_root: Path,
    config: Config | None,
    default_limit: int = DEFAULT_SESSION_LIST_LIMIT,
    archive: ArchiveStore | None = None,
) -> tuple[list[tuple[ArchiveSessionSearchHit, ArchiveSessionSummary]], str]:
    """Resolve a search plan to archive session hits paired with summaries.

    Returns ``(hits, resolved_lane)`` where each hit carries its
    :class:`ArchiveSessionSearchHit` plus the session summary, and ``resolved_lane``
    is the concrete lane that ran (``dialogue``/``semantic``/``hybrid``).
    """
    from polylogue.storage.search_providers import create_vector_provider, reciprocal_rank_fusion

    text = plan.similar_text or _plan_text_query(plan) or ""
    limit = plan.limit if plan.limit is not None else default_limit
    offset = plan.offset
    filter_kwargs = plan_filter_kwargs(plan)

    def read(archive: ArchiveStore) -> tuple[list[tuple[ArchiveSessionSearchHit, ArchiveSessionSummary]], str]:
        if plan.similar_session_id is not None:
            pool = max(limit + offset, limit) * 3
            scored = _session_seed_scored(plan, config=config, archive_root=archive_root, pool=pool)
            semantic_hits = archive.semantic_summaries(scored, limit=pool, offset=0, **filter_kwargs)
            return _pair_hits(archive, semantic_hits[offset : offset + limit]), "semantic"

        if plan.similar_text is None and plan.retrieval_lane in {"auto", "dialogue"}:
            hits = archive.search_summaries(
                text,
                limit=limit,
                offset=offset,
                sort=plan.sort,
                reverse=plan.reverse,
                **filter_kwargs,
            )
            return _pair_hits(archive, hits), "dialogue"

        vector_provider = plan.vector_provider
        if vector_provider is None and config is not None:
            vector_provider = create_vector_provider(config, db_path=archive_root / "embeddings.db")
        if vector_provider is None:
            from polylogue.core.errors import EmbeddingRetrievalNotReadyError

            if plan.retrieval_lane != "hybrid":
                raise EmbeddingRetrievalNotReadyError(
                    "semantic retrieval is unavailable: no configured/constructible vector backend; "
                    "configure Voyage/sqlite-vec and retry",
                    readiness_status="disabled",
                )
            # Hybrid is explicitly allowed to degrade, but must retain the
            # lexical evidence.  The envelope records the missing vector lane.
            pool = max(limit + offset, limit) * 3
            lexical_hits = archive.search_summaries(
                text, limit=pool, offset=0, sort=plan.sort, reverse=plan.reverse, **filter_kwargs
            )
            return _pair_hits(archive, lexical_hits[offset : offset + limit]), "dialogue"

        semantic_query = plan.similar_text or text
        pool = max(limit + offset, limit) * 3
        scored = vector_provider.query(semantic_query, limit=pool)
        semantic_hits = archive.semantic_summaries(scored, limit=pool, offset=0, **filter_kwargs)
        if plan.retrieval_lane != "hybrid":
            return _pair_hits(archive, semantic_hits[offset : offset + limit]), "semantic"

        lexical_hits = archive.search_summaries(
            text,
            limit=pool,
            offset=0,
            sort=plan.sort,
            reverse=plan.reverse,
            **filter_kwargs,
        )
        from dataclasses import replace as _replace

        hit_by_session: dict[str, ArchiveSessionSearchHit] = {}
        for hit in [*lexical_hits, *semantic_hits]:
            hit_by_session.setdefault(hit.session_id, hit)
        fused = reciprocal_rank_fusion(
            [(hit.session_id, 0.0) for hit in lexical_hits],
            [(hit.session_id, 0.0) for hit in semantic_hits],
        )
        page = fused[offset : offset + limit]
        text_ranks = {hit.session_id: rank for rank, hit in enumerate(lexical_hits, start=1)}
        vector_ranks = {hit.session_id: rank for rank, hit in enumerate(semantic_hits, start=1)}
        ranked = [
            _replace(
                hit_by_session[session_id],
                rank=offset + index,
                lane_ranks={
                    "text": text_ranks.get(session_id),
                    "vector": vector_ranks.get(session_id),
                },
            )
            for index, (session_id, _score) in enumerate(page, start=1)
            if session_id in hit_by_session
        ]
        return _pair_hits(archive, ranked), "hybrid"

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


async def delete_archive(
    plan: SessionQueryPlan,
    *,
    archive_root: Path,
    config: Config | None,
) -> int:
    if plan.can_use_summaries():
        summaries = await list_summaries_archive(
            plan,
            archive_root=archive_root,
            config=config,
        )
        session_ids = tuple(str(summary.id) for summary in summaries)
    else:
        sessions = await list_archive(plan, archive_root=archive_root, config=config)
        session_ids = tuple(str(session.id) for session in sessions)
    if not session_ids:
        return 0
    with _open_archive_for_write(archive_root) as archive:
        return archive.delete_sessions(session_ids)


__all__ = [
    "count_archive",
    "delete_archive",
    "first_archive",
    "list_archive",
    "list_summaries_archive",
]
