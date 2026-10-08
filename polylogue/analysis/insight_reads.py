"""Canonical insight pages over the caller's original pinned archive reader."""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Generator, Iterator
from contextlib import closing
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from polylogue.analysis.archive import (
    ArchiveCoverageInsightQuery,
    ArchiveDebtInsightQuery,
    ArchiveInsightModel,
    CostRollupInsightQuery,
    SessionCostInsight,
    SessionCostInsightQuery,
    SessionProfileInsightQuery,
    SessionTagRollupInsight,
    SessionTagRollupQuery,
    ThreadInsightQuery,
    UsageTimelineInsightQuery,
)
from polylogue.analysis.command_shapes import CommandShapeUsageQuery
from polylogue.analysis.cost_enrichment import enrich_session_cost_insight, enrich_session_cost_insights
from polylogue.analysis.tag_rollups import synthesize_origin_tag_rollups
from polylogue.analysis.tool_episodes import ToolEpisodeQuery
from polylogue.analysis.tool_usage import ToolUsageInsightQuery
from polylogue.archive.query.spec import parse_query_date

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _archive_query_date_ms(field: str, value: str | None) -> int | None:
    parsed = parse_query_date(field, value)
    return None if parsed is None else int(parsed.timestamp() * 1000)


def _session_cost_insight_page(archive: ArchiveStore, request: SessionCostInsightQuery) -> list[SessionCostInsight]:
    """One page of enriched session cost insights, filtered before the page cut.

    ``status`` and ``model`` are decided on the *enriched* estimate, so neither
    can be pushed into the archive query: filtering an already-cut page would
    answer "of the newest N, the matching ones". The matched scope is scanned
    in one forward pass until the requested page is full.
    """

    def scan(*, limit: int | None = None, offset: int = 0) -> Iterator[SessionCostInsight]:
        return archive.iter_session_cost_insights(
            session_id=request.session_id,
            time_basis=request.time_basis,
            origin=request.origin,
            since_ms=_archive_query_date_ms("since", request.since),
            until_ms=_archive_query_date_ms("until", request.until),
            limit=limit,
            offset=offset,
        )

    if request.status is None and request.model is None:
        return enrich_session_cost_insights(archive, list(scan(limit=request.limit, offset=request.offset)))

    def matches(insight: SessionCostInsight) -> bool:
        if request.status is not None and insight.estimate.status != request.status:
            return False
        return request.model is None or request.model in {
            insight.estimate.normalized_model,
            insight.estimate.model_name,
        }

    # One forward scan of the matched scope: ``islice`` skips the offset
    # without retaining it and stops as soon as the page is full.
    matching = (
        enriched
        for enriched in (enrich_session_cost_insight(archive, insight) for insight in scan())
        if matches(enriched)
    )
    start = max(int(request.offset), 0)
    stop = None if request.limit is None else start + max(int(request.limit), 0)
    return list(itertools.islice(matching, start, stop))


def _iter_tag_rollups(
    archive: ArchiveStore, request: SessionTagRollupQuery
) -> Generator[SessionTagRollupInsight, None, None]:
    since_ms = _archive_query_date_ms("since", request.since)
    until_ms = _archive_query_date_ms("until", request.until)
    origin_rollups = sorted(
        synthesize_origin_tag_rollups(
            archive,
            origin=request.origin,
            query=request.query,
            since_ms=since_ms,
            until_ms=until_ms,
            materialized_at=datetime.now(UTC).isoformat(),
        ),
        key=lambda row: (-row.session_count, row.tag),
    )
    with closing(
        archive.iter_session_tag_rollup_insights(
            origin=request.origin, query=request.query, since_ms=since_ms, until_ms=until_ms, limit=None, offset=0
        )
    ) as materialized:
        # Stable merge keeps materialized rows before equal synthesized rows, as the original sort did.
        merged = heapq.merge(materialized, origin_rollups, key=lambda row: (-row.session_count, row.tag))
        stop = None if request.limit is None else request.offset + max(int(request.limit), 0)
        yield from itertools.islice(merged, request.offset, stop)


def read_insight_page(archive: ArchiveStore, request: ArchiveInsightModel) -> list[ArchiveInsightModel]:
    """Use the canonical query filters and enrichment before its page cut."""
    if isinstance(
        request, (SessionTagRollupQuery, ThreadInsightQuery, ArchiveCoverageInsightQuery, SessionProfileInsightQuery)
    ):
        return list(iter_insight_rows(archive, request, checkpoint=archive.check_operation_read))
    if isinstance(request, ToolUsageInsightQuery):
        return list(archive.list_tool_usage_insights(request))
    if isinstance(request, ToolEpisodeQuery):
        return list(archive.list_tool_episode_insights(request))
    if isinstance(request, CommandShapeUsageQuery):
        return list(archive.list_command_shape_usage(request))
    if isinstance(request, SessionCostInsightQuery):
        return list(_session_cost_insight_page(archive, request))
    if isinstance(request, CostRollupInsightQuery):
        return list(
            archive.list_cost_rollup_insights(
                origin=request.origin,
                model=request.model,
                since_ms=_archive_query_date_ms("since", request.since),
                until_ms=_archive_query_date_ms("until", request.until),
                limit=request.limit,
                offset=request.offset,
            )
        )
    if isinstance(request, UsageTimelineInsightQuery):
        return list(
            archive.list_usage_timeline_insights(
                origin=request.origin,
                model=request.model,
                group_by=request.group_by,
                since_ms=_archive_query_date_ms("since", request.since),
                until_ms=_archive_query_date_ms("until", request.until),
                limit=request.limit,
                offset=request.offset,
            )
        )
    if isinstance(request, ArchiveDebtInsightQuery):
        return list(
            archive.list_archive_debt_insights(
                category=request.category,
                only_actionable=request.only_actionable,
                limit=request.limit,
                offset=request.offset,
            )
        )
    raise TypeError(f"insight query is not declared: {type(request).__name__}")


def iter_insight_rows(
    archive: ArchiveStore, request: ArchiveInsightModel, *, checkpoint: Callable[[], None] = lambda: None
) -> Generator[ArchiveInsightModel, None, None]:
    """One forward traversal of the exportable insight relation on the original reader."""
    checkpoint()
    row: ArchiveInsightModel
    if isinstance(request, SessionTagRollupQuery):
        with closing(_iter_tag_rollups(archive, request)) as tag_rows:
            for row in tag_rows:
                checkpoint()
                yield row
        return
    if isinstance(request, ThreadInsightQuery):
        with closing(
            archive.iter_thread_insights(
                query=request.query,
                since_ms=_archive_query_date_ms("since", request.since),
                until_ms=_archive_query_date_ms("until", request.until),
                limit=request.limit,
                offset=request.offset,
            )
        ) as thread_rows:
            for row in thread_rows:
                checkpoint()
                yield row
        return

    if isinstance(request, ArchiveCoverageInsightQuery):
        coverage_rows = archive.list_archive_coverage_insights(
            group_by=request.group_by,
            origin=request.origin,
            since_ms=_archive_query_date_ms("since", request.since),
            until_ms=_archive_query_date_ms("until", request.until),
            limit=request.limit,
            offset=request.offset,
        )

        for row in coverage_rows:
            checkpoint()
            yield row
        return
    if isinstance(request, SessionProfileInsightQuery):
        with closing(
            archive.iter_session_profile_insights(
                origin=request.origin,
                workflow_shape=request.workflow_shape,
                terminal_state=request.terminal_state,
                tag=request.tag,
                repo=request.repo,
                since_ms=_archive_query_date_ms("since", request.since),
                until_ms=_archive_query_date_ms("until", request.until),
                first_message_since=request.first_message_since,
                first_message_until=request.first_message_until,
                session_date_since=request.session_date_since,
                session_date_until=request.session_date_until,
                tier=request.tier,
                query=request.query,
                limit=request.limit,
                offset=request.offset,
                min_wallclock_seconds=request.min_wallclock_seconds,
                max_wallclock_seconds=request.max_wallclock_seconds,
                sort=request.sort,
            )
        ) as profile_rows:
            for row in profile_rows:
                checkpoint()
                yield row
        return

    raise TypeError(f"insight export query is not declared: {type(request).__name__}")
