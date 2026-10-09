"""Shared pinned selection for session-set read projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.hydration import archive_summary_to_domain
from polylogue.archive.query.sorting import OffsetSampledPage

if TYPE_CHECKING:
    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary, ArchiveStore

_POST_FILTER_CHUNK = 200


def select_read_view_summaries(
    plan: SessionQueryPlan,
    *,
    archive: ArchiveStore,
    default_limit: int,
) -> list[SessionSummary]:
    """Run query candidates through the plan's filters, order and page cut."""

    from dataclasses import replace

    from polylogue.archive.hydration import archive_envelope_to_session
    from polylogue.archive.query.archive_execution import (
        _COMPOSED_COUNT_SORTS,
        _archive_summaries,
        _ranked_window,
    )

    if plan.sort not in _COMPOSED_COUNT_SORTS:
        from polylogue.archive.query.archive_execution import _list_summaries_in_archive

        return _list_summaries_in_archive(
            plan, archive, config=None, archive_root=archive.archive_root, default_limit=default_limit
        )

    def keep_matching(rows: list[ArchiveSessionSummary]) -> list[ArchiveSessionSummary]:
        sessions = [
            archive_envelope_to_session(
                archive.read_session(row.session_id),
                display_label=row.display_label,
                display_label_source=row.display_label_source,
            )
            for row in rows
        ]
        ids = {str(session.id) for session in plan._apply_full_filters(sessions, sql_pushed=True)}
        return [row for row in rows if row.session_id in ids]

    if _ranked_window(plan):
        # Membership and comparison keys are settled on the held frame before
        # a page is hydrated. A sample retains only its requested summary rows.
        from polylogue.archive.query.sorting import SessionReservoir

        ranked_reservoir = SessionReservoir[ArchiveSessionSummary](plan.sample) if plan.sample is not None else None
        remaining_offset = plan.offset

        def sample_batch(rows: list[ArchiveSessionSummary]) -> None:
            nonlocal remaining_offset
            skipped = min(remaining_offset, len(rows))
            remaining_offset -= skipped
            assert ranked_reservoir is not None
            ranked_reservoir.offer(rows[skipped:])

        rows = _archive_summaries(
            plan,
            archive,
            config=None,
            archive_root=archive.archive_root,
            default_limit=default_limit,
            complete=ranked_reservoir is not None,
            on_batch=sample_batch if ranked_reservoir is not None else None,
            full_sort=True,
            keep=keep_matching if plan.has_post_filters() else None,
        )
        rows = ranked_reservoir.items() if ranked_reservoir is not None else rows[plan.offset :]
        return plan._finalize([archive_summary_to_domain(row) for row in rows])

    # The index stores a child's tail counters. Rank composed survivors before
    # the page cut, retaining only the requested best rows or sampled reservoir.
    fetch_plan = replace(plan, limit=None, offset=0)
    bound = plan.offset + (plan.limit if plan.limit is not None else default_limit)
    best: list[Session] = []
    reservoir: OffsetSampledPage[Session] | None = (
        OffsetSampledPage(offset=plan.offset, sample=plan.sample, sort=plan._sort_sessions) if plan.sample else None
    )
    summary_by_id: dict[str, SessionSummary] = {}

    def consume(rows: list[ArchiveSessionSummary]) -> None:
        nonlocal best
        for start in range(0, len(rows), _POST_FILTER_CHUNK):
            archive.check_operation_read()
            chunk = rows[start : start + _POST_FILTER_CHUNK]
            for row in chunk:
                summary_by_id[row.session_id] = archive_summary_to_domain(row)
            sessions = [
                archive_envelope_to_session(
                    archive.read_session(row.session_id),
                    display_label=row.display_label,
                    display_label_source=row.display_label_source,
                )
                for row in chunk
            ]
            kept = plan._apply_full_filters(sessions, sql_pushed=True)
            if reservoir is not None:
                reservoir.offer(kept)
                best = reservoir.items()
            else:
                best = plan._sort_sessions([*best, *kept])[:bound]
            retained = {str(session.id) for session in best}
            for session_id in [key for key in summary_by_id if key not in retained]:
                del summary_by_id[session_id]

    _archive_summaries(
        fetch_plan,
        archive,
        config=None,
        archive_root=archive.archive_root,
        default_limit=default_limit,
        complete=True,
        on_batch=consume,
    )
    ordered = [summary_by_id[str(session.id)] for session in best]
    if plan.offset:
        ordered = ordered[plan.offset :]
    return plan._finalize(ordered)
