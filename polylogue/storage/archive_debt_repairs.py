"""Derived-repair debt counters built from derived-model status snapshots."""

from __future__ import annotations

from polylogue.maintenance_models import DerivedModelStatus


def session_product_repair_count(
    derived_statuses: dict[str, DerivedModelStatus],
) -> int:
    session_profile_rows = derived_statuses.get("session_profile_rows")
    session_profile_merged_fts = derived_statuses.get("session_profile_merged_fts")
    session_profile_evidence_fts = derived_statuses.get("session_profile_evidence_fts")
    session_profile_inference_fts = derived_statuses.get("session_profile_inference_fts")
    session_profile_enrichment_fts = derived_statuses.get("session_profile_enrichment_fts")
    session_work_event_inference = derived_statuses.get("session_work_event_inference")
    session_work_event_inference_fts = derived_statuses.get("session_work_event_inference_fts")
    session_phase_inference = derived_statuses.get("session_phase_inference")
    work_threads = derived_statuses.get("work_threads")
    work_threads_fts = derived_statuses.get("work_threads_fts")
    session_tag_rollups = derived_statuses.get("session_tag_rollups")
    day_session_summaries = derived_statuses.get("day_session_summaries")
    week_session_summaries = derived_statuses.get("week_session_summaries")
    if not all(
        status is not None
        for status in (
            session_profile_rows,
            session_profile_merged_fts,
            session_profile_evidence_fts,
            session_profile_inference_fts,
            session_profile_enrichment_fts,
            session_work_event_inference,
            session_work_event_inference_fts,
            session_phase_inference,
            work_threads,
            work_threads_fts,
            session_tag_rollups,
            day_session_summaries,
            week_session_summaries,
        )
    ):
        return 0
    return (
        max(0, int(session_profile_rows.pending_documents or 0))
        + max(0, int(session_profile_rows.pending_rows or 0))
        + max(0, int(session_profile_rows.stale_rows or 0))
        + max(0, int(session_profile_rows.orphan_rows or 0))
        + max(0, int(session_profile_merged_fts.pending_rows or 0))
        + max(0, int(session_profile_merged_fts.stale_rows or 0))
        + max(0, int(session_profile_evidence_fts.pending_rows or 0))
        + max(0, int(session_profile_evidence_fts.stale_rows or 0))
        + max(0, int(session_profile_inference_fts.pending_rows or 0))
        + max(0, int(session_profile_inference_fts.stale_rows or 0))
        + max(0, int(session_profile_enrichment_fts.pending_rows or 0))
        + max(0, int(session_profile_enrichment_fts.stale_rows or 0))
        + max(0, int(session_work_event_inference.pending_rows or 0))
        + max(0, int(session_work_event_inference.stale_rows or 0))
        + max(0, int(session_work_event_inference.orphan_rows or 0))
        + max(0, int(session_work_event_inference_fts.pending_rows or 0))
        + max(0, int(session_work_event_inference_fts.stale_rows or 0))
        + max(0, int(session_phase_inference.pending_rows or 0))
        + max(0, int(session_phase_inference.stale_rows or 0))
        + max(0, int(session_phase_inference.orphan_rows or 0))
        + max(0, int(work_threads.pending_documents or 0))
        + max(0, int(work_threads.stale_rows or 0))
        + max(0, int(work_threads.orphan_rows or 0))
        + max(0, int(work_threads_fts.pending_rows or 0))
        + max(0, int(work_threads_fts.stale_rows or 0))
        + max(0, int(session_tag_rollups.pending_rows or 0))
        + max(0, int(session_tag_rollups.stale_rows or 0))
        + max(0, int(day_session_summaries.pending_rows or 0))
        + max(0, int(day_session_summaries.stale_rows or 0))
        + max(0, int(week_session_summaries.pending_rows or 0))
        + max(0, int(week_session_summaries.stale_rows or 0))
    )


def action_event_repair_count(
    derived_statuses: dict[str, DerivedModelStatus],
) -> int:
    action_events = derived_statuses.get("action_events")
    action_events_fts = derived_statuses.get("action_events_fts")
    if action_events is None or action_events_fts is None:
        return 0
    return (
        max(0, int(action_events.pending_documents or 0))
        + max(0, int(action_events.stale_rows or 0))
        + max(0, int(action_events_fts.pending_rows or 0))
    )


def dangling_fts_repair_count(
    derived_statuses: dict[str, DerivedModelStatus],
) -> int:
    messages_fts = derived_statuses.get("messages_fts")
    return max(0, int(messages_fts.pending_rows or 0)) if messages_fts is not None else 0


__all__ = [
    "action_event_repair_count",
    "dangling_fts_repair_count",
    "session_product_repair_count",
]
