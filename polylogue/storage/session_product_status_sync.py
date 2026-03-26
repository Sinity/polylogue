"""Synchronous session-product status and repair-candidate queries."""

# ruff: noqa: F403, F405

from __future__ import annotations

import sqlite3

from polylogue.storage.store import SESSION_PRODUCT_MATERIALIZER_VERSION

from .session_product_status_sql import *


def session_profile_repair_candidate_ids_sync(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        SESSION_PROFILE_REPAIR_CANDIDATES_SQL,
        (SESSION_PRODUCT_MATERIALIZER_VERSION,),
    ).fetchall()
    return [str(row["conversation_id"]) for row in rows]


def session_product_status_sync(conn: sqlite3.Connection) -> dict[str, int | bool]:
    session_profiles_exists = bool(conn.execute(SESSION_PROFILES_EXISTS_SQL).fetchone())
    session_profiles_fts_exists = bool(conn.execute(SESSION_PROFILES_FTS_EXISTS_SQL).fetchone())
    session_profile_evidence_fts_exists = bool(conn.execute(SESSION_PROFILE_EVIDENCE_FTS_EXISTS_SQL).fetchone())
    session_profile_inference_fts_exists = bool(conn.execute(SESSION_PROFILE_INFERENCE_FTS_EXISTS_SQL).fetchone())
    session_profile_enrichment_fts_exists = bool(conn.execute(SESSION_PROFILE_ENRICHMENT_FTS_EXISTS_SQL).fetchone())
    session_work_events_exists = bool(conn.execute(SESSION_WORK_EVENTS_EXISTS_SQL).fetchone())
    session_work_events_fts_exists = bool(conn.execute(SESSION_WORK_EVENTS_FTS_EXISTS_SQL).fetchone())
    session_phases_exists = bool(conn.execute(SESSION_PHASES_EXISTS_SQL).fetchone())
    work_threads_exists = bool(conn.execute(WORK_THREADS_EXISTS_SQL).fetchone())
    work_threads_fts_exists = bool(conn.execute(WORK_THREADS_FTS_EXISTS_SQL).fetchone())
    session_tag_rollups_exists = bool(conn.execute(SESSION_TAG_ROLLUPS_EXISTS_SQL).fetchone())
    day_session_summaries_exists = bool(conn.execute(DAY_SESSION_SUMMARIES_EXISTS_SQL).fetchone())

    total_conversations = int(conn.execute(TOTAL_CONVERSATIONS_SQL).fetchone()[0] or 0)
    root_threads = int(conn.execute(ROOT_THREAD_COUNT_SQL).fetchone()[0] or 0)
    profile_count = int(conn.execute(SESSION_PROFILE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    profile_merged_fts_count = int(conn.execute(SESSION_PROFILE_MERGED_FTS_DOC_COUNT_SQL).fetchone()[0] or 0) if session_profiles_fts_exists else 0
    profile_merged_fts_duplicate_count = int(conn.execute(SESSION_PROFILE_MERGED_FTS_DUPLICATE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_fts_exists else 0
    evidence_profile_fts_count = int(conn.execute(SESSION_PROFILE_EVIDENCE_FTS_DOC_COUNT_SQL).fetchone()[0] or 0) if session_profile_evidence_fts_exists else 0
    evidence_profile_fts_duplicate_count = int(conn.execute(SESSION_PROFILE_EVIDENCE_FTS_DUPLICATE_COUNT_SQL).fetchone()[0] or 0) if session_profile_evidence_fts_exists else 0
    inference_profile_fts_count = int(conn.execute(SESSION_PROFILE_INFERENCE_FTS_DOC_COUNT_SQL).fetchone()[0] or 0) if session_profile_inference_fts_exists else 0
    inference_profile_fts_duplicate_count = int(conn.execute(SESSION_PROFILE_INFERENCE_FTS_DUPLICATE_COUNT_SQL).fetchone()[0] or 0) if session_profile_inference_fts_exists else 0
    enrichment_profile_fts_count = int(conn.execute(SESSION_PROFILE_ENRICHMENT_FTS_DOC_COUNT_SQL).fetchone()[0] or 0) if session_profile_enrichment_fts_exists else 0
    enrichment_profile_fts_duplicate_count = int(conn.execute(SESSION_PROFILE_ENRICHMENT_FTS_DUPLICATE_COUNT_SQL).fetchone()[0] or 0) if session_profile_enrichment_fts_exists else 0
    work_event_inference_count = int(conn.execute(SESSION_WORK_EVENT_COUNT_SQL).fetchone()[0] or 0) if session_work_events_exists else 0
    work_event_inference_fts_count = int(conn.execute(SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL).fetchone()[0] or 0) if session_work_events_fts_exists else 0
    work_event_inference_fts_duplicate_count = int(conn.execute(SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL).fetchone()[0] or 0) if session_work_events_fts_exists else 0
    phase_inference_count = int(conn.execute(SESSION_PHASE_COUNT_SQL).fetchone()[0] or 0) if session_phases_exists else 0
    thread_count = int(conn.execute(WORK_THREAD_COUNT_SQL).fetchone()[0] or 0) if work_threads_exists else 0
    thread_fts_count = int(conn.execute(WORK_THREAD_FTS_DOC_COUNT_SQL).fetchone()[0] or 0) if work_threads_fts_exists else 0
    thread_fts_duplicate_count = int(conn.execute(WORK_THREAD_FTS_DUPLICATE_COUNT_SQL).fetchone()[0] or 0) if work_threads_fts_exists else 0
    tag_rollup_count = int(conn.execute(SESSION_TAG_ROLLUP_COUNT_SQL).fetchone()[0] or 0) if session_tag_rollups_exists else 0
    day_summary_count = int(conn.execute(DAY_SESSION_SUMMARY_COUNT_SQL).fetchone()[0] or 0) if day_session_summaries_exists else 0
    missing_profile_count = int(conn.execute(MISSING_SESSION_PROFILE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else total_conversations
    stale_profile_count = int(conn.execute(STALE_SESSION_PROFILE_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0) if session_profiles_exists else total_conversations
    orphan_profile_count = int(conn.execute(ORPHAN_SESSION_PROFILE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    expected_work_event_count = int(conn.execute(EXPECTED_WORK_EVENT_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    expected_phase_count = int(conn.execute(EXPECTED_PHASE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    stale_work_event_count = int(conn.execute(STALE_WORK_EVENT_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0) if session_work_events_exists else expected_work_event_count
    orphan_work_event_count = int(conn.execute(ORPHAN_SESSION_WORK_EVENT_COUNT_SQL).fetchone()[0] or 0) if session_work_events_exists else 0
    stale_phase_count = int(conn.execute(STALE_SESSION_PHASE_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0) if session_phases_exists else expected_phase_count
    orphan_phase_count = int(conn.execute(ORPHAN_SESSION_PHASE_COUNT_SQL).fetchone()[0] or 0) if session_phases_exists else 0
    stale_thread_count = int(conn.execute(STALE_WORK_THREAD_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0) if work_threads_exists else root_threads
    orphan_thread_count = int(conn.execute(ORPHAN_WORK_THREAD_COUNT_SQL).fetchone()[0] or 0) if work_threads_exists else 0
    expected_tag_rollup_count = int(conn.execute(EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    stale_tag_rollup_count = int(conn.execute(STALE_SESSION_TAG_ROLLUP_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0) if session_tag_rollups_exists else expected_tag_rollup_count
    expected_day_summary_count = int(conn.execute(EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    stale_day_summary_count = int(conn.execute(STALE_DAY_SESSION_SUMMARY_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0) if day_session_summaries_exists else expected_day_summary_count
    return {
        "total_conversations": total_conversations,
        "root_threads": root_threads,
        "profile_row_count": profile_count,
        "profile_merged_fts_count": profile_merged_fts_count,
        "profile_merged_fts_duplicate_count": profile_merged_fts_duplicate_count,
        "profile_evidence_fts_count": evidence_profile_fts_count,
        "profile_evidence_fts_duplicate_count": evidence_profile_fts_duplicate_count,
        "profile_inference_fts_count": inference_profile_fts_count,
        "profile_inference_fts_duplicate_count": inference_profile_fts_duplicate_count,
        "profile_enrichment_fts_count": enrichment_profile_fts_count,
        "profile_enrichment_fts_duplicate_count": enrichment_profile_fts_duplicate_count,
        "work_event_inference_count": work_event_inference_count,
        "work_event_inference_fts_count": work_event_inference_fts_count,
        "work_event_inference_fts_duplicate_count": work_event_inference_fts_duplicate_count,
        "phase_inference_count": phase_inference_count,
        "thread_count": thread_count,
        "thread_fts_count": thread_fts_count,
        "thread_fts_duplicate_count": thread_fts_duplicate_count,
        "tag_rollup_count": tag_rollup_count,
        "day_summary_count": day_summary_count,
        "missing_profile_row_count": missing_profile_count,
        "stale_profile_row_count": stale_profile_count,
        "orphan_profile_row_count": orphan_profile_count,
        "expected_work_event_inference_count": expected_work_event_count,
        "stale_work_event_inference_count": stale_work_event_count,
        "orphan_work_event_inference_count": orphan_work_event_count,
        "expected_phase_inference_count": expected_phase_count,
        "stale_phase_inference_count": stale_phase_count,
        "orphan_phase_inference_count": orphan_phase_count,
        "stale_thread_count": stale_thread_count,
        "orphan_thread_count": orphan_thread_count,
        "expected_tag_rollup_count": expected_tag_rollup_count,
        "stale_tag_rollup_count": stale_tag_rollup_count,
        "expected_day_summary_count": expected_day_summary_count,
        "stale_day_summary_count": stale_day_summary_count,
        "profile_rows_ready": session_profiles_exists and missing_profile_count == 0 and stale_profile_count == 0 and orphan_profile_count == 0,
        "profile_merged_fts_ready": session_profiles_fts_exists and profile_merged_fts_count == profile_count and profile_merged_fts_duplicate_count == 0,
        "profile_evidence_fts_ready": session_profile_evidence_fts_exists and evidence_profile_fts_count == profile_count and evidence_profile_fts_duplicate_count == 0,
        "profile_inference_fts_ready": session_profile_inference_fts_exists and inference_profile_fts_count == profile_count and inference_profile_fts_duplicate_count == 0,
        "profile_enrichment_fts_ready": session_profile_enrichment_fts_exists and enrichment_profile_fts_count == profile_count and enrichment_profile_fts_duplicate_count == 0,
        "work_event_inference_rows_ready": session_work_events_exists and work_event_inference_count == expected_work_event_count and stale_work_event_count == 0 and orphan_work_event_count == 0,
        "work_event_inference_fts_ready": session_work_events_fts_exists and work_event_inference_fts_count == work_event_inference_count and work_event_inference_fts_duplicate_count == 0,
        "phase_inference_rows_ready": session_phases_exists and phase_inference_count == expected_phase_count and stale_phase_count == 0 and orphan_phase_count == 0,
        "threads_ready": work_threads_exists and thread_count == root_threads and stale_thread_count == 0 and orphan_thread_count == 0,
        "threads_fts_ready": work_threads_fts_exists and thread_fts_count == thread_count and thread_fts_duplicate_count == 0,
        "tag_rollups_ready": session_tag_rollups_exists and tag_rollup_count == expected_tag_rollup_count and stale_tag_rollup_count == 0,
        "day_summaries_ready": day_session_summaries_exists and day_summary_count == expected_day_summary_count and stale_day_summary_count == 0,
        "week_summaries_ready": day_session_summaries_exists and day_summary_count == expected_day_summary_count and stale_day_summary_count == 0,
    }


__all__ = [
    "session_product_status_sync",
    "session_profile_repair_candidate_ids_sync",
]
