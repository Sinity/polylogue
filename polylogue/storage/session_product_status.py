"""Status, drift, and candidate discovery for durable session products."""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.storage.store import SESSION_PRODUCT_MATERIALIZER_VERSION

from .session_product_aggregates import _PROFILE_BUCKET_DAY_SQL

_SESSION_PROFILES_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
_SESSION_PROFILES_FTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles_fts'"
_SESSION_WORK_EVENTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_work_events'"
_SESSION_WORK_EVENTS_FTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_work_events_fts'"
_SESSION_PHASES_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_phases'"
_WORK_THREADS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='work_threads'"
_WORK_THREADS_FTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='work_threads_fts'"
_SESSION_TAG_ROLLUPS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_tag_rollups'"
_DAY_SESSION_SUMMARIES_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='day_session_summaries'"
_SESSION_PROFILE_COUNT_SQL = "SELECT COUNT(*) FROM session_profiles"
_SESSION_PROFILE_FTS_DOC_COUNT_SQL = "SELECT COUNT(DISTINCT conversation_id) FROM session_profiles_fts"
_SESSION_PROFILE_FTS_DUPLICATE_COUNT_SQL = "SELECT COUNT(*) - COUNT(DISTINCT conversation_id) FROM session_profiles_fts"
_SESSION_WORK_EVENT_COUNT_SQL = "SELECT COUNT(*) FROM session_work_events"
_SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL = "SELECT COUNT(DISTINCT event_id) FROM session_work_events_fts"
_SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL = "SELECT COUNT(*) - COUNT(DISTINCT event_id) FROM session_work_events_fts"
_SESSION_PHASE_COUNT_SQL = "SELECT COUNT(*) FROM session_phases"
_WORK_THREAD_COUNT_SQL = "SELECT COUNT(*) FROM work_threads"
_WORK_THREAD_FTS_DOC_COUNT_SQL = "SELECT COUNT(DISTINCT thread_id) FROM work_threads_fts"
_WORK_THREAD_FTS_DUPLICATE_COUNT_SQL = "SELECT COUNT(*) - COUNT(DISTINCT thread_id) FROM work_threads_fts"
_SESSION_TAG_ROLLUP_COUNT_SQL = "SELECT COUNT(*) FROM session_tag_rollups"
_DAY_SESSION_SUMMARY_COUNT_SQL = "SELECT COUNT(*) FROM day_session_summaries"
_TOTAL_CONVERSATIONS_SQL = "SELECT COUNT(*) FROM conversations"
_ROOT_THREAD_COUNT_SQL = """
    SELECT COUNT(*)
    FROM conversations c
    LEFT JOIN conversations parent ON c.parent_conversation_id = parent.conversation_id
    WHERE parent.conversation_id IS NULL
"""
_MISSING_SESSION_PROFILE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM conversations c
    LEFT JOIN session_profiles sp ON sp.conversation_id = c.conversation_id
    WHERE sp.conversation_id IS NULL
"""
_STALE_SESSION_PROFILE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM conversations c
    JOIN session_profiles sp ON sp.conversation_id = c.conversation_id
    WHERE sp.materializer_version != ?
       OR ABS(COALESCE(sp.source_sort_key, 0.0) - COALESCE(c.sort_key, 0.0)) > 0.000001
"""
_ORPHAN_SESSION_PROFILE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_profiles sp
    LEFT JOIN conversations c ON c.conversation_id = sp.conversation_id
    WHERE c.conversation_id IS NULL
"""
_EXPECTED_WORK_EVENT_COUNT_SQL = "SELECT COALESCE(SUM(work_event_count), 0) FROM session_profiles"
_EXPECTED_PHASE_COUNT_SQL = "SELECT COALESCE(SUM(phase_count), 0) FROM session_profiles"
_STALE_WORK_EVENT_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_work_events swe
    JOIN conversations c ON c.conversation_id = swe.conversation_id
    WHERE swe.materializer_version != ?
       OR ABS(COALESCE(swe.source_sort_key, 0.0) - COALESCE(c.sort_key, 0.0)) > 0.000001
"""
_ORPHAN_SESSION_WORK_EVENT_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_work_events swe
    LEFT JOIN conversations c ON c.conversation_id = swe.conversation_id
    WHERE c.conversation_id IS NULL
"""
_STALE_SESSION_PHASE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_phases sph
    JOIN conversations c ON c.conversation_id = sph.conversation_id
    WHERE sph.materializer_version != ?
       OR ABS(COALESCE(sph.source_sort_key, 0.0) - COALESCE(c.sort_key, 0.0)) > 0.000001
"""
_ORPHAN_SESSION_PHASE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_phases sph
    LEFT JOIN conversations c ON c.conversation_id = sph.conversation_id
    WHERE c.conversation_id IS NULL
"""
_STALE_WORK_THREAD_COUNT_SQL = """
    WITH RECURSIVE roots(root_id) AS (
        SELECT c.conversation_id
        FROM conversations c
        LEFT JOIN conversations parent ON c.parent_conversation_id = parent.conversation_id
        WHERE parent.conversation_id IS NULL
    ),
    descendants(root_id, conversation_id) AS (
        SELECT root_id, root_id FROM roots
        UNION ALL
        SELECT d.root_id, c.conversation_id
        FROM conversations c
        JOIN descendants d ON c.parent_conversation_id = d.conversation_id
    )
    SELECT COUNT(*)
    FROM work_threads wt
    WHERE wt.materializer_version != ?
       OR EXISTS (
            SELECT 1
            FROM descendants d
            JOIN session_profiles sp ON sp.conversation_id = d.conversation_id
            WHERE d.root_id = wt.thread_id
              AND sp.materialized_at > wt.materialized_at
       )
"""
_ORPHAN_WORK_THREAD_COUNT_SQL = """
    SELECT COUNT(*)
    FROM work_threads wt
    LEFT JOIN conversations c ON c.conversation_id = wt.root_id
    WHERE c.conversation_id IS NULL
"""
_EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL = f"""
    SELECT COUNT(*) FROM (
        SELECT sp.provider_name, {_PROFILE_BUCKET_DAY_SQL} AS bucket_day
        FROM session_profiles sp
        WHERE {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL
        GROUP BY sp.provider_name, bucket_day
    )
"""
_STALE_DAY_SESSION_SUMMARY_COUNT_SQL = f"""
    WITH expected AS (
        SELECT
            sp.provider_name AS provider_name,
            {_PROFILE_BUCKET_DAY_SQL} AS bucket_day,
            MAX(sp.materialized_at) AS max_profile_materialized_at
        FROM session_profiles sp
        WHERE {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL
        GROUP BY sp.provider_name, bucket_day
    )
    SELECT COUNT(*)
    FROM day_session_summaries dss
    LEFT JOIN expected e
      ON e.provider_name = dss.provider_name
     AND e.bucket_day = dss.day
    WHERE dss.materializer_version != ?
       OR e.bucket_day IS NULL
       OR COALESCE(e.max_profile_materialized_at, '') > COALESCE(dss.materialized_at, '')
"""
_EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL = f"""
    WITH tag_rows AS (
        SELECT sp.provider_name, {_PROFILE_BUCKET_DAY_SQL} AS bucket_day, tag.value AS tag
        FROM session_profiles sp, json_each(COALESCE(sp.tags_json, '[]')) tag
        WHERE {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL AND tag.value IS NOT NULL AND tag.value != ''
        UNION
        SELECT sp.provider_name, {_PROFILE_BUCKET_DAY_SQL} AS bucket_day, tag.value AS tag
        FROM session_profiles sp, json_each(COALESCE(sp.auto_tags_json, '[]')) tag
        WHERE {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL AND tag.value IS NOT NULL AND tag.value != ''
    )
    SELECT COUNT(*) FROM (
        SELECT provider_name, bucket_day, tag
        FROM tag_rows
        GROUP BY provider_name, bucket_day, tag
    )
"""
_STALE_SESSION_TAG_ROLLUP_COUNT_SQL = f"""
    WITH tag_rows AS (
        SELECT
            sp.provider_name AS provider_name,
            {_PROFILE_BUCKET_DAY_SQL} AS bucket_day,
            tag.value AS tag,
            sp.materialized_at AS profile_materialized_at
        FROM session_profiles sp, json_each(COALESCE(sp.tags_json, '[]')) tag
        WHERE {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL AND tag.value IS NOT NULL AND tag.value != ''
        UNION ALL
        SELECT
            sp.provider_name AS provider_name,
            {_PROFILE_BUCKET_DAY_SQL} AS bucket_day,
            tag.value AS tag,
            sp.materialized_at AS profile_materialized_at
        FROM session_profiles sp, json_each(COALESCE(sp.auto_tags_json, '[]')) tag
        WHERE {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL AND tag.value IS NOT NULL AND tag.value != ''
    ),
    expected AS (
        SELECT provider_name, bucket_day, tag, MAX(profile_materialized_at) AS max_profile_materialized_at
        FROM tag_rows
        GROUP BY provider_name, bucket_day, tag
    )
    SELECT COUNT(*)
    FROM session_tag_rollups str
    LEFT JOIN expected e
      ON e.provider_name = str.provider_name
     AND e.bucket_day = str.bucket_day
     AND e.tag = str.tag
    WHERE str.materializer_version != ?
       OR e.tag IS NULL
       OR COALESCE(e.max_profile_materialized_at, '') > COALESCE(str.materialized_at, '')
"""
_SESSION_PROFILE_REPAIR_CANDIDATES_SQL = """
    SELECT c.conversation_id
    FROM conversations c
    LEFT JOIN session_profiles sp ON sp.conversation_id = c.conversation_id
    WHERE sp.conversation_id IS NULL
       OR sp.materializer_version != ?
       OR COALESCE(sp.source_updated_at, '') != COALESCE(c.updated_at, '')
    ORDER BY c.conversation_id
"""


def session_profile_repair_candidate_ids_sync(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        _SESSION_PROFILE_REPAIR_CANDIDATES_SQL,
        (SESSION_PRODUCT_MATERIALIZER_VERSION,),
    ).fetchall()
    return [str(row["conversation_id"]) for row in rows]


async def session_profile_repair_candidate_ids_async(conn: aiosqlite.Connection) -> list[str]:
    rows = await (
        await conn.execute(
            _SESSION_PROFILE_REPAIR_CANDIDATES_SQL,
            (SESSION_PRODUCT_MATERIALIZER_VERSION,),
        )
    ).fetchall()
    return [str(row["conversation_id"]) for row in rows]


def session_product_status_sync(conn: sqlite3.Connection) -> dict[str, int | bool]:
    session_profiles_exists = bool(conn.execute(_SESSION_PROFILES_EXISTS_SQL).fetchone())
    session_profiles_fts_exists = bool(conn.execute(_SESSION_PROFILES_FTS_EXISTS_SQL).fetchone())
    session_work_events_exists = bool(conn.execute(_SESSION_WORK_EVENTS_EXISTS_SQL).fetchone())
    session_work_events_fts_exists = bool(conn.execute(_SESSION_WORK_EVENTS_FTS_EXISTS_SQL).fetchone())
    session_phases_exists = bool(conn.execute(_SESSION_PHASES_EXISTS_SQL).fetchone())
    work_threads_exists = bool(conn.execute(_WORK_THREADS_EXISTS_SQL).fetchone())
    work_threads_fts_exists = bool(conn.execute(_WORK_THREADS_FTS_EXISTS_SQL).fetchone())
    session_tag_rollups_exists = bool(conn.execute(_SESSION_TAG_ROLLUPS_EXISTS_SQL).fetchone())
    day_session_summaries_exists = bool(conn.execute(_DAY_SESSION_SUMMARIES_EXISTS_SQL).fetchone())

    total_conversations = int(conn.execute(_TOTAL_CONVERSATIONS_SQL).fetchone()[0] or 0)
    root_threads = int(conn.execute(_ROOT_THREAD_COUNT_SQL).fetchone()[0] or 0)
    profile_count = int(conn.execute(_SESSION_PROFILE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    profile_fts_count = int(conn.execute(_SESSION_PROFILE_FTS_DOC_COUNT_SQL).fetchone()[0] or 0) if session_profiles_fts_exists else 0
    profile_fts_duplicate_count = int(conn.execute(_SESSION_PROFILE_FTS_DUPLICATE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_fts_exists else 0
    work_event_count = int(conn.execute(_SESSION_WORK_EVENT_COUNT_SQL).fetchone()[0] or 0) if session_work_events_exists else 0
    work_event_fts_count = int(conn.execute(_SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL).fetchone()[0] or 0) if session_work_events_fts_exists else 0
    work_event_fts_duplicate_count = int(conn.execute(_SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL).fetchone()[0] or 0) if session_work_events_fts_exists else 0
    phase_count = int(conn.execute(_SESSION_PHASE_COUNT_SQL).fetchone()[0] or 0) if session_phases_exists else 0
    thread_count = int(conn.execute(_WORK_THREAD_COUNT_SQL).fetchone()[0] or 0) if work_threads_exists else 0
    thread_fts_count = int(conn.execute(_WORK_THREAD_FTS_DOC_COUNT_SQL).fetchone()[0] or 0) if work_threads_fts_exists else 0
    thread_fts_duplicate_count = int(conn.execute(_WORK_THREAD_FTS_DUPLICATE_COUNT_SQL).fetchone()[0] or 0) if work_threads_fts_exists else 0
    tag_rollup_count = int(conn.execute(_SESSION_TAG_ROLLUP_COUNT_SQL).fetchone()[0] or 0) if session_tag_rollups_exists else 0
    day_summary_count = int(conn.execute(_DAY_SESSION_SUMMARY_COUNT_SQL).fetchone()[0] or 0) if day_session_summaries_exists else 0
    missing_profile_count = int(conn.execute(_MISSING_SESSION_PROFILE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else total_conversations
    stale_profile_count = int(conn.execute(_STALE_SESSION_PROFILE_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0) if session_profiles_exists else total_conversations
    orphan_profile_count = int(conn.execute(_ORPHAN_SESSION_PROFILE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    expected_work_event_count = int(conn.execute(_EXPECTED_WORK_EVENT_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    expected_phase_count = int(conn.execute(_EXPECTED_PHASE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    stale_work_event_count = int(conn.execute(_STALE_WORK_EVENT_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0) if session_work_events_exists else expected_work_event_count
    orphan_work_event_count = int(conn.execute(_ORPHAN_SESSION_WORK_EVENT_COUNT_SQL).fetchone()[0] or 0) if session_work_events_exists else 0
    stale_phase_count = int(conn.execute(_STALE_SESSION_PHASE_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0) if session_phases_exists else expected_phase_count
    orphan_phase_count = int(conn.execute(_ORPHAN_SESSION_PHASE_COUNT_SQL).fetchone()[0] or 0) if session_phases_exists else 0
    stale_thread_count = int(conn.execute(_STALE_WORK_THREAD_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0) if work_threads_exists else root_threads
    orphan_thread_count = int(conn.execute(_ORPHAN_WORK_THREAD_COUNT_SQL).fetchone()[0] or 0) if work_threads_exists else 0
    expected_tag_rollup_count = int(conn.execute(_EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    stale_tag_rollup_count = int(conn.execute(_STALE_SESSION_TAG_ROLLUP_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0) if session_tag_rollups_exists else expected_tag_rollup_count
    expected_day_summary_count = int(conn.execute(_EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    stale_day_summary_count = int(conn.execute(_STALE_DAY_SESSION_SUMMARY_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0) if day_session_summaries_exists else expected_day_summary_count
    return {
        "total_conversations": total_conversations,
        "root_threads": root_threads,
        "profile_count": profile_count,
        "profile_fts_count": profile_fts_count,
        "profile_fts_duplicate_count": profile_fts_duplicate_count,
        "work_event_count": work_event_count,
        "work_event_fts_count": work_event_fts_count,
        "work_event_fts_duplicate_count": work_event_fts_duplicate_count,
        "phase_count": phase_count,
        "thread_count": thread_count,
        "thread_fts_count": thread_fts_count,
        "thread_fts_duplicate_count": thread_fts_duplicate_count,
        "tag_rollup_count": tag_rollup_count,
        "day_summary_count": day_summary_count,
        "missing_profile_count": missing_profile_count,
        "stale_profile_count": stale_profile_count,
        "orphan_profile_count": orphan_profile_count,
        "expected_work_event_count": expected_work_event_count,
        "stale_work_event_count": stale_work_event_count,
        "orphan_work_event_count": orphan_work_event_count,
        "expected_phase_count": expected_phase_count,
        "stale_phase_count": stale_phase_count,
        "orphan_phase_count": orphan_phase_count,
        "stale_thread_count": stale_thread_count,
        "orphan_thread_count": orphan_thread_count,
        "expected_tag_rollup_count": expected_tag_rollup_count,
        "stale_tag_rollup_count": stale_tag_rollup_count,
        "expected_day_summary_count": expected_day_summary_count,
        "stale_day_summary_count": stale_day_summary_count,
        "profiles_ready": session_profiles_exists and missing_profile_count == 0 and stale_profile_count == 0 and orphan_profile_count == 0,
        "profiles_fts_ready": session_profiles_fts_exists and profile_fts_count == profile_count and profile_fts_duplicate_count == 0,
        "work_events_ready": session_work_events_exists and work_event_count == expected_work_event_count and stale_work_event_count == 0 and orphan_work_event_count == 0,
        "work_events_fts_ready": session_work_events_fts_exists and work_event_fts_count == work_event_count and work_event_fts_duplicate_count == 0,
        "phases_ready": session_phases_exists and phase_count == expected_phase_count and stale_phase_count == 0 and orphan_phase_count == 0,
        "threads_ready": work_threads_exists and thread_count == root_threads and stale_thread_count == 0 and orphan_thread_count == 0,
        "threads_fts_ready": work_threads_fts_exists and thread_fts_count == thread_count and thread_fts_duplicate_count == 0,
        "tag_rollups_ready": session_tag_rollups_exists and tag_rollup_count == expected_tag_rollup_count and stale_tag_rollup_count == 0,
        "day_summaries_ready": day_session_summaries_exists and day_summary_count == expected_day_summary_count and stale_day_summary_count == 0,
        "week_summaries_ready": day_session_summaries_exists and day_summary_count == expected_day_summary_count and stale_day_summary_count == 0,
    }


async def session_product_status_async(conn: aiosqlite.Connection) -> dict[str, int | bool]:
    def to_int(row: tuple[object, ...] | None) -> int:
        return int(row[0] or 0) if row else 0

    session_profiles_exists = bool(await (await conn.execute(_SESSION_PROFILES_EXISTS_SQL)).fetchone())
    session_profiles_fts_exists = bool(await (await conn.execute(_SESSION_PROFILES_FTS_EXISTS_SQL)).fetchone())
    session_work_events_exists = bool(await (await conn.execute(_SESSION_WORK_EVENTS_EXISTS_SQL)).fetchone())
    session_work_events_fts_exists = bool(await (await conn.execute(_SESSION_WORK_EVENTS_FTS_EXISTS_SQL)).fetchone())
    session_phases_exists = bool(await (await conn.execute(_SESSION_PHASES_EXISTS_SQL)).fetchone())
    work_threads_exists = bool(await (await conn.execute(_WORK_THREADS_EXISTS_SQL)).fetchone())
    work_threads_fts_exists = bool(await (await conn.execute(_WORK_THREADS_FTS_EXISTS_SQL)).fetchone())
    session_tag_rollups_exists = bool(await (await conn.execute(_SESSION_TAG_ROLLUPS_EXISTS_SQL)).fetchone())
    day_session_summaries_exists = bool(await (await conn.execute(_DAY_SESSION_SUMMARIES_EXISTS_SQL)).fetchone())
    total_conversations = to_int(await (await conn.execute(_TOTAL_CONVERSATIONS_SQL)).fetchone())
    root_threads = to_int(await (await conn.execute(_ROOT_THREAD_COUNT_SQL)).fetchone())
    profile_count = to_int(await (await conn.execute(_SESSION_PROFILE_COUNT_SQL)).fetchone()) if session_profiles_exists else 0
    profile_fts_count = to_int(await (await conn.execute(_SESSION_PROFILE_FTS_DOC_COUNT_SQL)).fetchone()) if session_profiles_fts_exists else 0
    profile_fts_duplicate_count = to_int(await (await conn.execute(_SESSION_PROFILE_FTS_DUPLICATE_COUNT_SQL)).fetchone()) if session_profiles_fts_exists else 0
    work_event_count = to_int(await (await conn.execute(_SESSION_WORK_EVENT_COUNT_SQL)).fetchone()) if session_work_events_exists else 0
    work_event_fts_count = to_int(await (await conn.execute(_SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL)).fetchone()) if session_work_events_fts_exists else 0
    work_event_fts_duplicate_count = to_int(await (await conn.execute(_SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL)).fetchone()) if session_work_events_fts_exists else 0
    phase_count = to_int(await (await conn.execute(_SESSION_PHASE_COUNT_SQL)).fetchone()) if session_phases_exists else 0
    thread_count = to_int(await (await conn.execute(_WORK_THREAD_COUNT_SQL)).fetchone()) if work_threads_exists else 0
    thread_fts_count = to_int(await (await conn.execute(_WORK_THREAD_FTS_DOC_COUNT_SQL)).fetchone()) if work_threads_fts_exists else 0
    thread_fts_duplicate_count = to_int(await (await conn.execute(_WORK_THREAD_FTS_DUPLICATE_COUNT_SQL)).fetchone()) if work_threads_fts_exists else 0
    tag_rollup_count = to_int(await (await conn.execute(_SESSION_TAG_ROLLUP_COUNT_SQL)).fetchone()) if session_tag_rollups_exists else 0
    day_summary_count = to_int(await (await conn.execute(_DAY_SESSION_SUMMARY_COUNT_SQL)).fetchone()) if day_session_summaries_exists else 0
    missing_profile_count = to_int(await (await conn.execute(_MISSING_SESSION_PROFILE_COUNT_SQL)).fetchone()) if session_profiles_exists else total_conversations
    stale_profile_count = to_int(await (await conn.execute(_STALE_SESSION_PROFILE_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,))).fetchone()) if session_profiles_exists else total_conversations
    orphan_profile_count = to_int(await (await conn.execute(_ORPHAN_SESSION_PROFILE_COUNT_SQL)).fetchone()) if session_profiles_exists else 0
    expected_work_event_count = to_int(await (await conn.execute(_EXPECTED_WORK_EVENT_COUNT_SQL)).fetchone()) if session_profiles_exists else 0
    expected_phase_count = to_int(await (await conn.execute(_EXPECTED_PHASE_COUNT_SQL)).fetchone()) if session_profiles_exists else 0
    stale_work_event_count = to_int(await (await conn.execute(_STALE_WORK_EVENT_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,))).fetchone()) if session_work_events_exists else expected_work_event_count
    orphan_work_event_count = to_int(await (await conn.execute(_ORPHAN_SESSION_WORK_EVENT_COUNT_SQL)).fetchone()) if session_work_events_exists else 0
    stale_phase_count = to_int(await (await conn.execute(_STALE_SESSION_PHASE_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,))).fetchone()) if session_phases_exists else expected_phase_count
    orphan_phase_count = to_int(await (await conn.execute(_ORPHAN_SESSION_PHASE_COUNT_SQL)).fetchone()) if session_phases_exists else 0
    stale_thread_count = to_int(await (await conn.execute(_STALE_WORK_THREAD_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,))).fetchone()) if work_threads_exists else root_threads
    orphan_thread_count = to_int(await (await conn.execute(_ORPHAN_WORK_THREAD_COUNT_SQL)).fetchone()) if work_threads_exists else 0
    expected_tag_rollup_count = to_int(await (await conn.execute(_EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL)).fetchone()) if session_profiles_exists else 0
    stale_tag_rollup_count = to_int(await (await conn.execute(_STALE_SESSION_TAG_ROLLUP_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,))).fetchone()) if session_tag_rollups_exists else expected_tag_rollup_count
    expected_day_summary_count = to_int(await (await conn.execute(_EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL)).fetchone()) if session_profiles_exists else 0
    stale_day_summary_count = to_int(await (await conn.execute(_STALE_DAY_SESSION_SUMMARY_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,))).fetchone()) if day_session_summaries_exists else expected_day_summary_count
    return {
        "total_conversations": total_conversations,
        "root_threads": root_threads,
        "profile_count": profile_count,
        "profile_fts_count": profile_fts_count,
        "profile_fts_duplicate_count": profile_fts_duplicate_count,
        "work_event_count": work_event_count,
        "work_event_fts_count": work_event_fts_count,
        "work_event_fts_duplicate_count": work_event_fts_duplicate_count,
        "phase_count": phase_count,
        "thread_count": thread_count,
        "thread_fts_count": thread_fts_count,
        "thread_fts_duplicate_count": thread_fts_duplicate_count,
        "tag_rollup_count": tag_rollup_count,
        "day_summary_count": day_summary_count,
        "missing_profile_count": missing_profile_count,
        "stale_profile_count": stale_profile_count,
        "orphan_profile_count": orphan_profile_count,
        "expected_work_event_count": expected_work_event_count,
        "stale_work_event_count": stale_work_event_count,
        "orphan_work_event_count": orphan_work_event_count,
        "expected_phase_count": expected_phase_count,
        "stale_phase_count": stale_phase_count,
        "orphan_phase_count": orphan_phase_count,
        "stale_thread_count": stale_thread_count,
        "orphan_thread_count": orphan_thread_count,
        "expected_tag_rollup_count": expected_tag_rollup_count,
        "stale_tag_rollup_count": stale_tag_rollup_count,
        "expected_day_summary_count": expected_day_summary_count,
        "stale_day_summary_count": stale_day_summary_count,
        "profiles_ready": session_profiles_exists and missing_profile_count == 0 and stale_profile_count == 0 and orphan_profile_count == 0,
        "profiles_fts_ready": session_profiles_fts_exists and profile_fts_count == profile_count and profile_fts_duplicate_count == 0,
        "work_events_ready": session_work_events_exists and work_event_count == expected_work_event_count and stale_work_event_count == 0 and orphan_work_event_count == 0,
        "work_events_fts_ready": session_work_events_fts_exists and work_event_fts_count == work_event_count and work_event_fts_duplicate_count == 0,
        "phases_ready": session_phases_exists and phase_count == expected_phase_count and stale_phase_count == 0 and orphan_phase_count == 0,
        "threads_ready": work_threads_exists and thread_count == root_threads and stale_thread_count == 0 and orphan_thread_count == 0,
        "threads_fts_ready": work_threads_fts_exists and thread_fts_count == thread_count and thread_fts_duplicate_count == 0,
        "tag_rollups_ready": session_tag_rollups_exists and tag_rollup_count == expected_tag_rollup_count and stale_tag_rollup_count == 0,
        "day_summaries_ready": day_session_summaries_exists and day_summary_count == expected_day_summary_count and stale_day_summary_count == 0,
        "week_summaries_ready": day_session_summaries_exists and day_summary_count == expected_day_summary_count and stale_day_summary_count == 0,
    }


__all__ = [
    "session_product_status_async",
    "session_product_status_sync",
    "session_profile_repair_candidate_ids_async",
    "session_profile_repair_candidate_ids_sync",
]
