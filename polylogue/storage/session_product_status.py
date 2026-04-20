"""Canonical session-product status and repair-candidate queries."""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.storage.session_product_aggregates import _PROFILE_BUCKET_DAY_SQL
from polylogue.storage.session_product_runtime import SessionProductStatusSnapshot
from polylogue.storage.store import SESSION_PRODUCT_MATERIALIZER_VERSION

# ---------------------------------------------------------------------------
# SQL constants for session-product status and drift checks
# ---------------------------------------------------------------------------

SESSION_PROFILES_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
SESSION_PROFILES_FTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles_fts'"
SESSION_PROFILE_EVIDENCE_FTS_EXISTS_SQL = (
    "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profile_evidence_fts'"
)
SESSION_PROFILE_INFERENCE_FTS_EXISTS_SQL = (
    "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profile_inference_fts'"
)
SESSION_PROFILE_ENRICHMENT_FTS_EXISTS_SQL = (
    "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profile_enrichment_fts'"
)
SESSION_WORK_EVENTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_work_events'"
SESSION_WORK_EVENTS_FTS_EXISTS_SQL = (
    "SELECT name FROM sqlite_master WHERE type='table' AND name='session_work_events_fts'"
)
SESSION_PHASES_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_phases'"
WORK_THREADS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='work_threads'"
WORK_THREADS_FTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='work_threads_fts'"
SESSION_TAG_ROLLUPS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_tag_rollups'"
DAY_SESSION_SUMMARIES_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='day_session_summaries'"
SESSION_PROFILE_COUNT_SQL = "SELECT COUNT(*) FROM session_profiles"
SESSION_PROFILE_MERGED_FTS_DOC_COUNT_SQL = "SELECT COUNT(DISTINCT conversation_id) FROM session_profiles_fts"
SESSION_PROFILE_MERGED_FTS_DUPLICATE_COUNT_SQL = (
    "SELECT COUNT(*) - COUNT(DISTINCT conversation_id) FROM session_profiles_fts"
)
SESSION_PROFILE_EVIDENCE_FTS_DOC_COUNT_SQL = "SELECT COUNT(DISTINCT conversation_id) FROM session_profile_evidence_fts"
SESSION_PROFILE_EVIDENCE_FTS_DUPLICATE_COUNT_SQL = (
    "SELECT COUNT(*) - COUNT(DISTINCT conversation_id) FROM session_profile_evidence_fts"
)
SESSION_PROFILE_INFERENCE_FTS_DOC_COUNT_SQL = (
    "SELECT COUNT(DISTINCT conversation_id) FROM session_profile_inference_fts"
)
SESSION_PROFILE_INFERENCE_FTS_DUPLICATE_COUNT_SQL = (
    "SELECT COUNT(*) - COUNT(DISTINCT conversation_id) FROM session_profile_inference_fts"
)
SESSION_PROFILE_ENRICHMENT_FTS_DOC_COUNT_SQL = (
    "SELECT COUNT(DISTINCT conversation_id) FROM session_profile_enrichment_fts"
)
SESSION_PROFILE_ENRICHMENT_FTS_DUPLICATE_COUNT_SQL = (
    "SELECT COUNT(*) - COUNT(DISTINCT conversation_id) FROM session_profile_enrichment_fts"
)
SESSION_WORK_EVENT_COUNT_SQL = "SELECT COUNT(*) FROM session_work_events"
SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL = "SELECT COUNT(DISTINCT event_id) FROM session_work_events_fts"
SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL = "SELECT COUNT(*) - COUNT(DISTINCT event_id) FROM session_work_events_fts"
SESSION_PHASE_COUNT_SQL = "SELECT COUNT(*) FROM session_phases"
WORK_THREAD_COUNT_SQL = "SELECT COUNT(*) FROM work_threads"
WORK_THREAD_FTS_DOC_COUNT_SQL = "SELECT COUNT(DISTINCT thread_id) FROM work_threads_fts"
WORK_THREAD_FTS_DUPLICATE_COUNT_SQL = "SELECT COUNT(*) - COUNT(DISTINCT thread_id) FROM work_threads_fts"
SESSION_TAG_ROLLUP_COUNT_SQL = "SELECT COUNT(*) FROM session_tag_rollups"
DAY_SESSION_SUMMARY_COUNT_SQL = "SELECT COUNT(*) FROM day_session_summaries"
TOTAL_CONVERSATIONS_SQL = "SELECT COUNT(*) FROM conversations"
ROOT_THREAD_COUNT_SQL = """
    SELECT COUNT(*)
    FROM conversations c
    LEFT JOIN conversations parent ON c.parent_conversation_id = parent.conversation_id
    WHERE parent.conversation_id IS NULL
"""
MISSING_SESSION_PROFILE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM conversations c
    LEFT JOIN session_profiles sp ON sp.conversation_id = c.conversation_id
    WHERE sp.conversation_id IS NULL
"""
STALE_SESSION_PROFILE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM conversations c
    JOIN session_profiles sp ON sp.conversation_id = c.conversation_id
    WHERE sp.materializer_version != ?
       OR ABS(COALESCE(sp.source_sort_key, 0.0) - COALESCE(c.sort_key, 0.0)) > 0.000001
"""
ORPHAN_SESSION_PROFILE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_profiles sp
    LEFT JOIN conversations c ON c.conversation_id = sp.conversation_id
    WHERE c.conversation_id IS NULL
"""
EXPECTED_WORK_EVENT_COUNT_SQL = "SELECT COALESCE(SUM(work_event_count), 0) FROM session_profiles"
EXPECTED_PHASE_COUNT_SQL = "SELECT COALESCE(SUM(phase_count), 0) FROM session_profiles"
STALE_WORK_EVENT_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_work_events swe
    JOIN conversations c ON c.conversation_id = swe.conversation_id
    WHERE swe.materializer_version != ?
       OR ABS(COALESCE(swe.source_sort_key, 0.0) - COALESCE(c.sort_key, 0.0)) > 0.000001
"""
ORPHAN_SESSION_WORK_EVENT_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_work_events swe
    LEFT JOIN conversations c ON c.conversation_id = swe.conversation_id
    WHERE c.conversation_id IS NULL
"""
STALE_SESSION_PHASE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_phases sph
    JOIN conversations c ON c.conversation_id = sph.conversation_id
    WHERE sph.materializer_version != ?
       OR ABS(COALESCE(sph.source_sort_key, 0.0) - COALESCE(c.sort_key, 0.0)) > 0.000001
"""
ORPHAN_SESSION_PHASE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_phases sph
    LEFT JOIN conversations c ON c.conversation_id = sph.conversation_id
    WHERE c.conversation_id IS NULL
"""
STALE_WORK_THREAD_COUNT_SQL = """
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
ORPHAN_WORK_THREAD_COUNT_SQL = """
    SELECT COUNT(*)
    FROM work_threads wt
    LEFT JOIN conversations c ON c.conversation_id = wt.root_id
    WHERE c.conversation_id IS NULL
"""
EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL = f"""
    SELECT COUNT(*) FROM (
        SELECT sp.provider_name, {_PROFILE_BUCKET_DAY_SQL} AS bucket_day
        FROM session_profiles sp
        WHERE {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL
        GROUP BY sp.provider_name, bucket_day
    )
"""
STALE_DAY_SESSION_SUMMARY_COUNT_SQL = f"""
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
EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL = f"""
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
STALE_SESSION_TAG_ROLLUP_COUNT_SQL = f"""
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
SESSION_PROFILE_REPAIR_CANDIDATES_SQL = """
    SELECT c.conversation_id
    FROM conversations c
    LEFT JOIN session_profiles sp ON sp.conversation_id = c.conversation_id
    WHERE sp.conversation_id IS NULL
       OR sp.materializer_version != ?
       OR COALESCE(sp.source_updated_at, '') != COALESCE(c.updated_at, '')
    ORDER BY c.conversation_id
"""

_TABLE_SQLS = {
    "session_profiles": SESSION_PROFILES_EXISTS_SQL,
    "session_profiles_fts": SESSION_PROFILES_FTS_EXISTS_SQL,
    "session_profile_evidence_fts": SESSION_PROFILE_EVIDENCE_FTS_EXISTS_SQL,
    "session_profile_inference_fts": SESSION_PROFILE_INFERENCE_FTS_EXISTS_SQL,
    "session_profile_enrichment_fts": SESSION_PROFILE_ENRICHMENT_FTS_EXISTS_SQL,
    "session_work_events": SESSION_WORK_EVENTS_EXISTS_SQL,
    "session_work_events_fts": SESSION_WORK_EVENTS_FTS_EXISTS_SQL,
    "session_phases": SESSION_PHASES_EXISTS_SQL,
    "work_threads": WORK_THREADS_EXISTS_SQL,
    "work_threads_fts": WORK_THREADS_FTS_EXISTS_SQL,
    "session_tag_rollups": SESSION_TAG_ROLLUPS_EXISTS_SQL,
    "day_session_summaries": DAY_SESSION_SUMMARIES_EXISTS_SQL,
}


def _to_int(row: tuple[object, ...] | sqlite3.Row | None) -> int:
    if not row:
        return 0
    value = row[0]
    if isinstance(value, (int, float, str)):
        return int(value)
    return 0


def _status_payload(
    tables: dict[str, bool],
    counts: dict[str, int],
) -> SessionProductStatusSnapshot:
    profile_count = counts["profile_row_count"]
    work_event_count = counts["work_event_inference_count"]
    phase_count = counts["phase_inference_count"]
    thread_count = counts["thread_count"]
    tag_rollup_count = counts["tag_rollup_count"]
    day_summary_count = counts["day_summary_count"]
    missing_profile_count = counts["missing_profile_row_count"]
    stale_profile_count = counts["stale_profile_row_count"]
    orphan_profile_count = counts["orphan_profile_row_count"]
    expected_work_event_count = counts["expected_work_event_inference_count"]
    stale_work_event_count = counts["stale_work_event_inference_count"]
    orphan_work_event_count = counts["orphan_work_event_inference_count"]
    expected_phase_count = counts["expected_phase_inference_count"]
    stale_phase_count = counts["stale_phase_inference_count"]
    orphan_phase_count = counts["orphan_phase_inference_count"]
    stale_thread_count = counts["stale_thread_count"]
    orphan_thread_count = counts["orphan_thread_count"]
    expected_tag_rollup_count = counts["expected_tag_rollup_count"]
    stale_tag_rollup_count = counts["stale_tag_rollup_count"]
    expected_day_summary_count = counts["expected_day_summary_count"]
    stale_day_summary_count = counts["stale_day_summary_count"]

    return SessionProductStatusSnapshot(
        **counts,
        profile_rows_ready=tables["session_profiles"]
        and missing_profile_count == 0
        and stale_profile_count == 0
        and orphan_profile_count == 0,
        profile_merged_fts_ready=tables["session_profiles_fts"]
        and counts["profile_merged_fts_count"] == profile_count
        and counts["profile_merged_fts_duplicate_count"] == 0,
        profile_evidence_fts_ready=tables["session_profile_evidence_fts"]
        and counts["profile_evidence_fts_count"] == profile_count
        and counts["profile_evidence_fts_duplicate_count"] == 0,
        profile_inference_fts_ready=tables["session_profile_inference_fts"]
        and counts["profile_inference_fts_count"] == profile_count
        and counts["profile_inference_fts_duplicate_count"] == 0,
        profile_enrichment_fts_ready=tables["session_profile_enrichment_fts"]
        and counts["profile_enrichment_fts_count"] == profile_count
        and counts["profile_enrichment_fts_duplicate_count"] == 0,
        work_event_inference_rows_ready=tables["session_work_events"]
        and work_event_count == expected_work_event_count
        and stale_work_event_count == 0
        and orphan_work_event_count == 0,
        work_event_inference_fts_ready=tables["session_work_events_fts"]
        and counts["work_event_inference_fts_count"] == work_event_count
        and counts["work_event_inference_fts_duplicate_count"] == 0,
        phase_inference_rows_ready=tables["session_phases"]
        and phase_count == expected_phase_count
        and stale_phase_count == 0
        and orphan_phase_count == 0,
        threads_ready=tables["work_threads"]
        and thread_count == counts["root_threads"]
        and stale_thread_count == 0
        and orphan_thread_count == 0,
        threads_fts_ready=tables["work_threads_fts"]
        and counts["thread_fts_count"] == thread_count
        and counts["thread_fts_duplicate_count"] == 0,
        tag_rollups_ready=tables["session_tag_rollups"]
        and tag_rollup_count == expected_tag_rollup_count
        and stale_tag_rollup_count == 0,
        day_summaries_ready=tables["day_session_summaries"]
        and day_summary_count == expected_day_summary_count
        and stale_day_summary_count == 0,
        week_summaries_ready=tables["day_session_summaries"]
        and day_summary_count == expected_day_summary_count
        and stale_day_summary_count == 0,
    )


def session_profile_repair_candidate_ids_sync(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        SESSION_PROFILE_REPAIR_CANDIDATES_SQL,
        (SESSION_PRODUCT_MATERIALIZER_VERSION,),
    ).fetchall()
    return [str(row["conversation_id"]) for row in rows]


async def session_profile_repair_candidate_ids_async(conn: aiosqlite.Connection) -> list[str]:
    rows = await (
        await conn.execute(
            SESSION_PROFILE_REPAIR_CANDIDATES_SQL,
            (SESSION_PRODUCT_MATERIALIZER_VERSION,),
        )
    ).fetchall()
    return [str(row["conversation_id"]) for row in rows]


def session_product_status_sync(
    conn: sqlite3.Connection,
    *,
    verify_freshness: bool = True,
) -> SessionProductStatusSnapshot:
    tables = {key: bool(conn.execute(sql).fetchone()) for key, sql in _TABLE_SQLS.items()}

    def count(sql: str, *params: object) -> int:
        return _to_int(conn.execute(sql, params).fetchone())

    def table_count(table_exists: bool, table_name: str) -> int:
        return _to_int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()) if table_exists else 0

    total_conversations = count(TOTAL_CONVERSATIONS_SQL)
    thread_count = count(WORK_THREAD_COUNT_SQL) if tables["work_threads"] else 0
    root_threads = count(ROOT_THREAD_COUNT_SQL) if verify_freshness else thread_count
    profile_row_count = count(SESSION_PROFILE_COUNT_SQL) if tables["session_profiles"] else 0
    profile_merged_fts_count = (
        count(SESSION_PROFILE_MERGED_FTS_DOC_COUNT_SQL)
        if verify_freshness and tables["session_profiles_fts"]
        else table_count(tables["session_profiles_fts"], "session_profiles_fts")
    )
    profile_evidence_fts_count = (
        count(SESSION_PROFILE_EVIDENCE_FTS_DOC_COUNT_SQL)
        if verify_freshness and tables["session_profile_evidence_fts"]
        else table_count(tables["session_profile_evidence_fts"], "session_profile_evidence_fts")
    )
    profile_inference_fts_count = (
        count(SESSION_PROFILE_INFERENCE_FTS_DOC_COUNT_SQL)
        if verify_freshness and tables["session_profile_inference_fts"]
        else table_count(tables["session_profile_inference_fts"], "session_profile_inference_fts")
    )
    profile_enrichment_fts_count = (
        count(SESSION_PROFILE_ENRICHMENT_FTS_DOC_COUNT_SQL)
        if verify_freshness and tables["session_profile_enrichment_fts"]
        else table_count(tables["session_profile_enrichment_fts"], "session_profile_enrichment_fts")
    )
    work_event_inference_count = count(SESSION_WORK_EVENT_COUNT_SQL) if tables["session_work_events"] else 0
    work_event_inference_fts_count = (
        count(SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL)
        if verify_freshness and tables["session_work_events_fts"]
        else table_count(tables["session_work_events_fts"], "session_work_events_fts")
    )
    phase_inference_count = count(SESSION_PHASE_COUNT_SQL) if tables["session_phases"] else 0
    thread_fts_count = (
        count(WORK_THREAD_FTS_DOC_COUNT_SQL)
        if verify_freshness and tables["work_threads_fts"]
        else table_count(tables["work_threads_fts"], "work_threads_fts")
    )
    tag_rollup_count = count(SESSION_TAG_ROLLUP_COUNT_SQL) if tables["session_tag_rollups"] else 0
    day_summary_count = count(DAY_SESSION_SUMMARY_COUNT_SQL) if tables["day_session_summaries"] else 0

    counts = {
        "total_conversations": total_conversations,
        "root_threads": root_threads,
        "profile_row_count": profile_row_count,
        "profile_merged_fts_count": profile_merged_fts_count,
        "profile_merged_fts_duplicate_count": (
            count(SESSION_PROFILE_MERGED_FTS_DUPLICATE_COUNT_SQL)
            if verify_freshness and tables["session_profiles_fts"]
            else max(0, profile_merged_fts_count - profile_row_count)
        ),
        "profile_evidence_fts_count": profile_evidence_fts_count,
        "profile_evidence_fts_duplicate_count": (
            count(SESSION_PROFILE_EVIDENCE_FTS_DUPLICATE_COUNT_SQL)
            if verify_freshness and tables["session_profile_evidence_fts"]
            else max(0, profile_evidence_fts_count - profile_row_count)
        ),
        "profile_inference_fts_count": profile_inference_fts_count,
        "profile_inference_fts_duplicate_count": (
            count(SESSION_PROFILE_INFERENCE_FTS_DUPLICATE_COUNT_SQL)
            if verify_freshness and tables["session_profile_inference_fts"]
            else max(0, profile_inference_fts_count - profile_row_count)
        ),
        "profile_enrichment_fts_count": profile_enrichment_fts_count,
        "profile_enrichment_fts_duplicate_count": (
            count(SESSION_PROFILE_ENRICHMENT_FTS_DUPLICATE_COUNT_SQL)
            if verify_freshness and tables["session_profile_enrichment_fts"]
            else max(0, profile_enrichment_fts_count - profile_row_count)
        ),
        "work_event_inference_count": work_event_inference_count,
        "work_event_inference_fts_count": work_event_inference_fts_count,
        "work_event_inference_fts_duplicate_count": (
            count(SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL)
            if verify_freshness and tables["session_work_events_fts"]
            else max(0, work_event_inference_fts_count - work_event_inference_count)
        ),
        "phase_inference_count": phase_inference_count,
        "thread_count": thread_count,
        "thread_fts_count": thread_fts_count,
        "thread_fts_duplicate_count": (
            count(WORK_THREAD_FTS_DUPLICATE_COUNT_SQL)
            if verify_freshness and tables["work_threads_fts"]
            else max(0, thread_fts_count - thread_count)
        ),
        "tag_rollup_count": tag_rollup_count,
        "day_summary_count": day_summary_count,
    }
    counts["missing_profile_row_count"] = (
        count(MISSING_SESSION_PROFILE_COUNT_SQL) if tables["session_profiles"] else counts["total_conversations"]
    )
    counts["stale_profile_row_count"] = (
        count(STALE_SESSION_PROFILE_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION)
        if verify_freshness and tables["session_profiles"]
        else 0
    )
    counts["orphan_profile_row_count"] = (
        count(ORPHAN_SESSION_PROFILE_COUNT_SQL) if verify_freshness and tables["session_profiles"] else 0
    )
    counts["expected_work_event_inference_count"] = (
        count(EXPECTED_WORK_EVENT_COUNT_SQL) if tables["session_profiles"] else 0
    )
    counts["expected_phase_inference_count"] = count(EXPECTED_PHASE_COUNT_SQL) if tables["session_profiles"] else 0
    counts["stale_work_event_inference_count"] = (
        count(STALE_WORK_EVENT_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION)
        if verify_freshness and tables["session_work_events"]
        else 0
    )
    counts["orphan_work_event_inference_count"] = (
        count(ORPHAN_SESSION_WORK_EVENT_COUNT_SQL) if verify_freshness and tables["session_work_events"] else 0
    )
    counts["stale_phase_inference_count"] = (
        count(STALE_SESSION_PHASE_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION)
        if verify_freshness and tables["session_phases"]
        else 0
    )
    counts["orphan_phase_inference_count"] = (
        count(ORPHAN_SESSION_PHASE_COUNT_SQL) if verify_freshness and tables["session_phases"] else 0
    )
    counts["stale_thread_count"] = (
        count(STALE_WORK_THREAD_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION)
        if verify_freshness and tables["work_threads"]
        else 0
    )
    counts["orphan_thread_count"] = (
        count(ORPHAN_WORK_THREAD_COUNT_SQL) if verify_freshness and tables["work_threads"] else 0
    )
    counts["expected_tag_rollup_count"] = (
        count(EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL)
        if verify_freshness and tables["session_profiles"]
        else counts["tag_rollup_count"]
    )
    counts["stale_tag_rollup_count"] = (
        count(STALE_SESSION_TAG_ROLLUP_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION)
        if verify_freshness and tables["session_tag_rollups"]
        else 0
    )
    counts["expected_day_summary_count"] = (
        count(EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL)
        if verify_freshness and tables["session_profiles"]
        else counts["day_summary_count"]
    )
    counts["stale_day_summary_count"] = (
        count(STALE_DAY_SESSION_SUMMARY_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION)
        if verify_freshness and tables["day_session_summaries"]
        else 0
    )
    return _status_payload(tables, counts)


async def session_product_status_async(conn: aiosqlite.Connection) -> SessionProductStatusSnapshot:
    tables = {}
    for key, sql in _TABLE_SQLS.items():
        tables[key] = bool(await (await conn.execute(sql)).fetchone())

    async def count(sql: str, *params: object) -> int:
        return _to_int(await (await conn.execute(sql, params)).fetchone())

    counts = {
        "total_conversations": await count(TOTAL_CONVERSATIONS_SQL),
        "root_threads": await count(ROOT_THREAD_COUNT_SQL),
        "profile_row_count": await count(SESSION_PROFILE_COUNT_SQL) if tables["session_profiles"] else 0,
        "profile_merged_fts_count": await count(SESSION_PROFILE_MERGED_FTS_DOC_COUNT_SQL)
        if tables["session_profiles_fts"]
        else 0,
        "profile_merged_fts_duplicate_count": await count(SESSION_PROFILE_MERGED_FTS_DUPLICATE_COUNT_SQL)
        if tables["session_profiles_fts"]
        else 0,
        "profile_evidence_fts_count": await count(SESSION_PROFILE_EVIDENCE_FTS_DOC_COUNT_SQL)
        if tables["session_profile_evidence_fts"]
        else 0,
        "profile_evidence_fts_duplicate_count": await count(SESSION_PROFILE_EVIDENCE_FTS_DUPLICATE_COUNT_SQL)
        if tables["session_profile_evidence_fts"]
        else 0,
        "profile_inference_fts_count": await count(SESSION_PROFILE_INFERENCE_FTS_DOC_COUNT_SQL)
        if tables["session_profile_inference_fts"]
        else 0,
        "profile_inference_fts_duplicate_count": await count(SESSION_PROFILE_INFERENCE_FTS_DUPLICATE_COUNT_SQL)
        if tables["session_profile_inference_fts"]
        else 0,
        "profile_enrichment_fts_count": await count(SESSION_PROFILE_ENRICHMENT_FTS_DOC_COUNT_SQL)
        if tables["session_profile_enrichment_fts"]
        else 0,
        "profile_enrichment_fts_duplicate_count": await count(SESSION_PROFILE_ENRICHMENT_FTS_DUPLICATE_COUNT_SQL)
        if tables["session_profile_enrichment_fts"]
        else 0,
        "work_event_inference_count": await count(SESSION_WORK_EVENT_COUNT_SQL) if tables["session_work_events"] else 0,
        "work_event_inference_fts_count": await count(SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL)
        if tables["session_work_events_fts"]
        else 0,
        "work_event_inference_fts_duplicate_count": await count(SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL)
        if tables["session_work_events_fts"]
        else 0,
        "phase_inference_count": await count(SESSION_PHASE_COUNT_SQL) if tables["session_phases"] else 0,
        "thread_count": await count(WORK_THREAD_COUNT_SQL) if tables["work_threads"] else 0,
        "thread_fts_count": await count(WORK_THREAD_FTS_DOC_COUNT_SQL) if tables["work_threads_fts"] else 0,
        "thread_fts_duplicate_count": await count(WORK_THREAD_FTS_DUPLICATE_COUNT_SQL)
        if tables["work_threads_fts"]
        else 0,
        "tag_rollup_count": await count(SESSION_TAG_ROLLUP_COUNT_SQL) if tables["session_tag_rollups"] else 0,
        "day_summary_count": await count(DAY_SESSION_SUMMARY_COUNT_SQL) if tables["day_session_summaries"] else 0,
    }
    counts["missing_profile_row_count"] = (
        await count(MISSING_SESSION_PROFILE_COUNT_SQL) if tables["session_profiles"] else counts["total_conversations"]
    )
    counts["stale_profile_row_count"] = (
        await count(STALE_SESSION_PROFILE_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION)
        if tables["session_profiles"]
        else counts["total_conversations"]
    )
    counts["orphan_profile_row_count"] = (
        await count(ORPHAN_SESSION_PROFILE_COUNT_SQL) if tables["session_profiles"] else 0
    )
    counts["expected_work_event_inference_count"] = (
        await count(EXPECTED_WORK_EVENT_COUNT_SQL) if tables["session_profiles"] else 0
    )
    counts["expected_phase_inference_count"] = (
        await count(EXPECTED_PHASE_COUNT_SQL) if tables["session_profiles"] else 0
    )
    counts["stale_work_event_inference_count"] = (
        await count(STALE_WORK_EVENT_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION)
        if tables["session_work_events"]
        else counts["expected_work_event_inference_count"]
    )
    counts["orphan_work_event_inference_count"] = (
        await count(ORPHAN_SESSION_WORK_EVENT_COUNT_SQL) if tables["session_work_events"] else 0
    )
    counts["stale_phase_inference_count"] = (
        await count(STALE_SESSION_PHASE_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION)
        if tables["session_phases"]
        else counts["expected_phase_inference_count"]
    )
    counts["orphan_phase_inference_count"] = (
        await count(ORPHAN_SESSION_PHASE_COUNT_SQL) if tables["session_phases"] else 0
    )
    counts["stale_thread_count"] = (
        await count(STALE_WORK_THREAD_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION)
        if tables["work_threads"]
        else counts["root_threads"]
    )
    counts["orphan_thread_count"] = await count(ORPHAN_WORK_THREAD_COUNT_SQL) if tables["work_threads"] else 0
    counts["expected_tag_rollup_count"] = (
        await count(EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL) if tables["session_profiles"] else 0
    )
    counts["stale_tag_rollup_count"] = (
        await count(STALE_SESSION_TAG_ROLLUP_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION)
        if tables["session_tag_rollups"]
        else counts["expected_tag_rollup_count"]
    )
    counts["expected_day_summary_count"] = (
        await count(EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL) if tables["session_profiles"] else 0
    )
    counts["stale_day_summary_count"] = (
        await count(STALE_DAY_SESSION_SUMMARY_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION)
        if tables["day_session_summaries"]
        else counts["expected_day_summary_count"]
    )
    return _status_payload(tables, counts)


__all__ = [
    "session_product_status_async",
    "session_product_status_sync",
    "session_profile_repair_candidate_ids_async",
    "session_profile_repair_candidate_ids_sync",
]
