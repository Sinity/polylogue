"""Canonical session-product status and repair-candidate queries."""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.storage.store import SESSION_PRODUCT_MATERIALIZER_VERSION

from .session_product_status_sql import (
    DAY_SESSION_SUMMARIES_EXISTS_SQL,
    DAY_SESSION_SUMMARY_COUNT_SQL,
    EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL,
    EXPECTED_PHASE_COUNT_SQL,
    EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL,
    EXPECTED_WORK_EVENT_COUNT_SQL,
    MISSING_SESSION_PROFILE_COUNT_SQL,
    ORPHAN_SESSION_PHASE_COUNT_SQL,
    ORPHAN_SESSION_PROFILE_COUNT_SQL,
    ORPHAN_SESSION_WORK_EVENT_COUNT_SQL,
    ORPHAN_WORK_THREAD_COUNT_SQL,
    ROOT_THREAD_COUNT_SQL,
    SESSION_PHASE_COUNT_SQL,
    SESSION_PHASES_EXISTS_SQL,
    SESSION_PROFILE_COUNT_SQL,
    SESSION_PROFILE_ENRICHMENT_FTS_DOC_COUNT_SQL,
    SESSION_PROFILE_ENRICHMENT_FTS_DUPLICATE_COUNT_SQL,
    SESSION_PROFILE_ENRICHMENT_FTS_EXISTS_SQL,
    SESSION_PROFILE_EVIDENCE_FTS_DOC_COUNT_SQL,
    SESSION_PROFILE_EVIDENCE_FTS_DUPLICATE_COUNT_SQL,
    SESSION_PROFILE_EVIDENCE_FTS_EXISTS_SQL,
    SESSION_PROFILE_INFERENCE_FTS_DOC_COUNT_SQL,
    SESSION_PROFILE_INFERENCE_FTS_DUPLICATE_COUNT_SQL,
    SESSION_PROFILE_INFERENCE_FTS_EXISTS_SQL,
    SESSION_PROFILE_MERGED_FTS_DOC_COUNT_SQL,
    SESSION_PROFILE_MERGED_FTS_DUPLICATE_COUNT_SQL,
    SESSION_PROFILE_REPAIR_CANDIDATES_SQL,
    SESSION_PROFILES_EXISTS_SQL,
    SESSION_PROFILES_FTS_EXISTS_SQL,
    SESSION_TAG_ROLLUP_COUNT_SQL,
    SESSION_TAG_ROLLUPS_EXISTS_SQL,
    SESSION_WORK_EVENT_COUNT_SQL,
    SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL,
    SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL,
    SESSION_WORK_EVENTS_EXISTS_SQL,
    SESSION_WORK_EVENTS_FTS_EXISTS_SQL,
    STALE_DAY_SESSION_SUMMARY_COUNT_SQL,
    STALE_SESSION_PHASE_COUNT_SQL,
    STALE_SESSION_PROFILE_COUNT_SQL,
    STALE_SESSION_TAG_ROLLUP_COUNT_SQL,
    STALE_WORK_EVENT_COUNT_SQL,
    STALE_WORK_THREAD_COUNT_SQL,
    TOTAL_CONVERSATIONS_SQL,
    WORK_THREAD_COUNT_SQL,
    WORK_THREAD_FTS_DOC_COUNT_SQL,
    WORK_THREAD_FTS_DUPLICATE_COUNT_SQL,
    WORK_THREADS_EXISTS_SQL,
    WORK_THREADS_FTS_EXISTS_SQL,
)

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
    return int(row[0] or 0) if row else 0


def _status_payload(
    tables: dict[str, bool],
    counts: dict[str, int],
) -> dict[str, int | bool]:
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

    return {
        **counts,
        "profile_rows_ready": tables["session_profiles"]
        and missing_profile_count == 0
        and stale_profile_count == 0
        and orphan_profile_count == 0,
        "profile_merged_fts_ready": tables["session_profiles_fts"]
        and counts["profile_merged_fts_count"] == profile_count
        and counts["profile_merged_fts_duplicate_count"] == 0,
        "profile_evidence_fts_ready": tables["session_profile_evidence_fts"]
        and counts["profile_evidence_fts_count"] == profile_count
        and counts["profile_evidence_fts_duplicate_count"] == 0,
        "profile_inference_fts_ready": tables["session_profile_inference_fts"]
        and counts["profile_inference_fts_count"] == profile_count
        and counts["profile_inference_fts_duplicate_count"] == 0,
        "profile_enrichment_fts_ready": tables["session_profile_enrichment_fts"]
        and counts["profile_enrichment_fts_count"] == profile_count
        and counts["profile_enrichment_fts_duplicate_count"] == 0,
        "work_event_inference_rows_ready": tables["session_work_events"]
        and work_event_count == expected_work_event_count
        and stale_work_event_count == 0
        and orphan_work_event_count == 0,
        "work_event_inference_fts_ready": tables["session_work_events_fts"]
        and counts["work_event_inference_fts_count"] == work_event_count
        and counts["work_event_inference_fts_duplicate_count"] == 0,
        "phase_inference_rows_ready": tables["session_phases"]
        and phase_count == expected_phase_count
        and stale_phase_count == 0
        and orphan_phase_count == 0,
        "threads_ready": tables["work_threads"]
        and thread_count == counts["root_threads"]
        and stale_thread_count == 0
        and orphan_thread_count == 0,
        "threads_fts_ready": tables["work_threads_fts"]
        and counts["thread_fts_count"] == thread_count
        and counts["thread_fts_duplicate_count"] == 0,
        "tag_rollups_ready": tables["session_tag_rollups"]
        and tag_rollup_count == expected_tag_rollup_count
        and stale_tag_rollup_count == 0,
        "day_summaries_ready": tables["day_session_summaries"]
        and day_summary_count == expected_day_summary_count
        and stale_day_summary_count == 0,
        "week_summaries_ready": tables["day_session_summaries"]
        and day_summary_count == expected_day_summary_count
        and stale_day_summary_count == 0,
    }


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


def session_product_status_sync(conn: sqlite3.Connection) -> dict[str, int | bool]:
    tables = {
        key: bool(conn.execute(sql).fetchone())
        for key, sql in _TABLE_SQLS.items()
    }

    def count(sql: str, *params: object) -> int:
        return _to_int(conn.execute(sql, params).fetchone())

    counts = {
        "total_conversations": count(TOTAL_CONVERSATIONS_SQL),
        "root_threads": count(ROOT_THREAD_COUNT_SQL),
        "profile_row_count": count(SESSION_PROFILE_COUNT_SQL) if tables["session_profiles"] else 0,
        "profile_merged_fts_count": count(SESSION_PROFILE_MERGED_FTS_DOC_COUNT_SQL) if tables["session_profiles_fts"] else 0,
        "profile_merged_fts_duplicate_count": count(SESSION_PROFILE_MERGED_FTS_DUPLICATE_COUNT_SQL) if tables["session_profiles_fts"] else 0,
        "profile_evidence_fts_count": count(SESSION_PROFILE_EVIDENCE_FTS_DOC_COUNT_SQL) if tables["session_profile_evidence_fts"] else 0,
        "profile_evidence_fts_duplicate_count": count(SESSION_PROFILE_EVIDENCE_FTS_DUPLICATE_COUNT_SQL) if tables["session_profile_evidence_fts"] else 0,
        "profile_inference_fts_count": count(SESSION_PROFILE_INFERENCE_FTS_DOC_COUNT_SQL) if tables["session_profile_inference_fts"] else 0,
        "profile_inference_fts_duplicate_count": count(SESSION_PROFILE_INFERENCE_FTS_DUPLICATE_COUNT_SQL) if tables["session_profile_inference_fts"] else 0,
        "profile_enrichment_fts_count": count(SESSION_PROFILE_ENRICHMENT_FTS_DOC_COUNT_SQL) if tables["session_profile_enrichment_fts"] else 0,
        "profile_enrichment_fts_duplicate_count": count(SESSION_PROFILE_ENRICHMENT_FTS_DUPLICATE_COUNT_SQL) if tables["session_profile_enrichment_fts"] else 0,
        "work_event_inference_count": count(SESSION_WORK_EVENT_COUNT_SQL) if tables["session_work_events"] else 0,
        "work_event_inference_fts_count": count(SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL) if tables["session_work_events_fts"] else 0,
        "work_event_inference_fts_duplicate_count": count(SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL) if tables["session_work_events_fts"] else 0,
        "phase_inference_count": count(SESSION_PHASE_COUNT_SQL) if tables["session_phases"] else 0,
        "thread_count": count(WORK_THREAD_COUNT_SQL) if tables["work_threads"] else 0,
        "thread_fts_count": count(WORK_THREAD_FTS_DOC_COUNT_SQL) if tables["work_threads_fts"] else 0,
        "thread_fts_duplicate_count": count(WORK_THREAD_FTS_DUPLICATE_COUNT_SQL) if tables["work_threads_fts"] else 0,
        "tag_rollup_count": count(SESSION_TAG_ROLLUP_COUNT_SQL) if tables["session_tag_rollups"] else 0,
        "day_summary_count": count(DAY_SESSION_SUMMARY_COUNT_SQL) if tables["day_session_summaries"] else 0,
    }
    counts["missing_profile_row_count"] = count(MISSING_SESSION_PROFILE_COUNT_SQL) if tables["session_profiles"] else counts["total_conversations"]
    counts["stale_profile_row_count"] = count(STALE_SESSION_PROFILE_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION) if tables["session_profiles"] else counts["total_conversations"]
    counts["orphan_profile_row_count"] = count(ORPHAN_SESSION_PROFILE_COUNT_SQL) if tables["session_profiles"] else 0
    counts["expected_work_event_inference_count"] = count(EXPECTED_WORK_EVENT_COUNT_SQL) if tables["session_profiles"] else 0
    counts["expected_phase_inference_count"] = count(EXPECTED_PHASE_COUNT_SQL) if tables["session_profiles"] else 0
    counts["stale_work_event_inference_count"] = count(STALE_WORK_EVENT_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION) if tables["session_work_events"] else counts["expected_work_event_inference_count"]
    counts["orphan_work_event_inference_count"] = count(ORPHAN_SESSION_WORK_EVENT_COUNT_SQL) if tables["session_work_events"] else 0
    counts["stale_phase_inference_count"] = count(STALE_SESSION_PHASE_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION) if tables["session_phases"] else counts["expected_phase_inference_count"]
    counts["orphan_phase_inference_count"] = count(ORPHAN_SESSION_PHASE_COUNT_SQL) if tables["session_phases"] else 0
    counts["stale_thread_count"] = count(STALE_WORK_THREAD_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION) if tables["work_threads"] else counts["root_threads"]
    counts["orphan_thread_count"] = count(ORPHAN_WORK_THREAD_COUNT_SQL) if tables["work_threads"] else 0
    counts["expected_tag_rollup_count"] = count(EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL) if tables["session_profiles"] else 0
    counts["stale_tag_rollup_count"] = count(STALE_SESSION_TAG_ROLLUP_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION) if tables["session_tag_rollups"] else counts["expected_tag_rollup_count"]
    counts["expected_day_summary_count"] = count(EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL) if tables["session_profiles"] else 0
    counts["stale_day_summary_count"] = count(STALE_DAY_SESSION_SUMMARY_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION) if tables["day_session_summaries"] else counts["expected_day_summary_count"]
    return _status_payload(tables, counts)


async def session_product_status_async(conn: aiosqlite.Connection) -> dict[str, int | bool]:
    tables = {}
    for key, sql in _TABLE_SQLS.items():
        tables[key] = bool(await (await conn.execute(sql)).fetchone())

    async def count(sql: str, *params: object) -> int:
        return _to_int(await (await conn.execute(sql, params)).fetchone())

    counts = {
        "total_conversations": await count(TOTAL_CONVERSATIONS_SQL),
        "root_threads": await count(ROOT_THREAD_COUNT_SQL),
        "profile_row_count": await count(SESSION_PROFILE_COUNT_SQL) if tables["session_profiles"] else 0,
        "profile_merged_fts_count": await count(SESSION_PROFILE_MERGED_FTS_DOC_COUNT_SQL) if tables["session_profiles_fts"] else 0,
        "profile_merged_fts_duplicate_count": await count(SESSION_PROFILE_MERGED_FTS_DUPLICATE_COUNT_SQL) if tables["session_profiles_fts"] else 0,
        "profile_evidence_fts_count": await count(SESSION_PROFILE_EVIDENCE_FTS_DOC_COUNT_SQL) if tables["session_profile_evidence_fts"] else 0,
        "profile_evidence_fts_duplicate_count": await count(SESSION_PROFILE_EVIDENCE_FTS_DUPLICATE_COUNT_SQL) if tables["session_profile_evidence_fts"] else 0,
        "profile_inference_fts_count": await count(SESSION_PROFILE_INFERENCE_FTS_DOC_COUNT_SQL) if tables["session_profile_inference_fts"] else 0,
        "profile_inference_fts_duplicate_count": await count(SESSION_PROFILE_INFERENCE_FTS_DUPLICATE_COUNT_SQL) if tables["session_profile_inference_fts"] else 0,
        "profile_enrichment_fts_count": await count(SESSION_PROFILE_ENRICHMENT_FTS_DOC_COUNT_SQL) if tables["session_profile_enrichment_fts"] else 0,
        "profile_enrichment_fts_duplicate_count": await count(SESSION_PROFILE_ENRICHMENT_FTS_DUPLICATE_COUNT_SQL) if tables["session_profile_enrichment_fts"] else 0,
        "work_event_inference_count": await count(SESSION_WORK_EVENT_COUNT_SQL) if tables["session_work_events"] else 0,
        "work_event_inference_fts_count": await count(SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL) if tables["session_work_events_fts"] else 0,
        "work_event_inference_fts_duplicate_count": await count(SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL) if tables["session_work_events_fts"] else 0,
        "phase_inference_count": await count(SESSION_PHASE_COUNT_SQL) if tables["session_phases"] else 0,
        "thread_count": await count(WORK_THREAD_COUNT_SQL) if tables["work_threads"] else 0,
        "thread_fts_count": await count(WORK_THREAD_FTS_DOC_COUNT_SQL) if tables["work_threads_fts"] else 0,
        "thread_fts_duplicate_count": await count(WORK_THREAD_FTS_DUPLICATE_COUNT_SQL) if tables["work_threads_fts"] else 0,
        "tag_rollup_count": await count(SESSION_TAG_ROLLUP_COUNT_SQL) if tables["session_tag_rollups"] else 0,
        "day_summary_count": await count(DAY_SESSION_SUMMARY_COUNT_SQL) if tables["day_session_summaries"] else 0,
    }
    counts["missing_profile_row_count"] = await count(MISSING_SESSION_PROFILE_COUNT_SQL) if tables["session_profiles"] else counts["total_conversations"]
    counts["stale_profile_row_count"] = await count(STALE_SESSION_PROFILE_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION) if tables["session_profiles"] else counts["total_conversations"]
    counts["orphan_profile_row_count"] = await count(ORPHAN_SESSION_PROFILE_COUNT_SQL) if tables["session_profiles"] else 0
    counts["expected_work_event_inference_count"] = await count(EXPECTED_WORK_EVENT_COUNT_SQL) if tables["session_profiles"] else 0
    counts["expected_phase_inference_count"] = await count(EXPECTED_PHASE_COUNT_SQL) if tables["session_profiles"] else 0
    counts["stale_work_event_inference_count"] = await count(STALE_WORK_EVENT_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION) if tables["session_work_events"] else counts["expected_work_event_inference_count"]
    counts["orphan_work_event_inference_count"] = await count(ORPHAN_SESSION_WORK_EVENT_COUNT_SQL) if tables["session_work_events"] else 0
    counts["stale_phase_inference_count"] = await count(STALE_SESSION_PHASE_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION) if tables["session_phases"] else counts["expected_phase_inference_count"]
    counts["orphan_phase_inference_count"] = await count(ORPHAN_SESSION_PHASE_COUNT_SQL) if tables["session_phases"] else 0
    counts["stale_thread_count"] = await count(STALE_WORK_THREAD_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION) if tables["work_threads"] else counts["root_threads"]
    counts["orphan_thread_count"] = await count(ORPHAN_WORK_THREAD_COUNT_SQL) if tables["work_threads"] else 0
    counts["expected_tag_rollup_count"] = await count(EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL) if tables["session_profiles"] else 0
    counts["stale_tag_rollup_count"] = await count(STALE_SESSION_TAG_ROLLUP_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION) if tables["session_tag_rollups"] else counts["expected_tag_rollup_count"]
    counts["expected_day_summary_count"] = await count(EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL) if tables["session_profiles"] else 0
    counts["stale_day_summary_count"] = await count(STALE_DAY_SESSION_SUMMARY_COUNT_SQL, SESSION_PRODUCT_MATERIALIZER_VERSION) if tables["day_session_summaries"] else counts["expected_day_summary_count"]
    return _status_payload(tables, counts)


__all__ = [
    "session_product_status_async",
    "session_product_status_sync",
    "session_profile_repair_candidate_ids_async",
    "session_profile_repair_candidate_ids_sync",
]
