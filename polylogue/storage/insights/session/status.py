"""Canonical session-insight status and repair-candidate queries."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import TypeAlias

import aiosqlite

from polylogue.storage.insights.session.aggregates import _PROFILE_BUCKET_DAY_SQL
from polylogue.storage.insights.session.runtime import SessionInsightReadyFlag, SessionInsightStatusSnapshot
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION

TablePresence: TypeAlias = dict[str, bool]
StatusCounts: TypeAlias = dict[str, int]
CountEquality: TypeAlias = tuple[str, str]


@dataclass(frozen=True)
class SessionInsightTableDescriptor:
    """Table presence and optional row-count query for insight status."""

    key: str
    table_name: str
    count_key: str | None = None
    count_sql: str | None = None

    @property
    def exists_sql(self) -> str:
        return f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.table_name}'"

    def count_sync(self, conn: sqlite3.Connection, tables: TablePresence) -> tuple[str, int] | None:
        if self.count_key is None:
            return None
        if not tables[self.key]:
            return (self.count_key, 0)
        return (self.count_key, _count_sync(conn, self.count_sql or f"SELECT COUNT(*) FROM {self.table_name}"))

    async def count_async(self, conn: aiosqlite.Connection, tables: TablePresence) -> tuple[str, int] | None:
        if self.count_key is None:
            return None
        if not tables[self.key]:
            return (self.count_key, 0)
        return (self.count_key, await _count_async(conn, self.count_sql or f"SELECT COUNT(*) FROM {self.table_name}"))


@dataclass(frozen=True)
class SessionInsightFtsDescriptor:
    """FTS projection counts, duplicate checks, and readiness predicate."""

    table_key: str
    table_name: str
    count_key: str
    duplicate_count_key: str
    source_count_key: str
    distinct_sql: str
    duplicate_sql: str
    ready_key: SessionInsightReadyFlag

    def counts_sync(
        self,
        conn: sqlite3.Connection,
        tables: TablePresence,
        counts: StatusCounts,
        *,
        verify_freshness: bool,
    ) -> StatusCounts:
        if verify_freshness and tables[self.table_key]:
            indexed_count = _count_sync(conn, self.distinct_sql)
            duplicate_count = _count_sync(conn, self.duplicate_sql)
        else:
            indexed_count = _table_count_sync(conn, tables[self.table_key], self.table_name)
            duplicate_count = max(0, indexed_count - counts[self.source_count_key])
        return {
            self.count_key: indexed_count,
            self.duplicate_count_key: duplicate_count,
        }

    async def counts_async(
        self,
        conn: aiosqlite.Connection,
        tables: TablePresence,
        counts: StatusCounts,
        *,
        verify_freshness: bool,
    ) -> StatusCounts:
        if verify_freshness and tables[self.table_key]:
            indexed_count = await _count_async(conn, self.distinct_sql)
            duplicate_count = await _count_async(conn, self.duplicate_sql)
        else:
            indexed_count = await _table_count_async(conn, tables[self.table_key], self.table_name)
            duplicate_count = max(0, indexed_count - counts[self.source_count_key])
        return {
            self.count_key: indexed_count,
            self.duplicate_count_key: duplicate_count,
        }

    def ready(self, tables: TablePresence, counts: StatusCounts) -> bool:
        return (
            tables[self.table_key]
            and counts[self.count_key] == counts[self.source_count_key]
            and counts[self.duplicate_count_key] == 0
        )


@dataclass(frozen=True)
class SessionInsightCountDescriptor:
    """Status metric query with table/freshness gating and fallback semantics."""

    count_key: str
    sql: str
    table_key: str | None = None
    params: tuple[object, ...] = ()
    requires_freshness: bool = False
    fallback_count_key: str | None = None
    fallback_value: int = 0

    def _should_query(self, tables: TablePresence, *, verify_freshness: bool) -> bool:
        if self.requires_freshness and not verify_freshness:
            return False
        return self.table_key is None or tables[self.table_key]

    def _fallback(self, counts: StatusCounts) -> int:
        if self.fallback_count_key is not None:
            return counts[self.fallback_count_key]
        return self.fallback_value

    def count_sync(
        self,
        conn: sqlite3.Connection,
        tables: TablePresence,
        counts: StatusCounts,
        *,
        verify_freshness: bool,
    ) -> tuple[str, int]:
        if self._should_query(tables, verify_freshness=verify_freshness):
            return (self.count_key, _count_sync(conn, self.sql, *self.params))
        return (self.count_key, self._fallback(counts))

    async def count_async(
        self,
        conn: aiosqlite.Connection,
        tables: TablePresence,
        counts: StatusCounts,
        *,
        verify_freshness: bool,
    ) -> tuple[str, int]:
        if self._should_query(tables, verify_freshness=verify_freshness):
            return (self.count_key, await _count_async(conn, self.sql, *self.params))
        return (self.count_key, self._fallback(counts))


@dataclass(frozen=True)
class SessionInsightReadyDescriptor:
    """Readiness predicate over an insight table and named status counts."""

    ready_key: SessionInsightReadyFlag
    table_key: str
    equal_counts: tuple[CountEquality, ...] = ()
    zero_counts: tuple[str, ...] = ()

    def ready(self, tables: TablePresence, counts: StatusCounts) -> bool:
        return (
            tables[self.table_key]
            and all(counts[left] == counts[right] for left, right in self.equal_counts)
            and all(counts[key] == 0 for key in self.zero_counts)
        )


# ---------------------------------------------------------------------------
# SQL constants for session-insight status and drift checks
# ---------------------------------------------------------------------------

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

_TABLE_DESCRIPTORS: tuple[SessionInsightTableDescriptor, ...] = (
    SessionInsightTableDescriptor(
        key="session_profiles",
        table_name="session_profiles",
        count_key="profile_row_count",
        count_sql=SESSION_PROFILE_COUNT_SQL,
    ),
    SessionInsightTableDescriptor(
        key="session_profiles_fts",
        table_name="session_profiles_fts",
    ),
    SessionInsightTableDescriptor(
        key="session_profile_evidence_fts",
        table_name="session_profile_evidence_fts",
    ),
    SessionInsightTableDescriptor(
        key="session_profile_inference_fts",
        table_name="session_profile_inference_fts",
    ),
    SessionInsightTableDescriptor(
        key="session_profile_enrichment_fts",
        table_name="session_profile_enrichment_fts",
    ),
    SessionInsightTableDescriptor(
        key="session_work_events",
        table_name="session_work_events",
        count_key="work_event_inference_count",
        count_sql=SESSION_WORK_EVENT_COUNT_SQL,
    ),
    SessionInsightTableDescriptor(
        key="session_work_events_fts",
        table_name="session_work_events_fts",
    ),
    SessionInsightTableDescriptor(
        key="session_phases",
        table_name="session_phases",
        count_key="phase_inference_count",
        count_sql=SESSION_PHASE_COUNT_SQL,
    ),
    SessionInsightTableDescriptor(
        key="work_threads",
        table_name="work_threads",
        count_key="thread_count",
        count_sql=WORK_THREAD_COUNT_SQL,
    ),
    SessionInsightTableDescriptor(
        key="work_threads_fts",
        table_name="work_threads_fts",
    ),
    SessionInsightTableDescriptor(
        key="session_tag_rollups",
        table_name="session_tag_rollups",
        count_key="tag_rollup_count",
        count_sql=SESSION_TAG_ROLLUP_COUNT_SQL,
    ),
    SessionInsightTableDescriptor(
        key="day_session_summaries",
        table_name="day_session_summaries",
        count_key="day_summary_count",
        count_sql=DAY_SESSION_SUMMARY_COUNT_SQL,
    ),
)

_FTS_DESCRIPTORS: tuple[SessionInsightFtsDescriptor, ...] = (
    SessionInsightFtsDescriptor(
        table_key="session_profiles_fts",
        table_name="session_profiles_fts",
        count_key="profile_merged_fts_count",
        duplicate_count_key="profile_merged_fts_duplicate_count",
        source_count_key="profile_row_count",
        distinct_sql=SESSION_PROFILE_MERGED_FTS_DOC_COUNT_SQL,
        duplicate_sql=SESSION_PROFILE_MERGED_FTS_DUPLICATE_COUNT_SQL,
        ready_key="profile_merged_fts_ready",
    ),
    SessionInsightFtsDescriptor(
        table_key="session_profile_evidence_fts",
        table_name="session_profile_evidence_fts",
        count_key="profile_evidence_fts_count",
        duplicate_count_key="profile_evidence_fts_duplicate_count",
        source_count_key="profile_row_count",
        distinct_sql=SESSION_PROFILE_EVIDENCE_FTS_DOC_COUNT_SQL,
        duplicate_sql=SESSION_PROFILE_EVIDENCE_FTS_DUPLICATE_COUNT_SQL,
        ready_key="profile_evidence_fts_ready",
    ),
    SessionInsightFtsDescriptor(
        table_key="session_profile_inference_fts",
        table_name="session_profile_inference_fts",
        count_key="profile_inference_fts_count",
        duplicate_count_key="profile_inference_fts_duplicate_count",
        source_count_key="profile_row_count",
        distinct_sql=SESSION_PROFILE_INFERENCE_FTS_DOC_COUNT_SQL,
        duplicate_sql=SESSION_PROFILE_INFERENCE_FTS_DUPLICATE_COUNT_SQL,
        ready_key="profile_inference_fts_ready",
    ),
    SessionInsightFtsDescriptor(
        table_key="session_profile_enrichment_fts",
        table_name="session_profile_enrichment_fts",
        count_key="profile_enrichment_fts_count",
        duplicate_count_key="profile_enrichment_fts_duplicate_count",
        source_count_key="profile_row_count",
        distinct_sql=SESSION_PROFILE_ENRICHMENT_FTS_DOC_COUNT_SQL,
        duplicate_sql=SESSION_PROFILE_ENRICHMENT_FTS_DUPLICATE_COUNT_SQL,
        ready_key="profile_enrichment_fts_ready",
    ),
    SessionInsightFtsDescriptor(
        table_key="session_work_events_fts",
        table_name="session_work_events_fts",
        count_key="work_event_inference_fts_count",
        duplicate_count_key="work_event_inference_fts_duplicate_count",
        source_count_key="work_event_inference_count",
        distinct_sql=SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL,
        duplicate_sql=SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL,
        ready_key="work_event_inference_fts_ready",
    ),
    SessionInsightFtsDescriptor(
        table_key="work_threads_fts",
        table_name="work_threads_fts",
        count_key="thread_fts_count",
        duplicate_count_key="thread_fts_duplicate_count",
        source_count_key="thread_count",
        distinct_sql=WORK_THREAD_FTS_DOC_COUNT_SQL,
        duplicate_sql=WORK_THREAD_FTS_DUPLICATE_COUNT_SQL,
        ready_key="threads_fts_ready",
    ),
)

_COUNT_DESCRIPTORS: tuple[SessionInsightCountDescriptor, ...] = (
    SessionInsightCountDescriptor(
        count_key="missing_profile_row_count",
        table_key="session_profiles",
        sql=MISSING_SESSION_PROFILE_COUNT_SQL,
        fallback_count_key="total_conversations",
    ),
    SessionInsightCountDescriptor(
        count_key="stale_profile_row_count",
        table_key="session_profiles",
        sql=STALE_SESSION_PROFILE_COUNT_SQL,
        params=(SESSION_INSIGHT_MATERIALIZER_VERSION,),
        requires_freshness=True,
    ),
    SessionInsightCountDescriptor(
        count_key="orphan_profile_row_count",
        table_key="session_profiles",
        sql=ORPHAN_SESSION_PROFILE_COUNT_SQL,
        requires_freshness=True,
    ),
    SessionInsightCountDescriptor(
        count_key="expected_work_event_inference_count",
        table_key="session_profiles",
        sql=EXPECTED_WORK_EVENT_COUNT_SQL,
    ),
    SessionInsightCountDescriptor(
        count_key="expected_phase_inference_count",
        table_key="session_profiles",
        sql=EXPECTED_PHASE_COUNT_SQL,
    ),
    SessionInsightCountDescriptor(
        count_key="stale_work_event_inference_count",
        table_key="session_work_events",
        sql=STALE_WORK_EVENT_COUNT_SQL,
        params=(SESSION_INSIGHT_MATERIALIZER_VERSION,),
        requires_freshness=True,
    ),
    SessionInsightCountDescriptor(
        count_key="orphan_work_event_inference_count",
        table_key="session_work_events",
        sql=ORPHAN_SESSION_WORK_EVENT_COUNT_SQL,
        requires_freshness=True,
    ),
    SessionInsightCountDescriptor(
        count_key="stale_phase_inference_count",
        table_key="session_phases",
        sql=STALE_SESSION_PHASE_COUNT_SQL,
        params=(SESSION_INSIGHT_MATERIALIZER_VERSION,),
        requires_freshness=True,
    ),
    SessionInsightCountDescriptor(
        count_key="orphan_phase_inference_count",
        table_key="session_phases",
        sql=ORPHAN_SESSION_PHASE_COUNT_SQL,
        requires_freshness=True,
    ),
    SessionInsightCountDescriptor(
        count_key="stale_thread_count",
        table_key="work_threads",
        sql=STALE_WORK_THREAD_COUNT_SQL,
        params=(SESSION_INSIGHT_MATERIALIZER_VERSION,),
        requires_freshness=True,
    ),
    SessionInsightCountDescriptor(
        count_key="orphan_thread_count",
        table_key="work_threads",
        sql=ORPHAN_WORK_THREAD_COUNT_SQL,
        requires_freshness=True,
    ),
    SessionInsightCountDescriptor(
        count_key="expected_tag_rollup_count",
        table_key="session_profiles",
        sql=EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL,
        requires_freshness=True,
        fallback_count_key="tag_rollup_count",
    ),
    SessionInsightCountDescriptor(
        count_key="stale_tag_rollup_count",
        table_key="session_tag_rollups",
        sql=STALE_SESSION_TAG_ROLLUP_COUNT_SQL,
        params=(SESSION_INSIGHT_MATERIALIZER_VERSION,),
        requires_freshness=True,
    ),
    SessionInsightCountDescriptor(
        count_key="expected_day_summary_count",
        table_key="session_profiles",
        sql=EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL,
        requires_freshness=True,
        fallback_count_key="day_summary_count",
    ),
    SessionInsightCountDescriptor(
        count_key="stale_day_summary_count",
        table_key="day_session_summaries",
        sql=STALE_DAY_SESSION_SUMMARY_COUNT_SQL,
        params=(SESSION_INSIGHT_MATERIALIZER_VERSION,),
        requires_freshness=True,
    ),
)

_READY_DESCRIPTORS: tuple[SessionInsightReadyDescriptor, ...] = (
    SessionInsightReadyDescriptor(
        ready_key="profile_rows_ready",
        table_key="session_profiles",
        zero_counts=("missing_profile_row_count", "stale_profile_row_count", "orphan_profile_row_count"),
    ),
    SessionInsightReadyDescriptor(
        ready_key="work_event_inference_rows_ready",
        table_key="session_work_events",
        equal_counts=(("work_event_inference_count", "expected_work_event_inference_count"),),
        zero_counts=("stale_work_event_inference_count", "orphan_work_event_inference_count"),
    ),
    SessionInsightReadyDescriptor(
        ready_key="phase_inference_rows_ready",
        table_key="session_phases",
        equal_counts=(("phase_inference_count", "expected_phase_inference_count"),),
        zero_counts=("stale_phase_inference_count", "orphan_phase_inference_count"),
    ),
    SessionInsightReadyDescriptor(
        ready_key="threads_ready",
        table_key="work_threads",
        equal_counts=(("thread_count", "root_threads"),),
        zero_counts=("stale_thread_count", "orphan_thread_count"),
    ),
    SessionInsightReadyDescriptor(
        ready_key="tag_rollups_ready",
        table_key="session_tag_rollups",
        equal_counts=(("tag_rollup_count", "expected_tag_rollup_count"),),
        zero_counts=("stale_tag_rollup_count",),
    ),
    SessionInsightReadyDescriptor(
        ready_key="day_summaries_ready",
        table_key="day_session_summaries",
        equal_counts=(("day_summary_count", "expected_day_summary_count"),),
        zero_counts=("stale_day_summary_count",),
    ),
    SessionInsightReadyDescriptor(
        ready_key="week_summaries_ready",
        table_key="day_session_summaries",
        equal_counts=(("day_summary_count", "expected_day_summary_count"),),
        zero_counts=("stale_day_summary_count",),
    ),
)


def _to_int(row: tuple[object, ...] | sqlite3.Row | None) -> int:
    if not row:
        return 0
    value = row[0]
    if isinstance(value, (int, float, str)):
        return int(value)
    return 0


def _tables_sync(conn: sqlite3.Connection) -> TablePresence:
    return {descriptor.key: bool(conn.execute(descriptor.exists_sql).fetchone()) for descriptor in _TABLE_DESCRIPTORS}


async def _tables_async(conn: aiosqlite.Connection) -> TablePresence:
    tables: TablePresence = {}
    for descriptor in _TABLE_DESCRIPTORS:
        tables[descriptor.key] = bool(await (await conn.execute(descriptor.exists_sql)).fetchone())
    return tables


def _count_sync(conn: sqlite3.Connection, sql: str, *params: object) -> int:
    return _to_int(conn.execute(sql, params).fetchone())


async def _count_async(conn: aiosqlite.Connection, sql: str, *params: object) -> int:
    return _to_int(await (await conn.execute(sql, params)).fetchone())


def _table_count_sync(conn: sqlite3.Connection, table_exists: bool, table_name: str) -> int:
    return _to_int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()) if table_exists else 0


async def _table_count_async(conn: aiosqlite.Connection, table_exists: bool, table_name: str) -> int:
    return _to_int(await (await conn.execute(f"SELECT COUNT(*) FROM {table_name}")).fetchone()) if table_exists else 0


def _table_row_counts_sync(conn: sqlite3.Connection, tables: TablePresence) -> StatusCounts:
    counts: StatusCounts = {}
    for descriptor in _TABLE_DESCRIPTORS:
        count = descriptor.count_sync(conn, tables)
        if count is not None:
            counts[count[0]] = count[1]
    return counts


async def _table_row_counts_async(conn: aiosqlite.Connection, tables: TablePresence) -> StatusCounts:
    counts: StatusCounts = {}
    for descriptor in _TABLE_DESCRIPTORS:
        count = await descriptor.count_async(conn, tables)
        if count is not None:
            counts[count[0]] = count[1]
    return counts


def _fts_projection_counts_sync(
    conn: sqlite3.Connection,
    tables: TablePresence,
    *,
    counts: StatusCounts,
    verify_freshness: bool,
) -> StatusCounts:
    projection_counts: StatusCounts = {}
    for descriptor in _FTS_DESCRIPTORS:
        projection_counts.update(descriptor.counts_sync(conn, tables, counts, verify_freshness=verify_freshness))
    return projection_counts


async def _fts_projection_counts_async(
    conn: aiosqlite.Connection,
    tables: TablePresence,
    *,
    counts: StatusCounts,
    verify_freshness: bool,
) -> StatusCounts:
    projection_counts: StatusCounts = {}
    for descriptor in _FTS_DESCRIPTORS:
        projection_counts.update(await descriptor.counts_async(conn, tables, counts, verify_freshness=verify_freshness))
    return projection_counts


def _materialized_counts_sync(
    conn: sqlite3.Connection,
    tables: TablePresence,
    *,
    verify_freshness: bool,
) -> StatusCounts:
    counts: StatusCounts = {
        "total_conversations": _count_sync(conn, TOTAL_CONVERSATIONS_SQL),
    }
    counts.update(_table_row_counts_sync(conn, tables))
    counts["root_threads"] = _count_sync(conn, ROOT_THREAD_COUNT_SQL) if verify_freshness else counts["thread_count"]
    counts.update(_fts_projection_counts_sync(conn, tables, counts=counts, verify_freshness=verify_freshness))
    return counts


def _descriptor_counts_sync(
    conn: sqlite3.Connection,
    tables: TablePresence,
    counts: StatusCounts,
    *,
    verify_freshness: bool,
) -> StatusCounts:
    descriptor_counts: StatusCounts = {}
    for descriptor in _COUNT_DESCRIPTORS:
        key, value = descriptor.count_sync(
            conn, tables, {**counts, **descriptor_counts}, verify_freshness=verify_freshness
        )
        descriptor_counts[key] = value
    return descriptor_counts


def _status_counts_sync(
    conn: sqlite3.Connection,
    tables: TablePresence,
    *,
    verify_freshness: bool,
) -> StatusCounts:
    counts = _materialized_counts_sync(conn, tables, verify_freshness=verify_freshness)
    counts.update(_descriptor_counts_sync(conn, tables, counts, verify_freshness=verify_freshness))
    return counts


def _status_payload(
    tables: TablePresence,
    counts: StatusCounts,
) -> SessionInsightStatusSnapshot:
    ready_flags: dict[SessionInsightReadyFlag, bool] = {
        descriptor.ready_key: descriptor.ready(tables, counts) for descriptor in _READY_DESCRIPTORS
    }
    ready_flags.update({descriptor.ready_key: descriptor.ready(tables, counts) for descriptor in _FTS_DESCRIPTORS})
    return SessionInsightStatusSnapshot(
        **counts,
        profile_rows_ready=ready_flags["profile_rows_ready"],
        profile_merged_fts_ready=ready_flags["profile_merged_fts_ready"],
        profile_evidence_fts_ready=ready_flags["profile_evidence_fts_ready"],
        profile_inference_fts_ready=ready_flags["profile_inference_fts_ready"],
        profile_enrichment_fts_ready=ready_flags["profile_enrichment_fts_ready"],
        work_event_inference_rows_ready=ready_flags["work_event_inference_rows_ready"],
        work_event_inference_fts_ready=ready_flags["work_event_inference_fts_ready"],
        phase_inference_rows_ready=ready_flags["phase_inference_rows_ready"],
        threads_ready=ready_flags["threads_ready"],
        threads_fts_ready=ready_flags["threads_fts_ready"],
        tag_rollups_ready=ready_flags["tag_rollups_ready"],
        day_summaries_ready=ready_flags["day_summaries_ready"],
        week_summaries_ready=ready_flags["week_summaries_ready"],
    )


def session_profile_repair_candidate_ids_sync(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        SESSION_PROFILE_REPAIR_CANDIDATES_SQL,
        (SESSION_INSIGHT_MATERIALIZER_VERSION,),
    ).fetchall()
    return [str(row["conversation_id"]) for row in rows]


async def session_profile_repair_candidate_ids_async(conn: aiosqlite.Connection) -> list[str]:
    rows = await (
        await conn.execute(
            SESSION_PROFILE_REPAIR_CANDIDATES_SQL,
            (SESSION_INSIGHT_MATERIALIZER_VERSION,),
        )
    ).fetchall()
    return [str(row["conversation_id"]) for row in rows]


def session_insight_status_sync(
    conn: sqlite3.Connection,
    *,
    verify_freshness: bool = True,
) -> SessionInsightStatusSnapshot:
    tables = _tables_sync(conn)
    counts = _status_counts_sync(conn, tables, verify_freshness=verify_freshness)
    return _status_payload(tables, counts)


async def _materialized_counts_async(
    conn: aiosqlite.Connection,
    tables: TablePresence,
    *,
    verify_freshness: bool,
) -> StatusCounts:
    counts: StatusCounts = {
        "total_conversations": await _count_async(conn, TOTAL_CONVERSATIONS_SQL),
    }
    counts.update(await _table_row_counts_async(conn, tables))
    counts["root_threads"] = (
        await _count_async(conn, ROOT_THREAD_COUNT_SQL) if verify_freshness else counts["thread_count"]
    )
    counts.update(await _fts_projection_counts_async(conn, tables, counts=counts, verify_freshness=verify_freshness))
    return counts


async def _descriptor_counts_async(
    conn: aiosqlite.Connection,
    tables: TablePresence,
    counts: StatusCounts,
    *,
    verify_freshness: bool,
) -> StatusCounts:
    descriptor_counts: StatusCounts = {}
    for descriptor in _COUNT_DESCRIPTORS:
        key, value = await descriptor.count_async(
            conn,
            tables,
            {**counts, **descriptor_counts},
            verify_freshness=verify_freshness,
        )
        descriptor_counts[key] = value
    return descriptor_counts


async def _status_counts_async(
    conn: aiosqlite.Connection,
    tables: TablePresence,
    *,
    verify_freshness: bool,
) -> StatusCounts:
    counts = await _materialized_counts_async(conn, tables, verify_freshness=verify_freshness)
    counts.update(await _descriptor_counts_async(conn, tables, counts, verify_freshness=verify_freshness))
    return counts


async def session_insight_status_async(
    conn: aiosqlite.Connection,
    *,
    verify_freshness: bool = True,
) -> SessionInsightStatusSnapshot:
    """Return session-insight table/readiness status.

    With `verify_freshness=False`, the result is a lightweight approximation:
    `root_threads` falls back to `thread_count`, and readiness flags depending
    on freshness-gated stale/orphan/expected counts can appear ready without
    fresh verification. Use `verify_freshness=True` for full health checks.
    """

    tables = await _tables_async(conn)
    counts = await _status_counts_async(conn, tables, verify_freshness=verify_freshness)
    return _status_payload(tables, counts)


__all__ = [
    "SessionInsightCountDescriptor",
    "SessionInsightFtsDescriptor",
    "SessionInsightReadyDescriptor",
    "SessionInsightTableDescriptor",
    "session_insight_status_async",
    "session_insight_status_sync",
    "session_profile_repair_candidate_ids_async",
    "session_profile_repair_candidate_ids_sync",
]
