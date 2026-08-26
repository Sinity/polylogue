"""Connection-bound ArchiveStore capability for high-volume insight reads.

This is deliberately synchronous: :mod:`polylogue.archive.query.execution_control`
owns the async boundary, connection lifetime, snapshot, deadline, and cancellation.
The capability owns only the read SQL and its row-to-insight mapping.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime

from polylogue.insights.archive import ArchiveCoverageInsight
from polylogue.insights.archive_models import ArchiveInsightProvenance
from polylogue.insights.tool_usage import ToolUsageInsight, ToolUsageInsightQuery, build_tool_usage_insight
from polylogue.storage.sqlite.queries.tool_usage import ToolUsageOriginCoverageRow, ToolUsageRow

OriginNormalizer = Callable[[str | None], str | None]
IsoFromMilliseconds = Callable[[object], str | None]


class ArchiveReadInsights:
    """Insight reads over one caller-owned archive snapshot and connection."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        normalize_origin: OriginNormalizer,
        iso_from_milliseconds: IsoFromMilliseconds,
    ) -> None:
        self._conn = conn
        self._normalize_origin = normalize_origin
        self._iso_from_milliseconds = iso_from_milliseconds

    def list_tool_usage_insights(self, query: ToolUsageInsightQuery | None = None) -> list[ToolUsageInsight]:
        """Aggregate tool-usage insights from action rows."""
        request = query or ToolUsageInsightQuery()
        builder_request = self._tool_usage_builder_query(request)
        insight = build_tool_usage_insight(
            rows=self._tool_usage_rows(request),
            coverage_rows=self._tool_usage_origin_coverage_rows(),
            query=builder_request,
            materialized_at=datetime.now(UTC).isoformat(),
        )
        return [insight]

    def _tool_usage_rows(self, query: ToolUsageInsightQuery | None = None) -> list[ToolUsageRow]:
        request = query or ToolUsageInsightQuery()
        where: list[str] = []
        params: list[object] = []
        origin = self._normalize_origin(request.origin)
        if origin:
            where.append("s.origin = ?")
            params.append(origin)
        tool_expr = "COALESCE(NULLIF(LOWER(a.tool_name), ''), 'unknown')"
        if request.tool:
            where.append(f"{tool_expr} = LOWER(?)")
            params.append(request.tool)
        if request.mcp_server:
            mcp_prefix = f"mcp__{request.mcp_server.lower()}__"
            where.append(f"{tool_expr} >= ?")
            where.append(f"{tool_expr} < ?")
            params.append(mcp_prefix)
            params.append(f"{mcp_prefix}\U0010ffff")
        if request.action_kind:
            where.append("COALESCE(NULLIF(a.semantic_type, ''), 'tool_use') = ?")
            params.append(request.action_kind)
        if request.since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(request.since_ms)

        sql = """
            SELECT
                s.origin AS origin,
                {tool_expr} AS normalized_tool_name,
                COALESCE(NULLIF(a.semantic_type, ''), 'tool_use') AS action_kind,
                COUNT(*) AS call_count,
                COUNT(DISTINCT s.session_id) AS session_count,
                COUNT(DISTINCT a.message_id) AS message_count,
                COUNT(DISTINCT a.tool_use_block_id) AS distinct_tool_ids,
                SUM(CASE WHEN a.tool_path IS NOT NULL AND a.tool_path != '' THEN 1 ELSE 0 END) AS affected_path_calls,
                SUM(CASE WHEN a.output_text IS NOT NULL AND a.output_text != '' THEN 1 ELSE 0 END) AS output_text_calls
            FROM actions a
            JOIN sessions s ON s.session_id = a.session_id
            {where_clause}
            GROUP BY s.origin, normalized_tool_name, action_kind
            ORDER BY call_count DESC, s.origin ASC, normalized_tool_name ASC
            {limit_clause}
            """
        if request.limit is not None:
            limit_clause = "LIMIT ? OFFSET ?"
            params.extend((request.limit, request.offset))
        elif request.offset:
            limit_clause = "LIMIT -1 OFFSET ?"
            params.append(request.offset)
        else:
            limit_clause = ""
        rows = self._conn.execute(
            sql.format(
                tool_expr=tool_expr,
                where_clause=("WHERE " + " AND ".join(where)) if where else "",
                limit_clause=limit_clause,
            ),
            tuple(params),
        ).fetchall()
        return [
            {
                "origin": str(row["origin"] or "unknown-export"),
                "normalized_tool_name": str(row["normalized_tool_name"] or "unknown"),
                "action_kind": str(row["action_kind"] or "tool_use"),
                "call_count": int(row["call_count"] or 0),
                "session_count": int(row["session_count"] or 0),
                "message_count": int(row["message_count"] or 0),
                "distinct_tool_ids": int(row["distinct_tool_ids"] or 0),
                "affected_path_calls": int(row["affected_path_calls"] or 0),
                "output_text_calls": int(row["output_text_calls"] or 0),
            }
            for row in rows
        ]

    def _tool_usage_origin_coverage_rows(self) -> list[ToolUsageOriginCoverageRow]:
        rows = self._conn.execute(
            """
            SELECT
                s.origin AS origin,
                COUNT(DISTINCT s.session_id) AS session_count,
                COUNT(a.tool_use_block_id) AS action_count,
                COUNT(DISTINCT COALESCE(NULLIF(LOWER(a.tool_name), ''), 'unknown')) AS distinct_tool_count,
                COUNT(DISTINCT COALESCE(NULLIF(a.semantic_type, ''), 'tool_use')) AS distinct_action_kind_count,
                COUNT(a.tool_use_block_id) AS has_tool_id_signal,
                SUM(CASE WHEN a.tool_path IS NOT NULL AND a.tool_path != '' THEN 1 ELSE 0 END) AS has_affected_paths_signal,
                SUM(CASE WHEN a.output_text IS NOT NULL AND a.output_text != '' THEN 1 ELSE 0 END) AS has_output_text_signal
            FROM sessions s
            LEFT JOIN actions a ON a.session_id = s.session_id
            GROUP BY s.origin
            ORDER BY action_count DESC, session_count DESC, s.origin ASC
            """
        ).fetchall()
        return [
            {
                "origin": str(row["origin"] or "unknown-export"),
                "session_count": int(row["session_count"] or 0),
                "action_count": int(row["action_count"] or 0),
                "distinct_tool_count": int(row["distinct_tool_count"] or 0),
                "distinct_action_kind_count": int(row["distinct_action_kind_count"] or 0),
                "has_tool_id_signal": int(row["has_tool_id_signal"] or 0),
                "has_affected_paths_signal": int(row["has_affected_paths_signal"] or 0),
                "has_output_text_signal": int(row["has_output_text_signal"] or 0),
            }
            for row in rows
        ]

    def list_archive_coverage_insights(
        self,
        *,
        group_by: str = "origin",
        origin: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ArchiveCoverageInsight]:
        """Aggregate archive coverage from index tables."""
        normalized_origin = self._normalize_origin(origin)
        if group_by == "origin":
            return self._origin_coverage_insights(origin=normalized_origin, limit=limit, offset=offset)
        if group_by == "day":
            return self._time_bucket_coverage_insights(
                bucket_format="%Y-%m-%d",
                group_by="day",
                origin=normalized_origin,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=limit,
                offset=offset,
            )
        if group_by == "week":
            return self._time_bucket_coverage_insights(
                bucket_format="%Y-W%W",
                group_by="week",
                origin=normalized_origin,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=limit,
                offset=offset,
            )
        raise ValueError("archive coverage group_by must be one of: origin, day, week")

    def _origin_coverage_insights(
        self,
        *,
        origin: str | None,
        limit: int | None,
        offset: int,
    ) -> list[ArchiveCoverageInsight]:
        where = ""
        params: list[object] = []
        if origin is not None:
            where = "WHERE s.origin = ?"
            params.append(origin)
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT
                s.origin,
                COUNT(*) AS session_count,
                SUM(s.message_count) AS message_count,
                SUM(s.user_message_count) AS user_message_count,
                SUM(s.authored_user_message_count) AS authored_user_message_count,
                SUM(s.assistant_message_count) AS assistant_message_count,
                SUM(s.user_word_count) AS user_word_sum,
                SUM(s.authored_user_word_count) AS authored_user_word_sum,
                SUM(s.assistant_word_count) AS assistant_word_sum,
                SUM(s.tool_use_count) AS tool_use_count,
                SUM(s.thinking_count) AS thinking_count,
                SUM(CASE WHEN s.tool_use_count > 0 THEN 1 ELSE 0 END) AS sessions_with_tools,
                SUM(CASE WHEN s.thinking_count > 0 THEN 1 ELSE 0 END) AS sessions_with_thinking
            FROM sessions s
            {where}
            GROUP BY s.origin
            ORDER BY session_count DESC, s.origin
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [_origin_coverage_from_archive_row(row) for row in rows]

    def _time_bucket_coverage_insights(
        self,
        *,
        bucket_format: str,
        group_by: str,
        origin: str | None,
        since_ms: int | None,
        until_ms: int | None,
        limit: int | None,
        offset: int,
    ) -> list[ArchiveCoverageInsight]:
        where: list[str] = []
        params: list[object] = []
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT
                strftime('{bucket_format}', s.sort_key_ms / 1000, 'unixepoch') AS bucket,
                COUNT(DISTINCT s.session_id) AS session_count,
                COUNT(DISTINCT COALESCE(s.root_session_id, s.session_id)) AS logical_session_count,
                SUM(s.message_count) AS message_count,
                SUM(s.word_count) AS total_words,
                SUM(COALESCE((SELECT COALESCE(SUM(u.cost_usd), s.reported_cost_usd)
                              FROM session_model_usage u WHERE u.session_id = s.session_id), 0.0)) AS total_cost_usd,
                SUM(COALESCE(sp.duration_ms, 0)) AS total_duration_ms,
                SUM(COALESCE(sp.duration_ms, 0)) AS total_wall_duration_ms,
                MAX(s.sort_key_ms) AS source_sort_key_ms
            FROM sessions s
            LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
            {clause}
            GROUP BY bucket
            HAVING bucket IS NOT NULL
            ORDER BY bucket DESC
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [
            ArchiveCoverageInsight(
                group_by=group_by,
                bucket=str(row["bucket"]),
                session_count=int(row["session_count"] or 0),
                logical_session_count=int(row["logical_session_count"] or 0),
                message_count=int(row["message_count"] or 0),
                total_cost_usd=float(row["total_cost_usd"] or 0.0),
                total_duration_ms=int(row["total_duration_ms"] or 0),
                total_wall_duration_ms=int(row["total_wall_duration_ms"] or 0),
                total_words=int(row["total_words"] or 0),
                avg_messages_per_session=(
                    int(row["message_count"] or 0) / int(row["session_count"])
                    if int(row["session_count"] or 0)
                    else None
                ),
                work_event_breakdown=_coverage_work_event_breakdown(
                    self._conn, str(row["bucket"]), bucket_format, origin=origin, since_ms=since_ms, until_ms=until_ms
                ),
                repos_active=_coverage_repos_active(
                    self._conn, str(row["bucket"]), bucket_format, origin=origin, since_ms=since_ms, until_ms=until_ms
                ),
                origin_breakdown=_coverage_origin_breakdown(
                    self._conn, str(row["bucket"]), bucket_format, origin=origin, since_ms=since_ms, until_ms=until_ms
                ),
                provenance=ArchiveInsightProvenance(
                    materializer_version=1,
                    materialized_at=None,
                    source_updated_at=self._iso_from_milliseconds(row["source_sort_key_ms"]),
                    source_sort_key=(
                        float(row["source_sort_key_ms"]) / 1000.0 if row["source_sort_key_ms"] is not None else None
                    ),
                    input_high_water_mark=self._iso_from_milliseconds(row["source_sort_key_ms"]),
                    input_high_water_mark_source="sort_key" if row["source_sort_key_ms"] is not None else None,
                    time_confidence="estimated" if row["source_sort_key_ms"] is not None else "unknown",
                ),
            )
            for row in rows
        ]

    def _tool_usage_builder_query(self, query: ToolUsageInsightQuery) -> ToolUsageInsightQuery:
        origin = self._normalize_origin(query.origin)
        updates: dict[str, object] = {"limit": None, "offset": 0}
        if origin is None:
            return query.model_copy(update=updates)
        updates["origin"] = origin
        return query.model_copy(update=updates)


def _origin_coverage_from_archive_row(row: sqlite3.Row) -> ArchiveCoverageInsight:
    session_count = int(row["session_count"] or 0)
    message_count = int(row["message_count"] or 0)
    user_message_count = int(row["user_message_count"] or 0)
    authored_user_message_count = int(row["authored_user_message_count"] or 0)
    assistant_message_count = int(row["assistant_message_count"] or 0)
    user_word_sum = int(row["user_word_sum"] or 0)
    authored_user_word_sum = int(row["authored_user_word_sum"] or 0)
    assistant_word_sum = int(row["assistant_word_sum"] or 0)
    sessions_with_tools = int(row["sessions_with_tools"] or 0)
    sessions_with_thinking = int(row["sessions_with_thinking"] or 0)
    origin = str(row["origin"])
    return ArchiveCoverageInsight(
        group_by="origin",
        bucket=origin,
        origin=origin,
        session_count=session_count,
        message_count=message_count,
        user_message_count=user_message_count,
        authored_user_message_count=authored_user_message_count,
        assistant_message_count=assistant_message_count,
        avg_messages_per_session=(message_count / session_count if session_count else None),
        avg_user_words=(user_word_sum / user_message_count if user_message_count else None),
        avg_authored_user_words=(
            authored_user_word_sum / authored_user_message_count if authored_user_message_count else None
        ),
        avg_assistant_words=(assistant_word_sum / assistant_message_count if assistant_message_count else None),
        tool_use_count=int(row["tool_use_count"] or 0),
        thinking_count=int(row["thinking_count"] or 0),
        total_sessions_with_tools=sessions_with_tools,
        total_sessions_with_thinking=sessions_with_thinking,
        tool_use_percentage=((sessions_with_tools / session_count) * 100 if session_count else None),
        thinking_percentage=((sessions_with_thinking / session_count) * 100 if session_count else None),
    )


def _coverage_bucket_filter(
    bucket: str,
    bucket_format: str,
    *,
    origin: str | None,
    since_ms: int | None,
    until_ms: int | None,
) -> tuple[str, tuple[object, ...]]:
    clauses = ["strftime(?, s.sort_key_ms / 1000, 'unixepoch') = ?"]
    params: list[object] = [bucket_format, bucket]
    if origin is not None:
        clauses.append("s.origin = ?")
        params.append(origin)
    if since_ms is not None:
        clauses.append("s.sort_key_ms >= ?")
        params.append(since_ms)
    if until_ms is not None:
        clauses.append("s.sort_key_ms <= ?")
        params.append(until_ms)
    return "WHERE " + " AND ".join(clauses), tuple(params)


def _coverage_work_event_breakdown(
    conn: sqlite3.Connection,
    bucket: str,
    bucket_format: str,
    *,
    origin: str | None,
    since_ms: int | None,
    until_ms: int | None,
) -> dict[str, int]:
    where, params = _coverage_bucket_filter(bucket, bucket_format, origin=origin, since_ms=since_ms, until_ms=until_ms)
    rows = conn.execute(
        f"""
        SELECT e.work_event_type, COUNT(*) AS count
        FROM sessions s
        JOIN session_work_events e ON e.session_id = s.session_id
        {where}
        GROUP BY e.work_event_type
        ORDER BY count DESC, e.work_event_type
        """,
        params,
    ).fetchall()
    return {str(row["work_event_type"]): int(row["count"] or 0) for row in rows}


def _coverage_repos_active(
    conn: sqlite3.Connection,
    bucket: str,
    bucket_format: str,
    *,
    origin: str | None,
    since_ms: int | None,
    until_ms: int | None,
) -> tuple[str, ...]:
    where, params = _coverage_bucket_filter(bucket, bucket_format, origin=origin, since_ms=since_ms, until_ms=until_ms)
    rows = conn.execute(
        f"""
        SELECT DISTINCT COALESCE(NULLIF(r.repo_name, ''), NULLIF(r.root_path, ''), NULLIF(r.origin_url, '')) AS repo
        FROM sessions s
        JOIN session_repos sr ON sr.session_id = s.session_id
        JOIN repos r ON r.repo_id = sr.repo_id
        {where}
        ORDER BY repo
        """,
        params,
    ).fetchall()
    return tuple(str(row["repo"]) for row in rows if row["repo"])


def _coverage_origin_breakdown(
    conn: sqlite3.Connection,
    bucket: str,
    bucket_format: str,
    *,
    origin: str | None,
    since_ms: int | None,
    until_ms: int | None,
) -> dict[str, int]:
    where, params = _coverage_bucket_filter(bucket, bucket_format, origin=origin, since_ms=since_ms, until_ms=until_ms)
    rows = conn.execute(
        f"""
        SELECT s.origin, COUNT(DISTINCT s.session_id) AS count
        FROM sessions s
        {where}
        GROUP BY s.origin
        ORDER BY count DESC, s.origin
        """,
        params,
    ).fetchall()
    return {str(row["origin"]): int(row["count"] or 0) for row in rows}


__all__ = ["ArchiveReadInsights"]
