"""Tool-usage aggregation queries over canonical action_events.

These queries roll up tool calls per provider, per normalized tool name,
without inferring anything the substrate has not already classified.

The companion coverage query reports — for every provider observed in the
``sessions`` table — whether the action_events substrate has any rows
for that provider. This is the load-bearing distinction the issue calls
out: zero observed action_events for a provider may mean the provider does
not expose tool data at all, not that the user has not used tools.
"""

from __future__ import annotations

import aiosqlite
from typing_extensions import TypedDict

__all__ = [
    "ToolUsageProviderCoverageRow",
    "ToolUsageRow",
    "get_tool_usage_provider_coverage_rows",
    "get_tool_usage_rows",
]


class ToolUsageRow(TypedDict):
    """One row per (provider, normalized_tool_name)."""

    source_name: str
    normalized_tool_name: str
    action_kind: str
    call_count: int
    session_count: int
    message_count: int
    distinct_tool_ids: int
    affected_path_calls: int
    output_text_calls: int


class ToolUsageProviderCoverageRow(TypedDict):
    """Per-provider coverage signal — does the substrate carry tool data?"""

    source_name: str
    session_count: int
    action_event_count: int
    distinct_tool_count: int
    distinct_action_kind_count: int
    has_tool_id_signal: int
    has_affected_paths_signal: int
    has_output_text_signal: int


async def get_tool_usage_rows(
    conn: aiosqlite.Connection,
) -> list[ToolUsageRow]:
    """Aggregate tool usage from canonical ``action_events`` rows.

    The grouping key is ``(source_name, normalized_tool_name, action_kind)``
    because the same normalized tool name can fire under multiple action
    kinds (for example ``tool_use`` vs ``tool_result`` reflections).
    """

    cursor = await conn.execute(
        """
        SELECT
            COALESCE(NULLIF(source_name, ''), 'unknown')              AS source_name,
            normalized_tool_name                                        AS normalized_tool_name,
            action_kind                                                 AS action_kind,
            COUNT(*)                                                    AS call_count,
            COUNT(DISTINCT session_id)                             AS session_count,
            COUNT(DISTINCT message_id)                                  AS message_count,
            COUNT(DISTINCT tool_id)                                     AS distinct_tool_ids,
            SUM(CASE WHEN affected_paths_json IS NOT NULL AND affected_paths_json != '[]' THEN 1 ELSE 0 END)
                                                                        AS affected_path_calls,
            SUM(CASE WHEN output_text IS NOT NULL AND output_text != '' THEN 1 ELSE 0 END)
                                                                        AS output_text_calls
        FROM action_events
        GROUP BY
            COALESCE(NULLIF(source_name, ''), 'unknown'),
            normalized_tool_name,
            action_kind
        ORDER BY call_count DESC, source_name ASC, normalized_tool_name ASC
        """
    )
    rows = await cursor.fetchall()
    return [
        {
            "source_name": str(row["source_name"] or "unknown"),
            "normalized_tool_name": str(row["normalized_tool_name"] or "unknown"),
            "action_kind": str(row["action_kind"] or "unknown"),
            "call_count": int(row["call_count"] or 0),
            "session_count": int(row["session_count"] or 0),
            "message_count": int(row["message_count"] or 0),
            "distinct_tool_ids": int(row["distinct_tool_ids"] or 0),
            "affected_path_calls": int(row["affected_path_calls"] or 0),
            "output_text_calls": int(row["output_text_calls"] or 0),
        }
        for row in rows
    ]


async def get_tool_usage_provider_coverage_rows(
    conn: aiosqlite.Connection,
) -> list[ToolUsageProviderCoverageRow]:
    """Report tool-data coverage signals for every provider in the archive.

    Returns one row per provider that has at least one session. The
    ``action_event_count`` column is the load-bearing field — a provider
    with sessions but zero action events is explicitly visible as a
    coverage gap rather than collapsed into the rollup as a quiet zero.
    """

    cursor = await conn.execute(
        """
        SELECT
            c.source_name                                                AS source_name,
            COUNT(DISTINCT c.session_id)                              AS session_count,
            COALESCE(COUNT(ae.event_id), 0)                                AS action_event_count,
            COUNT(DISTINCT ae.normalized_tool_name)                        AS distinct_tool_count,
            COUNT(DISTINCT ae.action_kind)                                 AS distinct_action_kind_count,
            SUM(CASE WHEN ae.tool_id IS NOT NULL AND ae.tool_id != '' THEN 1 ELSE 0 END)
                                                                            AS has_tool_id_signal,
            SUM(CASE
                WHEN ae.affected_paths_json IS NOT NULL AND ae.affected_paths_json != '[]'
                THEN 1 ELSE 0 END)                                          AS has_affected_paths_signal,
            SUM(CASE WHEN ae.output_text IS NOT NULL AND ae.output_text != '' THEN 1 ELSE 0 END)
                                                                            AS has_output_text_signal
        FROM sessions c
        LEFT JOIN action_events ae ON ae.session_id = c.session_id
        GROUP BY c.source_name
        ORDER BY action_event_count DESC, session_count DESC, c.source_name ASC
        """
    )
    rows = await cursor.fetchall()
    return [
        {
            "source_name": str(row["source_name"] or "unknown"),
            "session_count": int(row["session_count"] or 0),
            "action_event_count": int(row["action_event_count"] or 0),
            "distinct_tool_count": int(row["distinct_tool_count"] or 0),
            "distinct_action_kind_count": int(row["distinct_action_kind_count"] or 0),
            "has_tool_id_signal": int(row["has_tool_id_signal"] or 0),
            "has_affected_paths_signal": int(row["has_affected_paths_signal"] or 0),
            "has_output_text_signal": int(row["has_output_text_signal"] or 0),
        }
        for row in rows
    ]
