"""Tool-usage aggregation queries over canonical action rows.

These queries roll up tool calls per origin, per normalized tool name,
without inferring anything the substrate has not already classified.

The companion coverage query reports — for every origin observed in the
``sessions`` table — whether the canonical ``actions`` view has any rows
for that origin.
"""

from __future__ import annotations

import aiosqlite
from typing_extensions import TypedDict

from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.insights.tool_usage import ToolUsageInsightQuery

__all__ = [
    "ToolUsageProviderCoverageRow",
    "ToolUsageRow",
    "get_tool_usage_provider_coverage_rows",
    "get_tool_usage_rows",
]


class ToolUsageRow(TypedDict):
    """One row per (origin, normalized_tool_name)."""

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
    """Per-origin coverage signal — does the substrate carry tool data?"""

    source_name: str
    session_count: int
    action_count: int
    distinct_tool_count: int
    distinct_action_kind_count: int
    has_tool_id_signal: int
    has_affected_paths_signal: int
    has_output_text_signal: int


async def get_tool_usage_rows(
    conn: aiosqlite.Connection,
    query: ToolUsageInsightQuery | None = None,
) -> list[ToolUsageRow]:
    """Aggregate tool usage from canonical ``actions`` rows.

    The grouping key is ``(source_name, normalized_tool_name, action_kind)``
    because the same normalized tool name can fire under multiple action
    categories.
    """

    request = query or ToolUsageInsightQuery()
    where: list[str] = []
    params: list[object] = []
    origin = _origin_for_tool_usage_filter(request.provider)
    if origin:
        where.append("s.origin = ?")
        params.append(origin)
    if request.tool:
        where.append("COALESCE(NULLIF(LOWER(a.tool_name), ''), 'unknown') = LOWER(?)")
        params.append(request.tool)
    if request.mcp_server:
        where.append("COALESCE(NULLIF(LOWER(a.tool_name), ''), 'unknown') LIKE ?")
        params.append(f"mcp__{request.mcp_server.lower()}__%")
    if request.action_kind:
        where.append("COALESCE(NULLIF(a.semantic_type, ''), 'tool_use') = ?")
        params.append(request.action_kind)
    if request.limit is not None:
        limit_clause = "LIMIT ? OFFSET ?"
        params.extend((request.limit, request.offset))
    elif request.offset:
        limit_clause = "LIMIT -1 OFFSET ?"
        params.append(request.offset)
    else:
        limit_clause = ""

    cursor = await conn.execute(
        f"""
        SELECT
            s.origin AS origin,
            COALESCE(NULLIF(LOWER(a.tool_name), ''), 'unknown') AS normalized_tool_name,
            COALESCE(NULLIF(a.semantic_type, ''), 'tool_use') AS action_kind,
            COUNT(*) AS call_count,
            COUNT(DISTINCT a.session_id) AS session_count,
            COUNT(DISTINCT a.message_id) AS message_count,
            COUNT(DISTINCT a.tool_use_block_id) AS distinct_tool_ids,
            SUM(CASE WHEN a.tool_path IS NOT NULL AND a.tool_path != '' THEN 1 ELSE 0 END) AS affected_path_calls,
            SUM(CASE WHEN a.output_text IS NOT NULL AND a.output_text != '' THEN 1 ELSE 0 END) AS output_text_calls
        FROM actions a
        JOIN sessions s ON s.session_id = a.session_id
        {"WHERE " + " AND ".join(where) if where else ""}
        GROUP BY
            s.origin,
            normalized_tool_name,
            action_kind
        ORDER BY call_count DESC, s.origin ASC, normalized_tool_name ASC
        {limit_clause}
        """,
        tuple(params),
    )
    rows = await cursor.fetchall()
    return [
        {
            "source_name": _provider_for_origin(str(row["origin"] or "unknown-export")).value,
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
    """Report tool-data coverage signals for every origin in the archive.

    Returns one row per origin that has at least one session. ``action_count``
    counts rows exposed by the canonical ``actions`` view over tool blocks.
    """

    cursor = await conn.execute(
        """
        SELECT
            s.origin AS source_name,
            COUNT(DISTINCT s.session_id) AS session_count,
            COALESCE(COUNT(a.tool_use_block_id), 0) AS action_count,
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
    )
    rows = await cursor.fetchall()
    return [
        {
            "source_name": str(row["source_name"] or "unknown"),
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


def _origin_for_tool_usage_filter(provider_or_origin: str | None) -> str | None:
    if provider_or_origin is None:
        return None
    known_origin = {
        "claude-code-session",
        "codex-session",
        "gemini-cli-session",
        "hermes-session",
        "antigravity-session",
        "chatgpt-export",
        "claude-ai-export",
        "aistudio-drive",
        "unknown-export",
    }
    if provider_or_origin in known_origin:
        return provider_or_origin
    return origin_from_provider(Provider.from_string(provider_or_origin)).value


def _provider_for_origin(origin: str) -> Provider:
    return {
        "claude-code-session": Provider.CLAUDE_CODE,
        "codex-session": Provider.CODEX,
        "gemini-cli-session": Provider.GEMINI_CLI,
        "hermes-session": Provider.HERMES,
        "antigravity-session": Provider.ANTIGRAVITY,
        "chatgpt-export": Provider.CHATGPT,
        "claude-ai-export": Provider.CLAUDE_AI,
        "aistudio-drive": Provider.GEMINI,
        "unknown-export": Provider.UNKNOWN,
    }.get(origin, Provider.UNKNOWN)
