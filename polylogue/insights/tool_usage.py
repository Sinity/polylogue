"""Tool usage insight contract — per-origin, per-tool rollups with coverage.

The contract has two halves that are surfaced together:

* a per-tool aggregation entry (``ToolUsageEntry``) — call counts,
  session counts, MCP-server identity when present, and which
  optional substrate fields are populated;
* a per-origin coverage entry (``ToolUsageCoverageEntry``) — for every
  origin that appears in the archive, whether the canonical ``actions``
  view exposes any rows. An origin with sessions but zero actions is the
  explicit "data unavailable" signal, not a quiet zero.

Both halves are returned in a single ``ToolUsageInsight`` envelope so MCP
and CLI consumers always see usage and coverage together. The presence of
coverage gaps is preserved through the envelope's ``has_coverage_gaps``
flag and the per-entry ``data_available`` field.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from polylogue.insights.archive import PaginatedInsightQuery
from polylogue.insights.archive_models import (
    ARCHIVE_INSIGHT_CONTRACT_VERSION,
    ArchiveInsightModel,
    ArchiveInsightProvenance,
)

if TYPE_CHECKING:
    from polylogue.storage.sqlite.queries.tool_usage import (
        ToolUsageProviderCoverageRow,
        ToolUsageRow,
    )

TOOL_USAGE_INSIGHT_VERSION = 1
"""Insight materializer version for ``tool_usage`` envelopes."""

_MCP_TOOL_PREFIX = "mcp__"


def extract_mcp_server(tool_name: str) -> str | None:
    """Return the MCP server identity prefix of a tool name, if any.

    MCP tool names follow the ``mcp__<server>__<tool>`` convention used by
    Claude Code, Codex, and the Polylogue MCP catalog. The server segment
    is the second underscored component. ``None`` is returned for tool
    names that do not match this shape so downstream callers can keep the
    coverage signal honest instead of inventing a synthetic server.
    """

    if not tool_name or not tool_name.startswith(_MCP_TOOL_PREFIX):
        return None
    remainder = tool_name[len(_MCP_TOOL_PREFIX) :]
    if "__" not in remainder:
        return None
    server, _, _ = remainder.partition("__")
    return server or None


class ToolUsageEntry(ArchiveInsightModel):
    """One per-(origin, tool, action_kind) rollup row."""

    source_name: str
    normalized_tool_name: str
    action_kind: str
    call_count: int
    session_count: int
    message_count: int
    distinct_tool_ids: int
    affected_path_calls: int
    output_text_calls: int
    mcp_server: str | None = None


class ToolUsageCoverageEntry(ArchiveInsightModel):
    """Per-origin tool-data coverage signal."""

    source_name: str
    session_count: int
    action_count: int
    distinct_tool_count: int
    distinct_action_kind_count: int
    has_tool_id_signal: bool
    has_affected_paths_signal: bool
    has_output_text_signal: bool
    data_available: bool
    """True when the provider exposes any action rows in the archive."""


class ToolUsageInsight(ArchiveInsightModel):
    """Aggregate envelope returned by ``list_tool_usage_insights``.

    The envelope itself is a single insight item. Each call returns
    exactly one ``ToolUsageInsight`` containing every aggregation entry
    and every provider coverage entry resolved by the query. This shape
    keeps usage and coverage atomically consistent — readers never see
    one half without the other.
    """

    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "tool_usage"
    materializer_version: int = TOOL_USAGE_INSIGHT_VERSION
    entries: tuple[ToolUsageEntry, ...] = ()
    provider_coverage: tuple[ToolUsageCoverageEntry, ...] = ()
    total_call_count: int = 0
    total_distinct_tools: int = 0
    providers_with_data: int = 0
    providers_without_data: int = 0
    has_coverage_gaps: bool = False
    provenance: ArchiveInsightProvenance


class ToolUsageInsightQuery(PaginatedInsightQuery):
    """Query parameters for ``list_tool_usage_insights``.

    ``provider`` and ``tool`` narrow the returned aggregation entries but
    never the coverage map — coverage gaps are always reported across all
    providers so a narrowed query never hides them.
    """

    provider: str | None = None
    tool: str | None = None
    mcp_server: str | None = None
    action_kind: str | None = None
    limit: int | None = None


def _tool_usage_entry(row: ToolUsageRow) -> ToolUsageEntry:
    tool_name = row["normalized_tool_name"]
    return ToolUsageEntry(
        source_name=row["source_name"],
        normalized_tool_name=tool_name,
        action_kind=row["action_kind"],
        call_count=row["call_count"],
        session_count=row["session_count"],
        message_count=row["message_count"],
        distinct_tool_ids=row["distinct_tool_ids"],
        affected_path_calls=row["affected_path_calls"],
        output_text_calls=row["output_text_calls"],
        mcp_server=extract_mcp_server(tool_name),
    )


def _tool_usage_coverage(row: ToolUsageProviderCoverageRow) -> ToolUsageCoverageEntry:
    action_count = row["action_count"]
    return ToolUsageCoverageEntry(
        source_name=row["source_name"],
        session_count=row["session_count"],
        action_count=action_count,
        distinct_tool_count=row["distinct_tool_count"],
        distinct_action_kind_count=row["distinct_action_kind_count"],
        has_tool_id_signal=row["has_tool_id_signal"] > 0,
        has_affected_paths_signal=row["has_affected_paths_signal"] > 0,
        has_output_text_signal=row["has_output_text_signal"] > 0,
        data_available=action_count > 0,
    )


def build_tool_usage_insight(
    *,
    rows: Sequence[ToolUsageRow],
    coverage_rows: Sequence[ToolUsageProviderCoverageRow],
    query: ToolUsageInsightQuery,
    materialized_at: str,
) -> ToolUsageInsight:
    """Assemble a ``ToolUsageInsight`` from raw substrate rows.

    Pure function — extracted so tests can exercise filtering, MCP-server
    derivation, and coverage-flag derivation without standing up a
    database. Query filters apply only to ``entries``; ``provider_coverage``
    stays exhaustive so coverage gaps are never hidden when a reader
    narrows the result set.
    """

    entries = [_tool_usage_entry(row) for row in rows]
    if query.provider:
        entries = [entry for entry in entries if entry.source_name == query.provider]
    if query.tool:
        entries = [entry for entry in entries if entry.normalized_tool_name == query.tool]
    if query.mcp_server:
        entries = [entry for entry in entries if entry.mcp_server == query.mcp_server]
    if query.action_kind:
        entries = [entry for entry in entries if entry.action_kind == query.action_kind]
    entries.sort(key=lambda entry: (-entry.call_count, entry.source_name, entry.normalized_tool_name))
    if query.offset:
        entries = entries[query.offset :]
    if query.limit is not None:
        entries = entries[: query.limit]

    coverage = tuple(_tool_usage_coverage(row) for row in coverage_rows)
    providers_with_data = sum(1 for entry in coverage if entry.data_available)
    providers_without_data = sum(1 for entry in coverage if not entry.data_available and entry.session_count > 0)
    total_calls = sum(entry.call_count for entry in entries)
    distinct_tools = len({(entry.source_name, entry.normalized_tool_name) for entry in entries})
    return ToolUsageInsight(
        entries=tuple(entries),
        provider_coverage=coverage,
        total_call_count=total_calls,
        total_distinct_tools=distinct_tools,
        providers_with_data=providers_with_data,
        providers_without_data=providers_without_data,
        has_coverage_gaps=providers_without_data > 0,
        provenance=ArchiveInsightProvenance(
            materializer_version=TOOL_USAGE_INSIGHT_VERSION,
            materialized_at=materialized_at,
            source_updated_at=None,
            source_sort_key=None,
        ),
    )


__all__ = [
    "TOOL_USAGE_INSIGHT_VERSION",
    "ToolUsageCoverageEntry",
    "ToolUsageEntry",
    "ToolUsageInsight",
    "ToolUsageInsightQuery",
    "build_tool_usage_insight",
    "extract_mcp_server",
]
