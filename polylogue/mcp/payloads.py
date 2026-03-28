"""Typed MCP payload models shared by server tools and resources."""

from __future__ import annotations

from polylogue.mcp.payload_archive import (
    MCPArchiveStatsPayload,
    MCPMetadataPayload,
    MCPMutationStatusPayload,
    MCPStatsByPayload,
    MCPTagCountsPayload,
)
from polylogue.mcp.payload_base import MCPPayload, MCPRootPayload
from polylogue.mcp.payload_conversations import (
    MCPConversationDetailPayload,
    MCPConversationSummaryListPayload,
    MCPConversationSummaryPayload,
    MCPErrorPayload,
    MCPMessagePayload,
)
from polylogue.mcp.payload_health import MCPHealthCheckPayload, MCPHealthReportPayload

__all__ = [
    "MCPArchiveStatsPayload",
    "MCPConversationDetailPayload",
    "MCPConversationSummaryListPayload",
    "MCPConversationSummaryPayload",
    "MCPErrorPayload",
    "MCPHealthCheckPayload",
    "MCPHealthReportPayload",
    "MCPMessagePayload",
    "MCPMetadataPayload",
    "MCPRootPayload",
    "MCPPayload",
    "MCPMutationStatusPayload",
    "MCPStatsByPayload",
    "MCPTagCountsPayload",
]
