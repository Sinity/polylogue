"""MCP semantic proof declarations."""

from __future__ import annotations

from polylogue.rendering.semantic_surface_contract_helpers import (
    declared_loss_contract as _declared_loss,
)
from polylogue.rendering.semantic_surface_contract_helpers import (
    preserve_contract as _preserve,
)
from polylogue.rendering.semantic_surface_models import SemanticSurfaceSpec

MCP_SUMMARY_CONTRACTS = (
    _preserve("conversation_id", "mcp_summary_json_v1 must preserve the conversation identifier", "conversation_id", "conversation_id"),
    _preserve("provider_identity", "mcp_summary_json_v1 must preserve provider identity", "provider", "provider"),
    _preserve("title_metadata", "mcp_summary_json_v1 must preserve the display title", "title", "title"),
    _preserve("message_count", "mcp_summary_json_v1 must preserve summary message counts", "messages", "messages"),
    _preserve("created_at", "mcp_summary_json_v1 must preserve the created_at timestamp when present", "created_at", "created_at"),
    _preserve("updated_at", "mcp_summary_json_v1 must preserve the updated_at timestamp when present", "updated_at", "updated_at"),
    _declared_loss(
        "tag_values",
        "mcp_summary_json_v1 intentionally omits tags",
        "tags",
        input_transform="len",
    ),
    _declared_loss(
        "summary_text",
        "mcp_summary_json_v1 intentionally omits summary text",
        "summary",
        input_transform="presence_count",
    ),
)

MCP_DETAIL_CONTRACTS = (
    _preserve("conversation_id", "mcp_detail_json_v1 must preserve the conversation identifier", "conversation_id", "conversation_id"),
    _preserve("provider_identity", "mcp_detail_json_v1 must preserve provider identity", "provider", "provider"),
    _preserve("title_metadata", "mcp_detail_json_v1 must preserve the display title", "title", "title"),
    _preserve("created_at", "mcp_detail_json_v1 must preserve the created_at timestamp when present", "created_at", "created_at"),
    _preserve("updated_at", "mcp_detail_json_v1 must preserve the updated_at timestamp when present", "updated_at", "updated_at"),
    _preserve("message_entries", "mcp_detail_json_v1 must preserve every message entry", "messages", "messages"),
    _preserve("message_ids", "mcp_detail_json_v1 must preserve message identifiers", "message_ids", "message_ids"),
    _preserve("role_entries", "mcp_detail_json_v1 must preserve message role distribution", "role_counts", "role_counts"),
    _preserve("timestamp_values", "mcp_detail_json_v1 must preserve message timestamps", "timestamped_messages", "timestamped_messages"),
    _declared_loss(
        "attachment_semantics",
        "mcp_detail_json_v1 intentionally omits attachment payload semantics",
        "attachment_count",
    ),
    _declared_loss(
        "thinking_semantics",
        "mcp_detail_json_v1 preserves display text but not typed thinking markers",
        "thinking_messages",
    ),
    _declared_loss(
        "tool_semantics",
        "mcp_detail_json_v1 preserves display text but not typed tool markers",
        "tool_messages",
    ),
    _declared_loss(
        "branch_structure",
        "mcp_detail_json_v1 intentionally omits explicit branch topology",
        "branch_messages",
    ),
)

MCP_SEMANTIC_SURFACE_SPECS: tuple[SemanticSurfaceSpec, ...] = (
    SemanticSurfaceSpec("mcp_summary_json_v1", "mcp", aliases=("mcp_summary",), contracts=MCP_SUMMARY_CONTRACTS),
    SemanticSurfaceSpec("mcp_detail_json_v1", "mcp", aliases=("mcp_detail",), contracts=MCP_DETAIL_CONTRACTS),
)

__all__ = ["MCP_SEMANTIC_SURFACE_SPECS"]
