"""Shared query-spec construction for MCP tools and prompts."""

from __future__ import annotations

from typing import Any

from polylogue.lib.query_spec import ConversationQuerySpec


def build_query_spec(**params: Any) -> ConversationQuerySpec:
    normalized = dict(params)
    if "has_tool_use" in normalized:
        normalized["filter_has_tool_use"] = normalized.pop("has_tool_use")
    if "has_thinking" in normalized:
        normalized["filter_has_thinking"] = normalized.pop("has_thinking")
    return ConversationQuerySpec.from_params(normalized)


__all__ = ["build_query_spec"]
