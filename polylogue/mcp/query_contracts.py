"""Shared MCP-side query normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.lib.query_spec import ConversationQuerySpec

_QUERY_PARAM_ALIASES = {
    "has_tool_use": "filter_has_tool_use",
    "has_thinking": "filter_has_thinking",
}


def normalize_query_params(params: Mapping[str, object]) -> dict[str, object]:
    """Normalize MCP query kwargs into substrate query-spec parameter names."""
    normalized = dict(params)
    for source_key, target_key in _QUERY_PARAM_ALIASES.items():
        if source_key in normalized:
            normalized[target_key] = normalized.pop(source_key)
    return normalized


def build_query_spec(**params: object) -> ConversationQuerySpec:
    """Build a ConversationQuerySpec from MCP-facing query kwargs."""
    return ConversationQuerySpec.from_params(normalize_query_params(params))


__all__ = ["build_query_spec", "normalize_query_params"]
