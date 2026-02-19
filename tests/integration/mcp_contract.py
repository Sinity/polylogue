"""Shared MCP server surface contract for integration tests."""

from __future__ import annotations

EXPECTED_TOOL_NAMES = {
    "search",
    "list_conversations",
    "get_conversation",
    "stats",
    "add_tag",
    "remove_tag",
    "list_tags",
    "get_metadata",
    "set_metadata",
    "delete_metadata",
    "delete_conversation",
    "get_conversation_summary",
    "get_session_tree",
    "get_stats_by",
    "health_check",
    "rebuild_index",
    "update_index",
    "export_conversation",
}

EXPECTED_RESOURCE_URIS = {
    "polylogue://stats",
    "polylogue://conversations",
    "polylogue://tags",
    "polylogue://health",
}

EXPECTED_RESOURCE_TEMPLATE_URIS = {
    "polylogue://conversation/{conv_id}",
}

EXPECTED_PROMPT_NAMES = {
    "analyze_errors",
    "summarize_week",
    "extract_code",
    "compare_conversations",
    "extract_patterns",
}

