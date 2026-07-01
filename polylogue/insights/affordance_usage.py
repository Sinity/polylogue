"""Shared affordance/tool usage classification helpers.

These helpers classify archive ``actions`` rows into operator-facing tool
families and evidence kinds. They intentionally keep raw tool names available
to callers; the normalized names are a read projection for analysis, not a
replacement for source evidence.
"""

from __future__ import annotations

from collections.abc import Mapping

DEFAULT_FAMILY_PATTERNS: tuple[str, ...] = (
    "serena",
    "codebase",
    "cclsp",
    "context7",
    "polylogue",
    "lynchpin",
)

FAMILY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("serena", ("serena",)),
    ("codebase-memory", ("codebase-memory", "codebase_memory", "codebasememory", "search_code")),
    ("cclsp", ("cclsp",)),
    ("context7", ("context7",)),
    ("polylogue", ("polylogue",)),
    ("lynchpin", ("lynchpin",)),
)

GENERIC_DETAIL_TOOL_FAMILIES: frozenset[str] = frozenset(
    {"exec_command", "functions", "functions.exec_command", "bash", "shell", "client"}
)


def clean_patterns(patterns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(pattern.strip().lower() for pattern in patterns if pattern.strip())


def like_param(pattern: str) -> str:
    escaped = pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def row_text(row: Mapping[str, object], key: str) -> str:
    return str(row.get(key) or "").lower()


def row_detail_text(row: Mapping[str, object]) -> str:
    return row_text(row, "match_detail") or row_text(row, "detail")


def matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return bool(patterns) and any(pattern in value for pattern in patterns)


def family_for_text(text: object) -> str | None:
    normalized = str(text or "").lower()
    for family, needles in FAMILY_ALIASES:
        if any(needle in normalized for needle in needles):
            return family
    return None


def family_for_tool(tool_name: object) -> str:
    normalized = str(tool_name or "").lower()
    if family := family_for_text(normalized):
        return family
    if normalized.startswith("mcp__"):
        parts = normalized.split("__")
        if len(parts) > 1 and parts[1]:
            return parts[1].replace("plugin_", "")
    return normalized or "unknown"


def family_for_row(row: Mapping[str, object]) -> str:
    tool_family = family_for_tool(row.get("tool_name"))
    if tool_family in GENERIC_DETAIL_TOOL_FAMILIES:
        return family_for_text(row_detail_text(row)) or tool_family
    return tool_family


def normalized_tool_name_for_row(row: Mapping[str, object]) -> str:
    raw = str(row.get("tool_name") or "").lower()
    family = family_for_row(row)
    if raw.startswith("mcp__"):
        parts = [part for part in raw.split("__") if part and part != "mcp"]
        if not parts:
            return "mcp/unknown"
        namespace = parts[0].removeprefix("plugin_")
        tool_parts = parts[1:]
        if namespace.startswith(f"{family}_"):
            namespace = family
        if tool_parts and tool_parts[0] == family:
            tool_parts = tool_parts[1:]
        tool = "__".join(tool_parts) if tool_parts else namespace
        return f"{family}/{tool}"
    if family in GENERIC_DETAIL_TOOL_FAMILIES:
        return raw or "unknown"
    if family_for_text(row_detail_text(row)) == family and raw in GENERIC_DETAIL_TOOL_FAMILIES:
        return f"{family}/command-detail"
    if family != raw and family_for_text(raw) == family:
        return f"{family}/{raw}"
    return raw or "unknown"


def evidence_kind_for_row(row: Mapping[str, object]) -> str:
    tool_name = row_text(row, "tool_name")
    tool_family = family_for_tool(tool_name)
    if tool_name.startswith("mcp__"):
        return "mcp_tool_call"
    if tool_family in GENERIC_DETAIL_TOOL_FAMILIES:
        return "command_detail"
    if tool_family in {"apply_patch", "edit"}:
        return "edit_content"
    if tool_family == "agent":
        return "subagent"
    if tool_family == "update_plan":
        return "agent_planning"
    if tool_family == "web_search_call":
        return "web_search"
    return "tool_call"


def matched_by_row(
    row: Mapping[str, object],
    *,
    tool_patterns: tuple[str, ...],
    detail_patterns: tuple[str, ...],
) -> str:
    tool_match = matches_any(row_text(row, "tool_name"), tool_patterns)
    detail_match = matches_any(row_detail_text(row), detail_patterns)
    if tool_match and detail_match:
        return "tool_name+detail"
    if detail_match:
        return "detail"
    if tool_match:
        return "tool_name"
    return "implicit-default"


__all__ = [
    "DEFAULT_FAMILY_PATTERNS",
    "FAMILY_ALIASES",
    "GENERIC_DETAIL_TOOL_FAMILIES",
    "clean_patterns",
    "evidence_kind_for_row",
    "family_for_row",
    "family_for_text",
    "family_for_tool",
    "like_param",
    "matched_by_row",
    "matches_any",
    "normalized_tool_name_for_row",
    "row_detail_text",
    "row_text",
]
