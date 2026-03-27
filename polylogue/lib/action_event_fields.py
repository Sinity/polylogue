"""Field extraction helpers for action-event normalization."""

from __future__ import annotations

import re

from polylogue.lib.action_event_parsing import _clean_str, _extract_first_string
from polylogue.lib.hashing import hash_text_short
from polylogue.lib.viewports import ToolCall, ToolCategory

_GIT_BRANCH_PATTERN = re.compile(r"git\s+(?:checkout|switch)\s+(?:-[bc]\s+)?(\S+)")
_QUERY_FIELDS = ("q", "query", "pattern", "search_query", "searchQuery", "needle", "term")
_URL_FIELDS = ("url", "uri", "href", "ref_id")
_CWD_FIELDS = ("cwd", "workdir", "directory")
_BRANCH_FIELDS = ("branch", "from_branch", "base", "base_ref", "head")


def extract_search_query(call: ToolCall) -> str | None:
    query = _extract_first_string(call.input, _QUERY_FIELDS)
    if query is not None:
        return query
    for key in ("queries", "patterns"):
        value = call.input.get(key)
        if isinstance(value, list):
            normalized = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            if normalized:
                return ", ".join(normalized)
    return None


def extract_command(call: ToolCall) -> str | None:
    return _extract_first_string(call.input, ("command", "cmd"))


def extract_url(call: ToolCall) -> str | None:
    for field in _URL_FIELDS:
        value = _clean_str(call.input.get(field))
        if value and (value.startswith("http://") or value.startswith("https://")):
            return value
    return None


def extract_cwd_path(call: ToolCall) -> str | None:
    value = _extract_first_string(call.input, _CWD_FIELDS)
    return value if value and value.startswith("/") else None


def extract_branch_names(call: ToolCall, command: str | None) -> tuple[str, ...]:
    branches: list[str] = []
    for field in _BRANCH_FIELDS:
        value = _clean_str(call.input.get(field))
        if value and not value.startswith("-") and "/" not in value[:1]:
            branches.append(value)
    if command:
        for match in _GIT_BRANCH_PATTERN.finditer(command):
            branch = match.group(1)
            if branch and not branch.startswith("-"):
                branches.append(branch)
    return tuple(dict.fromkeys(branches))


def normalize_output_text(call: ToolCall) -> str | None:
    output = _clean_str(call.output)
    if output is None:
        return None
    return " ".join(output.split())[:240]


def build_search_text(
    *,
    kind: ToolCategory,
    tool_name: str,
    affected_paths: tuple[str, ...],
    cwd_path: str | None,
    branch_names: tuple[str, ...],
    command: str | None,
    query: str | None,
    url: str | None,
    output_text: str | None,
) -> str:
    parts: list[str] = [kind.value, tool_name]
    parts.extend(affected_paths)
    if cwd_path:
        parts.append(cwd_path)
    parts.extend(branch_names)
    if command:
        parts.append(command)
    if query:
        parts.append(query)
    if url:
        parts.append(url)
    if output_text:
        parts.append(output_text)
    return " | ".join(part for part in parts if part)


def make_action_event_id(message_id: str, sequence_index: int, tool_id: str | None, tool_name: str) -> str:
    seed = f"{message_id}:{sequence_index}:{tool_id or ''}:{tool_name}"
    return f"act-{hash_text_short(seed)}"


__all__ = [
    "build_search_text",
    "extract_branch_names",
    "extract_command",
    "extract_cwd_path",
    "extract_search_query",
    "extract_url",
    "make_action_event_id",
    "normalize_output_text",
]
