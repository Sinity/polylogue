"""Canonical normalized action facts derived from tool calls.

Action facts sit above raw ToolCall viewports. They preserve the original
tool-call evidence while normalizing the fields that downstream consumers
actually care about: action kind, paths, cwd, branch names, commands, and
search/web targets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from polylogue.lib.viewports import ToolCall, ToolCategory
from polylogue.types import Provider

_GIT_BRANCH_PATTERN = re.compile(r"git\s+(?:checkout|switch)\s+(?:-[bc]\s+)?(\S+)")
_QUERY_FIELDS = ("q", "query", "pattern", "search_query", "searchQuery", "needle", "term")
_URL_FIELDS = ("url", "uri", "href", "ref_id")
_CWD_FIELDS = ("cwd", "workdir", "directory")
_BRANCH_FIELDS = ("branch", "from_branch", "base", "base_ref", "head")


def _clean_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate or None


def _extract_first_string(mapping: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = _clean_str(mapping.get(field))
        if value is not None:
            return value
    return None


def _extract_search_query(call: ToolCall) -> str | None:
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


def _extract_command(call: ToolCall) -> str | None:
    return _extract_first_string(call.input, ("command", "cmd"))


def _extract_url(call: ToolCall) -> str | None:
    for field in _URL_FIELDS:
        value = _clean_str(call.input.get(field))
        if value and (value.startswith("http://") or value.startswith("https://")):
            return value
    return None


def _extract_cwd_path(call: ToolCall) -> str | None:
    value = _extract_first_string(call.input, _CWD_FIELDS)
    return value if value and value.startswith("/") else None


def _extract_branch_names(call: ToolCall, command: str | None) -> tuple[str, ...]:
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


@dataclass(frozen=True, slots=True)
class ActionFact:
    """Normalized semantic action derived from a tool call."""

    kind: ToolCategory
    tool_name: str
    tool_id: str | None
    provider: Provider | None
    affected_paths: tuple[str, ...]
    cwd_path: str | None
    branch_names: tuple[str, ...]
    command: str | None
    query: str | None
    url: str | None
    raw: dict[str, Any]


def build_action_fact(call: ToolCall) -> ActionFact:
    command = _extract_command(call)
    return ActionFact(
        kind=call.category,
        tool_name=call.name,
        tool_id=call.id,
        provider=call.provider,
        affected_paths=tuple(call.affected_paths),
        cwd_path=_extract_cwd_path(call),
        branch_names=_extract_branch_names(call, command),
        command=command,
        query=_extract_search_query(call),
        url=_extract_url(call),
        raw=call.raw,
    )


def build_action_facts(calls: tuple[ToolCall, ...]) -> tuple[ActionFact, ...]:
    return tuple(build_action_fact(call) for call in calls)


__all__ = ["ActionFact", "build_action_fact", "build_action_facts"]
