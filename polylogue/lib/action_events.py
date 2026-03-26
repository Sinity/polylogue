"""Canonical semantic action events derived from tool calls.

Action events sit above raw ToolCall viewports. They preserve the original
tool-call evidence while normalizing the fields that downstream consumers
actually care about: action kind, paths, cwd, branch names, commands,
search/web targets, and message-scoped event identity.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from polylogue.lib.hashing import hash_text_short
from polylogue.lib.models import Message
from polylogue.lib.viewports import ToolCall, ToolCategory, classify_tool
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


def _normalized_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def build_tool_calls_from_content_blocks(
    *,
    provider: Provider | str | None,
    content_blocks: Sequence[Mapping[str, Any]],
) -> tuple[ToolCall, ...]:
    """Normalize canonical ToolCall viewports from content blocks.

    This is the shared bridge for semantic facts and any future persisted
    action indexing: one tool-use block becomes one ToolCall with the best
    available paired tool_result output from the same message block stream.
    """

    normalized_provider = Provider.from_string(provider) if provider is not None else None
    tool_result_outputs: dict[str, str] = {}
    for block in content_blocks:
        if str(block.get("type")) != "tool_result":
            continue
        tool_id = block.get("tool_id")
        text = block.get("text")
        if isinstance(tool_id, str) and tool_id and isinstance(text, str) and text:
            tool_result_outputs.setdefault(tool_id, text)

    calls: list[ToolCall] = []
    for block in content_blocks:
        if str(block.get("type")) != "tool_use":
            continue
        name = block.get("tool_name")
        if not isinstance(name, str) or not name:
            continue
        tool_id = block.get("tool_id")
        normalized_input = _normalized_mapping(block.get("tool_input"))
        semantic_category = _tool_category_from_semantic(block.get("semantic_type"))
        classified_category = classify_tool(name, normalized_input)
        category = classified_category if semantic_category in (None, ToolCategory.OTHER) else semantic_category
        raw = {
            "block_id": block.get("block_id"),
            "block_index": block.get("block_index"),
            "message_id": block.get("message_id"),
            "conversation_id": block.get("conversation_id"),
            "type": block.get("type"),
            "tool_name": name,
            "tool_id": tool_id,
            "tool_input": normalized_input,
            "media_type": block.get("media_type"),
            "metadata": _normalized_mapping(block.get("metadata")),
            "semantic_type": block.get("semantic_type"),
            "text": block.get("text"),
        }
        calls.append(
            ToolCall(
                name=name,
                id=tool_id if isinstance(tool_id, str) and tool_id else None,
                input=normalized_input,
                output=tool_result_outputs.get(tool_id) if isinstance(tool_id, str) else None,
                category=category,
                provider=normalized_provider,
                raw=raw,
            )
        )
    return tuple(calls)


def _tool_category_from_semantic(value: object) -> ToolCategory | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return ToolCategory(value)
    except ValueError:
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


def _normalize_output_text(call: ToolCall) -> str | None:
    output = _clean_str(call.output)
    if output is None:
        return None
    return " ".join(output.split())[:240]


def _build_search_text(
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


def _make_action_event_id(message_id: str, sequence_index: int, tool_id: str | None, tool_name: str) -> str:
    seed = f"{message_id}:{sequence_index}:{tool_id or ''}:{tool_name}"
    return f"act-{hash_text_short(seed)}"


@dataclass(frozen=True, slots=True)
class ActionEvent:
    """Normalized semantic event derived from a message tool call."""

    event_id: str
    message_id: str
    timestamp: datetime | None
    sequence_index: int
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
    output_text: str | None
    search_text: str
    raw: dict[str, Any]

    @property
    def normalized_tool_name(self) -> str:
        return (self.tool_name or "unknown").strip().lower()


def build_action_event(message: Message, call: ToolCall, *, sequence_index: int) -> ActionEvent:
    command = _extract_command(call)
    affected_paths = tuple(call.affected_paths)
    cwd_path = _extract_cwd_path(call)
    branch_names = _extract_branch_names(call, command)
    query = _extract_search_query(call)
    url = _extract_url(call)
    output_text = _normalize_output_text(call)
    return ActionEvent(
        event_id=_make_action_event_id(str(message.id), sequence_index, call.id, call.name),
        message_id=str(message.id),
        timestamp=message.timestamp,
        sequence_index=sequence_index,
        kind=call.category,
        tool_name=call.name,
        tool_id=call.id,
        provider=call.provider,
        affected_paths=affected_paths,
        cwd_path=cwd_path,
        branch_names=branch_names,
        command=command,
        query=query,
        url=url,
        output_text=output_text,
        search_text=_build_search_text(
            kind=call.category,
            tool_name=call.name,
            affected_paths=affected_paths,
            cwd_path=cwd_path,
            branch_names=branch_names,
            command=command,
            query=query,
            url=url,
            output_text=output_text,
        ),
        raw=call.raw,
    )


def build_action_events(message: Message, calls: tuple[ToolCall, ...]) -> tuple[ActionEvent, ...]:
    return tuple(
        build_action_event(message, call, sequence_index=index)
        for index, call in enumerate(calls)
    )


__all__ = [
    "ActionEvent",
    "build_action_event",
    "build_action_events",
    "build_tool_calls_from_content_blocks",
]
