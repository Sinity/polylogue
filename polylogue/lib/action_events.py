"""Canonical semantic action events derived from tool calls.

Action events sit above raw ToolCall viewports. They preserve the original
tool-call evidence while normalizing the fields that downstream consumers
actually care about: action kind, paths, cwd, branch names, commands,
search/web targets, and message-scoped event identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from polylogue.lib.action_event_fields import (
    build_search_text,
    extract_branch_names,
    extract_command,
    extract_cwd_path,
    extract_search_query,
    extract_url,
    make_action_event_id,
    normalize_output_text,
)
from polylogue.lib.action_event_parsing import build_tool_calls_from_content_blocks
from polylogue.lib.models import Message
from polylogue.lib.viewports import ToolCall, ToolCategory
from polylogue.types import Provider


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
    command = extract_command(call)
    affected_paths = tuple(call.affected_paths)
    cwd_path = extract_cwd_path(call)
    branch_names = extract_branch_names(call, command)
    query = extract_search_query(call)
    url = extract_url(call)
    output_text = normalize_output_text(call)
    return ActionEvent(
        event_id=make_action_event_id(str(message.id), sequence_index, call.id, call.name),
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
        search_text=build_search_text(
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
