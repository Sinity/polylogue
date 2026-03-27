"""Canonical semantic action events derived from tool calls.

Action events sit above raw ToolCall viewports. They preserve the original
tool-call evidence while normalizing the fields that downstream consumers
actually care about: action kind, paths, cwd, branch names, commands,
search/web targets, and message-scoped event identity.
"""

from __future__ import annotations

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
from polylogue.lib.action_event_models import ActionEvent
from polylogue.lib.action_event_parsing import build_tool_calls_from_content_blocks
from polylogue.lib.models import Message
from polylogue.lib.viewports import ToolCall


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
