"""Canonical semantic actions derived from tool calls.

Actions sit above raw ToolCall viewports. They preserve the original
tool-call evidence while normalizing the fields that downstream consumers
actually care about: action kind, paths, cwd, branch names, commands,
search/web targets, and message-scoped action identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from polylogue.archive.actions.fields import (
    build_search_text,
    extract_branch_names,
    extract_command,
    extract_cwd_path,
    extract_search_query,
    extract_url,
    make_action_id,
    normalize_output_text,
)
from polylogue.archive.actions.parsing import build_tool_calls_from_content_blocks
from polylogue.archive.viewport.viewports import ToolCall, ToolCategory
from polylogue.core.enums import Origin

_CANONICAL_TOOL_NAMES = {
    "bash": "bash",
    "shell": "bash",
    "terminal": "bash",
    "run": "bash",
    "exec": "bash",
    "exec_command": "bash",
    "functions.exec": "bash",
    "functions.exec_command": "bash",
    "shell_command": "bash",
    "spawn_agent": "task",
    "subagent": "task",
    "agent": "task",
    "update_plan": "todo",
}

_PATH_EXTRACTING_CATEGORIES = frozenset(
    {
        ToolCategory.FILE_READ,
        ToolCategory.FILE_WRITE,
        ToolCategory.FILE_EDIT,
        ToolCategory.SHELL,
        ToolCategory.GIT,
        ToolCategory.SEARCH,
        ToolCategory.WEB,
        ToolCategory.OTHER,
    }
)
_COMMAND_EXTRACTING_CATEGORIES = frozenset({ToolCategory.SHELL, ToolCategory.GIT, ToolCategory.OTHER})
_CWD_EXTRACTING_CATEGORIES = frozenset({ToolCategory.SHELL, ToolCategory.GIT, ToolCategory.OTHER})
_BRANCH_EXTRACTING_CATEGORIES = frozenset({ToolCategory.SHELL, ToolCategory.GIT, ToolCategory.OTHER})
_QUERY_EXTRACTING_CATEGORIES = frozenset({ToolCategory.SEARCH, ToolCategory.OTHER})
_URL_EXTRACTING_CATEGORIES = frozenset({ToolCategory.WEB, ToolCategory.OTHER})


def canonical_tool_name(name: str) -> str:
    """Return the cross-provider canonical tool name used for statistics."""
    normalized = (name or "unknown").strip().lower()
    return _CANONICAL_TOOL_NAMES.get(normalized, normalized)


@dataclass(frozen=True, slots=True)
class Action:
    """Normalized semantic action derived from a message tool call."""

    action_id: str
    message_id: str
    timestamp: datetime | None
    sequence_index: int
    kind: ToolCategory
    tool_name: str
    tool_id: str | None
    origin: Origin | None
    affected_paths: tuple[str, ...]
    cwd_path: str | None
    branch_names: tuple[str, ...]
    command: str | None
    query: str | None
    url: str | None
    output_text: str | None
    search_text: str
    raw: Mapping[str, object]
    #: Structural pass/fail from the paired tool_result's keystone columns
    #: (``tool_result_is_error``/``tool_result_exit_code``, see
    #: :func:`polylogue.archive.actions.parsing.tool_result_outcome`).
    #: ``True``/``False`` is a genuinely structural verdict; ``None`` means
    #: the origin never populated either column for this result -- not a
    #: negative claim, since most origins are structurally uncovered here
    #: (polylogue-9e5.3 audit). Consumers must not text-scan ``output_text``
    #: to recover a verdict this field reports as ``None``.
    tool_success: bool | None = None

    @property
    def normalized_tool_name(self) -> str:
        return canonical_tool_name(self.tool_name)


class ActionMessageLike(Protocol):
    @property
    def id(self) -> object: ...

    @property
    def timestamp(self) -> datetime | None: ...


def build_action(
    message: ActionMessageLike,
    call: ToolCall,
    *,
    sequence_index: int,
) -> Action:
    return _build_action_from_message_fields(
        message_id=str(message.id),
        timestamp=message.timestamp,
        call=call,
        sequence_index=sequence_index,
    )


def _build_action_from_message_fields(
    *,
    message_id: str,
    timestamp: datetime | None,
    call: ToolCall,
    sequence_index: int,
) -> Action:
    category = call.category
    command = extract_command(call) if category in _COMMAND_EXTRACTING_CATEGORIES else None
    affected_paths = tuple(call.affected_paths) if category in _PATH_EXTRACTING_CATEGORIES else ()
    cwd_path = extract_cwd_path(call) if category in _CWD_EXTRACTING_CATEGORIES else None
    branch_names = extract_branch_names(call, command) if category in _BRANCH_EXTRACTING_CATEGORIES else ()
    query = extract_search_query(call) if category in _QUERY_EXTRACTING_CATEGORIES else None
    url = extract_url(call) if category in _URL_EXTRACTING_CATEGORIES else None
    output_text = normalize_output_text(call)
    return Action(
        action_id=make_action_id(message_id, sequence_index, call.id, call.name),
        message_id=message_id,
        timestamp=timestamp,
        sequence_index=sequence_index,
        kind=category,
        tool_name=call.name,
        tool_id=call.id,
        origin=call.origin,
        affected_paths=affected_paths,
        cwd_path=cwd_path,
        branch_names=branch_names,
        command=command,
        query=query,
        url=url,
        output_text=output_text,
        search_text=build_search_text(
            kind=category,
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
        tool_success=call.success,
    )


def build_actions(
    message: ActionMessageLike,
    calls: tuple[ToolCall, ...],
) -> tuple[Action, ...]:
    if not calls:
        return ()
    message_id = str(message.id)
    timestamp = message.timestamp
    return tuple(
        _build_action_from_message_fields(
            message_id=message_id,
            timestamp=timestamp,
            call=call,
            sequence_index=index,
        )
        for index, call in enumerate(calls)
    )


__all__ = [
    "Action",
    "build_action",
    "build_actions",
    "build_tool_calls_from_content_blocks",
    "canonical_tool_name",
]
