"""Structured semantic metadata extraction for parsed tool blocks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from polylogue.lib.viewports import ToolCategory, classify_tool

ToolInputScalar: TypeAlias = str | int | float | bool | None
ToolInputPayload: TypeAlias = Mapping[str, Any]
ToolMetadata: TypeAlias = dict[str, Any]


def extract_tool_metadata(tool_name: str, tool_input: ToolInputPayload) -> ToolMetadata | None:
    """Return structured metadata for a parsed tool-use block."""
    category = classify_tool(tool_name, tool_input)
    if category == ToolCategory.GIT:
        command = _payload_string(tool_input, "command")
        return _parse_git_command(command) if command is not None else None
    if category == ToolCategory.SUBAGENT:
        return _parse_subagent_spawn(tool_input)
    if category == ToolCategory.AGENT:
        return _extract_agent_metadata(tool_name, tool_input)
    if category == ToolCategory.SEARCH:
        return _extract_search_metadata(tool_input)
    if category in (ToolCategory.FILE_READ, ToolCategory.FILE_WRITE, ToolCategory.FILE_EDIT):
        return _extract_file_paths(tool_name, tool_input)
    return None


def _payload_string(tool_input: ToolInputPayload, key: str) -> str | None:
    value = tool_input.get(key)
    return value if isinstance(value, str) and value else None


def _payload_list(tool_input: ToolInputPayload, key: str) -> list[Any] | None:
    value = tool_input.get(key)
    return value if isinstance(value, list) else None


def _quoted_flag_value(command: str, flag: str) -> str | None:
    marker = f"{flag} "
    start = command.find(marker)
    if start < 0:
        return None
    remaining = command[start + len(marker) :].lstrip()
    if not remaining:
        return None
    quote = remaining[0]
    if quote not in {'"', "'"}:
        return None
    end_quote = remaining.find(quote, 1)
    if end_quote <= 0:
        return None
    return remaining[1:end_quote]


def _parse_git_command(command: str) -> ToolMetadata:
    parts = command.strip().split()
    if len(parts) < 2:
        return {"full_command": command}

    git_command = parts[1]
    metadata: ToolMetadata = {
        "command": git_command,
        "full_command": command,
    }

    if git_command == "commit":
        message = _quoted_flag_value(command, "-m")
        if message is not None:
            metadata["message"] = message
        return metadata

    if git_command in {"checkout", "switch"}:
        for part in parts[2:]:
            if not part.startswith("-"):
                metadata["branch"] = part
                break
        return metadata

    if git_command == "push":
        non_flags = [part for part in parts[2:] if not part.startswith("-")]
        if len(non_flags) >= 2:
            metadata["remote"] = non_flags[0]
            metadata["branch"] = non_flags[1]
        elif len(non_flags) == 1:
            metadata["remote"] = non_flags[0]
        return metadata

    if git_command in {"add", "rm"}:
        metadata["files"] = [part for part in parts[2:] if not part.startswith("-")]
    return metadata


def _parse_subagent_spawn(tool_input: ToolInputPayload) -> ToolMetadata:
    prompt = _payload_string(tool_input, "prompt")
    metadata: ToolMetadata = {
        "agent_type": _payload_string(tool_input, "subagent_type") or "general-purpose",
        "run_in_background": bool(tool_input.get("run_in_background", False)),
    }
    description = _payload_string(tool_input, "description")
    if description is not None:
        metadata["description"] = description
    if prompt is not None:
        metadata["prompt_snippet"] = prompt[:200]
    return metadata


def _extract_file_paths(tool_name: str, tool_input: ToolInputPayload) -> ToolMetadata | None:
    path = _payload_string(tool_input, "file_path") or _payload_string(tool_input, "path")
    if path is None:
        return None

    metadata: ToolMetadata = {"path": path}
    if tool_name == "Write":
        content = _payload_string(tool_input, "content")
        if content is not None:
            metadata["new_content_snippet"] = content[:500]
    elif tool_name == "Edit":
        old_string = _payload_string(tool_input, "old_string")
        new_string = _payload_string(tool_input, "new_string")
        if old_string is not None:
            metadata["old_snippet"] = old_string[:200]
        if new_string is not None:
            metadata["new_snippet"] = new_string[:200]
    return metadata


def _extract_search_metadata(tool_input: ToolInputPayload) -> ToolMetadata | None:
    metadata: ToolMetadata = {}
    path = _payload_string(tool_input, "path")
    if path is not None:
        metadata["path"] = path
    pattern = _payload_string(tool_input, "pattern")
    if pattern is not None:
        metadata["pattern"] = pattern
    query = _payload_string(tool_input, "query")
    if query is not None:
        metadata["query"] = query[:500]
    return metadata or None


def _extract_agent_metadata(tool_name: str, tool_input: ToolInputPayload) -> ToolMetadata:
    metadata: ToolMetadata = {"tool": tool_name}
    for key, field_name in (
        ("description", "description"),
        ("prompt", "prompt_snippet"),
        ("plan", "plan_snippet"),
        ("subject", "subject"),
    ):
        value = _payload_string(tool_input, key)
        if value is not None:
            metadata[field_name] = value[:200]
    task_id = _payload_string(tool_input, "taskId") or _payload_string(tool_input, "task_id")
    if task_id is not None:
        metadata["task_id"] = task_id
    todos = _payload_list(tool_input, "todos")
    if todos is not None:
        metadata["todo_count"] = len(todos)
    questions = _payload_list(tool_input, "questions")
    if questions is not None:
        metadata["question_count"] = len(questions)
    return metadata


__all__ = [
    "ToolInputPayload",
    "ToolMetadata",
    "extract_tool_metadata",
]
