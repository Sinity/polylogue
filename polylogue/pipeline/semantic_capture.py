"""Raw-message semantic capture helpers retained for parser/test coverage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NotRequired

from typing_extensions import TypedDict

from polylogue.lib.json import JSONDocument, JSONValue, is_json_value, json_document, json_document_list
from polylogue.lib.payload_coercion import mapping_or_empty, optional_string
from polylogue.pipeline.semantic_metadata import (
    ToolInputPayload,
    ToolMetadata,
    _parse_git_command,
    _parse_subagent_spawn,
)


class ContextCompactionSummary(TypedDict):
    summary: str
    timestamp: JSONValue | None
    trigger: JSONValue | None
    pre_tokens: JSONValue | None
    preserved_segment_id: str | None
    is_modern: bool


class ThinkingTraceSummary(TypedDict):
    text: str
    token_count: int


class ToolInvocationSummary(TypedDict, total=False):
    tool_name: str | None
    tool_id: str | None
    input: JSONDocument
    is_file_operation: bool
    is_search_operation: bool
    is_subagent: bool
    is_git_operation: bool


class FileChangeSummary(TypedDict):
    path: str
    operation: str
    old_content: NotRequired[str]
    new_content: NotRequired[str]


class SubagentSpawnSummary(TypedDict):
    agent_type: str
    prompt: str
    description: str | None
    run_in_background: bool


def _json_value(value: object) -> JSONValue | None:
    return value if is_json_value(value) else None


def _tool_input_payload(value: object) -> ToolInputPayload:
    return json_document(value)


def _text_from_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    for block in json_document_list(content):
        if block.get("type") == "text":
            text = optional_string(block.get("text"))
            if text:
                return text
    return ""


def _summary_text(item: object) -> str:
    message = mapping_or_empty(mapping_or_empty(item).get("message"))
    return _text_from_message_content(message.get("content"))


def detect_context_compaction(item: Mapping[str, object]) -> ContextCompactionSummary | None:
    """Detect if a raw message item represents a context compaction event."""
    msg_type = item.get("type")

    if msg_type == "summary":
        return {
            "summary": _summary_text(item),
            "timestamp": _json_value(item.get("timestamp")),
            "trigger": None,
            "pre_tokens": None,
            "preserved_segment_id": None,
            "is_modern": False,
        }

    if msg_type == "system" and item.get("subtype") == "compact_boundary":
        meta = mapping_or_empty(item.get("compact_metadata"))
        preserved = mapping_or_empty(meta.get("preserved_segment"))
        return {
            "summary": _summary_text(item),
            "timestamp": _json_value(item.get("timestamp")),
            "trigger": _json_value(meta.get("trigger")),
            "pre_tokens": _json_value(meta.get("preTokens") or meta.get("pre_tokens")),
            "preserved_segment_id": optional_string(preserved.get("anchor_uuid")),
            "is_modern": True,
        }

    return None


def extract_thinking_traces(content_blocks: Sequence[Mapping[str, object]]) -> list[ThinkingTraceSummary]:
    traces: list[ThinkingTraceSummary] = []
    for block in (json_document(item) for item in content_blocks):
        if block.get("type") != "thinking":
            continue
        text = optional_string(block.get("thinking")) or optional_string(block.get("text")) or ""
        if text:
            traces.append({"text": text, "token_count": len(text.split())})
    return traces


def extract_tool_invocations(content_blocks: Sequence[Mapping[str, object]]) -> list[ToolInvocationSummary]:
    invocations: list[ToolInvocationSummary] = []
    for block in (json_document(item) for item in content_blocks):
        if block.get("type") != "tool_use":
            continue
        input_payload = json_document(block.get("input"))
        invocation: ToolInvocationSummary = {
            "tool_name": optional_string(block.get("name")),
            "tool_id": optional_string(block.get("id")),
            "input": input_payload,
        }
        tool_name = invocation.get("tool_name")
        if tool_name:
            invocation["is_file_operation"] = tool_name in {"Read", "Write", "Edit", "NotebookEdit"}
            invocation["is_search_operation"] = tool_name in {"Glob", "Grep", "WebSearch"}
            invocation["is_subagent"] = tool_name == "Task"
            if tool_name == "Bash":
                command = optional_string(input_payload.get("command")) or ""
                invocation["is_git_operation"] = command.strip().startswith("git ")
        invocations.append(invocation)
    return invocations


def parse_git_operation(tool_invocation: Mapping[str, object]) -> ToolMetadata | None:
    if tool_invocation.get("tool_name") != "Bash":
        return None
    command = optional_string(json_document(tool_invocation.get("input")).get("command"))
    if not command or not command.strip().startswith("git "):
        return None
    return _parse_git_command(command)


def extract_file_changes(tool_invocations: Sequence[Mapping[str, object]]) -> list[FileChangeSummary]:
    changes: list[FileChangeSummary] = []
    for invocation in tool_invocations:
        tool_name = invocation.get("tool_name")
        input_data = json_document(invocation.get("input"))
        path = optional_string(input_data.get("file_path")) or optional_string(input_data.get("path"))
        if not path:
            continue

        if tool_name == "Read":
            changes.append({"path": path, "operation": "read"})
        elif tool_name == "Write":
            content = optional_string(input_data.get("content"))
            entry: FileChangeSummary = {"path": path, "operation": "write"}
            if content is not None:
                entry["new_content"] = content[:500]
            changes.append(entry)
        elif tool_name == "Edit":
            entry = {"path": path, "operation": "edit"}
            old_string = optional_string(input_data.get("old_string"))
            new_string = optional_string(input_data.get("new_string"))
            if old_string is not None:
                entry["old_content"] = old_string[:200]
            if new_string is not None:
                entry["new_content"] = new_string[:200]
            changes.append(entry)
        elif tool_name == "NotebookEdit":
            new_source = optional_string(input_data.get("new_source"))
            entry: FileChangeSummary = {"path": path, "operation": "edit"}
            if new_source is not None:
                entry["new_content"] = new_source[:500]
            changes.append(entry)
    return changes


def extract_subagent_spawns(tool_invocations: Sequence[Mapping[str, object]]) -> list[SubagentSpawnSummary]:
    spawns: list[SubagentSpawnSummary] = []
    for invocation in tool_invocations:
        if invocation.get("tool_name") != "Task":
            continue
        input_data = json_document(invocation.get("input"))
        parsed = _parse_subagent_spawn(_tool_input_payload(input_data))
        agent_type = parsed.get("agent_type")
        spawns.append(
            {
                "agent_type": agent_type if isinstance(agent_type, str) else "general-purpose",
                "prompt": optional_string(input_data.get("prompt")) or "",
                "description": optional_string(parsed.get("description")),
                "run_in_background": bool(parsed.get("run_in_background", False)),
            }
        )
    return spawns


def extract_git_operations(tool_invocations: Sequence[Mapping[str, object]]) -> list[ToolMetadata]:
    operations: list[ToolMetadata] = []
    for invocation in tool_invocations:
        if git_op := parse_git_operation(invocation):
            operations.append(git_op)
    return operations


__all__ = [
    "detect_context_compaction",
    "extract_file_changes",
    "extract_git_operations",
    "extract_subagent_spawns",
    "extract_thinking_traces",
    "extract_tool_invocations",
    "parse_git_operation",
]
