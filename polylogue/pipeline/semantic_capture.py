"""Raw-message semantic capture helpers retained for parser/test coverage."""

from __future__ import annotations

from typing import Any

from polylogue.pipeline.semantic_metadata import _parse_git_command, _parse_subagent_spawn


def detect_context_compaction(item: dict[str, Any]) -> dict[str, Any] | None:
    """Detect if a raw message item represents a context compaction event.

    Recognises two record shapes:
    - **Legacy**: ``{"type": "summary", "message": {"content": "..."}, ...}``
    - **Modern**: ``{"type": "system", "subtype": "compact_boundary",
      "compact_metadata": {...}, ...}``

    Returns a normalised dict with ``summary``, ``timestamp``, ``trigger``,
    ``pre_tokens``, ``preserved_segment_id``, and ``is_modern``.
    """
    msg_type = item.get("type")

    # ------------------------------------------------------------------
    # Legacy format: type == "summary"
    # ------------------------------------------------------------------
    if msg_type == "summary":
        message = item.get("message", {})
        content = ""
        if isinstance(message, dict):
            content_raw = message.get("content")
            if isinstance(content_raw, str):
                content = content_raw
            elif isinstance(content_raw, list):
                for block in content_raw:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content = block.get("text", "")
                        break

        return {
            "summary": content,
            "timestamp": item.get("timestamp"),
            "trigger": None,
            "pre_tokens": None,
            "preserved_segment_id": None,
            "is_modern": False,
        }

    # ------------------------------------------------------------------
    # Modern format: type == "system", subtype == "compact_boundary"
    # ------------------------------------------------------------------
    if msg_type == "system" and item.get("subtype") == "compact_boundary":
        meta = item.get("compact_metadata") or {}
        # Extract summary text from message.content (same structure as legacy)
        message = item.get("message", {})
        content = ""
        if isinstance(message, dict):
            content_raw = message.get("content")
            if isinstance(content_raw, str):
                content = content_raw
            elif isinstance(content_raw, list):
                for block in content_raw:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content = block.get("text", "")
                        break

        # pre_tokens: accept both camelCase and snake_case
        pre_tokens = meta.get("preTokens") or meta.get("pre_tokens")

        # preserved_segment → anchor_uuid
        preserved = meta.get("preserved_segment") or {}
        anchor_uuid = preserved.get("anchor_uuid") if isinstance(preserved, dict) else None

        return {
            "summary": content,
            "timestamp": item.get("timestamp"),
            "trigger": meta.get("trigger"),
            "pre_tokens": pre_tokens,
            "preserved_segment_id": anchor_uuid,
            "is_modern": True,
        }

    return None


def extract_thinking_traces(content_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces = []
    for block in content_blocks:
        if block.get("type") == "thinking":
            text = block.get("thinking") or block.get("text") or ""
            if text:
                traces.append({
                    "text": text,
                    "token_count": len(text.split()),
                })
    return traces


def extract_tool_invocations(content_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    invocations = []
    for block in content_blocks:
        if block.get("type") == "tool_use":
            invocation: dict[str, Any] = {
                "tool_name": block.get("name"),
                "tool_id": block.get("id"),
                "input": block.get("input") or {},
            }
            tool_name = invocation["tool_name"]
            if tool_name:
                invocation["is_file_operation"] = tool_name in {"Read", "Write", "Edit", "NotebookEdit"}
                invocation["is_search_operation"] = tool_name in {"Glob", "Grep", "WebSearch"}
                invocation["is_subagent"] = tool_name == "Task"
                if tool_name == "Bash":
                    cmd = (invocation.get("input") or {}).get("command", "")
                    invocation["is_git_operation"] = isinstance(cmd, str) and cmd.strip().startswith("git ")
            invocations.append(invocation)
    return invocations


def parse_git_operation(tool_invocation: dict[str, Any]) -> dict[str, Any] | None:
    if tool_invocation.get("tool_name") != "Bash":
        return None
    command = tool_invocation.get("input", {}).get("command", "")
    if not isinstance(command, str) or not command.strip().startswith("git "):
        return None
    parts = command.strip().split()
    if len(parts) < 2:
        return None
    return _parse_git_command(command)


def extract_file_changes(tool_invocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes = []
    for invocation in tool_invocations:
        tool_name = invocation.get("tool_name")
        input_data = invocation.get("input", {})
        if tool_name == "Read":
            path = input_data.get("file_path") or input_data.get("path")
            if path:
                changes.append({"path": path, "operation": "read"})
        elif tool_name == "Write":
            path = input_data.get("file_path") or input_data.get("path")
            content = input_data.get("content")
            if path:
                changes.append({
                    "path": path,
                    "operation": "write",
                    "new_content": content[:500] if isinstance(content, str) else None,
                })
        elif tool_name == "Edit":
            path = input_data.get("file_path") or input_data.get("path")
            old_string = input_data.get("old_string")
            new_string = input_data.get("new_string")
            if path:
                changes.append({
                    "path": path,
                    "operation": "edit",
                    "old_content": old_string[:200] if isinstance(old_string, str) else None,
                    "new_content": new_string[:200] if isinstance(new_string, str) else None,
                })
    return changes


def extract_subagent_spawns(tool_invocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spawns = []
    for invocation in tool_invocations:
        if invocation.get("tool_name") != "Task":
            continue
        input_data = invocation.get("input", {})
        parsed = _parse_subagent_spawn(input_data)
        spawns.append({
            "agent_type": parsed["agent_type"],
            "prompt": input_data.get("prompt", ""),
            "description": parsed["description"],
            "run_in_background": parsed["run_in_background"],
        })
    return spawns


def extract_git_operations(tool_invocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    operations = []
    for invocation in tool_invocations:
        git_op = parse_git_operation(invocation)
        if git_op:
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
