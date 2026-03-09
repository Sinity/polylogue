"""Semantic extraction from parsed content blocks.

Computes structured metadata for tool_use blocks at ingest time.
Results are stored in content_blocks.semantic_type and metadata columns.

This is the canonical home for semantic extraction — originally these functions
lived in polylogue/sources/parsers/claude.py but are now provider-agnostic since
classify_tool() in viewports.py covers all providers.
"""

from __future__ import annotations

from typing import Any

from polylogue.lib.viewports import ToolCategory, classify_tool


def extract_tool_metadata(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any] | None:
    """Return structured metadata dict for a tool_use block, or None.

    Args:
        tool_name: Name of the tool (e.g. "Bash", "Read", "Task")
        tool_input: Tool input parameters (already-parsed dict)

    Returns:
        Structured metadata dict or None if no semantic metadata applies.
    """
    category = classify_tool(tool_name, tool_input)
    if category == ToolCategory.GIT:
        cmd = tool_input.get("command", "")
        if isinstance(cmd, str):
            return _parse_git_command(cmd)
        return None
    if category == ToolCategory.SUBAGENT:
        return _parse_subagent_spawn(tool_input)
    if category in (ToolCategory.FILE_READ, ToolCategory.FILE_WRITE, ToolCategory.FILE_EDIT):
        return _extract_file_paths(tool_name, tool_input)
    return None


def _parse_git_command(cmd: str) -> dict[str, Any]:
    """Parse git command string into structured metadata.

    Adapted from claude.py parse_git_operation() — operates on the command
    string directly rather than a raw tool invocation dict.
    """
    parts = cmd.strip().split()
    if len(parts) < 2:
        return {"full_command": cmd}

    git_cmd = parts[1]
    result: dict[str, Any] = {
        "command": git_cmd,
        "full_command": cmd,
    }

    if git_cmd == "commit":
        for i, part in enumerate(parts):
            if part == "-m" and i + 1 < len(parts):
                msg_start = cmd.find("-m") + 2
                if msg_start > 2:
                    remaining = cmd[msg_start:].strip()
                    if remaining.startswith('"'):
                        end_quote = remaining.find('"', 1)
                        if end_quote > 0:
                            result["message"] = remaining[1:end_quote]
                    elif remaining.startswith("'"):
                        end_quote = remaining.find("'", 1)
                        if end_quote > 0:
                            result["message"] = remaining[1:end_quote]

    elif git_cmd in ("checkout", "switch"):
        for part in parts[2:]:
            if not part.startswith("-"):
                result["branch"] = part
                break

    elif git_cmd == "push":
        non_flags = [p for p in parts[2:] if not p.startswith("-")]
        if len(non_flags) >= 2:
            result["remote"] = non_flags[0]
            result["branch"] = non_flags[1]
        elif len(non_flags) == 1:
            result["remote"] = non_flags[0]

    elif git_cmd in ("add", "rm"):
        result["files"] = [p for p in parts[2:] if not p.startswith("-")]

    return result


def _parse_subagent_spawn(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Extract subagent spawn metadata from Task tool input.

    Adapted from claude.py extract_subagent_spawns().
    """
    prompt = tool_input.get("prompt", "")
    return {
        "agent_type": tool_input.get("subagent_type", "general-purpose"),
        "description": tool_input.get("description"),
        "prompt_snippet": prompt[:200] if isinstance(prompt, str) else None,
        "run_in_background": tool_input.get("run_in_background", False),
    }


def _extract_file_paths(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any] | None:
    """Extract file path metadata from file operation tool input.

    Adapted from claude.py extract_file_changes().
    """
    path = tool_input.get("file_path") or tool_input.get("path")
    if not path:
        return None

    result: dict[str, Any] = {"path": path}

    if tool_name == "Write":
        content = tool_input.get("content")
        if isinstance(content, str):
            result["new_content_snippet"] = content[:500]
    elif tool_name == "Edit":
        old_string = tool_input.get("old_string")
        new_string = tool_input.get("new_string")
        if isinstance(old_string, str):
            result["old_snippet"] = old_string[:200]
        if isinstance(new_string, str):
            result["new_snippet"] = new_string[:200]

    return result


def detect_context_compaction(item: dict[str, Any]) -> dict[str, Any] | None:
    """Detect if a raw message item represents a context compaction event.

    Kept from claude.py — operates on raw message dicts, not content blocks.
    Used during claude.py parsing to detect summary messages.

    Args:
        item: Raw message dict from Claude Code JSONL

    Returns:
        Context compaction dict or None if not a compaction
    """
    msg_type = item.get("type")

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
        }

    return None


# =============================================================================
# Old-format API (raw dict-based) — kept for test coverage and internal callers
# These functions operate on raw API dict format (list of dicts with "type",
# "name", "input" keys) as opposed to ParsedContentBlock objects.
# =============================================================================


def extract_thinking_traces(content_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract structured thinking traces from raw content block dicts.

    Args:
        content_blocks: List of content block dicts from Claude Code message

    Returns:
        List of thinking trace dicts with text and token_count
    """
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
    """Extract structured tool invocations from raw content block dicts.

    Args:
        content_blocks: List of content block dicts from Claude Code message

    Returns:
        List of tool invocation dicts with name, id, input, and derived semantics
    """
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
    """Parse git operation details from a raw Bash tool invocation dict.

    Args:
        tool_invocation: Tool invocation dict with tool_name and input fields

    Returns:
        Git operation dict or None if not a git command
    """
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
    """Extract file change information from raw tool invocation dicts.

    Args:
        tool_invocations: List of tool invocation dicts with tool_name/input keys

    Returns:
        List of file change dicts with path, operation type
    """
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
    """Extract subagent spawn info from raw Task tool invocation dicts.

    Args:
        tool_invocations: List of tool invocation dicts with tool_name/input keys

    Returns:
        List of subagent spawn dicts
    """
    spawns = []
    for invocation in tool_invocations:
        if invocation.get("tool_name") != "Task":
            continue
        input_data = invocation.get("input", {})
        spawns.append({
            "agent_type": input_data.get("subagent_type", "general-purpose"),
            "prompt": input_data.get("prompt", ""),
            "description": input_data.get("description"),
            "run_in_background": input_data.get("run_in_background", False),
        })
    return spawns


def extract_git_operations(tool_invocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract all git operations from raw tool invocation dicts.

    Args:
        tool_invocations: List of tool invocation dicts

    Returns:
        List of git operation dicts
    """
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
    "extract_tool_metadata",
    "parse_git_operation",
]
