"""Structured semantic metadata extraction for parsed tool blocks."""

from __future__ import annotations

from typing import Any

from polylogue.lib.viewports import ToolCategory, classify_tool


def extract_tool_metadata(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any] | None:
    """Return structured metadata dict for a tool_use block, or None."""
    category = classify_tool(tool_name, tool_input)
    if category == ToolCategory.GIT:
        cmd = tool_input.get("command", "")
        if isinstance(cmd, str):
            return _parse_git_command(cmd)
        return None
    if category == ToolCategory.SUBAGENT:
        return _parse_subagent_spawn(tool_input)
    if category == ToolCategory.AGENT:
        return _extract_agent_metadata(tool_name, tool_input)
    if category == ToolCategory.SEARCH:
        return _extract_search_metadata(tool_input)
    if category in (ToolCategory.FILE_READ, ToolCategory.FILE_WRITE, ToolCategory.FILE_EDIT):
        return _extract_file_paths(tool_name, tool_input)
    return None


def _parse_git_command(cmd: str) -> dict[str, Any]:
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
    prompt = tool_input.get("prompt", "")
    return {
        "agent_type": tool_input.get("subagent_type", "general-purpose"),
        "description": tool_input.get("description"),
        "prompt_snippet": prompt[:200] if isinstance(prompt, str) else None,
        "run_in_background": tool_input.get("run_in_background", False),
    }


def _extract_file_paths(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any] | None:
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


def _extract_search_metadata(tool_input: dict[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    path = tool_input.get("path")
    if isinstance(path, str) and path:
        result["path"] = path
    pattern = tool_input.get("pattern")
    if isinstance(pattern, str) and pattern:
        result["pattern"] = pattern
    query = tool_input.get("query")
    if isinstance(query, str) and query:
        result["query"] = query[:500]
    return result or None


def _extract_agent_metadata(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {"tool": tool_name}
    description = tool_input.get("description")
    if isinstance(description, str) and description:
        result["description"] = description[:200]
    prompt = tool_input.get("prompt")
    if isinstance(prompt, str) and prompt:
        result["prompt_snippet"] = prompt[:200]
    plan = tool_input.get("plan")
    if isinstance(plan, str) and plan:
        result["plan_snippet"] = plan[:200]
    subject = tool_input.get("subject")
    if isinstance(subject, str) and subject:
        result["subject"] = subject[:200]
    task_id = tool_input.get("taskId") or tool_input.get("task_id")
    if isinstance(task_id, str) and task_id:
        result["task_id"] = task_id
    todos = tool_input.get("todos")
    if isinstance(todos, list):
        result["todo_count"] = len(todos)
    questions = tool_input.get("questions")
    if isinstance(questions, list):
        result["question_count"] = len(questions)
    return result


__all__ = ["extract_tool_metadata"]
