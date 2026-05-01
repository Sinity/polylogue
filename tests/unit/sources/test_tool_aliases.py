from __future__ import annotations

from polylogue.lib.action_event.action_events import canonical_tool_name
from polylogue.lib.viewport.enums import ToolCategory
from polylogue.lib.viewport.tools import classify_tool


def test_codex_shell_aliases_share_canonical_name_and_category() -> None:
    assert classify_tool("exec_command", {"cmd": "ls"}) is ToolCategory.SHELL
    assert classify_tool("shell_command", {"command": "git status"}) is ToolCategory.GIT
    assert canonical_tool_name("exec_command") == "bash"
    assert canonical_tool_name("shell_command") == "bash"
    assert canonical_tool_name("Bash") == "bash"


def test_codex_agent_aliases_share_canonical_names() -> None:
    assert classify_tool("spawn_agent", {}) is ToolCategory.SUBAGENT
    assert classify_tool("update_plan", {}) is ToolCategory.AGENT
    assert canonical_tool_name("spawn_agent") == "task"
    assert canonical_tool_name("update_plan") == "todo"
