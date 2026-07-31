from __future__ import annotations

from polylogue.archive.actions.actions import canonical_tool_name
from polylogue.archive.viewport.enums import ToolCategory
from polylogue.archive.viewport.tools import classify_tool


def test_codex_shell_aliases_share_canonical_name_and_category() -> None:
    assert classify_tool("exec_command", {"cmd": "ls"}) is ToolCategory.SHELL
    assert classify_tool("exec", {"command": "await tools.exec_command(...)"}) is ToolCategory.SHELL
    assert classify_tool("shell_command", {"command": "git status"}) is ToolCategory.GIT
    assert canonical_tool_name("exec_command") == "bash"
    assert canonical_tool_name("exec") == "bash"
    assert canonical_tool_name("shell_command") == "bash"
    assert canonical_tool_name("Bash") == "bash"


def test_codex_agent_aliases_share_canonical_names() -> None:
    assert classify_tool("spawn_agent", {}) is ToolCategory.SUBAGENT
    assert classify_tool("update_plan", {}) is ToolCategory.AGENT
    assert canonical_tool_name("spawn_agent") == "task"
    assert canonical_tool_name("update_plan") == "todo"


def test_agent_tool_classifies_as_subagent_dispatch() -> None:
    """polylogue-1vpm.7: the Claude Agent SDK's "Agent" tool is the direct
    successor to Claude Code's "Task" tool -- same dispatch shape (a
    ``prompt`` field the child's own first turn reproduces verbatim). It
    must land in ToolCategory.SUBAGENT, the category
    delegation_facts_source's `WHERE a.semantic_type = 'subagent'` filters
    on, not the generic ToolCategory.AGENT bucket (which also holds
    non-delegation control tools like askuserquestion/skill/batch/todo*).
    Without this, delegation resolution finds zero dispatch actions for any
    session that uses "Agent" instead of "Task" -- the join-key fix in this
    same bead has nothing to join for those sessions."""
    assert classify_tool("Agent", {"prompt": "do the thing", "model": "sonnet"}) is ToolCategory.SUBAGENT
    assert classify_tool("agent", {}) is ToolCategory.SUBAGENT


def test_codex_code_mode_child_aliases_have_queryable_categories() -> None:
    assert classify_tool("apply_patch", {"path": "src/example.py"}) is ToolCategory.FILE_EDIT
    assert classify_tool("wait", {"cell_id": "cell-7"}) is ToolCategory.AGENT
