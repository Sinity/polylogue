"""Tests for semantic classification of parsed blocks at ingest time.

Covers:
- classify_tool / extract_tool_metadata helpers used during parsing
- the live `_semantic_type` writer classifier: tool_use blocks are
  categorized (git, file_read, subagent, ...) and all other block types
  (thinking, text) carry no semantic_type
"""

from __future__ import annotations

from polylogue.archive.viewport.viewports import ToolCategory, classify_tool
from polylogue.core.enums import BlockType
from polylogue.core.json import JSONDocument, json_document
from polylogue.pipeline.semantic_capture import extract_subagent_spawns, parse_git_operation
from polylogue.pipeline.semantic_metadata import extract_tool_metadata
from polylogue.sources.parsers.base_models import ParsedContentBlock
from polylogue.storage.sqlite.archive_tiers.write import _semantic_type


def _payload(data: object) -> JSONDocument:
    return json_document(data)


def _string_list(value: object) -> list[str]:
    return [entry for entry in value if isinstance(entry, str)] if isinstance(value, list) else []


# =============================================================================
# Tests for classify_tool
# =============================================================================


class TestClassifyTool:
    def test_read_is_file_read(self) -> None:
        assert classify_tool("Read", {}) == ToolCategory.FILE_READ

    def test_write_is_file_write(self) -> None:
        assert classify_tool("Write", {}) == ToolCategory.FILE_WRITE

    def test_edit_is_file_edit(self) -> None:
        assert classify_tool("Edit", {}) == ToolCategory.FILE_EDIT

    def test_multiedit_is_file_edit(self) -> None:
        assert classify_tool("MultiEdit", {}) == ToolCategory.FILE_EDIT

    def test_bash_without_git_is_shell(self) -> None:
        assert classify_tool("Bash", {"command": "ls -la"}) == ToolCategory.SHELL

    def test_bash_with_git_is_git(self) -> None:
        assert classify_tool("Bash", {"command": "git status"}) == ToolCategory.GIT

    def test_task_is_subagent(self) -> None:
        assert classify_tool("Task", {}) == ToolCategory.SUBAGENT

    def test_webfetch_is_web(self) -> None:
        assert classify_tool("WebFetch", {}) == ToolCategory.WEB

    def test_glob_is_search(self) -> None:
        assert classify_tool("Glob", {}) == ToolCategory.SEARCH

    def test_agent_planning_tools_are_agent_category(self) -> None:
        assert classify_tool("TodoWrite", {}) == ToolCategory.AGENT
        assert classify_tool("TaskCreate", {}) == ToolCategory.AGENT
        assert classify_tool("EnterPlanMode", {}) == ToolCategory.AGENT
        assert classify_tool("Skill", {}) == ToolCategory.AGENT

    def test_shell_control_tools_are_shell_category(self) -> None:
        assert classify_tool("KillShell", {}) == ToolCategory.SHELL

    def test_lsp_inspection_tools_are_search_category(self) -> None:
        assert classify_tool("mcp__cclsp__find_definition", {}) == ToolCategory.SEARCH
        assert classify_tool("mcp__plugin_serena_serena__find_referencing_symbols", {}) == ToolCategory.SEARCH
        assert classify_tool("mcp__plugin_serena_serena__get_symbols_in_file", {}) == ToolCategory.SEARCH
        assert classify_tool("mcp__cclsp__get_diagnostics", {}) == ToolCategory.SEARCH

    def test_tabs_context_tools_are_web_category(self) -> None:
        assert classify_tool("mcp__claude-in-chrome__tabs_context_mcp", {}) == ToolCategory.WEB

    def test_ls_is_search(self) -> None:
        assert classify_tool("LS", {"path": "/workspace/polylogue"}) == ToolCategory.SEARCH

    def test_unknown_tool_is_other(self) -> None:
        assert classify_tool("FancyCustomTool", {}) == ToolCategory.OTHER


# =============================================================================
# Tests for extract_tool_metadata
# =============================================================================


class TestExtractToolMetadata:
    def test_git_status_returns_metadata(self) -> None:
        meta = extract_tool_metadata("Bash", _payload({"command": "git status"}))
        assert meta is not None
        assert meta["command"] == "status"
        assert "full_command" in meta

    def test_git_commit_extracts_message(self) -> None:
        meta = extract_tool_metadata("Bash", _payload({"command": 'git commit -m "fix: bug"'}))
        assert meta is not None
        assert meta["command"] == "commit"
        assert meta.get("message") == "fix: bug"

    def test_git_push_extracts_remote_branch(self) -> None:
        meta = extract_tool_metadata("Bash", _payload({"command": "git push origin main"}))
        assert meta is not None
        assert meta.get("remote") == "origin"
        assert meta.get("branch") == "main"

    def test_git_checkout_extracts_branch(self) -> None:
        meta = extract_tool_metadata("Bash", _payload({"command": "git checkout feature-x"}))
        assert meta is not None
        assert meta.get("branch") == "feature-x"

    def test_read_returns_file_path(self) -> None:
        meta = extract_tool_metadata("Read", _payload({"file_path": "/path/to/file.py"}))
        assert meta is not None
        assert meta["path"] == "/path/to/file.py"

    def test_write_returns_file_path_and_snippet(self) -> None:
        meta = extract_tool_metadata(
            "Write",
            _payload({"file_path": "/path/to/file.py", "content": "hello world"}),
        )
        assert meta is not None
        assert meta["path"] == "/path/to/file.py"
        assert "new_content_snippet" in meta

    def test_edit_returns_path_and_snippets(self) -> None:
        meta = extract_tool_metadata(
            "Edit",
            _payload(
                {
                    "file_path": "/path/to/file.py",
                    "old_string": "old code",
                    "new_string": "new code",
                }
            ),
        )
        assert meta is not None
        assert meta["path"] == "/path/to/file.py"
        assert "old_snippet" in meta
        assert "new_snippet" in meta

    def test_task_returns_agent_type(self) -> None:
        meta = extract_tool_metadata(
            "Task",
            _payload(
                {
                    "subagent_type": "general-purpose",
                    "description": "Do something",
                    "prompt": "Please do X",
                }
            ),
        )
        assert meta is not None
        assert meta["agent_type"] == "general-purpose"
        assert "prompt_snippet" in meta

    def test_shell_returns_none(self) -> None:
        meta = extract_tool_metadata("Bash", _payload({"command": "ls -la"}))
        assert meta is None

    def test_search_returns_path_and_pattern(self) -> None:
        meta = extract_tool_metadata(
            "Grep",
            _payload({"path": "/workspace/polylogue", "pattern": "build_session_profile"}),
        )
        assert meta is not None
        assert meta["path"] == "/workspace/polylogue"
        assert meta["pattern"] == "build_session_profile"

    def test_agent_todo_metadata_summarizes_todos(self) -> None:
        meta = extract_tool_metadata("TodoWrite", _payload({"todos": [{"id": "1"}, {"id": "2"}]}))
        assert meta is not None
        assert meta["tool"] == "TodoWrite"
        assert meta["todo_count"] == 2

    def test_other_tool_returns_none(self) -> None:
        meta = extract_tool_metadata("UnknownTool", _payload({}))
        assert meta is None


# =============================================================================
# Tests for the live block semantic_type classifier (write path)
# =============================================================================


class TestSemanticTypeClassifier:
    """`_semantic_type` is the function the archive writer runs over each parsed
    block to fill `blocks.semantic_type`. It classifies tool_use blocks by tool
    category and leaves every other block uncategorized."""

    def test_tool_use_git_block_classified_as_git(self) -> None:
        block = ParsedContentBlock(
            type=BlockType.TOOL_USE,
            tool_name="Bash",
            tool_id="t1",
            tool_input={"command": "git status"},
        )
        assert _semantic_type(block) == ToolCategory.GIT.value

    def test_tool_use_file_read_block_classified_as_file_read(self) -> None:
        block = ParsedContentBlock(
            type=BlockType.TOOL_USE,
            tool_name="Read",
            tool_id="t2",
            tool_input={"file_path": "/tmp/x.py"},
        )
        assert _semantic_type(block) == ToolCategory.FILE_READ.value

    def test_tool_use_subagent_block_classified_as_subagent(self) -> None:
        block = ParsedContentBlock(
            type=BlockType.TOOL_USE,
            tool_name="Task",
            tool_id="t3",
            tool_input={"prompt": "do work"},
        )
        assert _semantic_type(block) == ToolCategory.SUBAGENT.value

    def test_uncategorized_tool_use_block_has_no_semantic_type(self) -> None:
        block = ParsedContentBlock(
            type=BlockType.TOOL_USE,
            tool_name="UnknownTool",
            tool_id="t4",
            tool_input={},
        )
        assert _semantic_type(block) is None

    def test_thinking_block_has_no_semantic_type(self) -> None:
        # Thinking is carried by block_type=THINKING, not semantic_type; the
        # writer only assigns semantic_type to tool_use blocks.
        block = ParsedContentBlock(type=BlockType.THINKING, text="reasoning")
        assert _semantic_type(block) is None

    def test_text_block_has_no_semantic_type(self) -> None:
        block = ParsedContentBlock(type=BlockType.TEXT, text="hello")
        assert _semantic_type(block) is None


# =============================================================================
# Tests for parse_git_operation (old-format API, now in semantic.py)
# =============================================================================


class TestParseGitOperation:
    def test_git_status(self) -> None:
        result = parse_git_operation({"tool_name": "Bash", "input": {"command": "git status"}})
        assert result is not None
        assert result["command"] == "status"

    def test_non_git_bash(self) -> None:
        result = parse_git_operation({"tool_name": "Bash", "input": {"command": "ls -la"}})
        assert result is None

    def test_non_bash_tool(self) -> None:
        result = parse_git_operation({"tool_name": "Read", "input": {"file_path": "/foo"}})
        assert result is None

    def test_git_add(self) -> None:
        result = parse_git_operation({"tool_name": "Bash", "input": {"command": "git add file1.py file2.py"}})
        assert result is not None
        assert "file1.py" in _string_list(result.get("files"))


# =============================================================================
# Tests for extract_subagent_spawns (old-format API, now in semantic.py)
# =============================================================================


class TestExtractSubagentSpawns:
    def test_task_tool_extracted(self) -> None:
        invocations = [{"tool_name": "Task", "input": {"subagent_type": "general-purpose", "prompt": "Do X"}}]
        spawns = extract_subagent_spawns(invocations)
        assert len(spawns) == 1
        assert spawns[0]["agent_type"] == "general-purpose"

    def test_non_task_tools_skipped(self) -> None:
        invocations = [{"tool_name": "Bash", "input": {"command": "ls"}}]
        spawns = extract_subagent_spawns(invocations)
        assert len(spawns) == 0

    def test_empty_invocations(self) -> None:
        assert extract_subagent_spawns([]) == []
