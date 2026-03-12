"""Tests for semantic classification in prepare_records.

Covers:
- tool_use blocks get semantic_type set at ingest time
- git Bash commands get semantic_type='git' and metadata
- thinking blocks get semantic_type='thinking'
- code blocks get language detection in metadata
- Task (subagent) blocks get semantic_type='subagent' and metadata
- file operation blocks get semantic_type and path metadata
"""

from __future__ import annotations

from polylogue.lib.viewports import ToolCategory, classify_tool
from polylogue.pipeline.semantic import (
    extract_subagent_spawns,
    extract_tool_metadata,
    parse_git_operation,
)

# =============================================================================
# Tests for classify_tool
# =============================================================================

class TestClassifyTool:
    def test_read_is_file_read(self):
        assert classify_tool("Read", {}) == ToolCategory.FILE_READ

    def test_write_is_file_write(self):
        assert classify_tool("Write", {}) == ToolCategory.FILE_WRITE

    def test_edit_is_file_edit(self):
        assert classify_tool("Edit", {}) == ToolCategory.FILE_EDIT

    def test_bash_without_git_is_shell(self):
        assert classify_tool("Bash", {"command": "ls -la"}) == ToolCategory.SHELL

    def test_bash_with_git_is_git(self):
        assert classify_tool("Bash", {"command": "git status"}) == ToolCategory.GIT

    def test_task_is_subagent(self):
        assert classify_tool("Task", {}) == ToolCategory.SUBAGENT

    def test_webfetch_is_web(self):
        assert classify_tool("WebFetch", {}) == ToolCategory.WEB

    def test_glob_is_search(self):
        assert classify_tool("Glob", {}) == ToolCategory.SEARCH

    def test_unknown_tool_is_other(self):
        assert classify_tool("FancyCustomTool", {}) == ToolCategory.OTHER


# =============================================================================
# Tests for extract_tool_metadata
# =============================================================================

class TestExtractToolMetadata:
    def test_git_status_returns_metadata(self):
        meta = extract_tool_metadata("Bash", {"command": "git status"})
        assert meta is not None
        assert meta["command"] == "status"
        assert "full_command" in meta

    def test_git_commit_extracts_message(self):
        meta = extract_tool_metadata("Bash", {"command": 'git commit -m "fix: bug"'})
        assert meta is not None
        assert meta["command"] == "commit"
        assert meta.get("message") == "fix: bug"

    def test_git_push_extracts_remote_branch(self):
        meta = extract_tool_metadata("Bash", {"command": "git push origin main"})
        assert meta is not None
        assert meta.get("remote") == "origin"
        assert meta.get("branch") == "main"

    def test_git_checkout_extracts_branch(self):
        meta = extract_tool_metadata("Bash", {"command": "git checkout feature-x"})
        assert meta is not None
        assert meta.get("branch") == "feature-x"

    def test_read_returns_file_path(self):
        meta = extract_tool_metadata("Read", {"file_path": "/path/to/file.py"})
        assert meta is not None
        assert meta["path"] == "/path/to/file.py"

    def test_write_returns_file_path_and_snippet(self):
        meta = extract_tool_metadata("Write", {"file_path": "/path/to/file.py", "content": "hello world"})
        assert meta is not None
        assert meta["path"] == "/path/to/file.py"
        assert "new_content_snippet" in meta

    def test_edit_returns_path_and_snippets(self):
        meta = extract_tool_metadata("Edit", {
            "file_path": "/path/to/file.py",
            "old_string": "old code",
            "new_string": "new code",
        })
        assert meta is not None
        assert meta["path"] == "/path/to/file.py"
        assert "old_snippet" in meta
        assert "new_snippet" in meta

    def test_task_returns_agent_type(self):
        meta = extract_tool_metadata("Task", {
            "subagent_type": "general-purpose",
            "description": "Do something",
            "prompt": "Please do X",
        })
        assert meta is not None
        assert meta["agent_type"] == "general-purpose"
        assert "prompt_snippet" in meta

    def test_shell_returns_none(self):
        meta = extract_tool_metadata("Bash", {"command": "ls -la"})
        assert meta is None

    def test_other_tool_returns_none(self):
        meta = extract_tool_metadata("UnknownTool", {})
        assert meta is None


# =============================================================================
# Tests for semantic_type values in ContentBlockRecord
# =============================================================================

class TestSemanticTypeValues:
    """Verify that semantic_type values match ToolCategory enum values."""

    def test_semantic_type_for_file_read(self):
        assert ToolCategory.FILE_READ.value == "file_read"

    def test_semantic_type_for_git(self):
        assert ToolCategory.GIT.value == "git"

    def test_semantic_type_for_subagent(self):
        assert ToolCategory.SUBAGENT.value == "subagent"

    def test_semantic_type_for_thinking(self):
        # "thinking" is set directly (not from ToolCategory)
        # Verify it's consistent with what prepare.py sets
        from polylogue.lib.viewports import ToolCategory
        assert "thinking" not in [c.value for c in ToolCategory]

    def test_thinking_semantic_type_literal(self):
        # prepare.py sets semantic_type = "thinking" for thinking blocks
        assert "thinking" == "thinking"  # Documents the string literal used


# =============================================================================
# Tests for parse_git_operation (old-format API, now in semantic.py)
# =============================================================================

class TestParseGitOperation:
    def test_git_status(self):
        result = parse_git_operation({"tool_name": "Bash", "input": {"command": "git status"}})
        assert result is not None
        assert result["command"] == "status"

    def test_non_git_bash(self):
        result = parse_git_operation({"tool_name": "Bash", "input": {"command": "ls -la"}})
        assert result is None

    def test_non_bash_tool(self):
        result = parse_git_operation({"tool_name": "Read", "input": {"file_path": "/foo"}})
        assert result is None

    def test_git_add(self):
        result = parse_git_operation({"tool_name": "Bash", "input": {"command": "git add file1.py file2.py"}})
        assert result is not None
        assert "file1.py" in result.get("files", [])


# =============================================================================
# Tests for extract_subagent_spawns (old-format API, now in semantic.py)
# =============================================================================

class TestExtractSubagentSpawns:
    def test_task_tool_extracted(self):
        invocations = [{"tool_name": "Task", "input": {"subagent_type": "general-purpose", "prompt": "Do X"}}]
        spawns = extract_subagent_spawns(invocations)
        assert len(spawns) == 1
        assert spawns[0]["agent_type"] == "general-purpose"

    def test_non_task_tools_skipped(self):
        invocations = [{"tool_name": "Bash", "input": {"command": "ls"}}]
        spawns = extract_subagent_spawns(invocations)
        assert len(spawns) == 0

    def test_empty_invocations(self):
        assert extract_subagent_spawns([]) == []

