"""Tests for Claude Code semantic extraction.

These tests verify that the semantic extractors correctly parse
thinking traces, tool invocations, git operations, file changes,
subagent spawns, and context compaction events.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.importers.claude import (
    detect_context_compaction,
    extract_file_changes,
    extract_git_operations,
    extract_subagent_spawns,
    extract_thinking_traces,
    extract_tool_invocations,
    parse_code,
    parse_git_operation,
)


# =============================================================================
# Thinking Trace Extraction Tests
# =============================================================================


class TestThinkingTraces:
    """Tests for thinking trace extraction."""

    def test_extracts_thinking_blocks(self):
        """Thinking blocks are extracted with text and token count."""
        blocks = [
            {"type": "thinking", "thinking": "I need to analyze this carefully."},
            {"type": "text", "text": "Here is my response."},
        ]

        traces = extract_thinking_traces(blocks)

        assert len(traces) == 1
        assert "analyze" in traces[0]["text"]
        assert traces[0]["token_count"] > 0

    def test_handles_empty_thinking(self):
        """Empty thinking blocks are skipped."""
        blocks = [
            {"type": "thinking", "thinking": ""},
            {"type": "thinking", "thinking": None},
        ]

        traces = extract_thinking_traces(blocks)
        assert len(traces) == 0

    def test_handles_no_thinking_blocks(self):
        """No error when no thinking blocks present."""
        blocks = [
            {"type": "text", "text": "Hello"},
        ]

        traces = extract_thinking_traces(blocks)
        assert len(traces) == 0

    def test_handles_multiple_thinking_blocks(self):
        """Multiple thinking blocks are all extracted."""
        blocks = [
            {"type": "thinking", "thinking": "First thought"},
            {"type": "text", "text": "Response"},
            {"type": "thinking", "thinking": "Second thought"},
        ]

        traces = extract_thinking_traces(blocks)
        assert len(traces) == 2


# =============================================================================
# Tool Invocation Extraction Tests
# =============================================================================


class TestToolInvocations:
    """Tests for tool invocation extraction."""

    def test_extracts_tool_use_blocks(self):
        """Tool use blocks are extracted with name, id, and input."""
        blocks = [
            {
                "type": "tool_use",
                "name": "Read",
                "id": "tool_123",
                "input": {"file_path": "/test.py"},
            }
        ]

        invocations = extract_tool_invocations(blocks)

        assert len(invocations) == 1
        assert invocations[0]["tool_name"] == "Read"
        assert invocations[0]["tool_id"] == "tool_123"
        assert invocations[0]["input"]["file_path"] == "/test.py"

    def test_detects_file_operations(self):
        """File operations are correctly flagged."""
        file_tools = ["Read", "Write", "Edit", "NotebookEdit"]

        for tool in file_tools:
            blocks = [{"type": "tool_use", "name": tool, "id": "t1", "input": {}}]
            invocations = extract_tool_invocations(blocks)
            assert invocations[0]["is_file_operation"] == True, f"{tool} should be file operation"

    def test_detects_search_operations(self):
        """Search operations are correctly flagged."""
        search_tools = ["Glob", "Grep", "WebSearch"]

        for tool in search_tools:
            blocks = [{"type": "tool_use", "name": tool, "id": "t1", "input": {}}]
            invocations = extract_tool_invocations(blocks)
            assert invocations[0]["is_search_operation"] == True, f"{tool} should be search operation"

    def test_detects_git_operations(self):
        """Git commands in Bash are flagged."""
        blocks = [
            {
                "type": "tool_use",
                "name": "Bash",
                "id": "t1",
                "input": {"command": "git status"},
            }
        ]

        invocations = extract_tool_invocations(blocks)
        assert invocations[0]["is_git_operation"] == True

    def test_non_git_bash_not_flagged(self):
        """Non-git Bash commands are not flagged as git operations."""
        blocks = [
            {
                "type": "tool_use",
                "name": "Bash",
                "id": "t1",
                "input": {"command": "ls -la"},
            }
        ]

        invocations = extract_tool_invocations(blocks)
        assert invocations[0].get("is_git_operation") != True

    def test_detects_subagent_spawns(self):
        """Task tool is flagged as subagent."""
        blocks = [
            {
                "type": "tool_use",
                "name": "Task",
                "id": "t1",
                "input": {"subagent_type": "Explore"},
            }
        ]

        invocations = extract_tool_invocations(blocks)
        assert invocations[0]["is_subagent"] == True


# =============================================================================
# Git Operation Parsing Tests
# =============================================================================


class TestGitOperations:
    """Tests for git operation parsing."""

    def test_parse_git_status(self):
        """Git status command is parsed."""
        result = parse_git_operation({
            "tool_name": "Bash",
            "input": {"command": "git status"},
        })

        assert result["command"] == "status"

    def test_parse_git_commit_with_message(self):
        """Git commit message is extracted."""
        result = parse_git_operation({
            "tool_name": "Bash",
            "input": {"command": 'git commit -m "Fix bug"'},
        })

        assert result["command"] == "commit"
        assert result["message"] == "Fix bug"

    def test_parse_git_checkout_branch(self):
        """Git checkout branch is extracted."""
        result = parse_git_operation({
            "tool_name": "Bash",
            "input": {"command": "git checkout feature-branch"},
        })

        assert result["command"] == "checkout"
        assert result["branch"] == "feature-branch"

    def test_parse_git_push(self):
        """Git push remote and branch are extracted."""
        result = parse_git_operation({
            "tool_name": "Bash",
            "input": {"command": "git push origin main"},
        })

        assert result["command"] == "push"
        assert result["remote"] == "origin"
        assert result["branch"] == "main"

    def test_parse_git_add_files(self):
        """Git add files are extracted."""
        result = parse_git_operation({
            "tool_name": "Bash",
            "input": {"command": "git add file1.py file2.py"},
        })

        assert result["command"] == "add"
        assert "file1.py" in result["files"]
        assert "file2.py" in result["files"]

    def test_returns_none_for_non_git(self):
        """Non-git commands return None."""
        result = parse_git_operation({
            "tool_name": "Bash",
            "input": {"command": "ls -la"},
        })

        assert result is None

    def test_returns_none_for_non_bash(self):
        """Non-Bash tools return None."""
        result = parse_git_operation({
            "tool_name": "Read",
            "input": {"file_path": "/test.py"},
        })

        assert result is None


# =============================================================================
# File Change Extraction Tests
# =============================================================================


class TestFileChanges:
    """Tests for file change extraction."""

    def test_extract_read_operation(self):
        """Read operations are extracted."""
        invocations = [
            {"tool_name": "Read", "input": {"file_path": "/test.py"}},
        ]

        changes = extract_file_changes(invocations)

        assert len(changes) == 1
        assert changes[0]["path"] == "/test.py"
        assert changes[0]["operation"] == "read"

    def test_extract_write_operation(self):
        """Write operations are extracted with content."""
        invocations = [
            {
                "tool_name": "Write",
                "input": {"file_path": "/new.py", "content": "print('hello')"},
            },
        ]

        changes = extract_file_changes(invocations)

        assert len(changes) == 1
        assert changes[0]["operation"] == "write"
        assert changes[0]["new_content"] is not None

    def test_extract_edit_operation(self):
        """Edit operations are extracted with old and new content."""
        invocations = [
            {
                "tool_name": "Edit",
                "input": {
                    "file_path": "/test.py",
                    "old_string": "old code",
                    "new_string": "new code",
                },
            },
        ]

        changes = extract_file_changes(invocations)

        assert len(changes) == 1
        assert changes[0]["operation"] == "edit"
        assert changes[0]["old_content"] is not None
        assert changes[0]["new_content"] is not None

    def test_truncates_long_content(self):
        """Long content is truncated."""
        long_content = "x" * 1000
        invocations = [
            {"tool_name": "Write", "input": {"file_path": "/test.py", "content": long_content}},
        ]

        changes = extract_file_changes(invocations)

        assert len(changes[0]["new_content"]) <= 500


# =============================================================================
# Subagent Spawn Extraction Tests
# =============================================================================


class TestSubagentSpawns:
    """Tests for subagent spawn extraction."""

    def test_extract_task_invocation(self):
        """Task tool invocations are extracted as spawns."""
        invocations = [
            {
                "tool_name": "Task",
                "input": {
                    "subagent_type": "Explore",
                    "prompt": "Find all config files",
                    "description": "Search for configs",
                },
            },
        ]

        spawns = extract_subagent_spawns(invocations)

        assert len(spawns) == 1
        assert spawns[0]["agent_type"] == "Explore"
        assert spawns[0]["prompt"] == "Find all config files"
        assert spawns[0]["description"] == "Search for configs"

    def test_default_agent_type(self):
        """Missing subagent_type defaults to general-purpose."""
        invocations = [
            {"tool_name": "Task", "input": {"prompt": "Do something"}},
        ]

        spawns = extract_subagent_spawns(invocations)

        assert spawns[0]["agent_type"] == "general-purpose"

    def test_background_flag(self):
        """Background flag is captured."""
        invocations = [
            {
                "tool_name": "Task",
                "input": {"prompt": "Search", "run_in_background": True},
            },
        ]

        spawns = extract_subagent_spawns(invocations)

        assert spawns[0]["run_in_background"] == True

    def test_ignores_non_task_tools(self):
        """Non-Task tools are ignored."""
        invocations = [
            {"tool_name": "Read", "input": {"file_path": "/test.py"}},
            {"tool_name": "Bash", "input": {"command": "ls"}},
        ]

        spawns = extract_subagent_spawns(invocations)

        assert len(spawns) == 0


# =============================================================================
# Context Compaction Detection Tests
# =============================================================================


class TestContextCompaction:
    """Tests for context compaction detection."""

    def test_detects_summary_message(self):
        """Summary messages are detected as compaction events."""
        item = {
            "type": "summary",
            "message": {"content": "Summary of the conversation so far..."},
            "timestamp": 1704067200,
        }

        compaction = detect_context_compaction(item)

        assert compaction is not None
        assert "Summary" in compaction["summary"]
        assert compaction["timestamp"] == 1704067200

    def test_detects_summary_with_content_blocks(self):
        """Summary with content blocks is detected."""
        item = {
            "type": "summary",
            "message": {
                "content": [
                    {"type": "text", "text": "Conversation summary here"},
                ]
            },
        }

        compaction = detect_context_compaction(item)

        assert compaction is not None
        assert "summary" in compaction["summary"].lower()

    def test_returns_none_for_non_summary(self):
        """Non-summary messages return None."""
        item = {"type": "user", "message": {"content": "Hello"}}

        compaction = detect_context_compaction(item)

        assert compaction is None


# =============================================================================
# Integration Tests with parse_code
# =============================================================================


class TestParseCodeSemantic:
    """Integration tests for semantic extraction in parse_code."""

    def test_extracts_semantic_data_to_provider_meta(self):
        """parse_code extracts semantic data to message provider_meta."""
        payload = [
            {
                "type": "assistant",
                "uuid": "msg-1",
                "timestamp": 1704067200000,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me think..."},
                        {"type": "text", "text": "Here is my answer"},
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "id": "tool-1",
                            "input": {"file_path": "/test.py"},
                        },
                    ],
                },
            },
        ]

        result = parse_code(payload, "test-session")

        assert len(result.messages) == 1
        meta = result.messages[0].provider_meta

        # Check semantic data was extracted
        assert "thinking_traces" in meta
        assert len(meta["thinking_traces"]) == 1

        assert "tool_invocations" in meta
        assert len(meta["tool_invocations"]) == 1

        assert "file_changes" in meta
        assert len(meta["file_changes"]) == 1

    def test_extracts_git_operations(self):
        """parse_code extracts git operations from Bash commands."""
        payload = [
            {
                "type": "assistant",
                "uuid": "msg-1",
                "timestamp": 1704067200000,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "id": "tool-1",
                            "input": {"command": "git commit -m 'Fix bug'"},
                        },
                    ],
                },
            },
        ]

        result = parse_code(payload, "test-session")

        meta = result.messages[0].provider_meta
        assert "git_operations" in meta
        assert meta["git_operations"][0]["command"] == "commit"

    def test_extracts_subagent_spawns(self):
        """parse_code extracts subagent spawns from Task tools."""
        payload = [
            {
                "type": "assistant",
                "uuid": "msg-1",
                "timestamp": 1704067200000,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Task",
                            "id": "tool-1",
                            "input": {
                                "subagent_type": "Explore",
                                "prompt": "Find config files",
                            },
                        },
                    ],
                },
            },
        ]

        result = parse_code(payload, "test-session")

        meta = result.messages[0].provider_meta
        assert "subagent_spawns" in meta
        assert meta["subagent_spawns"][0]["agent_type"] == "Explore"

    def test_tracks_context_compactions(self):
        """parse_code tracks context compaction events at conversation level."""
        payload = [
            {"type": "user", "uuid": "msg-1", "message": {"content": "Hello"}},
            {
                "type": "summary",
                "message": {"content": "Summary of conversation"},
                "timestamp": 1704067200000,
            },
            {"type": "assistant", "uuid": "msg-2", "message": {"content": "Hi"}},
        ]

        result = parse_code(payload, "test-session")

        # Compaction should be in conversation provider_meta
        assert result.provider_meta is not None
        assert "context_compactions" in result.provider_meta
        assert len(result.provider_meta["context_compactions"]) == 1

    def test_aggregates_cost_and_duration(self):
        """parse_code aggregates total cost and duration."""
        payload = [
            {
                "type": "assistant",
                "uuid": "msg-1",
                "timestamp": 1704067200000,
                "costUSD": 0.01,
                "durationMs": 1000,
                "message": {"content": "Response 1"},
            },
            {
                "type": "assistant",
                "uuid": "msg-2",
                "timestamp": 1704067201000,
                "costUSD": 0.02,
                "durationMs": 2000,
                "message": {"content": "Response 2"},
            },
        ]

        result = parse_code(payload, "test-session")

        assert result.provider_meta is not None
        assert result.provider_meta["total_cost_usd"] == pytest.approx(0.03)
        assert result.provider_meta["total_duration_ms"] == 3000


# =============================================================================
# Property-Based Tests
# =============================================================================


@given(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=5))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_thinking_extraction_never_crashes(texts: list[str]):
    """Thinking trace extraction handles any input."""
    blocks = [{"type": "thinking", "thinking": text} for text in texts]
    traces = extract_thinking_traces(blocks)
    assert isinstance(traces, list)


@given(
    st.lists(
        st.sampled_from(["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task"]),
        min_size=0,
        max_size=10,
    )
)
@settings(max_examples=50)
def test_tool_extraction_never_crashes(tool_names: list[str]):
    """Tool invocation extraction handles any tools."""
    blocks = [
        {"type": "tool_use", "name": name, "id": f"t{i}", "input": {}}
        for i, name in enumerate(tool_names)
    ]
    invocations = extract_tool_invocations(blocks)
    assert len(invocations) == len(tool_names)
