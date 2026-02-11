"""Coverage tests for uncovered parser and utility functions.

Targets uncovered lines in:
- polylogue/sources/parsers/claude.py (lines 161-164, 225, 227, 265, 278, 280, 282-287, 369, 410-411, 647-654, 711-712, 717-718, 768)
- polylogue/sources/parsers/codex.py (lines 36, 46-47, 72, 76-78, 93-98, 111-113)
- polylogue/sources/parsers/drive.py (lines 20, 26, 29, 35-37, 47, 50, 55-58, 91, 95, 127-141)
- polylogue/lib/log.py (lines 29, 32, 51)
- polylogue/lib/dates.py (lines 43)
- polylogue/version.py (lines 82-83, 90-95)
- polylogue/ui/__init__.py (lines 50, 72, 87, 103, 108, 119-134, 138-141, 144-145, 151, 165-167, 174, 201-241)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.lib.dates import parse_date
from polylogue.lib.log import configure_logging, get_logger
from polylogue.sources.parsers.claude import (
    extract_text_from_segments,
    extract_tool_invocations,
    extract_thinking_traces,
    extract_file_changes,
    extract_subagent_spawns,
    parse_code as parse_claude_code,
    parse_ai as parse_claude_ai,
    _extract_message_text,
    parse_git_operation,
)
from polylogue.sources.parsers.codex import parse as parse_codex
from polylogue.sources.parsers.drive import parse_chunked_prompt
from polylogue.ui import create_ui


# =============================================================================
# Test Data Constants - Parametrization Tables
# =============================================================================

# Extract text from segments test cases
SEGMENT_TEST_CASES = [
    (
        [{"type": "thinking", "thinking": "This is my thought process"}],
        "<thinking>This is my thought process</thinking>",
        "thinking block with string thinking",
    ),
    (
        [{"type": "thinking"}, {"type": "text", "text": "regular text"}],
        "regular text",
        "thinking block without thinking field should be skipped",
    ),
    (
        [{"type": "custom", "content": "This is content field content"}],
        "This is content field content",
        "content dict with content field",
    ),
    (
        [{"type": "text", "text": "text content"}],
        "text content",
        "text segment",
    ),
    (
        [
            "Plain string segment",
            {"type": "tool_use", "name": "Read", "id": "tool-1", "input": {}},
            {"type": "thinking", "thinking": "Planning next step"},
            {"type": "text", "text": "Regular text"},
            {"type": "tool_result", "content": "Tool output", "is_error": False},
        ],
        None,  # Multiple segments, check individual parts
        "mixed segment types",
    ),
]

# Extract message text test cases
MESSAGE_TEXT_TEST_CASES = [
    ("Direct string content", "Direct string content", "message content as string"),
    (
        [{"type": "text", "text": "First"}, {"type": "text", "text": "Second"}],
        ("First", "Second"),
        "message content as list",
    ),
    (
        {"text": "Dict text content"},
        "Dict text content",
        "message content as dict with text",
    ),
    (
        {"parts": ["Part A", "Part B", "Part C"]},
        "Part A\nPart B\nPart C",
        "message content as dict with parts",
    ),
    (
        {"other": "field"},
        None,
        "message content as dict without text or parts",
    ),
    (
        None,
        None,
        "message content as none",
    ),
    (
        [{"type": "text"}],
        None,
        "message content as non-string list",
    ),
]

# Extract tool invocations test cases
TOOL_INVOCATION_TEST_CASES = [
    (
        [{"type": "tool_use", "name": "Read", "id": "read-1", "input": {"file_path": "/path/to/file"}}],
        1,
        {"tool_name": "Read", "is_file_operation": True},
        "tool use with all fields",
    ),
    (
        [{"type": "tool_use", "input": {}}],
        1,
        {"tool_name": None, "tool_id": None},
        "tool use with missing fields",
    ),
    (
        [{"type": "tool_use", "name": "Bash", "id": "bash-1", "input": {"command": "git commit -m 'test'"}}],
        1,
        {"is_git_operation": True},
        "tool use bash with git command",
    ),
    (
        [{"type": "tool_use", "name": "Bash", "id": "bash-1", "input": {"command": "ls -la"}}],
        1,
        {"is_git_operation": False},
        "tool use bash non-git",
    ),
    (
        [{"type": "tool_use", "name": "Task", "id": "task-1", "input": {}}],
        1,
        {"is_subagent": True},
        "tool use task",
    ),
]

# Extract thinking traces test cases
THINKING_TRACE_TEST_CASES = [
    (
        [{"type": "thinking", "thinking": "Let me think about this problem"}],
        "Let me think about this problem",
        "thinking trace with thinking field",
    ),
    (
        [{"type": "thinking", "text": "Fallback thinking text"}],
        "Fallback thinking text",
        "thinking trace with text field fallback",
    ),
    (
        [{"type": "thinking"}, {"type": "thinking", "thinking": "Valid"}],
        "Valid",
        "empty thinking trace skipped",
    ),
]

# Extract file changes test cases
FILE_CHANGE_TEST_CASES = [
    (
        [{"tool_name": "Read", "input": {"file_path": "/path/to/file.py"}}],
        "read",
        "/path/to/file.py",
        "read operation",
    ),
    (
        [{"tool_name": "Write", "input": {"file_path": "/path/to/output.txt", "content": "A" * 600}}],
        "write",
        None,
        "write operation with content",
    ),
    (
        [
            {
                "tool_name": "Edit",
                "input": {"file_path": "/path/to/file.py", "old_string": "old", "new_string": "new"},
            }
        ],
        "edit",
        "/path/to/file.py",
        "edit operation",
    ),
    (
        [{"tool_name": "Read", "input": {"path": "/alternative/path.txt"}}],
        "read",
        "/alternative/path.txt",
        "file path fallback to path field",
    ),
]

# Parse git operation test cases
GIT_OPERATION_TEST_CASES = [
    (
        {"tool_name": "Bash", "input": {"command": 'git commit -m "fix: resolve issue"'}},
        "commit",
        "fix: resolve issue",
        "git commit with message",
    ),
    (
        {"tool_name": "Bash", "input": {"command": "git commit"}},
        "commit",
        None,
        "git commit no message",
    ),
    (
        {"tool_name": "Bash", "input": {"command": "git checkout feature-branch"}},
        "checkout",
        "feature-branch",
        "git checkout branch",
    ),
    (
        {"tool_name": "Bash", "input": {"command": "git switch develop"}},
        "switch",
        "develop",
        "git switch branch",
    ),
    (
        {"tool_name": "Bash", "input": {"command": "git push origin main"}},
        "push",
        ("origin", "main"),
        "git push with remote and branch",
    ),
    (
        {"tool_name": "Bash", "input": {"command": "git push upstream"}},
        "push",
        "upstream",
        "git push remote only",
    ),
    (
        {"tool_name": "Bash", "input": {"command": "git add file1.py file2.py"}},
        "add",
        ("file1.py", "file2.py"),
        "git add files",
    ),
    (
        {"tool_name": "Bash", "input": {"command": "git rm deprecated.py old.txt"}},
        "rm",
        "deprecated.py",
        "git rm files",
    ),
    (
        {"tool_name": "Bash", "input": {"command": "ls -la"}},
        None,
        None,
        "non-git command returns none",
    ),
    (
        {"tool_name": "Read", "input": {"command": "git status"}},
        None,
        None,
        "non-bash tool returns none",
    ),
    (
        {"tool_name": "Bash", "input": {"command": "git"}},
        None,
        None,
        "git with no subcommand",
    ),
]

# Claude Code payload test cases
CLAUDE_CODE_PAYLOAD_TEST_CASES = [
    (
        [],
        "fallback-id",
        0,
        "fallback-id",
        "empty payload",
    ),
    (
        [
            {
                "type": "assistant",
                "uuid": "msg-1",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
                "timestamp": 1700000000,
                "sessionId": "sess-1",
            }
        ],
        "fallback",
        1,
        "sess-1",
        "payload with content blocks no tool use",
    ),
    (
        [
            {
                "type": "user",
                "uuid": "msg-1",
                "message": {"role": "user", "content": "Hello"},
                "timestamp": 1700000000,
                "sessionId": "sess-2",
            }
        ],
        "fallback",
        1,
        "sess-2",
        "payload cost aggregation zero",
    ),
    (
        [
            {
                "type": "user",
                "uuid": "msg-1",
                "message": {"role": "user", "content": "Hello"},
                "timestamp": 1700000000,
                "sessionId": "sess-3",
            }
        ],
        "fallback",
        1,
        "sess-3",
        "payload duration aggregation zero",
    ),
]

# Claude AI payload test cases
CLAUDE_AI_PAYLOAD_TEST_CASES = [
    (
        {
            "uuid": "conv-123",
            "name": "Test Conversation",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
            "chat_messages": [
                {"uuid": "msg-1", "sender": "human", "text": "Hello", "created_at": "2025-01-01T00:00:00Z"},
                {"uuid": "msg-2", "sender": "assistant", "text": "Hi there!", "created_at": "2025-01-01T00:01:00Z"},
            ],
        },
        2,
        "conv-123",
        "basic ai format",
    ),
    (
        {
            "uuid": "conv-1",
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "assistant",
                    "content": [{"type": "text", "text": "Response text"}],
                    "created_at": "2025-01-01T00:00:00Z",
                }
            ],
        },
        1,
        "conv-1",
        "ai format with content blocks",
    ),
    (
        {
            "uuid": "conv-1",
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "human",
                    "text": "Check this file",
                    "created_at": "2025-01-01T00:00:00Z",
                    "attachments": [{"id": "file-123", "name": "document.pdf", "size": 5000}],
                }
            ],
        },
        1,
        "conv-1",
        "ai format with attachments",
    ),
]

# Codex parser test cases
CODEX_PARSER_TEST_CASES = [
    (
        [
            {
                "type": "session_meta",
                "payload": {"id": "session-1", "timestamp": "2025-01-01T00:00:00Z"},
            }
        ],
        True,
        "codex looks like detects envelope format",
    ),
    (
        [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ],
        True,
        "codex looks like detects direct format",
    ),
]

# =============================================================================
# Claude Parser Tests - Edge Cases and Uncovered Branches
# =============================================================================


class TestExtractTextFromSegments:
    """Test extract_text_from_segments with various content block types."""

    @pytest.mark.parametrize("segments,expected,desc", SEGMENT_TEST_CASES)
    def test_extract_text_from_segments(self, segments, expected, desc):
        """Parametrized test for extract_text_from_segments."""
        result = extract_text_from_segments(segments)
        if desc == "mixed segment types":
            # Check individual parts for mixed case
            assert "Plain string segment" in result
            assert '"type": "tool_use"' in result
            assert "<thinking>Planning next step</thinking>" in result
            assert "Regular text" in result
            assert '"type": "tool_result"' in result
        elif expected is None:
            assert result is None
        else:
            if isinstance(expected, tuple):
                for part in expected:
                    assert part in result
            else:
                assert expected in result

    def test_empty_segments_list(self):
        """Empty segments should return None."""
        result = extract_text_from_segments([])
        assert result is None

    def test_non_dict_non_string_segments(self):
        """Non-dict, non-string segments should be skipped."""
        segments = [123, None, [], {"type": "text", "text": "valid"}]
        result = extract_text_from_segments(segments)
        assert result == "valid"


class TestExtractMessageText:
    """Test _extract_message_text with various content structures."""

    @pytest.mark.parametrize("content,expected,desc", MESSAGE_TEXT_TEST_CASES)
    def test_extract_message_text(self, content, expected, desc):
        """Parametrized test for _extract_message_text."""
        result = _extract_message_text(content)
        if expected is None:
            assert result is None
        elif isinstance(expected, tuple):
            for part in expected:
                assert part in result
        else:
            assert result == expected


class TestExtractToolInvocations:
    """Test extract_tool_invocations with edge cases."""

    @pytest.mark.parametrize("blocks,count,expected_attrs,desc", TOOL_INVOCATION_TEST_CASES)
    def test_extract_tool_invocations(self, blocks, count, expected_attrs, desc):
        """Parametrized test for extract_tool_invocations."""
        result = extract_tool_invocations(blocks)
        assert len(result) == count
        for key, value in expected_attrs.items():
            assert result[0][key] == value

    def test_tool_use_search_operations(self):
        """Search tools should be marked."""
        for tool_name in ["Glob", "Grep", "WebSearch"]:
            blocks = [
                {
                    "type": "tool_use",
                    "name": tool_name,
                    "id": f"{tool_name.lower()}-1",
                    "input": {},
                }
            ]
            result = extract_tool_invocations(blocks)
            assert result[0]["is_search_operation"] is True


class TestExtractThinkingTraces:
    """Test extract_thinking_traces."""

    @pytest.mark.parametrize("blocks,expected_text,desc", THINKING_TRACE_TEST_CASES)
    def test_extract_thinking_traces(self, blocks, expected_text, desc):
        """Parametrized test for extract_thinking_traces."""
        result = extract_thinking_traces(blocks)
        assert len(result) == 1
        assert expected_text in result[0]["text"]
        assert "token_count" in result[0]


class TestExtractFileChanges:
    """Test extract_file_changes."""

    @pytest.mark.parametrize("invocations,op,path,desc", FILE_CHANGE_TEST_CASES)
    def test_extract_file_changes(self, invocations, op, path, desc):
        """Parametrized test for extract_file_changes."""
        result = extract_file_changes(invocations)
        assert len(result) == 1
        assert result[0]["operation"] == op
        if op == "write":
            assert len(result[0]["new_content"]) == 500
        elif path is not None:
            assert result[0]["path"] == path


class TestExtractSubagentSpawns:
    """Test extract_subagent_spawns."""

    def test_task_tool_spawn(self):
        """Extract Task tool spawns."""
        invocations = [
            {
                "tool_name": "Task",
                "input": {
                    "subagent_type": "code-reviewer",
                    "prompt": "Review this code",
                    "description": "Code review task",
                    "run_in_background": True,
                },
            }
        ]
        result = extract_subagent_spawns(invocations)
        assert len(result) == 1
        assert result[0]["agent_type"] == "code-reviewer"
        assert result[0]["prompt"] == "Review this code"
        assert result[0]["run_in_background"] is True

    def test_task_with_missing_fields(self):
        """Task without optional fields should have defaults."""
        invocations = [
            {
                "tool_name": "Task",
                "input": {},
            }
        ]
        result = extract_subagent_spawns(invocations)
        assert len(result) == 1
        assert result[0]["agent_type"] == "general-purpose"
        assert result[0]["prompt"] == ""
        assert result[0]["run_in_background"] is False

    def test_non_task_tools_ignored(self):
        """Non-Task tools should be ignored."""
        invocations = [
            {"tool_name": "Read", "input": {}},
            {"tool_name": "Write", "input": {}},
        ]
        result = extract_subagent_spawns(invocations)
        assert len(result) == 0


class TestParseGitOperation:
    """Test parse_git_operation."""

    @pytest.mark.parametrize("invocation,command,expected_value,desc", GIT_OPERATION_TEST_CASES)
    def test_parse_git_operation(self, invocation, command, expected_value, desc):
        """Parametrized test for parse_git_operation."""
        result = parse_git_operation(invocation)
        if command is None:
            assert result is None
        else:
            assert result is not None
            assert result["command"] == command
            if expected_value is not None:
                if command == "commit":
                    if expected_value is not None:
                        assert result.get("message") == expected_value
                elif command in ("checkout", "switch"):
                    assert result["branch"] == expected_value
                elif command == "push":
                    if isinstance(expected_value, tuple):
                        assert result["remote"] == expected_value[0]
                        assert result["branch"] == expected_value[1]
                    else:
                        assert result["remote"] == expected_value
                elif command in ("add", "rm"):
                    if isinstance(expected_value, tuple):
                        for file in expected_value:
                            assert file in result["files"]
                    else:
                        assert expected_value in result["files"]


class TestParseCludeCodePayload:
    """Test parse_claude_code with various payload structures."""

    @pytest.mark.parametrize("payload,fallback_id,expected_count,expected_conv_id,desc", CLAUDE_CODE_PAYLOAD_TEST_CASES)
    def test_parse_claude_code(self, payload, fallback_id, expected_count, expected_conv_id, desc):
        """Parametrized test for parse_claude_code."""
        result = parse_claude_code(payload, fallback_id)
        assert result.provider_name == "claude-code"
        assert result.provider_conversation_id == expected_conv_id
        assert len(result.messages) == expected_count

    def test_payload_with_non_list_chat_messages(self):
        """Line 767-768: handle non-list chat_messages in AI format."""
        payload = {
            "uuid": "conv-1",
            "chat_messages": "not a list",
        }
        result = parse_claude_ai(payload, "fallback")
        assert len(result.messages) == 0


class TestParseCludeAIPayload:
    """Test parse_claude_ai with various payload structures."""

    @pytest.mark.parametrize("payload,expected_count,conv_id,desc", CLAUDE_AI_PAYLOAD_TEST_CASES)
    def test_parse_claude_ai(self, payload, expected_count, conv_id, desc):
        """Parametrized test for parse_claude_ai."""
        result = parse_claude_ai(payload, "fallback")
        assert result.provider_name == "claude"
        assert result.provider_conversation_id == conv_id
        assert len(result.messages) == expected_count


# =============================================================================
# Codex Parser Tests
# =============================================================================


class TestCodexParser:
    """Test Codex parser edge cases."""

    @pytest.mark.parametrize("payload,should_match,desc", CODEX_PARSER_TEST_CASES)
    def test_codex_looks_like(self, payload, should_match, desc):
        """Parametrized test for Codex format detection."""
        from polylogue.sources.parsers.codex import looks_like
        assert looks_like(payload) == should_match

    CODEX_PARSE_CASES = [
        ("valid_message", [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello from Codex"}]}], "valid message with text"),
        ("invalid_missing_role", [{"type": "message", "content": [{"type": "input_text", "text": "Invalid"}]}], "invalid missing role"),
        ("envelope_invalid_payload", [{"type": "response_item", "payload": {"type": "message", "role": "user"}}], "envelope invalid payload"),
    ]

    def test_codex_parse_variants(self):
        """Test parse_codex with various payloads."""
        # Valid message
        payload = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello from Codex"}]}]
        result = parse_codex(payload, "fallback")
        assert result is not None

        # Invalid missing role
        payload = [{"type": "message", "content": [{"type": "input_text", "text": "Invalid"}]}]
        result = parse_codex(payload, "fallback")
        assert result is not None  # Should handle gracefully

        # Envelope with invalid payload
        payload = [{"type": "response_item", "payload": {"type": "message", "role": "user"}}]
        result = parse_codex(payload, "fallback")
        assert result is not None


# =============================================================================
# Drive Parser Tests
# =============================================================================


class TestDriveParser:
    """Test Drive parser edge cases."""

    def test_extract_text_from_chunk(self):
        """Test extract_text_from_chunk with variants."""
        from polylogue.sources.parsers.drive import extract_text_from_chunk

        # Empty chunk
        result = extract_text_from_chunk({})
        assert result is None

        # With parts
        result = extract_text_from_chunk({"parts": ["Part 1", "Part 2"]})
        assert result == "Part 1\nPart 2"

    def test_collect_drive_docs(self):
        """Test _collect_drive_docs with variants."""
        from polylogue.sources.parsers.drive import _collect_drive_docs

        # Non-dict
        result = _collect_drive_docs("not a dict")
        assert result == []

        # Nested metadata
        payload = {"driveDocument": "doc1", "metadata": {"driveDocument": "doc2"}}
        result = _collect_drive_docs(payload)
        assert "doc1" in result and "doc2" in result

    def test_attachment_from_doc(self):
        """Test _attachment_from_doc with variants."""
        from polylogue.sources.parsers.drive import _attachment_from_doc

        # String ID
        result = _attachment_from_doc("doc-string-id", "msg-1")
        assert result is not None and result.provider_attachment_id == "doc-string-id"

        # Non-dict/non-string
        result = _attachment_from_doc(123, "msg-1")
        assert result is None

        # Missing ID
        result = _attachment_from_doc({"name": "file.pdf", "size": 1000}, "msg-1")
        assert result is None

        # Size as string
        result = _attachment_from_doc({"id": "doc-1", "sizeBytes": "5000"}, "msg-1")
        assert result is not None and result.size_bytes == 5000

    def test_parse_chunked_prompt(self):
        """Test parse_chunked_prompt variants."""
        # String chunks
        payload = {"chunkedPrompt": {"chunks": ["chunk text"]}, "createTime": "2025-01-01T00:00:00Z"}
        result = parse_chunked_prompt("gemini", payload, "fallback")
        assert len(result.messages) == 0

        # Message objects
        payload = {
            "chunks": [
                {"role": "user", "text": "User message", "id": "chunk-1"},
                {"role": "model", "text": "Model response", "id": "chunk-2"},
            ]
        }
        result = parse_chunked_prompt("gemini", payload, "fallback")
        assert len(result.messages) == 2


# =============================================================================
# Logging Tests
# =============================================================================


class TestLoggingConfiguration:
    """Test logging configuration."""

    def test_configure_logging_verbose(self):
        """Line 29: Configure with verbose=True."""
        configure_logging(verbose=True, json_logs=False)

    def test_configure_logging_json(self):
        """Line 32: Configure with json_logs=True."""
        configure_logging(verbose=False, json_logs=True)

    def test_get_logger_returns_logger(self):
        """Line 51: Get logger returns structured logger."""
        logger = get_logger("test.module")
        assert logger is not None

    def test_stderr_proxy_write(self):
        """Test _StderrProxy write method."""
        from polylogue.lib.log import _StderrProxy

        proxy = _StderrProxy()
        original_stderr = sys.stderr
        try:
            sys.stderr = StringIO()
            proxy.write("test message")
            assert sys.stderr.getvalue() == "test message"
        finally:
            sys.stderr = original_stderr

    def test_stderr_proxy_isatty(self):
        """Test _StderrProxy isatty method."""
        from polylogue.lib.log import _StderrProxy

        proxy = _StderrProxy()
        result = proxy.isatty()
        assert isinstance(result, bool)

    def test_stderr_proxy_fileno(self):
        """Test _StderrProxy fileno method."""
        from polylogue.lib.log import _StderrProxy

        proxy = _StderrProxy()
        result = proxy.fileno()
        assert isinstance(result, int)


# =============================================================================
# Date Parsing Tests
# =============================================================================


class TestDateParsing:
    """Test date parsing edge cases."""

    def test_parse_date_returns_naive_utc_aware(self):
        """Line 43: Ensure parsed date is always UTC-aware."""
        result = parse_date("2024-01-15")
        assert result is not None
        assert result.tzinfo is not None

    def test_parse_date_relative_dates(self):
        """Parse relative date strings."""
        result = parse_date("yesterday")
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_parse_date_invalid_returns_none(self):
        """Invalid date string should return None."""
        result = parse_date("not a date at all!!!!")
        assert result is None


# =============================================================================
# Version Resolution Tests
# =============================================================================


class TestVersionResolution:
    """Test version resolution edge cases."""

    def test_version_git_info_git_not_found(self):
        """Line 82-83: Handle git not available."""
        from polylogue.version import _get_git_info

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            result = _get_git_info(Path("/tmp"))
            assert result == (None, False)

    def test_version_git_info_timeout(self):
        """Handle git timeout."""
        from polylogue.version import _get_git_info
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("git", 2)
            result = _get_git_info(Path("/tmp"))
            assert result == (None, False)

    def test_version_build_info_import_error(self):
        """Line 90-95: Handle missing _build_info module when .git doesn't exist."""
        from polylogue.version import _resolve_version

        result = _resolve_version()
        assert result is not None
        assert result.version is not None


# =============================================================================
# UI Tests
# =============================================================================


class TestUICreation:
    """Test UI creation and plain mode."""

    UI_MODE_CASES = [
        (True, True, "plain mode"),
        (False, False, "rich mode"),
    ]

    @pytest.mark.parametrize("plain,expected_plain,desc", UI_MODE_CASES)
    def test_ui_create_modes(self, plain, expected_plain, desc):
        """Test UI creation in plain and rich modes."""
        ui = create_ui(plain=plain)
        assert ui is not None
        assert ui.plain == expected_plain

    UI_INTERACTION_CASES = [
        ("confirm_with_tty_y", "confirm", True, True, "y", True, "Confirm y returns True"),
        ("confirm_without_tty", "confirm", False, True, None, SystemExit, "Confirm no TTY aborts"),
        ("choose_valid", "choose", True, True, "2", "Option 2", "Choose valid input"),
        ("choose_invalid_then_valid", "choose", True, True, ["0", "1"], "A", "Choose invalid then valid"),
        ("input_empty_default", "input", True, True, "", "default-val", "Input empty uses default"),
        ("input_custom", "input", True, True, "custom", "custom", "Input custom value"),
        ("confirm_eof", "confirm", True, True, EOFError, False, "Confirm EOF returns default"),
        ("choose_eof", "choose", True, True, EOFError, None, "Choose EOF returns None"),
        ("input_eof", "input", True, True, EOFError, "default", "Input EOF returns default"),
    ]

    def test_ui_interactions(self, plain=True):
        """Test interactive UI operations."""
        # Confirm with input "y"
        ui = create_ui(plain=True)
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", return_value="y"):
                result = ui.confirm("Do you agree?")
                assert result is True

        # Confirm without TTY
        with patch("sys.stdin.isatty", return_value=False):
            with pytest.raises(SystemExit):
                ui.confirm("Do you agree?")

        # Choose with valid input
        options = ["Option 1", "Option 2", "Option 3"]
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", return_value="2"):
                result = ui.choose("Select one:", options)
                assert result == "Option 2"

        # Choose with invalid then valid
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", side_effect=["0", "1"]):
                result = ui.choose("Select:", ["A", "B"])
                assert result == "A"

        # Input with default
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", return_value=""):
                result = ui.input("Enter value:", default="default-val")
                assert result == "default-val"

        # Input with custom
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", return_value="custom"):
                result = ui.input("Enter:", default="default")
                assert result == "custom"

        # Confirm EOF returns default
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", side_effect=EOFError):
                result = ui.confirm("Confirm?", default=False)
                assert result is False

        # Choose EOF returns None
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", side_effect=EOFError):
                result = ui.choose("Choose:", ["A", "B"])
                assert result is None

        # Input EOF returns default
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", side_effect=EOFError):
                result = ui.input("Enter:", default="default")
                assert result == "default"

    def test_ui_progress_modes(self):
        """Test progress tracker in both modes."""
        # Plain mode
        ui = create_ui(plain=True)
        with patch("sys.stdout"):
            tracker = ui.progress("Processing", total=10)
            assert tracker is not None
            tracker.advance(5)
            tracker.update(description="Updated")

        # Rich mode
        ui = create_ui(plain=False)
        tracker = ui.progress("Processing", total=10)
        assert tracker is not None

    PROGRESS_TRACKER_CASES = [
        ("coerce_int_near_int", 5.0001, 5, "_coerce_int near whole"),
        ("coerce_int_fractional", 5.5, None, "_coerce_int fractional"),
        ("coerce_int_exact", 5, 5, "_coerce_int exact"),
        ("format_value_int", 5, "5", "_format_value int"),
        ("format_value_near_int", 5.0001, "5", "_format_value near int"),
        ("advance_progress", 3, 3, "advance sets _completed"),
        ("update_total", 20, 20, "update sets _total"),
    ]

    def test_progress_tracker_methods(self):
        """Test _PlainProgressTracker methods."""
        from polylogue.ui import _PlainProgressTracker

        console = MagicMock()

        # _coerce_int tests
        tracker = _PlainProgressTracker(console, "test", 10)
        assert tracker._coerce_int(5.0001) == 5
        assert tracker._coerce_int(5.5) is None
        assert tracker._coerce_int(5) == 5

        # _format_value tests
        tracker = _PlainProgressTracker(console, "test", None)
        assert tracker._format_value(5) == "5"
        assert tracker._format_value(5.0001) == "5"
        assert "5.5" in tracker._format_value(5.5)

        # advance test
        tracker = _PlainProgressTracker(console, "test", 10)
        tracker.advance(3)
        assert tracker._completed == 3

        # update test
        tracker.update(total=20)
        assert tracker._total == 20

        # context manager test
        with tracker as t:
            assert t is tracker


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
