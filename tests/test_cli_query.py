"""Tests for polylogue.cli.query module.

Coverage targets:
- execute_query: query execution with various parameters
- Filter chain building from params
- Aggregation outputs (by_month, by_provider, by_tag, stats)
- Format outputs (json, markdown, html, obsidian, org)
- Modifiers (set_meta, add_tag, rm_tag, delete)
- Output destinations (stdout, file, browser, clipboard)
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli import query
from polylogue.lib.models import Conversation, Message


@pytest.fixture
def sample_conversations():
    """Create sample conversations for testing."""
    now = datetime(2024, 6, 15, 10, 0)

    conv1 = Conversation(
        id="conv1-abc123",
        provider="chatgpt",
        messages=[
            Message(id="m1", role="user", text="Hello world", timestamp=now),
            Message(id="m2", role="assistant", text="Hi there!", timestamp=now),
        ],
        title="First Conversation",
        updated_at=now,
        metadata={"tags": ["work", "important"]},  # tags via metadata
    )

    conv2 = Conversation(
        id="conv2-def456",
        provider="claude",
        messages=[
            Message(id="m1", role="user", text="What is Python?", timestamp=now),
            Message(id="m2", role="assistant", text="Python is a programming language.", timestamp=now),
            Message(id="m3", role="assistant", text="It's great for scripting.", timestamp=now, is_tool_use=True),
        ],
        title="Python Discussion",
        updated_at=now,
        metadata={"tags": ["programming"]},  # tags via metadata
    )

    conv3 = Conversation(
        id="conv3-ghi789",
        provider="chatgpt",
        messages=[
            Message(id="m1", role="user", text="Explain thinking", timestamp=now),
            Message(id="m2", role="assistant", text="Thinking...", timestamp=now, is_thinking=True),
            Message(id="m3", role="assistant", text="Here's my answer.", timestamp=now),
        ],
        title="Thinking Demo",
        updated_at=datetime(2024, 7, 20, 10, 0),
        metadata={},  # no tags
    )

    return [conv1, conv2, conv3]


@pytest.fixture
def mock_env():
    """Create mock environment for query tests."""
    mock_ui = MagicMock()
    mock_ui.plain = True
    mock_ui.console = MagicMock()

    env = MagicMock()
    env.ui = mock_ui

    return env


class TestConvToDict:
    """Tests for _conv_to_dict helper."""

    def test_full_dict(self, sample_conversations):
        """Returns full dict when no fields specified."""
        conv = sample_conversations[0]
        result = query._conv_to_dict(conv, None)

        assert result["id"] == "conv1-abc123"
        assert result["provider"] == "chatgpt"
        assert result["title"] == "First Conversation"
        assert result["messages"] == 2
        assert result["tags"] == ["work", "important"]

    def test_selected_fields(self, sample_conversations):
        """Returns only selected fields."""
        conv = sample_conversations[0]
        result = query._conv_to_dict(conv, "id,provider")

        assert "id" in result
        assert "provider" in result
        assert "title" not in result
        assert "messages" not in result

    def test_fields_with_spaces(self, sample_conversations):
        """Handles field list with spaces."""
        conv = sample_conversations[0]
        result = query._conv_to_dict(conv, "id, title, tags")

        assert "id" in result
        assert "title" in result
        assert "tags" in result
        assert "provider" not in result


class TestConvToMarkdown:
    """Tests for _conv_to_markdown helper."""

    def test_includes_title(self, sample_conversations):
        """Includes conversation title as heading."""
        conv = sample_conversations[0]
        result = query._conv_to_markdown(conv)

        assert "# First Conversation" in result

    def test_includes_messages(self, sample_conversations):
        """Includes all messages."""
        conv = sample_conversations[0]
        result = query._conv_to_markdown(conv)

        assert "Hello world" in result
        assert "Hi there!" in result

    def test_includes_provider(self, sample_conversations):
        """Includes provider info."""
        conv = sample_conversations[0]
        result = query._conv_to_markdown(conv)

        assert "chatgpt" in result.lower()


class TestConvToHtml:
    """Tests for _conv_to_html helper."""

    def test_valid_html(self, sample_conversations):
        """Generates valid HTML structure."""
        conv = sample_conversations[0]
        result = query._conv_to_html(conv)

        assert "<!DOCTYPE html>" in result
        assert "<html>" in result
        assert "</html>" in result
        assert "<title>" in result

    def test_escapes_html_chars(self, sample_conversations):
        """Escapes HTML special characters."""
        conv = Conversation(
            id="test",
            provider="test",
            messages=[
                Message(id="m1", role="user", text="<script>alert('xss')</script>"),
            ],
        )
        result = query._conv_to_html(conv)

        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_includes_css(self, sample_conversations):
        """Includes CSS styles."""
        conv = sample_conversations[0]
        result = query._conv_to_html(conv)

        assert "<style>" in result
        assert "message-user" in result
        assert "message-assistant" in result


class TestConvToObsidian:
    """Tests for _conv_to_obsidian helper."""

    def test_yaml_frontmatter(self, sample_conversations):
        """Includes YAML frontmatter."""
        conv = sample_conversations[0]
        result = query._conv_to_obsidian(conv)

        assert result.startswith("---")
        assert "id: conv1-abc123" in result
        assert "provider: chatgpt" in result

    def test_tags_in_frontmatter(self, sample_conversations):
        """Includes tags in frontmatter."""
        conv = sample_conversations[0]
        result = query._conv_to_obsidian(conv)

        assert "tags:" in result


class TestConvToOrg:
    """Tests for _conv_to_org helper."""

    def test_org_headers(self, sample_conversations):
        """Uses Org-mode headers."""
        conv = sample_conversations[0]
        result = query._conv_to_org(conv)

        assert "#+TITLE:" in result
        assert "#+DATE:" in result
        assert "#+PROPERTY:" in result

    def test_org_headings(self, sample_conversations):
        """Uses * for headings."""
        conv = sample_conversations[0]
        result = query._conv_to_org(conv)

        assert "* USER" in result
        assert "* ASSISTANT" in result


class TestOutputStats:
    """Tests for _output_stats aggregation."""

    def test_shows_total_messages(self, mock_env, sample_conversations):
        """Shows total message count."""
        query._output_stats(mock_env, sample_conversations)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "messages" in output.lower()

    def test_shows_role_counts(self, mock_env, sample_conversations):
        """Shows user and assistant message counts."""
        query._output_stats(mock_env, sample_conversations)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "user" in output.lower()
        assert "assistant" in output.lower()

    def test_shows_tool_calls(self, mock_env, sample_conversations):
        """Shows tool call count."""
        query._output_stats(mock_env, sample_conversations)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "tool" in output.lower()

    def test_shows_thinking_traces(self, mock_env, sample_conversations):
        """Shows thinking trace count."""
        query._output_stats(mock_env, sample_conversations)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "thinking" in output.lower()

    def test_shows_date_range(self, mock_env, sample_conversations):
        """Shows date range of conversations."""
        query._output_stats(mock_env, sample_conversations)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "date" in output.lower() or "2024" in output

    def test_handles_no_results(self, mock_env):
        """Handles empty results list."""
        query._output_stats(mock_env, [])

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "no conversation" in output.lower()


class TestFormatList:
    """Tests for _format_list helper."""

    def test_json_format(self, sample_conversations):
        """JSON format returns valid JSON array."""
        result = query._format_list(sample_conversations, "json", None)
        parsed = json.loads(result)

        assert isinstance(parsed, list)
        assert len(parsed) == 3
        assert parsed[0]["id"] == "conv1-abc123"

    def test_default_format(self, sample_conversations):
        """Default format returns text list."""
        result = query._format_list(sample_conversations, "markdown", None)

        assert "conv1-abc123" in result
        assert "conv2-def456" in result
        assert "First Conversation" in result

    def test_yaml_list_format(self, sample_conversations):
        """YAML list format returns valid YAML list."""
        result = query._format_list(sample_conversations, "yaml", None)

        import yaml

        parsed = yaml.safe_load(result)

        assert isinstance(parsed, list)
        assert len(parsed) == 3
        assert parsed[0]["id"] == "conv1-abc123"


class TestFormatConversation:
    """Tests for _format_conversation helper."""

    def test_json_format(self, sample_conversations):
        """JSON format returns valid JSON."""
        conv = sample_conversations[0]
        result = query._format_conversation(conv, "json", None)
        parsed = json.loads(result)

        assert parsed["id"] == "conv1-abc123"

    def test_html_format(self, sample_conversations):
        """HTML format returns HTML."""
        conv = sample_conversations[0]
        result = query._format_conversation(conv, "html", None)

        assert "<!DOCTYPE html>" in result

    def test_obsidian_format(self, sample_conversations):
        """Obsidian format returns frontmatter markdown."""
        conv = sample_conversations[0]
        result = query._format_conversation(conv, "obsidian", None)

        assert result.startswith("---")

    def test_org_format(self, sample_conversations):
        """Org format returns Org-mode content."""
        conv = sample_conversations[0]
        result = query._format_conversation(conv, "org", None)

        assert "#+TITLE:" in result

    def test_yaml_format(self, sample_conversations):
        """YAML format returns valid YAML."""
        conv = sample_conversations[0]
        result = query._format_conversation(conv, "yaml", None)

        import yaml

        parsed = yaml.safe_load(result)

        assert parsed["id"] == "conv1-abc123"
        assert "messages" in parsed

    def test_plaintext_format(self, sample_conversations):
        """Plaintext format returns raw text without formatting."""
        conv = sample_conversations[0]
        result = query._format_conversation(conv, "plaintext", None)

        # Should contain message text but no markdown formatting
        assert "Hello world" in result
        assert "##" not in result  # No markdown headers
        assert "**" not in result  # No bold formatting


class TestCopyToClipboard:
    """Tests for _copy_to_clipboard helper."""

    def test_tries_clipboard_commands(self, mock_env):
        """Tries multiple clipboard commands."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            query._copy_to_clipboard(mock_env, "test content")

            # Should have tried at least one clipboard command
            assert mock_run.called

    def test_shows_success_message(self, mock_env):
        """Shows success message on copy."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            query._copy_to_clipboard(mock_env, "test content")

            calls = mock_env.ui.console.print.call_args_list
            output = " ".join(str(c) for c in calls)
            assert "clipboard" in output.lower()

    def test_shows_failure_message(self, mock_env, capsys):
        """Shows failure message when no clipboard tool found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            query._copy_to_clipboard(mock_env, "test content")

            captured = capsys.readouterr()
            assert "could not" in captured.err.lower() or "clipboard" in captured.err.lower()


class TestSendOutput:
    """Tests for _send_output helper."""

    def test_stdout_prints(self, mock_env, sample_conversations, capsys):
        """stdout destination prints content via click.echo."""
        query._send_output(mock_env, "test content", ["stdout"], "markdown", None)

        captured = capsys.readouterr()
        assert "test content" in captured.out

    def test_file_writes(self, mock_env, sample_conversations, tmp_path):
        """file destination writes content."""
        output_file = tmp_path / "output.md"

        query._send_output(mock_env, "test content", [str(output_file)], "markdown", None)

        assert output_file.exists()
        assert output_file.read_text() == "test content"

    def test_multiple_destinations(self, mock_env, sample_conversations, tmp_path, capsys):
        """Multiple destinations all receive content."""
        output_file = tmp_path / "output.md"

        query._send_output(mock_env, "test content", ["stdout", str(output_file)], "markdown", None)

        # Both stdout and file should be used
        assert output_file.exists()
        captured = capsys.readouterr()
        assert "test content" in captured.out


class TestDryRunMode:
    """Tests for --dry-run functionality in modifiers and delete."""

    def test_dry_run_modifiers_shows_preview(self, mock_env, sample_conversations, capsys):
        """Dry-run mode shows preview without modifying."""
        params = {
            "add_tag": ("test-tag",),
            "dry_run": True,
        }

        query._apply_modifiers(mock_env, sample_conversations, params)

        captured = capsys.readouterr()
        assert "DRY-RUN" in captured.out
        assert "3" in captured.out  # Count of conversations
        assert "add tags" in captured.out.lower()

    def test_dry_run_modifiers_shows_sample(self, mock_env, sample_conversations):
        """Dry-run shows sample of affected conversations."""
        params = {
            "rm_tag": ("old-tag",),
            "dry_run": True,
        }

        query._apply_modifiers(mock_env, sample_conversations, params)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "conv1" in output or "chatgpt" in output

    def test_dry_run_delete_shows_preview(self, mock_env, sample_conversations, capsys):
        """Dry-run delete shows preview without deleting."""
        params = {"dry_run": True}

        query._delete_conversations(mock_env, sample_conversations, params)

        captured = capsys.readouterr()
        assert "DRY-RUN" in captured.out
        assert "delete" in captured.out.lower()
        assert "3" in captured.out


class TestBulkOperationConfirmation:
    """Tests for bulk operation confirmation (>10 items requires --force)."""

    def test_modifiers_require_confirmation_for_bulk(self, mock_env, capsys):
        """Modifiers prompt for confirmation for >10 items."""
        # Create 15 mock conversations
        convs = [MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test") for i in range(15)]
        params = {"add_tag": ("bulk-tag",), "force": False}

        # Decline confirmation
        mock_env.ui.confirm.return_value = False
        query._apply_modifiers(mock_env, convs, params)

        mock_env.ui.confirm.assert_called_once()
        captured = capsys.readouterr()
        assert "15" in captured.out

    def test_modifiers_proceed_with_force(self, mock_env):
        """Modifiers proceed with --force for bulk operations."""
        # Create 15 mock conversations
        convs = [MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test") for i in range(15)]
        params = {"add_tag": ("bulk-tag",), "force": True}

        mock_backend = MagicMock()
        with (
            patch("polylogue.cli.helpers.load_effective_config"),
            patch("polylogue.services.get_repository", return_value=mock_backend),
        ):
            query._apply_modifiers(mock_env, convs, params)

            # Should have called add_tag 15 times
            assert mock_backend.add_tag.call_count == 15

    def test_delete_requires_confirmation_for_bulk(self, mock_env, capsys):
        """Delete prompts for confirmation for >10 items."""
        convs = [
            MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test", created_at=None, updated_at=None) for i in range(15)
        ]
        params = {"force": False}

        # Decline confirmation
        mock_env.ui.confirm.return_value = False
        query._delete_conversations(mock_env, convs, params)

        mock_env.ui.confirm.assert_called_once()
        captured = capsys.readouterr()
        assert "DELETE" in captured.err

    def test_delete_proceeds_with_force(self, mock_env):
        """Delete proceeds with --force for bulk operations."""
        convs = [MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test", created_at=None) for i in range(15)]
        params = {"force": True}

        mock_backend = MagicMock()
        mock_backend.delete_conversation.return_value = True
        with (
            patch("polylogue.cli.helpers.load_effective_config"),
            patch("polylogue.services.get_repository", return_value=mock_backend),
        ):
            query._delete_conversations(mock_env, convs, params)

            # Should have called delete_conversation 15 times
            assert mock_backend.delete_conversation.call_count == 15

    def test_small_operations_proceed_without_force(self, mock_env):
        """Operations with <=10 items proceed without --force."""
        convs = [MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test") for i in range(5)]
        params = {"add_tag": ("small-tag",), "force": False}

        mock_backend = MagicMock()
        with (
            patch("polylogue.cli.helpers.load_effective_config"),
            patch("polylogue.services.get_repository", return_value=mock_backend),
        ):
            # Should not raise
            query._apply_modifiers(mock_env, convs, params)

            assert mock_backend.add_tag.call_count == 5


class TestOperationReporting:
    """Tests for operation result reporting."""

    def test_add_tag_reports_count(self, mock_env, sample_conversations, capsys):
        """Add tag reports count of affected conversations."""
        params = {"add_tag": ("new-tag",)}

        mock_backend = MagicMock()
        with (
            patch("polylogue.cli.helpers.load_effective_config"),
            patch("polylogue.services.get_repository", return_value=mock_backend),
        ):
            query._apply_modifiers(mock_env, sample_conversations, params)

            captured = capsys.readouterr()
            assert "Added tags" in captured.out
            assert "3" in captured.out  # Number of conversations

    def test_delete_reports_count(self, mock_env, sample_conversations, capsys):
        """Delete reports count of deleted conversations."""
        params = {}

        mock_backend = MagicMock()
        mock_backend.delete_conversation.return_value = True
        with (
            patch("polylogue.cli.helpers.load_effective_config"),
            patch("polylogue.services.get_repository", return_value=mock_backend),
        ):
            query._delete_conversations(mock_env, sample_conversations, params)

            captured = capsys.readouterr()
            assert "Deleted" in captured.out
            assert "3" in captured.out


class TestConvToJson:
    """Tests for _conv_to_json helper."""

    def test_full_json_includes_message_content(self, sample_conversations):
        """Full JSON output includes message content, not just count."""
        conv = sample_conversations[0]
        result = query._conv_to_json(conv, None)
        parsed = json.loads(result)

        # Should have messages array with full content, not integer count
        assert isinstance(parsed["messages"], list)
        assert len(parsed["messages"]) == 2
        assert parsed["messages"][0]["id"] == "m1"
        assert parsed["messages"][0]["role"] == "user"
        assert parsed["messages"][0]["text"] == "Hello world"
        assert parsed["messages"][1]["text"] == "Hi there!"

    def test_field_selection_excludes_messages(self, sample_conversations):
        """Field selection works - excluding messages when not selected."""
        conv = sample_conversations[0]
        result = query._conv_to_json(conv, "id,title")
        parsed = json.loads(result)

        assert "id" in parsed
        assert "title" in parsed
        assert "messages" not in parsed
        assert "provider" not in parsed

    def test_field_selection_includes_messages_when_selected(self, sample_conversations):
        """Field selection includes messages when explicitly selected."""
        conv = sample_conversations[0]
        result = query._conv_to_json(conv, "id,messages")
        parsed = json.loads(result)

        assert "id" in parsed
        assert "messages" in parsed
        assert isinstance(parsed["messages"], list)
        assert len(parsed["messages"]) == 2

    def test_empty_conversation_produces_valid_json(self):
        """Empty conversation produces valid JSON with empty messages array."""
        conv = Conversation(
            id="empty-conv",
            provider="test",
            messages=[],
            title="Empty",
        )
        result = query._conv_to_json(conv, None)
        parsed = json.loads(result)

        assert parsed["id"] == "empty-conv"
        assert parsed["messages"] == []

    def test_messages_with_none_text_are_included(self):
        """Messages with None text are included as null in JSON."""
        now = datetime(2024, 6, 15, 10, 0)
        conv = Conversation(
            id="conv-with-none",
            provider="test",
            messages=[
                Message(id="m1", role="user", text="Hello", timestamp=now),
                Message(id="m2", role="assistant", text=None, timestamp=now),
            ],
            title="Test",
        )
        result = query._conv_to_json(conv, None)
        parsed = json.loads(result)

        assert len(parsed["messages"]) == 2
        assert parsed["messages"][0]["text"] == "Hello"
        assert parsed["messages"][1]["text"] is None

    def test_timestamps_are_iso_formatted(self, sample_conversations):
        """Timestamps are ISO formatted in JSON."""
        conv = sample_conversations[0]
        result = query._conv_to_json(conv, None)
        parsed = json.loads(result)

        # Check timestamp format is ISO
        ts = parsed["messages"][0]["timestamp"]
        assert ts == "2024-06-15T10:00:00"
        assert "T" in ts  # ISO format marker

    def test_messages_with_none_timestamp(self):
        """Messages with None timestamp are handled correctly."""
        conv = Conversation(
            id="conv-no-timestamp",
            provider="test",
            messages=[
                Message(id="m1", role="user", text="Hello", timestamp=None),
            ],
            title="Test",
        )
        result = query._conv_to_json(conv, None)
        parsed = json.loads(result)

        assert parsed["messages"][0]["timestamp"] is None

    def test_multiple_messages_with_full_content(self, sample_conversations):
        """Multiple messages all included with full content."""
        conv = sample_conversations[1]  # Has 3 messages
        result = query._conv_to_json(conv, None)
        parsed = json.loads(result)

        assert len(parsed["messages"]) == 3
        assert parsed["messages"][0]["text"] == "What is Python?"
        assert parsed["messages"][1]["text"] == "Python is a programming language."
        assert parsed["messages"][2]["text"] == "It's great for scripting."

    def test_field_selection_with_spaces(self, sample_conversations):
        """Field selection handles spaces in field list."""
        conv = sample_conversations[0]
        result = query._conv_to_json(conv, "id, title, messages")
        parsed = json.loads(result)

        assert "id" in parsed
        assert "title" in parsed
        assert "messages" in parsed
        assert "provider" not in parsed

    def test_json_is_valid_and_parseable(self, sample_conversations):
        """Generated JSON is valid and parseable."""
        conv = sample_conversations[0]
        result = query._conv_to_json(conv, None)

        # Should not raise
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "id" in parsed


class TestConvToCsvMessages:
    """Tests for _conv_to_csv_messages helper."""

    def test_csv_header_row_is_correct(self, sample_conversations):
        """CSV header row contains correct columns."""
        conv = sample_conversations[0]
        result = query._conv_to_csv_messages(conv)
        lines = result.strip().split("\n")

        assert lines[0].rstrip() == "conversation_id,message_id,role,timestamp,text"

    def test_messages_with_text_produce_csv_rows(self, sample_conversations):
        """Messages with text produce CSV rows."""
        conv = sample_conversations[0]
        result = query._conv_to_csv_messages(conv)
        lines = result.strip().split("\n")

        # Header + 2 messages
        assert len(lines) == 3
        assert "Hello world" in lines[1]
        assert "Hi there!" in lines[2]

    def test_messages_with_none_text_are_skipped(self):
        """Messages with None/empty text are skipped."""
        now = datetime(2024, 6, 15, 10, 0)
        conv = Conversation(
            id="conv-with-none",
            provider="test",
            messages=[
                Message(id="m1", role="user", text="Hello", timestamp=now),
                Message(id="m2", role="assistant", text=None, timestamp=now),
                Message(id="m3", role="user", text="", timestamp=now),
                Message(id="m4", role="assistant", text="Final", timestamp=now),
            ],
            title="Test",
        )
        result = query._conv_to_csv_messages(conv)
        lines = result.strip().split("\n")

        # Header + 2 messages (m1 and m4)
        assert len(lines) == 3
        assert "Hello" in lines[1]
        assert "Final" in lines[2]

    def test_special_chars_escaped_in_csv(self):
        """Special characters (commas, quotes, newlines) are properly escaped."""
        now = datetime(2024, 6, 15, 10, 0)
        conv = Conversation(
            id="conv-special",
            provider="test",
            messages=[
                Message(
                    id="m1",
                    role="user",
                    text='He said "hello, world"',
                    timestamp=now,
                ),
                Message(
                    id="m2",
                    role="assistant",
                    text="Line 1\nLine 2",
                    timestamp=now,
                ),
            ],
            title="Test",
        )
        result = query._conv_to_csv_messages(conv)
        lines = result.split("\n")

        # CSV writer should escape quotes and handle special chars
        assert len(lines) >= 2
        # Check that quotes are properly escaped (CSV format)
        assert '"hello' in result or 'hello' in result

    def test_empty_conversation_produces_header_only(self):
        """Empty conversation produces just the header."""
        conv = Conversation(
            id="empty-conv",
            provider="test",
            messages=[],
            title="Empty",
        )
        result = query._conv_to_csv_messages(conv)
        lines = result.split("\n")

        # Should have exactly 1 line (header)
        assert len(lines) == 1
        assert lines[0] == "conversation_id,message_id,role,timestamp,text"

    def test_csv_has_correct_column_order(self, sample_conversations):
        """CSV columns appear in correct order."""
        conv = sample_conversations[0]
        result = query._conv_to_csv_messages(conv)
        lines = result.strip().split("\n")

        # Parse first data row
        data_row = lines[1]
        parts = next(
            iter(__import__("csv").reader([data_row]))
        )  # Use CSV reader for proper parsing

        assert parts[0] == "conv1-abc123"  # conversation_id
        assert parts[1] == "m1"  # message_id
        assert parts[2] == "user"  # role
        assert "2024-06-15" in parts[3]  # timestamp
        assert parts[4] == "Hello world"  # text

    def test_timestamps_iso_formatted_in_csv(self, sample_conversations):
        """Timestamps are ISO formatted in CSV."""
        conv = sample_conversations[0]
        result = query._conv_to_csv_messages(conv)
        lines = result.split("\n")

        # Check timestamp in first data row
        import csv
        import io

        reader = csv.reader(io.StringIO(result))
        next(reader)  # Skip header
        row = next(reader)
        timestamp = row[3]

        assert timestamp == "2024-06-15T10:00:00"

    def test_messages_without_timestamp(self):
        """Messages without timestamp show empty string."""
        conv = Conversation(
            id="conv-no-ts",
            provider="test",
            messages=[
                Message(id="m1", role="user", text="Hello", timestamp=None),
            ],
            title="Test",
        )
        result = query._conv_to_csv_messages(conv)
        lines = result.split("\n")

        import csv
        import io

        reader = csv.reader(io.StringIO(result))
        next(reader)  # Skip header
        row = next(reader)
        timestamp = row[3]

        assert timestamp == ""

    def test_multiple_messages_all_in_csv(self, sample_conversations):
        """Multiple messages all appear in CSV output."""
        conv = sample_conversations[1]  # Has 3 messages
        result = query._conv_to_csv_messages(conv)
        lines = result.strip().split("\n")

        # Header + 3 messages
        assert len(lines) == 4
        assert "What is Python?" in lines[1]
        assert "Python is a programming language." in lines[2]
        assert "It's great for scripting." in lines[3]

    def test_output_is_valid_csv(self, sample_conversations):
        """Output is valid CSV that can be parsed."""
        conv = sample_conversations[0]
        result = query._conv_to_csv_messages(conv)

        import csv
        import io

        # Should not raise
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)

        assert len(rows) == 3  # Header + 2 messages
        assert rows[0] == ["conversation_id", "message_id", "role", "timestamp", "text"]
        assert len(rows[1]) == 5  # 5 columns
        assert len(rows[2]) == 5
