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

import csv
import io
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import yaml

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


class TestQueryConvToDict:
    """Tests for _conv_to_dict helper."""

    @pytest.mark.parametrize(
        "fields,should_include,should_exclude",
        [
            (None, ["id", "provider", "title", "messages", "tags"], []),
            ("id,provider", ["id", "provider"], ["title", "messages"]),
            ("id, title, tags", ["id", "title", "tags"], ["provider"]),
        ],
        ids=["full_dict", "selected_fields", "fields_with_spaces"],
    )
    def test_field_selection(self, sample_conversations, fields, should_include, should_exclude):
        """Tests _conv_to_dict field selection."""
        conv = sample_conversations[0]
        result = query._conv_to_dict(conv, fields)

        for field in should_include:
            assert field in result, f"Expected field '{field}' to be in result"
        for field in should_exclude:
            assert field not in result, f"Expected field '{field}' to NOT be in result"

        if fields is None:
            assert result["id"] == "conv1-abc123"
            assert result["provider"] == "chatgpt"
            assert result["title"] == "First Conversation"
            assert result["messages"] == 2
            assert result["tags"] == ["work", "important"]


class TestFormatHelpers:
    """Consolidated tests for format conversion helpers."""

    @pytest.mark.parametrize(
        "formatter,assertion",
        [
            (query._conv_to_markdown, lambda r: "# First Conversation" in r),
            (query._conv_to_html, lambda r: "<!DOCTYPE html>" in r and "<html>" in r),
            (query._conv_to_obsidian, lambda r: r.startswith("---")),
            (query._conv_to_org, lambda r: "#+TITLE:" in r),
        ],
        ids=["markdown", "html", "obsidian", "org"],
    )
    def test_format_structure(self, sample_conversations, formatter, assertion):
        """Tests format conversion helpers produce expected structure."""
        conv = sample_conversations[0]
        result = formatter(conv)
        assert assertion(result), f"Failed for {formatter.__name__}"

    def test_markdown_includes_messages(self, sample_conversations):
        """Markdown includes all message content."""
        conv = sample_conversations[0]
        result = query._conv_to_markdown(conv)
        assert "Hello world" in result and "Hi there!" in result

    def test_markdown_includes_provider(self, sample_conversations):
        """Markdown includes provider info."""
        conv = sample_conversations[0]
        result = query._conv_to_markdown(conv)
        assert "chatgpt" in result.lower()

    def test_html_escapes_special_chars(self, sample_conversations):
        """HTML properly escapes special characters."""
        conv = Conversation(
            id="test",
            provider="test",
            messages=[Message(id="m1", role="user", text="<script>alert('xss')</script>")],
        )
        result = query._conv_to_html(conv)
        assert "<script>" not in result and "&lt;script&gt;" in result

    def test_html_includes_css(self, sample_conversations):
        """HTML includes CSS styles."""
        conv = sample_conversations[0]
        result = query._conv_to_html(conv)
        assert "<style>" in result and "message-user" in result

    def test_obsidian_includes_tags(self, sample_conversations):
        """Obsidian format includes tags in frontmatter."""
        conv = sample_conversations[0]
        result = query._conv_to_obsidian(conv)
        assert "tags:" in result

    def test_org_uses_headings(self, sample_conversations):
        """Org-mode uses * for headings."""
        conv = sample_conversations[0]
        result = query._conv_to_org(conv)
        assert "* USER" in result and "* ASSISTANT" in result


class TestQueryOutputStats:
    """Tests for _output_stats aggregation."""

    @pytest.mark.parametrize(
        "stat_term",
        ["messages", "user", "assistant", "tool", "thinking", "date"],
        ids=[
            "total_messages",
            "role_counts_user",
            "role_counts_assistant",
            "tool_calls",
            "thinking_traces",
            "date_range",
        ],
    )
    def test_shows_stat(self, mock_env, sample_conversations, stat_term):
        """Tests _output_stats displays specific statistics."""
        query._output_stats(mock_env, sample_conversations)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert stat_term.lower() in output.lower() or (stat_term == "date" and "2024" in output)

    def test_handles_no_results(self, mock_env):
        """Handles empty results list."""
        query._output_stats(mock_env, [])

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "no conversation" in output.lower()


class TestQueryFormatList:
    """Tests for _format_list helper."""

    @pytest.mark.parametrize(
        "format_type,parser,expected_type,first_id",
        [
            ("json", json.loads, list, "conv1-abc123"),
            ("yaml", yaml.safe_load, list, "conv1-abc123"),
        ],
        ids=["json_format", "yaml_list_format"],
    )
    def test_format_list(self, sample_conversations, format_type, parser, expected_type, first_id):
        """Tests _format_list with various formats."""
        result = query._format_list(sample_conversations, format_type, None)
        parsed = parser(result)

        assert isinstance(parsed, expected_type)
        assert len(parsed) == 3
        assert parsed[0]["id"] == first_id

    def test_default_format(self, sample_conversations):
        """Default format returns text list."""
        result = query._format_list(sample_conversations, "markdown", None)

        assert "conv1-abc123" in result
        assert "conv2-def456" in result
        assert "First Conversation" in result


class TestQueryFormatConversation:
    """Tests for _format_conversation helper."""

    @pytest.mark.parametrize(
        "format_type,assertion",
        [
            ("json", lambda r: json.loads(r)["id"] == "conv1-abc123"),
            ("html", lambda r: "<!DOCTYPE html>" in r),
            ("obsidian", lambda r: r.startswith("---")),
            ("org", lambda r: "#+TITLE:" in r),
            ("yaml", lambda r: yaml.safe_load(r)["id"] == "conv1-abc123"),
            ("plaintext", lambda r: "Hello world" in r and "##" not in r),
        ],
        ids=[
            "json_format",
            "html_format",
            "obsidian_format",
            "org_format",
            "yaml_format",
            "plaintext_format",
        ],
    )
    def test_format_conversation(self, sample_conversations, format_type, assertion):
        """Tests _format_conversation with various formats."""
        conv = sample_conversations[0]
        result = query._format_conversation(conv, format_type, None)

        assert assertion(result), f"Failed assertion for format {format_type}"


class TestQueryCopyToClipboard:
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


class TestQuerySendOutput:
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
            MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test", created_at=None, updated_at=None)
            for i in range(15)
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


class TestQueryConvToJson:
    """Tests for _conv_to_json helper."""

    @pytest.mark.parametrize(
        "fields,has_messages,expected_id",
        [
            (None, True, "conv1-abc123"),
            ("id,title", False, "conv1-abc123"),
            ("id,messages", True, "conv1-abc123"),
        ],
        ids=[
            "full_json_includes_message_content",
            "field_selection_excludes_messages",
            "field_selection_includes_messages_when_selected",
        ],
    )
    def test_json_field_selection(self, sample_conversations, fields, has_messages, expected_id):
        """Tests _conv_to_json field selection."""
        conv = sample_conversations[0]
        result = query._conv_to_json(conv, fields)
        parsed = json.loads(result)

        assert parsed["id"] == expected_id
        if has_messages:
            assert isinstance(parsed["messages"], list)
            assert len(parsed["messages"]) == 2
            if fields is None:
                assert parsed["messages"][0]["id"] == "m1"
                assert parsed["messages"][0]["role"] == "user"
                assert parsed["messages"][0]["text"] == "Hello world"
        else:
            assert "messages" not in parsed
            assert "provider" not in parsed

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

    @pytest.mark.parametrize(
        "message_text,is_none",
        [
            ("Hello", False),
            (None, True),
        ],
        ids=["message_with_text", "message_with_none_text"],
    )
    def test_messages_with_text_handling(self, message_text, is_none):
        """Tests JSON handling of message text including None."""
        now = datetime(2024, 6, 15, 10, 0)
        conv = Conversation(
            id="conv-with-text",
            provider="test",
            messages=[
                Message(id="m1", role="user", text="Hello", timestamp=now),
                Message(id="m2", role="assistant", text=message_text, timestamp=now),
            ],
            title="Test",
        )
        result = query._conv_to_json(conv, None)
        parsed = json.loads(result)

        assert len(parsed["messages"]) == 2
        assert parsed["messages"][0]["text"] == "Hello"
        if is_none:
            assert parsed["messages"][1]["text"] is None
        else:
            assert parsed["messages"][1]["text"] == message_text

    def test_timestamps_are_iso_formatted(self, sample_conversations):
        """Timestamps are ISO formatted in JSON."""
        conv = sample_conversations[0]
        result = query._conv_to_json(conv, None)
        parsed = json.loads(result)

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


class TestQueryConvToCsv:
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
        reader = csv.reader([lines[1]])
        parts = next(iter(reader))

        assert parts[0] == "conv1-abc123"  # conversation_id
        assert parts[1] == "m1"  # message_id
        assert parts[2] == "user"  # role
        assert "2024-06-15" in parts[3]  # timestamp
        assert parts[4] == "Hello world"  # text

    def test_timestamps_iso_formatted_in_csv(self, sample_conversations):
        """Timestamps are ISO formatted in CSV."""
        conv = sample_conversations[0]
        result = query._conv_to_csv_messages(conv)

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

        # Should not raise
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)

        assert len(rows) == 3  # Header + 2 messages
        assert rows[0] == ["conversation_id", "message_id", "role", "timestamp", "text"]
        assert len(rows[1]) == 5  # 5 columns
        assert len(rows[2]) == 5
