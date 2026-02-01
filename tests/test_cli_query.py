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
from pathlib import Path
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


class TestOutputByMonth:
    """Tests for _output_by_month aggregation."""

    def test_groups_by_month(self, mock_env, sample_conversations):
        """Groups conversations by month."""
        query._output_by_month(mock_env, sample_conversations)

        # Should have been called with month groupings
        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "2024-06" in output or "2024-07" in output

    def test_handles_missing_dates(self, mock_env):
        """Handles conversations without dates."""
        conv = Conversation(
            id="no-date",
            provider="test",
            messages=[],
            updated_at=None,
        )

        query._output_by_month(mock_env, [conv])

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "unknown" in output


class TestOutputByProvider:
    """Tests for _output_by_provider aggregation."""

    def test_groups_by_provider(self, mock_env, sample_conversations):
        """Groups conversations by provider."""
        query._output_by_provider(mock_env, sample_conversations)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "chatgpt" in output or "claude" in output

    def test_shows_percentages(self, mock_env, sample_conversations):
        """Shows percentages for each provider."""
        query._output_by_provider(mock_env, sample_conversations)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "%" in output


class TestOutputByTag:
    """Tests for _output_by_tag aggregation."""

    def test_groups_by_tag(self, mock_env, sample_conversations):
        """Groups conversations by tag."""
        query._output_by_tag(mock_env, sample_conversations)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "work" in output or "programming" in output

    def test_counts_untagged(self, mock_env, sample_conversations):
        """Counts untagged conversations."""
        query._output_by_tag(mock_env, sample_conversations)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "untagged" in output


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

    def test_shows_failure_message(self, mock_env):
        """Shows failure message when no clipboard tool found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            query._copy_to_clipboard(mock_env, "test content")

            calls = mock_env.ui.console.print.call_args_list
            output = " ".join(str(c) for c in calls)
            assert "could not" in output.lower() or "clipboard" in output.lower()


class TestOutputCsv:
    """Tests for _output_csv helper."""

    def test_writes_csv_file(self, mock_env, sample_conversations, tmp_path):
        """Writes CSV file with header and data."""
        csv_path = tmp_path / "output.csv"
        query._output_csv(mock_env, sample_conversations, csv_path)

        assert csv_path.exists()
        content = csv_path.read_text()

        # Check header
        assert "source,provider,conversation_id" in content

    def test_includes_messages(self, mock_env, sample_conversations, tmp_path):
        """CSV includes message-level rows."""
        csv_path = tmp_path / "output.csv"
        query._output_csv(mock_env, sample_conversations, csv_path)

        content = csv_path.read_text()
        lines = content.strip().split("\n")

        # Header + messages from all conversations
        assert len(lines) > len(sample_conversations)

    def test_creates_parent_directories(self, mock_env, sample_conversations, tmp_path):
        """Creates parent directories if needed."""
        csv_path = tmp_path / "subdir" / "output.csv"
        query._output_csv(mock_env, sample_conversations, csv_path)

        assert csv_path.exists()


class TestAnnotateConversations:
    """Tests for _annotate_conversations helper."""

    def test_shows_not_implemented(self, mock_env, sample_conversations):
        """Shows not implemented message."""
        params = {"annotate": "summarize each conversation"}

        query._annotate_conversations(mock_env, sample_conversations, params)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "not" in output.lower() and "implement" in output.lower()

    def test_shows_count(self, mock_env, sample_conversations):
        """Shows count of conversations that would be annotated."""
        params = {"annotate": "test prompt"}

        query._annotate_conversations(mock_env, sample_conversations, params)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "3" in output  # Number of sample conversations

    def test_handles_no_results(self, mock_env):
        """Handles empty results."""
        params = {"annotate": "test"}

        query._annotate_conversations(mock_env, [], params)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "no conversation" in output.lower()


class TestSendOutput:
    """Tests for _send_output helper."""

    def test_stdout_prints(self, mock_env, sample_conversations):
        """stdout destination prints content."""
        query._send_output(mock_env, "test content", ["stdout"], "markdown", None)

        calls = mock_env.ui.console.print.call_args_list
        assert any("test content" in str(c) for c in calls)

    def test_file_writes(self, mock_env, sample_conversations, tmp_path):
        """file destination writes content."""
        output_file = tmp_path / "output.md"

        query._send_output(mock_env, "test content", [str(output_file)], "markdown", None)

        assert output_file.exists()
        assert output_file.read_text() == "test content"

    def test_multiple_destinations(self, mock_env, sample_conversations, tmp_path):
        """Multiple destinations all receive content."""
        output_file = tmp_path / "output.md"

        query._send_output(mock_env, "test content", ["stdout", str(output_file)], "markdown", None)

        # Both stdout and file should be used
        assert output_file.exists()
        calls = mock_env.ui.console.print.call_args_list
        assert any("test content" in str(c) for c in calls)


class TestDryRunMode:
    """Tests for --dry-run functionality in modifiers and delete."""

    def test_dry_run_modifiers_shows_preview(self, mock_env, sample_conversations):
        """Dry-run mode shows preview without modifying."""
        params = {
            "add_tag": ("test-tag",),
            "dry_run": True,
        }

        query._apply_modifiers(mock_env, sample_conversations, params)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "DRY-RUN" in output
        assert "3" in output  # Count of conversations
        assert "add tags" in output.lower()

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

    def test_dry_run_delete_shows_preview(self, mock_env, sample_conversations):
        """Dry-run delete shows preview without deleting."""
        params = {"dry_run": True}

        query._delete_conversations(mock_env, sample_conversations, params)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "DRY-RUN" in output
        assert "delete" in output.lower()
        assert "3" in output


class TestBulkOperationConfirmation:
    """Tests for bulk operation confirmation (>10 items requires --force)."""

    def test_modifiers_require_force_for_bulk(self, mock_env):
        """Modifiers require --force for >10 items."""
        # Create 15 mock conversations
        convs = [
            MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test")
            for i in range(15)
        ]
        params = {"add_tag": ("bulk-tag",), "force": False}

        with pytest.raises(SystemExit) as exc_info:
            query._apply_modifiers(mock_env, convs, params)

        assert exc_info.value.code == 1
        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "15" in output
        assert "--force" in output

    def test_modifiers_proceed_with_force(self, mock_env):
        """Modifiers proceed with --force for bulk operations."""
        # Create 15 mock conversations
        convs = [
            MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test")
            for i in range(15)
        ]
        params = {"add_tag": ("bulk-tag",), "force": True}

        with patch("polylogue.cli.helpers.load_effective_config"), \
             patch("polylogue.cli.container.create_repository") as mock_repo:
            mock_backend = MagicMock()
            mock_repo.return_value = mock_backend

            query._apply_modifiers(mock_env, convs, params)

            # Should have called add_tag 15 times
            assert mock_backend.add_tag.call_count == 15

    def test_delete_requires_force_for_bulk(self, mock_env):
        """Delete requires --force for >10 items."""
        convs = [
            MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test", updated_at=None)
            for i in range(15)
        ]
        params = {"force": False}

        with pytest.raises(SystemExit) as exc_info:
            query._delete_conversations(mock_env, convs, params)

        assert exc_info.value.code == 1
        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "DELETE" in output
        assert "--force" in output

    def test_delete_proceeds_with_force(self, mock_env):
        """Delete proceeds with --force for bulk operations."""
        convs = [
            MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test")
            for i in range(15)
        ]
        params = {"force": True}

        with patch("polylogue.cli.helpers.load_effective_config"), \
             patch("polylogue.cli.container.create_repository") as mock_repo:
            mock_backend = MagicMock()
            mock_backend.delete_conversation.return_value = True
            mock_repo.return_value = mock_backend

            query._delete_conversations(mock_env, convs, params)

            # Should have called delete_conversation 15 times
            assert mock_backend.delete_conversation.call_count == 15

    def test_small_operations_proceed_without_force(self, mock_env):
        """Operations with <=10 items proceed without --force."""
        convs = [
            MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test")
            for i in range(5)
        ]
        params = {"add_tag": ("small-tag",), "force": False}

        with patch("polylogue.cli.helpers.load_effective_config"), \
             patch("polylogue.cli.container.create_repository") as mock_repo:
            mock_backend = MagicMock()
            mock_repo.return_value = mock_backend

            # Should not raise
            query._apply_modifiers(mock_env, convs, params)

            assert mock_backend.add_tag.call_count == 5


class TestOperationReporting:
    """Tests for operation result reporting."""

    def test_add_tag_reports_count(self, mock_env, sample_conversations):
        """Add tag reports count of affected conversations."""
        params = {"add_tag": ("new-tag",)}

        with patch("polylogue.cli.helpers.load_effective_config"), \
             patch("polylogue.cli.container.create_repository") as mock_repo:
            mock_backend = MagicMock()
            mock_repo.return_value = mock_backend

            query._apply_modifiers(mock_env, sample_conversations, params)

            calls = mock_env.ui.console.print.call_args_list
            output = " ".join(str(c) for c in calls)
            assert "new-tag" in output
            assert "3" in output  # Number of conversations

    def test_rm_tag_reports_count(self, mock_env, sample_conversations):
        """Remove tag reports count of affected conversations."""
        params = {"rm_tag": ("old-tag",)}

        with patch("polylogue.cli.helpers.load_effective_config"), \
             patch("polylogue.cli.container.create_repository") as mock_repo:
            mock_backend = MagicMock()
            mock_repo.return_value = mock_backend

            query._apply_modifiers(mock_env, sample_conversations, params)

            calls = mock_env.ui.console.print.call_args_list
            output = " ".join(str(c) for c in calls)
            assert "old-tag" in output
            assert "3" in output

    def test_delete_reports_count(self, mock_env, sample_conversations):
        """Delete reports count of deleted conversations."""
        params = {}

        with patch("polylogue.cli.helpers.load_effective_config"), \
             patch("polylogue.cli.container.create_repository") as mock_repo:
            mock_backend = MagicMock()
            mock_backend.delete_conversation.return_value = True
            mock_repo.return_value = mock_backend

            query._delete_conversations(mock_env, sample_conversations, params)

            calls = mock_env.ui.console.print.call_args_list
            output = " ".join(str(c) for c in calls)
            assert "Deleted" in output
            assert "3" in output
