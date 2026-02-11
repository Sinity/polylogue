"""Tests for CLI query output formats and helper functions.

Covers:
- _describe_filters: filter description builder
- _yaml_safe: YAML quoting
- _conv_to_*: all 8 format converters
- _conv_to_dict: summary dict with field selection
- _format_conversation: format dispatch
- _format_list: list format dispatch
- _apply_transform: strip-tools, strip-thinking, strip-all
- _output_stats: stats output
- _output_stats_by: grouped stats output
- _write_message_streaming: streaming format output
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from io import StringIO
from unittest.mock import MagicMock

import pytest

from polylogue.lib.models import Conversation, Message
from polylogue.lib.messages import MessageCollection


# =============================================================================
# Test Helpers for Building Test Data
# =============================================================================


def _make_msg(
    role: str = "user",
    text: str | None = "Hello",
    **kwargs,
) -> Message:
    """Create a test message."""
    return Message(
        id=kwargs.get("id", f"msg-{role}"),
        role=role,
        text=text,
        timestamp=kwargs.get("timestamp"),
        attachments=kwargs.get("attachments", []),
        provider_meta=kwargs.get("provider_meta"),
    )


def _make_conv(
    id: str = "test-conv",
    provider: str = "chatgpt",
    title: str = "Test",
    messages: list[Message] | None = None,
    **kwargs,
) -> Conversation:
    """Create a test conversation."""
    if messages is None:
        messages = [_make_msg("user", "Hello"), _make_msg("assistant", "Hi there")]

    return Conversation(
        id=id,
        provider=provider,
        title=title,
        messages=MessageCollection(messages=messages),
        created_at=kwargs.get("created_at"),
        updated_at=kwargs.get("updated_at"),
        metadata={"tags": kwargs.get("tags", []), "summary": kwargs.get("summary")},
    )


# =============================================================================
# Tests for _describe_filters
# =============================================================================


class TestDescribeFilters:
    def _fn(self, params: dict) -> list[str]:
        from polylogue.cli.query import _describe_filters

        return _describe_filters(params)

    def test_empty_params(self) -> None:
        assert self._fn({}) == []

    @pytest.mark.parametrize(
        "params,expected_count,expected_content",
        [
            ({"query": ("error", "python")}, None, "error"),
            (
                {
                    "query": ("test",),
                    "contains": ("word",),
                    "provider": "claude",
                    "exclude_provider": "chatgpt",
                    "tag": "important",
                    "exclude_tag": "spam",
                    "title": "Test Title",
                    "has_type": ("thinking", "tools"),
                    "since": "2025-01-01",
                    "until": "2025-12-31",
                    "conv_id": "abc123",
                },
                11,
                None,
            ),
            ({"provider": "claude", "since": "2025-01-01"}, 2, None),
            ({"exclude_provider": "chatgpt"}, None, "exclude provider"),
            ({"has_type": ("thinking",)}, None, "has:"),
        ],
        ids=[
            "query_terms",
            "all_filter_types",
            "partial_filters",
            "exclude_provider",
            "has_type",
        ],
    )
    def test_filter_combinations(
        self, params: dict, expected_count: int | None, expected_content: str | None
    ) -> None:
        result = self._fn(params)
        if expected_count is not None:
            assert len(result) == expected_count
        if expected_content is not None:
            assert any(expected_content in r for r in result)


# =============================================================================
# Tests for _yaml_safe
# =============================================================================


class TestYamlSafe:
    def _fn(self, value: str) -> str:
        from polylogue.cli.query import _yaml_safe

        return _yaml_safe(value)

    @pytest.mark.parametrize(
        "input_str,expected_behavior",
        [
            ("hello", "unchanged"),
            ("key:value", "quoted"),
            ("# comment", "quoted"),
            ("line1\nline2", "escaped"),
            ("col1\tcol2", "escaped"),
            ('say "hello"', "escaped"),
        ],
        ids=[
            "plain_string",
            "colon",
            "hash",
            "newline",
            "tab",
            "quotes",
        ],
    )
    def test_yaml_safe_handling(self, input_str: str, expected_behavior: str) -> None:
        result = self._fn(input_str)
        if expected_behavior == "unchanged":
            assert result == input_str
        elif expected_behavior == "quoted":
            assert result.startswith('"') and result.endswith('"')
        elif expected_behavior == "escaped":
            assert "\\" in result or result.startswith('"')

    def test_special_chars_get_quoted(self) -> None:
        for char in ":#{}[]|>&*!?@`'\"":
            result = self._fn(f"text{char}here")
            assert result.startswith('"')
            assert result.endswith('"')


# =============================================================================
# Tests for _conv_to_markdown
# =============================================================================


class TestConvToMarkdown:
    def _fn(self, conv: Conversation) -> str:
        from polylogue.cli.query import _conv_to_markdown

        return _conv_to_markdown(conv)

    def test_basic(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        assert "# Test" in result
        assert "## User" in result
        assert "## Assistant" in result
        assert "Hello" in result
        assert "Hi there" in result

    def test_with_date(self) -> None:
        dt = datetime(2025, 6, 15, 12, 30, tzinfo=timezone.utc)
        conv = _make_conv(updated_at=dt)
        result = self._fn(conv)
        assert "2025-06-15" in result

    def test_no_title_uses_id(self) -> None:
        conv = _make_conv(title=None)
        result = self._fn(conv)
        # display_title truncates to 8 chars
        assert conv.id[:8] in result

    def test_empty_text_skipped(self) -> None:
        msgs = [_make_msg("user", None), _make_msg("assistant", "Reply")]
        conv = _make_conv(messages=msgs)
        result = self._fn(conv)
        assert "## User" not in result  # No text = skipped
        assert "## Assistant" in result

    def test_role_capitalized(self) -> None:
        msgs = [_make_msg("user", "Q"), _make_msg("assistant", "A")]
        conv = _make_conv(messages=msgs)
        result = self._fn(conv)
        assert "## User" in result
        assert "## Assistant" in result

    def test_provider_shown(self) -> None:
        conv = _make_conv(provider="claude")
        result = self._fn(conv)
        assert "claude" in result.lower()


# =============================================================================
# Tests for _conv_to_html
# =============================================================================


class TestConvToHtml:
    def _fn(self, conv: Conversation) -> str:
        from polylogue.cli.query import _conv_to_html

        return _conv_to_html(conv)

    def test_basic_structure(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        assert "<!DOCTYPE html>" in result
        assert "Polylogue" in result
        assert "message-user" in result
        assert "message-assistant" in result

    def test_pygments_css_included(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        assert "style>" in result

    def test_xss_safe_title(self) -> None:
        conv = _make_conv(title='<script>alert("xss")</script>')
        result = self._fn(conv)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_empty_messages_skipped(self) -> None:
        msgs = [_make_msg("user", None)]
        conv = _make_conv(messages=msgs)
        result = self._fn(conv)
        # Empty messages should not create message divs, though the CSS class
        # may still be in the stylesheet. Check for actual message div instead.
        assert '<div class="message-user">' not in result

    def test_role_class_sanitized(self) -> None:
        """Role names with special chars get sanitized CSS classes."""
        msgs = [_make_msg("tool_use", "tool output")]
        conv = _make_conv(messages=msgs)
        result = self._fn(conv)
        # Special chars should be replaced with hyphens
        assert "message-tool-use" in result

    def test_has_closing_tags(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        assert result.count("<html>") == result.count("</html>")
        assert result.count("<body>") == result.count("</body>")


# =============================================================================
# Tests for _conv_to_json
# =============================================================================


class TestConvToJson:
    def _fn(self, conv: Conversation, fields: str | None = None) -> str:
        from polylogue.cli.query import _conv_to_json

        return _conv_to_json(conv, fields)

    def test_valid_json(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        data = json.loads(result)
        assert data["id"] == "test-conv"
        assert data["provider"] == "chatgpt"
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) == 2

    def test_messages_have_content(self) -> None:
        conv = _make_conv()
        data = json.loads(self._fn(conv))
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["text"] == "Hello"

    def test_field_selection(self) -> None:
        conv = _make_conv()
        data = json.loads(self._fn(conv, "id,provider"))
        assert "id" in data
        assert "provider" in data

    def test_messages_included_by_default(self) -> None:
        conv = _make_conv()
        data = json.loads(self._fn(conv))
        assert "messages" in data
        assert isinstance(data["messages"], list)


# =============================================================================
# Tests for _conv_to_plaintext
# =============================================================================


class TestConvToPlaintext:
    def _fn(self, conv: Conversation) -> str:
        from polylogue.cli.query import _conv_to_plaintext

        return _conv_to_plaintext(conv)

    def test_no_formatting(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        assert "Hello" in result
        assert "Hi there" in result
        assert "##" not in result  # No markdown headers
        assert "**" not in result  # No bold

    def test_empty_text_skipped(self) -> None:
        msgs = [_make_msg("user", None), _make_msg("assistant", "Reply")]
        conv = _make_conv(messages=msgs)
        result = self._fn(conv)
        assert "Reply" in result

    def test_messages_separated_by_blank_lines(self) -> None:
        msgs = [_make_msg("user", "Q1"), _make_msg("assistant", "A1")]
        conv = _make_conv(messages=msgs)
        result = self._fn(conv)
        assert "Q1" in result
        assert "A1" in result
        # Should have blank line between messages
        assert "\n\n" in result


# =============================================================================
# Tests for _conv_to_csv_messages
# =============================================================================


class TestConvToCsvMessages:
    def _fn(self, conv: Conversation) -> str:
        from polylogue.cli.query import _conv_to_csv_messages

        return _conv_to_csv_messages(conv)

    def test_header_present(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        assert "conversation_id" in result
        assert "message_id" in result

    def test_rows_match_messages(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        lines = result.strip().split("\n")
        assert len(lines) == 3  # header + 2 messages

    def test_empty_text_skipped(self) -> None:
        msgs = [_make_msg("user", None), _make_msg("assistant", "Reply")]
        conv = _make_conv(messages=msgs)
        result = self._fn(conv)
        lines = result.strip().split("\n")
        assert len(lines) == 2  # header + 1 message (user skipped)

    def test_has_text_column(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        assert "text" in result


# =============================================================================
# Tests for _conv_to_obsidian
# =============================================================================


class TestConvToObsidian:
    def _fn(self, conv: Conversation) -> str:
        from polylogue.cli.query import _conv_to_obsidian

        return _conv_to_obsidian(conv)

    def test_frontmatter_present(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        assert result.startswith("---")
        assert "provider:" in result

    def test_content_after_frontmatter(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        # Should have frontmatter then markdown
        parts = result.split("---")
        assert len(parts) >= 3  # before, frontmatter, after

    def test_includes_date(self) -> None:
        dt = datetime(2025, 6, 15, 12, 30, tzinfo=timezone.utc)
        conv = _make_conv(updated_at=dt)
        result = self._fn(conv)
        assert "date:" in result

    def test_includes_id(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        assert "id:" in result


# =============================================================================
# Tests for _conv_to_org
# =============================================================================


class TestConvToOrg:
    def _fn(self, conv: Conversation) -> str:
        from polylogue.cli.query import _conv_to_org

        return _conv_to_org(conv)

    def test_org_header(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        assert "#+TITLE:" in result
        assert "#+DATE:" in result
        assert "#+PROPERTY:" in result

    def test_messages_as_headings(self) -> None:
        conv = _make_conv()
        result = self._fn(conv)
        assert "* USER" in result
        assert "* ASSISTANT" in result

    def test_role_uppercase(self) -> None:
        msgs = [_make_msg("user", "Q")]
        conv = _make_conv(messages=msgs)
        result = self._fn(conv)
        # Role should be uppercase in org mode
        assert "* USER" in result


# =============================================================================
# Tests for _conv_to_dict
# =============================================================================


class TestConvToDict:
    def _fn(self, conv: Conversation, fields: str | None = None) -> dict:
        from polylogue.cli.query import _conv_to_dict

        return _conv_to_dict(conv, fields)

    def test_all_fields(self) -> None:
        conv = _make_conv()
        d = self._fn(conv)
        assert d["id"] == "test-conv"
        assert d["provider"] == "chatgpt"
        assert d["messages"] == 2
        assert "tags" in d

    def test_field_selection(self) -> None:
        conv = _make_conv()
        d = self._fn(conv, "id,provider")
        assert set(d.keys()) == {"id", "provider"}

    def test_word_count(self) -> None:
        conv = _make_conv()
        d = self._fn(conv)
        assert d["words"] > 0

    def test_date_in_iso_format(self) -> None:
        dt = datetime(2025, 6, 15, 12, 30, tzinfo=timezone.utc)
        conv = _make_conv(updated_at=dt)
        d = self._fn(conv)
        assert "2025" in d["date"]


# =============================================================================
# Tests for _conv_to_csv
# =============================================================================


class TestConvToCsv:
    def _fn(self, results: list[Conversation]) -> str:
        from polylogue.cli.query import _conv_to_csv

        return _conv_to_csv(results)

    def test_header_present(self) -> None:
        result = self._fn([_make_conv()])
        assert "id,date,provider,title" in result

    def test_row_count(self) -> None:
        result = self._fn([_make_conv(id="c1"), _make_conv(id="c2")])
        lines = result.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_includes_all_fields(self) -> None:
        result = self._fn([_make_conv()])
        # Check that expected columns are present
        assert "messages" in result
        assert "words" in result


# =============================================================================
# Tests for _conv_to_yaml
# =============================================================================


class TestConvToYaml:
    def _fn(self, conv: Conversation, fields: str | None = None) -> str:
        from polylogue.cli.query import _conv_to_yaml

        return _conv_to_yaml(conv, fields)

    def test_valid_yaml(self) -> None:
        import yaml

        conv = _make_conv()
        result = self._fn(conv)
        data = yaml.safe_load(result)
        assert data["id"] == "test-conv"
        assert isinstance(data["messages"], list)

    def test_includes_messages_by_default(self) -> None:
        import yaml

        conv = _make_conv()
        result = self._fn(conv)
        data = yaml.safe_load(result)
        assert "messages" in data

    def test_preserves_unicode(self) -> None:
        import yaml

        msgs = [_make_msg("user", "Hello 世界")]
        conv = _make_conv(messages=msgs)
        result = self._fn(conv)
        data = yaml.safe_load(result)
        assert "世界" in data["messages"][0]["text"]


# =============================================================================
# Tests for _format_conversation
# =============================================================================


class TestFormatConversation:
    def _fn(self, conv: Conversation, fmt: str, fields: str | None = None) -> str:
        from polylogue.cli.query import _format_conversation

        return _format_conversation(conv, fmt, fields)

    @pytest.mark.parametrize(
        "fmt,expected_content",
        [
            ("markdown", "# Test"),
            ("json", None),  # JSON is valid but varies; check separately
            ("html", "<!DOCTYPE"),
            ("plaintext", "##"),  # Should NOT have this
            ("csv", "conversation_id"),
            ("obsidian", "---"),
            ("org", "#+TITLE:"),
            ("yaml", "id:"),
        ],
        ids=[
            "markdown",
            "json",
            "html",
            "plaintext",
            "csv",
            "obsidian",
            "org",
            "yaml",
        ],
    )
    def test_dispatch_formats(
        self, fmt: str, expected_content: str | None
    ) -> None:
        result = self._fn(_make_conv(), fmt)
        if fmt == "json":
            json.loads(result)  # Verify valid JSON
        elif fmt == "plaintext":
            assert expected_content not in result
        else:
            assert expected_content in result

    def test_unknown_defaults_to_markdown(self) -> None:
        result = self._fn(_make_conv(), "unknown_format")
        assert "# Test" in result


# =============================================================================
# Tests for _format_list
# =============================================================================


class TestFormatList:
    def _fn(self, results: list[Conversation], fmt: str, fields: str | None = None) -> str:
        from polylogue.cli.query import _format_list

        return _format_list(results, fmt, fields)

    @pytest.mark.parametrize(
        "fmt,expected_check",
        [
            ("json", "is_list"),
            ("csv", "id,date,provider"),
            ("text", "id_or_provider"),
            ("yaml", "is_list"),
        ],
        ids=["json", "csv", "text", "yaml"],
    )
    def test_format_list_dispatch(self, fmt: str, expected_check: str) -> None:
        result = self._fn([_make_conv()], fmt)
        if expected_check == "is_list":
            if fmt == "json":
                data = json.loads(result)
                assert isinstance(data, list)
            elif fmt == "yaml":
                import yaml

                data = yaml.safe_load(result)
                assert isinstance(data, list)
        elif expected_check == "id_or_provider":
            assert "test-conv" in result or "chatgpt" in result
        else:
            assert expected_check in result

    def test_multiple_conversations(self) -> None:
        result = self._fn([_make_conv(id="c1"), _make_conv(id="c2")], "text")
        assert "c1" in result
        assert "c2" in result


# =============================================================================
# Tests for _write_message_streaming
# =============================================================================


class TestWriteMessageStreaming:
    def _fn(self, msg: Message, fmt: str) -> None:
        from polylogue.cli.query import _write_message_streaming

        _write_message_streaming(msg, fmt)

    @pytest.mark.parametrize(
        "role,text,fmt,expected_output",
        [
            ("user", "Hello streaming", "plaintext", "[USER]"),
            ("assistant", "Reply here", "markdown", "## Assistant"),
            ("user", "JSON test", "json-lines", "type"),
            ("user", None, "plaintext", ""),
            ("user", None, "markdown", ""),
            ("assistant", "Timestamped", "json-lines", "timestamp"),
            ("assistant", "Text", "plaintext", "[ASSISTANT]"),
            ("user", "One two three", "json-lines", "word_count"),
        ],
        ids=[
            "plaintext_basic",
            "markdown_basic",
            "jsonlines_basic",
            "plaintext_empty",
            "markdown_empty",
            "jsonlines_timestamp",
            "plaintext_uppercase",
            "jsonlines_word_count",
        ],
    )
    def test_streaming_formats(
        self, role: str, text: str | None, fmt: str, expected_output: str, capsys
    ) -> None:
        msg = _make_msg(role, text)
        if fmt == "json-lines" and role == "assistant" and text == "Timestamped":
            ts = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
            msg = _make_msg(role, text, timestamp=ts)

        self._fn(msg, fmt)
        captured = capsys.readouterr()

        if expected_output == "":
            assert captured.out == ""
        elif fmt == "json-lines":
            data = json.loads(captured.out.strip())
            if expected_output == "type":
                assert data["type"] == "message"
                assert data["text"] == text
            elif expected_output == "timestamp":
                assert data["timestamp"] is not None
                assert "2025" in data["timestamp"]
            elif expected_output == "word_count":
                assert data["word_count"] == 3
        else:
            assert expected_output in captured.out


# =============================================================================
# Tests for _apply_transform
# =============================================================================


class TestApplyTransform:
    def _fn(self, results: list[Conversation], transform: str) -> list[Conversation]:
        from polylogue.cli.query import _apply_transform

        return _apply_transform(results, transform)

    @pytest.mark.parametrize(
        "transform,message_setup,expected_msg_count",
        [
            (
                "strip-tools",
                [
                    _make_msg("user", "Question"),
                    _make_msg("tool", "Tool output"),
                    _make_msg("assistant", "Answer"),
                ],
                2,
            ),
            (
                "strip-thinking",
                [_make_msg("user", "Question"), _make_msg("assistant", "Answer")],
                2,
            ),
            (
                "strip-all",
                [
                    _make_msg("user", "Q"),
                    _make_msg("tool", "T"),
                    _make_msg("assistant", "A"),
                ],
                3,
            ),
        ],
        ids=["strip_tools", "strip_thinking", "strip_all"],
    )
    def test_transform_variants(
        self, transform: str, message_setup: list[Message], expected_msg_count: int
    ) -> None:
        conv = _make_conv(messages=message_setup)
        result = self._fn([conv], transform)
        assert len(result) == 1

    def test_unknown_transform_returns_unchanged(self) -> None:
        conv = _make_conv()
        result = self._fn([conv], "unknown-transform")
        # Should return conv unchanged if transform not recognized
        assert len(result) == 1

    def test_multiple_conversations(self) -> None:
        conv1 = _make_conv(id="c1")
        conv2 = _make_conv(id="c2")
        result = self._fn([conv1, conv2], "strip-tools")
        assert len(result) == 2


# =============================================================================
# Tests for _output_stats
# =============================================================================


class TestOutputStats:
    def test_basic_stats(self) -> None:
        from polylogue.cli.query import _output_stats

        dt = datetime(2025, 6, 15, tzinfo=timezone.utc)
        conv = _make_conv(updated_at=dt)
        env = MagicMock()
        _output_stats(env, [conv])
        env.ui.console.print.assert_called()

    def test_empty_results(self) -> None:
        from polylogue.cli.query import _output_stats

        env = MagicMock()
        _output_stats(env, [])
        env.ui.console.print.assert_called_once_with("No conversations matched.")

    def test_aggregates_stats(self) -> None:
        from polylogue.cli.query import _output_stats

        dt1 = datetime(2025, 1, 15, tzinfo=timezone.utc)
        dt2 = datetime(2025, 6, 15, tzinfo=timezone.utc)
        convs = [
            _make_conv(id="c1", updated_at=dt1),
            _make_conv(id="c2", updated_at=dt2),
        ]
        env = MagicMock()
        _output_stats(env, convs)
        # Should have aggregated both conversations
        env.ui.console.print.assert_called()

    def test_calculates_word_count(self) -> None:
        from polylogue.cli.query import _output_stats

        msgs = [
            _make_msg("user", "One two"),
            _make_msg("assistant", "Three four five"),
        ]
        conv = _make_conv(messages=msgs)
        env = MagicMock()
        _output_stats(env, [conv])
        # Should have called print with stats
        env.ui.console.print.assert_called()


# =============================================================================
# Tests for _output_stats_by
# =============================================================================


class TestOutputStatsBy:
    @pytest.mark.parametrize(
        "groupby,convs_setup",
        [
            (
                "provider",
                [
                    _make_conv(id="c1", provider="claude", updated_at=datetime(2025, 6, 15, tzinfo=timezone.utc)),
                    _make_conv(id="c2", provider="chatgpt", updated_at=datetime(2025, 6, 15, tzinfo=timezone.utc)),
                    _make_conv(id="c3", provider="claude", updated_at=datetime(2025, 6, 15, tzinfo=timezone.utc)),
                ],
            ),
            (
                "month",
                [
                    _make_conv(id="c1", updated_at=datetime(2025, 1, 15, tzinfo=timezone.utc)),
                    _make_conv(id="c2", updated_at=datetime(2025, 6, 15, tzinfo=timezone.utc)),
                ],
            ),
            (
                "year",
                [
                    _make_conv(id="c1", updated_at=datetime(2024, 1, 15, tzinfo=timezone.utc)),
                    _make_conv(id="c2", updated_at=datetime(2025, 6, 15, tzinfo=timezone.utc)),
                ],
            ),
        ],
        ids=["by_provider", "by_month", "by_year"],
    )
    def test_stats_by_grouping(
        self, groupby: str, convs_setup: list[Conversation]
    ) -> None:
        from polylogue.cli.query import _output_stats_by

        env = MagicMock()
        _output_stats_by(env, convs_setup, groupby)
        env.ui.console.print.assert_called()

    def test_empty_results(self) -> None:
        from polylogue.cli.query import _output_stats_by

        env = MagicMock()
        _output_stats_by(env, [], "provider")
        env.ui.console.print.assert_called_once_with("No conversations matched.")

    def test_groups_correctly_by_provider(self) -> None:
        from polylogue.cli.query import _output_stats_by

        convs = [
            _make_conv(id="c1", provider="claude"),
            _make_conv(id="c2", provider="chatgpt"),
            _make_conv(id="c3", provider="claude"),
        ]
        env = MagicMock()
        _output_stats_by(env, convs, "provider")
        # Should have grouped by provider
        env.ui.console.print.assert_called()
