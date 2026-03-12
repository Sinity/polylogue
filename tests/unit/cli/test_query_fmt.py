"""Focused contracts for query formatting helpers.

This file owns pure formatting/projection behavior:
- filter description rendering
- YAML escaping
- single-conversation formatting across output formats
- list formatting across output formats
- streaming record rendering
- grouped stats rendering

Query execution, routing, transforms, and output destination handling live in
``test_query_exec_laws.py``.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import yaml
from rich.console import Console

from polylogue.cli.query_helpers import describe_query_filters
from polylogue.cli.query_output import _format_list, _output_stats_by, _write_message_streaming
from polylogue.rendering.formatting import _conv_to_dict, _yaml_safe, format_conversation
from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, Message


@dataclass(frozen=True)
class FilterCase:
    name: str
    params: dict[str, object]
    expected: tuple[str, ...]


@dataclass(frozen=True)
class ConversationFormatCase:
    name: str
    output_format: str
    fields: str | None
    expected: tuple[str, ...]
    excluded: tuple[str, ...] = ()


@dataclass(frozen=True)
class ListFormatCase:
    name: str
    output_format: str
    fields: str | None
    expected: tuple[str, ...]


FILTER_CASES = (
    FilterCase("empty", {}, ()),
    FilterCase(
        "mixed_filters",
        {
            "query": ("python", "errors"),
            "provider": "claude-ai",
            "exclude_provider": "chatgpt",
            "tag": "important",
            "exclude_tag": "spam",
            "title": "Test Title",
            "has_type": ("thinking", "tools"),
            "since": "2025-01-01",
            "until": "2025-12-31",
            "conv_id": "abc123",
        },
        (
            "search: python errors",
            "provider: claude-ai",
            "exclude provider: chatgpt",
            "tag: important",
            "exclude tag: spam",
            "has: thinking, tools",
            "since:",
            "until:",
            "title: Test Title",
            "id: abc123",
        ),
    ),
    FilterCase(
        "contains_and_negative",
        {"contains": ("fallback",), "exclude_text": ("internal",), "provider": "codex"},
        ("provider: codex", "contains: fallback", "exclude text: internal"),
    ),
)


CONVERSATION_FORMAT_CASES = (
    ConversationFormatCase(
        "markdown",
        "markdown",
        None,
        ("# Example Conversation", "## User", "## Assistant", "Hello", "Response"),
    ),
    ConversationFormatCase(
        "html",
        "html",
        None,
        ("<!DOCTYPE html>", "&lt;script&gt;alert(&#34;xss&#34;)&lt;/script&gt;", "message-user", "message-assistant"),
        excluded=("<script>",),
    ),
    ConversationFormatCase(
        "plaintext",
        "plaintext",
        None,
        ("Hello", "Response"),
        excluded=("## User", "**Provider**"),
    ),
    ConversationFormatCase(
        "obsidian",
        "obsidian",
        None,
        ("---", "provider: claude-ai", "tags:", "# Example Conversation"),
    ),
    ConversationFormatCase(
        "org",
        "org",
        None,
        ("#+TITLE: Example Conversation", "* USER", "* ASSISTANT"),
    ),
    ConversationFormatCase(
        "json_full",
        "json",
        None,
        ('"provider": "claude-ai"', '"messages": [', '"role": "assistant"'),
    ),
    ConversationFormatCase(
        "json_selected",
        "json",
        "id,provider,title",
        ('"provider": "claude-ai"', '"title": "Example Conversation"'),
        excluded=('"messages": [',),
    ),
    ConversationFormatCase(
        "yaml_full",
        "yaml",
        None,
        ("provider: claude-ai", "messages:", "- id: msg-user"),
    ),
    ConversationFormatCase(
        "yaml_selected",
        "yaml",
        "id,title",
        ("title: Example Conversation",),
        excluded=("messages:", "provider:"),
    ),
)


LIST_FORMAT_CASES = (
    ListFormatCase(
        "text",
        "text",
        None,
        ("conv-1234567890abcdef", "claude-ai", "Example Conversation"),
    ),
    ListFormatCase(
        "json",
        "json",
        None,
        ('"provider": "claude-ai"', '"summary": "Synthetic summary"'),
    ),
    ListFormatCase(
        "yaml",
        "yaml",
        None,
        ("provider: claude-ai", "summary: Synthetic summary"),
    ),
    ListFormatCase(
        "csv",
        "csv",
        None,
        ("id,date,provider,title,messages,words,tags,summary", "conv-1234567890abcdef"),
    ),
    ListFormatCase(
        "json_selected",
        "json",
        "id,title",
        ('"id": "conv-1234567890abcdef"', '"title": "Example Conversation"'),
    ),
)


STREAM_CASES = (
    ("plaintext", "[ASSISTANT]", "Hello from assistant"),
    ("markdown", "## Assistant", "Hello from assistant"),
    ("json-lines", '"type": "message"', '"role": "assistant"'),
)


STATS_CASES = (
    (
        "provider",
        [
            ("conv-1", "claude-ai", datetime(2025, 3, 1, tzinfo=timezone.utc), ["hello there", "more words"]),
            ("conv-2", "chatgpt", datetime(2025, 2, 1, tzinfo=timezone.utc), ["other text"]),
            ("conv-3", "claude-ai", datetime(2025, 1, 1, tzinfo=timezone.utc), ["final message"]),
        ],
        ("Matched: 3 conversations (by provider)", "claude-ai", "chatgpt", "TOTAL"),
    ),
    (
        "month",
        [
            ("conv-1", "claude-ai", datetime(2025, 3, 5, tzinfo=timezone.utc), ["month three"]),
            ("conv-2", "claude-ai", datetime(2025, 1, 6, tzinfo=timezone.utc), ["month one"]),
        ],
        ("Matched: 2 conversations (by month)", "2025-03", "2025-01", "TOTAL"),
    ),
)


def _make_msg(
    role: str = "user",
    text: str | None = "Hello",
    **kwargs: object,
) -> Message:
    return Message(
        id=str(kwargs.get("id", f"msg-{role}")),
        role=role,
        text=text,
        timestamp=kwargs.get("timestamp"),
        attachments=kwargs.get("attachments", []),
        provider_meta=kwargs.get("provider_meta"),
    )


def _make_conv(
    id: str = "conv-1234567890abcdef",
    provider: str = "claude-ai",
    title: str | None = "Example Conversation",
    messages: list[Message] | None = None,
    **kwargs: object,
) -> Conversation:
    if messages is None:
        messages = [
            _make_msg("user", "Hello", id="msg-user"),
            _make_msg("assistant", "Response", id="msg-assistant"),
        ]
    return Conversation(
        id=id,
        provider=provider,
        title=title,
        messages=MessageCollection(messages=messages),
        created_at=kwargs.get("created_at"),
        updated_at=kwargs.get("updated_at"),
        metadata={
            "tags": kwargs.get("tags", ["law", "example"]),
            "summary": kwargs.get("summary", "Synthetic summary"),
        },
    )


@pytest.fixture
def sample_conversation() -> Conversation:
    return _make_conv(
        updated_at=datetime(2025, 6, 15, 12, 30, tzinfo=timezone.utc),
        messages=[
            _make_msg("user", "Hello", id="msg-user"),
            _make_msg(
                "assistant",
                "Response",
                id="msg-assistant",
                provider_meta={"content_blocks": [{"type": "thinking", "text": "step one"}]},
            ),
        ],
    )


class TestFilterDescriptions:
    @pytest.mark.parametrize("case", FILTER_CASES, ids=lambda case: case.name)
    def test_describe_filters_contract_matrix(self, case: FilterCase) -> None:
        result = describe_query_filters(case.params)
        if not case.expected:
            assert result == []
            return
        for token in case.expected:
            assert any(token in item for item in result), (case.name, token, result)


class TestYamlEscaping:
    @pytest.mark.parametrize(
        ("value", "expected", "quoted"),
        [
            ("hello", "hello", False),
            ("key:value", "key:value", True),
            ("line1\nline2", "line1\\nline2", True),
            ('say "hello"', 'say \\"hello\\"', True),
            ("tab\there", "tab\\there", True),
        ],
    )
    def test_yaml_safe_contract(self, value: str, expected: str, quoted: bool) -> None:
        result = _yaml_safe(value)
        if quoted:
            assert result.startswith('"') and result.endswith('"')
            assert expected in result
        else:
            assert result == expected


class TestConversationFormatting:
    @pytest.mark.parametrize("case", CONVERSATION_FORMAT_CASES, ids=lambda case: case.name)
    def test_format_conversation_matrix(self, sample_conversation: Conversation, case: ConversationFormatCase) -> None:
        conversation = sample_conversation
        if case.output_format == "html":
            conversation = conversation.model_copy(update={"title": '<script>alert("xss")</script>'})
        rendered = format_conversation(conversation, case.output_format, case.fields)
        for token in case.expected:
            assert token in rendered, (case.name, token)
        for token in case.excluded:
            assert token not in rendered, (case.name, token)

    def test_conv_to_dict_field_selection_contract(self, sample_conversation: Conversation) -> None:
        selected = _conv_to_dict(sample_conversation, "id,title")
        assert selected == {
            "id": "conv-1234567890abcdef",
            "title": "Example Conversation",
        }

    def test_json_and_yaml_roundtrip_contract(self, sample_conversation: Conversation) -> None:
        json_data = json.loads(format_conversation(sample_conversation, "json", None))
        yaml_data = yaml.safe_load(format_conversation(sample_conversation, "yaml", None))
        assert json_data["id"] == yaml_data["id"] == "conv-1234567890abcdef"
        assert len(json_data["messages"]) == len(yaml_data["messages"]) == 2
        assert json_data["messages"][1]["text"] == yaml_data["messages"][1]["text"] == "Response"

    def test_csv_messages_skips_empty_text(self) -> None:
        conv = _make_conv(messages=[_make_msg("user", None, id="empty"), _make_msg("assistant", "Reply", id="reply")])
        rendered = format_conversation(conv, "csv", None)
        assert "empty" not in rendered
        assert "reply" in rendered


class TestListFormatting:
    @pytest.mark.parametrize("case", LIST_FORMAT_CASES, ids=lambda case: case.name)
    def test_format_list_contract_matrix(self, sample_conversation: Conversation, case: ListFormatCase) -> None:
        other = _make_conv(
            id="conv-bbbbbbbbbbbbbbbb",
            provider="chatgpt",
            title="Second Conversation",
            messages=[_make_msg("user", "Question"), _make_msg("assistant", "Answer")],
            updated_at=datetime(2025, 6, 16, 12, 30, tzinfo=timezone.utc),
            summary="Second summary",
            tags=["second"],
        )
        rendered = _format_list([sample_conversation, other], case.output_format, case.fields)
        for token in case.expected:
            assert token in rendered, (case.name, token)
        if case.name == "json_selected":
            payload = json.loads(rendered)
            assert payload[0] == {"id": "conv-1234567890abcdef", "title": "Example Conversation"}
        if case.name == "yaml":
            payload = yaml.safe_load(rendered)
            assert payload[0]["id"] == "conv-1234567890abcdef"
            assert payload[0]["provider"] == "claude-ai"


class TestStreamingOutput:
    @pytest.mark.parametrize("output_format,expected_role,expected_text", STREAM_CASES)
    def test_write_message_streaming_matrix(self, output_format: str, expected_role: str, expected_text: str) -> None:
        message = _make_msg(
            role="assistant",
            text="Hello from assistant",
            id="stream-1",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            _write_message_streaming(message, output_format)
        output = buffer.getvalue()
        assert expected_role in output
        assert expected_text in output
        if output_format == "json-lines":
            payload = json.loads(output)
            assert payload["id"] == "stream-1"
            assert payload["word_count"] == message.word_count


class TestGroupedStatsOutput:
    @pytest.mark.parametrize("dimension,raw_cases,expected_tokens", STATS_CASES, ids=[case[0] for case in STATS_CASES])
    def test_output_stats_by_contract_matrix(self, dimension: str, raw_cases, expected_tokens) -> None:
        conversations = [
            _make_conv(
                id=conv_id,
                provider=provider,
                updated_at=updated_at,
                messages=[_make_msg("assistant", text, id=f"{conv_id}-{index}") for index, text in enumerate(texts)],
            )
            for conv_id, provider, updated_at, texts in raw_cases
        ]
        console_buffer = io.StringIO()
        env = MagicMock()
        env.ui.console = Console(file=console_buffer, force_terminal=False, color_system=None, width=120)

        _output_stats_by(env, conversations, dimension)

        output = console_buffer.getvalue()
        for token in expected_tokens:
            assert token in output

    def test_output_stats_by_empty_contract(self) -> None:
        env = MagicMock()
        env.ui.console = MagicMock()
        _output_stats_by(env, [], "provider")
        env.ui.console.print.assert_called_once_with("No conversations matched.")
