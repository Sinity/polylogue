"""Focused contracts for query formatting helpers.

This file owns pure formatting/projection behavior:
- filter description rendering
- YAML escaping
- single-session formatting across output formats
- list formatting across output formats
- streaming record rendering

Query execution, routing, transforms, and output destination handling live in
``test_query_exec_laws.py``. Grouped-stats rendering (origin/date/semantic/
profile) was removed with the dead ``output_stats_by_sessions`` family
(polylogue-t46.6) — see ``polylogue/cli/query_stats.py``'s module docstring.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import yaml

from polylogue.archive.attachment.models import Attachment
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.roles import Role
from polylogue.archive.models import Message, Session, SessionSummary
from polylogue.archive.query.search_hits import DEFAULT_SEARCH_SNIPPET_MAX_CHARS, SessionSearchHit
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.cli.query_contracts import describe_query_filters
from polylogue.cli.query_output import (
    _format_list,
    _write_message_streaming,
    format_search_hit_list,
    format_summary_list,
    render_stream_transcript,
)
from polylogue.cli.query_output_contracts import StructuredRowsDocument
from polylogue.core.enums import Origin, Provider
from polylogue.core.types import SessionId
from polylogue.rendering.formatting import _conv_to_dict, _yaml_safe, format_session
from tests.infra.builders import make_conv as build_conv
from tests.infra.builders import make_msg as build_msg


@dataclass(frozen=True)
class FilterCase:
    name: str
    params: dict[str, object]
    expected: tuple[str, ...]


@dataclass(frozen=True)
class SessionFormatCase:
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
            "origin": "claude-ai-export",
            "exclude_origin": "chatgpt-export",
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
            "origin: claude-ai-export",
            "exclude origin: chatgpt-export",
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
        {"contains": ("fallback",), "exclude_text": ("internal",), "origin": "codex-session"},
        ("origin: codex-session", "contains: fallback", "exclude text: internal"),
    ),
)


SESSION_FORMAT_CASES = (
    SessionFormatCase(
        "markdown",
        "markdown",
        None,
        ("# Example Session", "## user", "## assistant", "Hello", "Response"),
    ),
    SessionFormatCase(
        "html",
        "html",
        None,
        ("<!DOCTYPE html>", "&lt;script&gt;alert(&#34;xss&#34;)&lt;/script&gt;", "message-user", "message-assistant"),
        excluded=("<script>",),
    ),
    SessionFormatCase(
        "plaintext",
        "plaintext",
        None,
        ("Hello", "Response"),
        excluded=("## User", "**Origin**"),
    ),
    SessionFormatCase(
        "obsidian",
        "obsidian",
        None,
        ("---", "origin: claude-ai-export", "tags:", "# Example Session"),
    ),
    SessionFormatCase(
        "org",
        "org",
        None,
        ("#+TITLE: Example Session", "* USER", "* ASSISTANT"),
    ),
    SessionFormatCase(
        "json_full",
        "json",
        None,
        ('"origin": "claude-ai-export"', '"messages": [', '"role": "assistant"'),
    ),
    SessionFormatCase(
        "json_selected",
        "json",
        "id,origin,title",
        ('"origin": "claude-ai-export"', '"title": "Example Session"'),
        excluded=('"messages": [',),
    ),
    SessionFormatCase(
        "yaml_full",
        "yaml",
        None,
        ("origin: claude-ai-export", "messages:", "- id: msg-user"),
    ),
    SessionFormatCase(
        "yaml_selected",
        "yaml",
        "id,title",
        ("title: Example Session",),
        excluded=("messages:", "origin:"),
    ),
)


LIST_FORMAT_CASES = (
    ListFormatCase(
        "text",
        "text",
        None,
        ("conv-1234567890abcdef", "claude-ai", "Example Session"),
    ),
    ListFormatCase(
        "json",
        "json",
        None,
        ('"origin": "claude-ai-export"', '"summary": "Synthetic summary"'),
    ),
    ListFormatCase(
        "yaml",
        "yaml",
        None,
        ("origin: claude-ai-export", "summary: Synthetic summary"),
    ),
    ListFormatCase(
        "csv",
        "csv",
        None,
        ("id,date,origin,title,messages,words,tags,summary", "conv-1234567890abcdef"),
    ),
    ListFormatCase(
        "json_selected",
        "json",
        "id,title",
        ('"id": "conv-1234567890abcdef"', '"title": "Example Session"'),
    ),
)


STREAM_CASES = (
    ("plaintext", "[ASSISTANT]", "Hello from assistant"),
    ("markdown", "## Assistant", "Hello from assistant"),
    ("json-lines", '"type": "message"', '"role": "assistant"'),
)


def _make_msg(
    role: str = "user",
    text: str | None = "Hello",
    *,
    id: str | None = None,
    timestamp: datetime | None = None,
    attachments: list[Attachment] | None = None,
    blocks: list[dict[str, object]] | None = None,
    provider_meta: dict[str, object] | None = None,
) -> Message:
    return build_msg(
        id=id or f"msg-{role}",
        role=Role.normalize(role),
        text=text,
        timestamp=timestamp,
        attachments=attachments or [],
        blocks=blocks or [],
        provider_meta=provider_meta,
    )


def _make_conv(
    id: str = "conv-1234567890abcdef",
    provider: str = "claude-ai",
    title: str | None = "Example Session",
    messages: list[Message] | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    tags: list[str] | None = None,
    summary: str = "Synthetic summary",
) -> Session:
    default_messages: Sequence[Message] | MessageCollection | None = messages
    if default_messages is None:
        default_messages = [
            _make_msg("user", "Hello", id="msg-user"),
            _make_msg("assistant", "Response", id="msg-assistant"),
        ]
    return build_conv(
        id=id,
        provider=Provider.from_string(provider),
        title=title,
        messages=default_messages,
        created_at=created_at,
        updated_at=updated_at,
        metadata={
            "tags": tags or ["law", "example"],
            "summary": summary,
        },
    )


@pytest.fixture
def sample_session() -> Session:
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


class TestSessionFormatting:
    @pytest.mark.parametrize("case", SESSION_FORMAT_CASES, ids=lambda case: case.name)
    def test_format_session_matrix(self, sample_session: Session, case: SessionFormatCase) -> None:
        session = sample_session
        if case.output_format == "html":
            session = session.model_copy(update={"title": '<script>alert("xss")</script>'})
        rendered = format_session(session, case.output_format, case.fields)
        for token in case.expected:
            assert token in rendered, (case.name, token)
        for token in case.excluded:
            assert token not in rendered, (case.name, token)

    def test_conv_to_dict_field_selection_contract(self, sample_session: Session) -> None:
        selected = _conv_to_dict(sample_session, "id,title")
        assert selected == {
            "id": "conv-1234567890abcdef",
            "title": "Example Session",
        }

    def test_json_and_yaml_roundtrip_contract(self, sample_session: Session) -> None:
        json_data = json.loads(format_session(sample_session, "json", None))
        yaml_data = yaml.safe_load(format_session(sample_session, "yaml", None))
        assert json_data["id"] == yaml_data["id"] == "conv-1234567890abcdef"
        assert len(json_data["messages"]) == len(yaml_data["messages"]) == 2
        assert json_data["messages"][1]["text"] == yaml_data["messages"][1]["text"] == "Response"

    def test_csv_messages_skips_empty_text(self) -> None:
        conv = _make_conv(messages=[_make_msg("user", None, id="empty"), _make_msg("assistant", "Reply", id="reply")])
        rendered = format_session(conv, "csv", None)
        assert "empty" not in rendered
        assert "reply" in rendered

    def test_format_session_applies_content_projection(self) -> None:
        conv = _make_conv(
            messages=[
                _make_msg(
                    "assistant",
                    "Alpha\n\n```python\nprint('x')\n```\n\nOmega",
                    id="projected",
                )
            ]
        )

        rendered = format_session(conv, "plaintext", None, content_projection=ContentProjectionSpec.prose_only())

        assert rendered == "Alpha\n\nOmega"


class TestListFormatting:
    def test_structured_rows_document_centralizes_machine_and_plain_rendering(self) -> None:
        document = StructuredRowsDocument(
            rows=(
                {
                    "id": "conv-a",
                    "provider": "claude-ai",
                    "title": "A",
                    "messages": 2,
                },
            ),
            csv_headers=("id", "provider", "title", "messages"),
            csv_rows=(("conv-a", "claude-ai", "A", 2),),
            text_lines=("conv-a  claude-ai  A (2 msgs)",),
        )

        # #1618: JSON/YAML render via envelope, not bare array.
        json_payload = json.loads(document.with_selected_fields("id,title").render("json"))
        assert json_payload["items"] == [{"id": "conv-a", "title": "A"}]
        assert json_payload["total"] == 1
        yaml_payload = yaml.safe_load(document.render("yaml"))
        assert yaml_payload["items"][0]["provider"] == "claude-ai"
        assert document.render("csv").splitlines()[0] == "id,provider,title,messages"
        assert document.render("text") == "conv-a  claude-ai  A (2 msgs)"

    @pytest.mark.parametrize("case", LIST_FORMAT_CASES, ids=lambda case: case.name)
    def test_format_list_contract_matrix(self, sample_session: Session, case: ListFormatCase) -> None:
        other = _make_conv(
            id="conv-bbbbbbbbbbbbbbbb",
            provider="chatgpt",
            title="Second Session",
            messages=[_make_msg("user", "Question"), _make_msg("assistant", "Answer")],
            updated_at=datetime(2025, 6, 16, 12, 30, tzinfo=timezone.utc),
            summary="Second summary",
            tags=["second"],
        )
        rendered = _format_list([sample_session, other], case.output_format, case.fields)
        for token in case.expected:
            assert token in rendered, (case.name, token)
        if case.name == "json_selected":
            payload = json.loads(rendered)
            # #1618: envelope shape, not bare array.
            assert payload["items"][0] == {"id": "conv-1234567890abcdef", "title": "Example Session"}
            assert payload["total"] == 2
        if case.name == "yaml":
            payload = yaml.safe_load(rendered)
            assert payload["items"][0]["id"] == "conv-1234567890abcdef"
            assert payload["items"][0]["origin"] == "claude-ai-export"

    @pytest.mark.parametrize("output_format", ["json", "yaml", "csv", "text"])
    def test_format_summary_list_contract(self, output_format: str) -> None:
        summary = SessionSummary(
            id=SessionId("conv-summary-1"),
            origin=Origin.CLAUDE_AI_EXPORT,
            title="Summary Session",
            created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            updated_at=datetime(2025, 6, 2, tzinfo=timezone.utc),
            metadata={"tags": ["alpha", "beta"], "summary": "Summary text"},
        )

        rendered = format_summary_list(
            [summary],
            output_format,
            None,
            message_counts={"conv-summary-1": 7},
        )

        if output_format == "json":
            payload = json.loads(rendered)
            # #1618: envelope shape, not bare array.
            assert payload["items"][0]["id"] == "conv-summary-1"
            assert payload["items"][0]["message_count"] == 7
            assert payload["items"][0]["tags"] == ["alpha", "beta"]
            assert payload["total"] == 1
        elif output_format == "yaml":
            payload = yaml.safe_load(rendered)
            assert payload["items"][0]["origin"] == "claude-ai-export"
            assert payload["items"][0]["summary"] == "Summary text"
        elif output_format == "csv":
            assert "id,date,origin,title,messages,tags,summary" in rendered
            assert "conv-summary-1" in rendered
            assert "alpha,beta" in rendered
        else:
            assert "conv-summary-1" in rendered
            assert "claude-ai" in rendered
            assert "(7 msgs)" in rendered

    @pytest.mark.parametrize("output_format", ["json", "yaml", "csv", "text"])
    def test_format_search_hit_list_contract(self, output_format: str) -> None:
        summary = SessionSummary(
            id=SessionId("conv-hit-1"),
            origin=Origin.CLAUDE_AI_EXPORT,
            title="Hit Session",
            created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            updated_at=datetime(2025, 6, 2, tzinfo=timezone.utc),
            metadata={"summary": "Summary text"},
        )
        hit = SessionSearchHit(
            summary=summary,
            rank=1,
            retrieval_lane="dialogue",
            match_surface="message",
            message_id="msg-hit-1",
            snippet="[needle] in context",
            score=-2.5,
        )

        rendered = format_search_hit_list(
            [hit],
            output_format,
            None,
            message_counts={"conv-hit-1": 4},
        )

        if output_format == "json":
            payload = json.loads(rendered)
            # #1618: envelope shape, not bare array.
            assert payload["items"][0]["session"]["id"] == "conv-hit-1"
            assert payload["items"][0]["session"]["message_count"] == 4
            assert payload["items"][0]["match"]["message_id"] == "msg-hit-1"
            assert payload["items"][0]["match"]["snippet"] == "[needle] in context"
        elif output_format == "yaml":
            payload = yaml.safe_load(rendered)
            assert payload["items"][0]["session"]["origin"] == "claude-ai-export"
            assert payload["items"][0]["match"]["retrieval_lane"] == "dialogue"
        elif output_format == "csv":
            assert "id,date,origin,title,messages,rank,retrieval_lane,match_surface,message_id,snippet" in rendered
            assert "conv-hit-1" in rendered
            assert "msg-hit-1" in rendered
        else:
            assert "conv-hit-1" in rendered
            assert "match[1]: message/dialogue/message msg-hit-1" in rendered
            assert "[needle] in context" in rendered

    def test_format_search_hit_list_exposes_attachment_identity_evidence(self) -> None:
        summary = SessionSummary(
            id=SessionId("conv-attachment-hit"),
            origin=Origin.AISTUDIO_DRIVE,
            title="Attachment Hit",
            created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            updated_at=datetime(2025, 6, 2, tzinfo=timezone.utc),
        )
        hit = SessionSearchHit(
            summary=summary,
            rank=1,
            retrieval_lane="attachment",
            match_surface="attachment",
            message_id="msg-doc",
            snippet='attachment identity attachment.provider_file_id=drive-file-1 name="Project Plan"',
        )

        rendered = format_search_hit_list([hit], "json", None, message_counts={"conv-attachment-hit": 1})
        payload = json.loads(rendered)

        # #1618: envelope shape, not bare array.
        assert payload["items"][0]["match"]["match_surface"] == "attachment"
        assert payload["items"][0]["match"]["retrieval_lane"] == "attachment"
        assert payload["items"][0]["match"]["message_id"] == "msg-doc"
        assert "attachment.provider_file_id=drive-file-1" in payload["items"][0]["match"]["snippet"]

    @pytest.mark.parametrize("output_format", ["json", "yaml", "csv", "text"])
    def test_format_search_hit_list_bounds_giant_snippets(self, output_format: str) -> None:
        summary = SessionSummary(
            id=SessionId("conv-giant-hit"),
            origin=Origin.CLAUDE_AI_EXPORT,
            title="Giant Hit",
            created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            updated_at=datetime(2025, 6, 2, tzinfo=timezone.utc),
        )
        giant_snippet = "needle " + ("full transcript payload " * 300)
        hit = SessionSearchHit(
            summary=summary,
            rank=1,
            retrieval_lane="dialogue",
            match_surface="message",
            message_id="msg-giant",
            snippet=giant_snippet,
        )

        rendered = format_search_hit_list([hit], output_format, None, message_counts={"conv-giant-hit": 1})

        assert "full transcript payload " * 20 not in rendered
        if output_format == "json":
            snippet = json.loads(rendered)["items"][0]["match"]["snippet"]
        elif output_format == "yaml":
            snippet = yaml.safe_load(rendered)["items"][0]["match"]["snippet"]
        elif output_format == "csv":
            snippet = rendered.rsplit(",", maxsplit=1)[-1].strip()
        else:
            snippet = rendered.split("needle", maxsplit=1)[1]
        assert len(snippet) <= DEFAULT_SEARCH_SNIPPET_MAX_CHARS + 64
        assert "..." in snippet

    def test_cli_list_json_envelope_matches_mcp_list_sessions_shape(self) -> None:
        """#1618: CLI ``--format json`` list output emits the same
        ``{"items": [...], "total": N, "limit": N, "offset": N}`` envelope
        that MCP ``list_sessions`` returns. Bare-array output was the
        pre-fix shape; both surfaces now align.
        """
        document = StructuredRowsDocument(
            rows=(
                {"id": "conv-a", "provider": "claude-ai", "title": "A", "messages": 3},
                {"id": "conv-b", "provider": "claude-ai", "title": "B", "messages": 5},
            ),
            csv_headers=("id", "provider", "title", "messages"),
            csv_rows=(("conv-a", "claude-ai", "A", 3), ("conv-b", "claude-ai", "B", 5)),
            text_lines=("conv-a  claude-ai  A (3 msgs)", "conv-b  claude-ai  B (5 msgs)"),
        )
        json_payload = json.loads(document.render("json"))
        yaml_payload = yaml.safe_load(document.render("yaml"))

        for payload in (json_payload, yaml_payload):
            assert set(payload.keys()) == {"items", "total", "limit", "offset"}, (
                f"#1618: envelope must carry items/total/limit/offset, got {sorted(payload.keys())}"
            )
            assert isinstance(payload["items"], list)
            assert len(payload["items"]) == 2
            assert payload["total"] == 2
            assert payload["offset"] == 0
            # ``limit`` mirrors ``total`` until CLI pagination lands.
            assert payload["limit"] == 2

        # ndjson stays a streaming form — no envelope. Pinned so a future
        # refactor doesn't accidentally wrap ndjson too (would break shell
        # pipelines that read line-by-line).
        ndjson_lines = [line for line in document.render("ndjson").splitlines() if line]
        parsed = [json.loads(line) for line in ndjson_lines]
        assert [row["id"] for row in parsed] == ["conv-a", "conv-b"]

    def test_format_summary_and_search_use_provider_display_label(self) -> None:
        summary = SessionSummary(
            id=SessionId("gemini:gemini-20250422-1234"),
            origin=Origin.AISTUDIO_DRIVE,
            title="gemini-20250422-1234",
            metadata={"title": "Project Plan: Please review the attached project plan."},
            created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            updated_at=datetime(2025, 6, 2, tzinfo=timezone.utc),
        )
        hit = SessionSearchHit(
            summary=summary,
            rank=1,
            retrieval_lane="dialogue",
            match_surface="message",
            message_id="msg-user",
            snippet="Please review the attached project plan.",
        )

        summary_payload = json.loads(format_summary_list([summary], "json", None, message_counts={str(summary.id): 1}))
        search_payload = json.loads(format_search_hit_list([hit], "json", None, message_counts={str(summary.id): 1}))

        # #1618: envelope shape, not bare array.
        assert summary_payload["items"][0]["title"] == "Project Plan: Please review the attached project plan."
        assert (
            search_payload["items"][0]["session"]["title"] == "Project Plan: Please review the attached project plan."
        )


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

    @pytest.mark.parametrize(
        ("output_format", "expected_tokens"),
        [
            (
                "markdown",
                (
                    "# Example Session",
                    "**Origin**: claude-ai-export",
                    "**Date**: 2025-06-15 12:30",
                    "_Streamed 2 messages_",
                ),
            ),
            (
                "json-lines",
                (
                    '"type": "header"',
                    '"origin": "claude-ai-export"',
                    '"date": "2025-06-15T12:30:00+00:00"',
                    '"type": "footer"',
                ),
            ),
            ("plaintext", ("[USER]", "[ASSISTANT]")),
        ],
    )
    def test_render_stream_transcript_contract(
        self, sample_session: Session, output_format: str, expected_tokens: tuple[str, ...]
    ) -> None:
        rendered, emitted = render_stream_transcript(
            session_id=str(sample_session.id),
            title=sample_session.display_title,
            origin=str(sample_session.origin),
            display_date=sample_session.display_date,
            messages=list(sample_session.messages),
            output_format=output_format,
        )

        assert emitted == 2
        for token in expected_tokens:
            assert token in rendered
