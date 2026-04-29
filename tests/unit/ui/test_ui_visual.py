"""Consolidated UI visual/rendering tests.

MERGED: test_rendering.py (renderer implementations, core formatting) +
        test_golden.py (golden/snapshot reference tests)

To regenerate golden files after intentional rendering changes:
    UPDATE_GOLDEN=1 pytest tests/unit/ui/test_ui_visual.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TypeAlias

import pytest

from polylogue.config import Config
from polylogue.rendering.core import (
    ConversationFormatter,
    FormattedConversation,
    FormattedConversationMetadata,
)
from polylogue.rendering.renderers import (
    HTMLRenderer,
    MarkdownRenderer,
    create_renderer,
    list_formats,
)
from polylogue.rendering.renderers.markdown import MarkdownRenderer as MarkdownRendererDirect
from polylogue.storage.backends.connection import open_connection
from tests.infra import GOLDEN_DIR
from tests.infra.assertions import (
    assert_contains_all,
    assert_messages_ordered,
    assert_not_contains_any,
)
from tests.infra.storage_records import (
    ConversationBuilder,
    DbFactory,
    db_setup,
    make_conversation,
    make_message,
    store_records,
)

JSONRecord: TypeAlias = dict[str, object]
WorkspaceEnv: TypeAlias = dict[str, Path]
MessageOrderingRow: TypeAlias = tuple[str, str, str | None]
AttachmentCaseMeta: TypeAlias = JSONRecord | list[JSONRecord] | None
AttachmentExpected: TypeAlias = str | list[str]

# =============================================================================
# RENDERER IMPLEMENTATION TESTS (from test_rendering.py)
# =============================================================================


@pytest.fixture
def sample_conversation_id() -> str:
    """Create a sample conversation for testing."""
    conversation = make_conversation(
        "test-conv-1",
        provider_name="test-provider",
        title="Test Conversation",
        created_at="2024-01-01T10:00:00Z",
        updated_at="2024-01-01T10:00:10Z",
        provider_meta={"source": "test"},
    )

    messages = [
        make_message("msg1", "test-conv-1", text="Hello, can you help me?", timestamp="2024-01-01T10:00:00Z"),
        make_message(
            "msg2",
            "test-conv-1",
            role="assistant",
            text="Of course! How can I help you today?",
            timestamp="2024-01-01T10:00:05Z",
        ),
        make_message("msg3", "test-conv-1", text="I need help with Python testing", timestamp="2024-01-01T10:00:10Z"),
    ]

    with open_connection(None) as conn:
        store_records(conversation=conversation, messages=messages, attachments=[], conn=conn)

    return "test-conv-1"


@pytest.fixture
def sample_conversation_with_json() -> str:
    """Create a conversation with JSON content (tool use)."""
    conversation = make_conversation(
        "test-conv-json",
        provider_name="test-provider",
        title="JSON Test",
        created_at="2024-01-01T10:00:00Z",
        updated_at="2024-01-01T10:00:05Z",
        provider_meta={"source": "test"},
    )

    messages = [
        make_message("msg1", "test-conv-json", text="Search for Python testing", timestamp="2024-01-01T10:00:00Z"),
        make_message(
            "msg2",
            "test-conv-json",
            role="assistant",
            text='{"query": "Python testing", "results": ["pytest", "unittest"]}',
            timestamp="2024-01-01T10:00:05Z",
        ),
    ]

    with open_connection(None) as conn:
        store_records(conversation=conversation, messages=messages, attachments=[], conn=conn)

    return "test-conv-json"


class TestMarkdownRenderer:
    """Tests for MarkdownRenderer."""

    def test_supports_format(self, workspace_env: WorkspaceEnv) -> None:
        renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        assert renderer.supports_format() == "markdown"

    @pytest.mark.asyncio
    async def test_render_basic_conversation(self, workspace_env: WorkspaceEnv, sample_conversation_id: str) -> None:
        renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "render"
        result_path = await renderer.render(sample_conversation_id, output_path)
        assert result_path.exists()
        assert result_path.suffix == ".md"
        content = result_path.read_text()
        assert "# Test Conversation" in content
        assert "Provider: test-provider" in content
        assert "Conversation ID: test-conv-1" in content
        assert "## user" in content
        assert "Hello, can you help me?" in content
        assert "## assistant" in content
        assert "Of course! How can I help you today?" in content
        assert "I need help with Python testing" in content

    @pytest.mark.asyncio
    async def test_render_with_json_formatting(
        self, workspace_env: WorkspaceEnv, sample_conversation_with_json: str
    ) -> None:
        renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "render"
        result_path = await renderer.render(sample_conversation_with_json, output_path)
        content = result_path.read_text()
        assert "```json" in content
        assert '"query": "Python testing"' in content
        assert '"results"' in content

    @pytest.mark.asyncio
    async def test_render_nonexistent_conversation(self, workspace_env: WorkspaceEnv) -> None:
        renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "render"
        with pytest.raises(ValueError, match="Conversation not found"):
            await renderer.render("nonexistent-id", output_path)

    @pytest.mark.asyncio
    async def test_render_creates_output_directory(
        self, workspace_env: WorkspaceEnv, sample_conversation_id: str
    ) -> None:
        renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "custom" / "nested" / "render"
        assert not output_path.exists()
        result_path = await renderer.render(sample_conversation_id, output_path)
        assert result_path.exists()
        assert result_path.parent.exists()


class TestHTMLRenderer:
    """Tests for HTMLRenderer."""

    def test_supports_format(self, workspace_env: WorkspaceEnv) -> None:
        renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])
        assert renderer.supports_format() == "html"

    @pytest.mark.asyncio
    async def test_render_basic_conversation(self, workspace_env: WorkspaceEnv, sample_conversation_id: str) -> None:
        renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "render"
        result_path = await renderer.render(sample_conversation_id, output_path)
        assert result_path.exists()
        assert result_path.suffix == ".html"
        content = result_path.read_text()
        assert "<!doctype html>" in content.lower()
        assert "<html" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "Test Conversation" in content
        assert "<title>" in content
        assert "test-provider" in content
        assert "Hello, can you help me?" in content

    @pytest.mark.asyncio
    async def test_render_nonexistent_conversation(self, workspace_env: WorkspaceEnv) -> None:
        renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "render"
        with pytest.raises(ValueError, match="Conversation not found"):
            await renderer.render("nonexistent-id", output_path)

    @pytest.mark.asyncio
    async def test_render_with_json_content(
        self, workspace_env: WorkspaceEnv, sample_conversation_with_json: str
    ) -> None:
        renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "render"
        result_path = await renderer.render(sample_conversation_with_json, output_path)
        content = result_path.read_text()
        assert "Python testing" in content
        assert "query" in content


class TestRendererFactory:
    """Tests for renderer factory functions."""

    def test_create_markdown_renderer(self, workspace_env: WorkspaceEnv) -> None:
        config = Config(
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
            sources=[],
        )
        renderer = create_renderer("markdown", config)
        assert isinstance(renderer, MarkdownRenderer)
        assert renderer.supports_format() == "markdown"

    def test_create_html_renderer(self, workspace_env: WorkspaceEnv) -> None:
        config = Config(
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
            sources=[],
        )
        renderer = create_renderer("html", config)
        assert isinstance(renderer, HTMLRenderer)
        assert renderer.supports_format() == "html"

    def test_create_renderer_case_insensitive(self, workspace_env: WorkspaceEnv) -> None:
        config = Config(
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
            sources=[],
        )
        renderer1 = create_renderer("HTML", config)
        renderer2 = create_renderer("Html", config)
        renderer3 = create_renderer("MARKDOWN", config)
        assert isinstance(renderer1, HTMLRenderer)
        assert isinstance(renderer2, HTMLRenderer)
        assert isinstance(renderer3, MarkdownRenderer)

    def test_create_renderer_unsupported_format(self, workspace_env: WorkspaceEnv) -> None:
        config = Config(
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
            sources=[],
        )
        with pytest.raises(ValueError, match="Unsupported format: json"):
            create_renderer("json", config)

    def test_list_formats(self) -> None:
        formats = list_formats()
        assert isinstance(formats, list)
        assert "markdown" in formats
        assert "html" in formats
        assert len(formats) >= 2


class TestRendererIntegration:
    """Integration tests for renderers."""

    @pytest.mark.asyncio
    async def test_both_renderers_produce_output(
        self, workspace_env: WorkspaceEnv, sample_conversation_id: str
    ) -> None:
        md_renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        html_renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "render"
        md_path = await md_renderer.render(sample_conversation_id, output_path)
        html_path = await html_renderer.render(sample_conversation_id, output_path)
        assert md_path.exists()
        assert html_path.exists()
        md_content = md_path.read_text()
        html_content = html_path.read_text()
        assert "Test Conversation" in md_content
        assert "Test Conversation" in html_content
        assert "Hello, can you help me?" in md_content
        assert "Hello, can you help me?" in html_content

    def test_protocol_compliance(self, workspace_env: WorkspaceEnv) -> None:
        from polylogue.protocols import OutputRenderer

        md_renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        html_renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])
        assert isinstance(md_renderer, OutputRenderer)
        assert isinstance(html_renderer, OutputRenderer)
        assert hasattr(md_renderer, "render")
        assert hasattr(md_renderer, "supports_format")
        assert hasattr(html_renderer, "render")
        assert hasattr(html_renderer, "supports_format")


# =============================================================================
# RENDERING CORE TESTS (from test_rendering.py)
# =============================================================================


def test_formatted_conversation_dataclass_fields() -> None:
    fc = FormattedConversation(
        title="Test Title",
        provider="chatgpt",
        conversation_id="conv-123",
        markdown_text="# Test\n\nContent",
        metadata=FormattedConversationMetadata(
            message_count=5,
            attachment_count=0,
            created_at=None,
            updated_at=None,
        ),
    )
    assert fc.title == "Test Title"
    assert fc.provider == "chatgpt"
    assert fc.conversation_id == "conv-123"
    assert fc.markdown_text == "# Test\n\nContent"
    assert fc.metadata == FormattedConversationMetadata(
        message_count=5,
        attachment_count=0,
        created_at=None,
        updated_at=None,
    )


def test_formatted_conversation_dataclass_equality() -> None:
    fc1 = FormattedConversation(
        title="Test",
        provider="claude-ai",
        conversation_id="c1",
        markdown_text="md",
        metadata=FormattedConversationMetadata(
            message_count=0,
            attachment_count=0,
            created_at=None,
            updated_at=None,
        ),
    )
    fc2 = FormattedConversation(
        title="Test",
        provider="claude-ai",
        conversation_id="c1",
        markdown_text="md",
        metadata=FormattedConversationMetadata(
            message_count=0,
            attachment_count=0,
            created_at=None,
            updated_at=None,
        ),
    )
    assert fc1 == fc2


INIT_CASES = [
    ("basic path", "accepts and stores"),
    ("path object", "works with Path object"),
]


@pytest.mark.parametrize("label,desc", INIT_CASES)
def test_formatter_initialization_comprehensive(tmp_path: Path, label: str, desc: str) -> None:
    if label == "basic path":
        formatter = ConversationFormatter(tmp_path)
        assert formatter.archive_root == tmp_path
    elif label == "path object":
        archive = tmp_path / "archive"
        archive.mkdir()
        formatter = ConversationFormatter(archive)
        assert formatter.archive_root == archive


FORMAT_CASES = [
    ("missing conversation", "nonexistent-conv", "raises ValueError"),
    ("basic conversation", "basic-conv", "returns FormattedConversation"),
    ("null title", "no-title-conv", "uses conversation_id as title"),
]


@pytest.mark.parametrize("label,conv_id,desc", FORMAT_CASES)
@pytest.mark.asyncio
async def test_formatter_format_comprehensive(workspace_env: WorkspaceEnv, label: str, conv_id: str, desc: str) -> None:
    db_path = db_setup(workspace_env)
    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)

    if label == "missing conversation":
        with pytest.raises(ValueError, match="Conversation not found"):
            await formatter.format(conv_id)
    elif label == "basic conversation":
        (ConversationBuilder(db_path, conv_id).add_message("m1", role="user", text="Hello!").save())
        result = await formatter.format(conv_id)
        assert isinstance(result, FormattedConversation)
        assert result.title == "Test Conversation"
        assert result.provider == "test"
        assert result.conversation_id == conv_id
        assert "Hello!" in result.markdown_text
        assert result.metadata.message_count == 1
    elif label == "null title":
        (ConversationBuilder(db_path, conv_id).title(None).save())
        result = await formatter.format(conv_id)
        assert result.title == conv_id
        assert f"# {conv_id}" in result.markdown_text


MESSAGE_ORDERING_CASES = [
    (
        "timestamp order",
        "ordered-conv",
        [
            ("m3", "Third", "2024-01-01T12:00:30Z"),
            ("m1", "First", "2024-01-01T12:00:10Z"),
            ("m2", "Second", "2024-01-01T12:00:20Z"),
        ],
        "timestamp ascending",
    ),
    (
        "null timestamps",
        "null-ts-conv",
        [
            ("m1", "Timestamped", "2024-01-01T12:00:00Z"),
            ("m2", "NoTimestamp", None),
        ],
        "null timestamps sort last",
    ),
    (
        "epoch timestamps",
        "epoch-conv",
        [
            ("m1", "LaterEpoch", "1704110400.5"),
            ("m2", "EarlierEpoch", "1704106800"),
        ],
        "numeric epoch timestamps",
    ),
]


@pytest.mark.parametrize("label,conv_id,message_data,desc", MESSAGE_ORDERING_CASES)
@pytest.mark.asyncio
async def test_message_ordering_comprehensive(
    workspace_env: WorkspaceEnv, label: str, conv_id: str, message_data: list[MessageOrderingRow], desc: str
) -> None:
    db_path = db_setup(workspace_env)
    builder = ConversationBuilder(db_path, conv_id)
    for i, (msg_id, text, timestamp) in enumerate(message_data):
        role = "user" if i % 2 == 0 else "assistant"
        builder.add_message(msg_id, role=role, text=text, timestamp=timestamp)
    builder.save()
    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = await formatter.format(conv_id)
    if label == "timestamp order":
        assert_messages_ordered(result.markdown_text, "First", "Second", "Third")
    elif label == "null timestamps":
        assert_messages_ordered(result.markdown_text, "Timestamped", "NoTimestamp")
    elif label == "epoch timestamps":
        assert_messages_ordered(result.markdown_text, "EarlierEpoch", "LaterEpoch")


JSON_WRAPPING_CASES = [
    ('{"key": "value", "count": 42}', True, "JSON object wrapped"),
    ('[1, 2, 3, "four"]', True, "JSON array wrapped"),
    ("{malformed json without closing", False, "invalid JSON not wrapped"),
    ("{this is not json}", False, "JSON-like but not JSON"),
    ("This is just regular text.", False, "plain text not wrapped"),
]


@pytest.mark.parametrize("text,wrapped,desc", JSON_WRAPPING_CASES)
@pytest.mark.asyncio
async def test_json_text_wrapping_comprehensive(
    workspace_env: WorkspaceEnv, text: str, wrapped: bool, desc: str
) -> None:
    db_path = db_setup(workspace_env)
    conv_id = f"json-{hash(text) % 10000}-conv"
    (ConversationBuilder(db_path, conv_id).title("Test").add_message("m1", role="tool", text=text).save())
    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = await formatter.format(conv_id)
    if wrapped:
        assert "```json" in result.markdown_text, f"Failed {desc}"
        assert "```" in result.markdown_text
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            assert any(key in result.markdown_text for key in parsed), f"Failed {desc}: JSON keys not found"
        elif isinstance(parsed, list):
            assert "[" in result.markdown_text and "]" in result.markdown_text, (
                f"Failed {desc}: JSON array markers not found"
            )
    else:
        assert "```json" not in result.markdown_text, f"Failed {desc}"
        assert text in result.markdown_text, f"Failed {desc}: text not found"


TIMESTAMP_RENDERING_CASES = [
    ("2024-01-15T10:30:00Z", True, "timestamps rendered"),
    (None, False, "no timestamp when null"),
]


@pytest.mark.parametrize("timestamp,rendered,desc", TIMESTAMP_RENDERING_CASES)
@pytest.mark.asyncio
async def test_timestamp_rendering_comprehensive(
    workspace_env: WorkspaceEnv, timestamp: str | None, rendered: bool, desc: str
) -> None:
    db_path = db_setup(workspace_env)
    conv_id = f"ts-{hash(str(timestamp)) % 10000}-conv"
    (
        ConversationBuilder(db_path, conv_id)
        .title("Test")
        .add_message("m1", role="user", text="Hello", timestamp=timestamp)
        .save()
    )
    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = await formatter.format(conv_id)
    if rendered:
        assert f"_Timestamp: {timestamp}_" in result.markdown_text, f"Failed {desc}"
    else:
        assert "_Timestamp:" not in result.markdown_text, f"Failed {desc}"


ATTACHMENT_CASES = [
    ("meta.name", {"name": "MyFile.pdf"}, "MyFile.pdf", "name from meta.name"),
    ("meta.provider_id", {"provider_id": "provider_file_123"}, "provider_file_123", "name from provider_id"),
    ("meta.drive_id", {"drive_id": "1ABC123XYZ"}, "1ABC123XYZ", "name from drive_id"),
    ("fallback ID", None, "att-fallback-123", "fallback to attachment_id"),
    ("empty meta", {}, "att-empty-meta", "fallback when meta empty"),
    (
        "multiple",
        [
            {"id": "att1", "meta": {"name": "File1.png"}},
            {"id": "att2", "meta": {"name": "File2.jpg"}},
            {"id": "att3", "meta": {"name": "File3.txt"}},
        ],
        ["File1.png", "File2.jpg", "File3.txt"],
        "multiple attachments",
    ),
    ("path", {"name": "Doc.pdf"}, "custom/path/to/file.pdf", "uses explicit path"),
]


@pytest.mark.parametrize("label,meta,expected,desc", ATTACHMENT_CASES)
@pytest.mark.asyncio
async def test_attachment_handling_comprehensive(
    workspace_env: WorkspaceEnv, label: str, meta: AttachmentCaseMeta, expected: AttachmentExpected, desc: str
) -> None:
    db_path = db_setup(workspace_env)
    conv_id = f"att-{label}-conv"
    builder = ConversationBuilder(db_path, conv_id).title("Test").add_message("m1", role="user", text="See attachment")
    if label == "multiple":
        assert isinstance(meta, list)
        for att in meta:
            attachment_id = att["id"]
            provider_meta = att.get("meta")
            assert isinstance(attachment_id, str)
            assert provider_meta is None or isinstance(provider_meta, dict)
            builder.add_attachment(
                attachment_id=attachment_id,
                message_id="m1",
                provider_meta=provider_meta,
            )
    elif label == "path":
        assert isinstance(expected, str)
        assert meta is None or isinstance(meta, dict)
        builder.add_attachment(
            attachment_id="att1",
            message_id="m1",
            path=expected,
            provider_meta=meta,
        )
    else:
        assert isinstance(expected, str)
        assert meta is None or isinstance(meta, dict)
        att_id = expected if meta is None or meta == {} else "att1"
        builder.add_attachment(
            attachment_id=att_id,
            message_id="m1",
            provider_meta=meta,
        )
    builder.save()
    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = await formatter.format(conv_id)
    if isinstance(expected, list):
        for exp in expected:
            assert exp in result.markdown_text, f"Failed {desc}: {exp} not found"
    else:
        assert expected in result.markdown_text, f"Failed {desc}"


@pytest.mark.asyncio
async def test_orphaned_attachments_section(workspace_env: WorkspaceEnv) -> None:
    db_path = db_setup(workspace_env)
    conv_id = "orphan-att-conv"
    (
        ConversationBuilder(db_path, conv_id)
        .title("Test")
        .add_message("m1", role="user", text="Hello")
        .add_attachment(
            attachment_id="orphan-att",
            message_id=None,
            mime_type="image/png",
            size_bytes=2048,
            provider_meta={"name": "OrphanFile.png"},
        )
        .save()
    )
    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = await formatter.format(conv_id)
    assert "## attachments" in result.markdown_text
    assert "OrphanFile.png" in result.markdown_text


METADATA_CASES = [
    ("counts", {"messages": 5, "attachments": 3}, "message and attachment counts"),
    ("timestamps", {"created": "2024-01-01T10:00:00Z", "updated": "2024-01-15T15:30:00Z"}, "created_at and updated_at"),
]


@pytest.mark.parametrize("label,data,desc", METADATA_CASES)
@pytest.mark.asyncio
async def test_metadata_comprehensive(workspace_env: WorkspaceEnv, label: str, data: JSONRecord, desc: str) -> None:
    db_path = db_setup(workspace_env)
    conv_id = f"meta-{label}-conv"
    if label == "counts":
        message_count = data["messages"]
        attachment_count = data["attachments"]
        assert isinstance(message_count, int)
        assert isinstance(attachment_count, int)
        builder = ConversationBuilder(db_path, conv_id).title("Test")
        for i in range(message_count):
            role = "user" if i % 2 == 0 else "assistant"
            builder.add_message(f"m{i}", role=role, text=f"Message {i}")
        for i in range(attachment_count):
            builder.add_attachment(
                attachment_id=f"att{i}",
                message_id="m0",
                mime_type="text/plain",
                size_bytes=100,
            )
        builder.save()
        formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
        result = await formatter.format(conv_id)
        assert result.metadata.message_count == message_count, f"Failed {desc}"
        assert result.metadata.attachment_count == attachment_count, f"Failed {desc}"
    elif label == "timestamps":
        created = data["created"]
        updated = data["updated"]
        assert isinstance(created, str)
        assert isinstance(updated, str)
        (ConversationBuilder(db_path, conv_id).title("Test").created_at(created).updated_at(updated).save())
        formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
        result = await formatter.format(conv_id)
        assert result.metadata.created_at == created, f"Failed {desc}"
        assert result.metadata.updated_at == updated, f"Failed {desc}"


MARKDOWN_STRUCTURE_CASES = [
    ("header", [], "header structure"),
    (
        "roles",
        [
            {"role": "user", "text": "User message"},
            {"role": "assistant", "text": "Assistant message"},
            {"role": "system", "text": "System message"},
        ],
        "role sections",
    ),
    (
        "empty messages",
        [
            {"role": "user", "text": "Real content"},
            {"role": "tool", "text": ""},
            {"role": "system", "text": "   "},
        ],
        "empty messages skipped",
    ),
    ("null role", [{"role": None, "text": "No role"}], "null role defaults to 'message'"),
]


@pytest.mark.parametrize("label,messages_data,desc", MARKDOWN_STRUCTURE_CASES)
@pytest.mark.asyncio
async def test_markdown_structure_comprehensive(
    workspace_env: WorkspaceEnv, label: str, messages_data: list[JSONRecord], desc: str
) -> None:
    db_path = db_setup(workspace_env)
    conv_id = f"md-{label}-conv"
    builder = ConversationBuilder(db_path, conv_id).provider("chatgpt").title("My Chat Title")
    for i, msg in enumerate(messages_data):
        role = msg["role"]
        text = msg["text"]
        assert role is None or isinstance(role, str)
        assert isinstance(text, str)
        builder.add_message(
            f"m{i}",
            role=role,
            text=text,
            timestamp=f"2024-01-01T10:00:{i:02d}Z",
        )
    builder.save()
    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = await formatter.format(conv_id)
    if label == "header":
        assert_contains_all(
            result.markdown_text,
            "# My Chat Title",
            "Provider: chatgpt",
            f"Conversation ID: {conv_id}",
        )
    elif label == "roles":
        assert_contains_all(result.markdown_text, "## user", "## assistant", "## system")
    elif label == "empty messages":
        assert "## user" in result.markdown_text
        assert_not_contains_any(result.markdown_text, "## tool", "## system")
    elif label == "null role":
        assert "## message" in result.markdown_text


# =============================================================================
# GOLDEN / SNAPSHOT TESTS (from test_golden.py)
# =============================================================================


def normalize_markdown(text: str) -> str:
    """Normalize markdown for comparison (handle whitespace differences)."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"/tmp/(?:nix-shell\.[^/]+/)?pytest-of-[^)>\s]+", "$TMPDIR", text)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip() + "\n"


def assert_golden(name: str, actual: str) -> None:
    """Compare rendered output against a golden reference file."""
    golden_path = GOLDEN_DIR / f"{name}.md"
    normalized = normalize_markdown(actual)

    if os.environ.get("UPDATE_GOLDEN"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(normalized)
        return

    if not golden_path.exists():
        pytest.fail(f"Golden file not found: {golden_path}\nRun with UPDATE_GOLDEN=1 to generate golden files.")

    expected = normalize_markdown(golden_path.read_text())
    assert normalized == expected, (
        f"Rendered output differs from golden file: {golden_path}\n"
        "Run with UPDATE_GOLDEN=1 to regenerate after intentional changes."
    )


class TestGoldenMarkdownRendering:
    """Test markdown rendering against golden reference files."""

    @pytest.mark.asyncio
    async def test_chatgpt_simple_conversation(
        self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path
    ) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-chatgpt-simple"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Simple ChatGPT Conversation",
            messages=[
                {"id": "msg1", "role": "user", "text": "Hello, how are you?", "timestamp": "2024-01-01T12:00:00Z"},
                {
                    "id": "msg2",
                    "role": "assistant",
                    "text": "I'm doing well, thank you for asking! How can I help you today?",
                    "timestamp": "2024-01-01T12:00:05Z",
                },
                {
                    "id": "msg3",
                    "role": "user",
                    "text": "Can you explain what markdown is?",
                    "timestamp": "2024-01-01T12:00:15Z",
                },
                {
                    "id": "msg4",
                    "role": "assistant",
                    "text": "Markdown is a lightweight markup language for creating formatted text using a plain-text editor.",
                    "timestamp": "2024-01-01T12:00:20Z",
                },
            ],
        )
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = await formatter.format(conv_id)
        assert "# Simple ChatGPT Conversation" in formatted.markdown_text
        assert "Provider: chatgpt" in formatted.markdown_text
        assert "## user" in formatted.markdown_text
        assert "## assistant" in formatted.markdown_text
        assert "Hello, how are you?" in formatted.markdown_text
        assert "Markdown is a lightweight markup language" in formatted.markdown_text
        assert "_Timestamp: 2024-01-01T12:00:00Z_" in formatted.markdown_text
        assert_golden("chatgpt-simple", formatted.markdown_text)

    @pytest.mark.asyncio
    async def test_claude_with_thinking_blocks(
        self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path
    ) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-claude-thinking"
        factory.create_conversation(
            id=conv_id,
            provider="claude-ai",
            title="Claude with Thinking",
            messages=[
                {"id": "msg1", "role": "user", "text": "What is 2+2?", "timestamp": "2024-01-15T10:00:00Z"},
                {
                    "id": "msg2",
                    "role": "assistant",
                    "text": "<thinking>\nThis is a simple arithmetic question. 2+2 equals 4.\n</thinking>\n\nThe answer is 4.",
                    "timestamp": "2024-01-15T10:00:05Z",
                },
            ],
        )
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = await formatter.format(conv_id)
        assert "<thinking>" in formatted.markdown_text
        assert "</thinking>" in formatted.markdown_text
        assert "This is a simple arithmetic question" in formatted.markdown_text
        assert "The answer is 4" in formatted.markdown_text
        assert_golden("claude-thinking", formatted.markdown_text)

    @pytest.mark.asyncio
    async def test_json_tool_use_formatted(self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-tool-use"
        factory.create_conversation(
            id=conv_id,
            provider="claude-code",
            title="Tool Use Example",
            messages=[
                {
                    "id": "msg1",
                    "role": "user",
                    "text": "List files in current directory",
                    "timestamp": "2024-02-01T09:00:00Z",
                },
                {
                    "id": "msg2",
                    "role": "assistant",
                    "text": '{"tool": "bash", "command": "ls -la"}',
                    "timestamp": "2024-02-01T09:00:03Z",
                },
            ],
        )
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = await formatter.format(conv_id)
        assert "```json" in formatted.markdown_text
        assert '"tool": "bash"' in formatted.markdown_text
        assert "```" in formatted.markdown_text
        assert_golden("tool-use-json", formatted.markdown_text)

    @pytest.mark.asyncio
    async def test_empty_messages_skipped(self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-empty-messages"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Conversation with Empty Messages",
            messages=[
                {"id": "msg1", "role": "user", "text": "Hello", "timestamp": "2024-03-01T14:00:00Z"},
                {"id": "msg2", "role": "assistant", "text": "", "timestamp": "2024-03-01T14:00:01Z"},
                {"id": "msg3", "role": "user", "text": "Are you there?", "timestamp": "2024-03-01T14:00:10Z"},
                {"id": "msg4", "role": "assistant", "text": "Yes, I'm here!", "timestamp": "2024-03-01T14:00:15Z"},
            ],
        )
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = await formatter.format(conv_id)
        assistant_count = formatted.markdown_text.count("## assistant")
        assert assistant_count == 1, f"Expected 1 assistant section, got {assistant_count}"
        assert_golden("empty-messages", formatted.markdown_text)

    @pytest.mark.asyncio
    async def test_unicode_content_preserved(self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-unicode"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Unicode Test: \u4f60\u597d\u4e16\u754c \U0001f30d",
            messages=[
                {
                    "id": "msg1",
                    "role": "user",
                    "text": "Hello in Chinese: \u4f60\u597d",
                    "timestamp": "2024-04-01T08:00:00Z",
                },
                {
                    "id": "msg2",
                    "role": "assistant",
                    "text": "\u4f60\u597d! That means 'hello' in Chinese. \U0001f1e8\U0001f1f3",
                    "timestamp": "2024-04-01T08:00:05Z",
                },
                {
                    "id": "msg3",
                    "role": "user",
                    "text": "Math symbols: \u2211, \u222b, \u221a, \u03c0, \u221e",
                    "timestamp": "2024-04-01T08:00:15Z",
                },
            ],
        )
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = await formatter.format(conv_id)
        assert "\u4f60\u597d\u4e16\u754c \U0001f30d" in formatted.markdown_text
        assert "\u4f60\u597d" in formatted.markdown_text
        assert "\U0001f1e8\U0001f1f3" in formatted.markdown_text
        assert "\u2211, \u222b, \u221a, \u03c0, \u221e" in formatted.markdown_text
        assert_golden("unicode", formatted.markdown_text)

    @pytest.mark.asyncio
    async def test_attachments_formatted_as_links(
        self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path
    ) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-attachments"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Conversation with Attachments",
            messages=[
                {
                    "id": "msg1",
                    "role": "user",
                    "text": "Here's a screenshot",
                    "timestamp": "2024-05-01T11:00:00Z",
                    "attachments": [
                        {
                            "id": "att1",
                            "mime_type": "image/png",
                            "size_bytes": 12345,
                            "meta": {"name": "screenshot.png"},
                        },
                    ],
                },
            ],
        )
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = await formatter.format(conv_id)
        assert "- Attachment:" in formatted.markdown_text
        has_name_or_id = "screenshot.png" in formatted.markdown_text or "att1" in formatted.markdown_text
        assert has_name_or_id, "Attachment reference not found in output"
        assert_golden("attachments", formatted.markdown_text)

    @pytest.mark.asyncio
    async def test_message_ordering_by_timestamp(
        self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path
    ) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-ordering"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Message Ordering Test",
            messages=[
                {"id": "msg3", "role": "user", "text": "Third message", "timestamp": "2024-01-01T12:00:30Z"},
                {"id": "msg1", "role": "user", "text": "First message", "timestamp": "2024-01-01T12:00:00Z"},
                {"id": "msg2", "role": "assistant", "text": "Second message", "timestamp": "2024-01-01T12:00:15Z"},
            ],
        )
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = await formatter.format(conv_id)
        first_pos = formatted.markdown_text.find("First message")
        second_pos = formatted.markdown_text.find("Second message")
        third_pos = formatted.markdown_text.find("Third message")
        assert first_pos < second_pos < third_pos, "Messages should be ordered by timestamp"
        assert_golden("ordering", formatted.markdown_text)


class TestGoldenFileStructure:
    """Test file structure and naming conventions."""

    @pytest.mark.asyncio
    async def test_markdown_renderer_output_path(
        self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path
    ) -> None:
        factory = DbFactory(db_path)
        conv_id = "test-file-structure"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="File Structure Test",
            messages=[{"id": "msg1", "role": "user", "text": "Test message"}],
        )
        renderer = MarkdownRendererDirect(workspace_env["archive_root"])
        output_path = await renderer.render(conv_id, tmp_path)
        assert output_path.exists()
        assert output_path.name == "conversation.md"
        assert "chatgpt" in str(output_path.parent)

    @pytest.mark.asyncio
    async def test_multiple_conversations_isolated(
        self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path
    ) -> None:
        factory = DbFactory(db_path)
        conv1_id = "test-conv-1"
        conv2_id = "test-conv-2"
        factory.create_conversation(
            id=conv1_id,
            provider="chatgpt",
            title="Conversation 1",
            messages=[{"id": "msg1", "role": "user", "text": "Message 1"}],
        )
        factory.create_conversation(
            id=conv2_id,
            provider="claude-ai",
            title="Conversation 2",
            messages=[{"id": "msg2", "role": "user", "text": "Message 2"}],
        )
        renderer = MarkdownRendererDirect(workspace_env["archive_root"])
        path1 = await renderer.render(conv1_id, tmp_path)
        path2 = await renderer.render(conv2_id, tmp_path)
        assert path1.parent != path2.parent
        assert path1.exists() and path2.exists()


class TestGoldenEdgeCases:
    """Test edge cases in rendering."""

    @pytest.mark.asyncio
    async def test_very_long_text_not_truncated(
        self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path
    ) -> None:
        factory = DbFactory(db_path)
        long_text = "This is a very long message. " * 1000
        conv_id = "golden-long-text"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Long Text Test",
            messages=[{"id": "msg1", "role": "user", "text": long_text}],
        )
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = await formatter.format(conv_id)
        expected_min_len = len(long_text)
        actual_len = len(formatted.markdown_text)
        assert actual_len >= expected_min_len
        occurrences = formatted.markdown_text.count("This is a very long message.")
        assert occurrences >= 999

    @pytest.mark.asyncio
    async def test_special_markdown_chars_not_double_escaped(
        self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path
    ) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-markdown-chars"
        text_with_markdown = "This has **bold**, *italic*, `code`, [links](url), and # headers"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Markdown Chars Test",
            messages=[{"id": "msg1", "role": "user", "text": text_with_markdown, "timestamp": "2024-06-01T16:00:00Z"}],
        )
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = await formatter.format(conv_id)
        assert "**bold**" in formatted.markdown_text
        assert "*italic*" in formatted.markdown_text
        assert "`code`" in formatted.markdown_text
        assert "[links](url)" in formatted.markdown_text
        assert "# headers" in formatted.markdown_text
        assert_golden("markdown-chars", formatted.markdown_text)

    @pytest.mark.asyncio
    async def test_messages_with_timestamps_rendered(
        self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path
    ) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-with-timestamp"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Timestamp Test",
            messages=[
                {"id": "msg1", "role": "user", "text": "Message with timestamp", "timestamp": "2024-01-01T12:00:00Z"}
            ],
        )
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = await formatter.format(conv_id)
        assert "_Timestamp: 2024-01-01T12:00:00Z_" in formatted.markdown_text
        assert_golden("timestamp", formatted.markdown_text)
