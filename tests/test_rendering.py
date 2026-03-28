"""Consolidated rendering tests.

SYSTEMATIZATION: Merged from:
- test_renderers.py (Renderer implementations)
- test_rendering_core.py (Core formatting)

This file contains tests for:
- Renderer factory and implementations (HTML, Markdown)
- Conversation formatting
- Message ordering
- JSON output
- Attachments
"""
from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.rendering.core import ConversationFormatter, FormattedConversation
from polylogue.rendering.renderers import (
    HTMLRenderer,
    MarkdownRenderer,
    create_renderer,
    list_formats,
)
from polylogue.storage.backends.sqlite import open_connection
from polylogue.storage.store import store_records
from tests.helpers import (
    ConversationBuilder,
    assert_contains_all,
    assert_messages_ordered,
    assert_not_contains_any,
    db_setup,
    make_conversation,
    make_message,
)


# =============================================================================
# RENDERER IMPLEMENTATION TESTS (from test_renderers.py)
# =============================================================================


@pytest.fixture
def sample_conversation_id():
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
        make_message("msg2", "test-conv-1", role="assistant", text="Of course! How can I help you today?", timestamp="2024-01-01T10:00:05Z"),
        make_message("msg3", "test-conv-1", text="I need help with Python testing", timestamp="2024-01-01T10:00:10Z"),
    ]

    with open_connection(None) as conn:
        store_records(conversation=conversation, messages=messages, attachments=[], conn=conn)

    return "test-conv-1"


@pytest.fixture
def sample_conversation_with_json():
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
        make_message("msg2", "test-conv-json", role="assistant", text='{"query": "Python testing", "results": ["pytest", "unittest"]}', timestamp="2024-01-01T10:00:05Z"),
    ]

    with open_connection(None) as conn:
        store_records(conversation=conversation, messages=messages, attachments=[], conn=conn)

    return "test-conv-json"


class TestMarkdownRenderer:
    """Tests for MarkdownRenderer."""

    def test_supports_format(self, workspace_env):
        """Test that MarkdownRenderer reports correct format."""
        renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        assert renderer.supports_format() == "markdown"

    def test_render_basic_conversation(self, workspace_env, sample_conversation_id):
        """Test rendering a basic conversation to Markdown."""
        renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "render"

        result_path = renderer.render(sample_conversation_id, output_path)

        assert result_path.exists()
        assert result_path.suffix == ".md"
        content = result_path.read_text()

        # Verify basic structure
        assert "# Test Conversation" in content
        assert "Provider: test-provider" in content
        assert "Conversation ID: test-conv-1" in content

        # Verify messages
        assert "## user" in content
        assert "Hello, can you help me?" in content
        assert "## assistant" in content
        assert "Of course! How can I help you today?" in content
        assert "I need help with Python testing" in content

    def test_render_with_json_formatting(self, workspace_env, sample_conversation_with_json):
        """Test that JSON content is formatted in code blocks."""
        renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "render"

        result_path = renderer.render(sample_conversation_with_json, output_path)
        content = result_path.read_text()

        # Should wrap JSON in code block
        assert "```json" in content
        assert '"query": "Python testing"' in content
        assert '"results"' in content

    def test_render_nonexistent_conversation(self, workspace_env):
        """Test that rendering nonexistent conversation raises error."""
        renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "render"

        with pytest.raises(ValueError, match="Conversation not found"):
            renderer.render("nonexistent-id", output_path)

    def test_render_creates_output_directory(self, workspace_env, sample_conversation_id):
        """Test that render creates output directory if it doesn't exist."""
        renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "custom" / "nested" / "render"

        assert not output_path.exists()

        result_path = renderer.render(sample_conversation_id, output_path)

        assert result_path.exists()
        assert result_path.parent.exists()


class TestHTMLRenderer:
    """Tests for HTMLRenderer."""

    def test_supports_format(self, workspace_env):
        """Test that HTMLRenderer reports correct format."""
        renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])
        assert renderer.supports_format() == "html"

    def test_render_basic_conversation(self, workspace_env, sample_conversation_id):
        """Test rendering a basic conversation to HTML."""
        renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "render"

        result_path = renderer.render(sample_conversation_id, output_path)

        assert result_path.exists()
        assert result_path.suffix == ".html"
        content = result_path.read_text()

        # Verify HTML structure
        assert "<!doctype html>" in content.lower()
        assert "<html" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "<title>Test Conversation</title>" in content

        # Verify content is present (rendered from markdown)
        assert "Test Conversation" in content
        assert "test-provider" in content
        assert "Hello, can you help me?" in content

    def test_render_nonexistent_conversation(self, workspace_env):
        """Test that rendering nonexistent conversation raises error."""
        renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "render"

        with pytest.raises(ValueError, match="Conversation not found"):
            renderer.render("nonexistent-id", output_path)

    def test_render_with_json_content(self, workspace_env, sample_conversation_with_json):
        """Test rendering conversation with JSON content."""
        renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])
        output_path = workspace_env["archive_root"] / "render"

        result_path = renderer.render(sample_conversation_with_json, output_path)
        content = result_path.read_text()

        # JSON should be in code block
        assert "<code>" in content or "<pre>" in content
        assert "Python testing" in content


class TestRendererFactory:
    """Tests for renderer factory functions."""

    def test_create_markdown_renderer(self, workspace_env):
        """Test creating a Markdown renderer via factory."""

        config = Config(
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
            sources=[],
        )

        renderer = create_renderer("markdown", config)

        assert isinstance(renderer, MarkdownRenderer)
        assert renderer.supports_format() == "markdown"

    def test_create_html_renderer(self, workspace_env):
        """Test creating an HTML renderer via factory."""

        config = Config(
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
            sources=[],
        )

        renderer = create_renderer("html", config)

        assert isinstance(renderer, HTMLRenderer)
        assert renderer.supports_format() == "html"

    def test_create_renderer_case_insensitive(self, workspace_env):
        """Test that format parameter is case-insensitive."""

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

    def test_create_renderer_unsupported_format(self, workspace_env):
        """Test that unsupported format raises ValueError."""

        config = Config(
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
            sources=[],
        )

        with pytest.raises(ValueError, match="Unsupported format: json"):
            create_renderer("json", config)

    def test_list_formats(self):
        """Test listing supported formats."""
        formats = list_formats()

        assert isinstance(formats, list)
        assert "markdown" in formats
        assert "html" in formats
        assert len(formats) == 2


class TestRendererIntegration:
    """Integration tests for renderers."""

    def test_both_renderers_produce_output(self, workspace_env, sample_conversation_id):
        """Test that both renderers produce valid output for the same conversation."""
        md_renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        html_renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])

        output_path = workspace_env["archive_root"] / "render"

        md_path = md_renderer.render(sample_conversation_id, output_path)
        html_path = html_renderer.render(sample_conversation_id, output_path)

        # Both should exist
        assert md_path.exists()
        assert html_path.exists()

        # Both should contain the conversation content
        md_content = md_path.read_text()
        html_content = html_path.read_text()

        assert "Test Conversation" in md_content
        assert "Test Conversation" in html_content
        assert "Hello, can you help me?" in md_content
        assert "Hello, can you help me?" in html_content

    def test_protocol_compliance(self, workspace_env):
        """Test that renderers implement the OutputRenderer protocol."""
        from polylogue.protocols import OutputRenderer

        md_renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
        html_renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])

        # Check protocol compliance
        assert isinstance(md_renderer, OutputRenderer)
        assert isinstance(html_renderer, OutputRenderer)

        # Check required methods exist
        assert hasattr(md_renderer, "render")
        assert hasattr(md_renderer, "supports_format")
        assert hasattr(html_renderer, "render")
        assert hasattr(html_renderer, "supports_format")

# =============================================================================
# RENDERING CORE TESTS (from test_rendering_core.py)
# =============================================================================


# =============================================================================
# FORMATTED CONVERSATION DATACLASS - KEPT AS-IS (2 tests)
# =============================================================================


def test_formatted_conversation_dataclass_fields():
    """FormattedConversation has expected fields."""
    fc = FormattedConversation(
        title="Test Title",
        provider="chatgpt",
        conversation_id="conv-123",
        markdown_text="# Test\n\nContent",
        metadata={"message_count": 5},
    )
    assert fc.title == "Test Title"
    assert fc.provider == "chatgpt"
    assert fc.conversation_id == "conv-123"
    assert fc.markdown_text == "# Test\n\nContent"
    assert fc.metadata == {"message_count": 5}


def test_formatted_conversation_dataclass_equality():
    """Two FormattedConversations with same data are equal."""
    fc1 = FormattedConversation(
        title="Test", provider="claude", conversation_id="c1", markdown_text="md", metadata={}
    )
    fc2 = FormattedConversation(
        title="Test", provider="claude", conversation_id="c1", markdown_text="md", metadata={}
    )
    assert fc1 == fc2


# =============================================================================
# INITIALIZATION - PARAMETRIZED (1 test replacing 2)
# =============================================================================


INIT_CASES = [
    ("basic path", "accepts and stores"),
    ("path object", "works with Path object"),
]


@pytest.mark.parametrize("label,desc", INIT_CASES)
def test_formatter_initialization_comprehensive(tmp_path, label, desc):
    """Comprehensive initialization test.

    Replaces 2 individual tests from TestConversationFormatterInit.
    """
    if label == "basic path":
        formatter = ConversationFormatter(tmp_path)
        assert formatter.archive_root == tmp_path

    elif label == "path object":
        archive = tmp_path / "archive"
        archive.mkdir()
        formatter = ConversationFormatter(archive)
        assert formatter.archive_root == archive


# =============================================================================
# FORMAT METHOD - PARAMETRIZED (1 test replacing 3)
# =============================================================================


FORMAT_CASES = [
    ("missing conversation", "nonexistent-conv", "raises ValueError"),
    ("basic conversation", "basic-conv", "returns FormattedConversation"),
    ("null title", "no-title-conv", "uses conversation_id as title"),
]


@pytest.mark.parametrize("label,conv_id,desc", FORMAT_CASES)
def test_formatter_format_comprehensive(workspace_env, label, conv_id, desc):
    """Comprehensive format method test.

    Replaces 3 individual tests from TestConversationFormatterFormat.
    """
    db_path = db_setup(workspace_env)
    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)

    if label == "missing conversation":
        # Don't create conversation
        with pytest.raises(ValueError, match="Conversation not found"):
            formatter.format(conv_id)

    elif label == "basic conversation":
        # Create conversation with message
        (ConversationBuilder(db_path, conv_id)
         .add_message("m1", role="user", text="Hello!")
         .save())

        result = formatter.format(conv_id)

        assert isinstance(result, FormattedConversation)
        assert result.title == "Test Conversation"
        assert result.provider == "test"
        assert result.conversation_id == conv_id
        assert "Hello!" in result.markdown_text
        assert result.metadata["message_count"] == 1

    elif label == "null title":
        # Create conversation with title=None
        (ConversationBuilder(db_path, conv_id)
         .title(None)
         .save())

        result = formatter.format(conv_id)

        assert result.title == conv_id
        assert f"# {conv_id}" in result.markdown_text


# =============================================================================
# MESSAGE ORDERING - PARAMETRIZED (1 test replacing 3)
# =============================================================================


MESSAGE_ORDERING_CASES = [
    ("timestamp order", "ordered-conv", [
        ("m3", "Third", "2024-01-01T12:00:30Z"),
        ("m1", "First", "2024-01-01T12:00:10Z"),
        ("m2", "Second", "2024-01-01T12:00:20Z"),
    ], "timestamp ascending"),

    ("null timestamps", "null-ts-conv", [
        ("m1", "Timestamped", "2024-01-01T12:00:00Z"),
        ("m2", "NoTimestamp", None),
    ], "null timestamps sort last"),

    ("epoch timestamps", "epoch-conv", [
        ("m1", "LaterEpoch", "1704110400.5"),
        ("m2", "EarlierEpoch", "1704106800"),
    ], "numeric epoch timestamps"),
]


@pytest.mark.parametrize("label,conv_id,message_data,desc", MESSAGE_ORDERING_CASES)
def test_message_ordering_comprehensive(workspace_env, label, conv_id, message_data, desc):
    """Comprehensive message ordering test.

    Replaces 3 individual tests from TestMessageOrdering.
    """
    db_path = db_setup(workspace_env)

    # Build messages from test data
    builder = ConversationBuilder(db_path, conv_id)
    for i, (msg_id, text, timestamp) in enumerate(message_data):
        role = "user" if i % 2 == 0 else "assistant"
        builder.add_message(msg_id, role=role, text=text, timestamp=timestamp)
    builder.save()

    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = formatter.format(conv_id)

    if label == "timestamp order":
        # Messages should appear in chronological order
        assert_messages_ordered(result.markdown_text, "First", "Second", "Third")

    elif label == "null timestamps":
        # Timestamped before null
        assert_messages_ordered(result.markdown_text, "Timestamped", "NoTimestamp")

    elif label == "epoch timestamps":
        # Earlier epoch before later
        assert_messages_ordered(result.markdown_text, "EarlierEpoch", "LaterEpoch")


# =============================================================================
# JSON TEXT WRAPPING - PARAMETRIZED (1 test replacing 5)
# =============================================================================


JSON_WRAPPING_CASES = [
    # Valid JSON is wrapped in ```json blocks
    ('{"key": "value", "count": 42}', True, "JSON object wrapped"),
    ('[1, 2, 3, "four"]', True, "JSON array wrapped"),
    ('{malformed json without closing', False, "invalid JSON not wrapped"),
    ("{this is not json}", False, "JSON-like but not JSON"),
    ("This is just regular text.", False, "plain text not wrapped"),
]


@pytest.mark.parametrize("text,wrapped,desc", JSON_WRAPPING_CASES)
def test_json_text_wrapping_comprehensive(workspace_env, text, wrapped, desc):
    """Comprehensive JSON text wrapping test.

    Replaces 5 individual tests from TestJSONTextWrapping.
    """
    db_path = db_setup(workspace_env)
    conv_id = f"json-{hash(text) % 10000}-conv"

    # Create conversation with tool message containing the text
    (ConversationBuilder(db_path, conv_id)
     .title("Test")
     .add_message("m1", role="tool", text=text)
     .save())

    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = formatter.format(conv_id)

    if wrapped:
        # Should have JSON code block
        assert "```json" in result.markdown_text, f"Failed {desc}"
        assert "```" in result.markdown_text
        # JSON may be pretty-printed, so check for key elements instead of exact match
        import json
        try:
            parsed = json.loads(text)
            # For objects, check for a key
            if isinstance(parsed, dict):
                # Check one of the keys is present
                assert any(key in result.markdown_text for key in parsed.keys()), f"Failed {desc}: JSON keys not found"
            elif isinstance(parsed, list):
                # Check list markers are present
                assert "[" in result.markdown_text and "]" in result.markdown_text, f"Failed {desc}: JSON array markers not found"
        except json.JSONDecodeError:
            pass  # If it can't be parsed, test will fail elsewhere
    else:
        # Should NOT have JSON code block
        assert "```json" not in result.markdown_text, f"Failed {desc}"
        # Original text should be preserved as-is
        assert text in result.markdown_text, f"Failed {desc}: text not found"


# =============================================================================
# TIMESTAMP RENDERING - PARAMETRIZED (1 test replacing 2)
# =============================================================================


TIMESTAMP_RENDERING_CASES = [
    ("2024-01-15T10:30:00Z", True, "timestamps rendered"),
    (None, False, "no timestamp when null"),
]


@pytest.mark.parametrize("timestamp,rendered,desc", TIMESTAMP_RENDERING_CASES)
def test_timestamp_rendering_comprehensive(workspace_env, timestamp, rendered, desc):
    """Comprehensive timestamp rendering test.

    Replaces 2 individual tests from TestTimestampRendering.
    """
    db_path = db_setup(workspace_env)
    conv_id = f"ts-{hash(str(timestamp)) % 10000}-conv"

    # Create conversation with message having given timestamp
    (ConversationBuilder(db_path, conv_id)
     .title("Test")
     .add_message("m1", role="user", text="Hello", timestamp=timestamp)
     .save())

    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = formatter.format(conv_id)

    if rendered:
        # Should show timestamp line
        assert f"_Timestamp: {timestamp}_" in result.markdown_text, f"Failed {desc}"
    else:
        # Should NOT show timestamp line
        assert "_Timestamp:" not in result.markdown_text, f"Failed {desc}"


# =============================================================================
# ATTACHMENT HANDLING - PARAMETRIZED (1 test replacing 8)
# =============================================================================


ATTACHMENT_CASES = [
    # Name extraction precedence
    ("meta.name", {"name": "MyFile.pdf"}, "MyFile.pdf", "name from meta.name"),
    ("meta.provider_id", {"provider_id": "provider_file_123"}, "provider_file_123", "name from provider_id"),
    ("meta.drive_id", {"drive_id": "1ABC123XYZ"}, "1ABC123XYZ", "name from drive_id"),
    ("fallback ID", None, "att-fallback-123", "fallback to attachment_id"),
    ("empty meta", {}, "att-empty-meta", "fallback when meta empty"),

    # Multiple attachments
    ("multiple", [
        {"id": "att1", "meta": {"name": "File1.png"}},
        {"id": "att2", "meta": {"name": "File2.jpg"}},
        {"id": "att3", "meta": {"name": "File3.txt"}},
    ], ["File1.png", "File2.jpg", "File3.txt"], "multiple attachments"),

    # Path usage
    ("path", {"name": "Doc.pdf"}, "/custom/path/to/file.pdf", "uses explicit path"),
]


@pytest.mark.parametrize("label,meta,expected,desc", ATTACHMENT_CASES)
def test_attachment_handling_comprehensive(workspace_env, label, meta, expected, desc):
    """Comprehensive attachment handling test.

    Replaces 8 individual tests from TestAttachmentHandling.
    """
    db_path = db_setup(workspace_env)
    conv_id = f"att-{label}-conv"

    # Create conversation with attachments
    builder = (ConversationBuilder(db_path, conv_id)
               .title("Test")
               .add_message("m1", role="user", text="See attachment"))

    # Build attachment records
    if label == "multiple":
        # Multiple attachments
        for att in meta:
            builder.add_attachment(
                attachment_id=att["id"],
                message_id="m1",
                provider_meta=att.get("meta"),
            )
    elif label == "path":
        # Attachment with explicit path
        builder.add_attachment(
            attachment_id="att1",
            message_id="m1",
            path=expected,
            provider_meta=meta,
        )
    else:
        # Single attachment with various meta scenarios
        att_id = expected if meta is None or meta == {} else "att1"
        builder.add_attachment(
            attachment_id=att_id,
            message_id="m1",
            provider_meta=meta,
        )

    builder.save()

    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = formatter.format(conv_id)

    # Verify expected string(s) in output
    if isinstance(expected, list):
        for exp in expected:
            assert exp in result.markdown_text, f"Failed {desc}: {exp} not found"
    else:
        assert expected in result.markdown_text, f"Failed {desc}"


# =============================================================================
# ORPHANED ATTACHMENTS - KEPT AS-IS (1 test)
# =============================================================================


def test_orphaned_attachments_section(workspace_env):
    """Attachments without message_id grouped in ## attachments section."""
    db_path = db_setup(workspace_env)
    conv_id = "orphan-att-conv"

    # Orphaned attachment - message_id is None
    (ConversationBuilder(db_path, conv_id)
     .title("Test")
     .add_message("m1", role="user", text="Hello")
     .add_attachment(
         attachment_id="orphan-att",
         message_id=None,  # No associated message
         mime_type="image/png",
         size_bytes=2048,
         provider_meta={"name": "OrphanFile.png"},
     )
     .save())

    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = formatter.format(conv_id)

    assert "## attachments" in result.markdown_text
    assert "OrphanFile.png" in result.markdown_text


# =============================================================================
# METADATA - PARAMETRIZED (1 test replacing 2)
# =============================================================================


METADATA_CASES = [
    ("counts", {"messages": 5, "attachments": 3}, "message and attachment counts"),
    ("timestamps", {"created": "2024-01-01T10:00:00Z", "updated": "2024-01-15T15:30:00Z"}, "created_at and updated_at"),
]


@pytest.mark.parametrize("label,data,desc", METADATA_CASES)
def test_metadata_comprehensive(workspace_env, label, data, desc):
    """Comprehensive metadata test.

    Replaces 2 individual tests from TestMetadata.
    """
    db_path = db_setup(workspace_env)
    conv_id = f"meta-{label}-conv"

    if label == "counts":
        # Create conversation with messages and attachments
        builder = ConversationBuilder(db_path, conv_id).title("Test")

        for i in range(data["messages"]):
            role = "user" if i % 2 == 0 else "assistant"
            builder.add_message(f"m{i}", role=role, text=f"Message {i}")

        for i in range(data["attachments"]):
            builder.add_attachment(
                attachment_id=f"att{i}",
                message_id="m0",
                mime_type="text/plain",
                size_bytes=100,
            )

        builder.save()

        formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
        result = formatter.format(conv_id)

        assert result.metadata["message_count"] == data["messages"], f"Failed {desc}"
        assert result.metadata["attachment_count"] == data["attachments"], f"Failed {desc}"

    elif label == "timestamps":
        # Create conversation with specific timestamps
        (ConversationBuilder(db_path, conv_id)
         .title("Test")
         .created_at(data["created"])
         .updated_at(data["updated"])
         .save())

        formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
        result = formatter.format(conv_id)

        assert result.metadata["created_at"] == data["created"], f"Failed {desc}"
        assert result.metadata["updated_at"] == data["updated"], f"Failed {desc}"


# =============================================================================
# MARKDOWN STRUCTURE - PARAMETRIZED (1 test replacing 4)
# =============================================================================


MARKDOWN_STRUCTURE_CASES = [
    ("header", [], "header structure"),
    ("roles", [
        {"role": "user", "text": "User message"},
        {"role": "assistant", "text": "Assistant message"},
        {"role": "system", "text": "System message"},
    ], "role sections"),
    ("empty messages", [
        {"role": "user", "text": "Real content"},
        {"role": "tool", "text": ""},  # Empty
        {"role": "system", "text": "   "},  # Whitespace
    ], "empty messages skipped"),
    ("null role", [{"role": None, "text": "No role"}], "null role defaults to 'message'"),
]


@pytest.mark.parametrize("label,messages_data,desc", MARKDOWN_STRUCTURE_CASES)
def test_markdown_structure_comprehensive(workspace_env, label, messages_data, desc):
    """Comprehensive markdown structure test.

    Replaces 4 individual tests from TestMarkdownStructure.
    """
    db_path = db_setup(workspace_env)
    conv_id = f"md-{label}-conv"

    # Build conversation
    builder = (ConversationBuilder(db_path, conv_id)
               .provider("chatgpt")
               .title("My Chat Title"))

    # Build messages
    for i, msg in enumerate(messages_data):
        builder.add_message(
            f"m{i}",
            role=msg["role"],
            text=msg["text"],
            timestamp=f"2024-01-01T10:00:{i:02d}Z",
        )

    builder.save()

    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = formatter.format(conv_id)

    if label == "header":
        # Check header structure
        assert_contains_all(
            result.markdown_text,
            "# My Chat Title",
            "Provider: chatgpt",
            f"Conversation ID: {conv_id}",
        )

    elif label == "roles":
        # Each role should have section
        assert_contains_all(result.markdown_text, "## user", "## assistant", "## system")

    elif label == "empty messages":
        # Only non-empty message should appear
        assert "## user" in result.markdown_text
        assert_not_contains_any(result.markdown_text, "## tool", "## system")

    elif label == "null role":
        # Null role defaults to 'message'
        assert "## message" in result.markdown_text
