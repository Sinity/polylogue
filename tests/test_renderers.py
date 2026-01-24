"""Tests for renderer implementations."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.rendering.renderers import (
    HTMLRenderer,
    MarkdownRenderer,
    create_renderer,
    list_formats,
)
from polylogue.storage.db import open_connection
from polylogue.storage.store import ConversationRecord, MessageRecord, store_records


@pytest.fixture
def sample_conversation_id():
    """Create a sample conversation for testing."""
    conversation = ConversationRecord(
        conversation_id="test-conv-1",
        provider_name="test-provider",
        provider_conversation_id="test-conv-1",
        title="Test Conversation",
        created_at="2024-01-01T10:00:00Z",
        updated_at="2024-01-01T10:00:10Z",
        content_hash="test-hash-1",
        provider_meta={"source": "test"},
    )

    messages = [
        MessageRecord(
            message_id="msg1",
            conversation_id="test-conv-1",
            role="user",
            text="Hello, can you help me?",
            timestamp="2024-01-01T10:00:00Z",
            content_hash="hash-msg1",
        ),
        MessageRecord(
            message_id="msg2",
            conversation_id="test-conv-1",
            role="assistant",
            text="Of course! How can I help you today?",
            timestamp="2024-01-01T10:00:05Z",
            content_hash="hash-msg2",
        ),
        MessageRecord(
            message_id="msg3",
            conversation_id="test-conv-1",
            role="user",
            text="I need help with Python testing",
            timestamp="2024-01-01T10:00:10Z",
            content_hash="hash-msg3",
        ),
    ]

    with open_connection(None) as conn:
        store_records(conversation=conversation, messages=messages, attachments=[], conn=conn)

    return "test-conv-1"


@pytest.fixture
def sample_conversation_with_json():
    """Create a conversation with JSON content (tool use)."""
    conversation = ConversationRecord(
        conversation_id="test-conv-json",
        provider_name="test-provider",
        provider_conversation_id="test-conv-json",
        title="JSON Test",
        created_at="2024-01-01T10:00:00Z",
        updated_at="2024-01-01T10:00:05Z",
        content_hash="test-hash-json",
        provider_meta={"source": "test"},
    )

    messages = [
        MessageRecord(
            message_id="msg1",
            conversation_id="test-conv-json",
            role="user",
            text="Search for Python testing",
            timestamp="2024-01-01T10:00:00Z",
            content_hash="hash-json1",
        ),
        MessageRecord(
            message_id="msg2",
            conversation_id="test-conv-json",
            role="assistant",
            text='{"query": "Python testing", "results": ["pytest", "unittest"]}',
            timestamp="2024-01-01T10:00:05Z",
            content_hash="hash-json2",
        ),
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
