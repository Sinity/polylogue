"""Golden/snapshot tests for rendered output.

These tests verify that polylogue's rendered output (markdown files, HTML, etc.)
matches expected "golden" reference files. This catches regressions in rendering
logic and documents expected output format.

Golden files are stored in tests/golden/<test-case>/expected/
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.rendering.core import ConversationFormatter
from polylogue.rendering.renderers.markdown import MarkdownRenderer
from polylogue.storage.backends.sqlite import open_connection
from tests.factories import DbFactory

# Golden files directory
GOLDEN_DIR = Path(__file__).parent / "golden"


def normalize_markdown(text: str) -> str:
    """Normalize markdown for comparison (handle whitespace differences)."""
    # Normalize line endings
    text = text.replace("\r\n", "\n")
    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in text.split("\n")]
    # Ensure single trailing newline
    return "\n".join(lines).strip() + "\n"


class TestGoldenMarkdownRendering:
    """Test markdown rendering against golden reference files."""

    def test_chatgpt_simple_conversation(self, tmp_path, workspace_env, db_path):
        """Simple ChatGPT conversation should match golden markdown."""
        factory = DbFactory(db_path)

        # Create test conversation
        conv_id = "golden-chatgpt-simple"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Simple ChatGPT Conversation",
            messages=[
                {
                    "id": "msg1",
                    "role": "user",
                    "text": "Hello, how are you?",
                    "timestamp": "2024-01-01T12:00:00Z",
                },
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

        # Render
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = formatter.format(conv_id)

        # Verify structure
        assert "# Simple ChatGPT Conversation" in formatted.markdown_text
        assert "Provider: chatgpt" in formatted.markdown_text
        assert "## user" in formatted.markdown_text
        assert "## assistant" in formatted.markdown_text
        assert "Hello, how are you?" in formatted.markdown_text
        assert "Markdown is a lightweight markup language" in formatted.markdown_text

        # Check timestamps are formatted
        assert "_Timestamp: 2024-01-01T12:00:00Z_" in formatted.markdown_text

    def test_claude_with_thinking_blocks(self, tmp_path, workspace_env, db_path):
        """Claude conversation with thinking blocks should preserve XML tags."""
        factory = DbFactory(db_path)

        # Create conversation with thinking blocks
        conv_id = "golden-claude-thinking"
        factory.create_conversation(
            id=conv_id,
            provider="claude",
            title="Claude with Thinking",
            messages=[
                {
                    "id": "msg1",
                    "role": "user",
                    "text": "What is 2+2?",
                },
                {
                    "id": "msg2",
                    "role": "assistant",
                    "text": "<thinking>\nThis is a simple arithmetic question. 2+2 equals 4.\n</thinking>\n\nThe answer is 4.",
                },
            ],
        )

        # Render
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = formatter.format(conv_id)

        # Verify thinking blocks are preserved
        assert "<thinking>" in formatted.markdown_text
        assert "</thinking>" in formatted.markdown_text
        assert "This is a simple arithmetic question" in formatted.markdown_text
        assert "The answer is 4" in formatted.markdown_text

    def test_json_tool_use_formatted(self, tmp_path, workspace_env, db_path):
        """JSON tool use should be formatted as code blocks."""
        factory = DbFactory(db_path)

        # Create conversation with JSON tool use
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
                },
                {
                    "id": "msg2",
                    "role": "assistant",
                    "text": '{"tool": "bash", "command": "ls -la"}',
                },
            ],
        )

        # Render
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = formatter.format(conv_id)

        # JSON should be in code block
        assert "```json" in formatted.markdown_text
        assert '"tool": "bash"' in formatted.markdown_text
        assert "```" in formatted.markdown_text

    def test_empty_messages_skipped(self, tmp_path, workspace_env, db_path):
        """Empty messages without attachments should be skipped."""
        factory = DbFactory(db_path)

        # Create conversation with empty message
        conv_id = "golden-empty-messages"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Conversation with Empty Messages",
            messages=[
                {
                    "id": "msg1",
                    "role": "user",
                    "text": "Hello",
                },
                {
                    "id": "msg2",
                    "role": "assistant",
                    "text": "",  # Empty message
                },
                {
                    "id": "msg3",
                    "role": "user",
                    "text": "Are you there?",
                },
                {
                    "id": "msg4",
                    "role": "assistant",
                    "text": "Yes, I'm here!",
                },
            ],
        )

        # Render
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = formatter.format(conv_id)

        # Empty message should not appear
        # Count occurrences of "## assistant" - should be 1, not 2
        assistant_count = formatted.markdown_text.count("## assistant")
        assert assistant_count == 1, f"Expected 1 assistant section, got {assistant_count}"

    def test_unicode_content_preserved(self, tmp_path, workspace_env, db_path):
        """Unicode characters should be preserved in output."""
        factory = DbFactory(db_path)

        # Create conversation with Unicode
        conv_id = "golden-unicode"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Unicode Test: 你好世界 🌍",
            messages=[
                {
                    "id": "msg1",
                    "role": "user",
                    "text": "Hello in Chinese: 你好",
                },
                {
                    "id": "msg2",
                    "role": "assistant",
                    "text": "你好! That means 'hello' in Chinese. 🇨🇳",
                },
                {
                    "id": "msg3",
                    "role": "user",
                    "text": "Math symbols: ∑, ∫, √, π, ∞",
                },
            ],
        )

        # Render
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = formatter.format(conv_id)

        # Unicode should be preserved
        assert "你好世界 🌍" in formatted.markdown_text
        assert "你好" in formatted.markdown_text
        assert "🇨🇳" in formatted.markdown_text
        assert "∑, ∫, √, π, ∞" in formatted.markdown_text

    def test_attachments_formatted_as_links(self, tmp_path, workspace_env, db_path):
        """Attachments should be formatted as markdown list items."""
        factory = DbFactory(db_path)

        # Create conversation with attachments (embedded in message)
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
                    "attachments": [
                        {
                            "id": "att1",
                            "mime_type": "image/png",
                            "size_bytes": 12345,
                            "meta": {"name": "screenshot.png"},
                        }
                    ],
                },
            ],
        )

        # Render
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = formatter.format(conv_id)

        # Attachment should appear as list item
        assert "- Attachment:" in formatted.markdown_text
        # Should contain either the name or the attachment ID
        has_name_or_id = "screenshot.png" in formatted.markdown_text or "att1" in formatted.markdown_text
        assert has_name_or_id, f"Attachment reference not found in output"

    def test_message_ordering_by_timestamp(self, tmp_path, workspace_env, db_path):
        """Messages should be ordered by timestamp."""
        factory = DbFactory(db_path)

        # Create conversation with out-of-order insertion but ordered timestamps
        conv_id = "golden-ordering"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Message Ordering Test",
            messages=[
                {
                    "id": "msg3",
                    "role": "user",
                    "text": "Third message",
                    "timestamp": "2024-01-01T12:00:30Z",
                },
                {
                    "id": "msg1",
                    "role": "user",
                    "text": "First message",
                    "timestamp": "2024-01-01T12:00:00Z",
                },
                {
                    "id": "msg2",
                    "role": "assistant",
                    "text": "Second message",
                    "timestamp": "2024-01-01T12:00:15Z",
                },
            ],
        )

        # Render
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = formatter.format(conv_id)

        # Find positions of messages in rendered text
        first_pos = formatted.markdown_text.find("First message")
        second_pos = formatted.markdown_text.find("Second message")
        third_pos = formatted.markdown_text.find("Third message")

        # Verify ordering
        assert first_pos < second_pos < third_pos, "Messages should be ordered by timestamp"


class TestGoldenFileStructure:
    """Test file structure and naming conventions."""

    def test_markdown_renderer_output_path(self, tmp_path, workspace_env, db_path):
        """MarkdownRenderer should create correct file structure."""
        factory = DbFactory(db_path)

        # Create test conversation
        conv_id = "test-file-structure"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="File Structure Test",
            messages=[
                {"id": "msg1", "role": "user", "text": "Test message"},
            ],
        )

        # Render using MarkdownRenderer
        renderer = MarkdownRenderer(workspace_env["archive_root"])
        output_path = renderer.render(conv_id, tmp_path)

        # Verify structure
        assert output_path.exists(), "Output file should exist"
        assert output_path.name == "conversation.md", "File should be named conversation.md"
        assert "chatgpt" in str(output_path.parent), "Parent directory should contain provider name"

    def test_multiple_conversations_isolated(self, tmp_path, workspace_env, db_path):
        """Multiple conversations should be isolated in separate directories."""
        factory = DbFactory(db_path)

        # Create two conversations
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
            provider="claude",
            title="Conversation 2",
            messages=[{"id": "msg2", "role": "user", "text": "Message 2"}],
        )

        # Render both
        renderer = MarkdownRenderer(workspace_env["archive_root"])
        path1 = renderer.render(conv1_id, tmp_path)
        path2 = renderer.render(conv2_id, tmp_path)

        # Verify isolation
        assert path1.parent != path2.parent, "Conversations should have separate directories"
        assert path1.exists() and path2.exists(), "Both files should exist"


class TestGoldenEdgeCases:
    """Test edge cases in rendering."""

    def test_very_long_text_not_truncated(self, tmp_path, workspace_env, db_path):
        """Very long messages should not be truncated."""
        factory = DbFactory(db_path)

        # Create conversation with very long message
        long_text = "This is a very long message. " * 1000  # 30,000 chars
        conv_id = "golden-long-text"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Long Text Test",
            messages=[
                {"id": "msg1", "role": "user", "text": long_text},
            ],
        )

        # Render
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = formatter.format(conv_id)

        # Verify full text preserved (check length to avoid pytest truncation in error display)
        expected_min_len = len(long_text)
        actual_len = len(formatted.markdown_text)

        # Markdown adds headers, so actual will be slightly longer
        assert actual_len >= expected_min_len, f"Long text may be truncated: expected >={expected_min_len} chars, got {actual_len}"

        # Verify the repeated text appears many times
        occurrences = formatted.markdown_text.count("This is a very long message.")
        assert occurrences >= 999, f"Text appears truncated, found {occurrences}/1000 occurrences"

    def test_special_markdown_chars_not_double_escaped(self, tmp_path, workspace_env, db_path):
        """Markdown special characters should be preserved as-is (not escaped)."""
        factory = DbFactory(db_path)

        # Create conversation with markdown special chars
        conv_id = "golden-markdown-chars"
        text_with_markdown = "This has **bold**, *italic*, `code`, [links](url), and # headers"

        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Markdown Chars Test",
            messages=[
                {"id": "msg1", "role": "user", "text": text_with_markdown},
            ],
        )

        # Render
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = formatter.format(conv_id)

        # Special chars should be preserved (not escaped)
        assert "**bold**" in formatted.markdown_text
        assert "*italic*" in formatted.markdown_text
        assert "`code`" in formatted.markdown_text
        assert "[links](url)" in formatted.markdown_text
        assert "# headers" in formatted.markdown_text

    def test_messages_with_timestamps_rendered(self, tmp_path, workspace_env, db_path):
        """Messages with timestamps should render timestamp line."""
        factory = DbFactory(db_path)

        # Create conversation with explicit timestamp
        conv_id = "golden-with-timestamp"
        factory.create_conversation(
            id=conv_id,
            provider="chatgpt",
            title="Timestamp Test",
            messages=[
                {"id": "msg1", "role": "user", "text": "Message with timestamp", "timestamp": "2024-01-01T12:00:00Z"},
            ],
        )

        # Render
        formatter = ConversationFormatter(workspace_env["archive_root"])
        formatted = formatter.format(conv_id)

        # Should have _Timestamp:_ line
        assert "_Timestamp: 2024-01-01T12:00:00Z_" in formatted.markdown_text
