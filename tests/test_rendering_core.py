"""Tests for rendering/core.py - Core rendering utilities.

This test module covers:
- ConversationFormatter initialization and format() method
- Message ordering by timestamp
- Attachment handling and metadata extraction
- JSON text wrapping in code blocks
- Orphaned attachments section
- Edge cases and error handling
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from polylogue.rendering.core import ConversationFormatter, FormattedConversation
from polylogue.storage.db import open_connection
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord, store_records


class TestFormattedConversation:
    """Tests for the FormattedConversation dataclass."""

    def test_dataclass_fields(self):
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

    def test_dataclass_equality(self):
        """Two FormattedConversations with same data are equal."""
        fc1 = FormattedConversation(
            title="Test", provider="claude", conversation_id="c1", markdown_text="md", metadata={}
        )
        fc2 = FormattedConversation(
            title="Test", provider="claude", conversation_id="c1", markdown_text="md", metadata={}
        )
        assert fc1 == fc2


class TestConversationFormatterInit:
    """Tests for ConversationFormatter initialization."""

    def test_initialization_sets_archive_root(self, tmp_path):
        """ConversationFormatter accepts and stores archive_root."""
        formatter = ConversationFormatter(tmp_path)
        assert formatter.archive_root == tmp_path

    def test_initialization_with_path_object(self, tmp_path):
        """ConversationFormatter works with Path object."""
        archive = tmp_path / "archive"
        archive.mkdir()
        formatter = ConversationFormatter(archive)
        assert formatter.archive_root == archive


class TestConversationFormatterFormat:
    """Tests for ConversationFormatter.format() method."""

    def _setup_db(self, workspace_env):
        """Helper to set up database path."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def _create_conversation(
        self,
        db_path: Path,
        conversation_id: str = "test-conv",
        provider: str = "test",
        title: str | None = "Test Conversation",
        messages: list[MessageRecord] | None = None,
        attachments: list[AttachmentRecord] | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """Helper to create a conversation in the database."""
        now = datetime.now(timezone.utc).isoformat()
        conv = ConversationRecord(
            conversation_id=conversation_id,
            provider_name=provider,
            provider_conversation_id=f"ext-{conversation_id}",
            title=title,
            created_at=created_at or now,
            updated_at=updated_at or now,
            content_hash=uuid4().hex,
            version=1,
        )
        with open_connection(db_path) as conn:
            store_records(
                conversation=conv,
                messages=messages or [],
                attachments=attachments or [],
                conn=conn,
            )

    def test_format_missing_conversation_raises(self, workspace_env):
        """ValueError raised for non-existent conversation."""
        self._setup_db(workspace_env)
        formatter = ConversationFormatter(workspace_env["archive_root"])

        with pytest.raises(ValueError, match="Conversation not found"):
            formatter.format("nonexistent-conv")

    def test_format_basic_conversation(self, workspace_env):
        """Returns FormattedConversation with all fields."""
        db_path = self._setup_db(workspace_env)
        conv_id = "basic-conv"

        messages = [
            MessageRecord(
                message_id="m1",
                conversation_id=conv_id,
                role="user",
                text="Hello!",
                timestamp=datetime.now(timezone.utc).isoformat(),
                content_hash="hash1",
                version=1,
            )
        ]
        self._create_conversation(db_path, conv_id, messages=messages)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert isinstance(result, FormattedConversation)
        assert result.title == "Test Conversation"
        assert result.provider == "test"
        assert result.conversation_id == conv_id
        assert "Hello!" in result.markdown_text
        assert result.metadata["message_count"] == 1

    def test_format_uses_conversation_id_when_title_null(self, workspace_env):
        """Uses conversation_id as title when title is None."""
        db_path = self._setup_db(workspace_env)
        conv_id = "no-title-conv"

        self._create_conversation(db_path, conv_id, title=None)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert result.title == conv_id
        assert f"# {conv_id}" in result.markdown_text


class TestMessageOrdering:
    """Tests for message ordering by timestamp."""

    def _setup_db(self, workspace_env):
        """Helper to set up database path."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def _create_conversation(self, db_path: Path, conversation_id: str, messages: list[MessageRecord]):
        """Helper to create a conversation with messages."""
        conv = ConversationRecord(
            conversation_id=conversation_id,
            provider_name="test",
            provider_conversation_id=f"ext-{conversation_id}",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash=uuid4().hex,
            version=1,
        )
        with open_connection(db_path) as conn:
            store_records(conversation=conv, messages=messages, attachments=[], conn=conn)

    def test_messages_ordered_by_timestamp(self, workspace_env):
        """Messages are ordered by timestamp ascending."""
        db_path = self._setup_db(workspace_env)
        conv_id = "ordered-conv"

        # Create messages out of order
        messages = [
            MessageRecord(
                message_id="m3",
                conversation_id=conv_id,
                role="user",
                text="Third",
                timestamp="2024-01-01T12:00:30Z",
                content_hash="h3",
                version=1,
            ),
            MessageRecord(
                message_id="m1",
                conversation_id=conv_id,
                role="user",
                text="First",
                timestamp="2024-01-01T12:00:10Z",
                content_hash="h1",
                version=1,
            ),
            MessageRecord(
                message_id="m2",
                conversation_id=conv_id,
                role="assistant",
                text="Second",
                timestamp="2024-01-01T12:00:20Z",
                content_hash="h2",
                version=1,
            ),
        ]
        self._create_conversation(db_path, conv_id, messages)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        # Check order in markdown
        first_idx = result.markdown_text.index("First")
        second_idx = result.markdown_text.index("Second")
        third_idx = result.markdown_text.index("Third")
        assert first_idx < second_idx < third_idx

    def test_null_timestamps_sort_last(self, workspace_env):
        """Messages with null timestamps appear after timestamped messages."""
        db_path = self._setup_db(workspace_env)
        conv_id = "null-ts-conv"

        messages = [
            MessageRecord(
                message_id="m1",
                conversation_id=conv_id,
                role="user",
                text="Timestamped",
                timestamp="2024-01-01T12:00:00Z",
                content_hash="h1",
                version=1,
            ),
            MessageRecord(
                message_id="m2",
                conversation_id=conv_id,
                role="assistant",
                text="NoTimestamp",
                timestamp=None,
                content_hash="h2",
                version=1,
            ),
        ]
        self._create_conversation(db_path, conv_id, messages)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        # Timestamped should appear before NoTimestamp
        ts_idx = result.markdown_text.index("Timestamped")
        no_ts_idx = result.markdown_text.index("NoTimestamp")
        assert ts_idx < no_ts_idx

    def test_numeric_epoch_timestamps(self, workspace_env):
        """Numeric epoch timestamps are handled correctly."""
        db_path = self._setup_db(workspace_env)
        conv_id = "epoch-conv"

        messages = [
            MessageRecord(
                message_id="m1",
                conversation_id=conv_id,
                role="user",
                text="LaterEpoch",
                timestamp="1704110400.5",  # Numeric epoch with decimal
                content_hash="h1",
                version=1,
            ),
            MessageRecord(
                message_id="m2",
                conversation_id=conv_id,
                role="assistant",
                text="EarlierEpoch",
                timestamp="1704106800",  # Earlier numeric epoch
                content_hash="h2",
                version=1,
            ),
        ]
        self._create_conversation(db_path, conv_id, messages)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        # Earlier should appear first
        earlier_idx = result.markdown_text.index("EarlierEpoch")
        later_idx = result.markdown_text.index("LaterEpoch")
        assert earlier_idx < later_idx


class TestJSONTextWrapping:
    """Tests for JSON text wrapping in code blocks."""

    def _setup_db(self, workspace_env):
        """Helper to set up database path."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def _create_conversation(self, db_path: Path, conversation_id: str, text: str):
        """Helper to create a conversation with a single message."""
        conv = ConversationRecord(
            conversation_id=conversation_id,
            provider_name="test",
            provider_conversation_id=f"ext-{conversation_id}",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash=uuid4().hex,
            version=1,
        )
        msg = MessageRecord(
            message_id="m1",
            conversation_id=conversation_id,
            role="tool",
            text=text,
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_hash="h1",
            version=1,
        )
        with open_connection(db_path) as conn:
            store_records(conversation=conv, messages=[msg], attachments=[], conn=conn)

    def test_json_object_wrapped_in_code_blocks(self, workspace_env):
        """JSON object text is wrapped in ```json code block."""
        db_path = self._setup_db(workspace_env)
        conv_id = "json-obj-conv"
        json_text = '{"key": "value", "count": 42}'

        self._create_conversation(db_path, conv_id, json_text)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "```json" in result.markdown_text
        assert '"key": "value"' in result.markdown_text
        assert "```" in result.markdown_text

    def test_json_array_wrapped_in_code_blocks(self, workspace_env):
        """JSON array text is wrapped in ```json code block."""
        db_path = self._setup_db(workspace_env)
        conv_id = "json-arr-conv"
        json_text = '[1, 2, 3, "four"]'

        self._create_conversation(db_path, conv_id, json_text)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "```json" in result.markdown_text
        assert "```" in result.markdown_text

    def test_invalid_json_not_wrapped(self, workspace_env):
        """Invalid JSON (e.g., {malformed) stays as plain text."""
        db_path = self._setup_db(workspace_env)
        conv_id = "bad-json-conv"
        bad_json = '{malformed json without closing'

        self._create_conversation(db_path, conv_id, bad_json)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        # Should not have json code block
        assert "```json" not in result.markdown_text
        # Original text should be preserved
        assert bad_json in result.markdown_text

    def test_json_like_but_not_json(self, workspace_env):
        """Text starting with { but not valid JSON is not wrapped."""
        db_path = self._setup_db(workspace_env)
        conv_id = "fake-json-conv"
        not_json = "{this is not json}"

        self._create_conversation(db_path, conv_id, not_json)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "```json" not in result.markdown_text
        assert not_json in result.markdown_text

    def test_plain_text_not_wrapped(self, workspace_env):
        """Plain text is not wrapped in code blocks."""
        db_path = self._setup_db(workspace_env)
        conv_id = "plain-conv"
        plain_text = "This is just regular text."

        self._create_conversation(db_path, conv_id, plain_text)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "```json" not in result.markdown_text
        assert plain_text in result.markdown_text


class TestTimestampRendering:
    """Tests for timestamp rendering in messages."""

    def _setup_db(self, workspace_env):
        """Helper to set up database path."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def _create_conversation(self, db_path: Path, conversation_id: str, timestamp: str | None):
        """Helper to create a conversation with a message having given timestamp."""
        conv = ConversationRecord(
            conversation_id=conversation_id,
            provider_name="test",
            provider_conversation_id=f"ext-{conversation_id}",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash=uuid4().hex,
            version=1,
        )
        msg = MessageRecord(
            message_id="m1",
            conversation_id=conversation_id,
            role="user",
            text="Hello",
            timestamp=timestamp,
            content_hash="h1",
            version=1,
        )
        with open_connection(db_path) as conn:
            store_records(conversation=conv, messages=[msg], attachments=[], conn=conn)

    def test_timestamps_rendered(self, workspace_env):
        """Messages with timestamps show _Timestamp: ..._ line."""
        db_path = self._setup_db(workspace_env)
        conv_id = "ts-conv"
        timestamp = "2024-01-15T10:30:00Z"

        self._create_conversation(db_path, conv_id, timestamp)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert f"_Timestamp: {timestamp}_" in result.markdown_text

    def test_no_timestamp_line_when_null(self, workspace_env):
        """No timestamp line when timestamp is None."""
        db_path = self._setup_db(workspace_env)
        conv_id = "no-ts-conv"

        self._create_conversation(db_path, conv_id, None)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "_Timestamp:" not in result.markdown_text


class TestAttachmentHandling:
    """Tests for attachment handling and rendering."""

    def _setup_db(self, workspace_env):
        """Helper to set up database path."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def _create_conversation_with_attachments(
        self,
        db_path: Path,
        conversation_id: str,
        attachments: list[dict],
        message_id: str = "m1",
    ):
        """Helper to create a conversation with attachments."""
        conv = ConversationRecord(
            conversation_id=conversation_id,
            provider_name="test",
            provider_conversation_id=f"ext-{conversation_id}",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash=uuid4().hex,
            version=1,
        )
        msg = MessageRecord(
            message_id=message_id,
            conversation_id=conversation_id,
            role="user",
            text="See attachment",
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_hash="h1",
            version=1,
        )
        att_recs = [
            AttachmentRecord(
                attachment_id=att["id"],
                conversation_id=conversation_id,
                message_id=att.get("message_id", message_id),
                mime_type=att.get("mime_type", "application/octet-stream"),
                size_bytes=att.get("size_bytes", 1024),
                path=att.get("path"),
                provider_meta=att.get("meta"),  # dict or None
            )
            for att in attachments
        ]
        with open_connection(db_path) as conn:
            store_records(conversation=conv, messages=[msg], attachments=att_recs, conn=conn)

    def test_attachment_name_from_meta_name(self, workspace_env):
        """Uses provider_meta.name first for attachment label."""
        db_path = self._setup_db(workspace_env)
        conv_id = "att-name-conv"

        self._create_conversation_with_attachments(
            db_path,
            conv_id,
            [{"id": "att1", "meta": {"name": "MyFile.pdf"}}],
        )

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "MyFile.pdf" in result.markdown_text

    def test_attachment_name_from_provider_id(self, workspace_env):
        """Falls back to provider_meta.provider_id for attachment label."""
        db_path = self._setup_db(workspace_env)
        conv_id = "att-pid-conv"

        self._create_conversation_with_attachments(
            db_path,
            conv_id,
            [{"id": "att1", "meta": {"provider_id": "provider_file_123"}}],
        )

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "provider_file_123" in result.markdown_text

    def test_attachment_name_from_drive_id(self, workspace_env):
        """Falls back to provider_meta.drive_id for attachment label."""
        db_path = self._setup_db(workspace_env)
        conv_id = "att-drive-conv"

        self._create_conversation_with_attachments(
            db_path,
            conv_id,
            [{"id": "att1", "meta": {"drive_id": "1ABC123XYZ"}}],
        )

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "1ABC123XYZ" in result.markdown_text

    def test_attachment_name_fallback_to_id(self, workspace_env):
        """Uses attachment_id as last resort for label."""
        db_path = self._setup_db(workspace_env)
        conv_id = "att-id-conv"

        self._create_conversation_with_attachments(
            db_path,
            conv_id,
            [{"id": "att-fallback-123", "meta": None}],
        )

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "att-fallback-123" in result.markdown_text

    def test_attachment_name_empty_meta(self, workspace_env):
        """Falls back to attachment_id when provider_meta is empty dict."""
        db_path = self._setup_db(workspace_env)
        conv_id = "att-empty-meta-conv"

        self._create_conversation_with_attachments(
            db_path,
            conv_id,
            [{"id": "att-empty-meta", "meta": {}}],
        )

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "att-empty-meta" in result.markdown_text

    def test_multiple_attachments_per_message(self, workspace_env):
        """All attachments on a message are listed."""
        db_path = self._setup_db(workspace_env)
        conv_id = "multi-att-conv"

        self._create_conversation_with_attachments(
            db_path,
            conv_id,
            [
                {"id": "att1", "meta": {"name": "File1.png"}},
                {"id": "att2", "meta": {"name": "File2.jpg"}},
                {"id": "att3", "meta": {"name": "File3.txt"}},
            ],
        )

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "File1.png" in result.markdown_text
        assert "File2.jpg" in result.markdown_text
        assert "File3.txt" in result.markdown_text

    def test_attachment_path_used_when_present(self, workspace_env):
        """Uses explicit path when provided."""
        db_path = self._setup_db(workspace_env)
        conv_id = "att-path-conv"
        custom_path = "/custom/path/to/file.pdf"

        self._create_conversation_with_attachments(
            db_path,
            conv_id,
            [{"id": "att1", "path": custom_path, "meta": {"name": "Doc.pdf"}}],
        )

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert custom_path in result.markdown_text


class TestOrphanedAttachments:
    """Tests for orphaned attachments handling."""

    def _setup_db(self, workspace_env):
        """Helper to set up database path."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def test_orphaned_attachments_section(self, workspace_env):
        """Attachments without message_id grouped in ## attachments section."""
        db_path = self._setup_db(workspace_env)
        conv_id = "orphan-att-conv"

        conv = ConversationRecord(
            conversation_id=conv_id,
            provider_name="test",
            provider_conversation_id=f"ext-{conv_id}",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash=uuid4().hex,
            version=1,
        )
        msg = MessageRecord(
            message_id="m1",
            conversation_id=conv_id,
            role="user",
            text="Hello",
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_hash="h1",
            version=1,
        )
        # Orphaned attachment - message_id is None
        orphan_att = AttachmentRecord(
            attachment_id="orphan-att",
            conversation_id=conv_id,
            message_id=None,  # No associated message
            mime_type="image/png",
            size_bytes=2048,
            path=None,
            provider_meta={"name": "OrphanFile.png"},
        )

        with open_connection(db_path) as conn:
            store_records(conversation=conv, messages=[msg], attachments=[orphan_att], conn=conn)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "## attachments" in result.markdown_text
        assert "OrphanFile.png" in result.markdown_text


class TestMetadata:
    """Tests for metadata extraction."""

    def _setup_db(self, workspace_env):
        """Helper to set up database path."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def test_metadata_counts(self, workspace_env):
        """message_count and attachment_count are correct."""
        db_path = self._setup_db(workspace_env)
        conv_id = "counts-conv"

        conv = ConversationRecord(
            conversation_id=conv_id,
            provider_name="test",
            provider_conversation_id=f"ext-{conv_id}",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash=uuid4().hex,
            version=1,
        )
        messages = [
            MessageRecord(
                message_id=f"m{i}",
                conversation_id=conv_id,
                role="user" if i % 2 == 0 else "assistant",
                text=f"Message {i}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                content_hash=f"h{i}",
                version=1,
            )
            for i in range(5)
        ]
        attachments = [
            AttachmentRecord(
                attachment_id=f"att{i}",
                conversation_id=conv_id,
                message_id="m0",
                mime_type="text/plain",
                size_bytes=100,
                path=None,
                provider_meta=None,
            )
            for i in range(3)
        ]

        with open_connection(db_path) as conn:
            store_records(conversation=conv, messages=messages, attachments=attachments, conn=conn)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert result.metadata["message_count"] == 5
        assert result.metadata["attachment_count"] == 3

    def test_metadata_timestamps(self, workspace_env):
        """created_at and updated_at are extracted."""
        db_path = self._setup_db(workspace_env)
        conv_id = "ts-meta-conv"
        created = "2024-01-01T10:00:00Z"
        updated = "2024-01-15T15:30:00Z"

        conv = ConversationRecord(
            conversation_id=conv_id,
            provider_name="test",
            provider_conversation_id=f"ext-{conv_id}",
            title="Test",
            created_at=created,
            updated_at=updated,
            content_hash=uuid4().hex,
            version=1,
        )

        with open_connection(db_path) as conn:
            store_records(conversation=conv, messages=[], attachments=[], conn=conn)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert result.metadata["created_at"] == created
        assert result.metadata["updated_at"] == updated


class TestMarkdownStructure:
    """Tests for markdown output structure."""

    def _setup_db(self, workspace_env):
        """Helper to set up database path."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def test_markdown_header_structure(self, workspace_env):
        """Title, provider, conversation_id in header."""
        db_path = self._setup_db(workspace_env)
        conv_id = "header-conv"

        conv = ConversationRecord(
            conversation_id=conv_id,
            provider_name="chatgpt",
            provider_conversation_id=f"ext-{conv_id}",
            title="My Chat Title",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash=uuid4().hex,
            version=1,
        )

        with open_connection(db_path) as conn:
            store_records(conversation=conv, messages=[], attachments=[], conn=conn)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "# My Chat Title" in result.markdown_text
        assert "Provider: chatgpt" in result.markdown_text
        assert f"Conversation ID: {conv_id}" in result.markdown_text

    def test_role_sections(self, workspace_env):
        """Each message has ## {role} header."""
        db_path = self._setup_db(workspace_env)
        conv_id = "roles-conv"

        conv = ConversationRecord(
            conversation_id=conv_id,
            provider_name="test",
            provider_conversation_id=f"ext-{conv_id}",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash=uuid4().hex,
            version=1,
        )
        messages = [
            MessageRecord(
                message_id="m1",
                conversation_id=conv_id,
                role="user",
                text="User message",
                timestamp="2024-01-01T10:00:00Z",
                content_hash="h1",
                version=1,
            ),
            MessageRecord(
                message_id="m2",
                conversation_id=conv_id,
                role="assistant",
                text="Assistant message",
                timestamp="2024-01-01T10:00:10Z",
                content_hash="h2",
                version=1,
            ),
            MessageRecord(
                message_id="m3",
                conversation_id=conv_id,
                role="system",
                text="System message",
                timestamp="2024-01-01T10:00:20Z",
                content_hash="h3",
                version=1,
            ),
        ]

        with open_connection(db_path) as conn:
            store_records(conversation=conv, messages=messages, attachments=[], conn=conn)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "## user" in result.markdown_text
        assert "## assistant" in result.markdown_text
        assert "## system" in result.markdown_text

    def test_empty_messages_skipped(self, workspace_env):
        """Messages without text/attachments are omitted."""
        db_path = self._setup_db(workspace_env)
        conv_id = "empty-msg-conv"

        conv = ConversationRecord(
            conversation_id=conv_id,
            provider_name="test",
            provider_conversation_id=f"ext-{conv_id}",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash=uuid4().hex,
            version=1,
        )
        messages = [
            MessageRecord(
                message_id="m1",
                conversation_id=conv_id,
                role="user",
                text="Real content",
                timestamp="2024-01-01T10:00:00Z",
                content_hash="h1",
                version=1,
            ),
            MessageRecord(
                message_id="m2",
                conversation_id=conv_id,
                role="tool",
                text="",  # Empty text
                timestamp="2024-01-01T10:00:10Z",
                content_hash="h2",
                version=1,
            ),
            MessageRecord(
                message_id="m3",
                conversation_id=conv_id,
                role="system",
                text="   ",  # Whitespace only
                timestamp="2024-01-01T10:00:20Z",
                content_hash="h3",
                version=1,
            ),
        ]

        with open_connection(db_path) as conn:
            store_records(conversation=conv, messages=messages, attachments=[], conn=conn)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "## user" in result.markdown_text
        # Empty tool and whitespace-only system messages should be skipped
        assert "## tool" not in result.markdown_text
        assert "## system" not in result.markdown_text

    def test_null_role_uses_message_default(self, workspace_env):
        """Null role defaults to 'message'."""
        db_path = self._setup_db(workspace_env)
        conv_id = "null-role-conv"

        conv = ConversationRecord(
            conversation_id=conv_id,
            provider_name="test",
            provider_conversation_id=f"ext-{conv_id}",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash=uuid4().hex,
            version=1,
        )
        msg = MessageRecord(
            message_id="m1",
            conversation_id=conv_id,
            role=None,  # Null role
            text="No role",
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_hash="h1",
            version=1,
        )

        with open_connection(db_path) as conn:
            store_records(conversation=conv, messages=[msg], attachments=[], conn=conn)

        formatter = ConversationFormatter(workspace_env["archive_root"])
        result = formatter.format(conv_id)

        assert "## message" in result.markdown_text
