"""Tests for bugs fixed in this session.

Covers:
1. --stream filter bypass (uses filter_chain when filters active)
2. --delete safety guard (rejects delete without filters)
3. delete_conversation orphaned attachment cleanup
4. ChatGPT multimodal text dict filtering
5. LIKE wildcard injection escaping in title_contains
6. Claude AI ZIP filtering (only conversations.json)
7. detect_provider underscore variant path detection
8. YAML fields split fix (exact field matching)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.storage.backends.sqlite import SQLiteBackend, connection_context
from polylogue.storage.store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
)
from polylogue.lib.models import Conversation, ConversationId, Message


def make_hash(s: str) -> str:
    """Create a 16-char content hash."""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# =============================================================================
# Fixture: In-memory SQLiteBackend with schema
# =============================================================================


@pytest.fixture
def backend():
    """Create an in-memory SQLiteBackend with schema for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        with connection_context(db_path) as conn:
            pass  # Schema is created automatically
        backend = SQLiteBackend(db_path)
        yield backend


# =============================================================================
# 1. --stream filter bypass fix (query.py)
# =============================================================================


class TestStreamFilterBypass:
    """Test that streaming path respects filters via _describe_filters."""

    def test_describe_filters_detects_provider(self):
        """_describe_filters should return truthy when provider filter present."""
        from polylogue.cli.query import _describe_filters

        params = {"provider": "claude", "query": ()}
        result = _describe_filters(params)
        assert result
        assert any("provider: claude" in part for part in result)

    def test_describe_filters_empty_when_no_filters(self):
        """_describe_filters should return empty list when no filters."""
        from polylogue.cli.query import _describe_filters

        params = {"query": (), "latest": True}
        result = _describe_filters(params)
        assert not result

    def test_describe_filters_detects_tag(self):
        """_describe_filters should detect tag filter."""
        from polylogue.cli.query import _describe_filters

        params = {"tag": "important", "query": ()}
        result = _describe_filters(params)
        assert result
        assert any("tag: important" in part for part in result)

    def test_describe_filters_detects_exclude_provider(self):
        """_describe_filters should detect exclude_provider filter."""
        from polylogue.cli.query import _describe_filters

        params = {"exclude_provider": "chatgpt", "query": ()}
        result = _describe_filters(params)
        assert result
        assert any("exclude provider: chatgpt" in part for part in result)

    def test_describe_filters_detects_title(self):
        """_describe_filters should detect title filter."""
        from polylogue.cli.query import _describe_filters

        params = {"title": "test", "query": ()}
        result = _describe_filters(params)
        assert result
        assert any("title: test" in part for part in result)

    def test_describe_filters_detects_since(self):
        """_describe_filters should detect since filter."""
        from polylogue.cli.query import _describe_filters

        params = {"since": "2025-01-01", "query": ()}
        result = _describe_filters(params)
        assert result
        assert any("since: 2025-01-01" in part for part in result)

    def test_describe_filters_detects_until(self):
        """_describe_filters should detect until filter."""
        from polylogue.cli.query import _describe_filters

        params = {"until": "2025-12-31", "query": ()}
        result = _describe_filters(params)
        assert result
        assert any("until: 2025-12-31" in part for part in result)

    def test_describe_filters_detects_conv_id(self):
        """_describe_filters should detect conv_id filter."""
        from polylogue.cli.query import _describe_filters

        params = {"conv_id": "abc123", "query": ()}
        result = _describe_filters(params)
        assert result
        assert any("id: abc123" in part for part in result)

    def test_describe_filters_multiple_filters(self):
        """_describe_filters should detect multiple filters."""
        from polylogue.cli.query import _describe_filters

        params = {
            "provider": "claude",
            "tag": "work",
            "since": "2025-01-01",
            "query": (),
        }
        result = _describe_filters(params)
        assert len(result) == 3


# =============================================================================
# 2. --delete safety guard (query.py)
# =============================================================================


class TestDeleteSafetyGuard:
    """Test that execute_query rejects --delete without filters."""

    def test_delete_without_filters_raises_exit(self, workspace_env):
        """execute_query should raise SystemExit(1) when delete_matched=True and no filters."""
        from polylogue.cli.query import execute_query
        from polylogue.cli.types import AppEnv

        # Mock AppEnv
        env = MagicMock(spec=AppEnv)
        env.ui = MagicMock()
        env.ui.select = MagicMock(return_value=False)

        params = {
            "delete_matched": True,
            # No filters
            "query": (),
            "contains": (),
            "exclude_text": (),
            "provider": None,
            "exclude_provider": None,
            "tag": None,
            "exclude_tag": None,
            "title": None,
            "has_type": (),
            "since": None,
            "until": None,
            "conv_id": None,
            "latest": False,
            "sort": None,
            "reverse": False,
            "limit": None,
            "sample": None,
            "count_only": False,
            "list_mode": True,
            "transform": None,
            "dialogue_only": False,
            "stream": False,
            "set_meta": None,
            "add_tag": None,
        }

        with pytest.raises(SystemExit) as exc_info:
            execute_query(env, params)
        assert exc_info.value.code == 1


# =============================================================================
# 3. delete_conversation orphaned attachment cleanup (sqlite.py)
# =============================================================================


class TestDeleteConversationAttachmentCleanup:
    """Test that delete_conversation properly cleans orphaned attachments."""

    def test_delete_conversation_cleans_orphaned_attachments(self, backend):
        """After deleting a conversation, orphaned attachments should be removed."""
        conn = backend._get_connection()

        # Create a conversation directly via SQL
        conn.execute(
            """INSERT INTO conversations
               (conversation_id, provider_name, provider_conversation_id, title, created_at, updated_at, content_hash, version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "conv-1",
                "test",
                "prov-1",
                "Test Conv",
                "2025-01-01",
                "2025-01-01",
                make_hash("test conv 1"),
                1,
            ),
        )

        # Create a message
        conn.execute(
            """INSERT INTO messages
               (message_id, conversation_id, role, text, content_hash, timestamp, version)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "msg-1",
                "conv-1",
                "user",
                "Hello",
                make_hash("Hello"),
                "2025-01-01T00:00:00Z",
                1,
            ),
        )

        # Create an attachment
        conn.execute(
            """INSERT INTO attachments
               (attachment_id, mime_type, size_bytes, ref_count)
               VALUES (?, ?, ?, ?)""",
            ("att-1", "text/plain", 100, 1),
        )

        # Create an attachment_ref linking message to attachment
        conn.execute(
            """INSERT INTO attachment_refs (ref_id, attachment_id, conversation_id, message_id)
               VALUES (?, ?, ?, ?)""",
            ("ref-1", "att-1", "conv-1", "msg-1"),
        )
        conn.commit()

        # Verify attachment exists
        existing = conn.execute(
            "SELECT 1 FROM attachments WHERE attachment_id = ?", ("att-1",)
        ).fetchone()
        assert existing is not None

        # Delete conversation
        backend.delete_conversation("conv-1")

        # Verify attachment is gone
        orphaned = conn.execute(
            "SELECT 1 FROM attachments WHERE attachment_id = ?", ("att-1",)
        ).fetchone()
        assert orphaned is None

    def test_delete_conversation_preserves_shared_attachments(self, backend):
        """Shared attachments should survive when only one conversation is deleted."""
        conn = backend._get_connection()

        # Create two conversations via SQL
        conn.execute(
            """INSERT INTO conversations
               (conversation_id, provider_name, provider_conversation_id, title, created_at, updated_at, content_hash, version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "conv-1",
                "test",
                "prov-1",
                "Test Conv 1",
                "2025-01-01",
                "2025-01-01",
                make_hash("test conv 1"),
                1,
            ),
        )
        conn.execute(
            """INSERT INTO conversations
               (conversation_id, provider_name, provider_conversation_id, title, created_at, updated_at, content_hash, version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "conv-2",
                "test",
                "prov-2",
                "Test Conv 2",
                "2025-01-02",
                "2025-01-02",
                make_hash("test conv 2"),
                1,
            ),
        )

        # Create messages in each
        conn.execute(
            """INSERT INTO messages
               (message_id, conversation_id, role, text, content_hash, timestamp, version)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("msg-1", "conv-1", "user", "Hello", make_hash("Hello"), "2025-01-01T00:00:00Z", 1),
        )
        conn.execute(
            """INSERT INTO messages
               (message_id, conversation_id, role, text, content_hash, timestamp, version)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("msg-2", "conv-2", "user", "Hello", make_hash("Hello"), "2025-01-02T00:00:00Z", 1),
        )

        # Create a shared attachment referenced by both
        conn.execute(
            """INSERT INTO attachments
               (attachment_id, mime_type, size_bytes, ref_count)
               VALUES (?, ?, ?, ?)""",
            ("att-1", "text/plain", 100, 2),
        )

        # Create attachment_refs for both messages
        conn.execute(
            "INSERT INTO attachment_refs (ref_id, attachment_id, conversation_id, message_id) VALUES (?, ?, ?, ?)",
            ("ref-1", "att-1", "conv-1", "msg-1"),
        )
        conn.execute(
            "INSERT INTO attachment_refs (ref_id, attachment_id, conversation_id, message_id) VALUES (?, ?, ?, ?)",
            ("ref-2", "att-1", "conv-2", "msg-2"),
        )
        conn.commit()

        # Delete first conversation
        backend.delete_conversation("conv-1")

        # Verify attachment still exists with reduced ref_count
        row = conn.execute(
            "SELECT ref_count FROM attachments WHERE attachment_id = ?", ("att-1",)
        ).fetchone()
        assert row is not None
        assert row["ref_count"] == 1

        # Delete second conversation
        backend.delete_conversation("conv-2")

        # Now attachment should be gone
        orphaned = conn.execute(
            "SELECT 1 FROM attachments WHERE attachment_id = ?", ("att-1",)
        ).fetchone()
        assert orphaned is None


# =============================================================================
# 4. ChatGPT multimodal text fix (chatgpt.py)
# =============================================================================


class TestChatGPTMultimodalText:
    """Test that ChatGPT multimodal parts are handled correctly."""

    def test_chatgpt_multimodal_filters_dict_parts(self):
        """Dict parts (like image_asset_pointer) should not produce Python repr."""
        payload = {
            "title": "Test",
            "mapping": {
                "node1": {
                    "id": "node1",
                    "children": ["node2"],
                    "message": None,
                },
                "node2": {
                    "id": "node2",
                    "parent": "node1",
                    "children": [],
                    "message": {
                        "id": "msg1",
                        "author": {"role": "user"},
                        "content": {
                            "content_type": "multimodal_text",
                            "parts": [
                                "Hello, look at this image:",
                                {
                                    "content_type": "image_asset_pointer",
                                    "asset_pointer": "file-service://file-abc123",
                                    "size_bytes": 123,
                                },
                            ],
                        },
                        "create_time": 1700000000.0,
                    },
                },
            },
        }
        from polylogue.sources.parsers.chatgpt import parse

        conv = parse(payload, "test-fallback")
        assert len(conv.messages) > 0
        text = conv.messages[0].text

        # Should NOT contain Python dict repr
        assert "asset_pointer" not in text
        assert "<dict" not in text.lower()
        assert "file-service" not in text

        # Should contain the text part
        assert "Hello, look at this image:" in text

    def test_chatgpt_tether_quote_text_extracted(self):
        """Dict parts with 'text' key should have their text extracted."""
        payload = {
            "title": "Test",
            "mapping": {
                "node1": {
                    "id": "node1",
                    "children": ["node2"],
                    "message": None,
                },
                "node2": {
                    "id": "node2",
                    "parent": "node1",
                    "children": [],
                    "message": {
                        "id": "msg1",
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": [
                                "Here is the quote:",
                                {
                                    "content_type": "tether_quote",
                                    "text": "This is quoted text",
                                    "metadata": {},
                                },
                            ],
                        },
                        "create_time": 1700000000.0,
                    },
                },
            },
        }
        from polylogue.sources.parsers.chatgpt import parse

        conv = parse(payload, "test-fallback")
        assert len(conv.messages) > 0
        text = conv.messages[0].text

        # Should contain both the intro text and the quoted text
        assert "Here is the quote:" in text
        assert "This is quoted text" in text


# =============================================================================
# 5. LIKE wildcard injection fix (sqlite.py)
# =============================================================================


class TestTitleContainsWildcardEscaping:
    """Test that LIKE wildcards in title_contains are escaped."""

    def test_title_contains_escapes_percent_wildcard(self, backend):
        """LIKE % wildcard should be escaped in title search."""
        # Create conversations with titles containing % and x
        conv1 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="test",
            provider_conversation_id="prov-1",
            title="100% done",
            created_at="2025-01-01",
            updated_at="2025-01-01",
            content_hash=make_hash("100% done"),
        )
        conv2 = ConversationRecord(
            conversation_id="conv-2",
            provider_name="test",
            provider_conversation_id="prov-2",
            title="100x done",
            created_at="2025-01-02",
            updated_at="2025-01-02",
            content_hash=make_hash("100x done"),
        )
        backend.save_conversation(conv1)
        backend.save_conversation(conv2)

        # Search for "100%"
        results = backend.list_conversations(title_contains="100%")
        assert len(results) == 1
        assert results[0].title == "100% done"

    def test_title_contains_escapes_underscore_wildcard(self, backend):
        """LIKE _ wildcard should be escaped in title search."""
        # Create conversations
        conv1 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="test",
            provider_conversation_id="prov-1",
            title="100_ done",
            created_at="2025-01-01",
            updated_at="2025-01-01",
            content_hash=make_hash("100_ done"),
        )
        conv2 = ConversationRecord(
            conversation_id="conv-2",
            provider_name="test",
            provider_conversation_id="prov-2",
            title="100x done",
            created_at="2025-01-02",
            updated_at="2025-01-02",
            content_hash=make_hash("100x done"),
        )
        backend.save_conversation(conv1)
        backend.save_conversation(conv2)

        # Search for "100_"
        results = backend.list_conversations(title_contains="100_")
        assert len(results) == 1
        assert results[0].title == "100_ done"

    def test_title_contains_escapes_backslash(self, backend):
        """Backslashes should be escaped in title search."""
        # Create a conversation with backslash
        conv = ConversationRecord(
            conversation_id="conv-1",
            provider_name="test",
            provider_conversation_id="prov-1",
            title="C:\\Users\\test",
            created_at="2025-01-01",
            updated_at="2025-01-01",
            content_hash=make_hash("C:\\Users\\test"),
        )
        backend.save_conversation(conv)

        # Search for "C:\Users\test" - should find it
        results = backend.list_conversations(title_contains="C:\\Users\\test")
        assert len(results) == 1
        assert "C:" in results[0].title


# =============================================================================
# 6. Claude AI ZIP filter (source.py)
# =============================================================================


class TestClaudeAIZIPFiltering:
    """Test that Claude AI ZIP exports only process conversations.json."""

    def test_iter_source_conversations_skips_non_conversations_in_claude_zip(
        self, tmp_path
    ):
        """Claude AI ZIP exports should only process conversations.json."""
        zip_path = tmp_path / "claude-export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            # Add conversations.json (real data)
            zf.writestr(
                "conversations.json",
                json.dumps(
                    [
                        {
                            "title": "Test Conversation",
                            "mapping": {
                                "n1": {"id": "n1", "children": ["n2"], "message": None},
                                "n2": {
                                    "id": "n2",
                                    "parent": "n1",
                                    "children": [],
                                    "message": {
                                        "id": "m1",
                                        "author": {"role": "user"},
                                        "content": {
                                            "content_type": "text",
                                            "parts": ["Hi there"],
                                        },
                                        "create_time": 1700000000.0,
                                    },
                                },
                            },
                        }
                    ]
                ),
            )
            # Add metadata.json (should be skipped)
            zf.writestr(
                "metadata.json",
                json.dumps(
                    {"account": "test@example.com", "export_date": "2025-01-01"}
                ),
            )

        from polylogue.sources.source import iter_source_conversations
        from polylogue.config import Source

        source = Source(name="claude", path=zip_path)
        convs = list(iter_source_conversations(source))

        # Should only have 1 conversation from conversations.json
        assert len(convs) == 1
        assert convs[0].title == "Test Conversation"


# =============================================================================
# 7. detect_provider path gap (source.py)
# =============================================================================


class TestDetectProviderPath:
    """Test provider detection from file paths."""

    def test_detect_provider_claude_code_underscore_in_path(self):
        """Underscore variant in path should detect claude-code."""
        from pathlib import Path
        from polylogue.sources.source import detect_provider

        p = Path("/exports/claude_code/session.jsonl")
        assert detect_provider(None, p) == "claude-code"

    def test_detect_provider_claude_code_dash_in_path(self):
        """Dash variant should also detect claude-code."""
        from pathlib import Path
        from polylogue.sources.source import detect_provider

        p = Path("/exports/claude-code/session.jsonl")
        assert detect_provider(None, p) == "claude-code"

    def test_detect_provider_claude_without_code(self):
        """Plain claude path should detect claude."""
        from pathlib import Path
        from polylogue.sources.source import detect_provider

        p = Path("/exports/claude/data.json")
        assert detect_provider(None, p) == "claude"

    def test_detect_provider_chatgpt(self):
        """ChatGPT path should detect chatgpt."""
        from pathlib import Path
        from polylogue.sources.source import detect_provider

        p = Path("/exports/chatgpt/conversations.json")
        assert detect_provider(None, p) == "chatgpt"

    def test_detect_provider_codex(self):
        """Codex path should detect codex."""
        from pathlib import Path
        from polylogue.sources.source import detect_provider

        p = Path("/exports/codex/sessions.jsonl")
        assert detect_provider(None, p) == "codex"


# =============================================================================
# 8. YAML fields split fix (query.py)
# =============================================================================


class TestYAMLFieldsExactMatch:
    """Test that YAML output uses exact field matching, not substring."""

    def test_conv_to_yaml_excludes_messages_when_not_in_fields(self):
        """YAML output should not include full messages when 'messages' not in fields."""
        from polylogue.cli.query import _conv_to_yaml

        conv = Conversation(
            id=ConversationId("test-1"),
            provider="test",
            title="Test",
            messages=[Message(id="m1", role="user", text="hello world")],
        )

        # With fields="id,title" (no "messages"), should NOT include message list
        result = _conv_to_yaml(conv, "id,title")
        import yaml

        data = yaml.safe_load(result)

        # If messages key exists, it should NOT be a full list of message dicts
        if "messages" in data:
            # Should either be absent or count, not list of dicts
            if isinstance(data["messages"], list):
                # If it's a list, dicts should not have text field
                for msg in data["messages"]:
                    if isinstance(msg, dict):
                        assert "text" not in msg or msg.get("text") is None

    def test_conv_to_yaml_includes_messages_when_in_fields(self):
        """YAML output should include full messages when 'messages' in fields."""
        from polylogue.cli.query import _conv_to_yaml

        conv = Conversation(
            id=ConversationId("test-1"),
            provider="test",
            title="Test",
            messages=[Message(id="m1", role="user", text="hello world")],
        )

        # With fields="id,title,messages", should include messages
        result = _conv_to_yaml(conv, "id,title,messages")
        import yaml

        data = yaml.safe_load(result)

        # Should have messages key with list
        assert "messages" in data
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) > 0
        assert data["messages"][0]["text"] == "hello world"

    def test_conv_to_yaml_includes_all_when_no_fields(self):
        """YAML output should include messages when no fields specified."""
        from polylogue.cli.query import _conv_to_yaml

        conv = Conversation(
            id=ConversationId("test-1"),
            provider="test",
            title="Test",
            messages=[Message(id="m1", role="user", text="hello world")],
        )

        # With fields=None, should include everything
        result = _conv_to_yaml(conv, None)
        import yaml

        data = yaml.safe_load(result)

        # Should have messages with full content
        assert "messages" in data
        assert isinstance(data["messages"], list)
        assert data["messages"][0]["text"] == "hello world"
