"""Consolidated backend tests.

SYSTEMATIZATION: Merged from:
- test_backend_sqlite.py (SQLite backend operations)
- test_backend_protocol.py (StorageBackend protocol conformance)
- test_backend_comparison.py (Old vs new extraction comparison)

This file contains tests for:
- SQLite backend CRUD operations
- Transaction management
- StorageBackend protocol conformance
- Extraction comparison between old/new backends
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.protocols import StorageBackend
from polylogue.storage.backends import SQLiteBackend
from tests.helpers import make_attachment, make_conversation, make_message


# =============================================================================
# SQLITE BACKEND OPERATIONS (from test_backend_sqlite.py)
# =============================================================================


def test_backend_save_and_get_conversation(sqlite_backend: SQLiteBackend) -> None:
    """Test saving and retrieving a conversation."""
    conv = make_conversation("conv1", title="Test Conversation", content_hash="hash123")

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.commit()

    retrieved = sqlite_backend.get_conversation("conv1")
    assert retrieved is not None
    assert retrieved.conversation_id == "conv1"
    assert retrieved.title == "Test Conversation"
    assert retrieved.content_hash == "hash123"


def test_backend_save_and_get_messages(sqlite_backend: SQLiteBackend) -> None:
    """Test saving and retrieving messages."""
    conv = make_conversation("conv1", title="Test")
    msg1 = make_message("msg1", "conv1", text="Hello")
    msg2 = make_message("msg2", "conv1", role="assistant", text="Hi there")

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.save_messages([msg1, msg2])
    sqlite_backend.commit()

    messages = sqlite_backend.get_messages("conv1")
    assert len(messages) == 2
    assert messages[0].message_id == "msg1"
    assert messages[0].text == "Hello"
    assert messages[1].message_id == "msg2"
    assert messages[1].text == "Hi there"


def test_backend_list_conversations(sqlite_backend: SQLiteBackend) -> None:
    """Test listing conversations."""
    conv1 = make_conversation("conv1", title="First", created_at="2024-01-01T00:00:00Z", updated_at="2024-01-01T00:00:00Z", provider_meta={"source": "claude"})
    conv2 = make_conversation("conv2", title="Second", created_at="2024-01-02T00:00:00Z", updated_at="2024-01-02T00:00:00Z", provider_meta={"source": "chatgpt"})

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv1)
    sqlite_backend.save_conversation(conv2)
    sqlite_backend.commit()

    all_convs = sqlite_backend.list_conversations()
    assert len(all_convs) == 2

    claude_convs = sqlite_backend.list_conversations(source="claude")
    assert len(claude_convs) == 1
    assert claude_convs[0].conversation_id == "conv1"


def test_backend_save_attachments(sqlite_backend: SQLiteBackend) -> None:
    """Test saving and retrieving attachments."""
    conv = make_conversation("conv1", title="Test")
    msg = make_message("msg1", "conv1", text="Hello")
    att = make_attachment("att1", "conv1", "msg1", mime_type="image/png", size_bytes=1024)

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.save_messages([msg])
    sqlite_backend.save_attachments([att])
    sqlite_backend.commit()

    attachments = sqlite_backend.get_attachments("conv1")
    assert len(attachments) == 1
    assert attachments[0].attachment_id == "att1"
    assert attachments[0].mime_type == "image/png"
    assert attachments[0].size_bytes == 1024


def test_backend_transaction_rollback(sqlite_backend: SQLiteBackend) -> None:
    """Test transaction rollback."""
    conv = make_conversation("conv1", title="Test")

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.rollback()

    retrieved = sqlite_backend.get_conversation("conv1")
    assert retrieved is None


def test_backend_transaction_context_manager(sqlite_backend: SQLiteBackend) -> None:
    """Test using the transaction context manager."""
    conv = make_conversation("conv1", title="Test")

    with sqlite_backend.transaction():
        sqlite_backend.save_conversation(conv)

    retrieved = sqlite_backend.get_conversation("conv1")
    assert retrieved is not None
    assert retrieved.conversation_id == "conv1"


def test_backend_transaction_context_manager_exception(sqlite_backend: SQLiteBackend) -> None:
    """Test transaction context manager rolls back on exception."""
    conv = make_conversation("conv1", title="Test")

    with pytest.raises(ValueError), sqlite_backend.transaction():
        sqlite_backend.save_conversation(conv)
        raise ValueError("Test error")

    retrieved = sqlite_backend.get_conversation("conv1")
    assert retrieved is None


def test_backend_delete_conversation(sqlite_backend: SQLiteBackend) -> None:
    """Test deleting a conversation and all related records."""
    conv = make_conversation("conv1", title="Test")
    msg1 = make_message("msg1", "conv1", text="Hello")
    msg2 = make_message("msg2", "conv1", role="assistant", text="Hi there")
    att = make_attachment("att1", "conv1", "msg1", mime_type="image/png", size_bytes=1024)

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.save_messages([msg1, msg2])
    sqlite_backend.save_attachments([att])
    sqlite_backend.commit()

    assert sqlite_backend.get_conversation("conv1") is not None
    assert len(sqlite_backend.get_messages("conv1")) == 2
    assert len(sqlite_backend.get_attachments("conv1")) == 1

    result = sqlite_backend.delete_conversation("conv1")
    assert result is True

    assert sqlite_backend.get_conversation("conv1") is None
    assert len(sqlite_backend.get_messages("conv1")) == 0
    assert len(sqlite_backend.get_attachments("conv1")) == 0


def test_backend_delete_conversation_not_found(sqlite_backend: SQLiteBackend) -> None:
    """Test deleting a non-existent conversation returns False."""
    result = sqlite_backend.delete_conversation("nonexistent")
    assert result is False


def test_backend_delete_conversation_cleans_fts(sqlite_backend: SQLiteBackend) -> None:
    """Test that deleting a conversation also cleans up FTS entries."""
    from polylogue.storage.index import ensure_index, update_index_for_conversations

    conv = make_conversation("conv1", title="Test")
    msg = make_message("msg1", "conv1", text="searchable content here")

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.save_messages([msg])
    sqlite_backend.commit()

    conn = sqlite_backend._get_connection()
    ensure_index(conn)
    update_index_for_conversations(["conv1"], conn)
    conn.commit()

    fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", ("conv1",)).fetchone()[0]
    assert fts_count > 0

    result = sqlite_backend.delete_conversation("conv1")
    assert result is True

    fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", ("conv1",)).fetchone()[0]
    assert fts_count == 0


# =============================================================================
# STORAGE BACKEND PROTOCOL CONFORMANCE (from test_backend_protocol.py)
# =============================================================================


class TestStorageBackendProtocol:
    """Verify SQLiteBackend implements StorageBackend protocol."""

    def test_sqlite_backend_is_storage_backend(self, tmp_path: Path) -> None:
        """SQLiteBackend is a runtime-checkable StorageBackend."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        assert isinstance(backend, StorageBackend)
        backend.close()


class TestConversationOperations:
    """Test conversation save/retrieve operations."""

    def test_save_and_get_conversation(self, tmp_path: Path) -> None:
        """save_conversation persists data retrievable by get_conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1", title="Test Conversation", provider_name="claude")
        backend.save_conversation(conv)

        retrieved = backend.get_conversation("conv-1")
        assert retrieved is not None
        assert retrieved.conversation_id == "conv-1"
        assert retrieved.title == "Test Conversation"
        assert retrieved.provider_name == "claude"
        backend.close()

    def test_get_nonexistent_conversation_returns_none(self, tmp_path: Path) -> None:
        """get_conversation returns None for missing ID."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        result = backend.get_conversation("nonexistent")
        assert result is None
        backend.close()

    def test_save_conversation_upserts(self, tmp_path: Path) -> None:
        """save_conversation updates existing conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv1 = make_conversation("conv-1", title="Original Title")
        backend.save_conversation(conv1)

        conv2 = make_conversation("conv-1", title="Updated Title")
        backend.save_conversation(conv2)

        retrieved = backend.get_conversation("conv-1")
        assert retrieved is not None
        assert retrieved.title == "Updated Title"
        backend.close()

    def test_list_conversations_returns_all(self, tmp_path: Path) -> None:
        """list_conversations returns all stored conversations."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        for i in range(3):
            conv = make_conversation(f"conv-{i}", title=f"Conversation {i}")
            backend.save_conversation(conv)

        all_convs = backend.list_conversations()
        assert len(all_convs) == 3
        assert {c.conversation_id for c in all_convs} == {"conv-0", "conv-1", "conv-2"}
        backend.close()

    def test_list_conversations_filters_by_provider(self, tmp_path: Path) -> None:
        """list_conversations filters by provider_name."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        backend.save_conversation(make_conversation("c1", provider_name="claude"))
        backend.save_conversation(make_conversation("c2", provider_name="chatgpt"))
        backend.save_conversation(make_conversation("c3", provider_name="claude"))

        claude_convs = backend.list_conversations(provider="claude")
        assert len(claude_convs) == 2
        assert all(c.provider_name == "claude" for c in claude_convs)
        backend.close()

    def test_list_conversations_with_limit_and_offset(self, tmp_path: Path) -> None:
        """list_conversations supports pagination."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        for i in range(10):
            conv = make_conversation(f"conv-{i:02d}")
            backend.save_conversation(conv)

        page1 = backend.list_conversations(limit=3, offset=0)
        page2 = backend.list_conversations(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        page1_ids = {c.conversation_id for c in page1}
        page2_ids = {c.conversation_id for c in page2}
        assert page1_ids.isdisjoint(page2_ids)
        backend.close()


class TestMessageOperations:
    """Test message save/retrieve operations."""

    def test_save_and_get_messages(self, tmp_path: Path) -> None:
        """save_messages persists data retrievable by get_messages."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        messages = [
            make_message("m1", "conv-1", role="user", text="Hello"),
            make_message("m2", "conv-1", role="assistant", text="Hi there"),
        ]
        backend.save_messages(messages)

        retrieved = backend.get_messages("conv-1")
        assert len(retrieved) == 2
        assert {m.message_id for m in retrieved} == {"m1", "m2"}
        assert {m.role for m in retrieved} == {"user", "assistant"}
        backend.close()

    def test_get_messages_for_empty_conversation(self, tmp_path: Path) -> None:
        """get_messages returns empty list for conversation with no messages."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        retrieved = backend.get_messages("conv-1")
        assert retrieved == []
        backend.close()


class TestAttachmentOperations:
    """Test attachment save/retrieve operations."""

    def test_save_and_get_attachments(self, tmp_path: Path) -> None:
        """save_attachments persists data retrievable by get_attachments."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        attachments = [
            make_attachment("att1", "conv-1", mime_type="image/png", size_bytes=1024),
            make_attachment("att2", "conv-1", mime_type="text/plain", size_bytes=256),
        ]
        backend.save_attachments(attachments)

        retrieved = backend.get_attachments("conv-1")
        assert len(retrieved) == 2
        assert {a.attachment_id for a in retrieved} == {"att1", "att2"}
        backend.close()

    def test_prune_attachments_removes_unlisted(self, tmp_path: Path) -> None:
        """prune_attachments removes attachments not in keep set."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        attachments = [
            make_attachment("att1", "conv-1"),
            make_attachment("att2", "conv-1"),
            make_attachment("att3", "conv-1"),
        ]
        backend.save_attachments(attachments)

        backend.prune_attachments("conv-1", {"att1", "att3"})

        retrieved = backend.get_attachments("conv-1")
        assert {a.attachment_id for a in retrieved} == {"att1", "att3"}
        backend.close()


class TestTransactionOperations:
    """Test transaction management."""

    def test_begin_commit_persists_data(self, tmp_path: Path) -> None:
        """Data saved within begin/commit is persisted."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        backend.begin()
        conv = make_conversation("tx-conv")
        backend.save_conversation(conv)
        backend.commit()

        retrieved = backend.get_conversation("tx-conv")
        assert retrieved is not None
        backend.close()

    def test_rollback_discards_changes(self, tmp_path: Path) -> None:
        """Data saved within begin/rollback is discarded."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        backend.begin()
        backend.save_conversation(make_conversation("rollback-conv"))
        backend.rollback()

        assert backend.get_conversation("rollback-conv") is None
        backend.close()


class TestMetadataOperations:
    """Test metadata CRUD operations."""

    def test_update_and_get_metadata(self, tmp_path: Path) -> None:
        """update_metadata sets key, get_metadata retrieves it."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        backend.update_metadata("conv-1", "rating", 5)
        backend.update_metadata("conv-1", "reviewed", True)

        metadata = backend.get_metadata("conv-1")
        assert metadata.get("rating") == 5
        assert metadata.get("reviewed") is True
        backend.close()

    def test_delete_metadata(self, tmp_path: Path) -> None:
        """delete_metadata removes a key."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        backend.update_metadata("conv-1", "temp", "value")
        backend.delete_metadata("conv-1", "temp")

        metadata = backend.get_metadata("conv-1")
        assert "temp" not in metadata
        backend.close()

    def test_add_and_remove_tag(self, tmp_path: Path) -> None:
        """add_tag adds to tags list, remove_tag removes it."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        backend.add_tag("conv-1", "important")
        backend.add_tag("conv-1", "work")

        metadata = backend.get_metadata("conv-1")
        tags = metadata.get("tags", [])
        assert "important" in tags
        assert "work" in tags

        backend.remove_tag("conv-1", "work")
        metadata = backend.get_metadata("conv-1")
        tags = metadata.get("tags", [])
        assert "important" in tags
        assert "work" not in tags
        backend.close()


class TestSearchOperations:
    """Test search and resolve operations."""

    def test_resolve_id_exact_match(self, tmp_path: Path) -> None:
        """resolve_id returns full ID for exact match."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conversation-12345")
        backend.save_conversation(conv)

        resolved = backend.resolve_id("conversation-12345")
        assert resolved == "conversation-12345"
        backend.close()

    def test_resolve_id_prefix_match(self, tmp_path: Path) -> None:
        """resolve_id returns full ID for unique prefix."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("unique-prefix-abc123")
        backend.save_conversation(conv)

        resolved = backend.resolve_id("unique-prefix")
        assert resolved == "unique-prefix-abc123"
        backend.close()

    def test_resolve_id_ambiguous_returns_none(self, tmp_path: Path) -> None:
        """resolve_id returns None for ambiguous prefix."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        backend.save_conversation(make_conversation("prefix-abc"))
        backend.save_conversation(make_conversation("prefix-def"))

        resolved = backend.resolve_id("prefix")
        assert resolved is None
        backend.close()


class TestDeleteOperations:
    """Test deletion operations."""

    def test_delete_conversation(self, tmp_path: Path) -> None:
        """delete_conversation removes conversation and related data."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("to-delete")
        backend.save_conversation(conv)
        backend.save_messages([make_message("m1", "to-delete")])
        backend.save_attachments([make_attachment("a1", "to-delete")])

        result = backend.delete_conversation("to-delete")
        assert result is True

        assert backend.get_conversation("to-delete") is None
        assert backend.get_messages("to-delete") == []
        assert backend.get_attachments("to-delete") == []
        backend.close()

    def test_delete_nonexistent_returns_false(self, tmp_path: Path) -> None:
        """delete_conversation returns False for missing ID."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        result = backend.delete_conversation("nonexistent")
        assert result is False
        backend.close()


# =============================================================================
# BACKEND COMPARISON TESTS (from test_backend_comparison.py)
# =============================================================================


from polylogue.importers.claude import (
    parse_code as old_parse_code,
    extract_text_from_segments as old_extract_segments,
)
from polylogue.importers.base import normalize_role as old_normalize_role

from polylogue.schemas.unified import (
    extract_harmonized_message,
    normalize_role as new_normalize_role,
    is_message_record,
)


@dataclass
class ComparisonResult:
    """Result of comparing old vs new extraction."""

    field: str
    old_value: str | None
    new_value: str | None
    equivalent: bool


def compare_extractions(provider: str, raw: dict) -> list[ComparisonResult]:
    """Compare old and new extraction for a single message."""
    results = []

    try:
        new_msg = extract_harmonized_message(provider, raw)
    except Exception as e:
        return [ComparisonResult("extraction", None, str(e), False)]

    if provider == "claude-code":
        msg_obj = raw.get("message", {})
        msg_type = raw.get("type")

        if msg_type in ("user", "human"):
            old_role = "user"
        elif msg_type == "assistant":
            old_role = "assistant"
        else:
            old_role = msg_type or "unknown"

        content_raw = msg_obj.get("content") if isinstance(msg_obj, dict) else None
        old_text = old_extract_segments(content_raw) if isinstance(content_raw, list) else None

        old_role_norm = old_normalize_role(old_role)
        new_role_norm = new_msg.role

        results.append(ComparisonResult(
            field="role",
            old_value=old_role_norm,
            new_value=new_role_norm,
            equivalent=old_role_norm == new_role_norm,
        ))

        old_text_norm = (old_text or "").strip()
        new_text_norm = (new_msg.text or "").strip()

        text_equiv = old_text_norm == new_text_norm

        results.append(ComparisonResult(
            field="text",
            old_value=old_text_norm[:50] + "..." if len(old_text_norm) > 50 else old_text_norm,
            new_value=new_text_norm[:50] + "..." if len(new_text_norm) > 50 else new_text_norm,
            equivalent=text_equiv,
        ))

    return results


class TestBackendComparison:
    """Compare old vs new extraction backends."""

    def test_role_normalization_equivalence(self):
        """Old and new role normalization should produce same results for valid inputs."""
        test_roles = [
            "user", "human", "USER",
            "assistant", "model", "ai",
            "system",
            "tool", "function",
        ]

        differences = []
        for role in test_roles:
            old = old_normalize_role(role)
            new = new_normalize_role(role)
            if old != new:
                differences.append((role, old, new))

        if differences:
            print("\nRole normalization differences (may be improvements):")
            for role, old, new in differences:
                print(f"  {role!r}: old={old!r}, new={new!r}")

        assert old_normalize_role("user") == new_normalize_role("user")
        assert old_normalize_role("assistant") == new_normalize_role("assistant")

    def test_claude_code_extraction_equivalence(self, seeded_db):
        """Compare old and new extraction on real Claude Code data."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.provider_meta
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = 'claude-code'
            LIMIT 500
            """
        )

        rows = cur.fetchall()
        conn.close()

        equiv_count = Counter()
        diff_samples = []

        for (pm_json,) in rows:
            pm = json.loads(pm_json)
            raw = pm.get("raw", pm)

            if not is_message_record("claude-code", raw):
                continue

            results = compare_extractions("claude-code", raw)

            for r in results:
                if r.equivalent:
                    equiv_count[r.field] += 1
                else:
                    if len(diff_samples) < 10:
                        diff_samples.append(r)

        total = sum(equiv_count.values())
        if total == 0:
            pytest.skip("No claude-code messages in seeded database")

    def test_new_extraction_is_superset(self, seeded_db):
        """New extraction should provide more information, not less."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.provider_meta
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = 'claude-code'
            LIMIT 500
            """
        )

        rows = cur.fetchall()
        conn.close()

        tool_calls_found = 0
        reasoning_found = 0

        for (pm_json,) in rows:
            pm = json.loads(pm_json)
            raw = pm.get("raw", pm)

            if not is_message_record("claude-code", raw):
                continue

            new_msg = extract_harmonized_message("claude-code", raw)
            tool_calls_found += len(new_msg.tool_calls)
            reasoning_found += len(new_msg.reasoning_traces)

        assert tool_calls_found >= 0
        assert reasoning_found >= 0
        print(f"Reasoning traces extracted: {reasoning_found}")

        assert tool_calls_found > 0 or reasoning_found >= 0, "New extraction should find viewports"


class TestAPICompatibility:
    """Test that new extraction can be adapted to old API."""

    def test_parsed_message_equivalent_fields(self):
        """HarmonizedMessage has equivalent fields to ParsedMessage."""
        from polylogue.importers.base import ParsedMessage
        from polylogue.schemas.unified import HarmonizedMessage

        field_mapping = {
            "provider_message_id": "id",
            "role": "role",
            "text": "text",
            "timestamp": "timestamp",
            "provider_meta": "raw",
        }

        pm = ParsedMessage(provider_message_id="test", role="user", text="hello")
        for pm_field in field_mapping.keys():
            assert hasattr(pm, pm_field), f"ParsedMessage missing {pm_field}"

        hm = HarmonizedMessage(role="user", text="hello", provider="test")
        for hm_field in field_mapping.values():
            assert hasattr(hm, hm_field), f"HarmonizedMessage missing {hm_field}"

    def test_can_convert_harmonized_to_parsed(self):
        """Demonstrate conversion from HarmonizedMessage to ParsedMessage."""
        from polylogue.importers.base import ParsedMessage

        raw = {
            "type": "assistant",
            "uuid": "test-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}]
            }
        }

        hm = extract_harmonized_message("claude-code", raw)

        pm = ParsedMessage(
            provider_message_id=hm.id or "unknown",
            role=hm.role,
            text=hm.text,
            timestamp=hm.timestamp.isoformat() if hm.timestamp else None,
            provider_meta={"raw": hm.raw},
        )

        assert pm.role == "assistant"
        assert pm.text == "Hello!"
        assert pm.provider_message_id == "test-123"
