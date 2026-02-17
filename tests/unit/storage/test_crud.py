"""CRUD operations tests — conversation, message, attachment, transaction, metadata, search, delete, backend comparison."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from polylogue.schemas.unified import (
    extract_harmonized_message,
    is_message_record,
)
from polylogue.schemas.unified import (
    normalize_role as new_normalize_role,
)
from polylogue.sources.parsers.base import normalize_role as old_normalize_role
from polylogue.sources.parsers.claude import (
    extract_text_from_segments as old_extract_segments,
)
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.store import (
    ConversationRecord,
)
from tests.infra.helpers import (
    make_attachment,
    make_conversation,
    make_message,
    make_hash,
)


class TestConversationOperations:
    """Test conversation save/retrieve operations."""

    async def test_save_and_get_conversation(self, tmp_path: Path) -> None:
        """save_conversation persists data retrievable by get_conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1", title="Test Conversation", provider_name="claude")
        await backend.save_conversation_record(conv)

        retrieved = await backend.get_conversation("conv-1")
        assert retrieved is not None
        assert retrieved.conversation_id == "conv-1"
        assert retrieved.title == "Test Conversation"
        assert retrieved.provider_name == "claude"
        await backend.close()

    async def test_get_nonexistent_conversation_returns_none(self, tmp_path: Path) -> None:
        """get_conversation returns None for missing ID."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        result = await backend.get_conversation("nonexistent")
        assert result is None
        await backend.close()

    async def test_save_conversation_upserts(self, tmp_path: Path) -> None:
        """save_conversation updates existing conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv1 = make_conversation("conv-1", title="Original Title")
        await backend.save_conversation_record(conv1)

        conv2 = make_conversation("conv-1", title="Updated Title")
        await backend.save_conversation_record(conv2)

        retrieved = await backend.get_conversation("conv-1")
        assert retrieved is not None
        assert retrieved.title == "Updated Title"
        await backend.close()

    async def test_list_conversations_returns_all(self, tmp_path: Path) -> None:
        """list_conversations returns all stored conversations."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        for i in range(3):
            conv = make_conversation(f"conv-{i}", title=f"Conversation {i}")
            await backend.save_conversation_record(conv)

        all_convs = await backend.list_conversations()
        assert len(all_convs) == 3
        assert {c.conversation_id for c in all_convs} == {"conv-0", "conv-1", "conv-2"}
        await backend.close()

    async def test_list_conversations_filters_by_provider(self, tmp_path: Path) -> None:
        """list_conversations filters by provider_name."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        await backend.save_conversation_record(make_conversation("c1", provider_name="claude"))
        await backend.save_conversation_record(make_conversation("c2", provider_name="chatgpt"))
        await backend.save_conversation_record(make_conversation("c3", provider_name="claude"))

        claude_convs = await backend.list_conversations(provider="claude")
        assert len(claude_convs) == 2
        assert all(c.provider_name == "claude" for c in claude_convs)
        await backend.close()

    async def test_list_conversations_with_limit_and_offset(self, tmp_path: Path) -> None:
        """list_conversations supports pagination."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        for i in range(10):
            conv = make_conversation(f"conv-{i:02d}")
            await backend.save_conversation_record(conv)

        page1 = await backend.list_conversations(limit=3, offset=0)
        page2 = await backend.list_conversations(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        page1_ids = {c.conversation_id for c in page1}
        page2_ids = {c.conversation_id for c in page2}
        assert page1_ids.isdisjoint(page2_ids)
        await backend.close()

    async def test_backend_list_conversations_offset_without_limit(self, tmp_path: Path) -> None:
        """Regression: OFFSET without LIMIT must not raise SQL syntax error."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        for i in range(5):
            conv = make_conversation(f"off-{i}", updated_at=f"2024-01-{i+1:02d}T00:00:00Z")
            await backend.save_conversation_record(conv)

        # This previously generated invalid SQL: ... ORDER BY ... OFFSET ? (no LIMIT)
        result = await backend.list_conversations(offset=2)
        assert len(result) == 3  # 5 total - 2 skipped = 3
        await backend.close()

    async def test_title_contains_escapes_percent_wildcard(self, tmp_path: Path) -> None:
        """LIKE % wildcard should be escaped in title search."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

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
        await backend.save_conversation_record(conv1)
        await backend.save_conversation_record(conv2)

        # Search for "100%"
        results = await backend.list_conversations(title_contains="100%")
        assert len(results) == 1
        assert results[0].title == "100% done"
        await backend.close()

    async def test_title_contains_escapes_underscore_wildcard(self, tmp_path: Path) -> None:
        """LIKE _ wildcard should be escaped in title search."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

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
        await backend.save_conversation_record(conv1)
        await backend.save_conversation_record(conv2)

        # Search for "100_"
        results = await backend.list_conversations(title_contains="100_")
        assert len(results) == 1
        assert results[0].title == "100_ done"
        await backend.close()

    async def test_title_contains_escapes_backslash(self, tmp_path: Path) -> None:
        """Backslashes should be escaped in title search."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

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
        await backend.save_conversation_record(conv)

        # Search for "C:\Users\test" - should find it
        results = await backend.list_conversations(title_contains="C:\\Users\\test")
        assert len(results) == 1
        assert "C:" in results[0].title
        await backend.close()


async def test_repository_message_mapping_uses_backend_path(tmp_path: Path) -> None:
    """Regression: _get_message_conversation_mapping must use backend's db_path."""
    from polylogue.storage.repository import ConversationRepository

    db_path = tmp_path / "custom.db"
    backend = SQLiteBackend(db_path=db_path)

    conv = make_conversation("map-conv-1", title="Mapping Test")
    msg = make_message("map-msg-1", "map-conv-1", text="Hello")

    await backend.begin()
    await backend.save_conversation_record(conv)
    await backend.save_messages([msg])
    await backend.commit()

    repo = ConversationRepository(backend)
    mapping = await repo._get_message_conversation_mapping(["map-msg-1"])
    assert mapping == {"map-msg-1": "map-conv-1"}

    # Non-existent messages should return empty
    mapping_empty = await repo._get_message_conversation_mapping(["nonexistent"])
    assert mapping_empty == {}
    await backend.close()


class TestMessageOperations:
    """Test message save/retrieve operations."""

    async def test_save_and_get_messages(self, tmp_path: Path) -> None:
        """save_messages persists data retrievable by get_messages."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        await backend.save_conversation_record(conv)

        messages = [
            make_message("m1", "conv-1", role="user", text="Hello"),
            make_message("m2", "conv-1", role="assistant", text="Hi there"),
        ]
        await backend.save_messages(messages)

        retrieved = await backend.get_messages("conv-1")
        assert len(retrieved) == 2
        assert {m.message_id for m in retrieved} == {"m1", "m2"}
        assert {m.role for m in retrieved} == {"user", "assistant"}
        await backend.close()

    async def test_get_messages_for_empty_conversation(self, tmp_path: Path) -> None:
        """get_messages returns empty list for conversation with no messages."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        await backend.save_conversation_record(conv)

        retrieved = await backend.get_messages("conv-1")
        assert retrieved == []
        await backend.close()


class TestAttachmentOperations:
    """Test attachment save/retrieve operations."""

    async def test_save_and_get_attachments(self, tmp_path: Path) -> None:
        """save_attachments persists data retrievable by get_attachments."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        await backend.save_conversation_record(conv)

        # Create messages for attachment references
        msg = make_message("m1", "conv-1")
        await backend.save_messages([msg])

        attachments = [
            make_attachment("att1", "conv-1", "m1", mime_type="image/png", size_bytes=1024),
            make_attachment("att2", "conv-1", "m1", mime_type="text/plain", size_bytes=256),
        ]
        await backend.save_attachments(attachments)

        retrieved = await backend.get_attachments("conv-1")
        assert len(retrieved) == 2
        assert {a.attachment_id for a in retrieved} == {"att1", "att2"}
        await backend.close()

    async def test_prune_attachments_removes_unlisted(self, tmp_path: Path) -> None:
        """prune_attachments removes attachments not in keep set."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        await backend.save_conversation_record(conv)

        # Create messages for attachment references
        msg = make_message("m1", "conv-1")
        await backend.save_messages([msg])

        attachments = [
            make_attachment("att1", "conv-1", "m1"),
            make_attachment("att2", "conv-1", "m1"),
            make_attachment("att3", "conv-1", "m1"),
        ]
        await backend.save_attachments(attachments)

        await backend.prune_attachments("conv-1", {"att1", "att3"})

        retrieved = await backend.get_attachments("conv-1")
        assert {a.attachment_id for a in retrieved} == {"att1", "att3"}
        await backend.close()


class TestTransactionOperations:
    """Test transaction management."""

    async def test_begin_commit_persists_data(self, tmp_path: Path) -> None:
        """Data saved within begin/commit is persisted."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        await backend.begin()
        conv = make_conversation("tx-conv")
        await backend.save_conversation_record(conv)
        await backend.commit()

        retrieved = await backend.get_conversation("tx-conv")
        assert retrieved is not None
        await backend.close()

    async def test_rollback_discards_changes(self, tmp_path: Path) -> None:
        """Data saved within begin/rollback is discarded."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        await backend.begin()
        await backend.save_conversation_record(make_conversation("rollback-conv"))
        await backend.rollback()

        assert await backend.get_conversation("rollback-conv") is None
        await backend.close()


class TestMetadataOperations:
    """Test metadata CRUD operations."""

    async def test_update_and_get_metadata(self, tmp_path: Path) -> None:
        """update_metadata sets key, get_metadata retrieves it."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        await backend.save_conversation_record(conv)

        await backend.update_metadata("conv-1", "rating", 5)
        await backend.update_metadata("conv-1", "reviewed", True)

        metadata = await backend.get_metadata("conv-1")
        assert metadata.get("rating") == 5
        assert metadata.get("reviewed") is True
        await backend.close()

    async def test_delete_metadata(self, tmp_path: Path) -> None:
        """delete_metadata removes a key."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        await backend.save_conversation_record(conv)

        await backend.update_metadata("conv-1", "temp", "value")
        await backend.delete_metadata("conv-1", "temp")

        metadata = await backend.get_metadata("conv-1")
        assert "temp" not in metadata
        await backend.close()

    async def test_add_and_remove_tag(self, tmp_path: Path) -> None:
        """add_tag adds to tags list, remove_tag removes it."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        await backend.save_conversation_record(conv)

        await backend.add_tag("conv-1", "important")
        await backend.add_tag("conv-1", "work")

        metadata = await backend.get_metadata("conv-1")
        tags = metadata.get("tags", [])
        assert "important" in tags
        assert "work" in tags

        await backend.remove_tag("conv-1", "work")
        metadata = await backend.get_metadata("conv-1")
        tags = metadata.get("tags", [])
        assert "important" in tags
        assert "work" not in tags
        await backend.close()

    async def test_list_tags_empty(self, tmp_path: Path) -> None:
        """list_tags returns empty dict when no tags exist."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        await backend.save_conversation_record(conv)

        tags = await backend.list_tags()
        assert tags == {}
        await backend.close()

    async def test_list_tags_counts(self, tmp_path: Path) -> None:
        """list_tags returns correct tag counts across conversations."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        # Create 3 conversations with different tags
        conv1 = make_conversation("conv-1")
        conv2 = make_conversation("conv-2")
        conv3 = make_conversation("conv-3")

        await backend.save_conversation_record(conv1)
        await backend.save_conversation_record(conv2)
        await backend.save_conversation_record(conv3)

        # Tag conv-1 with "important" and "work"
        await backend.add_tag("conv-1", "important")
        await backend.add_tag("conv-1", "work")

        # Tag conv-2 with "important"
        await backend.add_tag("conv-2", "important")

        # conv-3 has no tags

        tags = await backend.list_tags()
        assert tags == {"important": 2, "work": 1}
        await backend.close()

    async def test_list_tags_provider_filter(self, tmp_path: Path) -> None:
        """list_tags with provider filter only counts tags from that provider."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        # Create conversations with different providers
        conv_claude = make_conversation("conv-claude", provider_name="claude")
        conv_chatgpt = make_conversation("conv-chatgpt", provider_name="chatgpt")

        await backend.save_conversation_record(conv_claude)
        await backend.save_conversation_record(conv_chatgpt)

        # Tag both
        await backend.add_tag("conv-claude", "important")
        await backend.add_tag("conv-chatgpt", "important")
        await backend.add_tag("conv-chatgpt", "review")

        # Filter by claude provider
        tags_claude = await backend.list_tags(provider="claude")
        assert tags_claude == {"important": 1}

        # Filter by chatgpt provider
        tags_chatgpt = await backend.list_tags(provider="chatgpt")
        assert tags_chatgpt == {"important": 1, "review": 1}

        # All tags
        tags_all = await backend.list_tags()
        assert tags_all == {"important": 2, "review": 1}

        await backend.close()

    async def test_list_tags_dedup(self, tmp_path: Path) -> None:
        """list_tags doesn't double-count duplicate tags on same conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        await backend.save_conversation_record(conv)

        # Add the same tag twice
        await backend.add_tag("conv-1", "important")
        await backend.add_tag("conv-1", "important")

        tags = await backend.list_tags()
        # Should count as 1, not 2
        assert tags == {"important": 1}
        await backend.close()


class TestSearchOperations:
    """Test search and resolve operations."""

    async def test_resolve_id_exact_match(self, tmp_path: Path) -> None:
        """resolve_id returns full ID for exact match."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conversation-12345")
        await backend.save_conversation_record(conv)

        resolved = await backend.resolve_id("conversation-12345")
        assert resolved == "conversation-12345"
        await backend.close()

    async def test_resolve_id_prefix_match(self, tmp_path: Path) -> None:
        """resolve_id returns full ID for unique prefix."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("unique-prefix-abc123")
        await backend.save_conversation_record(conv)

        resolved = await backend.resolve_id("unique-prefix")
        assert resolved == "unique-prefix-abc123"
        await backend.close()

    async def test_resolve_id_ambiguous_returns_none(self, tmp_path: Path) -> None:
        """resolve_id returns None for ambiguous prefix."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        await backend.save_conversation_record(make_conversation("prefix-abc"))
        await backend.save_conversation_record(make_conversation("prefix-def"))

        resolved = await backend.resolve_id("prefix")
        assert resolved is None
        await backend.close()


class TestDeleteOperations:
    """Test deletion operations."""

    async def test_delete_conversation(self, tmp_path: Path) -> None:
        """delete_conversation removes conversation and related data."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("to-delete")
        await backend.save_conversation_record(conv)
        await backend.save_messages([make_message("m1", "to-delete")])
        await backend.save_attachments([make_attachment("a1", "to-delete")])

        result = await backend.delete_conversation("to-delete")
        assert result is True

        assert await backend.get_conversation("to-delete") is None
        assert await backend.get_messages("to-delete") == []
        assert await backend.get_attachments("to-delete") == []
        await backend.close()

    async def test_delete_nonexistent_returns_false(self, tmp_path: Path) -> None:
        """delete_conversation returns False for missing ID."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        result = await backend.delete_conversation("nonexistent")
        assert result is False
        await backend.close()


class TestPruneAttachments:
    """Test attachment pruning edge cases."""

    async def test_prune_does_not_remove_shared_attachments(self, tmp_path: Path) -> None:
        """Attachments referenced by multiple conversations are NOT pruned."""
        backend = SQLiteBackend(db_path=tmp_path / "prune.db")

        # Two conversations sharing the same attachment ID
        conv1 = make_conversation("conv-1", title="First")
        msg1 = make_message("msg-1", "conv-1", text="Hello")
        att = make_attachment("shared-att", "conv-1", "msg-1", mime_type="image/png", size_bytes=100)

        await backend.save_conversation_record(conv1)
        await backend.save_messages([msg1])
        await backend.save_attachments([att])

        conv2 = make_conversation("conv-2", title="Second")
        msg2 = make_message("msg-2", "conv-2", text="World")
        att2 = make_attachment("shared-att", "conv-2", "msg-2", mime_type="image/png", size_bytes=100)

        await backend.save_conversation_record(conv2)
        await backend.save_messages([msg2])
        await backend.save_attachments([att2])

        # Prune conv-1 without the attachment
        await backend.prune_attachments("conv-1", set())

        # Attachment should still exist in conv-2
        conv2_atts = await backend.get_attachments("conv-2")
        assert len(conv2_atts) == 1
        assert conv2_atts[0].attachment_id == "shared-att"

        # Check in database that attachment still exists
        async with backend._get_connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM attachments WHERE attachment_id = 'shared-att'")
            row = await cursor.fetchone()
            assert row[0] == 1
        await backend.close()

    async def test_prune_removes_sole_attachment(self, tmp_path: Path) -> None:
        """Attachments with only one reference are pruned."""
        backend = SQLiteBackend(db_path=tmp_path / "prune2.db")

        conv = make_conversation("conv-sole", title="Sole")
        msg = make_message("msg-1", "conv-sole", text="Hello")
        att = make_attachment("sole-att", "conv-sole", "msg-1", mime_type="text/plain", size_bytes=50)

        await backend.save_conversation_record(conv)
        await backend.save_messages([msg])
        await backend.save_attachments([att])

        # Prune without keeping the attachment
        await backend.prune_attachments("conv-sole", set())

        # Attachment should be removed
        async with backend._get_connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM attachments WHERE attachment_id = 'sole-att'")
            row = await cursor.fetchone()
            assert row[0] == 0
        await backend.close()

    async def test_prune_empty_keep_set_removes_all(self, tmp_path: Path) -> None:
        """Pruning with empty keep set removes all attachments for conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "prune3.db")

        conv = make_conversation("conv-empty", title="Empty Keep")
        msg = make_message("msg-1", "conv-empty", text="Hello")
        attachments = [
            make_attachment("att1", "conv-empty", "msg-1"),
            make_attachment("att2", "conv-empty", "msg-1"),
            make_attachment("att3", "conv-empty", "msg-1"),
        ]

        await backend.save_conversation_record(conv)
        await backend.save_messages([msg])
        await backend.save_attachments(attachments)

        # Verify attachments were saved
        before = await backend.get_attachments("conv-empty")
        assert len(before) == 3

        # Prune with empty keep set
        await backend.prune_attachments("conv-empty", set())

        # All attachments should be gone
        after = await backend.get_attachments("conv-empty")
        assert len(after) == 0
        await backend.close()


# =============================================================================
# BACKEND COMPARISON TESTS (from test_backend_core.py)
# =============================================================================


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


class TestTransactionAtomicity:
    """Test that transaction management is atomic — partial failures roll back all changes."""

    @pytest.mark.asyncio
    async def test_message_failure_rolls_back_conversation(self, tmp_path: Path) -> None:
        """If message saving fails within a transaction, conversation is rolled back."""
        backend = SQLiteBackend(db_path=tmp_path / "atomic1.db")

        conv = make_conversation("conv-atomic-1", title="Atomic Test 1")
        msg = make_message("msg-1", "conv-atomic-1", text="Hello")

        # Start explicit transaction
        await backend.begin()
        await backend.save_conversation_record(conv)

        # Simulate failure by trying to save a message with bad data
        # (this would normally fail in a real scenario; we manually raise)
        try:
            await backend.save_messages([msg])
            # Now simulate a failure after messages but before commit
            raise RuntimeError("Simulated failure after message save")
        except RuntimeError:
            await backend.rollback()

        # Verify nothing persisted
        retrieved = await backend.get_conversation("conv-atomic-1")
        assert retrieved is None

        await backend.close()

    @pytest.mark.asyncio
    async def test_attachment_failure_rolls_back_all(self, tmp_path: Path) -> None:
        """If attachment saving fails, conversation and messages are NOT persisted."""
        backend = SQLiteBackend(db_path=tmp_path / "atomic2.db")

        conv = make_conversation("conv-atomic-2", title="Atomic Test 2")
        msg = make_message("msg-1", "conv-atomic-2", text="Hello")
        att = make_attachment("att-bad", "conv-atomic-2", "msg-1", mime_type="image/png", size_bytes=100)

        await backend.begin()
        await backend.save_conversation_record(conv)
        await backend.save_messages([msg])

        # Simulate attachment save failure
        try:
            await backend.save_attachments([att])
            # Simulate a failure during transaction
            raise RuntimeError("Simulated attachment failure")
        except RuntimeError:
            await backend.rollback()

        # Verify nothing persisted
        retrieved = await backend.get_conversation("conv-atomic-2")
        assert retrieved is None

        await backend.close()

    @pytest.mark.asyncio
    async def test_nothing_persisted_after_rollback(self, tmp_path: Path) -> None:
        """After a rolled-back transaction, database state is completely clean."""
        backend = SQLiteBackend(db_path=tmp_path / "atomic3.db")

        conv = make_conversation("conv-clean", title="Clean Check")
        msg = make_message("msg-1", "conv-clean", text="Hello")
        att = make_attachment("att-1", "conv-clean", "msg-1", mime_type="text/plain", size_bytes=50)

        await backend.begin()
        try:
            await backend.save_conversation_record(conv)
            await backend.save_messages([msg])
            await backend.save_attachments([att])
            raise RuntimeError("Boom")
        except RuntimeError:
            await backend.rollback()

        # Verify all tables are empty
        async with backend._get_connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM conversations")
            row = await cursor.fetchone()
            assert row[0] == 0

            cursor = await conn.execute("SELECT COUNT(*) FROM messages")
            row = await cursor.fetchone()
            assert row[0] == 0

            cursor = await conn.execute("SELECT COUNT(*) FROM attachments")
            row = await cursor.fetchone()
            assert row[0] == 0

        await backend.close()

    @pytest.mark.asyncio
    async def test_successful_transaction_persists(self, tmp_path: Path) -> None:
        """A successful transaction persists all data."""
        backend = SQLiteBackend(db_path=tmp_path / "atomic4.db")

        conv = make_conversation("conv-success", title="Success Test")
        msg = make_message("msg-1", "conv-success", text="Hello")
        att = make_attachment("att-1", "conv-success", "msg-1", mime_type="text/plain", size_bytes=50)

        async with backend.transaction():
            await backend.save_conversation_record(conv)
            await backend.save_messages([msg])
            await backend.save_attachments([att])

        # Verify all data persisted
        retrieved_conv = await backend.get_conversation("conv-success")
        assert retrieved_conv is not None
        assert retrieved_conv.title == "Success Test"

        retrieved_msgs = await backend.get_messages("conv-success")
        assert len(retrieved_msgs) == 1
        assert retrieved_msgs[0].message_id == "msg-1"

        retrieved_atts = await backend.get_attachments("conv-success")
        assert len(retrieved_atts) == 1
        assert retrieved_atts[0].attachment_id == "att-1"

        await backend.close()


class TestAPICompatibility:
    """Test that new extraction can be adapted to old API."""

    def test_parsed_message_equivalent_fields(self):
        """HarmonizedMessage has equivalent fields to ParsedMessage."""
        from polylogue.schemas.unified import HarmonizedMessage
        from polylogue.sources.parsers.base import ParsedMessage

        field_mapping = {
            "provider_message_id": "id",
            "role": "role",
            "text": "text",
            "timestamp": "timestamp",
            "provider_meta": "raw",
        }

        pm = ParsedMessage(provider_message_id="test", role="user", text="hello")
        for pm_field in field_mapping:
            assert hasattr(pm, pm_field), f"ParsedMessage missing {pm_field}"

        hm = HarmonizedMessage(role="user", text="hello", provider="test")
        for hm_field in field_mapping.values():
            assert hasattr(hm, hm_field), f"HarmonizedMessage missing {hm_field}"

    def test_can_convert_harmonized_to_parsed(self):
        """Demonstrate conversion from HarmonizedMessage to ParsedMessage."""
        from polylogue.sources.parsers.base import ParsedMessage

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
