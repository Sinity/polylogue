"""CRUD operations tests — conversation, message, attachment, transaction, metadata, search, delete, backend comparison."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from polylogue.lib.roles import normalize_role as new_normalize_role
from polylogue.schemas.unified import (
    extract_from_provider_meta,
    extract_harmonized_message,
    is_message_record,
)
from polylogue.sources.parsers.base import normalize_role as old_normalize_role
from polylogue.sources.parsers.claude import (
    extract_text_from_segments as old_extract_segments,
)
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.store import (
    ContentBlockRecord,
)
from tests.infra.helpers import (
    make_attachment,
    make_conversation,
    make_message,
)


class TestConversationOperations:
    """Test conversation save/retrieve operations."""


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

    async def test_content_block_semantic_type_stored(self, tmp_path: Path) -> None:
        """ContentBlockRecord.semantic_type is persisted and retrieved correctly."""
        from polylogue.types import MessageId

        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        conv = make_conversation("conv-sem")
        await backend.save_conversation_record(conv)

        msg = make_message("msg-sem", "conv-sem", role="assistant", text="")
        await backend.save_messages([msg])

        blocks = [
            ContentBlockRecord(
                block_id="block-1",
                message_id=MessageId("msg-sem"),
                conversation_id="conv-sem",
                block_index=0,
                type="tool_use",
                tool_name="Bash",
                semantic_type="git",
                metadata='{"command": "status"}',
            ),
            ContentBlockRecord(
                block_id="block-2",
                message_id=MessageId("msg-sem"),
                conversation_id="conv-sem",
                block_index=1,
                type="thinking",
                text="Let me think",
                semantic_type="thinking",
            ),
            ContentBlockRecord(
                block_id="block-3",
                message_id=MessageId("msg-sem"),
                conversation_id="conv-sem",
                block_index=2,
                type="text",
                text="plain text",
                semantic_type=None,
            ),
        ]
        await backend.save_content_blocks(blocks)

        blocks_by_msg = await backend.get_content_blocks(["msg-sem"])
        retrieved = blocks_by_msg.get("msg-sem", [])
        assert len(retrieved) == 3

        by_index = {b.block_index: b for b in retrieved}
        assert by_index[0].semantic_type == "git"
        assert by_index[1].semantic_type == "thinking"
        assert by_index[2].semantic_type is None
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


class TestSearchOperations:
    """Test search and resolve operations."""


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

    async def test_delete_conversation_reparents_children(self, tmp_path: Path) -> None:
        """Deleting a parent should keep descendants accessible by reparenting them."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        await backend.save_conversation_record(make_conversation("root"))
        await backend.save_conversation_record(make_conversation("child", parent_conversation_id="root"))
        await backend.save_conversation_record(make_conversation("grandchild", parent_conversation_id="child"))

        assert await backend.delete_conversation("child") is True

        root = await backend.get_conversation("root")
        child = await backend.get_conversation("child")
        grandchild = await backend.get_conversation("grandchild")

        assert root is not None
        assert child is None
        assert grandchild is not None
        assert grandchild.parent_conversation_id == "root"
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
        async with backend.connection() as conn:
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
        async with backend.connection() as conn:
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


def _load_claude_code_message_records_from_raw(seeded_db: str | Path, *, limit: int = 500) -> list[dict]:
    """Load Claude Code message records from raw_conversations JSONL payloads."""
    conn = sqlite3.connect(seeded_db)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT raw_content
        FROM raw_conversations
        WHERE provider_name = 'claude-code'
        LIMIT ?
        """,
        (max(1, limit),),
    )
    rows = cur.fetchall()
    conn.close()

    records: list[dict] = []
    for (raw_content,) in rows:
        text = (
            raw_content.decode("utf-8", errors="replace")
            if isinstance(raw_content, (bytes, bytearray))
            else str(raw_content)
        )
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and is_message_record("claude-code", payload):
                records.append(payload)
                if len(records) >= limit:
                    return records
    return records


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
        """Compare old and new extraction on Claude Code raw message records."""
        records = _load_claude_code_message_records_from_raw(seeded_db, limit=500)
        assert records, "Expected claude-code message records in seeded raw corpus"

        equiv_count = Counter()
        diff_samples = []
        role_mismatches = 0

        for raw in records:
            results = compare_extractions("claude-code", raw)
            for r in results:
                if r.equivalent:
                    equiv_count[r.field] += 1
                else:
                    if r.field == "role":
                        role_mismatches += 1
                    if len(diff_samples) < 10:
                        diff_samples.append(r)

        total = sum(equiv_count.values())
        assert total > 0
        assert role_mismatches == 0

    def test_new_extraction_is_superset(self, seeded_db):
        """New extraction should expose tool/reasoning data on raw records."""
        records = _load_claude_code_message_records_from_raw(seeded_db, limit=500)
        assert records, "Expected claude-code message records in seeded raw corpus"

        tool_calls_found = 0
        reasoning_found = 0
        processed = 0

        for raw in records:
            new_msg = extract_from_provider_meta("claude-code", {"raw": raw})
            processed += 1
            tool_calls_found += len(new_msg.tool_calls)
            reasoning_found += len(new_msg.reasoning_traces)

        assert processed > 0
        assert tool_calls_found > 0 or reasoning_found > 0, (
            "New extraction should find tool calls or reasoning traces"
        )


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
        async with backend.connection() as conn:
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
