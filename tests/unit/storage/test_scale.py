"""Scale tests for storage operations.

These tests verify correct behavior with 200+ conversations to catch:
- N+1 query patterns (asyncio.gather spawning N connections)
- Memory issues with large result sets
- Correct data association across batch queries

The original production failure was a connection storm with 3000+
concurrent SQLite connections. These tests ensure batch query
patterns work correctly at moderate scale.
"""

from __future__ import annotations

import asyncio

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord


# Number of conversations for scale tests.
# 200 is enough to expose N+1 patterns while keeping tests fast (<2s).
SCALE_COUNT = 200


async def _seed_conversations(
    backend: SQLiteBackend, count: int, msgs_per_conv: int = 3
) -> list[str]:
    """Seed the database with conversations and messages.

    Returns list of conversation IDs.
    """
    ids = []
    for i in range(count):
        cid = f"scale-conv-{i:04d}"
        ids.append(cid)
        await backend.save_conversation_record(
            ConversationRecord(
                conversation_id=cid,
                provider_name="chatgpt" if i % 2 == 0 else "claude",
                provider_conversation_id=f"prov-{cid}",
                title=f"Scale Test Conversation {i}",
                created_at=f"2025-01-{(i % 28) + 1:02d}T00:00:00Z",
                updated_at=f"2025-01-{(i % 28) + 1:02d}T00:00:00Z",
                content_hash=f"hash-{cid}",
            )
        )
        msgs = [
            MessageRecord(
                message_id=f"{cid}-m{j}",
                conversation_id=cid,
                role="user" if j % 2 == 0 else "assistant",
                text=f"Message {j} in conversation {i}",
                timestamp=f"2025-01-01T00:{j:02d}:00Z",
                content_hash=f"hash-{cid}-m{j}",
            )
            for j in range(msgs_per_conv)
        ]
        await backend.save_messages(msgs)
    return ids


class TestGetManyScale:
    """Test repository._get_many() with 200+ conversations."""

    async def test_get_many_returns_all_conversations(self, tmp_path):
        """_get_many with 200 IDs returns all conversations with messages."""
        from polylogue.storage.repository import ConversationRepository

        db_path = tmp_path / "scale.db"
        backend = SQLiteBackend(db_path=db_path)
        ids = await _seed_conversations(backend, SCALE_COUNT)

        repo = ConversationRepository(backend=backend)
        convos = await repo._get_many(ids)

        assert len(convos) == SCALE_COUNT
        # Verify each conversation has its messages
        for convo in convos:
            assert (
                len(convo.messages) == 3
            ), f"Conv {convo.id} has {len(convo.messages)} messages, expected 3"

    async def test_get_many_preserves_order(self, tmp_path):
        """_get_many returns conversations in input order."""
        from polylogue.storage.repository import ConversationRepository

        db_path = tmp_path / "scale.db"
        backend = SQLiteBackend(db_path=db_path)
        ids = await _seed_conversations(backend, 50)

        repo = ConversationRepository(backend=backend)
        # Request in reverse order
        reversed_ids = list(reversed(ids))
        convos = await repo._get_many(reversed_ids)

        assert [c.id for c in convos] == reversed_ids

    async def test_get_many_messages_correctly_associated(self, tmp_path):
        """Each conversation's messages actually belong to that conversation."""
        from polylogue.storage.repository import ConversationRepository

        db_path = tmp_path / "scale.db"
        backend = SQLiteBackend(db_path=db_path)
        ids = await _seed_conversations(backend, SCALE_COUNT)

        repo = ConversationRepository(backend=backend)
        convos = await repo._get_many(ids)

        for convo in convos:
            for msg in convo.messages:
                assert msg.id.startswith(convo.id + "-"), (
                    f"Message {msg.id} doesn't belong to conversation {convo.id}"
                )

    async def test_get_many_varying_message_counts(self, tmp_path):
        """Conversations with different message counts are handled correctly."""
        from polylogue.storage.repository import ConversationRepository

        db_path = tmp_path / "scale.db"
        backend = SQLiteBackend(db_path=db_path)

        # Create conversations with 1, 5, and 10 messages
        test_cases = [(1, 1), (5, 5), (10, 10)]
        all_ids = []
        for count, msgs in test_cases:
            for i in range(20):
                cid = f"var-{msgs}msg-{i:03d}"
                all_ids.append(cid)
                await backend.save_conversation_record(
                    ConversationRecord(
                        conversation_id=cid,
                        provider_name="test",
                        provider_conversation_id=f"prov-{cid}",
                        title=f"Var {msgs} msgs {i}",
                        created_at="2025-01-01T00:00:00Z",
                        updated_at="2025-01-01T00:00:00Z",
                        content_hash=f"hash-{cid}",
                    )
                )
                msg_records = [
                    MessageRecord(
                        message_id=f"{cid}-m{j}",
                        conversation_id=cid,
                        role="user",
                        text=f"msg {j}",
                        timestamp=f"2025-01-01T00:{j:02d}:00Z",
                        content_hash=f"hash-{cid}-m{j}",
                    )
                    for j in range(msgs)
                ]
                await backend.save_messages(msg_records)

        repo = ConversationRepository(backend=backend)
        convos = await repo._get_many(all_ids)
        assert len(convos) == 60  # 20 * 3

        # Verify message counts
        for convo in convos:
            if "var-1msg" in convo.id:
                assert len(convo.messages) == 1
            elif "var-5msg" in convo.id:
                assert len(convo.messages) == 5
            elif "var-10msg" in convo.id:
                assert len(convo.messages) == 10


class TestFacadeScale:
    """Test facade batch operations at scale."""

    async def test_get_conversations_at_scale(self, tmp_path):
        """Facade.get_conversations with 200 IDs returns correct results."""
        from polylogue.facade import Polylogue

        db_path = tmp_path / "facade_scale.db"
        backend = SQLiteBackend(db_path=db_path)
        ids = await _seed_conversations(backend, SCALE_COUNT)

        archive = Polylogue(db_path=db_path)
        convos = await archive.get_conversations(ids)
        await archive.close()

        assert len(convos) == SCALE_COUNT
        for convo in convos:
            assert len(convo.messages) == 3

    async def test_get_conversations_empty_input(self, tmp_path):
        """Facade.get_conversations with empty list returns empty."""
        from polylogue.facade import Polylogue

        db_path = tmp_path / "facade_empty.db"
        # Just create the backend to initialize schema
        SQLiteBackend(db_path=db_path)

        archive = Polylogue(db_path=db_path)
        convos = await archive.get_conversations([])
        await archive.close()

        assert convos == []

    async def test_list_conversations_at_scale(self, tmp_path):
        """Facade.list_conversations with 200 conversations works."""
        from polylogue.facade import Polylogue

        db_path = tmp_path / "facade_list.db"
        backend = SQLiteBackend(db_path=db_path)
        await _seed_conversations(backend, SCALE_COUNT)

        archive = Polylogue(db_path=db_path)
        convos = await archive.list_conversations()
        await archive.close()

        assert len(convos) == SCALE_COUNT

    async def test_list_conversations_with_provider_filter(self, tmp_path):
        """Facade.list_conversations filtered by provider at scale."""
        from polylogue.facade import Polylogue

        db_path = tmp_path / "facade_filter.db"
        backend = SQLiteBackend(db_path=db_path)
        await _seed_conversations(backend, SCALE_COUNT)  # alternates chatgpt/claude

        archive = Polylogue(db_path=db_path)
        convos = await archive.list_conversations(provider="chatgpt")
        await archive.close()

        assert len(convos) == SCALE_COUNT // 2
        for convo in convos:
            assert convo.provider == "chatgpt"


class TestBatchQueryScale:
    """Test backend batch queries at scale."""

    async def test_messages_batch_200_conversations(self, tmp_path):
        """get_messages_batch with 200 conversation IDs returns correct grouping."""
        db_path = tmp_path / "batch_scale.db"
        backend = SQLiteBackend(db_path=db_path)
        ids = await _seed_conversations(backend, SCALE_COUNT, msgs_per_conv=5)

        result = await backend.get_messages_batch(ids)

        assert len(result) == SCALE_COUNT
        for cid in ids:
            assert (
                len(result[cid]) == 5
            ), f"Conv {cid} has {len(result[cid])} messages, expected 5"

    async def test_conversations_batch_200_ids(self, tmp_path):
        """get_conversations_batch with 200 IDs returns all."""
        db_path = tmp_path / "conv_batch_scale.db"
        backend = SQLiteBackend(db_path=db_path)
        ids = await _seed_conversations(backend, SCALE_COUNT)

        records = await backend.get_conversations_batch(ids)
        assert len(records) == SCALE_COUNT
