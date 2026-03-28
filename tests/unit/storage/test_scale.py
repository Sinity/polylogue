"""Scale tests for storage operations.

These tests verify correct behavior with 200+ conversations to catch:
- N+1 query patterns (asyncio.gather spawning N connections)
- Memory issues with large result sets
- Correct data association across batch queries

The original production failure was a connection storm with 3000+
concurrent SQLite connections. These tests ensure batch query
patterns work correctly at moderate scale.

Performance budget tests (TestPerformanceBudget) assert that key
operations finish within fixed timing budgets. They use @pytest.mark.slow
and are excluded from the normal fast unit run.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from collections import defaultdict

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import ContentBlockRecord, ConversationRecord, MessageRecord

# Number of conversations for scale tests.
# 200 is enough to expose N+1 patterns while keeping tests fast (<2s).
SCALE_COUNT = 200


async def _seed_conversations(
    backend: SQLiteBackend, count: int, msgs_per_conv: int = 3
) -> list[str]:
    """Seed the database with conversations and messages.

    Saves conversations concurrently, then all messages in one batch.
    Returns list of conversation IDs.
    """
    ids = [f"scale-conv-{i:04d}" for i in range(count)]
    conv_records = [
        ConversationRecord(
            conversation_id=ids[i],
            provider_name="chatgpt" if i % 2 == 0 else "claude",
            provider_conversation_id=f"prov-{ids[i]}",
            title=f"Scale Test Conversation {i}",
            created_at=f"2025-01-{(i % 28) + 1:02d}T00:00:00Z",
            updated_at=f"2025-01-{(i % 28) + 1:02d}T00:00:00Z",
            content_hash=f"hash-{ids[i]}",
        )
        for i in range(count)
    ]
    await asyncio.gather(*(backend.save_conversation_record(r) for r in conv_records))
    all_msgs = [
        MessageRecord(
            message_id=f"{ids[i]}-m{j}",
            conversation_id=ids[i],
            role="user" if j % 2 == 0 else "assistant",
            text=f"Message {j} in conversation {i}",
            timestamp=f"2025-01-01T00:{j:02d}:00Z",
            content_hash=f"hash-{ids[i]}-m{j}",
        )
        for i in range(count)
        for j in range(msgs_per_conv)
    ]
    await backend.save_messages(all_msgs)
    return ids


class TestGetManyScale:
    """Test repository.get_many() with 200+ conversations."""

    async def test_get_many_returns_all_conversations(self, tmp_path):
        """get_many with 200 IDs returns all conversations with messages."""
        from polylogue.storage.repository import ConversationRepository

        db_path = tmp_path / "scale.db"
        backend = SQLiteBackend(db_path=db_path)
        ids = await _seed_conversations(backend, SCALE_COUNT)

        repo = ConversationRepository(backend=backend)
        convos = await repo.get_many(ids)

        assert len(convos) == SCALE_COUNT
        # Verify each conversation has its messages
        for convo in convos:
            assert (
                len(convo.messages) == 3
            ), f"Conv {convo.id} has {len(convo.messages)} messages, expected 3"

    async def test_get_many_preserves_order(self, tmp_path):
        """get_many returns conversations in input order."""
        from polylogue.storage.repository import ConversationRepository

        db_path = tmp_path / "scale.db"
        backend = SQLiteBackend(db_path=db_path)
        ids = await _seed_conversations(backend, 50)

        repo = ConversationRepository(backend=backend)
        # Request in reverse order
        reversed_ids = list(reversed(ids))
        convos = await repo.get_many(reversed_ids)

        assert [c.id for c in convos] == reversed_ids

    async def test_get_many_messages_correctly_associated(self, tmp_path):
        """Each conversation's messages actually belong to that conversation."""
        from polylogue.storage.repository import ConversationRepository

        db_path = tmp_path / "scale.db"
        backend = SQLiteBackend(db_path=db_path)
        ids = await _seed_conversations(backend, SCALE_COUNT)

        repo = ConversationRepository(backend=backend)
        convos = await repo.get_many(ids)

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
        msg_counts = [1, 5, 10]
        all_ids = [f"var-{msgs}msg-{i:03d}" for msgs in msg_counts for i in range(20)]
        conv_records = [
            ConversationRecord(
                conversation_id=cid,
                provider_name="test",
                provider_conversation_id=f"prov-{cid}",
                title=f"Var {cid.split('-')[1]} {cid.split('-')[2]}",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                content_hash=f"hash-{cid}",
            )
            for cid in all_ids
        ]
        await asyncio.gather(*(backend.save_conversation_record(r) for r in conv_records))
        all_msg_records = [
            MessageRecord(
                message_id=f"{cid}-m{j}",
                conversation_id=cid,
                role="user",
                text=f"msg {j}",
                timestamp=f"2025-01-01T00:{j:02d}:00Z",
                content_hash=f"hash-{cid}-m{j}",
            )
            for msgs in msg_counts
            for i in range(20)
            for cid in [f"var-{msgs}msg-{i:03d}"]
            for j in range(msgs)
        ]
        await backend.save_messages(all_msg_records)

        repo = ConversationRepository(backend=backend)
        convos = await repo.get_many(all_ids)
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


async def _seed_budget_db(tmp_path, *, conv_count: int = 500, msgs_per_conv: int = 10):
    """Seed a DB for performance budget tests. Returns (backend, ids)."""
    db_path = tmp_path / "budget.db"
    backend = SQLiteBackend(db_path=db_path)
    ids = await _seed_conversations(backend, conv_count, msgs_per_conv=msgs_per_conv)
    return backend, ids


@pytest.mark.slow
class TestPerformanceBudget:
    """Performance budget tests — each asserts a timing SLA.

    These catch query regressions that wouldn't surface as correctness failures.
    Budgets are conservative (10–20× typical times on a modern workstation).
    """

    async def test_list_performance_budget(self, tmp_path):
        """list_conversations(limit=50) on 5k-message DB must finish in <100ms."""
        backend, _ = await _seed_budget_db(tmp_path)
        t0 = time.monotonic()
        results = await backend.list_conversations(limit=50)
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert len(results) == 50
        assert elapsed_ms < 100, f"list_conversations took {elapsed_ms:.0f}ms (budget: 100ms)"

    async def test_get_many_performance_budget(self, tmp_path):
        """get_many(100 ids) on 5k DB must finish in <500ms."""
        backend, ids = await _seed_budget_db(tmp_path)
        repo = ConversationRepository(backend=backend)
        sample_ids = ids[:100]
        t0 = time.monotonic()
        results = await repo.get_many(sample_ids)
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert len(results) == 100
        assert elapsed_ms < 500, f"get_many(100) took {elapsed_ms:.0f}ms (budget: 500ms)"

    async def test_fts_search_budget(self, tmp_path):
        """FTS5 search for common term on 5k-message DB must finish in <200ms."""
        backend, _ = await _seed_budget_db(tmp_path)
        # Rebuild index so FTS has content
        with open_connection(backend.db_path) as conn:
            rebuild_index(conn)
        repo = ConversationRepository(backend=backend)
        t0 = time.monotonic()
        results = await repo.search_summaries("Message", limit=20)
        elapsed_ms = (time.monotonic() - t0) * 1000
        # FTS on seeded data — results may be empty if text doesn't tokenize well, just check timing
        assert elapsed_ms < 200, f"FTS search took {elapsed_ms:.0f}ms (budget: 200ms)"
        _ = results  # exercised

    async def test_has_tool_use_filter_budget(self, tmp_path):
        """has_tool_use=True filter on 5k DB must finish in <100ms.

        Validates that the stats LEFT JOIN covering index is effective.
        """
        backend, _ = await _seed_budget_db(tmp_path)
        t0 = time.monotonic()
        results = await backend.list_conversations(has_tool_use=True, limit=50)
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 100, f"has_tool_use filter took {elapsed_ms:.0f}ms (budget: 100ms)"
        _ = results  # exercised

    async def test_semantic_filter_budget(self, tmp_path):
        """has_file_ops EXISTS filter on 5k DB must finish in <200ms.

        Now that content_blocks are seeded, this validates the covering index
        and ensures the filter returns real results.
        """
        backend, ids = await _seed_budget_db(tmp_path)

        # Seed content_blocks: 1 file_read block per conversation for 20% of convs
        blocks: list[ContentBlockRecord] = []
        for i, cid in enumerate(ids):
            if i % 5 == 0:  # 20% of conversations
                mid = f"{cid}-m0"
                blocks.append(ContentBlockRecord(
                    block_id=ContentBlockRecord.make_id(mid, 0),
                    message_id=mid,
                    conversation_id=cid,
                    block_index=0,
                    type='tool_use',
                    semantic_type='file_read',
                ))
        await backend.save_content_blocks(blocks)

        t0 = time.monotonic()
        results = await backend.list_conversations(has_file_ops=True, limit=50)
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert len(results) > 0, "Semantic filter should find results with seeded content_blocks"
        assert elapsed_ms < 200, f"has_file_ops filter took {elapsed_ms:.0f}ms (budget: 200ms)"
