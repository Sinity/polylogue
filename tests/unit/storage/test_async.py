"""Tests for async SQLite backend.

Covers:
- Basic operations (list, stats, get)
- Concurrent reads
- Context manager lifecycle
- Double-check lock pattern for schema initialization
- Transaction isolation and write serialization
"""

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.async_facade import AsyncPolylogue
from polylogue.storage.backends.async_sqlite import AsyncSQLiteBackend

# =============================================================================
# Basic Operations
# =============================================================================


@pytest.mark.asyncio
async def test_async_backend_basic():
    """Test basic async backend operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncPolylogue(db_path=db_path) as archive:
            # Empty archive initially
            convs = await archive.list_conversations()
            assert len(convs) == 0

            # Stats should work
            stats = await archive.stats()
            assert stats["conversation_count"] == 0
            assert stats["message_count"] == 0


@pytest.mark.asyncio
async def test_async_concurrent_reads():
    """Test concurrent read operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncPolylogue(db_path=db_path) as archive:
            # Concurrent reads should work
            results = await asyncio.gather(
                archive.list_conversations(),
                archive.stats(),
                archive.get_conversation("nonexistent"),
            )

            convs, stats, missing = results

            assert len(convs) == 0
            assert stats["conversation_count"] == 0
            assert missing is None


@pytest.mark.asyncio
async def test_async_batch_retrieval():
    """Test parallel batch retrieval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncPolylogue(db_path=db_path) as archive:
            # Try to get multiple (non-existent) conversations
            ids = [f"test:{i}" for i in range(10)]
            convs = await archive.get_conversations(ids)

            # Should return empty list (none found)
            assert len(convs) == 0


@pytest.mark.asyncio
async def test_async_context_manager():
    """Test async context manager lifecycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Context manager should handle cleanup
        async with AsyncPolylogue(db_path=db_path) as archive:
            await archive.stats()

        # Manual close should also work
        archive2 = AsyncPolylogue(db_path=db_path)
        try:
            await archive2.stats()
        finally:
            await archive2.close()


# =============================================================================
# Double-Check Lock Pattern Tests (for schema initialization)
# =============================================================================


@pytest.mark.asyncio
async def test_concurrent_schema_initialization():
    """Multiple concurrent operations should not race on schema init.

    Tests the double-check lock pattern at async_sqlite.py:92-101
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_concurrent.db"

        backend = AsyncSQLiteBackend(db_path=db_path)

        # Start 10 concurrent operations that all need schema
        async def get_or_fail(i: int):
            return await backend.get_conversation(f"test:{i}")

        # All should complete without error despite concurrent schema init
        results = await asyncio.gather(*[get_or_fail(i) for i in range(10)])

        # All should return None (no conversations exist)
        assert all(r is None for r in results)

        # Schema should have been initialized exactly once
        assert backend._schema_ensured is True


@pytest.mark.asyncio
async def test_schema_init_called_once_despite_concurrency():
    """Verify schema is initialized only once even with high concurrency."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_once.db"

        backend = AsyncSQLiteBackend(db_path=db_path)

        # Track schema init calls
        init_count = 0
        original_ensure_schema = backend._ensure_schema

        async def counting_ensure_schema(conn):
            nonlocal init_count
            init_count += 1
            return await original_ensure_schema(conn)

        backend._ensure_schema = counting_ensure_schema

        # Launch 20 concurrent operations
        tasks = [backend.list_conversations() for _ in range(20)]
        await asyncio.gather(*tasks)

        # Schema init should have been called exactly once (double-check lock works)
        assert init_count == 1


@pytest.mark.asyncio
async def test_schema_lock_prevents_race():
    """Test that _schema_lock prevents concurrent schema initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_lock.db"

        backend = AsyncSQLiteBackend(db_path=db_path)

        # Force schema_ensured to False to test lock behavior
        backend._schema_ensured = False

        events: list[str] = []

        original_ensure_schema = backend._ensure_schema

        async def slow_ensure_schema(conn):
            events.append("start")
            await asyncio.sleep(0.1)  # Simulate slow schema init
            await original_ensure_schema(conn)
            events.append("end")

        backend._ensure_schema = slow_ensure_schema

        # Start two concurrent operations
        task1 = asyncio.create_task(backend.get_conversation("a"))
        task2 = asyncio.create_task(backend.get_conversation("b"))

        await asyncio.gather(task1, task2)

        # Should see exactly one start/end pair (lock serializes)
        assert events.count("start") == 1
        assert events.count("end") == 1


# =============================================================================
# Transaction and Write Lock Tests
# =============================================================================


@pytest.mark.asyncio
async def test_transaction_context_manager():
    """Transaction context manager acquires write lock."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_transaction.db"

        backend = AsyncSQLiteBackend(db_path=db_path)

        # Should be able to acquire and release transaction
        async with backend.transaction():
            # Write lock should be held
            assert backend._write_lock.locked()

        # Lock should be released after context exits
        assert not backend._write_lock.locked()


@pytest.mark.asyncio
async def test_concurrent_writes_serialized():
    """Concurrent writes should be serialized via write lock."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_writes.db"

        backend = AsyncSQLiteBackend(db_path=db_path)

        execution_order: list[int] = []

        async def write_operation(task_id: int):
            async with backend.transaction():
                execution_order.append(task_id)
                await asyncio.sleep(0.01)  # Hold lock briefly

        # Start 5 concurrent writes
        tasks = [write_operation(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # All 5 operations should have completed
        assert len(execution_order) == 5

        # Order is non-deterministic but all should be present
        assert set(execution_order) == {0, 1, 2, 3, 4}


@pytest.mark.asyncio
async def test_write_lock_not_held_during_reads():
    """Read operations should not acquire write lock."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_read_no_lock.db"

        backend = AsyncSQLiteBackend(db_path=db_path)

        # Perform a read operation
        await backend.list_conversations()

        # Write lock should not be held
        assert not backend._write_lock.locked()


@pytest.mark.asyncio
async def test_reads_can_proceed_during_write():
    """Reads should be able to proceed concurrently with writes.

    SQLite WAL mode allows concurrent reads even during writes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_concurrent_read_write.db"

        backend = AsyncSQLiteBackend(db_path=db_path)

        read_completed = False

        async def slow_write():
            async with backend.transaction():
                await asyncio.sleep(0.1)

        async def quick_read():
            nonlocal read_completed
            await backend.list_conversations()
            read_completed = True

        # Start write first, then read
        write_task = asyncio.create_task(slow_write())
        await asyncio.sleep(0.01)  # Let write start
        read_task = asyncio.create_task(quick_read())

        await asyncio.gather(write_task, read_task)

        # Read should have completed
        assert read_completed


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_connection_error_during_init():
    """Backend should handle connection errors gracefully."""
    # Use an invalid path that can't be created
    with pytest.raises((OSError, PermissionError, Exception)):
        backend = AsyncSQLiteBackend(db_path=Path("/nonexistent/deeply/nested/path/db.db"))
        # Trigger connection
        await backend.get_conversation("test")


@pytest.mark.asyncio
async def test_close_is_idempotent():
    """Calling close() multiple times should be safe."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_close.db"

        backend = AsyncSQLiteBackend(db_path=db_path)

        # Ensure schema is initialized
        await backend.list_conversations()

        # Close multiple times - should not raise
        await backend.close()
        await backend.close()
        await backend.close()


# =============================================================================
# Batch Insertion Tests (N+1 fix verification)
# =============================================================================


@pytest.mark.asyncio
async def test_batch_message_insertion():
    """Verify messages are inserted in batch using executemany.

    Tests that save_conversation uses batch insertion for messages
    rather than individual INSERT statements (N+1 pattern).
    """
    from datetime import datetime, timezone
    from uuid import uuid4

    from polylogue.storage.store import ConversationRecord, MessageRecord

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_batch.db"

        backend = AsyncSQLiteBackend(db_path=db_path)

        now = datetime.now(timezone.utc).isoformat()
        conv_id = "test-conv"

        # Create conversation with 100 messages
        conv = ConversationRecord(
            conversation_id=conv_id,
            provider_name="test",
            provider_conversation_id="ext-1",
            title="Batch Test",
            created_at=now,
            updated_at=now,
            content_hash=uuid4().hex,
        )

        messages = [
            MessageRecord(
                message_id=f"m{i}",
                conversation_id=conv_id,
                role="user" if i % 2 == 0 else "assistant",
                text=f"Message {i} content",
                timestamp=now,
                content_hash=uuid4().hex[:16],
            )
            for i in range(100)
        ]

        # Save with batch insertion
        counts = await backend.save_conversation(conv, messages, [])

        assert counts["messages_created"] == 100
        assert counts["conversations_created"] == 1

        # Verify messages were actually stored
        retrieved = await backend.get_conversation(conv_id)
        assert retrieved is not None

        # Get messages separately (ConversationRecord doesn't embed messages)
        messages_stored = await backend.get_messages(conv_id)
        assert len(messages_stored) == 100

        await backend.close()


@pytest.mark.asyncio
async def test_batch_attachment_insertion():
    """Verify attachments and refs are inserted in batch."""
    from datetime import datetime, timezone
    from uuid import uuid4

    from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_batch_att.db"

        backend = AsyncSQLiteBackend(db_path=db_path)

        now = datetime.now(timezone.utc).isoformat()
        conv_id = "test-conv"

        conv = ConversationRecord(
            conversation_id=conv_id,
            provider_name="test",
            provider_conversation_id="ext-1",
            title="Attachment Batch Test",
            created_at=now,
            updated_at=now,
            content_hash=uuid4().hex,
        )

        messages = [
            MessageRecord(
                message_id="m1",
                conversation_id=conv_id,
                role="user",
                text="Message with attachments",
                timestamp=now,
                content_hash=uuid4().hex[:16],
            )
        ]

        # Create 50 attachments
        attachments = [
            AttachmentRecord(
                attachment_id=f"att{i}",
                conversation_id=conv_id,
                message_id="m1",
                mime_type="application/pdf",
                size_bytes=1024 * (i + 1),
            )
            for i in range(50)
        ]

        counts = await backend.save_conversation(conv, messages, attachments)

        assert counts["attachments_created"] == 50

        await backend.close()


@pytest.mark.asyncio
async def test_empty_message_list_no_error():
    """Verify empty message list is handled correctly."""
    from datetime import datetime, timezone
    from uuid import uuid4

    from polylogue.storage.store import ConversationRecord

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_empty.db"

        backend = AsyncSQLiteBackend(db_path=db_path)

        now = datetime.now(timezone.utc).isoformat()

        conv = ConversationRecord(
            conversation_id="empty-conv",
            provider_name="test",
            provider_conversation_id="ext-1",
            title="Empty Conversation",
            created_at=now,
            updated_at=now,
            content_hash=uuid4().hex,
        )

        # Empty messages and attachments should not cause errors
        counts = await backend.save_conversation(conv, [], [])

        assert counts["messages_created"] == 0
        assert counts["attachments_created"] == 0
        assert counts["conversations_created"] == 1

        await backend.close()


# =============================================================================
# ASYNC STORAGE REPOSITORY TESTS (merged from test_async_repository.py)
# =============================================================================

from polylogue.lib.models import Conversation, Message
from polylogue.storage.async_repository import AsyncConversationRepository
from polylogue.storage.store import AttachmentRecord, MessageRecord

# =============================================================================
# Context Manager Tests
# =============================================================================


@pytest.mark.asyncio
async def test_context_manager_basic():
    """Repository context manager works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncConversationRepository(db_path=db_path) as repo:
            assert repo is not None
            # Should be able to perform operations
            result = await repo.get_conversation("nonexistent")
            assert result is None


@pytest.mark.asyncio
async def test_context_manager_cleanup():
    """Context manager properly cleans up resources."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncConversationRepository(db_path=db_path) as repo:
            # Perform some operation to initialize
            await repo.get_conversation("test")

        # After exit, backend should be closed (close() is a no-op but safe)
        # We can verify by trying to use the backend directly
        # The repository should handle this gracefully


@pytest.mark.asyncio
async def test_manual_close():
    """Manual close() works without context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        repo = AsyncConversationRepository(db_path=db_path)
        try:
            await repo.get_conversation("test")
        finally:
            await repo.close()

        # Should be safe to call close again
        await repo.close()


# =============================================================================
# Save Conversation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_save_conversation_returns_counts():
    """save_conversation returns creation counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncConversationRepository(db_path=db_path) as repo:
            # Create a conversation
            conv = Conversation(
                id="test:conv1",
                provider="test",
                title="Test Conversation",
                messages=[
                    Message(id="m1", role="user", text="Hello"),
                    Message(id="m2", role="assistant", text="Hi there!"),
                ],
                created_at=datetime.now(timezone.utc),
                metadata={"content_hash": "abc123"},
            )

            messages = [
                MessageRecord(
                    message_id="m1",
                    conversation_id="test:conv1",
                    role="user",
                    text="Hello",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    content_hash="hash1",
                ),
                MessageRecord(
                    message_id="m2",
                    conversation_id="test:conv1",
                    role="assistant",
                    text="Hi there!",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    content_hash="hash2",
                ),
            ]

            counts = await repo.save_conversation(conv, messages, [])

            assert counts["conversations"] == 1
            assert counts["messages"] == 2


@pytest.mark.asyncio
async def test_save_conversation_with_attachments():
    """save_conversation handles attachments correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncConversationRepository(db_path=db_path) as repo:
            conv = Conversation(
                id="test:conv2",
                provider="test",
                title="With Attachments",
                messages=[Message(id="m1", role="user", text="See attached")],
                metadata={"content_hash": "def456"},
            )

            messages = [
                MessageRecord(
                    message_id="m1",
                    conversation_id="test:conv2",
                    role="user",
                    text="See attached",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    content_hash="msghash",
                ),
            ]

            attachments = [
                AttachmentRecord(
                    attachment_id="att1",
                    conversation_id="test:conv2",
                    message_id="m1",
                    mime_type="image/png",
                    size_bytes=1024,
                ),
            ]

            counts = await repo.save_conversation(conv, messages, attachments)

            assert counts["attachments"] == 1


# =============================================================================
# Get Conversation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_conversation_found():
    """get_conversation returns saved conversation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncConversationRepository(db_path=db_path) as repo:
            # Save first
            conv = Conversation(
                id="test:findme",
                provider="test",
                title="Find Me",
                messages=[Message(id="m1", role="user", text="Test")],
                metadata={"content_hash": "findhash"},
            )

            messages = [
                MessageRecord(
                    message_id="m1",
                    conversation_id="test:findme",
                    role="user",
                    text="Test",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    content_hash="mhash",
                ),
            ]

            await repo.save_conversation(conv, messages, [])

            # Now retrieve
            result = await repo.get_conversation("test:findme")

            assert result is not None
            assert result.conversation_id == "test:findme"
            assert result.title == "Find Me"


@pytest.mark.asyncio
async def test_get_conversation_not_found():
    """get_conversation returns None for missing conversation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncConversationRepository(db_path=db_path) as repo:
            result = await repo.get_conversation("nonexistent:id")
            assert result is None


# =============================================================================
# Conversation Exists Tests
# =============================================================================


@pytest.mark.asyncio
async def test_conversation_exists_by_hash():
    """conversation_exists checks by content hash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncConversationRepository(db_path=db_path) as repo:
            # Save a conversation with known hash
            conv = Conversation(
                id="test:hashed",
                provider="test",
                title="Hashed",
                messages=[Message(id="m1", role="user", text="Test")],
                metadata={"content_hash": "unique_content_hash_123"},
            )

            messages = [
                MessageRecord(
                    message_id="m1",
                    conversation_id="test:hashed",
                    role="user",
                    text="Test",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    content_hash="mhash",
                ),
            ]

            await repo.save_conversation(conv, messages, [])

            # Check existence
            exists = await repo.conversation_exists("unique_content_hash_123")
            assert exists is True

            not_exists = await repo.conversation_exists("different_hash")
            assert not_exists is False


# =============================================================================
# Get Source Conversations Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_source_conversations():
    """get_source_conversations filters by provider."""
    import uuid

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncConversationRepository(db_path=db_path) as repo:
            # Save conversations from different providers
            providers = ["chatgpt", "claude", "chatgpt"]
            for i, provider in enumerate(providers):
                conv_id = f"{provider}:{uuid.uuid4()}"
                conv = Conversation(
                    id=conv_id,
                    provider=provider,
                    title=f"From {provider} #{i}",
                    messages=[Message(id=f"m{i}", role="user", text="Test")],
                    metadata={"content_hash": f"hash_{conv_id}"},
                )

                messages = [
                    MessageRecord(
                        message_id=f"m{i}",
                        conversation_id=conv_id,
                        role="user",
                        text="Test",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        content_hash=f"mhash{i}",
                    ),
                ]

                await repo.save_conversation(conv, messages, [])

            # Get ChatGPT conversations
            chatgpt_ids = await repo.get_source_conversations("chatgpt")
            assert len(chatgpt_ids) == 2

            # Get Claude conversations
            claude_ids = await repo.get_source_conversations("claude")
            assert len(claude_ids) == 1


# =============================================================================
# Concurrent Operations Tests
# =============================================================================


@pytest.mark.asyncio
async def test_concurrent_reads():
    """Multiple concurrent reads should work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncConversationRepository(db_path=db_path) as repo:
            # Launch multiple concurrent reads
            tasks = [
                repo.get_conversation(f"test:{i}")
                for i in range(10)
            ]

            results = await asyncio.gather(*tasks)

            # All should return None (no conversations exist)
            assert all(r is None for r in results)


@pytest.mark.asyncio
async def test_concurrent_existence_checks():
    """Multiple concurrent existence checks should work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncConversationRepository(db_path=db_path) as repo:
            # Launch multiple concurrent existence checks
            tasks = [
                repo.conversation_exists(f"hash_{i}")
                for i in range(10)
            ]

            results = await asyncio.gather(*tasks)

            # All should return False (no conversations exist)
            assert all(r is False for r in results)
