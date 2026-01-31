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
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.core.async_facade import AsyncPolylogue
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
