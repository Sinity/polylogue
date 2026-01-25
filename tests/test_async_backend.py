"""Tests for async SQLite backend."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from polylogue.core.async_facade import AsyncPolylogue


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
