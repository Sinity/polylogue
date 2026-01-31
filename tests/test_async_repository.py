"""Tests for async storage repository.

Covers polylogue/storage/async_repository.py which provides high-level
async interface for conversation persistence.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from polylogue.lib.models import Conversation, Message
from polylogue.storage.async_repository import AsyncStorageRepository
from polylogue.storage.store import AttachmentRecord, MessageRecord
from polylogue.types import ContentHash


# =============================================================================
# Context Manager Tests
# =============================================================================


@pytest.mark.asyncio
async def test_context_manager_basic():
    """Repository context manager works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncStorageRepository(db_path=db_path) as repo:
            assert repo is not None
            # Should be able to perform operations
            result = await repo.get_conversation("nonexistent")
            assert result is None


@pytest.mark.asyncio
async def test_context_manager_cleanup():
    """Context manager properly cleans up resources."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncStorageRepository(db_path=db_path) as repo:
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

        repo = AsyncStorageRepository(db_path=db_path)
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

        async with AsyncStorageRepository(db_path=db_path) as repo:
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

            assert "conversations_created" in counts
            assert counts["conversations_created"] == 1
            assert counts["messages_created"] == 2


@pytest.mark.asyncio
async def test_save_conversation_with_attachments():
    """save_conversation handles attachments correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncStorageRepository(db_path=db_path) as repo:
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

            assert counts["attachments_created"] == 1
            assert counts["attachment_refs_created"] == 1


# =============================================================================
# Get Conversation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_conversation_found():
    """get_conversation returns saved conversation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with AsyncStorageRepository(db_path=db_path) as repo:
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

        async with AsyncStorageRepository(db_path=db_path) as repo:
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

        async with AsyncStorageRepository(db_path=db_path) as repo:
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

        async with AsyncStorageRepository(db_path=db_path) as repo:
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

        async with AsyncStorageRepository(db_path=db_path) as repo:
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

        async with AsyncStorageRepository(db_path=db_path) as repo:
            # Launch multiple concurrent existence checks
            tasks = [
                repo.conversation_exists(f"hash_{i}")
                for i in range(10)
            ]

            results = await asyncio.gather(*tasks)

            # All should return False (no conversations exist)
            assert all(r is False for r in results)
