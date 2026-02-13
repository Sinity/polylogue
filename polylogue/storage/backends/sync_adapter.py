"""Synchronous adapter for the async SQLite backend.

Provides ``SyncFromAsyncBackend``, a thin wrapper that enables the
``AsyncSQLiteBackend`` to be used in synchronous contexts (CLI, pipeline)
via ``asyncio.run()``.  This is the bridge toward making the async backend
the single canonical implementation.

Usage::

    from polylogue.storage.backends.sync_adapter import SyncFromAsyncBackend

    backend = SyncFromAsyncBackend()
    conv = backend.get_conversation("abc123")  # runs async code synchronously
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from polylogue.lib.log import get_logger
from polylogue.storage.backends.async_sqlite import AsyncSQLiteBackend
from polylogue.storage.store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    RunRecord,
)

LOGGER = get_logger(__name__)


def _run_sync(coro):
    """Run an async coroutine synchronously.

    Uses an existing event loop if available (e.g. inside Jupyter or
    nested async contexts), otherwise creates a new one.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside an async context — use asyncio.run_coroutine_threadsafe
        # with a new thread's event loop, or just use asyncio.run in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class SyncFromAsyncBackend:
    """Synchronous wrapper around AsyncSQLiteBackend.

    Translates synchronous method calls into async calls via asyncio.run().
    Implements the same interface as SQLiteBackend so it can be used as a
    drop-in replacement.

    This is a bridge pattern: as the codebase migrates to async-first,
    callers can gradually switch from this adapter to using the async
    backend directly.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._async_backend = AsyncSQLiteBackend(db_path)
        self._db_path = db_path or self._async_backend._db_path
        # Transaction tracking for sync compatibility
        self._transaction_depth = 0

    # --- Connection lifecycle ---

    def close(self) -> None:
        """Close database connections."""
        _run_sync(self._async_backend.close())

    # --- Transaction methods ---
    # Note: These are no-ops for the adapter since each _run_sync call
    # gets its own connection context. For transactional operations,
    # use save_conversation() which handles its own transaction.

    def begin(self) -> None:
        """Begin a transaction (no-op in adapter)."""
        self._transaction_depth += 1

    def commit(self) -> None:
        """Commit a transaction (no-op in adapter)."""
        self._transaction_depth = max(0, self._transaction_depth - 1)

    def rollback(self) -> None:
        """Rollback a transaction (no-op in adapter)."""
        self._transaction_depth = max(0, self._transaction_depth - 1)

    # --- Read methods ---

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Retrieve a conversation by ID.

        Args:
            conversation_id: Conversation ID

        Returns:
            ConversationRecord if found, None otherwise
        """
        return _run_sync(self._async_backend.get_conversation(conversation_id))

    def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        """Get all messages for a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            List of message records (ordered by timestamp)
        """
        return _run_sync(self._async_backend.get_messages(conversation_id))

    def get_attachments(self, conversation_id: str) -> list[AttachmentRecord]:
        """Get all attachments for a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            List of attachment records

        Note:
            Currently not implemented in AsyncSQLiteBackend.
            Will raise NotImplementedError.
        """
        return _run_sync(self._async_backend.get_attachments(conversation_id))

    def list_conversations(self, **kwargs: Any) -> list[ConversationRecord]:
        """List conversations with optional filtering.

        Args:
            **kwargs: Forwarded to async backend (provider, limit, offset)

        Returns:
            List of conversation records
        """
        return _run_sync(self._async_backend.list_conversations(**kwargs))

    def count_conversations(self, **kwargs: Any) -> int:
        """Count conversations with optional filtering.

        Args:
            **kwargs: Forwarded to async backend (provider)

        Returns:
            Count of matching conversations
        """
        return _run_sync(self._async_backend.count_conversations(**kwargs))

    # --- Write methods ---

    def save_conversation(self, conversation: ConversationRecord) -> None:
        """Save a conversation (without messages or attachments).

        Args:
            conversation: Conversation record to save

        Note:
            This saves only the conversation record. To save a complete
            conversation with messages and attachments, use the async
            backend's save_conversation(conv, msgs, atts) directly.
        """
        _run_sync(self._async_backend.save_conversation(conversation, [], []))

    def save_messages(self, messages: list[MessageRecord]) -> None:
        """Save messages (not implemented in adapter).

        Args:
            messages: List of message records

        Note:
            The async backend handles messages in save_conversation().
            For standalone message saving, use the sync SQLiteBackend
            or extend the async backend.
        """
        # The async backend handles messages in save_conversation
        # For standalone message saving, we need a direct path
        raise NotImplementedError(
            "save_messages() not implemented in SyncFromAsyncBackend. "
            "Messages should be saved via save_conversation() or use "
            "the sync SQLiteBackend for standalone message operations."
        )

    def save_attachments(self, attachments: list[AttachmentRecord]) -> None:
        """Save attachments (not implemented in adapter).

        Args:
            attachments: List of attachment records

        Note:
            The async backend handles attachments in save_conversation().
            For standalone attachment saving, use the sync SQLiteBackend
            or extend the async backend.
        """
        raise NotImplementedError(
            "save_attachments() not implemented in SyncFromAsyncBackend. "
            "Attachments should be saved via save_conversation() or use "
            "the sync SQLiteBackend for standalone attachment operations."
        )

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all associated records.

        Args:
            conversation_id: Conversation ID to delete

        Returns:
            True if conversation was deleted, False if it didn't exist
        """
        return _run_sync(self._async_backend.delete_conversation(conversation_id))

    # --- Metadata ---

    def get_metadata(self, conversation_id: str) -> dict[str, object]:
        """Get metadata for a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Metadata dictionary (empty dict if conversation doesn't exist)
        """
        return _run_sync(self._async_backend.get_metadata(conversation_id))

    def update_metadata(self, conversation_id: str, key: str, value: object) -> None:
        """Update a single metadata key for a conversation.

        Args:
            conversation_id: Conversation ID
            key: Metadata key to update
            value: New value (will be JSON-serialized)
        """
        _run_sync(self._async_backend.update_metadata(conversation_id, key, value))

    def record_run(self, record: RunRecord) -> None:
        """Record a pipeline run.

        Args:
            record: Run record to save
        """
        _run_sync(self._async_backend.record_run(record))


__all__ = [
    "SyncFromAsyncBackend",
]
