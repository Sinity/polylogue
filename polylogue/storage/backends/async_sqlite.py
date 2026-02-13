"""Async SQLite storage backend implementation using aiosqlite.

This backend provides async/await API for all database operations, enabling
concurrent queries and parallel processing without blocking.

Performance characteristics:
- Parallel reads: 5-10x faster for batch operations
- Write serialization: Still uses exclusive locks (SQLite limitation)
- Connection pooling: Each async context gets its own connection
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

import polylogue.paths as _paths
from polylogue.lib.log import get_logger
from polylogue.storage.backends.sqlite import _row_to_conversation, _row_to_message
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord, RunRecord

LOGGER = get_logger(__name__)


def default_db_path() -> Path:
    """Return the default database path (same as sync backend).

    Reads from polylogue.paths at call time for test isolation.
    """
    return _paths.data_home() / "polylogue.db"


class AsyncSQLiteBackend:
    """Async SQLite storage backend implementation.

    This backend provides async/await API for database operations, enabling
    true concurrency for read operations while maintaining write safety.

    Thread Safety:
        - Each async task gets its own connection via asynccontextmanager
        - Write operations still use exclusive locks (SQLite limitation)
        - Safe for concurrent async tasks

    Transaction Management:
        - Use async with backend.transaction(): ... for transactions
        - All write operations should be within a transaction
        - Nested transactions use SAVEPOINTs

    Example:
        backend = AsyncSQLiteBackend()

        # Concurrent reads
        conv1, conv2 = await asyncio.gather(
            backend.get_conversation("id1"),
            backend.get_conversation("id2")
        )

        # Transaction for writes
        async with backend.transaction():
            await backend.save_conversation(conv, msgs, atts)
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize async SQLite backend.

        Args:
            db_path: Path to SQLite database file. If None, uses default path.
        """
        self._db_path = Path(db_path) if db_path is not None else default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Write lock for serializing write operations
        self._write_lock = asyncio.Lock()

        # Lock and flag for schema initialization (prevents race condition)
        self._schema_lock = asyncio.Lock()
        self._schema_ensured = False

    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get async database connection with schema ensured.

        Each connection is independent and can be used concurrently.
        Uses a lock to prevent race conditions during schema initialization.
        """
        # Ensure schema is initialized before any operation (uses lock to prevent race)
        if not self._schema_ensured:
            async with self._schema_lock:
                # Double-check after acquiring lock
                if not self._schema_ensured:
                    async with aiosqlite.connect(self._db_path, timeout=30) as init_conn:
                        init_conn.row_factory = aiosqlite.Row
                        await init_conn.execute("PRAGMA journal_mode=WAL")
                        await init_conn.execute("PRAGMA busy_timeout = 30000")
                        await self._ensure_schema(init_conn)
                    self._schema_ensured = True

        async with aiosqlite.connect(self._db_path, timeout=30) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout = 30000")
            yield conn

    async def _ensure_schema(self, conn: aiosqlite.Connection) -> None:
        """Ensure database schema exists and is at current version.

        For fresh databases (version 0), creates the schema directly.
        For existing databases, runs migrations via the sync backend's
        migration logic (wrapped in asyncio.to_thread since migrations
        are synchronous and run once).
        """
        import asyncio
        import sqlite3

        from polylogue.storage.backends.sqlite import (
            SCHEMA_VERSION,
            _load_sqlite_vec,
        )
        from polylogue.storage.backends.sqlite import (
            _ensure_schema as _sync_ensure_schema,
        )

        # Check current schema version
        cursor = await conn.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        current_version = row[0] if row else 0

        if current_version == 0:
            # Fresh database - apply schema directly via async
            from polylogue.storage.backends.sqlite import SCHEMA_DDL

            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.executescript(SCHEMA_DDL)

            # Try to create vec0 table if sqlite-vec is available
            try:
                await conn.execute("SELECT vec_version()")
                from polylogue.storage.backends.sqlite import _VEC0_DDL
                await conn.execute(_VEC0_DDL)
            except Exception:
                pass  # sqlite-vec not available

            await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            await conn.commit()
        elif current_version < SCHEMA_VERSION:
            # Existing database needs migration - use sync migration logic
            # Migrations are synchronous, run once, and involve DDL changes
            # that are best handled by the battle-tested sync path
            def _run_sync_migration() -> None:
                sync_conn = sqlite3.connect(self._db_path, timeout=30)
                sync_conn.row_factory = sqlite3.Row
                try:
                    sync_conn.execute("PRAGMA foreign_keys = ON")
                    sync_conn.execute("PRAGMA journal_mode=WAL")
                    sync_conn.execute("PRAGMA busy_timeout = 30000")
                    _load_sqlite_vec(sync_conn)
                    _sync_ensure_schema(sync_conn)
                finally:
                    sync_conn.close()

            await asyncio.to_thread(_run_sync_migration)
        elif current_version > SCHEMA_VERSION:
            from polylogue.storage.backends.sqlite import DatabaseError

            raise DatabaseError(
                f"Unsupported DB schema version {current_version} (expected {SCHEMA_VERSION})"
            )

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Context manager for database transactions.

        Acquires write lock to serialize write operations.
        """
        async with self._write_lock:
            yield

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Retrieve a conversation by ID."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            return _row_to_conversation(row) if row is not None else None

    async def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        """Get all messages for a conversation."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,),
            )
            rows = await cursor.fetchall()
            return [_row_to_message(row) for row in rows]

    async def list_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ConversationRecord]:
        """List conversations with optional filtering."""
        async with self._get_connection() as conn:
            query = "SELECT * FROM conversations"
            params: list[object] = []

            if provider:
                query += " WHERE provider_name = ?"
                params.append(provider)

            query += " ORDER BY updated_at DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)
            elif offset:
                # SQLite requires LIMIT before OFFSET; -1 means unlimited
                query += " LIMIT -1"

            if offset:
                query += " OFFSET ?"
                params.append(offset)

            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            return [_row_to_conversation(row) for row in rows]

    async def conversation_exists_by_hash(self, content_hash: str) -> bool:
        """Check if conversation with given content hash exists.

        Args:
            content_hash: SHA-256 hash of conversation content

        Returns:
            True if conversation exists, False otherwise
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT 1 FROM conversations WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            )
            row = await cursor.fetchone()
            return row is not None

    async def save_conversation(
        self,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
    ) -> dict[str, int]:
        """Save a full conversation with messages and attachments.

        Args:
            conversation: Conversation record
            messages: List of message records
            attachments: List of attachment records

        Returns:
            Dictionary with counts of created/updated records
        """
        counts = {
            "conversations_created": 0,
            "conversations_updated": 0,
            "messages_created": 0,
            "attachments_created": 0,
            "attachment_refs_created": 0,
        }

        async with self.transaction(), self._get_connection() as conn:
            import json

            # 1. Save Conversation
            # Check if exists to determine created vs updated
            cursor = await conn.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?",
                (conversation.conversation_id,),
            )
            exists = (await cursor.fetchone()) is not None
            if exists:
                counts["conversations_updated"] = 1
            else:
                counts["conversations_created"] = 1

            await conn.execute(
                """
                    INSERT INTO conversations (
                        conversation_id, provider_name, provider_conversation_id,
                        title, created_at, updated_at, content_hash,
                        provider_meta, metadata, version,
                        parent_conversation_id, branch_type, raw_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                        title = excluded.title,
                        created_at = excluded.created_at,
                        updated_at = excluded.updated_at,
                        content_hash = excluded.content_hash,
                        provider_meta = excluded.provider_meta,
                        parent_conversation_id = excluded.parent_conversation_id,
                        branch_type = excluded.branch_type,
                        raw_id = COALESCE(excluded.raw_id, conversations.raw_id)
                    WHERE
                        content_hash != excluded.content_hash
                        OR IFNULL(title, '') != IFNULL(excluded.title, '')
                        OR IFNULL(created_at, '') != IFNULL(excluded.created_at, '')
                        OR IFNULL(updated_at, '') != IFNULL(excluded.updated_at, '')
                        OR IFNULL(provider_meta, '') != IFNULL(excluded.provider_meta, '')
                        OR IFNULL(parent_conversation_id, '') != IFNULL(excluded.parent_conversation_id, '')
                        OR IFNULL(branch_type, '') != IFNULL(excluded.branch_type, '')
                        OR IFNULL(raw_id, '') != IFNULL(excluded.raw_id, '')
                    """,
                (
                    conversation.conversation_id,
                    conversation.provider_name,
                    conversation.provider_conversation_id,
                    conversation.title,
                    conversation.created_at,
                    conversation.updated_at,
                    conversation.content_hash,
                    json.dumps(conversation.provider_meta) if conversation.provider_meta else None,
                    json.dumps(conversation.metadata) if conversation.metadata else "{}",
                    conversation.version,
                    conversation.parent_conversation_id,
                    conversation.branch_type,
                    conversation.raw_id,
                ),
            )

            # 2. Save Messages (batched with executemany for performance)
            if messages:
                message_data = [
                    (
                        msg.message_id,
                        msg.conversation_id,
                        msg.provider_message_id,
                        msg.role,
                        msg.text,
                        msg.timestamp,
                        msg.content_hash,
                        json.dumps(msg.provider_meta) if msg.provider_meta else None,
                        msg.version or 1,
                        msg.parent_message_id,
                        msg.branch_index,
                    )
                    for msg in messages
                ]
                await conn.executemany(
                    """
                        INSERT INTO messages (
                            message_id, conversation_id, provider_message_id,
                            role, text, timestamp, content_hash,
                            provider_meta, version,
                            parent_message_id, branch_index
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(message_id) DO UPDATE SET
                            role = excluded.role,
                            text = excluded.text,
                            timestamp = excluded.timestamp,
                            content_hash = excluded.content_hash,
                            provider_meta = excluded.provider_meta,
                            parent_message_id = excluded.parent_message_id,
                            branch_index = excluded.branch_index
                        WHERE
                            content_hash != excluded.content_hash
                            OR IFNULL(role, '') != IFNULL(excluded.role, '')
                            OR IFNULL(text, '') != IFNULL(excluded.text, '')
                            OR IFNULL(timestamp, '') != IFNULL(excluded.timestamp, '')
                            OR IFNULL(provider_meta, '') != IFNULL(excluded.provider_meta, '')
                            OR IFNULL(parent_message_id, '') != IFNULL(excluded.parent_message_id, '')
                            OR branch_index != excluded.branch_index
                        """,
                    message_data,
                )
            counts["messages_created"] = len(messages)

            # 3. Save Attachments (batched)
            if attachments:
                attachment_data = [
                    (
                        att.attachment_id,
                        att.mime_type,
                        att.size_bytes,
                        str(att.path) if att.path else None,
                        json.dumps(att.provider_meta) if att.provider_meta else None,
                    )
                    for att in attachments
                ]
                await conn.executemany(
                    """
                        INSERT OR IGNORE INTO attachments (
                            attachment_id, mime_type, size_bytes,
                            path, ref_count, provider_meta
                        ) VALUES (?, ?, ?, ?, 0, ?)
                        """,
                    attachment_data,
                )
            counts["attachments_created"] = len(attachments)

            # 4. Save Attachment Refs (batched)
            from polylogue.storage.store import _make_ref_id

            if attachments:
                ref_data = [
                    (
                        _make_ref_id(att.attachment_id, att.conversation_id, att.message_id),
                        att.attachment_id,
                        att.conversation_id,
                        att.message_id,
                        json.dumps(att.provider_meta) if att.provider_meta else None,
                    )
                    for att in attachments
                ]
                await conn.executemany(
                    """
                        INSERT OR IGNORE INTO attachment_refs (
                            ref_id, attachment_id, conversation_id, message_id, provider_meta
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                    ref_data,
                )
            counts["attachment_refs_created"] = len(attachments)

            await conn.commit()

        return counts

    async def close(self) -> None:
        """Close database connections.

        Note: Connections are managed per-context, so this is a no-op.
        Kept for API compatibility with sync backend.
        """
        pass

    async def get_attachments(self, conversation_id: str) -> list[AttachmentRecord]:
        """Get all attachments for a conversation.

        Joins the attachments table with attachment_refs to retrieve
        attachment metadata along with the message_id linkage.
        """
        from polylogue.storage.backends.sqlite import _parse_json
        from polylogue.types import ConversationId

        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT a.*, r.message_id
                FROM attachments a
                JOIN attachment_refs r ON a.attachment_id = r.attachment_id
                WHERE r.conversation_id = ?
                """,
                (conversation_id,),
            )
            rows = await cursor.fetchall()

        return [
            AttachmentRecord(
                attachment_id=row["attachment_id"],
                conversation_id=ConversationId(conversation_id),
                message_id=row["message_id"],
                mime_type=row["mime_type"],
                size_bytes=row["size_bytes"],
                path=row["path"],
                provider_meta=_parse_json(row["provider_meta"], field="provider_meta", record_id=row["attachment_id"]),
            )
            for row in rows
        ]

    async def count_conversations(
        self,
        provider: str | None = None,
    ) -> int:
        """Count conversations with optional filtering.

        Args:
            provider: Optional provider name to filter by

        Returns:
            Count of matching conversations
        """
        async with self._get_connection() as conn:
            query = "SELECT COUNT(*) FROM conversations"
            params: list[object] = []

            if provider:
                query += " WHERE provider_name = ?"
                params.append(provider)

            cursor = await conn.execute(query, params)
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def get_metadata(self, conversation_id: str) -> dict[str, object]:
        """Get metadata for a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Metadata dictionary (empty dict if conversation doesn't exist)
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT metadata FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return {}

            import json
            metadata_str = row[0]
            if metadata_str is None or metadata_str == "":
                return {}
            return json.loads(metadata_str)

    async def update_metadata(self, conversation_id: str, key: str, value: object) -> None:
        """Update a single metadata key for a conversation.

        Args:
            conversation_id: Conversation ID
            key: Metadata key to update
            value: New value (will be JSON-serialized)
        """
        async with self.transaction(), self._get_connection() as conn:
            import json

            # Read current metadata
            cursor = await conn.execute(
                "SELECT metadata FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                # Conversation doesn't exist - this is a no-op (matches sync behavior)
                return

            # Update metadata
            metadata_str = row[0]
            metadata = {} if metadata_str is None or metadata_str == "" else json.loads(metadata_str)

            metadata[key] = value

            # Write back
            await conn.execute(
                "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
                (json.dumps(metadata), conversation_id),
            )
            await conn.commit()

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all associated records.

        Args:
            conversation_id: Conversation ID to delete

        Returns:
            True if conversation was deleted, False if it didn't exist
        """
        async with self.transaction(), self._get_connection() as conn:
            # Check if exists
            cursor = await conn.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            exists = (await cursor.fetchone()) is not None

            if not exists:
                return False

            # Delete messages (will cascade delete attachment_refs via FK)
            await conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )

            # Delete attachment refs directly (if not cascaded)
            await conn.execute(
                "DELETE FROM attachment_refs WHERE conversation_id = ?",
                (conversation_id,),
            )

            # Delete conversation
            await conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )

            await conn.commit()
            return True

    async def record_run(self, record: RunRecord) -> None:
        """Record a pipeline run.

        Args:
            record: Run record to save
        """
        from polylogue.storage.store import _json_or_none

        async with self.transaction(), self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO runs (
                    run_id, timestamp, plan_snapshot, counts_json, drift_json, indexed, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.timestamp,
                    _json_or_none(record.plan_snapshot),
                    _json_or_none(record.counts),
                    _json_or_none(record.drift),
                    record.indexed,
                    record.duration_ms,
                ),
            )
            await conn.commit()
