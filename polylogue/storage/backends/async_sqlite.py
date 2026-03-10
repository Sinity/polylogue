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
from polylogue.lib.json import dumps as json_dumps
from polylogue.lib.log import get_logger
from polylogue.storage.backends.connection import (
    DB_TIMEOUT,
    _build_conversation_filters,
    _build_source_scope_filter,
    _needs_stats_join,
)
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    RawConversationRecord,
    RawConversationState,
    RunRecord,
    _json_or_none,
    _make_ref_id,
    _parse_json,
    _row_to_content_block,
    _row_to_conversation,
    _row_to_message,
    _row_to_raw_conversation,
)
from polylogue.types import ConversationId

logger = get_logger(__name__)



def default_db_path() -> Path:
    """Return the default database path (same as sync backend).

    Reads from polylogue.paths at call time for test isolation.
    """
    return _paths.data_home() / "polylogue.db"


class SQLiteBackend:
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
        backend = SQLiteBackend()

        # Concurrent reads
        conv1, conv2 = await asyncio.gather(
            backend.get_conversation("id1"),
            backend.get_conversation("id2")
        )

        # Transaction for writes
        async with backend.transaction():
            await backend.save_conversation_record(conv)
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

        # Transaction depth tracking for savepoint nesting
        self._transaction_depth = 0

        # Persistent connection for explicit transaction control
        self._txn_conn: aiosqlite.Connection | None = None

        # Reusable connection for bulk operations (e.g., acquisition)
        self._bulk_conn: aiosqlite.Connection | None = None

        # Connection pool for concurrent read operations (e.g., rendering)
        self._read_pool: asyncio.Queue[aiosqlite.Connection] | None = None

    @property
    def db_path(self) -> Path:
        """Return the backing SQLite database path."""
        return self._db_path

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Public connection context for read/query helpers."""
        async with self._get_connection() as conn:
            yield conn

    async def _ensure_schema_once(self) -> None:
        """Ensure schema is initialized exactly once (thread-safe via asyncio lock)."""
        if self._schema_ensured:
            return
        async with self._schema_lock:
            if self._schema_ensured:
                return
            async with aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT) as init_conn:
                init_conn.row_factory = aiosqlite.Row
                await init_conn.execute("PRAGMA journal_mode=WAL")
                await init_conn.execute(f"PRAGMA busy_timeout = {DB_TIMEOUT * 1000}")
                await self._ensure_schema(init_conn)
            self._schema_ensured = True

    @asynccontextmanager
    async def bulk_connection(self) -> AsyncIterator[None]:
        """Keep a single connection alive for many sequential operations.

        When active, ``_get_connection()`` reuses this connection instead of
        opening a new one per call.  All writes are grouped in a single
        transaction — call ``bulk_flush()`` periodically for intermediate
        durability.  On exit the final transaction is committed.

        This avoids both connection-per-call overhead (fd/WAL exhaustion)
        and per-item fsync overhead (each commit forces a WAL flush).
        """
        await self._ensure_schema_once()
        conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute(f"PRAGMA busy_timeout = {DB_TIMEOUT * 1000}")
        await conn.execute("BEGIN IMMEDIATE")
        self._bulk_conn = conn
        # Suppress per-item commits in methods that check _transaction_depth
        self._transaction_depth += 1
        try:
            yield
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
        finally:
            self._transaction_depth -= 1
            self._bulk_conn = None
            await conn.close()

    async def bulk_flush(self) -> None:
        """Commit the current bulk transaction and start a new one.

        Call periodically during long bulk operations for intermediate
        durability.  Safe to call outside ``bulk_connection()`` (no-op).
        """
        if self._bulk_conn is not None:
            await self._bulk_conn.commit()
            await self._bulk_conn.execute("BEGIN IMMEDIATE")

    @asynccontextmanager
    async def read_pool(self, size: int = 4) -> AsyncIterator[None]:
        """Open a pool of reusable read connections for concurrent operations.

        While active, ``_get_connection()`` borrows from the pool instead of
        opening a fresh connection per call.  This eliminates per-call
        overhead: thread spawn, PRAGMA negotiation, and schema checks.

        Use for read-heavy phases like rendering where many concurrent
        tasks each need short-lived DB access.

        Args:
            size: Number of connections in the pool (match worker count)
        """
        await self._ensure_schema_once()
        pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
        connections: list[aiosqlite.Connection] = []

        for _ in range(size):
            conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute(f"PRAGMA busy_timeout = {DB_TIMEOUT * 1000}")
            connections.append(conn)
            pool.put_nowait(conn)

        self._read_pool = pool
        try:
            yield
        finally:
            self._read_pool = None
            for conn in connections:
                await conn.close()

    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get async database connection with schema ensured.

        Connection reuse priority:
        1. Transaction connection (_txn_conn) when inside begin/commit block
        2. Bulk connection (_bulk_conn) when inside bulk_connection() context
        3. Read pool connection when inside read_pool() context
        4. Fresh connection per call (fallback)
        """
        await self._ensure_schema_once()

        # Reuse transaction connection when inside begin/commit block
        if self._txn_conn is not None and self._transaction_depth > 0:
            yield self._txn_conn
            return

        # Reuse bulk connection when inside bulk_connection() context
        if self._bulk_conn is not None:
            yield self._bulk_conn
            return

        # Borrow from read pool when available
        if self._read_pool is not None:
            conn = await self._read_pool.get()
            try:
                yield conn
            finally:
                self._read_pool.put_nowait(conn)
            return

        async with aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute(f"PRAGMA busy_timeout = {DB_TIMEOUT * 1000}")
            yield conn

    async def _ensure_schema(self, conn: aiosqlite.Connection) -> None:
        """Ensure database schema exists and is at the current schema version.

        For fresh databases (version 0): apply DDL and set the current version.
        For the current version: nothing to do.
        For any other version: raise — wipe DB and re-run.
        """
        from polylogue.storage.backends.schema import (
            _VEC0_DDL,
            SCHEMA_DDL,
            SCHEMA_VERSION,
        )

        cursor = await conn.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        current_version = row[0] if row else 0

        if current_version == 0:
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.executescript(SCHEMA_DDL)
            try:
                await conn.execute("SELECT vec_version()")
                await conn.execute(_VEC0_DDL)
            except Exception:
                pass  # sqlite-vec not available
            await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            await conn.commit()
        elif current_version == SCHEMA_VERSION:
            pass  # Already at target version
        else:
            from polylogue.errors import DatabaseError

            raise DatabaseError(
                f"Database schema version {current_version} is incompatible with expected version {SCHEMA_VERSION}. "
                f"Delete the database file and re-run polylogue to create a fresh v{SCHEMA_VERSION} schema."
            )

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Context manager for database transactions.

        Acquires write lock to serialize write operations and manages
        explicit transaction control with savepoint nesting.
        """
        async with self._write_lock:
            # Create persistent connection for explicit transaction if needed
            if self._txn_conn is None:
                self._txn_conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
                self._txn_conn.row_factory = aiosqlite.Row
                await self._txn_conn.execute("PRAGMA foreign_keys = ON")
                await self._txn_conn.execute("PRAGMA journal_mode=WAL")
                await self._txn_conn.execute(f"PRAGMA busy_timeout = {DB_TIMEOUT * 1000}")

            await self.begin()
            try:
                yield
                await self.commit()
            except Exception:
                await self.rollback()
                raise

    async def begin(self) -> None:
        """Begin a transaction or nested savepoint."""
        await self._ensure_schema_once()
        if self._txn_conn is None:
            self._txn_conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
            self._txn_conn.row_factory = aiosqlite.Row
            await self._txn_conn.execute("PRAGMA foreign_keys = ON")
            await self._txn_conn.execute("PRAGMA journal_mode=WAL")
            await self._txn_conn.execute(f"PRAGMA busy_timeout = {DB_TIMEOUT * 1000}")

        if self._transaction_depth == 0:
            await self._txn_conn.execute("BEGIN IMMEDIATE")
        else:
            await self._txn_conn.execute(f"SAVEPOINT sp_{self._transaction_depth}")
        self._transaction_depth += 1

    async def commit(self) -> None:
        """Commit the current transaction or release savepoint."""
        if self._transaction_depth <= 0:
            from polylogue.errors import DatabaseError

            raise DatabaseError("No active transaction to commit")

        if self._txn_conn is None:
            from polylogue.errors import DatabaseError

            raise DatabaseError("No transaction connection")

        self._transaction_depth -= 1

        if self._transaction_depth == 0:
            await self._txn_conn.commit()
            await self._txn_conn.close()
            self._txn_conn = None
        else:
            await self._txn_conn.execute(f"RELEASE SAVEPOINT sp_{self._transaction_depth}")

    async def rollback(self) -> None:
        """Rollback to the last begin() or savepoint."""
        if self._transaction_depth <= 0:
            from polylogue.errors import DatabaseError

            raise DatabaseError("No active transaction to rollback")

        if self._txn_conn is None:
            from polylogue.errors import DatabaseError

            raise DatabaseError("No transaction connection")

        self._transaction_depth -= 1

        if self._transaction_depth == 0:
            await self._txn_conn.rollback()
            await self._txn_conn.close()
            self._txn_conn = None
        else:
            await self._txn_conn.execute(f"ROLLBACK TO SAVEPOINT sp_{self._transaction_depth}")

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Retrieve a conversation by ID."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            return _row_to_conversation(row) if row is not None else None

    async def get_conversations_batch(self, ids: list[str]) -> list[ConversationRecord]:
        """Retrieve multiple conversations in a single query.

        Preserves the order of input IDs. Missing IDs are silently skipped.

        Args:
            ids: List of conversation IDs to fetch

        Returns:
            List of ConversationRecord objects in the order of input IDs
        """
        if not ids:
            return []

        async with self._get_connection() as conn:
            placeholders = ",".join("?" for _ in ids)
            cursor = await conn.execute(
                f"SELECT * FROM conversations WHERE conversation_id IN ({placeholders})",
                ids,
            )
            rows = await cursor.fetchall()

        # Build lookup for order preservation
        by_id = {}
        for row in rows:
            by_id[row["conversation_id"]] = _row_to_conversation(row)

        return [by_id[cid] for cid in ids if cid in by_id]

    async def list_conversations(
        self,
        source: str | None = None,
        provider: str | None = None,
        providers: list[str] | None = None,
        parent_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        has_file_ops: bool = False,
        has_git_ops: bool = False,
        has_subagent: bool = False,
    ) -> list[ConversationRecord]:
        """List conversations with optional filtering and pagination.

        Args:
            source: Filter by source name
            provider: Filter by single provider name
            providers: Filter by multiple provider names (OR match, also matches source_name)
            parent_id: Filter by parent conversation ID
            since: Filter to conversations updated on/after this ISO date string
            until: Filter to conversations updated on/before this ISO date string
            title_contains: Filter to conversations whose title contains this text (case-insensitive)
            limit: Maximum number of records to return
            offset: Number of records to skip
            has_tool_use: Only conversations with tool_use blocks
            has_thinking: Only conversations with thinking blocks
            min_messages: Minimum message count
            max_messages: Maximum message count
            min_words: Minimum total word count
            has_file_ops: Only conversations with file operations (read/write/edit)
            has_git_ops: Only conversations with git operations
            has_subagent: Only conversations that spawned subagents
        """
        use_stats_join = _needs_stats_join(
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
        )
        async with self._get_connection() as conn:
            # Build query with filters
            where_sql, params = _build_conversation_filters(
                source=source,
                provider=provider,
                providers=providers,
                parent_id=parent_id,
                since=since,
                until=until,
                title_contains=title_contains,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                has_file_ops=has_file_ops,
                has_git_ops=has_git_ops,
                has_subagent=has_subagent,
            )

            if use_stats_join:
                from_clause = "FROM conversations c LEFT JOIN conversation_stats cs ON cs.conversation_id = c.conversation_id"
                select_clause = "SELECT c.*"
                order_clause = "ORDER BY (c.sort_key IS NULL) ASC, c.sort_key DESC, c.conversation_id DESC"
            else:
                from_clause = "FROM conversations"
                select_clause = "SELECT *"
                order_clause = "ORDER BY (sort_key IS NULL) ASC, sort_key DESC, conversation_id DESC"

            # Build full query with ordering and pagination
            query = f"""
                {select_clause} {from_clause}
                {where_sql}
                {order_clause}
            """

            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            elif offset > 0:
                # SQLite requires LIMIT before OFFSET; use -1 for unlimited
                query += " LIMIT -1"

            if offset > 0:
                query += " OFFSET ?"
                params.append(offset)

            cursor = await conn.execute(query, tuple(params))
            rows = await cursor.fetchall()

        return [_row_to_conversation(row) for row in rows]

    async def count_conversations(
        self,
        source: str | None = None,
        provider: str | None = None,
        providers: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        has_file_ops: bool = False,
        has_git_ops: bool = False,
        has_subagent: bool = False,
    ) -> int:
        """Count conversations matching filters without loading records.

        Accepts the same filter params as list_conversations but returns
        just the count via COUNT(*) for maximum efficiency.
        """
        use_stats_join = _needs_stats_join(
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
        )
        async with self._get_connection() as conn:
            where_sql, params = _build_conversation_filters(
                source=source,
                provider=provider,
                providers=providers,
                since=since,
                until=until,
                title_contains=title_contains,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                has_file_ops=has_file_ops,
                has_git_ops=has_git_ops,
                has_subagent=has_subagent,
            )
            if use_stats_join:
                sql = f"SELECT COUNT(*) as cnt FROM conversations c LEFT JOIN conversation_stats cs ON cs.conversation_id = c.conversation_id {where_sql}"
            else:
                sql = f"SELECT COUNT(*) as cnt FROM conversations {where_sql}"
            cursor = await conn.execute(sql, tuple(params))
            row = await cursor.fetchone()
            return int(row["cnt"])

    async def aggregate_message_stats(
        self,
        conversation_ids: list[str] | None = None,
    ) -> dict[str, int]:
        """Compute aggregate message statistics via SQL.

        Returns dict with keys: total, user, assistant, system, words_approx,
        attachments, min_sort_key, max_sort_key, providers.

        Uses index-only scans wherever possible.  The messages table rows are
        large (text + provider_meta), so full-table scans are avoided.
        - Message count: GROUP BY conversation_id uses the conversation_id
          index (reads ~4K small index entries, not 1.6M data pages).
        - Role/word breakdown: skipped (would require full-table scan).
        - Provider/date/attachments: derived from the small conversations table.
        """
        async with self._get_connection() as conn:
            if conversation_ids is not None:
                # --- Filtered path ---
                await conn.execute("CREATE TEMP TABLE IF NOT EXISTS _stat_ids (cid TEXT PRIMARY KEY)")
                await conn.execute("DELETE FROM _stat_ids")
                await conn.executemany(
                    "INSERT OR IGNORE INTO _stat_ids (cid) VALUES (?)",
                    [(cid,) for cid in conversation_ids],
                )

                # Message count via index-only GROUP BY (0.16s for 1.6M msgs)
                msg_row = await (await conn.execute("""
                    SELECT SUM(cnt) FROM (
                        SELECT COUNT(*) as cnt FROM messages
                        WHERE conversation_id IN (SELECT cid FROM _stat_ids)
                        GROUP BY conversation_id
                    )
                """)).fetchone()
                msg_total = msg_row[0] or 0

                # Date range + provider breakdown from conversations table
                date_row = await (await conn.execute("""
                    SELECT MIN(sort_key) as min_sk, MAX(sort_key) as max_sk
                    FROM conversations WHERE conversation_id IN (SELECT cid FROM _stat_ids)
                """)).fetchone()

                prov_rows = await (await conn.execute("""
                    SELECT provider_name, COUNT(*) as cnt
                    FROM conversations WHERE conversation_id IN (SELECT cid FROM _stat_ids)
                    GROUP BY provider_name ORDER BY cnt DESC
                """)).fetchall()
                providers = {r["provider_name"]: r["cnt"] for r in prov_rows}

                # Attachment count
                att_row = await (await conn.execute("""
                    SELECT COUNT(*) as cnt FROM attachment_refs
                    WHERE conversation_id IN (SELECT cid FROM _stat_ids)
                """)).fetchone()

                await conn.execute("DROP TABLE IF EXISTS _stat_ids")

                return {
                    "total": msg_total,
                    "user": 0,
                    "assistant": 0,
                    "system": 0,
                    "words_approx": 0,
                    "attachments": att_row["cnt"] or 0,
                    "min_sort_key": date_row["min_sk"],
                    "max_sort_key": date_row["max_sk"],
                    "providers": providers,
                }

            # --- Unfiltered path ---
            msg_total = (await (await conn.execute("SELECT COUNT(*) FROM messages")).fetchone())[0] or 0

            date_row = await (await conn.execute(
                "SELECT MIN(sort_key) as min_sk, MAX(sort_key) as max_sk FROM conversations"
            )).fetchone()

            prov_rows = await (await conn.execute(
                "SELECT provider_name, COUNT(*) as cnt FROM conversations GROUP BY provider_name ORDER BY cnt DESC"
            )).fetchall()
            providers = {r["provider_name"]: r["cnt"] for r in prov_rows}

            att_cnt = (await (await conn.execute("SELECT COUNT(*) FROM attachment_refs")).fetchone())[0] or 0

            return {
                "total": msg_total,
                "user": 0,
                "assistant": 0,
                "system": 0,
                "words_approx": 0,
                "attachments": att_cnt,
                "min_sort_key": date_row["min_sk"],
                "max_sort_key": date_row["max_sk"],
                "providers": providers,
            }

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

    async def save_conversation_record(self, record: ConversationRecord) -> None:
        """Persist a conversation record with upsert semantics.

        Note: metadata is NOT updated via upsert - it's user-editable and
        should only be modified via update_metadata/add_tag/remove_tag methods.

        Args:
            record: Conversation record to save
        """
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO conversations (
                    conversation_id,
                    provider_name,
                    provider_conversation_id,
                    title,
                    created_at,
                    updated_at,
                    sort_key,
                    content_hash,
                    provider_meta,
                    metadata,
                    version,
                    parent_conversation_id,
                    branch_type,
                    raw_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    title = excluded.title,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    sort_key = excluded.sort_key,
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
                    record.conversation_id,
                    record.provider_name,
                    record.provider_conversation_id,
                    record.title,
                    record.created_at,
                    record.updated_at,
                    record.sort_key,
                    record.content_hash,
                    _json_or_none(record.provider_meta),
                    _json_or_none(record.metadata) or "{}",
                    record.version,
                    record.parent_conversation_id,
                    record.branch_type,
                    record.raw_id,
                ),
            )
            if self._transaction_depth == 0:
                await conn.commit()

    async def save_conversation(
        self,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
    ) -> dict[str, int]:
        """Save a conversation with messages and attachments atomically.

        Convenience method that wraps the lower-level operations.
        Does not use explicit transactions - each operation uses its own connection.

        Args:
            conversation: Conversation record to save
            messages: List of message records
            attachments: List of attachment records

        Returns:
            Dictionary with counts:
                - conversations_created: Number of new conversations
                - messages_created: Number of new messages
                - attachments_created: Number of new attachments
        """
        counts: dict[str, int] = {
            "conversations_created": 0,
            "messages_created": 0,
            "attachments_created": 0,
        }

        # Save conversation
        existing = await self.get_conversation(conversation.conversation_id)
        await self.save_conversation_record(conversation)
        if not existing or existing.content_hash != conversation.content_hash:
            counts["conversations_created"] = 1

        # Save messages
        if messages:
            existing_messages = {
                msg.message_id: msg for msg in await self.get_messages(conversation.conversation_id)
            }
            for message in messages:
                existing_msg = existing_messages.get(message.message_id)
                if not existing_msg or existing_msg.content_hash != message.content_hash:
                    counts["messages_created"] += 1
            await self.save_messages(messages)

        # Save attachments
        if attachments:
            await self.save_attachments(attachments)
            counts["attachments_created"] = len(attachments)

        return counts

    async def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        """Get all messages for a conversation, with content_blocks attached."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY (sort_key IS NULL), sort_key, message_id",
                (conversation_id,),
            )
            rows = await cursor.fetchall()
        messages = [_row_to_message(row) for row in rows]
        if messages:
            msg_ids = [m.message_id for m in messages]
            blocks_by_msg = await self.get_content_blocks(msg_ids)
            messages = [
                m.model_copy(update={"content_blocks": blocks_by_msg.get(m.message_id, [])})
                for m in messages
            ]
        return messages

    async def get_messages_batch(self, conversation_ids: list[str]) -> dict[str, list[MessageRecord]]:
        """Get messages for multiple conversations in a single query, with content_blocks.

        Returns a dict mapping conversation_id → list of MessageRecords.
        Missing conversations produce empty lists.
        """
        if not conversation_ids:
            return {}

        result: dict[str, list[MessageRecord]] = {cid: [] for cid in conversation_ids}
        all_messages: list[MessageRecord] = []
        async with self._get_connection() as conn:
            placeholders = ",".join("?" for _ in conversation_ids)
            cursor = await conn.execute(
                f"SELECT * FROM messages WHERE conversation_id IN ({placeholders}) ORDER BY (sort_key IS NULL), sort_key, message_id",
                conversation_ids,
            )
            rows = await cursor.fetchall()

        for row in rows:
            cid = row["conversation_id"]
            msg = _row_to_message(row)
            if cid in result:
                result[cid].append(msg)
            all_messages.append(msg)

        if all_messages:
            msg_ids = [m.message_id for m in all_messages]
            blocks_by_msg = await self.get_content_blocks(msg_ids)
            for cid in result:
                result[cid] = [
                    m.model_copy(update={"content_blocks": blocks_by_msg.get(m.message_id, [])})
                    for m in result[cid]
                ]

        return result

    @staticmethod
    def _topo_sort_messages(records: list[MessageRecord]) -> list[MessageRecord]:
        """Sort messages so parents come before children (for FK constraint).

        Cross-conversation parent references (parent outside this batch) are left
        as-is — those FKs are never set by prepare.py anyway (only within-conversation
        parent_message_id is resolved).
        """
        ids_in_batch = {r.message_id for r in records}
        # Separate records with intra-batch parent from those without
        no_parent: list[MessageRecord] = []
        has_parent: list[MessageRecord] = []
        for r in records:
            if r.parent_message_id and r.parent_message_id in ids_in_batch:
                has_parent.append(r)
            else:
                no_parent.append(r)
        if not has_parent:
            return records
        # Build simple ordering: insert parents before children
        ordered: list[MessageRecord] = list(no_parent)
        inserted_ids = {r.message_id for r in ordered}
        remaining = list(has_parent)
        max_passes = len(remaining) + 1
        for _ in range(max_passes):
            if not remaining:
                break
            next_remaining: list[MessageRecord] = []
            for r in remaining:
                if r.parent_message_id in inserted_ids:
                    ordered.append(r)
                    inserted_ids.add(r.message_id)
                else:
                    next_remaining.append(r)
            remaining = next_remaining
        ordered.extend(remaining)  # Append anything still stuck (cycles)
        return ordered

    async def save_messages(self, records: list[MessageRecord]) -> None:
        """Persist multiple message records using bulk insert."""
        if not records:
            return
        records = self._topo_sort_messages(records)
        async with self._get_connection() as conn:
            query = """
                INSERT INTO messages (
                    message_id,
                    conversation_id,
                    provider_message_id,
                    role,
                    text,
                    sort_key,
                    content_hash,
                    version,
                    parent_message_id,
                    branch_index,
                    provider_name,
                    word_count,
                    has_tool_use,
                    has_thinking
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    role = excluded.role,
                    text = excluded.text,
                    sort_key = excluded.sort_key,
                    content_hash = excluded.content_hash,
                    parent_message_id = excluded.parent_message_id,
                    branch_index = excluded.branch_index,
                    provider_name = excluded.provider_name,
                    word_count = excluded.word_count,
                    has_tool_use = excluded.has_tool_use,
                    has_thinking = excluded.has_thinking
                WHERE
                    content_hash != excluded.content_hash
                    OR IFNULL(role, '') != IFNULL(excluded.role, '')
                    OR IFNULL(text, '') != IFNULL(excluded.text, '')
                    OR IFNULL(parent_message_id, '') != IFNULL(excluded.parent_message_id, '')
                    OR branch_index != excluded.branch_index
            """
            data = [
                (
                    r.message_id,
                    r.conversation_id,
                    r.provider_message_id,
                    r.role,
                    r.text,
                    r.sort_key,
                    r.content_hash,
                    r.version,
                    r.parent_message_id,
                    r.branch_index,
                    r.provider_name,
                    r.word_count,
                    r.has_tool_use,
                    r.has_thinking,
                )
                for r in records
            ]
            await conn.executemany(query, data)
            if self._transaction_depth == 0:
                await conn.commit()

    async def save_content_blocks(self, records: list[ContentBlockRecord]) -> None:
        """Persist content block records using bulk insert."""
        if not records:
            return
        async with self._get_connection() as conn:
            query = """
                INSERT INTO content_blocks (
                    block_id,
                    message_id,
                    conversation_id,
                    block_index,
                    type,
                    text,
                    tool_name,
                    tool_id,
                    tool_input,
                    media_type,
                    metadata,
                    semantic_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id, block_index) DO UPDATE SET
                    type = excluded.type,
                    text = excluded.text,
                    tool_name = excluded.tool_name,
                    tool_id = excluded.tool_id,
                    tool_input = excluded.tool_input,
                    media_type = excluded.media_type,
                    metadata = excluded.metadata,
                    semantic_type = excluded.semantic_type
            """
            data = [
                (
                    r.block_id,
                    r.message_id,
                    r.conversation_id,
                    r.block_index,
                    r.type,
                    r.text,
                    r.tool_name,
                    r.tool_id,
                    r.tool_input,
                    r.media_type,
                    r.metadata,
                    r.semantic_type,
                )
                for r in records
            ]
            await conn.executemany(query, data)
            if self._transaction_depth == 0:
                await conn.commit()

    async def upsert_conversation_stats(
        self,
        conversation_id: str,
        provider_name: str,
        messages: list[MessageRecord],
    ) -> None:
        """Upsert precomputed per-conversation aggregate stats.

        Called after save_messages to keep conversation_stats in sync.
        """
        message_count = len(messages)
        word_count = sum(m.word_count for m in messages)
        tool_use_count = sum(1 for m in messages if m.has_tool_use)
        thinking_count = sum(1 for m in messages if m.has_thinking)
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO conversation_stats
                    (conversation_id, provider_name, message_count, word_count, tool_use_count, thinking_count)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    provider_name  = excluded.provider_name,
                    message_count  = excluded.message_count,
                    word_count     = excluded.word_count,
                    tool_use_count = excluded.tool_use_count,
                    thinking_count = excluded.thinking_count
                """,
                (conversation_id, provider_name, message_count, word_count, tool_use_count, thinking_count),
            )
            if self._transaction_depth == 0:
                await conn.commit()

    async def get_content_blocks(self, message_ids: list[str]) -> dict[str, list[ContentBlockRecord]]:
        """Get content blocks for a list of message IDs.

        Returns a dict mapping message_id → ordered list of ContentBlockRecords.
        """
        if not message_ids:
            return {}
        result: dict[str, list[ContentBlockRecord]] = {mid: [] for mid in message_ids}
        async with self._get_connection() as conn:
            placeholders = ",".join("?" for _ in message_ids)
            cursor = await conn.execute(
                f"SELECT * FROM content_blocks WHERE message_id IN ({placeholders}) ORDER BY message_id, block_index",
                message_ids,
            )
            rows = await cursor.fetchall()
        for row in rows:
            mid = row["message_id"]
            if mid in result:
                result[mid].append(_row_to_content_block(row))
        return result

    async def get_attachments(self, conversation_id: str) -> list[AttachmentRecord]:
        """Get all attachments for a conversation.

        Joins the attachments table with attachment_refs to retrieve
        attachment metadata along with the message_id linkage.
        """
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
                provider_meta=_parse_json(
                    row["provider_meta"], field="provider_meta", record_id=row["attachment_id"]
                ),
            )
            for row in rows
        ]

    async def get_attachments_batch(
        self, conversation_ids: list[str]
    ) -> dict[str, list[AttachmentRecord]]:
        """Get attachments for multiple conversations in a single query.

        Returns a dict mapping conversation_id → list of AttachmentRecords.
        Missing conversations produce empty lists.
        """
        if not conversation_ids:
            return {}

        result: dict[str, list[AttachmentRecord]] = {cid: [] for cid in conversation_ids}
        async with self._get_connection() as conn:
            placeholders = ",".join("?" for _ in conversation_ids)
            cursor = await conn.execute(
                f"""
                SELECT a.*, r.message_id, r.conversation_id
                FROM attachments a
                JOIN attachment_refs r ON a.attachment_id = r.attachment_id
                WHERE r.conversation_id IN ({placeholders})
                """,
                conversation_ids,
            )
            rows = await cursor.fetchall()

        for row in rows:
            cid = row["conversation_id"]
            if cid in result:
                result[cid].append(
                    AttachmentRecord(
                        attachment_id=row["attachment_id"],
                        conversation_id=ConversationId(cid),
                        message_id=row["message_id"],
                        mime_type=row["mime_type"],
                        size_bytes=row["size_bytes"],
                        path=row["path"],
                        provider_meta=_parse_json(
                            row["provider_meta"],
                            field="provider_meta",
                            record_id=row["attachment_id"],
                        ),
                    )
                )

        return result

    async def save_attachments(self, records: list[AttachmentRecord]) -> None:
        """Persist attachment records with reference counting using bulk operations.

        Args:
            records: List of attachment records to save
        """
        if not records:
            return
        async with self._get_connection() as conn:
            # 1. Bulk Upsert attachments metadata
            att_query = """
                INSERT INTO attachments (
                    attachment_id, mime_type, size_bytes, path, ref_count, provider_meta
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(attachment_id) DO UPDATE SET
                    mime_type = COALESCE(excluded.mime_type, attachments.mime_type),
                    size_bytes = COALESCE(excluded.size_bytes, attachments.size_bytes),
                    path = COALESCE(excluded.path, attachments.path),
                    provider_meta = COALESCE(excluded.provider_meta, attachments.provider_meta)
            """
            att_data = [
                (r.attachment_id, r.mime_type, r.size_bytes, r.path, 0, _json_or_none(r.provider_meta))
                for r in records
            ]
            await conn.executemany(att_query, att_data)

            # 2. Bulk Insert or Ignore refs
            ref_query = """
                INSERT OR IGNORE INTO attachment_refs (
                    ref_id, attachment_id, conversation_id, message_id, provider_meta
                ) VALUES (?, ?, ?, ?, ?)
            """
            ref_data = []
            for r in records:
                ref_id = _make_ref_id(r.attachment_id, r.conversation_id, r.message_id)
                ref_data.append(
                    (ref_id, r.attachment_id, r.conversation_id, r.message_id, _json_or_none(r.provider_meta))
                )

            await conn.executemany(ref_query, ref_data)

            # 3. Recalculate ref counts in a single statement
            affected_aids = list({r.attachment_id for r in records})
            placeholders = ", ".join("?" for _ in affected_aids)
            await conn.execute(
                f"""
                UPDATE attachments
                SET ref_count = (
                    SELECT COUNT(*)
                    FROM attachment_refs
                    WHERE attachment_refs.attachment_id = attachments.attachment_id
                )
                WHERE attachment_id IN ({placeholders})
                """,
                tuple(affected_aids),
            )
            if self._transaction_depth == 0:
                await conn.commit()

    async def prune_attachments(
        self, conversation_id: str, keep_attachment_ids: set[str]
    ) -> None:
        """Remove attachment refs not in keep set and clean up orphaned attachments.

        Args:
            conversation_id: The conversation to prune attachments for
            keep_attachment_ids: Set of attachment IDs to keep (prune all others)
        """
        async with self._get_connection() as conn:
            # Find refs to remove (refs for this conversation not in keep set)
            if keep_attachment_ids:
                placeholders = ",".join("?" * len(keep_attachment_ids))
                cursor = await conn.execute(
                    f"""
                    SELECT attachment_id FROM attachment_refs
                    WHERE conversation_id = ? AND attachment_id NOT IN ({placeholders})
                    """,
                    (conversation_id, *keep_attachment_ids),
                )
                refs_to_remove = await cursor.fetchall()
            else:
                # No attachments to keep - remove all refs for this conversation
                cursor = await conn.execute(
                    "SELECT attachment_id FROM attachment_refs WHERE conversation_id = ?",
                    (conversation_id,),
                )
                refs_to_remove = await cursor.fetchall()

            if not refs_to_remove:
                return

            # Extract IDs
            attachment_ids_to_check = {row[0] for row in refs_to_remove}

            # Bulk Delete refs
            if keep_attachment_ids:
                placeholders = ",".join("?" * len(keep_attachment_ids))
                await conn.execute(
                    f"DELETE FROM attachment_refs WHERE conversation_id = ? AND attachment_id NOT IN ({placeholders})",
                    (conversation_id, *keep_attachment_ids),
                )
            else:
                await conn.execute(
                    "DELETE FROM attachment_refs WHERE conversation_id = ?",
                    (conversation_id,),
                )

            # Update ref counts in bulk, then clean up orphans
            aids_list = list(attachment_ids_to_check)
            aid_placeholders = ", ".join("?" for _ in aids_list)
            await conn.execute(
                f"""
                UPDATE attachments
                SET ref_count = (
                    SELECT COUNT(*)
                    FROM attachment_refs
                    WHERE attachment_refs.attachment_id = attachments.attachment_id
                )
                WHERE attachment_id IN ({aid_placeholders})
                """,
                tuple(aids_list),
            )

            # Clean up orphaned attachments (ref_count <= 0)
            await conn.execute("DELETE FROM attachments WHERE ref_count <= 0")

            if self._transaction_depth == 0:
                await conn.commit()

    async def list_conversations_by_parent(self, parent_id: str) -> list[ConversationRecord]:
        """List all conversations that have the given conversation as parent.

        Args:
            parent_id: The parent conversation ID

        Returns:
            List of child conversation records
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM conversations
                WHERE parent_conversation_id = ?
                ORDER BY created_at ASC
                """,
                (parent_id,),
            )
            rows = await cursor.fetchall()

        return [_row_to_conversation(row) for row in rows]

    async def resolve_id(self, id_prefix: str) -> str | None:
        """Resolve a partial conversation ID to a full ID.

        Supports both exact matches and prefix matches. If multiple
        conversations match the prefix, returns None (ambiguous).

        Args:
            id_prefix: Full or partial conversation ID to resolve

        Returns:
            The full conversation ID if exactly one match found, None otherwise
        """
        async with self._get_connection() as conn:
            # Try exact match first
            cursor = await conn.execute(
                "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
                (id_prefix,),
            )
            row = await cursor.fetchone()
            if row:
                return str(row["conversation_id"])

            # Try prefix match
            cursor = await conn.execute(
                "SELECT conversation_id FROM conversations WHERE conversation_id LIKE ? LIMIT 2",
                (f"{id_prefix}%",),
            )
            rows = await cursor.fetchall()

        if len(rows) == 1:
            return str(rows[0]["conversation_id"])

        return None  # No match or ambiguous

    async def get_last_sync_timestamp(self) -> str | None:
        """Return the timestamp of the most recent ingestion run, or None."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT MAX(timestamp) as last FROM runs")
            row = await cursor.fetchone()
            return row["last"] if row and row["last"] else None

    def _conversation_id_query(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Build the canonical scoped conversation-ID query."""
        predicate, params = _build_source_scope_filter(
            source_names,
            provider_column="provider_name",
            source_column="source_name",
        )
        sql = "SELECT conversation_id FROM conversations"
        if predicate:
            sql += f" WHERE {predicate}"
        sql += " ORDER BY sort_key DESC, conversation_id ASC"
        return sql, tuple(params)

    async def count_conversation_ids(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> int:
        """Count conversation IDs, optionally scoped to source names or legacy provider names."""
        predicate, params = _build_source_scope_filter(
            source_names,
            provider_column="provider_name",
            source_column="source_name",
        )
        sql = "SELECT COUNT(*) AS count FROM conversations"
        if predicate:
            sql += f" WHERE {predicate}"

        async with self._get_connection() as conn:
            cursor = await conn.execute(sql, tuple(params))
            row = await cursor.fetchone()

        return int(row["count"]) if row is not None else 0

    async def iter_conversation_ids(
        self,
        *,
        source_names: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        """Iterate conversation IDs in bounded fetch batches."""
        sql, params = self._conversation_id_query(source_names=source_names)
        async with self._get_connection() as conn:
            cursor = await conn.execute(sql, params)
            while True:
                rows = await cursor.fetchmany(page_size)
                if not rows:
                    break
                for row in rows:
                    yield str(row["conversation_id"])

    def _raw_id_query(
        self,
        *,
        source_names: list[str] | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Build the canonical scoped raw-ID query."""
        where_clauses: list[str] = []
        params: list[str] = []

        if require_unparsed:
            where_clauses.append("parsed_at IS NULL")
        if require_unvalidated:
            where_clauses.append("validated_at IS NULL")
        if validation_statuses:
            placeholders = ",".join("?" for _ in validation_statuses)
            where_clauses.append(f"validation_status IN ({placeholders})")
            params.extend(validation_statuses)

        predicate, scope_params = _build_source_scope_filter(
            source_names,
            provider_column="provider_name",
            source_column="source_name",
        )
        if predicate:
            where_clauses.append(predicate)
            params.extend(scope_params)

        sql = "SELECT raw_id FROM raw_conversations"
        if where_clauses:
            sql += f" WHERE {' AND '.join(where_clauses)}"
        sql += " ORDER BY acquired_at DESC, raw_id ASC"
        return sql, tuple(params)

    async def iter_raw_ids(
        self,
        *,
        source_names: list[str] | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        """Iterate raw conversation IDs for a pipeline state slice in bounded batches."""
        sql, params = self._raw_id_query(
            source_names=source_names,
            require_unparsed=require_unparsed,
            require_unvalidated=require_unvalidated,
            validation_statuses=validation_statuses,
        )

        async with self._get_connection() as conn:
            cursor = await conn.execute(sql, params)
            while True:
                rows = await cursor.fetchmany(page_size)
                if not rows:
                    break
                for row in rows:
                    yield str(row["raw_id"])

    async def search_conversations(
        self, query: str, limit: int = 100, providers: list[str] | None = None
    ) -> list[str]:
        """Search conversations using the canonical ranked FTS conversation query.

        Args:
            query: Raw search query string (will be escaped for FTS5)
            limit: Maximum number of conversation IDs to return
            providers: Optional list of provider names to filter by

        Returns:
            List of conversation IDs matching the query, ordered by relevance
        """
        from polylogue.storage.search import build_ranked_conversation_search_query

        async with self._get_connection() as conn:
            # Check if FTS table exists before querying
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            )
            exists = await cursor.fetchone()

            if not exists:
                from polylogue.errors import DatabaseError

                raise DatabaseError("Search index not built. Run indexing first or use a different backend.")

            query_spec = build_ranked_conversation_search_query(
                query=query,
                limit=limit,
                scope_names=providers,
            )
            if query_spec is None:
                return []

            sql, params = query_spec

            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()

        return [str(row["conversation_id"]) for row in rows]

    # --- Metadata CRUD ---

    async def get_metadata(self, conversation_id: str) -> dict[str, object]:
        """Get metadata dict for a conversation.

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
            return _parse_json(row["metadata"], field="metadata", record_id=conversation_id) or {}

    async def _metadata_read_modify_write(
        self, conversation_id: str, mutator: callable[[dict[str, object]], bool]
    ) -> None:
        """Atomically read-modify-write conversation metadata.

        Acquires the write lock and uses explicit transaction control
        for nested atomicity.

        Args:
            conversation_id: Target conversation
            mutator: Receives current metadata dict, mutates it in place,
                     returns True if a write is needed
        """
        async with self._write_lock:
            if self._txn_conn is None:
                self._txn_conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
                self._txn_conn.row_factory = aiosqlite.Row
                await self._txn_conn.execute("PRAGMA foreign_keys = ON")
                await self._txn_conn.execute("PRAGMA journal_mode=WAL")
                await self._txn_conn.execute(f"PRAGMA busy_timeout = {DB_TIMEOUT * 1000}")

            try:
                await self._txn_conn.execute("BEGIN IMMEDIATE")
                current = await self.get_metadata(conversation_id)
                if mutator(current):
                    await self._txn_conn.execute(
                        "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
                        (json_dumps(current), conversation_id),
                    )
                await self._txn_conn.commit()
            except Exception:
                await self._txn_conn.rollback()
                raise
            finally:
                if self._txn_conn is not None:
                    await self._txn_conn.close()
                    self._txn_conn = None

    async def update_metadata(self, conversation_id: str, key: str, value: object) -> None:
        """Set a single metadata key.

        Args:
            conversation_id: Conversation ID
            key: Metadata key to update
            value: New value (will be JSON-serialized)
        """

        def _set(meta: dict[str, object]) -> bool:
            meta[key] = value
            return True

        await self._metadata_read_modify_write(conversation_id, _set)

    async def delete_metadata(self, conversation_id: str, key: str) -> None:
        """Remove a metadata key.

        Args:
            conversation_id: Conversation ID
            key: Metadata key to delete
        """

        def _delete(meta: dict[str, object]) -> bool:
            if key in meta:
                del meta[key]
                return True
            return False

        await self._metadata_read_modify_write(conversation_id, _delete)

    async def add_tag(self, conversation_id: str, tag: str) -> None:
        """Add a tag to the conversation's tags list.

        Args:
            conversation_id: Conversation ID
            tag: Tag to add
        """

        def _add(meta: dict[str, object]) -> bool:
            tags = meta.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            if tag not in tags:
                tags.append(tag)
                meta["tags"] = tags
                return True
            return False

        await self._metadata_read_modify_write(conversation_id, _add)

    async def remove_tag(self, conversation_id: str, tag: str) -> None:
        """Remove a tag from the conversation's tags list.

        Args:
            conversation_id: Conversation ID
            tag: Tag to remove
        """

        def _remove(meta: dict[str, object]) -> bool:
            tags = meta.get("tags", [])
            if isinstance(tags, list) and tag in tags:
                tags.remove(tag)
                meta["tags"] = tags
                return True
            return False

        await self._metadata_read_modify_write(conversation_id, _remove)

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        """List all tags with counts, using json_each for efficiency.

        Args:
            provider: Optional provider filter.

        Returns:
            Dict of tag → count, sorted by count descending.
        """
        async with self._get_connection() as conn:
            where = "WHERE metadata IS NOT NULL AND json_extract(metadata, '$.tags') IS NOT NULL"
            params: tuple[str, ...] = ()
            if provider:
                where += " AND provider_name = ?"
                params = (provider,)
            cursor = await conn.execute(
                f"""
                SELECT tag.value AS tag_name, COUNT(*) AS cnt
                FROM conversations,
                     json_each(json_extract(metadata, '$.tags')) AS tag
                {where}
                GROUP BY tag.value
                ORDER BY cnt DESC
                """,
                params,
            )
            rows = await cursor.fetchall()

        return {row["tag_name"]: row["cnt"] for row in rows}

    async def set_metadata(self, conversation_id: str, metadata: dict[str, object]) -> None:
        """Replace entire metadata dict.

        Args:
            conversation_id: Conversation ID
            metadata: New metadata dictionary
        """
        async with self._get_connection() as conn:
            await conn.execute(
                "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
                (json_dumps(metadata), conversation_id),
            )
            if self._transaction_depth == 0:
                await conn.commit()

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and all related records.

        Removes conversation, messages, attachment refs, and FTS index entries.
        Does NOT delete attachments themselves (handled by ref counting).

        Args:
            conversation_id: ID of the conversation to delete

        Returns:
            True if deleted, False if not found
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

            # Collect attachment IDs that may become orphaned after CASCADE
            cursor = await conn.execute(
                """SELECT DISTINCT ar.attachment_id FROM attachment_refs ar
                   JOIN messages m ON ar.message_id = m.message_id
                   WHERE m.conversation_id = ?""",
                (conversation_id,),
            )
            affected_attachments = [row[0] for row in await cursor.fetchall()]

            # Delete conversation (CASCADE handles messages, attachment_refs, + FTS)
            await conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )

            # Recalculate ref_count for affected attachments and delete orphans
            if affected_attachments:
                placeholders = ",".join("?" * len(affected_attachments))
                await conn.execute(
                    f"""UPDATE attachments SET ref_count = (
                            SELECT COUNT(*) FROM attachment_refs
                            WHERE attachment_refs.attachment_id = attachments.attachment_id
                        ) WHERE attachment_id IN ({placeholders})""",
                    affected_attachments,
                )
                await conn.execute(
                    f"DELETE FROM attachments WHERE attachment_id IN ({placeholders}) AND ref_count <= 0",
                    affected_attachments,
                )

            if self._transaction_depth == 0:
                await conn.commit()
            return True

    async def iter_messages(
        self,
        conversation_id: str,
        *,
        chunk_size: int = 100,
        dialogue_only: bool = False,
        limit: int | None = None,
    ) -> AsyncIterator[MessageRecord]:
        """Stream messages in chunks instead of loading all at once.

        This is the memory-efficient alternative to get_messages() for large
        conversations. Uses cursor-based pagination with LIMIT/OFFSET to
        avoid loading the entire result set into memory.

        Args:
            conversation_id: ID of the conversation to stream messages from
            chunk_size: Number of messages to fetch per database round-trip.
                       Larger values = fewer queries but more memory per chunk.
            dialogue_only: If True, only yield user/assistant messages (skip
                          tool, system, etc.). Filtered at SQL level for efficiency.
            limit: Maximum total messages to yield. None = no limit.

        Yields:
            MessageRecord objects one at a time
        """
        offset = 0
        yielded = 0

        while True:
            async with self._get_connection() as conn:
                # Build query with optional role filter
                query = "SELECT * FROM messages WHERE conversation_id = ?"
                params: list[str | int] = [conversation_id]

                if dialogue_only:
                    query += " AND role IN ('user', 'assistant', 'human')"

                query += " ORDER BY (sort_key IS NULL), sort_key, message_id"

                # Calculate how many to fetch this round
                fetch_limit = chunk_size
                if limit is not None:
                    remaining = limit - yielded
                    if remaining <= 0:
                        break
                    fetch_limit = min(chunk_size, remaining)

                query += " LIMIT ? OFFSET ?"
                params.extend([fetch_limit, offset])

                cursor = await conn.execute(query, tuple(params))
                rows = await cursor.fetchall()

            if not rows:
                break

            for row in rows:
                yield _row_to_message(row)
                yielded += 1

                if limit is not None and yielded >= limit:
                    return

            offset += len(rows)

            # If we got fewer rows than requested, we've reached the end
            if len(rows) < fetch_limit:
                break

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, int]:
        """Get message counts without loading messages.

        Useful for UI display and deciding whether to use streaming.

        Args:
            conversation_id: ID of the conversation

        Returns:
            Dict with counts: total_messages, dialogue_messages, tool_messages
        """
        async with self._get_connection() as conn:
            # Total messages
            cursor = await conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            total = (await cursor.fetchone())["cnt"]

            # Dialogue messages (user + assistant)
            cursor = await conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ? AND role IN ('user', 'assistant', 'human')",
                (conversation_id,),
            )
            dialogue = (await cursor.fetchone())["cnt"]

        return {
            "total_messages": total,
            "dialogue_messages": dialogue,
            "tool_messages": total - dialogue,
        }

    async def get_message_counts_batch(self, conversation_ids: list[str]) -> dict[str, int]:
        """Get message counts for multiple conversations in a single query.

        Args:
            conversation_ids: List of conversation IDs

        Returns:
            Dict mapping conversation_id to message count
        """
        if not conversation_ids:
            return {}

        async with self._get_connection() as conn:
            placeholders = ",".join("?" for _ in conversation_ids)
            cursor = await conn.execute(
                f"""
                SELECT conversation_id, COUNT(*) as cnt
                FROM messages
                WHERE conversation_id IN ({placeholders})
                GROUP BY conversation_id
                """,
                conversation_ids,
            )
            rows = await cursor.fetchall()

        return {row["conversation_id"]: row["cnt"] for row in rows}

    async def get_stats_by(self, group_by: str = "provider") -> dict[str, int]:
        """Get conversation counts grouped by provider, month, or year."""
        async with self._get_connection() as conn:
            if group_by == "month":
                cursor = await conn.execute(
                    """
                    SELECT strftime('%Y-%m', updated_at) as period, COUNT(*) as count
                    FROM conversations
                    WHERE updated_at IS NOT NULL
                    GROUP BY period ORDER BY period DESC
                    """
                )
            elif group_by == "year":
                cursor = await conn.execute(
                    """
                    SELECT strftime('%Y', updated_at) as period, COUNT(*) as count
                    FROM conversations
                    WHERE updated_at IS NOT NULL
                    GROUP BY period ORDER BY period DESC
                    """
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT provider_name as period, COUNT(*) as count
                    FROM conversations
                    GROUP BY provider_name ORDER BY count DESC
                    """
                )
            rows = await cursor.fetchall()
        return {row["period"]: row["count"] for row in rows}

    async def get_provider_conversation_counts(self) -> list[dict[str, object]]:
        """Return conversation counts per provider — fast, conversations-table-only query.

        Used for the non-verbose stats bar chart which only needs conversation_count.
        Avoids the 29s LEFT JOIN over 1.67M message rows.
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT provider_name, COUNT(*) AS conversation_count
                FROM conversations
                GROUP BY provider_name
                ORDER BY conversation_count DESC
                """
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_provider_metrics_rows(self) -> list[dict[str, object]]:
        """Return raw provider aggregation rows for analytics reporting.

        Single-table GROUP BY on messages using the covering index
        idx_messages_provider_stats — index-only scan, no JOINs needed.
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT
                    provider_name,
                    COUNT(DISTINCT conversation_id)                                                AS conversation_count,
                    COUNT(*)                                                                       AS message_count,
                    SUM(CASE WHEN role = 'user'      THEN 1 ELSE 0 END)                           AS user_message_count,
                    SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END)                           AS assistant_message_count,
                    SUM(CASE WHEN role = 'user'      THEN word_count ELSE 0 END)                  AS user_word_sum,
                    SUM(CASE WHEN role = 'assistant' THEN word_count ELSE 0 END)                  AS assistant_word_sum,
                    SUM(has_tool_use)                                                              AS tool_use_count,
                    SUM(has_thinking)                                                              AS thinking_count,
                    COUNT(DISTINCT CASE WHEN has_tool_use = 1 THEN conversation_id END)           AS conversations_with_tools,
                    COUNT(DISTINCT CASE WHEN has_thinking = 1 THEN conversation_id END)           AS conversations_with_thinking
                FROM messages
                GROUP BY provider_name
                ORDER BY conversation_count DESC
                """
            )
            rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    async def get_latest_run(self) -> RunRecord | None:
        """Fetch the most recent pipeline run record."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM runs ORDER BY timestamp DESC LIMIT 1"
            )
            row = await cursor.fetchone()

        if not row:
            return None

        return RunRecord(
            run_id=row["run_id"],
            timestamp=row["timestamp"],
            plan_snapshot=_parse_json(row["plan_snapshot"], field="plan_snapshot", record_id=row["run_id"]),
            counts=_parse_json(row["counts_json"], field="counts_json", record_id=row["run_id"]),
            drift=_parse_json(row["drift_json"], field="drift_json", record_id=row["run_id"]),
            indexed=bool(row["indexed"]) if row["indexed"] is not None else None,
            duration_ms=row["duration_ms"],
        )

    async def close(self) -> None:
        """Close database connections.

        Note: Connections are managed per-context, so this only closes
        active transaction connections.
        """
        if self._txn_conn is not None:
            await self._txn_conn.close()
            self._txn_conn = None
        self._transaction_depth = 0

    async def record_run(self, record: RunRecord) -> None:
        """Record a pipeline run audit entry.

        Args:
            record: Run record containing execution metadata
        """
        async with self.transaction(), self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO runs (
                    run_id,
                    timestamp,
                    plan_snapshot,
                    counts_json,
                    drift_json,
                    indexed,
                    duration_ms
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
            if self._transaction_depth == 0:
                await conn.commit()

    # --- Raw Conversation Storage ---

    async def save_raw_conversation(self, record: RawConversationRecord) -> bool:
        """Save a raw conversation record.

        Uses INSERT OR IGNORE to avoid duplicates (raw_id is SHA256 of content).

        Args:
            record: Raw conversation record to save

        Returns:
            True if inserted, False if already exists
        """
        async with self._get_connection() as conn:
            # Two-step: try INSERT OR IGNORE first (cheap for the common
            # "already exists" case), then update mtime metadata if needed.
            cursor = await conn.execute(
                """
                INSERT OR IGNORE INTO raw_conversations (
                    raw_id,
                    provider_name,
                    payload_provider,
                    source_name,
                    source_path,
                    source_index,
                    raw_content,
                    acquired_at,
                    file_mtime,
                    parsed_at,
                    parse_error,
                    validated_at,
                    validation_status,
                    validation_error,
                    validation_drift_count,
                    validation_provider,
                    validation_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.raw_id,
                    record.provider_name,
                    record.payload_provider,
                    record.source_name,
                    record.source_path,
                    record.source_index,
                    record.raw_content,
                    record.acquired_at,
                    record.file_mtime,
                    record.parsed_at,
                    record.parse_error,
                    record.validated_at,
                    record.validation_status,
                    record.validation_error,
                    record.validation_drift_count,
                    record.validation_provider,
                    record.validation_mode,
                ),
            )
            inserted = bool(cursor.rowcount > 0)

            # If the record already existed, update mtime/source_path so
            # the mtime-based skip can settle on subsequent runs (a file
            # may be renamed or touched without changing content).
            if not inserted and record.file_mtime is not None:
                await conn.execute(
                    "UPDATE raw_conversations SET file_mtime = ?, source_path = ? "
                    "WHERE raw_id = ? AND (file_mtime IS NOT ? OR source_path IS NOT ?)",
                    (record.file_mtime, record.source_path,
                     record.raw_id, record.file_mtime, record.source_path),
                )

            if self._transaction_depth == 0:
                await conn.commit()
            return inserted

    async def get_raw_conversation(self, raw_id: str) -> RawConversationRecord | None:
        """Retrieve a raw conversation by ID.

        Args:
            raw_id: SHA256 hash of the raw content

        Returns:
            RawConversationRecord if found, None otherwise
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM raw_conversations WHERE raw_id = ?",
                (raw_id,),
            )
            row = await cursor.fetchone()

            if row is None:
                return None

            return _row_to_raw_conversation(row)

    async def mark_raw_parsed(
        self,
        raw_id: str,
        *,
        error: str | None = None,
        payload_provider: str | None = None,
    ) -> None:
        """Mark a raw conversation as parsed (or record a parse error).

        On success (error=None): sets parsed_at, clears parse_error.
        On failure (error=...): sets parse_error, leaves parsed_at NULL
        so the record will be retried on next run.

        Args:
            raw_id: Raw conversation ID to update
            error: If not None, the parse error message
            payload_provider: Durable provider classification derived from payload decoding
        """
        from datetime import datetime, timezone

        async with self._get_connection() as conn:
            if error is None:
                await conn.execute(
                    "UPDATE raw_conversations "
                    "SET parsed_at = ?, parse_error = NULL, payload_provider = COALESCE(?, payload_provider) "
                    "WHERE raw_id = ?",
                    (datetime.now(timezone.utc).isoformat(), payload_provider, raw_id),
                )
            else:
                await conn.execute(
                    "UPDATE raw_conversations "
                    "SET parse_error = ?, payload_provider = COALESCE(?, payload_provider) "
                    "WHERE raw_id = ?",
                    (error[:2000], payload_provider, raw_id),  # Truncate to avoid bloating the DB
                )
            if self._transaction_depth == 0:
                await conn.commit()

    async def mark_raw_validated(
        self,
        raw_id: str,
        *,
        status: str,
        error: str | None = None,
        drift_count: int = 0,
        provider: str | None = None,
        mode: str | None = None,
        payload_provider: str | None = None,
    ) -> None:
        """Persist validation status for a raw conversation record.

        Args:
            raw_id: Raw conversation ID to update
            status: Validation status ("passed", "failed", "skipped")
            error: Optional validation error text
            drift_count: Number of drift warnings observed for this payload
            provider: Canonical provider schema used during validation
            mode: Validation mode ("off", "advisory", "strict")
            payload_provider: Durable provider classification derived from payload decoding
        """
        from datetime import datetime, timezone

        if status not in {"passed", "failed", "skipped"}:
            raise ValueError(f"Invalid validation status: {status}")

        async with self._get_connection() as conn:
            await conn.execute(
                """
                UPDATE raw_conversations
                SET validated_at = ?,
                    validation_status = ?,
                    validation_error = ?,
                    validation_drift_count = ?,
                    validation_provider = ?,
                    validation_mode = ?,
                    payload_provider = COALESCE(?, payload_provider)
                WHERE raw_id = ?
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    status,
                    (error[:2000] if error else None),
                    max(0, int(drift_count)),
                    provider,
                    mode,
                    payload_provider,
                    raw_id,
                ),
            )
            if self._transaction_depth == 0:
                await conn.commit()

    async def get_known_source_mtimes(self) -> dict[str, str]:
        """Return {source_path: file_mtime} for all raw records with an mtime.

        Used by the acquisition stage to skip files whose mtime hasn't changed
        since the last run, replacing a full file read + SHA256 hash with a
        single stat() call.

        Returns:
            Dict mapping source_path to its last-known file_mtime
        """
        result: dict[str, str] = {}
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT source_path, file_mtime FROM raw_conversations WHERE file_mtime IS NOT NULL"
            )
            while True:
                rows = await cursor.fetchmany(1000)
                if not rows:
                    break
                for row in rows:
                    result[row["source_path"]] = row["file_mtime"]
        return result

    async def reset_parse_status(self, *, provider: str | None = None) -> int:
        """Clear parsed_at/parse_error to force re-parsing on next run.

        Args:
            provider: If set, only reset records for this provider.
                      If None, reset all records.

        Returns:
            Number of records reset
        """
        async with self._get_connection() as conn:
            if provider is not None:
                cursor = await conn.execute(
                    "UPDATE raw_conversations SET parsed_at = NULL, parse_error = NULL "
                    "WHERE provider_name = ? AND (parsed_at IS NOT NULL OR parse_error IS NOT NULL)",
                    (provider,),
                )
            else:
                cursor = await conn.execute(
                    "UPDATE raw_conversations SET parsed_at = NULL, parse_error = NULL "
                    "WHERE parsed_at IS NOT NULL OR parse_error IS NOT NULL"
                )
            if self._transaction_depth == 0:
                await conn.commit()
            return cursor.rowcount

    async def reset_validation_status(self, *, provider: str | None = None) -> int:
        """Clear validation tracking to force re-validation on next run.

        Args:
            provider: If set, only reset records for this provider.
                      If None, reset all records.

        Returns:
            Number of records reset
        """
        async with self._get_connection() as conn:
            if provider is not None:
                cursor = await conn.execute(
                    "UPDATE raw_conversations "
                    "SET validated_at = NULL, validation_status = NULL, validation_error = NULL, "
                    "validation_drift_count = NULL, validation_provider = NULL, validation_mode = NULL "
                    "WHERE provider_name = ? "
                    "AND (validated_at IS NOT NULL OR validation_status IS NOT NULL OR validation_error IS NOT NULL)",
                    (provider,),
                )
            else:
                cursor = await conn.execute(
                    "UPDATE raw_conversations "
                    "SET validated_at = NULL, validation_status = NULL, validation_error = NULL, "
                    "validation_drift_count = NULL, validation_provider = NULL, validation_mode = NULL "
                    "WHERE validated_at IS NOT NULL OR validation_status IS NOT NULL OR validation_error IS NOT NULL"
                )
            if self._transaction_depth == 0:
                await conn.commit()
            return cursor.rowcount

    async def get_raw_conversations_batch(
        self, raw_ids: list[str],
    ) -> list[RawConversationRecord]:
        """Fetch multiple raw conversations in a single query.

        Replaces sequential per-ID queries with one ``IN (?)`` query,
        eliminating N connection round-trips per batch.

        Args:
            raw_ids: List of raw_id hashes to fetch

        Returns:
            List of found records (missing IDs are silently skipped)
        """
        if not raw_ids:
            return []
        async with self._get_connection() as conn:
            placeholders = ",".join("?" * len(raw_ids))
            cursor = await conn.execute(
                f"SELECT * FROM raw_conversations WHERE raw_id IN ({placeholders})",  # noqa: S608
                raw_ids,
            )
            records: list[RawConversationRecord] = []
            while True:
                rows = await cursor.fetchmany(200)
                if not rows:
                    break
                records.extend(_row_to_raw_conversation(row) for row in rows)
            return records

    async def get_raw_conversation_states(
        self,
        raw_ids: list[str],
    ) -> dict[str, RawConversationState]:
        """Fetch persisted processing state for raw conversation IDs."""
        if not raw_ids:
            return {}

        async with self._get_connection() as conn:
            placeholders = ",".join("?" * len(raw_ids))
            cursor = await conn.execute(
                f"""
                SELECT
                    raw_id,
                    source_name,
                    source_path,
                    parsed_at,
                    parse_error,
                    payload_provider,
                    validation_status,
                    validation_provider
                FROM raw_conversations
                WHERE raw_id IN ({placeholders})
                """,
                raw_ids,
            )
            rows = await cursor.fetchall()

        return {
            row["raw_id"]: RawConversationState(
                raw_id=row["raw_id"],
                source_name=row["source_name"],
                source_path=row["source_path"],
                parsed_at=row["parsed_at"],
                parse_error=row["parse_error"],
                payload_provider=row["payload_provider"],
                validation_status=row["validation_status"],
                validation_provider=row["validation_provider"],
            )
            for row in rows
        }

    async def iter_raw_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[RawConversationRecord]:
        """Iterate over raw conversation records.

        Args:
            provider: Optional provider name to filter by
            limit: Optional maximum number of records to return

        Yields:
            RawConversationRecord objects
        """
        offset = 0
        yielded = 0

        while True:
            async with self._get_connection() as conn:
                query = "SELECT * FROM raw_conversations"
                params: list[str | int] = []

                if provider is not None:
                    query += " WHERE provider_name = ?"
                    params.append(provider)

                query += " ORDER BY acquired_at DESC"

                # Fetch in chunks
                chunk_size = 100
                query_with_limit = query + " LIMIT ? OFFSET ?"
                params.extend([chunk_size, offset])

                cursor = await conn.execute(query_with_limit, tuple(params))
                rows = await cursor.fetchall()

            if not rows:
                break

            for row in rows:
                yield _row_to_raw_conversation(row)
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

            offset += len(rows)

            # If we got fewer rows than requested, we've reached the end
            if len(rows) < chunk_size:
                break

    async def get_raw_conversation_count(self, provider: str | None = None) -> int:
        """Get count of raw conversations.

        Args:
            provider: Optional provider name to filter by

        Returns:
            Count of raw conversation records
        """
        async with self._get_connection() as conn:
            query = "SELECT COUNT(*) as cnt FROM raw_conversations"
            params: tuple[str, ...] = ()
            if provider is not None:
                query += " WHERE provider_name = ?"
                params = (provider,)
            cursor = await conn.execute(query, params)
            row = await cursor.fetchone()
            return int(row["cnt"])


__all__ = [
    "SQLiteBackend",
    "default_db_path",
]
