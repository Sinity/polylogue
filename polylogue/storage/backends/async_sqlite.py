"""Async SQLite storage backend implementation using aiosqlite.

This backend provides async/await API for all database operations, enabling
concurrent queries and parallel processing without blocking.

Performance characteristics:
- Parallel reads: 5-10x faster for batch operations
- Write serialization: Still uses exclusive locks (SQLite limitation)
- Connection pooling: Each async context gets its own connection
"""

from __future__ import annotations

from pathlib import Path

<<<<<<< HEAD
import aiosqlite

import polylogue.paths as _paths
from polylogue.logging import get_logger
from polylogue.storage.backends.connection import (
    DB_TIMEOUT,
||||||| parent of 91024994 (refactor: split async sqlite and check plain roots)
import aiosqlite

from polylogue.storage.backends.async_sqlite_archive import SQLiteArchiveMixin
from polylogue.storage.backends.async_sqlite_derived_actions import (
    SQLiteDerivedActionsMixin,
=======
from polylogue.storage.backends.async_sqlite_archive import SQLiteArchiveMixin
from polylogue.storage.backends.async_sqlite_connections import (
    bulk_connection as backend_bulk_connection,
)
from polylogue.storage.backends.async_sqlite_connections import (
    bulk_flush as backend_bulk_flush,
)
from polylogue.storage.backends.async_sqlite_connections import (
    connection as backend_connection,
)
from polylogue.storage.backends.async_sqlite_connections import (
    get_connection as backend_get_connection,
)
from polylogue.storage.backends.async_sqlite_connections import (
    read_pool as backend_read_pool,
)
from polylogue.storage.backends.async_sqlite_derived_actions import (
    SQLiteDerivedActionsMixin,
>>>>>>> 91024994 (refactor: split async sqlite and check plain roots)
)
<<<<<<< HEAD
from polylogue.storage.backends.queries import (
    artifacts as artifacts_q,
)
from polylogue.storage.backends.queries import (
    attachments as attachments_q,
||||||| parent of c5d6c6a9 (refactor: narrow governance/health/repair (27 files deleted))
from polylogue.storage.backends.async_sqlite_derived_maintenance import (
    SQLiteDerivedMaintenanceMixin,
)
from polylogue.storage.backends.async_sqlite_derived_products import (
    SQLiteDerivedProductsMixin,
=======
from polylogue.storage.backends.async_sqlite_derived_products import (
    SQLiteDerivedProductsMixin,
>>>>>>> c5d6c6a9 (refactor: narrow governance/health/repair (27 files deleted))
)
<<<<<<< HEAD
from polylogue.storage.backends.queries import (
    conversations as conversations_q,
)
from polylogue.storage.backends.queries import (
    messages as messages_q,
)
from polylogue.storage.backends.queries import (
    publications as publications_q,
)
from polylogue.storage.backends.queries import (
    raw as raw_queries,
)
from polylogue.storage.backends.queries import (
    runs as runs_q,
)
from polylogue.storage.backends.queries import (
    stats as stats_q,
||||||| parent of 91024994 (refactor: split async sqlite and check plain roots)
from polylogue.storage.backends.async_sqlite_derived_stats import SQLiteDerivedStatsMixin
from polylogue.storage.backends.async_sqlite_raw import SQLiteRawMixin
from polylogue.storage.backends.async_sqlite_schema import ensure_schema
from polylogue.storage.backends.async_sqlite_support import (
    configure_connection,
    default_db_path,
=======
from polylogue.storage.backends.async_sqlite_derived_stats import SQLiteDerivedStatsMixin
from polylogue.storage.backends.async_sqlite_raw import SQLiteRawMixin
from polylogue.storage.backends.async_sqlite_runtime import (
    ensure_schema as ensure_backend_schema,
>>>>>>> 91024994 (refactor: split async sqlite and check plain roots)
)
<<<<<<< HEAD
from polylogue.storage.backends.query_store import SQLiteQueryStore
from polylogue.storage.store import (
    ArtifactObservationRecord,
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    PublicationRecord,
    RawConversationRecord,
    RawConversationState,
    RunRecord,
)
from polylogue.types import Provider, ValidationMode, ValidationStatus

logger = get_logger(__name__)
||||||| parent of 91024994 (refactor: split async sqlite and check plain roots)
from polylogue.storage.backends.connection import DB_TIMEOUT
from polylogue.storage.backends.query_store import SQLiteQueryStore
=======
from polylogue.storage.backends.async_sqlite_runtime import (
    ensure_schema_once,
    initialize_backend_state,
)
from polylogue.storage.backends.async_sqlite_support import default_db_path
from polylogue.storage.backends.async_sqlite_transactions import (
    begin as backend_begin,
)
from polylogue.storage.backends.async_sqlite_transactions import (
    close_backend,
)
from polylogue.storage.backends.async_sqlite_transactions import (
    commit as backend_commit,
)
from polylogue.storage.backends.async_sqlite_transactions import (
    rollback as backend_rollback,
)
from polylogue.storage.backends.async_sqlite_transactions import (
    transaction as backend_transaction,
)
from polylogue.storage.backends.schema import SCHEMA_DDL
>>>>>>> 91024994 (refactor: split async sqlite and check plain roots)


<<<<<<< HEAD
def default_db_path() -> Path:
    """Return the default database path (same as sync backend).

    Reads from polylogue.paths at call time for test isolation.
    """
    return _paths.data_home() / "polylogue.db"


class SQLiteBackend:
||||||| parent of c5d6c6a9 (refactor: narrow governance/health/repair (27 files deleted))
class SQLiteBackend(
    SQLiteArchiveMixin,
    SQLiteDerivedActionsMixin,
    SQLiteDerivedProductsMixin,
    SQLiteDerivedMaintenanceMixin,
    SQLiteDerivedStatsMixin,
    SQLiteRawMixin,
):
=======
class SQLiteBackend(
    SQLiteArchiveMixin,
    SQLiteDerivedActionsMixin,
    SQLiteDerivedProductsMixin,
    SQLiteDerivedStatsMixin,
    SQLiteRawMixin,
):
>>>>>>> c5d6c6a9 (refactor: narrow governance/health/repair (27 files deleted))
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
        initialize_backend_state(self, db_path)

    @property
    def db_path(self) -> Path:
        """Return the backing SQLite database path."""
        return self._db_path

<<<<<<< HEAD
    @asynccontextmanager
    async def connection(self) -> AsyncIterator[aiosqlite.Connection]:
||||||| parent of 91024994 (refactor: split async sqlite and check plain roots)
    @property
    def transaction_depth(self) -> int:
        """Return the active transaction nesting depth."""
        return self._transaction_depth

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[aiosqlite.Connection]:
=======
    @property
    def transaction_depth(self) -> int:
        """Return the active transaction nesting depth."""
        return self._transaction_depth

    def connection(self):
>>>>>>> 91024994 (refactor: split async sqlite and check plain roots)
        """Public connection context for read/query helpers."""
        return backend_connection(self)

    async def _ensure_schema_once(self) -> None:
        """Ensure schema is initialized exactly once (thread-safe via asyncio lock)."""
<<<<<<< HEAD
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
||||||| parent of 91024994 (refactor: split async sqlite and check plain roots)
        if self._schema_ensured:
            return
        async with self._schema_lock:
            if self._schema_ensured:
                return
            async with aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT) as init_conn:
                await configure_connection(init_conn)
                await self._ensure_schema(init_conn)
            self._schema_ensured = True
=======
        await ensure_schema_once(self)
>>>>>>> 91024994 (refactor: split async sqlite and check plain roots)

    def bulk_connection(self):
        """Keep a single connection alive for many sequential operations.

        When active, ``_get_connection()`` reuses this connection instead of
        opening a new one per call.  All writes are grouped in a single
        transaction — call ``bulk_flush()`` periodically for intermediate
        durability.  On exit the final transaction is committed.

        This avoids both connection-per-call overhead (fd/WAL exhaustion)
        and per-item fsync overhead (each commit forces a WAL flush).
        """
<<<<<<< HEAD
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
||||||| parent of 91024994 (refactor: split async sqlite and check plain roots)
        await self._ensure_schema_once()
        conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
        await configure_connection(conn)
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
=======
        return backend_bulk_connection(self)
>>>>>>> 91024994 (refactor: split async sqlite and check plain roots)

    async def bulk_flush(self) -> None:
        """Commit the current bulk transaction and start a new one.

        Call periodically during long bulk operations for intermediate
        durability.  Safe to call outside ``bulk_connection()`` (no-op).
        """
        await backend_bulk_flush(self)

    def read_pool(self, size: int = 4):
        """Open a pool of reusable read connections for concurrent operations.

        While active, ``_get_connection()`` borrows from the pool instead of
        opening a fresh connection per call.  This eliminates per-call
        overhead: thread spawn, PRAGMA negotiation, and schema checks.

        Use for read-heavy phases like rendering where many concurrent
        tasks each need short-lived DB access.

        Args:
            size: Number of connections in the pool (match worker count)
        """
        return backend_read_pool(self, size=size)

<<<<<<< HEAD
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
||||||| parent of 91024994 (refactor: split async sqlite and check plain roots)
        for _ in range(size):
            conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
            await configure_connection(conn)
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
=======
    def _get_connection(self):
>>>>>>> 91024994 (refactor: split async sqlite and check plain roots)
        """Get async database connection with schema ensured.

        Connection reuse priority:
        1. Transaction connection (_txn_conn) when inside begin/commit block
        2. Bulk connection (_bulk_conn) when inside bulk_connection() context
        3. Read pool connection when inside read_pool() context
        4. Fresh connection per call (fallback)
        """
        return backend_get_connection(self)

<<<<<<< HEAD
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
            _ARTIFACT_OBSERVATION_DDL,
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
            await conn.executescript(_ARTIFACT_OBSERVATION_DDL)
        else:
            from polylogue.errors import DatabaseError

            raise DatabaseError(
                f"Database schema version {current_version} is incompatible with expected version {SCHEMA_VERSION}. "
                f"Delete the database file and re-run polylogue to create a fresh v{SCHEMA_VERSION} schema."
            )
||||||| parent of 91024994 (refactor: split async sqlite and check plain roots)
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
            await configure_connection(conn)
            yield conn

    async def _ensure_schema(self, conn: aiosqlite.Connection) -> None:
        """Ensure database schema exists and is at the current schema version."""
        await ensure_schema(conn)
=======
    async def _ensure_schema(self, conn) -> None:
        """Ensure database schema exists and is at the current schema version."""
        await ensure_backend_schema(conn)
>>>>>>> 91024994 (refactor: split async sqlite and check plain roots)

    def transaction(self):
        """Context manager for database transactions.

        Acquires write lock to serialize write operations and manages
        explicit transaction control with savepoint nesting.
        """
<<<<<<< HEAD
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
||||||| parent of 91024994 (refactor: split async sqlite and check plain roots)
        async with self._write_lock:
            # Create persistent connection for explicit transaction if needed
            if self._txn_conn is None:
                self._txn_conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
                await configure_connection(self._txn_conn)

            await self.begin()
            try:
                yield
                await self.commit()
            except Exception:
                await self.rollback()
                raise
=======
        return backend_transaction(self)
>>>>>>> 91024994 (refactor: split async sqlite and check plain roots)

    async def begin(self) -> None:
        """Begin a transaction or nested savepoint."""
<<<<<<< HEAD
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
||||||| parent of 91024994 (refactor: split async sqlite and check plain roots)
        await self._ensure_schema_once()
        if self._txn_conn is None:
            self._txn_conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
            await configure_connection(self._txn_conn)

        if self._transaction_depth == 0:
            await self._txn_conn.execute("BEGIN IMMEDIATE")
        else:
            await self._txn_conn.execute(f"SAVEPOINT sp_{self._transaction_depth}")
        self._transaction_depth += 1
=======
        await backend_begin(self)
>>>>>>> 91024994 (refactor: split async sqlite and check plain roots)

    async def commit(self) -> None:
        """Commit the current transaction or release savepoint."""
        await backend_commit(self)

    async def rollback(self) -> None:
        """Rollback to the last begin() or savepoint."""
        await backend_rollback(self)

    # --- Conversation CRUD ---

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Retrieve a conversation by ID."""
        return await self.queries.get_conversation(conversation_id)

    async def get_conversations_batch(self, ids: list[str]) -> list[ConversationRecord]:
        """Retrieve multiple conversations in a single query.

        Preserves the order of input IDs. Missing IDs are silently skipped.
        """
        return await self.queries.get_conversations_batch(ids)

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
        """List conversations with optional filtering and pagination."""
        return await self.queries.list_conversations(
            source=source,
            provider=provider,
            providers=providers,
            parent_id=parent_id,
            since=since,
            until=until,
            title_contains=title_contains,
            limit=limit,
            offset=offset,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            has_file_ops=has_file_ops,
            has_git_ops=has_git_ops,
            has_subagent=has_subagent,
        )

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
        """Count conversations matching filters without loading records."""
        return await self.queries.count_conversations(
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

    async def aggregate_message_stats(
        self,
        conversation_ids: list[str] | None = None,
    ) -> dict[str, int]:
        """Compute aggregate message statistics via SQL."""
        return await self.queries.aggregate_message_stats(conversation_ids)

    async def conversation_exists_by_hash(self, content_hash: str) -> bool:
        """Check if conversation with given content hash exists."""
        return await self.queries.conversation_exists_by_hash(content_hash)

    async def save_conversation_record(self, record: ConversationRecord) -> None:
        """Persist a conversation record with upsert semantics."""
        async with self._get_connection() as conn:
            await conversations_q.save_conversation_record(
                conn, record, self._transaction_depth
            )

    async def save_conversation(
        self,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
    ) -> dict[str, int]:
        """Save a conversation with messages and attachments atomically."""
        counts: dict[str, int] = {
            "conversations_created": 0,
            "messages_created": 0,
            "attachments_created": 0,
        }

        existing = await self.get_conversation(conversation.conversation_id)
        await self.save_conversation_record(conversation)
        if not existing or existing.content_hash != conversation.content_hash:
            counts["conversations_created"] = 1

        if messages:
            existing_messages = {
                msg.message_id: msg for msg in await self.get_messages(conversation.conversation_id)
            }
            for message in messages:
                existing_msg = existing_messages.get(message.message_id)
                if not existing_msg or existing_msg.content_hash != message.content_hash:
                    counts["messages_created"] += 1
            await self.save_messages(messages)

        if attachments:
            await self.save_attachments(attachments)
            counts["attachments_created"] = len(attachments)

        return counts

    # --- Message CRUD ---

    async def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        """Get all messages for a conversation, with content_blocks attached."""
        return await self.queries.get_messages(conversation_id)

    async def get_messages_batch(self, conversation_ids: list[str]) -> dict[str, list[MessageRecord]]:
        """Get messages for multiple conversations in a single query, with content_blocks."""
        return await self.queries.get_messages_batch(conversation_ids)

    @staticmethod
    def _topo_sort_messages(records: list[MessageRecord]) -> list[MessageRecord]:
        """Sort messages so parents come before children (for FK constraint)."""
        return messages_q.topo_sort_messages(records)

    async def save_messages(self, records: list[MessageRecord]) -> None:
        """Persist multiple message records using bulk insert."""
        async with self._get_connection() as conn:
            await messages_q.save_messages(conn, records, self._transaction_depth)

    async def save_content_blocks(self, records: list[ContentBlockRecord]) -> None:
        """Persist content block records using bulk insert."""
        async with self._get_connection() as conn:
            await attachments_q.save_content_blocks(conn, records, self._transaction_depth)

    async def upsert_conversation_stats(
        self,
        conversation_id: str,
        provider_name: str,
        messages: list[MessageRecord],
    ) -> None:
        """Upsert precomputed per-conversation aggregate stats."""
        async with self._get_connection() as conn:
            await stats_q.upsert_conversation_stats(
                conn, conversation_id, provider_name, messages, self._transaction_depth
            )

    async def get_content_blocks(self, message_ids: list[str]) -> dict[str, list[ContentBlockRecord]]:
        """Get content blocks for a list of message IDs."""
        return await self.queries.get_content_blocks(message_ids)

    async def get_attachments(self, conversation_id: str) -> list[AttachmentRecord]:
        """Get all attachments for a conversation."""
        async with self._get_connection() as conn:
            return await attachments_q.get_attachments(conn, conversation_id)

    async def get_attachments_batch(
        self, conversation_ids: list[str]
    ) -> dict[str, list[AttachmentRecord]]:
        """Get attachments for multiple conversations in a single query."""
        async with self._get_connection() as conn:
            return await attachments_q.get_attachments_batch(conn, conversation_ids)

    async def save_attachments(self, records: list[AttachmentRecord]) -> None:
        """Persist attachment records with reference counting."""
        async with self._get_connection() as conn:
            await attachments_q.save_attachments(conn, records, self._transaction_depth)

    async def prune_attachments(
        self, conversation_id: str, keep_attachment_ids: set[str]
    ) -> None:
        """Remove attachment refs not in keep set and clean up orphaned attachments."""
        async with self._get_connection() as conn:
            await attachments_q.prune_attachments(
                conn, conversation_id, keep_attachment_ids, self._transaction_depth
            )

    async def list_conversations_by_parent(self, parent_id: str) -> list[ConversationRecord]:
        """List all conversations that have the given conversation as parent."""
        async with self._get_connection() as conn:
            return await conversations_q.list_conversations_by_parent(conn, parent_id)

    async def resolve_id(self, id_prefix: str) -> str | None:
        """Resolve a partial conversation ID to a full ID."""
        return await self.queries.resolve_id(id_prefix)

    async def get_last_sync_timestamp(self) -> str | None:
        """Return the timestamp of the most recent ingestion run, or None."""
        return await self.queries.get_last_sync_timestamp()

    def _conversation_id_query(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Build the canonical scoped conversation-ID query."""
        return self.queries.conversation_id_query(source_names=source_names)

    async def count_conversation_ids(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> int:
        """Count conversation IDs, optionally scoped to source names."""
        return await self.queries.count_conversation_ids(source_names=source_names)

    async def iter_conversation_ids(
        self,
        *,
        source_names: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        """Iterate conversation IDs in bounded fetch batches."""
        async for cid in self.queries.iter_conversation_ids(
            source_names=source_names, page_size=page_size
        ):
            yield cid

    def _raw_id_query(
        self,
        *,
        source_names: list[str] | None = None,
        provider_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Build the canonical scoped raw-ID query."""
        return self.queries.raw_id_query(
            source_names=source_names,
            provider_name=provider_name,
            require_unparsed=require_unparsed,
            require_unvalidated=require_unvalidated,
            validation_statuses=validation_statuses,
        )

    async def iter_raw_ids(
        self,
        *,
        source_names: list[str] | None = None,
        provider_name: str | None = None,
        require_unparsed: bool = False,
        require_unvalidated: bool = False,
        validation_statuses: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        """Iterate raw conversation IDs for a pipeline state slice."""
        async for rid in self.queries.iter_raw_ids(
            source_names=source_names,
            provider_name=provider_name,
            require_unparsed=require_unparsed,
            require_unvalidated=require_unvalidated,
            validation_statuses=validation_statuses,
            page_size=page_size,
        ):
            yield rid

    async def search_conversations(
        self, query: str, limit: int = 100, providers: list[str] | None = None
    ) -> list[str]:
        """Search conversations using the canonical ranked FTS conversation query."""
        return await self.queries.search_conversations(query, limit, providers)

    # --- Metadata CRUD ---

    async def get_metadata(self, conversation_id: str) -> dict[str, object]:
        """Get metadata dict for a conversation."""
        async with self._get_connection() as conn:
            return await conversations_q.get_metadata(conn, conversation_id)

    async def _metadata_read_modify_write(
        self, conversation_id: str, mutator: callable[[dict[str, object]], bool]
    ) -> None:
        """Atomically read-modify-write conversation metadata."""
        async with self._write_lock:
            if self._txn_conn is None:
                self._txn_conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
                self._txn_conn.row_factory = aiosqlite.Row
                await self._txn_conn.execute("PRAGMA foreign_keys = ON")
                await self._txn_conn.execute("PRAGMA journal_mode=WAL")
                await self._txn_conn.execute(f"PRAGMA busy_timeout = {DB_TIMEOUT * 1000}")

            try:
                await self._txn_conn.execute("BEGIN IMMEDIATE")
                current = await conversations_q.get_metadata(self._txn_conn, conversation_id)
                if mutator(current):
                    await conversations_q.update_metadata_raw(
                        self._txn_conn, conversation_id, current
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
        """Set a single metadata key."""

        def _set(meta: dict[str, object]) -> bool:
            meta[key] = value
            return True

        await self._metadata_read_modify_write(conversation_id, _set)

    async def delete_metadata(self, conversation_id: str, key: str) -> None:
        """Remove a metadata key."""

        def _delete(meta: dict[str, object]) -> bool:
            if key in meta:
                del meta[key]
                return True
            return False

        await self._metadata_read_modify_write(conversation_id, _delete)

    async def add_tag(self, conversation_id: str, tag: str) -> None:
        """Add a tag to the conversation's tags list."""

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
        """Remove a tag from the conversation's tags list."""

        def _remove(meta: dict[str, object]) -> bool:
            tags = meta.get("tags", [])
            if isinstance(tags, list) and tag in tags:
                tags.remove(tag)
                meta["tags"] = tags
                return True
            return False

        await self._metadata_read_modify_write(conversation_id, _remove)

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        """List all tags with counts."""
        async with self._get_connection() as conn:
            return await conversations_q.list_tags(conn, provider=provider)

    async def set_metadata(self, conversation_id: str, metadata: dict[str, object]) -> None:
        """Replace entire metadata dict."""
        async with self._get_connection() as conn:
            await conversations_q.set_metadata(
                conn, conversation_id, metadata, self._transaction_depth
            )

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and all related records."""
        async with self.transaction(), self._get_connection() as conn:
            return await conversations_q.delete_conversation_sql(
                conn, conversation_id, self._transaction_depth
            )

    async def iter_messages(
        self,
        conversation_id: str,
        *,
        chunk_size: int = 100,
        dialogue_only: bool = False,
        limit: int | None = None,
    ) -> AsyncIterator[MessageRecord]:
        """Stream messages in chunks instead of loading all at once."""
        async with self._get_connection() as conn:
            async for msg in messages_q.iter_messages(
                conn,
                conversation_id,
                chunk_size=chunk_size,
                dialogue_only=dialogue_only,
                limit=limit,
            ):
                yield msg

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, int]:
        """Get message counts without loading messages."""
        return await self.queries.get_conversation_stats(conversation_id)

    async def get_message_counts_batch(self, conversation_ids: list[str]) -> dict[str, int]:
        """Get message counts for multiple conversations in a single query."""
        return await self.queries.get_message_counts_batch(conversation_ids)

    async def get_stats_by(self, group_by: str = "provider") -> dict[str, int]:
        """Get conversation counts grouped by provider, month, or year."""
        return await self.queries.get_stats_by(group_by)

    async def get_provider_conversation_counts(self) -> list[dict[str, object]]:
        """Return conversation counts per provider."""
        return await self.queries.get_provider_conversation_counts()

    async def get_provider_metrics_rows(self) -> list[dict[str, object]]:
        """Return raw provider aggregation rows for analytics reporting."""
        return await self.queries.get_provider_metrics_rows()

    async def get_latest_run(self) -> RunRecord | None:
        """Fetch the most recent pipeline run record."""
        return await self.queries.get_latest_run()

    async def get_latest_publication(
        self,
        publication_kind: str,
    ) -> PublicationRecord | None:
        """Fetch the most recent publication record for one publication kind."""
        return await self.queries.get_latest_publication(publication_kind)

    async def close(self) -> None:
        """Close database connections."""
        await close_backend(self)

    async def record_run(self, record: RunRecord) -> None:
        """Record a pipeline run audit entry."""
        async with self.transaction(), self._get_connection() as conn:
            await runs_q.record_run(conn, record, self._transaction_depth)

    async def record_publication(self, record: PublicationRecord) -> None:
        """Persist one publication manifest."""
        async with self.transaction(), self._get_connection() as conn:
            await publications_q.record_publication(conn, record, self._transaction_depth)

    # --- Raw Conversation Storage ---

    async def save_raw_conversation(self, record: RawConversationRecord) -> bool:
        """Save a raw conversation record. Returns True if inserted."""
        async with self._get_connection() as conn:
            return await raw_queries.save_raw_conversation(
                conn, record, self._transaction_depth
            )

    async def save_artifact_observation(self, record: ArtifactObservationRecord) -> bool:
        """Persist or refresh one durable artifact observation."""
        async with self._get_connection() as conn:
            return await artifacts_q.save_artifact_observation(
                conn, record, self._transaction_depth
            )

    async def get_raw_conversation(self, raw_id: str) -> RawConversationRecord | None:
        """Retrieve a raw conversation by ID."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_conversation(conn, raw_id)

    async def mark_raw_parsed(
        self,
        raw_id: str,
        *,
        error: str | None = None,
        payload_provider: Provider | str | None = None,
    ) -> None:
        """Mark a raw conversation as parsed (or record a parse error)."""
        async with self._get_connection() as conn:
            await raw_queries.mark_raw_parsed(
                conn, raw_id,
                error=error,
                payload_provider=payload_provider,
                transaction_depth=self._transaction_depth,
            )

    async def mark_raw_validated(
        self,
        raw_id: str,
        *,
        status: ValidationStatus | str,
        error: str | None = None,
        drift_count: int = 0,
        provider: Provider | str | None = None,
        mode: ValidationMode | str | None = None,
        payload_provider: Provider | str | None = None,
    ) -> None:
        """Persist validation status for a raw conversation record."""
        async with self._get_connection() as conn:
            await raw_queries.mark_raw_validated(
                conn, raw_id,
                status=status,
                error=error,
                drift_count=drift_count,
                provider=provider,
                mode=mode,
                payload_provider=payload_provider,
                transaction_depth=self._transaction_depth,
            )

    async def get_known_source_mtimes(self) -> dict[str, str]:
        """Return {source_path: file_mtime} for all raw records with an mtime."""
        return await self.queries.get_known_source_mtimes()

    async def reset_parse_status(self, *, provider: str | None = None) -> int:
        """Clear parsed_at/parse_error to force re-parsing on next run."""
        async with self._get_connection() as conn:
            return await raw_queries.reset_parse_status(
                conn, provider=provider, transaction_depth=self._transaction_depth
            )

    async def reset_validation_status(self, *, provider: str | None = None) -> int:
        """Clear validation tracking to force re-validation on next run."""
        async with self._get_connection() as conn:
            return await raw_queries.reset_validation_status(
                conn, provider=provider, transaction_depth=self._transaction_depth
            )

    async def get_raw_conversations_batch(
        self, raw_ids: list[str],
    ) -> list[RawConversationRecord]:
        """Fetch multiple raw conversations in a single query."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_conversations_batch(conn, raw_ids)

    async def get_raw_conversation_states(
        self,
        raw_ids: list[str],
    ) -> dict[str, RawConversationState]:
        """Fetch persisted processing state for raw conversation IDs."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_conversation_states(conn, raw_ids)

    async def iter_raw_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[RawConversationRecord]:
        """Iterate over raw conversation records."""
        async with self._get_connection() as conn:
            async for record in raw_queries.iter_raw_conversations(conn, provider, limit):
                yield record

    async def get_raw_conversation_count(self, provider: str | None = None) -> int:
        """Get count of raw conversations."""
        async with self._get_connection() as conn:
            return await raw_queries.get_raw_conversation_count(conn, provider)


__all__ = [
    "SCHEMA_DDL",
    "SQLiteBackend",
    "default_db_path",
]
