"""SQLite storage backend implementation."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from polylogue.errors import PolylogueError
from polylogue.lib.json import dumps as json_dumps
from polylogue.lib.log import get_logger
from polylogue.storage.backends.connection import (
    _load_sqlite_vec,
    connection_context,
    create_default_backend,
    default_db_path,
    open_connection,
)

# Re-export from submodules for backward compatibility
from polylogue.storage.backends.schema import (
    _MIGRATIONS,
    _VEC0_DDL,
    SCHEMA_DDL,
    SCHEMA_VERSION,
    _apply_schema,
    _ensure_schema,
    _ensure_vec0_table,
    _run_migrations,
)
from polylogue.storage.store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    RawConversationRecord,
    RunRecord,
    _json_or_none,
)
from polylogue.types import ConversationId

LOGGER = get_logger(__name__)


def _parse_json(raw: str | None, *, field: str = "", record_id: str = "") -> Any:
    """Parse a JSON string with diagnostic context on failure."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatabaseError(
            f"Corrupt JSON in {field} for {record_id}: {exc} (value starts: {raw[:80]!r})"
        ) from exc


def _row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    """Get a column value, returning default if the column doesn't exist.

    Handles schema version differences where optional columns may be absent.
    """
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


def _row_to_conversation(row: sqlite3.Row) -> ConversationRecord:
    """Map a SQLite row to a ConversationRecord.

    Handles missing optional columns (parent_conversation_id, branch_type,
    raw_id) for compatibility with older schema versions.
    """
    return ConversationRecord(
        conversation_id=row["conversation_id"],
        provider_name=row["provider_name"],
        provider_conversation_id=row["provider_conversation_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        content_hash=row["content_hash"],
        provider_meta=_parse_json(row["provider_meta"], field="provider_meta", record_id=row["conversation_id"]),
        metadata=_parse_json(row["metadata"], field="metadata", record_id=row["conversation_id"]),
        version=row["version"],
        parent_conversation_id=_row_get(row, "parent_conversation_id"),
        branch_type=_row_get(row, "branch_type"),
        raw_id=_row_get(row, "raw_id"),
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    """Map a SQLite row to a MessageRecord.

    Handles missing optional columns (parent_message_id, branch_index)
    for compatibility with older schema versions.
    """
    return MessageRecord(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        provider_message_id=row["provider_message_id"],
        role=row["role"],
        text=row["text"],
        timestamp=row["timestamp"],
        content_hash=row["content_hash"],
        provider_meta=_parse_json(row["provider_meta"], field="provider_meta", record_id=row["message_id"]),
        version=row["version"],
        parent_message_id=_row_get(row, "parent_message_id"),
        branch_index=_row_get(row, "branch_index", 0) or 0,
    )


def _row_to_raw_conversation(row: sqlite3.Row) -> RawConversationRecord:
    """Map a SQLite row to a RawConversationRecord."""
    return RawConversationRecord(
        raw_id=row["raw_id"],
        provider_name=row["provider_name"],
        source_name=row["source_name"],
        source_path=row["source_path"],
        source_index=row["source_index"],
        raw_content=row["raw_content"],
        acquired_at=row["acquired_at"],
        file_mtime=row["file_mtime"],
    )


def _update_ref_counts(conn: sqlite3.Connection, attachment_ids: set[str]) -> None:
    """Recalculate ref_count for the given attachment IDs from attachment_refs."""
    for aid in attachment_ids:
        conn.execute(
            """
            UPDATE attachments
            SET ref_count = (SELECT COUNT(*) FROM attachment_refs WHERE attachment_id = ?)
            WHERE attachment_id = ?
            """,
            (aid, aid),
        )


def _build_conversation_filters(
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    parent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
) -> tuple[str, list[str | int]]:
    """Build WHERE clause and params for conversation queries."""
    where_clauses: list[str] = []
    params: list[str | int] = []

    if source is not None:
        where_clauses.append("source_name = ?")
        params.append(source)
    if provider is not None:
        where_clauses.append("provider_name = ?")
        params.append(provider)
    if providers:
        placeholders = ",".join("?" for _ in providers)
        where_clauses.append(
            f"(provider_name IN ({placeholders}) OR source_name IN ({placeholders}))"
        )
        params.extend(providers)
        params.extend(providers)
    if parent_id is not None:
        where_clauses.append("parent_conversation_id = ?")
        params.append(parent_id)
    if since is not None:
        where_clauses.append("updated_at >= ?")
        params.append(since)
    if until is not None:
        where_clauses.append("updated_at <= ?")
        params.append(until)
    if title_contains is not None:
        escaped = title_contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where_clauses.append("title LIKE ? ESCAPE '\\'")
        params.append(f"%{escaped}%")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    return where_sql, params


class DatabaseError(PolylogueError):
    """Base class for database errors."""


class SQLiteBackend:
    """SQLite storage backend implementation.

    This backend provides SQLite-based storage for conversations, messages,
    and attachments. It handles schema creation, migrations, and all CRUD
    operations.

    Thread Safety:
        - Each backend instance maintains a single connection
        - Connections are NOT thread-safe
        - Use separate backend instances per thread
        - Transactions (begin/commit/rollback) are connection-scoped

    Transaction Management:
        - begin() starts a transaction (or nested savepoint)
        - commit() commits the transaction
        - rollback() rolls back to the last begin()
        - All write operations should be within a transaction
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize SQLite backend.

        Thread-safe: Each thread gets its own connection via threading.local().

        Args:
            db_path: Path to SQLite database file. If None, uses default path.
        """
        self._db_path = Path(db_path) if db_path is not None else default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        import threading

        self._local = threading.local()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create thread-local database connection.

        Each thread gets its own connection for thread safety.
        sqlite-vec extension is loaded if available.
        """
        # Check if this thread has a connection
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path, timeout=30)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA busy_timeout = 30000")
            # Load sqlite-vec extension if available (optional for vector search)
            _load_sqlite_vec(self._local.conn)
            _ensure_schema(self._local.conn)
            self._local.transaction_depth = 0
        conn: sqlite3.Connection = self._local.conn
        return conn

    def begin(self) -> None:
        """Begin a transaction or nested savepoint.

        Uses ``BEGIN IMMEDIATE`` for top-level transactions, which acquires
        a reserved lock immediately.  This prevents SQLITE_BUSY errors that
        would otherwise occur if a deferred transaction tried to promote to
        a write lock while another connection holds one.  Combined with
        ``busy_timeout=30000``, this replaces the need for a Python-level
        write lock in the repository layer.
        """
        conn = self._get_connection()
        if self._local.transaction_depth == 0:
            conn.execute("BEGIN IMMEDIATE")
        else:
            conn.execute(f"SAVEPOINT sp_{self._local.transaction_depth}")
        self._local.transaction_depth += 1

    def commit(self) -> None:
        """Commit the current transaction or release savepoint."""
        if self._local.transaction_depth <= 0:
            raise DatabaseError("No active transaction to commit")

        conn = self._get_connection()
        self._local.transaction_depth -= 1

        if self._local.transaction_depth == 0:
            conn.commit()
        else:
            conn.execute(f"RELEASE SAVEPOINT sp_{self._local.transaction_depth}")

    def rollback(self) -> None:
        """Rollback to the last begin() or savepoint."""
        if self._local.transaction_depth <= 0:
            raise DatabaseError("No active transaction to rollback")

        conn = self._get_connection()
        self._local.transaction_depth -= 1

        if self._local.transaction_depth == 0:
            conn.rollback()
        else:
            conn.execute(f"ROLLBACK TO SAVEPOINT sp_{self._local.transaction_depth}")

    def record_run(self, record: RunRecord) -> None:
        """Record a pipeline run audit entry.

        Args:
            record: Run record containing execution metadata
        """
        conn = self._get_connection()
        conn.execute(
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
        conn.commit()

    def get_conversation(self, id: str) -> ConversationRecord | None:
        """Retrieve a conversation by ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (id,),
        ).fetchone()

        if row is None:
            return None

        return _row_to_conversation(row)

    def get_conversations_batch(self, ids: list[str]) -> list[ConversationRecord]:
        """Retrieve multiple conversations in a single query.

        Preserves the order of input IDs.  Missing IDs are silently skipped.

        Args:
            ids: List of conversation IDs to fetch

        Returns:
            List of ConversationRecord objects in the order of input IDs
        """
        if not ids:
            return []

        conn = self._get_connection()
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT * FROM conversations WHERE conversation_id IN ({placeholders})",
            ids,
        ).fetchall()

        # Build lookup for order preservation
        by_id = {}
        for row in rows:
            by_id[row["conversation_id"]] = _row_to_conversation(row)

        return [by_id[cid] for cid in ids if cid in by_id]

    def list_conversations(
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
    ) -> list[ConversationRecord]:
        """List conversations with optional filtering and pagination.

        Args:
            source: Filter by source name
            provider: Filter by single provider name (for backwards compat)
            providers: Filter by multiple provider names (OR match, also matches source_name)
            parent_id: Filter by parent conversation ID
            since: Filter to conversations updated on/after this ISO date string
            until: Filter to conversations updated on/before this ISO date string
            title_contains: Filter to conversations whose title contains this text (case-insensitive)
            limit: Maximum number of records to return
            offset: Number of records to skip
        """
        conn = self._get_connection()

        # Build query with filters
        where_sql, params = _build_conversation_filters(
            source=source, provider=provider, providers=providers,
            parent_id=parent_id,
            since=since, until=until, title_contains=title_contains,
        )

        # Build full query with ordering and pagination
        query = f"""
            SELECT * FROM conversations
            {where_sql}
            ORDER BY
                CASE WHEN updated_at IS NULL OR updated_at = '' THEN 1 ELSE 0 END,
                updated_at DESC,
                conversation_id DESC
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

        rows = conn.execute(query, tuple(params)).fetchall()

        return [_row_to_conversation(row) for row in rows]

    def count_conversations(
        self,
        source: str | None = None,
        provider: str | None = None,
        providers: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
    ) -> int:
        """Count conversations matching filters without loading records.

        Accepts the same filter params as list_conversations but returns
        just the count via COUNT(*) for maximum efficiency.
        """
        conn = self._get_connection()
        where_sql, params = _build_conversation_filters(
            source=source, provider=provider, providers=providers,
            since=since, until=until, title_contains=title_contains,
        )
        row = conn.execute(
            f"SELECT COUNT(*) as cnt FROM conversations {where_sql}",
            tuple(params),
        ).fetchone()
        return int(row["cnt"])

    def save_conversation(self, record: ConversationRecord) -> None:
        """Persist a conversation record with upsert semantics.

        Note: metadata is NOT updated via upsert - it's user-editable and
        should only be modified via update_metadata/add_tag/remove_tag methods.
        """
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO conversations (
                conversation_id,
                provider_name,
                provider_conversation_id,
                title,
                created_at,
                updated_at,
                content_hash,
                provider_meta,
                metadata,
                version,
                parent_conversation_id,
                branch_type,
                raw_id
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
                record.conversation_id,
                record.provider_name,
                record.provider_conversation_id,
                record.title,
                record.created_at,
                record.updated_at,
                record.content_hash,
                _json_or_none(record.provider_meta),
                _json_or_none(record.metadata) or "{}",
                record.version,
                record.parent_conversation_id,
                record.branch_type,
                record.raw_id,
            ),
        )

    def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        """Retrieve all messages for a conversation."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,),
        ).fetchall()

        return [_row_to_message(row) for row in rows]

    def save_messages(self, records: list[MessageRecord]) -> None:
        """Persist multiple message records using bulk insert."""
        if not records:
            return
        conn = self._get_connection()
        query = """
            INSERT INTO messages (
                message_id,
                conversation_id,
                provider_message_id,
                role,
                text,
                timestamp,
                content_hash,
                provider_meta,
                version,
                parent_message_id,
                branch_index
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
        """
        data = [
            (
                r.message_id,
                r.conversation_id,
                r.provider_message_id,
                r.role,
                r.text,
                r.timestamp,
                r.content_hash,
                _json_or_none(r.provider_meta),
                r.version,
                r.parent_message_id,
                r.branch_index,
            )
            for r in records
        ]
        conn.executemany(query, data)

    def get_attachments(self, conversation_id: str) -> list[AttachmentRecord]:
        """Retrieve all attachments for a conversation."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT a.*, r.message_id
            FROM attachments a
            JOIN attachment_refs r ON a.attachment_id = r.attachment_id
            WHERE r.conversation_id = ?
            """,
            (conversation_id,),
        ).fetchall()

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

    def prune_attachments(self, conversation_id: str, keep_attachment_ids: set[str]) -> None:
        """Remove attachment refs not in keep set and clean up orphaned attachments.

        Args:
            conversation_id: The conversation to prune attachments for
            keep_attachment_ids: Set of attachment IDs to keep (prune all others)
        """
        conn = self._get_connection()

        # Find refs to remove (refs for this conversation not in keep set)
        if keep_attachment_ids:
            placeholders = ",".join("?" * len(keep_attachment_ids))
            refs_to_remove = conn.execute(
                f"""
                SELECT attachment_id FROM attachment_refs
                WHERE conversation_id = ? AND attachment_id NOT IN ({placeholders})
                """,
                (conversation_id, *keep_attachment_ids),
            ).fetchall()
        else:
            # No attachments to keep - remove all refs for this conversation
            refs_to_remove = conn.execute(
                "SELECT attachment_id FROM attachment_refs WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchall()

        if not refs_to_remove:
            return

        # Extract IDs
        attachment_ids_to_check = {row[0] for row in refs_to_remove}

        # Bulk Delete refs
        if keep_attachment_ids:
            placeholders = ",".join("?" * len(keep_attachment_ids))
            conn.execute(
                f"DELETE FROM attachment_refs WHERE conversation_id = ? AND attachment_id NOT IN ({placeholders})",
                (conversation_id, *keep_attachment_ids),
            )
        else:
            conn.execute(
                "DELETE FROM attachment_refs WHERE conversation_id = ?",
                (conversation_id,),
            )

        _update_ref_counts(conn, attachment_ids_to_check)

        # Clean up orphaned attachments (ref_count <= 0)
        conn.execute("DELETE FROM attachments WHERE ref_count <= 0")

    def save_attachments(self, records: list[AttachmentRecord]) -> None:
        """Persist attachment records with reference counting using bulk operations."""
        if not records:
            return
        conn = self._get_connection()

        # Import here to avoid circular dependency
        from polylogue.storage.store import _make_ref_id

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
            (r.attachment_id, r.mime_type, r.size_bytes, r.path, 0, _json_or_none(r.provider_meta)) for r in records
        ]
        conn.executemany(att_query, att_data)

        # 2. Bulk Insert or Ignore refs
        ref_query = """
            INSERT OR IGNORE INTO attachment_refs (
                ref_id, attachment_id, conversation_id, message_id, provider_meta
            ) VALUES (?, ?, ?, ?, ?)
        """
        ref_data = []
        for r in records:
            ref_id = _make_ref_id(r.attachment_id, r.conversation_id, r.message_id)
            ref_data.append((ref_id, r.attachment_id, r.conversation_id, r.message_id, _json_or_none(r.provider_meta)))

        conn.executemany(ref_query, ref_data)

        # 3. Recalculate ref counts for all affected attachments in this batch
        _update_ref_counts(conn, {r.attachment_id for r in records})

    def list_conversations_by_parent(self, parent_id: str) -> list[ConversationRecord]:
        """List all conversations that have the given conversation as parent.

        Args:
            parent_id: The parent conversation ID

        Returns:
            List of child conversation records
        """
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT * FROM conversations
            WHERE parent_conversation_id = ?
            ORDER BY created_at ASC
            """,
            (parent_id,),
        ).fetchall()

        return [_row_to_conversation(row) for row in rows]

    def resolve_id(self, id_prefix: str) -> str | None:
        """Resolve a partial conversation ID to a full ID.

        Supports both exact matches and prefix matches. If multiple
        conversations match the prefix, returns None (ambiguous).

        Args:
            id_prefix: Full or partial conversation ID to resolve

        Returns:
            The full conversation ID if exactly one match found, None otherwise
        """
        conn = self._get_connection()

        # Try exact match first
        row = conn.execute(
            "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
            (id_prefix,),
        ).fetchone()
        if row:
            return str(row["conversation_id"])

        # Try prefix match
        rows = conn.execute(
            "SELECT conversation_id FROM conversations WHERE conversation_id LIKE ? LIMIT 2",
            (f"{id_prefix}%",),
        ).fetchall()

        if len(rows) == 1:
            return str(rows[0]["conversation_id"])

        return None  # No match or ambiguous

    def search_conversations(
        self, query: str, limit: int = 100, providers: list[str] | None = None
    ) -> list[str]:
        """Search conversations using full-text search with BM25 ranking.

        Escapes user input for safe FTS5 MATCH, then ranks results using
        BM25 (via FTS5's built-in rank function). Results are grouped by
        conversation with the best matching message determining position.

        Args:
            query: Raw search query string (will be escaped for FTS5)
            limit: Maximum number of conversation IDs to return
            providers: Optional list of provider names to filter by

        Returns:
            List of conversation IDs matching the query, ordered by relevance
        """
        from polylogue.storage.search import escape_fts5_query

        conn = self._get_connection()

        # Check if FTS table exists before querying
        exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()

        if not exists:
            raise DatabaseError("Search index not built. Run indexing first or use a different backend.")

        fts_query = escape_fts5_query(query)
        if not fts_query:
            return []

        if providers:
            placeholders = ",".join("?" for _ in providers)
            from_clause = "messages_fts JOIN conversations ON conversations.conversation_id = messages_fts.conversation_id"
            provider_filter = f" AND (conversations.provider_name IN ({placeholders}) OR conversations.source_name IN ({placeholders}))"
            params: tuple[Any, ...] = (fts_query, *providers, *providers, limit)
        else:
            from_clause = "messages_fts"
            provider_filter = ""
            params = (fts_query, limit)
        rows = conn.execute(
            f"SELECT DISTINCT messages_fts.conversation_id FROM {from_clause} WHERE messages_fts MATCH ?{provider_filter} LIMIT ?",
            params,
        ).fetchall()

        return [str(row["conversation_id"]) for row in rows]

    # --- Metadata CRUD ---

    def get_metadata(self, conversation_id: str) -> dict[str, object]:
        """Get metadata dict for a conversation."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT metadata FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()

        if row is None:
            return {}
        return _parse_json(row["metadata"], field="metadata", record_id=conversation_id) or {}

    def _metadata_read_modify_write(
        self, conversation_id: str, mutator: Callable[[dict[str, object]], bool]
    ) -> None:
        """Atomically read-modify-write conversation metadata.

        Uses BEGIN IMMEDIATE to prevent concurrent writers from
        interleaving between the read and write (TOCTOU prevention).
        Falls back to SAVEPOINT when already inside a transaction.

        Args:
            conversation_id: Target conversation
            mutator: Receives current metadata dict, mutates it in place,
                     returns True if a write is needed
        """
        conn = self._get_connection()

        if conn.in_transaction:
            # Already in a transaction — use savepoint for nested atomicity
            conn.execute("SAVEPOINT metadata_rmw")
            try:
                current = self.get_metadata(conversation_id)
                if mutator(current):
                    conn.execute(
                        "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
                        (json_dumps(current), conversation_id),
                    )
                conn.execute("RELEASE SAVEPOINT metadata_rmw")
            except Exception:
                conn.execute("ROLLBACK TO SAVEPOINT metadata_rmw")
                raise
        else:
            # No active transaction — use BEGIN IMMEDIATE for exclusive lock
            conn.execute("BEGIN IMMEDIATE")
            try:
                current = self.get_metadata(conversation_id)
                if mutator(current):
                    conn.execute(
                        "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
                        (json_dumps(current), conversation_id),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def update_metadata(self, conversation_id: str, key: str, value: object) -> None:
        """Set a single metadata key."""

        def _set(meta: dict[str, object]) -> bool:
            meta[key] = value
            return True

        self._metadata_read_modify_write(conversation_id, _set)

    def delete_metadata(self, conversation_id: str, key: str) -> None:
        """Remove a metadata key."""

        def _delete(meta: dict[str, object]) -> bool:
            if key in meta:
                del meta[key]
                return True
            return False

        self._metadata_read_modify_write(conversation_id, _delete)

    def add_tag(self, conversation_id: str, tag: str) -> None:
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

        self._metadata_read_modify_write(conversation_id, _add)

    def remove_tag(self, conversation_id: str, tag: str) -> None:
        """Remove a tag from the conversation's tags list."""

        def _remove(meta: dict[str, object]) -> bool:
            tags = meta.get("tags", [])
            if isinstance(tags, list) and tag in tags:
                tags.remove(tag)
                meta["tags"] = tags
                return True
            return False

        self._metadata_read_modify_write(conversation_id, _remove)

    def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        """List all tags with counts, using json_each for efficiency.

        Args:
            provider: Optional provider filter.

        Returns:
            Dict of tag → count, sorted by count descending.
        """
        conn = self._get_connection()
        where = "WHERE metadata IS NOT NULL AND json_extract(metadata, '$.tags') IS NOT NULL"
        params: tuple[str, ...] = ()
        if provider:
            where += " AND provider_name = ?"
            params = (provider,)
        rows = conn.execute(
            f"""
            SELECT tag.value AS tag_name, COUNT(*) AS cnt
            FROM conversations,
                 json_each(json_extract(metadata, '$.tags')) AS tag
            {where}
            GROUP BY tag.value
            ORDER BY cnt DESC
            """,
            params,
        ).fetchall()
        return {row["tag_name"]: row["cnt"] for row in rows}

    def set_metadata(self, conversation_id: str, metadata: dict[str, object]) -> None:
        """Replace entire metadata dict."""
        conn = self._get_connection()
        conn.execute(
            "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
            (json_dumps(metadata), conversation_id),
        )
        conn.commit()

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and all related records.

        Removes conversation, messages, attachment refs, and FTS index entries.
        Does NOT delete attachments themselves (handled by ref counting).

        Args:
            conversation_id: ID of the conversation to delete

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()

        # Check if exists
        exists = conn.execute(
            "SELECT 1 FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()

        if not exists:
            return False

        # FTS cleanup is handled automatically by the messages_fts_delete trigger
        # (added in schema v11) when CASCADE deletes the messages.
        try:
            # Collect attachment IDs that may become orphaned after CASCADE
            affected_attachments = [
                row[0]
                for row in conn.execute(
                    """SELECT DISTINCT ar.attachment_id FROM attachment_refs ar
                       JOIN messages m ON ar.message_id = m.message_id
                       WHERE m.conversation_id = ?""",
                    (conversation_id,),
                ).fetchall()
            ]

            # Delete conversation (CASCADE handles messages, attachment_refs, + FTS)
            conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )

            # Recalculate ref_count for affected attachments and delete orphans
            if affected_attachments:
                placeholders = ",".join("?" * len(affected_attachments))
                conn.execute(
                    f"""UPDATE attachments SET ref_count = (
                            SELECT COUNT(*) FROM attachment_refs
                            WHERE attachment_refs.attachment_id = attachments.attachment_id
                        ) WHERE attachment_id IN ({placeholders})""",
                    affected_attachments,
                )
                conn.execute(
                    f"DELETE FROM attachments WHERE attachment_id IN ({placeholders}) AND ref_count <= 0",
                    affected_attachments,
                )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return True

    def iter_messages(
        self,
        conversation_id: str,
        *,
        chunk_size: int = 100,
        dialogue_only: bool = False,
        limit: int | None = None,
    ) -> Iterator[MessageRecord]:
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
        conn = self._get_connection()
        offset = 0
        yielded = 0

        while True:
            # Build query with optional role filter
            query = "SELECT * FROM messages WHERE conversation_id = ?"
            params: list[str | int] = [conversation_id]

            if dialogue_only:
                query += " AND role IN ('user', 'assistant', 'human')"

            query += " ORDER BY timestamp"

            # Calculate how many to fetch this round
            fetch_limit = chunk_size
            if limit is not None:
                remaining = limit - yielded
                if remaining <= 0:
                    break
                fetch_limit = min(chunk_size, remaining)

            query += " LIMIT ? OFFSET ?"
            params.extend([fetch_limit, offset])

            rows = conn.execute(query, tuple(params)).fetchall()
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

    def get_conversation_stats(self, conversation_id: str) -> dict[str, int]:
        """Get message counts without loading messages.

        Useful for UI display and deciding whether to use streaming.

        Args:
            conversation_id: ID of the conversation

        Returns:
            Dict with counts: total_messages, dialogue_messages, tool_messages
        """
        conn = self._get_connection()

        # Total messages
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()["cnt"]

        # Dialogue messages (user + assistant)
        dialogue = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ? AND role IN ('user', 'assistant', 'human')",
            (conversation_id,),
        ).fetchone()["cnt"]

        return {
            "total_messages": total,
            "dialogue_messages": dialogue,
            "tool_messages": total - dialogue,
        }

    def get_message_counts_batch(self, conversation_ids: list[str]) -> dict[str, int]:
        """Get message counts for multiple conversations in a single query.

        Args:
            conversation_ids: List of conversation IDs

        Returns:
            Dict mapping conversation_id to message count
        """
        if not conversation_ids:
            return {}

        conn = self._get_connection()
        placeholders = ",".join("?" for _ in conversation_ids)
        rows = conn.execute(
            f"""
            SELECT conversation_id, COUNT(*) as cnt
            FROM messages
            WHERE conversation_id IN ({placeholders})
            GROUP BY conversation_id
            """,
            conversation_ids,
        ).fetchall()

        return {row["conversation_id"]: row["cnt"] for row in rows}

    def close(self) -> None:
        """Close the database connection for this thread."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
            self._local.transaction_depth = 0

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Context manager for transactions.

        Example:
            with backend.transaction():
                backend.save_conversation(conv)
                backend.save_messages(messages)
        """
        self.begin()
        try:
            yield
            self.commit()
        except Exception:
            self.rollback()
            raise

    # --- Raw Conversation Storage ---

    def save_raw_conversation(self, record: RawConversationRecord) -> bool:
        """Save a raw conversation record.

        Uses INSERT OR IGNORE to avoid duplicates (raw_id is SHA256 of content).

        Args:
            record: Raw conversation record to save

        Returns:
            True if inserted, False if already exists
        """
        conn = self._get_connection()
        result = conn.execute(
            """
            INSERT OR IGNORE INTO raw_conversations (
                raw_id,
                provider_name,
                source_name,
                source_path,
                source_index,
                raw_content,
                acquired_at,
                file_mtime
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.raw_id,
                record.provider_name,
                record.source_name,
                record.source_path,
                record.source_index,
                record.raw_content,
                record.acquired_at,
                record.file_mtime,
            ),
        )
        return bool(result.rowcount > 0)

    def get_raw_conversation(self, raw_id: str) -> RawConversationRecord | None:
        """Retrieve a raw conversation by ID.

        Args:
            raw_id: SHA256 hash of the raw content

        Returns:
            RawConversationRecord if found, None otherwise
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM raw_conversations WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()

        if row is None:
            return None

        return _row_to_raw_conversation(row)

    def iter_raw_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
    ) -> Iterator[RawConversationRecord]:
        """Iterate over raw conversation records.

        Args:
            provider: Optional provider name to filter by
            limit: Optional maximum number of records to return

        Yields:
            RawConversationRecord objects
        """
        conn = self._get_connection()

        query = "SELECT * FROM raw_conversations"
        params: list[str | int] = []

        if provider is not None:
            query += " WHERE provider_name = ?"
            params.append(provider)

        query += " ORDER BY acquired_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(query, tuple(params)).fetchall()

        for row in rows:
            yield _row_to_raw_conversation(row)

    def get_raw_conversation_count(self, provider: str | None = None) -> int:
        """Get count of raw conversations.

        Args:
            provider: Optional provider name to filter by

        Returns:
            Count of raw conversation records
        """
        conn = self._get_connection()
        query = "SELECT COUNT(*) as cnt FROM raw_conversations"
        params: tuple[str, ...] = ()
        if provider is not None:
            query += " WHERE provider_name = ?"
            params = (provider,)
        row = conn.execute(query, params).fetchone()
        return int(row["cnt"])



__all__ = [
    "SQLiteBackend",
    "DatabaseError",
    "default_db_path",
    "connection_context",
    "open_connection",
    "create_default_backend",
    "SCHEMA_VERSION",
    "SCHEMA_DDL",
    "_VEC0_DDL",
    "_apply_schema",
    "_MIGRATIONS",
    "_run_migrations",
    "_ensure_schema",
    "_ensure_vec0_table",
    "_load_sqlite_vec",
]
