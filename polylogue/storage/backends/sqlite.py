"""SQLite storage backend implementation."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from polylogue.core.json import dumps as json_dumps
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord
from polylogue.types import ConversationId

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 4


class DatabaseError(Exception):
    """Database-related errors (schema, connection, migration)."""


def default_db_path() -> Path:
    import os

    raw_state_root = os.environ.get("XDG_STATE_HOME")
    state_root = Path(raw_state_root).expanduser() if raw_state_root else Path.home() / ".local/state"
    return state_root / "polylogue" / "polylogue.db"


def _json_or_none(value: dict[str, object] | None) -> str | None:
    if value is None:
        return None
    return json_dumps(value)


def _make_ref_id(attachment_id: str, conversation_id: str, message_id: str | None) -> str:
    seed = f"{attachment_id}:{conversation_id}:{message_id or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"ref-{digest}"


def _apply_schema(conn: sqlite3.Connection) -> None:
    """Apply fresh schema at version SCHEMA_VERSION."""
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            source_name TEXT GENERATED ALWAYS AS (json_extract(provider_meta, '$.source')) STORED,
            version INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_provider
        ON conversations(provider_name, provider_conversation_id);

        CREATE INDEX IF NOT EXISTS idx_conversations_source_name
        ON conversations(source_name) WHERE source_name IS NOT NULL;

        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            provider_message_id TEXT,
            role TEXT,
            text TEXT,
            timestamp TEXT,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            version INTEGER NOT NULL,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(conversation_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages(conversation_id);

        CREATE TABLE IF NOT EXISTS attachments (
            attachment_id TEXT PRIMARY KEY,
            mime_type TEXT,
            size_bytes INTEGER,
            path TEXT,
            ref_count INTEGER NOT NULL DEFAULT 0,
            provider_meta TEXT,
            UNIQUE (attachment_id)
        );

        CREATE TABLE IF NOT EXISTS attachment_refs (
            ref_id TEXT PRIMARY KEY,
            attachment_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            message_id TEXT,
            provider_meta TEXT,
            FOREIGN KEY (attachment_id)
                REFERENCES attachments(attachment_id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            FOREIGN KEY (message_id)
                REFERENCES messages(message_id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_attachment_refs_conversation
        ON attachment_refs(conversation_id);

        CREATE INDEX IF NOT EXISTS idx_attachment_refs_message
        ON attachment_refs(message_id);

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            plan_snapshot TEXT,
            counts_json TEXT,
            drift_json TEXT,
            indexed INTEGER,
            duration_ms INTEGER
        );
        """
    )
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Migrate from v1 to v2: add attachment reference counting."""
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("ALTER TABLE attachments RENAME TO attachment_refs_old")
    conn.executescript(
        """
        CREATE TABLE attachments (
            attachment_id TEXT PRIMARY KEY,
            mime_type TEXT,
            size_bytes INTEGER,
            path TEXT,
            ref_count INTEGER NOT NULL DEFAULT 0,
            provider_meta TEXT,
            UNIQUE (attachment_id)
        );

        CREATE TABLE attachment_refs (
            ref_id TEXT PRIMARY KEY,
            attachment_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            message_id TEXT,
            provider_meta TEXT,
            FOREIGN KEY (attachment_id)
                REFERENCES attachments(attachment_id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            FOREIGN KEY (message_id)
                REFERENCES messages(message_id) ON DELETE SET NULL
        );

        CREATE INDEX idx_attachment_refs_conversation
        ON attachment_refs(conversation_id);

        CREATE INDEX idx_attachment_refs_message
        ON attachment_refs(message_id);
        """
    )
    rows = conn.execute("SELECT * FROM attachment_refs_old").fetchall()
    for row in rows:
        attachment_id = row["attachment_id"]
        conn.execute(
            """
            INSERT OR IGNORE INTO attachments (
                attachment_id,
                mime_type,
                size_bytes,
                path,
                ref_count,
                provider_meta
            ) VALUES (?, ?, ?, ?, 0, ?)
            """,
            (
                attachment_id,
                row["mime_type"],
                row["size_bytes"],
                row["path"],
                row["provider_meta"],
            ),
        )
        ref_id = _make_ref_id(attachment_id, row["conversation_id"], row["message_id"])
        conn.execute(
            """
            INSERT OR IGNORE INTO attachment_refs (
                ref_id,
                attachment_id,
                conversation_id,
                message_id,
                provider_meta
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                ref_id,
                attachment_id,
                row["conversation_id"],
                row["message_id"],
                row["provider_meta"],
            ),
        )
    conn.execute(
        """
        UPDATE attachments
        SET ref_count = (
            SELECT COUNT(*)
            FROM attachment_refs
            WHERE attachment_refs.attachment_id = attachments.attachment_id
        )
        """
    )
    conn.execute("DROP TABLE attachment_refs_old")
    conn.execute("PRAGMA foreign_keys = ON")


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Migrate from v2 to v3: update runs table schema."""
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("ALTER TABLE runs RENAME TO runs_old")
    conn.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            plan_snapshot TEXT,
            counts_json TEXT,
            drift_json TEXT,
            indexed INTEGER,
            duration_ms INTEGER
        );
        """
    )
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
        )
        SELECT
            run_id,
            timestamp,
            plan_snapshot,
            counts_json,
            drift_json,
            indexed,
            duration_ms
        FROM runs_old
        """
    )
    conn.execute("DROP TABLE runs_old")
    conn.execute("PRAGMA foreign_keys = ON")


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    """Migrate from v3 to v4: add computed source_name column."""
    conn.execute("PRAGMA foreign_keys = OFF")
    # Drop existing index before renaming table to avoid conflicts
    conn.execute("DROP INDEX IF EXISTS idx_conversations_provider")
    conn.execute("ALTER TABLE conversations RENAME TO conversations_old")
    conn.executescript(
        """
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            source_name TEXT GENERATED ALWAYS AS (json_extract(provider_meta, '$.source')) STORED,
            version INTEGER NOT NULL
        );

        CREATE INDEX idx_conversations_provider
        ON conversations(provider_name, provider_conversation_id);

        CREATE INDEX idx_conversations_source_name
        ON conversations(source_name) WHERE source_name IS NOT NULL;
        """
    )
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
            version
        )
        SELECT
            conversation_id,
            provider_name,
            provider_conversation_id,
            title,
            created_at,
            updated_at,
            content_hash,
            provider_meta,
            version
        FROM conversations_old
        """
    )
    conn.execute("DROP TABLE conversations_old")
    conn.execute("PRAGMA foreign_keys = ON")


# Migration registry: maps source version to migration function
_MIGRATIONS = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
    3: _migrate_v3_to_v4,
}


def _run_migrations(conn: sqlite3.Connection, current_version: int, target_version: int) -> None:
    """Run migrations from current_version to target_version.

    Note: SQLite DDL statements (ALTER TABLE, CREATE TABLE, etc.) auto-commit
    and cannot be rolled back. If a migration fails mid-way, the database may
    be in an inconsistent state. Always backup before migrations.

    Each successful migration updates the schema version before proceeding to
    the next, so partial progress is preserved.

    Args:
        conn: Database connection
        current_version: Starting schema version
        target_version: Target schema version

    Raises:
        RuntimeError: If any migration fails, with details about which migration
                     failed and at what version the database remains.
    """
    if current_version >= target_version:
        return

    for version in range(current_version, target_version):
        migration_func = _MIGRATIONS.get(version)
        if migration_func is None:
            continue

        LOGGER.info("Running migration v%d -> v%d", version, version + 1)

        try:
            migration_func(conn)
            conn.execute(f"PRAGMA user_version = {version + 1}")
            LOGGER.info("Migration v%d -> v%d completed", version, version + 1)
        except Exception as exc:
            LOGGER.error("Migration v%d -> v%d failed: %s", version, version + 1, exc)
            raise RuntimeError(
                f"Migration from v{version} to v{version + 1} failed. "
                f"Database may be in inconsistent state. Error: {exc}"
            ) from exc


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure database schema is at current version, running migrations if needed.

    For fresh databases (version 0), creates the schema directly.
    For existing databases, runs migrations sequentially with rollback support.

    Args:
        conn: Database connection

    Raises:
        DatabaseError: If schema version is unsupported or migration fails
    """
    row = conn.execute("PRAGMA user_version").fetchone()
    current_version = row[0] if row else 0

    if current_version == 0:
        # Fresh database - create schema directly
        _apply_schema(conn)
        return

    if current_version < SCHEMA_VERSION:
        # Run migrations with rollback support
        try:
            _run_migrations(conn, current_version, SCHEMA_VERSION)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return

    if current_version != SCHEMA_VERSION:
        raise DatabaseError(f"Unsupported DB schema version {current_version} (expected {SCHEMA_VERSION})")


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

        Args:
            db_path: Path to SQLite database file. If None, uses default path.
        """
        self._db_path = Path(db_path) if db_path is not None else default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._transaction_depth = 0

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, timeout=30)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA busy_timeout = 30000")
            _ensure_schema(self._conn)
        return self._conn

    def begin(self) -> None:
        """Begin a transaction or nested savepoint."""
        conn = self._get_connection()
        if self._transaction_depth == 0:
            conn.execute("BEGIN")
        else:
            conn.execute(f"SAVEPOINT sp_{self._transaction_depth}")
        self._transaction_depth += 1

    def commit(self) -> None:
        """Commit the current transaction or release savepoint."""
        if self._transaction_depth <= 0:
            raise DatabaseError("No active transaction to commit")

        conn = self._get_connection()
        self._transaction_depth -= 1

        if self._transaction_depth == 0:
            conn.commit()
        else:
            conn.execute(f"RELEASE SAVEPOINT sp_{self._transaction_depth}")

    def rollback(self) -> None:
        """Rollback to the last begin() or savepoint."""
        if self._transaction_depth <= 0:
            raise DatabaseError("No active transaction to rollback")

        conn = self._get_connection()
        self._transaction_depth -= 1

        if self._transaction_depth == 0:
            conn.rollback()
        else:
            conn.execute(f"ROLLBACK TO SAVEPOINT sp_{self._transaction_depth}")

    def get_conversation(self, id: str) -> ConversationRecord | None:
        """Retrieve a conversation by ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (id,),
        ).fetchone()

        if row is None:
            return None

        import json

        return ConversationRecord(
            conversation_id=row["conversation_id"],
            provider_name=row["provider_name"],
            provider_conversation_id=row["provider_conversation_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            content_hash=row["content_hash"],
            provider_meta=json.loads(row["provider_meta"]) if row["provider_meta"] else None,
            version=row["version"],
        )

    def list_conversations(self, source: str | None = None) -> list[ConversationRecord]:
        """List all conversations, optionally filtered by source."""
        conn = self._get_connection()

        if source is not None:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE source_name = ? ORDER BY created_at DESC",
                (source,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM conversations ORDER BY created_at DESC").fetchall()

        import json

        return [
            ConversationRecord(
                conversation_id=row["conversation_id"],
                provider_name=row["provider_name"],
                provider_conversation_id=row["provider_conversation_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                content_hash=row["content_hash"],
                provider_meta=json.loads(row["provider_meta"]) if row["provider_meta"] else None,
                version=row["version"],
            )
            for row in rows
        ]

    def save_conversation(self, record: ConversationRecord) -> None:
        """Persist a conversation record with upsert semantics."""
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
                version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                title = excluded.title,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                content_hash = excluded.content_hash,
                provider_meta = excluded.provider_meta
            WHERE
                content_hash != excluded.content_hash
                OR IFNULL(title, '') != IFNULL(excluded.title, '')
                OR IFNULL(created_at, '') != IFNULL(excluded.created_at, '')
                OR IFNULL(updated_at, '') != IFNULL(excluded.updated_at, '')
                OR IFNULL(provider_meta, '') != IFNULL(excluded.provider_meta, '')
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
                record.version,
            ),
        )

    def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        """Retrieve all messages for a conversation."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,),
        ).fetchall()

        import json

        return [
            MessageRecord(
                message_id=row["message_id"],
                conversation_id=row["conversation_id"],
                provider_message_id=row["provider_message_id"],
                role=row["role"],
                text=row["text"],
                timestamp=row["timestamp"],
                content_hash=row["content_hash"],
                provider_meta=json.loads(row["provider_meta"]) if row["provider_meta"] else None,
                version=row["version"],
            )
            for row in rows
        ]

    def save_messages(self, records: list[MessageRecord]) -> None:
        """Persist multiple message records."""
        conn = self._get_connection()
        for record in records:
            conn.execute(
                """
                INSERT INTO messages (
                    message_id,
                    conversation_id,
                    provider_message_id,
                    role,
                    text,
                    timestamp,
                    content_hash,
                    provider_meta,
                    version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    role = excluded.role,
                    text = excluded.text,
                    timestamp = excluded.timestamp,
                    content_hash = excluded.content_hash,
                    provider_meta = excluded.provider_meta
                WHERE
                    content_hash != excluded.content_hash
                    OR IFNULL(role, '') != IFNULL(excluded.role, '')
                    OR IFNULL(text, '') != IFNULL(excluded.text, '')
                    OR IFNULL(timestamp, '') != IFNULL(excluded.timestamp, '')
                    OR IFNULL(provider_meta, '') != IFNULL(excluded.provider_meta, '')
                """,
                (
                    record.message_id,
                    record.conversation_id,
                    record.provider_message_id,
                    record.role,
                    record.text,
                    record.timestamp,
                    record.content_hash,
                    _json_or_none(record.provider_meta),
                    record.version,
                ),
            )

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

        import json

        return [
            AttachmentRecord(
                attachment_id=row["attachment_id"],
                conversation_id=ConversationId(conversation_id),
                message_id=row["message_id"],
                mime_type=row["mime_type"],
                size_bytes=row["size_bytes"],
                path=row["path"],
                provider_meta=json.loads(row["provider_meta"]) if row["provider_meta"] else None,
            )
            for row in rows
        ]

    def save_attachments(self, records: list[AttachmentRecord]) -> None:
        """Persist attachment records with reference counting."""
        conn = self._get_connection()

        for record in records:
            # Ensure attachment metadata exists (idempotent, doesn't touch ref_count)
            conn.execute(
                """
                INSERT INTO attachments (
                    attachment_id,
                    mime_type,
                    size_bytes,
                    path,
                    ref_count,
                    provider_meta
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(attachment_id) DO UPDATE SET
                    mime_type = COALESCE(excluded.mime_type, attachments.mime_type),
                    size_bytes = COALESCE(excluded.size_bytes, attachments.size_bytes),
                    path = COALESCE(excluded.path, attachments.path),
                    provider_meta = COALESCE(excluded.provider_meta, attachments.provider_meta)
                """,
                (
                    record.attachment_id,
                    record.mime_type,
                    record.size_bytes,
                    record.path,
                    0,
                    _json_or_none(record.provider_meta),
                ),
            )

            # Atomically insert ref and increment count
            ref_id = _make_ref_id(record.attachment_id, record.conversation_id, record.message_id)
            res = conn.execute(
                """
                INSERT OR IGNORE INTO attachment_refs (
                    ref_id,
                    attachment_id,
                    conversation_id,
                    message_id,
                    provider_meta
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ref_id,
                    record.attachment_id,
                    record.conversation_id,
                    record.message_id,
                    _json_or_none(record.provider_meta),
                ),
            )

            # Only increment if we actually inserted a new ref
            if res.rowcount > 0:
                conn.execute(
                    "UPDATE attachments SET ref_count = ref_count + 1 WHERE attachment_id = ?",
                    (record.attachment_id,),
                )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._transaction_depth = 0

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


__all__ = ["SQLiteBackend", "DatabaseError", "default_db_path"]
