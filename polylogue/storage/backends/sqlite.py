"""SQLite storage backend implementation."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from polylogue.lib.json import dumps as json_dumps
from polylogue.lib.log import get_logger
from polylogue.paths import DATA_HOME
from polylogue.storage.store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    RawConversationRecord,
    RunRecord,
)
from polylogue.types import ConversationId


class DatabaseError(Exception):
    """Base class for database errors."""

    pass


@contextmanager
def connection_context(db_path: Path | str | sqlite3.Connection | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager for managing sqlite3 connections.

    Args:
        db_path: Path to the database file, or an existing connection.
                 If None, uses default path.

    Yields:
        An open sqlite3.Connection with Row factory and WAL mode enabled.
        sqlite-vec extension is loaded if available.
    """
    if isinstance(db_path, sqlite3.Connection):
        _load_sqlite_vec(db_path)
        yield db_path
        return

    path = Path(db_path) if db_path else default_db_path()
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds for lock contention
        # Load sqlite-vec extension if available (optional for vector search)
        _load_sqlite_vec(conn)
        # Ensure schema exists and is at current version
        _ensure_schema(conn)
        yield conn
    finally:
        conn.close()


# Alias for backward compatibility
open_connection = connection_context


LOGGER = get_logger(__name__)
SCHEMA_VERSION = 11


def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Attempt to load sqlite-vec extension.

    Returns True if loaded successfully, False otherwise.
    The extension is optional - vector search is simply unavailable without it.
    Silent on failure since this is called on every connection.

    Note: enable_load_extension(True) is required before loading native SQLite
    extensions. We re-disable it after loading for security (prevents untrusted
    SQL from loading arbitrary extensions).
    """
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
            return True
        finally:
            conn.enable_load_extension(False)
    except ImportError:
        return False
    except Exception as exc:
        LOGGER.warning("sqlite-vec extension load failed: %s", exc)
        return False


def create_default_backend() -> SQLiteBackend:
    """Create a SQLiteBackend with the default database path.

    This is a convenience function for creating backends when
    no custom path is needed.

    Returns:
        SQLiteBackend connected to the default database location
    """
    return SQLiteBackend(db_path=None)


def default_db_path() -> Path:
    """Return the default database path.

    Uses XDG_DATA_HOME/polylogue/polylogue.db (semantic data, not ephemeral state).
    """
    return DATA_HOME / "polylogue.db"


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
        CREATE TABLE IF NOT EXISTS raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            raw_content BLOB NOT NULL,
            acquired_at TEXT NOT NULL,
            file_mtime TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_raw_conv_provider
        ON raw_conversations(provider_name);

        CREATE INDEX IF NOT EXISTS idx_raw_conv_source
        ON raw_conversations(source_path);

        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            metadata TEXT DEFAULT '{}',
            source_name TEXT GENERATED ALWAYS AS (json_extract(provider_meta, '$.source')) STORED,
            version INTEGER NOT NULL,
            parent_conversation_id TEXT REFERENCES conversations(conversation_id),
            branch_type TEXT CHECK (branch_type IN ('continuation', 'sidechain', 'fork') OR branch_type IS NULL),
            raw_id TEXT REFERENCES raw_conversations(raw_id)
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_provider
        ON conversations(provider_name, provider_conversation_id);

        CREATE INDEX IF NOT EXISTS idx_conversations_source_name
        ON conversations(source_name) WHERE source_name IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_conversations_parent
        ON conversations(parent_conversation_id) WHERE parent_conversation_id IS NOT NULL;

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
            parent_message_id TEXT,
            branch_index INTEGER DEFAULT 0,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(conversation_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages(conversation_id);

        CREATE INDEX IF NOT EXISTS idx_messages_parent
        ON messages(parent_message_id) WHERE parent_message_id IS NOT NULL;

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

        CREATE INDEX IF NOT EXISTS idx_attachment_refs_attachment
        ON attachment_refs(attachment_id);

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            plan_snapshot TEXT,
            counts_json TEXT,
            drift_json TEXT,
            indexed INTEGER,
            duration_ms INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_raw_id
        ON conversations(raw_id) WHERE raw_id IS NOT NULL;

        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            message_id UNINDEXED,
            conversation_id UNINDEXED,
            content
        );

        CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages
        BEGIN
            INSERT OR IGNORE INTO messages_fts(rowid, message_id, conversation_id, content)
            VALUES (new.rowid, new.message_id, new.conversation_id, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE OF text ON messages
        BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
            INSERT INTO messages_fts(rowid, message_id, conversation_id, content)
            VALUES (new.rowid, new.message_id, new.conversation_id, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages
        BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
        END;

        -- v10: Embedding metadata table (always created)
        CREATE TABLE IF NOT EXISTS embeddings_meta (
            target_id TEXT PRIMARY KEY,
            target_type TEXT NOT NULL CHECK (target_type IN ('message', 'conversation')),
            model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            embedded_at TEXT NOT NULL,
            content_hash TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_embeddings_meta_type
        ON embeddings_meta(target_type);

        -- v10: Embedding status tracking per conversation
        CREATE TABLE IF NOT EXISTS embedding_status (
            conversation_id TEXT PRIMARY KEY REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            message_count_embedded INTEGER DEFAULT 0,
            last_embedded_at TEXT,
            needs_reindex INTEGER DEFAULT 1,
            error_message TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_embedding_status_needs
        ON embedding_status(needs_reindex) WHERE needs_reindex = 1;
        """
    )

    # v10: Create vec0 table if sqlite-vec is available
    vec_available = False
    try:
        conn.execute("SELECT vec_version()")
        vec_available = True
    except sqlite3.OperationalError:
        pass

    if vec_available:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
                message_id TEXT PRIMARY KEY,
                embedding float[1024],
                +provider_name TEXT,
                +conversation_id TEXT
            )
        """)

    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Migrate from v1 to v2: add attachment reference counting."""
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


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Migrate from v2 to v3: update runs table schema."""
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


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    """Migrate from v3 to v4: add computed source_name column."""
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


def _migrate_v4_to_v5(conn: sqlite3.Connection) -> None:
    """Migrate from v4 to v5: add metadata column for user-editable fields."""
    # Simple column addition - SQLite supports ADD COLUMN
    conn.execute("ALTER TABLE conversations ADD COLUMN metadata TEXT DEFAULT '{}'")


def _migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    """Migrate from v5 to v6: add index on attachment_refs(attachment_id)."""
    conn.execute("CREATE INDEX IF NOT EXISTS idx_attachment_refs_attachment ON attachment_refs(attachment_id)")


def _migrate_v6_to_v7(conn: sqlite3.Connection) -> None:
    """Migrate from v6 to v7: add session/message branching support.

    Adds to conversations:
    - parent_conversation_id: links to parent session (for continuations/sidechains)
    - branch_type: 'continuation', 'sidechain', or 'fork'

    Adds to messages:
    - parent_message_id: links to parent message (for edit branches)
    - branch_index: sibling position (0 = mainline, >0 = branch)
    """
    # Add conversation branching columns
    conn.execute(
        "ALTER TABLE conversations ADD COLUMN parent_conversation_id TEXT REFERENCES conversations(conversation_id)"
    )
    conn.execute(
        "ALTER TABLE conversations ADD COLUMN branch_type TEXT CHECK (branch_type IN ('continuation', 'sidechain', 'fork') OR branch_type IS NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_parent ON conversations(parent_conversation_id) WHERE parent_conversation_id IS NOT NULL"
    )

    # Add message branching columns
    conn.execute("ALTER TABLE messages ADD COLUMN parent_message_id TEXT")
    conn.execute("ALTER TABLE messages ADD COLUMN branch_index INTEGER DEFAULT 0")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_message_id) WHERE parent_message_id IS NOT NULL"
    )


def _migrate_v7_to_v8(conn: sqlite3.Connection) -> None:
    """Migrate from v7 to v8: add raw storage with proper FK direction.

    Creates raw_conversations table for storing original JSON/JSONL bytes,
    and adds raw_id column to conversations (FK points FROM conversations TO raw).

    Data flow: raw_conversations → conversations (not the reverse).
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            raw_content BLOB NOT NULL,
            acquired_at TEXT NOT NULL,
            file_mtime TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_raw_conv_provider
        ON raw_conversations(provider_name);

        CREATE INDEX IF NOT EXISTS idx_raw_conv_source
        ON raw_conversations(source_path);
        """
    )

    # Add raw_id to conversations (FK to raw_conversations)
    conn.execute("ALTER TABLE conversations ADD COLUMN raw_id TEXT REFERENCES raw_conversations(raw_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_raw_id ON conversations(raw_id) WHERE raw_id IS NOT NULL"
    )


def _migrate_v8_to_v9(conn: sqlite3.Connection) -> None:
    """Migrate from v8 to v9: add source_name to raw_conversations.

    The source_name field stores the config source name (e.g., "inbox")
    separately from provider_name (e.g., "chatgpt") and source_path (file path).
    """
    # Check if column already exists (idempotent migration)
    cursor = conn.execute("PRAGMA table_info(raw_conversations)")
    columns = {row[1] for row in cursor.fetchall()}
    if "source_name" not in columns:
        conn.execute("ALTER TABLE raw_conversations ADD COLUMN source_name TEXT")


def _migrate_v9_to_v10(conn: sqlite3.Connection) -> None:
    """Migrate from v9 to v10: add sqlite-vec vector storage tables.

    Creates:
    - embeddings_meta: Tracks embedding provenance (model, dimension, timestamps)
    - message_embeddings: vec0 virtual table for message-level embeddings (if sqlite-vec available)

    The vec0 tables are only created if sqlite-vec extension is loaded.
    Without the extension, the metadata table is still created but vector
    search will be unavailable.
    """
    # Metadata table for embedding provenance (always created)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings_meta (
            target_id TEXT PRIMARY KEY,
            target_type TEXT NOT NULL CHECK (target_type IN ('message', 'conversation')),
            model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            embedded_at TEXT NOT NULL,
            content_hash TEXT
        )
    """)

    # Check if sqlite-vec is available by trying to create a vec0 table
    vec_available = False
    try:
        # Test if vec0 module is available
        conn.execute("SELECT vec_version()")
        vec_available = True
    except sqlite3.OperationalError:
        LOGGER.info("sqlite-vec not available, skipping vec0 table creation")

    if vec_available:
        # vec0 virtual table for message embeddings
        # Using 1024 dimensions as default for voyage-4
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
                message_id TEXT PRIMARY KEY,
                embedding float[1024],
                +provider_name TEXT,
                +conversation_id TEXT
            )
        """)

        # Index on conversation_id for filtering
        # Note: vec0 tables handle their own indexing, but we add metadata for filtering
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_meta_type
            ON embeddings_meta(target_type)
        """)

    # Embedding status tracking per conversation
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_status (
            conversation_id TEXT PRIMARY KEY REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            message_count_embedded INTEGER DEFAULT 0,
            last_embedded_at TEXT,
            needs_reindex INTEGER DEFAULT 1,
            error_message TEXT
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_embedding_status_needs
        ON embedding_status(needs_reindex) WHERE needs_reindex = 1
    """)


def _migrate_v10_to_v11(conn: sqlite3.Connection) -> None:
    """Migrate from v10 to v11: add FTS UPDATE/DELETE triggers.

    Previously only INSERT trigger existed, so edited or deleted messages
    remained searchable as ghost results.
    """
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE OF text ON messages
        BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
            INSERT INTO messages_fts(rowid, message_id, conversation_id, content)
            VALUES (new.rowid, new.message_id, new.conversation_id, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages
        BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
        END;
    """)


# Migration registry: maps source version to migration function
_MIGRATIONS = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
    3: _migrate_v3_to_v4,
    4: _migrate_v4_to_v5,
    5: _migrate_v5_to_v6,
    6: _migrate_v6_to_v7,
    7: _migrate_v7_to_v8,
    8: _migrate_v8_to_v9,
    9: _migrate_v9_to_v10,
    10: _migrate_v10_to_v11,
}


def _run_migrations(conn: sqlite3.Connection, current_version: int, target_version: int) -> None:
    """Run migrations from current_version to target_version.

    Each migration step is committed individually (ratcheting), ensuring that
    even if the full sequence fails, the database remains in a valid intermediate
    state (the last successful version).

    Args:
        conn: Database connection
        current_version: Starting schema version
        target_version: Target schema version

    Raises:
        RuntimeError: If any migration fails.
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
            conn.commit()  # Commit each step successfully
            LOGGER.info("Migration v%d -> v%d completed", version, version + 1)
        except Exception as exc:
            LOGGER.error("Migration v%d -> v%d failed: %s", version, version + 1, exc)
            conn.rollback()  # Rollback changes from the failed step
            raise RuntimeError(
                f"Migration from v{version} to v{version + 1} failed. Database remains at v{version}. Error: {exc}"
            ) from exc


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure database schema is at current version, running migrations if needed.

    For fresh databases (version 0), creates the schema directly.
    For existing databases, runs migrations sequentially.

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
        # To perform schema changes (especially recreations), we disable FKs.
        # This requires no active transaction.
        if conn.in_transaction:
            conn.commit()

        # Disable foreign keys globally for the migration process
        conn.execute("PRAGMA foreign_keys = OFF")

        try:
            _run_migrations(conn, current_version, SCHEMA_VERSION)
        finally:
            # Always re-enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
        return

    if current_version != SCHEMA_VERSION:
        raise DatabaseError(f"Unsupported DB schema version {current_version} (expected {SCHEMA_VERSION})")

    # Ensure vec0 table exists if sqlite-vec is now available.
    # This handles databases created when sqlite-vec couldn't load (e.g., before
    # the enable_load_extension fix). The v10 migration would have skipped vec0
    # creation, so we create it retroactively.
    _ensure_vec0_table(conn)


def _ensure_vec0_table(conn: sqlite3.Connection) -> None:
    """Create vec0 table if sqlite-vec is available but table is missing.

    This is idempotent and handles the case where the v10 migration ran
    without sqlite-vec available (e.g., due to missing enable_load_extension).
    """
    try:
        conn.execute("SELECT vec_version()")
    except sqlite3.OperationalError:
        return  # sqlite-vec not available, nothing to do

    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='message_embeddings'"
    ).fetchone()
    if not exists:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
                message_id TEXT PRIMARY KEY,
                embedding float[1024],
                +provider_name TEXT,
                +conversation_id TEXT
            )
        """)
        conn.commit()
        LOGGER.info("Created missing message_embeddings vec0 table")


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
        """Begin a transaction or nested savepoint."""
        conn = self._get_connection()
        if self._local.transaction_depth == 0:
            conn.execute("BEGIN")
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
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            version=row["version"],
            parent_conversation_id=row["parent_conversation_id"],
            branch_type=row["branch_type"],
            raw_id=row["raw_id"],
        )

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

        import json

        conn = self._get_connection()
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT * FROM conversations WHERE conversation_id IN ({placeholders})",
            ids,
        ).fetchall()

        # Build lookup for order preservation
        by_id = {}
        for row in rows:
            by_id[row["conversation_id"]] = ConversationRecord(
                conversation_id=row["conversation_id"],
                provider_name=row["provider_name"],
                provider_conversation_id=row["provider_conversation_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                content_hash=row["content_hash"],
                provider_meta=json.loads(row["provider_meta"]) if row["provider_meta"] else None,
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                version=row["version"],
                parent_conversation_id=row["parent_conversation_id"],
                branch_type=row["branch_type"],
                raw_id=row["raw_id"],
            )

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
        where_clauses = []
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
            where_clauses.append("title LIKE ?")
            params.append(f"%{title_contains}%")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

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

        if offset > 0:
            query += " OFFSET ?"
            params.append(offset)

        rows = conn.execute(query, tuple(params)).fetchall()

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
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                version=row["version"],
                parent_conversation_id=row["parent_conversation_id"],
                branch_type=row["branch_type"],
                raw_id=row["raw_id"],
            )
            for row in rows
        ]

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
        where_clauses = []
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
        if since is not None:
            where_clauses.append("updated_at >= ?")
            params.append(since)
        if until is not None:
            where_clauses.append("updated_at <= ?")
            params.append(until)
        if title_contains is not None:
            where_clauses.append("title LIKE ?")
            params.append(f"%{title_contains}%")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
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
                parent_message_id=row["parent_message_id"],
                branch_index=row["branch_index"] or 0,
            )
            for row in rows
        ]

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

        # Batch update ref counts for affected attachments
        for aid in attachment_ids_to_check:
            conn.execute(
                """
                UPDATE attachments
                SET ref_count = (SELECT COUNT(*) FROM attachment_refs WHERE attachment_id = ?)
                WHERE attachment_id = ?
                """,
                (aid, aid),
            )

        # Clean up orphaned attachments (ref_count <= 0)
        conn.execute("DELETE FROM attachments WHERE ref_count <= 0")

    def save_attachments(self, records: list[AttachmentRecord]) -> None:
        """Persist attachment records with reference counting using bulk operations."""
        if not records:
            return
        conn = self._get_connection()

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
        attachment_ids = {r.attachment_id for r in records}
        for aid in attachment_ids:
            conn.execute(
                """
                UPDATE attachments
                SET ref_count = (SELECT COUNT(*) FROM attachment_refs WHERE attachment_id = ?)
                WHERE attachment_id = ?
                """,
                (aid, aid),
            )

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
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                version=row["version"],
                parent_conversation_id=row["parent_conversation_id"],
                branch_type=row["branch_type"],
                raw_id=row["raw_id"],
            )
            for row in rows
        ]

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
            rows = conn.execute(
                f"""
                SELECT DISTINCT messages_fts.conversation_id
                FROM messages_fts
                JOIN conversations ON conversations.conversation_id = messages_fts.conversation_id
                WHERE messages_fts MATCH ?
                  AND (conversations.provider_name IN ({placeholders})
                       OR conversations.source_name IN ({placeholders}))
                LIMIT ?
                """,
                (fts_query, *providers, *providers, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT DISTINCT conversation_id
                FROM messages_fts
                WHERE messages_fts MATCH ?
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()

        return [str(row["conversation_id"]) for row in rows]

    # --- Metadata CRUD ---

    def get_metadata(self, conversation_id: str) -> dict[str, object]:
        """Get metadata dict for a conversation."""
        import json

        conn = self._get_connection()
        row = conn.execute(
            "SELECT metadata FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()

        if row is None:
            return {}
        return json.loads(row["metadata"]) if row["metadata"] else {}

    def update_metadata(self, conversation_id: str, key: str, value: object) -> None:
        """Set a single metadata key."""
        import json

        conn = self._get_connection()
        current = self.get_metadata(conversation_id)
        current[key] = value
        conn.execute(
            "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
            (json.dumps(current), conversation_id),
        )
        conn.commit()

    def delete_metadata(self, conversation_id: str, key: str) -> None:
        """Remove a metadata key."""
        import json

        conn = self._get_connection()
        current = self.get_metadata(conversation_id)
        if key in current:
            del current[key]
            conn.execute(
                "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
                (json.dumps(current), conversation_id),
            )
            conn.commit()

    def add_tag(self, conversation_id: str, tag: str) -> None:
        """Add a tag to the conversation's tags list."""
        import json

        conn = self._get_connection()
        current = self.get_metadata(conversation_id)
        tags = current.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        if tag not in tags:
            tags.append(tag)
            current["tags"] = tags
            conn.execute(
                "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
                (json.dumps(current), conversation_id),
            )
            conn.commit()

    def remove_tag(self, conversation_id: str, tag: str) -> None:
        """Remove a tag from the conversation's tags list."""
        import json

        conn = self._get_connection()
        current = self.get_metadata(conversation_id)
        tags = current.get("tags", [])
        if isinstance(tags, list) and tag in tags:
            tags.remove(tag)
            current["tags"] = tags
            conn.execute(
                "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
                (json.dumps(current), conversation_id),
            )
            conn.commit()

    def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        """List all tags with counts, using json_each for efficiency.

        Args:
            provider: Optional provider filter.

        Returns:
            Dict of tag → count, sorted by count descending.
        """
        conn = self._get_connection()
        if provider:
            rows = conn.execute(
                """
                SELECT tag.value AS tag_name, COUNT(*) AS cnt
                FROM conversations,
                     json_each(json_extract(metadata, '$.tags')) AS tag
                WHERE metadata IS NOT NULL
                  AND json_extract(metadata, '$.tags') IS NOT NULL
                  AND provider_name = ?
                GROUP BY tag.value
                ORDER BY cnt DESC
                """,
                (provider,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT tag.value AS tag_name, COUNT(*) AS cnt
                FROM conversations,
                     json_each(json_extract(metadata, '$.tags')) AS tag
                WHERE metadata IS NOT NULL
                  AND json_extract(metadata, '$.tags') IS NOT NULL
                GROUP BY tag.value
                ORDER BY cnt DESC
                """,
            ).fetchall()
        return {row["tag_name"]: row["cnt"] for row in rows}

    def set_metadata(self, conversation_id: str, metadata: dict[str, object]) -> None:
        """Replace entire metadata dict."""
        import json

        conn = self._get_connection()
        conn.execute(
            "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
            (json.dumps(metadata), conversation_id),
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
            # Delete conversation (CASCADE handles messages + FTS automatically)
            conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )

            # Clean up orphaned attachments (ref_count <= 0)
            conn.execute("DELETE FROM attachments WHERE ref_count <= 0")

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
        import json

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
                yield MessageRecord(
                    message_id=row["message_id"],
                    conversation_id=row["conversation_id"],
                    provider_message_id=row["provider_message_id"],
                    role=row["role"],
                    text=row["text"],
                    timestamp=row["timestamp"],
                    content_hash=row["content_hash"],
                    provider_meta=json.loads(row["provider_meta"]) if row["provider_meta"] else None,
                    version=row["version"],
                    parent_message_id=row["parent_message_id"],
                    branch_index=row["branch_index"] or 0,
                )
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
            yield RawConversationRecord(
                raw_id=row["raw_id"],
                provider_name=row["provider_name"],
                source_name=row["source_name"],
                source_path=row["source_path"],
                source_index=row["source_index"],
                raw_content=row["raw_content"],
                acquired_at=row["acquired_at"],
                file_mtime=row["file_mtime"],
            )

    def get_raw_conversation_count(self, provider: str | None = None) -> int:
        """Get count of raw conversations.

        Args:
            provider: Optional provider name to filter by

        Returns:
            Count of raw conversation records
        """
        conn = self._get_connection()

        if provider is not None:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM raw_conversations WHERE provider_name = ?",
                (provider,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) as cnt FROM raw_conversations").fetchone()

        return int(row["cnt"])


__all__ = ["SQLiteBackend", "DatabaseError", "default_db_path"]
