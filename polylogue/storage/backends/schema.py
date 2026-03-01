"""SQLite schema management: DDL, migrations, and version control."""

from __future__ import annotations

import sqlite3

from polylogue.lib.log import get_logger
from polylogue.storage.store import _make_ref_id

logger = get_logger(__name__)
SCHEMA_VERSION = 16


_VEC0_DDL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
        message_id TEXT PRIMARY KEY,
        embedding float[1024],
        +provider_name TEXT,
        +conversation_id TEXT
    )
"""


# Core DDL applied by SQLiteBackend during schema initialization.
SCHEMA_DDL = """
        CREATE TABLE IF NOT EXISTS raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            raw_content BLOB NOT NULL,
            acquired_at TEXT NOT NULL,
            file_mtime TEXT,
            parsed_at TEXT,
            parse_error TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_raw_conv_provider
        ON raw_conversations(provider_name);

        CREATE INDEX IF NOT EXISTS idx_raw_conv_source
        ON raw_conversations(source_path);

        CREATE INDEX IF NOT EXISTS idx_raw_conv_source_mtime
        ON raw_conversations(source_path, file_mtime);

        CREATE INDEX IF NOT EXISTS idx_raw_conv_unparsed
        ON raw_conversations(raw_id) WHERE parsed_at IS NULL;

        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            sort_key REAL,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            metadata TEXT DEFAULT '{}',
            source_name TEXT GENERATED ALWAYS AS (json_extract(provider_meta, '$.source')) STORED,
            version INTEGER NOT NULL,
            parent_conversation_id TEXT REFERENCES conversations(conversation_id),
            branch_type TEXT CHECK (branch_type IN ('continuation', 'sidechain', 'fork', 'subagent') OR branch_type IS NULL),
            raw_id TEXT REFERENCES raw_conversations(raw_id)
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_provider
        ON conversations(provider_name, provider_conversation_id);

        CREATE INDEX IF NOT EXISTS idx_conversations_source_name
        ON conversations(source_name) WHERE source_name IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_conversations_parent
        ON conversations(parent_conversation_id) WHERE parent_conversation_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_conversations_content_hash
        ON conversations(content_hash);

        CREATE INDEX IF NOT EXISTS idx_conversations_sortkey
        ON conversations(sort_key);

        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            provider_message_id TEXT,
            role TEXT,
            text TEXT,
            timestamp TEXT,
            sort_key REAL,
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

        CREATE INDEX IF NOT EXISTS idx_messages_conversation_sortkey
        ON messages(conversation_id, sort_key);

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


def _apply_schema(conn: sqlite3.Connection) -> None:
    """Apply fresh schema at version SCHEMA_VERSION."""
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_DDL)

    # v10: Create vec0 table if sqlite-vec is available
    vec_available = False
    try:
        conn.execute("SELECT vec_version()")
        vec_available = True
    except sqlite3.OperationalError:
        pass

    if vec_available:
        conn.execute(_VEC0_DDL)

    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Migrate from v1 to v2: add attachment reference counting."""
    conn.execute("ALTER TABLE attachments RENAME TO attachment_refs_old")
    conn.execute("""
        CREATE TABLE attachments (
            attachment_id TEXT PRIMARY KEY,
            mime_type TEXT,
            size_bytes INTEGER,
            path TEXT,
            ref_count INTEGER NOT NULL DEFAULT 0,
            provider_meta TEXT,
            UNIQUE (attachment_id)
        )
    """)
    conn.execute("""
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
        )
    """)
    conn.execute("CREATE INDEX idx_attachment_refs_conversation ON attachment_refs(conversation_id)")
    conn.execute("CREATE INDEX idx_attachment_refs_message ON attachment_refs(message_id)")
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
    conn.execute("""
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            plan_snapshot TEXT,
            counts_json TEXT,
            drift_json TEXT,
            indexed INTEGER,
            duration_ms INTEGER
        )
    """)
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
    conn.execute("""
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
        )
    """)
    conn.execute("CREATE INDEX idx_conversations_provider ON conversations(provider_name, provider_conversation_id)")
    conn.execute("CREATE INDEX idx_conversations_source_name ON conversations(source_name) WHERE source_name IS NOT NULL")
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            raw_content BLOB NOT NULL,
            acquired_at TEXT NOT NULL,
            file_mtime TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_conv_provider ON raw_conversations(provider_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_conv_source ON raw_conversations(source_path)")

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
        logger.info("sqlite-vec not available, skipping vec0 table creation")

    if vec_available:
        conn.execute(_VEC0_DDL)

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
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE OF text ON messages
        BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
            INSERT INTO messages_fts(rowid, message_id, conversation_id, content)
            VALUES (new.rowid, new.message_id, new.conversation_id, new.text);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages
        BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
        END
    """)


def _migrate_v11_to_v12(conn: sqlite3.Connection) -> None:
    """Migrate from v11 to v12: add parse tracking columns to raw_conversations.

    Adds:
    - parsed_at TEXT: timestamp when the record was successfully parsed
    - parse_error TEXT: error message from last failed parse attempt
    - idx_raw_conv_source_mtime: composite index for mtime-based acquisition skip

    Backfill: marks existing raw records that have linked conversations as already
    parsed, preventing the first post-migration run from re-parsing everything.
    """
    conn.execute("ALTER TABLE raw_conversations ADD COLUMN parsed_at TEXT")
    conn.execute("ALTER TABLE raw_conversations ADD COLUMN parse_error TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_raw_conv_source_mtime "
        "ON raw_conversations(source_path, file_mtime)"
    )

    # Backfill: mark records that already have parsed conversations
    conn.execute("""
        UPDATE raw_conversations SET parsed_at = acquired_at
        WHERE raw_id IN (
            SELECT DISTINCT raw_id FROM conversations WHERE raw_id IS NOT NULL
        )
    """)


def _migrate_v12_to_v13(conn: sqlite3.Connection) -> None:
    """Migrate from v12 to v13: add performance indices.

    Adds three indices identified via query pattern analysis:
    - idx_raw_conv_unparsed: partial index on unparsed raw records
      (shrinks as records get parsed — near-empty at steady state)
    - idx_conversations_content_hash: speeds up dedup hash lookups
    - idx_messages_conversation_ts: composite covering ORDER BY timestamp
      (superseded by sort_key index in v14, but created here for intermediate state)
    """
    # Partial index: only indexes unparsed rows, so it stays small after
    # initial catch-up and shrinks toward zero at steady state.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_raw_conv_unparsed "
        "ON raw_conversations(raw_id) WHERE parsed_at IS NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_content_hash "
        "ON conversations(content_hash)"
    )
    # Composite index: superseded by idx_messages_conversation_sortkey in v14,
    # but still created here for databases running intermediate migrations.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_conversation_ts "
        "ON messages(conversation_id, timestamp)"
    )


def _migrate_v13_to_v14(conn: sqlite3.Connection) -> None:
    """Migrate from v13 to v14: add pre-computed sort_key to messages.

    Adds a `sort_key REAL` column for O(1) timestamp sorting, replacing the
    per-row CASE/GLOB/strftime computation in ORDER BY clauses. The column
    stores Unix epoch seconds, computed from either numeric or ISO-8601
    timestamp formats.

    Backfills all existing rows and creates a covering index for the most
    common query pattern: messages for a conversation ordered by time.
    """
    conn.execute("ALTER TABLE messages ADD COLUMN sort_key REAL")

    # Backfill: convert existing timestamps to numeric sort keys using the
    # same CASE/GLOB logic the queries used to do at runtime.
    conn.execute("""
        UPDATE messages SET sort_key = CASE
            WHEN timestamp IS NULL THEN NULL
            WHEN timestamp GLOB '*[^0-9.]*' THEN CAST(strftime('%s', timestamp) AS REAL)
            ELSE CAST(timestamp AS REAL)
        END
    """)

    # Replace the old (conversation_id, timestamp) index with a sort_key one.
    # The old index can't be used for ORDER BY because queries used computed
    # expressions, not the raw timestamp column.
    conn.execute("DROP INDEX IF EXISTS idx_messages_conversation_ts")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_conversation_sortkey "
        "ON messages(conversation_id, sort_key)"
    )


def _migrate_v14_to_v15(conn: sqlite3.Connection) -> None:
    """Migrate from v14 to v15: add pre-computed sort_key to conversations.

    Mirrors the messages.sort_key column (v14) for the conversations table,
    eliminating GLOB-based bifurcated timestamp comparisons in WHERE and
    ORDER BY clauses for conversation listing/filtering.

    Backfills from updated_at using the same numeric/ISO-8601 detection.
    """
    conn.execute("ALTER TABLE conversations ADD COLUMN sort_key REAL")

    # Backfill: same CASE/GLOB logic as messages sort_key (v14)
    conn.execute("""
        UPDATE conversations SET sort_key = CASE
            WHEN updated_at IS NULL THEN NULL
            WHEN updated_at GLOB '*[^0-9.]*' THEN CAST(strftime('%s', updated_at) AS REAL)
            ELSE CAST(updated_at AS REAL)
        END
    """)

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_sortkey "
        "ON conversations(sort_key)"
    )


def _migrate_v15_to_v16(conn: sqlite3.Connection) -> None:
    """Migrate from v15 to v16: add 'subagent' to branch_type CHECK constraint.

    SQLite has no ALTER COLUMN — must recreate the conversations table with
    the updated CHECK constraint.  FK enforcement is already disabled by the
    migration runner, so child tables (messages, attachment_refs,
    embedding_status) remain intact throughout.
    """
    # 1. Drop indexes that reference conversations (will be recreated)
    conn.execute("DROP INDEX IF EXISTS idx_conversations_provider")
    conn.execute("DROP INDEX IF EXISTS idx_conversations_source_name")
    conn.execute("DROP INDEX IF EXISTS idx_conversations_parent")
    conn.execute("DROP INDEX IF EXISTS idx_conversations_content_hash")
    conn.execute("DROP INDEX IF EXISTS idx_conversations_sortkey")
    conn.execute("DROP INDEX IF EXISTS idx_conversations_raw_id")

    # 2. Rename existing table
    conn.execute("ALTER TABLE conversations RENAME TO conversations_old")

    # 3. Create new table with updated CHECK constraint
    conn.execute("""
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            sort_key REAL,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            metadata TEXT DEFAULT '{}',
            source_name TEXT GENERATED ALWAYS AS (json_extract(provider_meta, '$.source')) STORED,
            version INTEGER NOT NULL,
            parent_conversation_id TEXT REFERENCES conversations(conversation_id),
            branch_type TEXT CHECK (branch_type IN ('continuation', 'sidechain', 'fork', 'subagent') OR branch_type IS NULL),
            raw_id TEXT REFERENCES raw_conversations(raw_id)
        )
    """)

    # 4. Copy data (exclude generated column source_name)
    conn.execute("""
        INSERT INTO conversations (
            conversation_id, provider_name, provider_conversation_id,
            title, created_at, updated_at, sort_key, content_hash,
            provider_meta, metadata, version,
            parent_conversation_id, branch_type, raw_id
        )
        SELECT
            conversation_id, provider_name, provider_conversation_id,
            title, created_at, updated_at, sort_key, content_hash,
            provider_meta, metadata, version,
            parent_conversation_id, branch_type, raw_id
        FROM conversations_old
    """)

    # 5. Drop old table
    conn.execute("DROP TABLE conversations_old")

    # 6. Recreate all indexes
    conn.execute(
        "CREATE INDEX idx_conversations_provider "
        "ON conversations(provider_name, provider_conversation_id)"
    )
    conn.execute(
        "CREATE INDEX idx_conversations_source_name "
        "ON conversations(source_name) WHERE source_name IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX idx_conversations_parent "
        "ON conversations(parent_conversation_id) WHERE parent_conversation_id IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX idx_conversations_content_hash "
        "ON conversations(content_hash)"
    )
    conn.execute(
        "CREATE INDEX idx_conversations_sortkey "
        "ON conversations(sort_key)"
    )
    conn.execute(
        "CREATE INDEX idx_conversations_raw_id "
        "ON conversations(raw_id) WHERE raw_id IS NOT NULL"
    )


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
    11: _migrate_v11_to_v12,
    12: _migrate_v12_to_v13,
    13: _migrate_v13_to_v14,
    14: _migrate_v14_to_v15,
    15: _migrate_v15_to_v16,
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

        logger.info("Running migration v%d -> v%d", version, version + 1)

        try:
            migration_func(conn)
            conn.execute(f"PRAGMA user_version = {version + 1}")
            conn.commit()  # Commit each step successfully
            logger.info("Migration v%d -> v%d completed", version, version + 1)
        except Exception as exc:
            logger.error("Migration v%d -> v%d failed: %s", version, version + 1, exc)
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
    from polylogue.errors import DatabaseError

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
        conn.execute(_VEC0_DDL)
        conn.commit()
        logger.info("Created missing message_embeddings vec0 table")


__all__ = [
    "SCHEMA_DDL",
    "SCHEMA_VERSION",
]
