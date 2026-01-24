from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 5
_LOCAL = threading.local()


class DatabaseError(Exception):
    """Database-related errors (schema, connection, migration)."""


def default_db_path() -> Path:
    """Return the default database path.

    Uses XDG_DATA_HOME/polylogue/polylogue.db (semantic data, not ephemeral state).
    """
    from polylogue.paths import DATA_HOME
    return DATA_HOME / "polylogue.db"


def _apply_schema(conn: sqlite3.Connection) -> None:
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
            metadata TEXT DEFAULT '{}',
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

        CREATE INDEX IF NOT EXISTS idx_messages_conversation_covering
        ON messages(conversation_id, message_id, content_hash);

        CREATE INDEX IF NOT EXISTS idx_messages_conversation_ordering
        ON messages(conversation_id, timestamp, message_id);

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


def _make_ref_id(attachment_id: str, conversation_id: str, message_id: str | None) -> str:
    import hashlib

    seed = f"{attachment_id}:{conversation_id}:{message_id or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"ref-{digest}"


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
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
    """Add computed source_name column to conversations for faster filtering."""
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


def _migrate_v4_to_v5(conn: sqlite3.Connection) -> None:
    """Add metadata column for user-editable fields."""
    conn.execute("ALTER TABLE conversations ADD COLUMN metadata TEXT DEFAULT '{}'")


# Migration registry: maps source version to migration function
_MIGRATIONS = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
    3: _migrate_v3_to_v4,
    4: _migrate_v4_to_v5,
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


def _ensure_schema(conn: sqlite3.Connection, max_retries: int = 5) -> None:
    """Ensure database schema is at current version, running migrations if needed.

    For fresh databases (version 0), creates the schema directly.
    For existing databases, runs migrations sequentially with rollback support.

    Thread safety: When multiple threads simultaneously try to create the schema,
    SQLite may return "database is locked" errors. This function retries with
    exponential backoff to handle concurrent schema initialization.

    Args:
        conn: Database connection
        max_retries: Maximum number of retries on lock contention (default: 5)

    Raises:
        DatabaseError: If schema version is unsupported or migration fails
    """
    import time

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
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

            return  # Schema is at correct version

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                last_error = e
                # Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
                sleep_time = 0.1 * (2**attempt)
                LOGGER.debug("Schema creation locked (attempt %d/%d), retrying in %.1fs", attempt + 1, max_retries, sleep_time)
                time.sleep(sleep_time)
            else:
                raise

    # All retries exhausted
    raise DatabaseError(f"Failed to initialize schema after {max_retries} retries: {last_error}")


def _execute_pragma_with_retry(conn: sqlite3.Connection, pragma: str, max_retries: int = 5) -> None:
    """Execute a PRAGMA statement with retry on database lock.

    Some PRAGMAs like journal_mode=WAL can fail with "database is locked" when
    multiple processes/threads initialize the database simultaneously.

    Args:
        conn: Database connection
        pragma: PRAGMA statement to execute
        max_retries: Maximum retries on lock contention
    """
    import time

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            conn.execute(pragma)
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                last_error = e
                sleep_time = 0.1 * (2**attempt)
                LOGGER.debug("PRAGMA locked (attempt %d/%d), retrying in %.1fs", attempt + 1, max_retries, sleep_time)
                time.sleep(sleep_time)
            else:
                raise

    raise DatabaseError(f"Failed to execute {pragma} after {max_retries} retries: {last_error}")


def _get_state() -> dict[str, object]:
    state: dict[str, object] | None = getattr(_LOCAL, "state", None)
    if state is None:
        state = {"conn": None, "path": None, "depth": 0}
        _LOCAL.state = state
    return state


@contextmanager
def open_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    target_path = Path(db_path) if db_path is not None else default_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    state = _get_state()
    state_conn: sqlite3.Connection | None = state["conn"]  # type: ignore[assignment]
    state_path: Path | None = state["path"]  # type: ignore[assignment]
    state_depth: int = state["depth"]  # type: ignore[assignment]

    if state_conn is not None and state_path != target_path:
        raise DatabaseError(f"Existing connection opened for {state_path}, cannot open {target_path}")

    created_here = False
    if state_conn is None:
        conn = sqlite3.connect(target_path, timeout=30)
        conn.row_factory = sqlite3.Row
        # Set busy_timeout FIRST to handle concurrent access during PRAGMA setup
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        # WAL mode can fail with "database is locked" during concurrent initialization
        _execute_pragma_with_retry(conn, "PRAGMA journal_mode=WAL;")
        _ensure_schema(conn)
        state["conn"] = conn
        state["path"] = target_path
        state_conn = conn
        created_here = True
    state["depth"] = state_depth + 1
    exc_info = None
    try:
        yield state_conn
    except Exception:
        exc_info = True
        raise
    finally:
        state_depth_val: int = state["depth"]  # type: ignore[assignment]
        state_depth_val = state_depth_val - 1
        state["depth"] = state_depth_val
        if state_depth_val <= 0 and created_here:
            try:
                if exc_info:
                    state_conn.rollback()
                else:
                    state_conn.commit()
            finally:
                try:
                    state_conn.close()
                finally:
                    state["conn"] = None
                    state["path"] = None
                    state["depth"] = 0


@contextmanager
def connection_context(conn: sqlite3.Connection | None = None, db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Use a provided connection or open a new one via open_connection.

    This is a convenience wrapper that allows code to optionally accept a connection
    or create one automatically. It enables both standalone usage and composition
    within larger transactions.

    Transaction ownership and commit behavior:
        - If conn is provided: The caller owns the connection and is responsible for
          commit/rollback. This function does NOT commit or rollback. The connection
          remains open after the context exits.

        - If conn is None: A new connection is opened via open_connection(), which
          automatically commits on successful exit and closes the connection. The
          caller does NOT need to commit.

    Thread safety:
        - Each thread has its own connection state (via threading.local in open_connection).
        - Passing a connection between threads is NOT safe - SQLite connections are
          not thread-safe.
        - If you need concurrent database access, each thread should call this with
          conn=None to get its own connection.

    Context manager behavior:
        - Always use this as a context manager (with statement).
        - The yielded connection is valid only within the context.
        - Do not store the connection for use outside the context.

    Args:
        conn: Optional existing connection to use. If provided, caller retains ownership.
        db_path: Optional database path. Only used if conn is None. Defaults to
                 default_db_path() if not specified.

    Yields:
        sqlite3.Connection: The connection to use (either provided or newly opened).

    Example:
        # Standalone usage (auto-commit, auto-close):
        with connection_context() as conn:
            conn.execute("INSERT INTO ...")

        # Composed within a transaction:
        with open_connection() as conn:
            with connection_context(conn) as same_conn:
                # Operations here are part of the outer transaction
                same_conn.execute("INSERT INTO ...")
            # Commit happens here when open_connection context exits
    """
    if conn:
        yield conn
    else:
        with open_connection(db_path) as new_conn:
            yield new_conn


__all__ = ["open_connection", "default_db_path", "SCHEMA_VERSION", "connection_context"]
