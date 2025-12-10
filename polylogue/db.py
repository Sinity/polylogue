from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .paths import STATE_HOME

DB_PATH = STATE_HOME / "polylogue.db"

_LOCAL = threading.local()

SCHEMA_VERSION = 5


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return False
    return any(row[1] == column for row in rows)


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    version_row = conn.execute("PRAGMA user_version").fetchone()
    current_version = version_row[0] if version_row else 0

    # Base tables (created idempotently to avoid destructive upgrades)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            slug TEXT NOT NULL,
            title TEXT,
            current_branch TEXT,
            root_message_id TEXT,
            last_updated TEXT,
            content_hash TEXT,
            metadata_json TEXT,
            PRIMARY KEY (provider, conversation_id)
        );

        CREATE TABLE IF NOT EXISTS branches (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            parent_branch_id TEXT,
            label TEXT,
            depth INTEGER DEFAULT 0,
            is_current INTEGER DEFAULT 0,
            metadata_json TEXT,
            PRIMARY KEY (provider, conversation_id, branch_id),
            FOREIGN KEY (provider, conversation_id) REFERENCES conversations(provider, conversation_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS messages (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            parent_id TEXT,
            position INTEGER NOT NULL,
            timestamp TEXT,
            role TEXT,
            model TEXT,
            content_hash TEXT,
            content_text TEXT,
            content_json TEXT,
            rendered_text TEXT,
            raw_json TEXT,
            token_count INTEGER DEFAULT 0,
            word_count INTEGER DEFAULT 0,
            attachment_count INTEGER DEFAULT 0,
            attachment_names TEXT,
            is_leaf INTEGER DEFAULT 0,
            metadata_json TEXT,
            PRIMARY KEY (provider, conversation_id, branch_id, position),
            FOREIGN KEY (provider, conversation_id, branch_id) REFERENCES branches(provider, conversation_id, branch_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cmd TEXT NOT NULL,
            count INTEGER DEFAULT 0,
            attachments INTEGER DEFAULT 0,
            attachment_bytes INTEGER DEFAULT 0,
            tokens INTEGER DEFAULT 0,
            skipped INTEGER DEFAULT 0,
            pruned INTEGER DEFAULT 0,
            diffs INTEGER DEFAULT 0,
            duration REAL,
            out TEXT,
            provider TEXT,
            branch_id TEXT,
            metadata_json TEXT
        );

        CREATE TABLE IF NOT EXISTS state_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS raw_imports (
            hash TEXT PRIMARY KEY,
            imported_at INTEGER DEFAULT (unixepoch()),
            provider TEXT NOT NULL,
            source_path TEXT,
            blob BLOB,
            parser_version TEXT,
            parse_status TEXT DEFAULT 'pending',
            error_message TEXT,
            metadata_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_raw_imports_status
            ON raw_imports(parse_status);
        CREATE INDEX IF NOT EXISTS idx_raw_imports_provider
            ON raw_imports(provider, imported_at DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            provider,
            conversation_id,
            branch_id,
            message_id,
            content,
            tokenize='unicode61'
        );
        """
    )

    # Ensure newer columns exist when upgrading from v2
    if not _column_exists(conn, "messages", "attachment_names"):
        conn.execute("ALTER TABLE messages ADD COLUMN attachment_names TEXT")

    # Idempotent indexes
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_branch_order ON messages(provider, conversation_id, branch_id, position)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_conversation_message ON messages(provider, conversation_id, message_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_branches_conversation ON branches(provider, conversation_id)"
    )

    # Attachments indexing (added in schema v3)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS attachments (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            branch_id TEXT,
            message_id TEXT,
            attachment_name TEXT NOT NULL,
            attachment_path TEXT,
            size_bytes INTEGER,
            content_hash TEXT,
            mime_type TEXT,
            text_content TEXT,
            text_bytes INTEGER,
            truncated INTEGER DEFAULT 0,
            ocr_used INTEGER DEFAULT 0,
            PRIMARY KEY (provider, conversation_id, branch_id, message_id, attachment_name),
            FOREIGN KEY (provider, conversation_id) REFERENCES conversations(provider, conversation_id) ON DELETE CASCADE,
            FOREIGN KEY (provider, conversation_id, branch_id) REFERENCES branches(provider, conversation_id, branch_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_attachments_conversation
            ON attachments(provider, conversation_id);

        CREATE INDEX IF NOT EXISTS idx_attachments_name
            ON attachments(attachment_name);

        CREATE VIRTUAL TABLE IF NOT EXISTS attachments_fts USING fts5(
            provider,
            conversation_id,
            branch_id,
            message_id,
            attachment_name,
            content,
            tokenize='unicode61'
        );
        """
    )

    # Schema v5: Assets table with SHA-256 keys and message_assets junction table
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY,
            mime_type TEXT,
            size_bytes INTEGER,
            data BLOB,
            local_path TEXT,
            created_at INTEGER DEFAULT (unixepoch())
        );

        CREATE TABLE IF NOT EXISTS message_assets (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            asset_id TEXT NOT NULL,
            filename TEXT,
            PRIMARY KEY (provider, conversation_id, branch_id, message_id, asset_id),
            FOREIGN KEY (provider, conversation_id, branch_id, message_id)
                REFERENCES messages(provider, conversation_id, branch_id, message_id)
                ON DELETE CASCADE,
            FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_assets_size ON assets(size_bytes);
        CREATE INDEX IF NOT EXISTS idx_message_assets_message
            ON message_assets(provider, conversation_id, branch_id, message_id);
        CREATE INDEX IF NOT EXISTS idx_message_assets_asset ON message_assets(asset_id);
        """
    )

    # Schema v5: FTS5 triggers for automatic sync
    conn.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(provider, conversation_id, branch_id, message_id, content)
            VALUES (new.provider, new.conversation_id, new.branch_id, new.message_id,
                    COALESCE(new.content_text, '') || ' ' || COALESCE(new.rendered_text, ''));
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
            DELETE FROM messages_fts
            WHERE provider = old.provider
              AND conversation_id = old.conversation_id
              AND branch_id = old.branch_id
              AND message_id = old.message_id;
            INSERT INTO messages_fts(provider, conversation_id, branch_id, message_id, content)
            VALUES (new.provider, new.conversation_id, new.branch_id, new.message_id,
                    COALESCE(new.content_text, '') || ' ' || COALESCE(new.rendered_text, ''));
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts
            WHERE provider = old.provider
              AND conversation_id = old.conversation_id
              AND branch_id = old.branch_id
              AND message_id = old.message_id;
        END;
        """
    )

    # Schema v5: Materialized view for canonical paths
    conn.executescript(
        """
        CREATE VIEW IF NOT EXISTS view_canonical_transcript AS
        WITH RECURSIVE chat_tree(
            provider, conversation_id, branch_id, message_id, parent_id,
            position, content_text, role, timestamp, depth
        ) AS (
            -- Start from root (position 0)
            SELECT
                provider, conversation_id, branch_id, message_id, parent_id,
                position, content_text, role, timestamp, 0 as depth
            FROM messages
            WHERE position = 0
            UNION ALL
            -- Traverse down the tree
            SELECT
                m.provider, m.conversation_id, m.branch_id, m.message_id, m.parent_id,
                m.position, m.content_text, m.role, m.timestamp, ct.depth + 1
            FROM messages m
            JOIN chat_tree ct ON m.parent_id = ct.message_id
                AND m.provider = ct.provider
                AND m.conversation_id = ct.conversation_id
        )
        SELECT * FROM chat_tree ORDER BY provider, conversation_id, position;
        """
    )

    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def _get_state() -> dict:
    state = getattr(_LOCAL, "state", None)
    if state is None:
        state = {"conn": None, "path": None, "depth": 0}
        _LOCAL.state = state
    return state


@contextmanager
def open_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    target_path = Path(db_path) if db_path is not None else DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    state = _get_state()
    if state["conn"] is not None and state["path"] != target_path:
        raise RuntimeError(f"Existing connection opened for {state['path']}, cannot open {target_path}")

    created_here = False
    if state["conn"] is None:
        conn = sqlite3.connect(target_path)
        conn.row_factory = sqlite3.Row
        _apply_schema(conn)
        conn.execute("PRAGMA journal_mode=WAL;")
        state["conn"] = conn
        state["path"] = target_path
        created_here = True
    state["depth"] += 1
    try:
        yield state["conn"]
    finally:
        state["depth"] -= 1
        if state["depth"] <= 0:
            if created_here:
                try:
                    state["conn"].commit()
                finally:
                    try:
                        state["conn"].close()
                    finally:
                        state["conn"] = None
                        state["path"] = None
                        state["depth"] = 0


def upsert_conversation(
    conn: sqlite3.Connection,
    *,
    provider: str,
    conversation_id: str,
    slug: str,
    title: Optional[str],
    current_branch: Optional[str],
    root_message_id: Optional[str],
    last_updated: Optional[str],
    content_hash: Optional[str],
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO conversations (
            provider, conversation_id, slug, title, current_branch,
            root_message_id, last_updated, content_hash, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, conversation_id) DO UPDATE SET
            slug=excluded.slug,
            title=excluded.title,
            current_branch=excluded.current_branch,
            root_message_id=excluded.root_message_id,
            last_updated=excluded.last_updated,
            content_hash=excluded.content_hash,
            metadata_json=excluded.metadata_json;
        """,
        (
            provider,
            conversation_id,
            slug,
            title,
            current_branch,
            root_message_id,
            last_updated,
            content_hash,
            json.dumps(metadata) if metadata else None,
        ),
    )


def replace_branches(
    conn: sqlite3.Connection,
    *,
    provider: str,
    conversation_id: str,
    branches: Sequence[Dict[str, object]],
) -> None:
    conn.execute(
        "DELETE FROM branches WHERE provider = ? AND conversation_id = ?",
        (provider, conversation_id),
    )
    conn.executemany(
        """
        INSERT INTO branches (
            provider, conversation_id, branch_id, parent_branch_id, label,
            depth, is_current, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                provider,
                conversation_id,
                b["branch_id"],
                b.get("parent_branch_id"),
                b.get("label"),
                b.get("depth", 0),
                int(b.get("is_current", False)),
                json.dumps(b.get("metadata")) if b.get("metadata") else None,
            )
            for b in branches
        ],
    )


def replace_messages(
    conn: sqlite3.Connection,
    *,
    provider: str,
    conversation_id: str,
    branch_id: str,
    messages: Sequence[Dict[str, object]],
) -> None:
    """Replace all messages for a given branch with transaction protection.

    Uses SAVEPOINT to ensure atomicity: if the operation is interrupted after
    DELETE but before INSERT completes, the transaction is rolled back and no
    data is lost.
    """
    # Use SAVEPOINT for transaction protection to prevent data loss if interrupted
    conn.execute("SAVEPOINT replace_messages_sp")
    try:
        conn.execute(
            "DELETE FROM messages WHERE provider = ? AND conversation_id = ? AND branch_id = ?",
            (provider, conversation_id, branch_id),
        )
        conn.execute(
            "DELETE FROM messages_fts WHERE provider = ? AND conversation_id = ? AND branch_id = ?",
            (provider, conversation_id, branch_id),
        )
        if not messages:
            conn.execute("RELEASE replace_messages_sp")
            return
        conn.executemany(
            """
            INSERT INTO messages (
                provider, conversation_id, branch_id, message_id, parent_id,
                position, timestamp, role, model, content_hash, content_text,
                content_json, rendered_text, raw_json, token_count, word_count,
                attachment_count, attachment_names, is_leaf, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    provider,
                    conversation_id,
                    branch_id,
                    msg["message_id"],
                    msg.get("parent_id"),
                    msg.get("position"),
                    msg.get("timestamp"),
                    msg.get("role"),
                    msg.get("model"),
                    msg.get("content_hash"),
                    msg.get("content_text"),
                    msg.get("content_json"),
                    msg.get("rendered_text"),
                    msg.get("raw_json"),
                    msg.get("token_count", 0),
                    msg.get("word_count", 0),
                    msg.get("attachment_count", 0),
                    msg.get("attachment_names"),
                    msg.get("is_leaf", 0),
                    json.dumps(msg.get("metadata")) if msg.get("metadata") else None,
                )
                for msg in messages
            ],
        )
        conn.execute(
            """
            INSERT INTO messages_fts
                (provider, conversation_id, branch_id, message_id, content)
            SELECT provider, conversation_id, branch_id, message_id, rendered_text
              FROM messages
             WHERE provider = ? AND conversation_id = ? AND branch_id = ?
            """,
            (provider, conversation_id, branch_id),
        )
        # Commit the savepoint on success
        conn.execute("RELEASE replace_messages_sp")
    except Exception:
        # Rollback to savepoint on failure to prevent data loss
        conn.execute("ROLLBACK TO replace_messages_sp")
        raise


def replace_attachments(
    conn: sqlite3.Connection,
    *,
    provider: str,
    conversation_id: str,
    attachments: Sequence[Dict[str, object]],
) -> None:
    """Replace all attachments for a conversation with transaction protection."""

    conn.execute("SAVEPOINT replace_attachments_sp")
    try:
        conn.execute(
            "DELETE FROM attachments WHERE provider = ? AND conversation_id = ?",
            (provider, conversation_id),
        )
        conn.execute(
            "DELETE FROM attachments_fts WHERE provider = ? AND conversation_id = ?",
            (provider, conversation_id),
        )
        if attachments:
            conn.executemany(
                """
                INSERT INTO attachments (
                    provider, conversation_id, branch_id, message_id,
                    attachment_name, attachment_path, size_bytes,
                    content_hash, mime_type, text_content, text_bytes,
                    truncated, ocr_used
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        provider,
                        conversation_id,
                        att.get("branch_id"),
                        att.get("message_id"),
                        att.get("attachment_name"),
                        att.get("attachment_path"),
                        att.get("size_bytes"),
                        att.get("content_hash"),
                        att.get("mime_type"),
                        att.get("text_content"),
                        att.get("text_bytes"),
                        int(bool(att.get("truncated", False))),
                        int(bool(att.get("ocr_used", False))),
                    )
                    for att in attachments
                ],
            )
            conn.execute(
                """
                INSERT INTO attachments_fts
                    (provider, conversation_id, branch_id, message_id, attachment_name, content)
                SELECT provider, conversation_id, branch_id, message_id, attachment_name, COALESCE(text_content, '')
                  FROM attachments
                 WHERE provider = ? AND conversation_id = ?
                """,
                (provider, conversation_id),
            )
        conn.execute("RELEASE replace_attachments_sp")
    except Exception:
        conn.execute("ROLLBACK TO replace_attachments_sp")
        raise


def record_run(
    conn: sqlite3.Connection,
    *,
    timestamp: str,
    cmd: str,
    count: int,
    attachments: int,
    attachment_bytes: int,
    tokens: int,
    skipped: int,
    pruned: int,
    diffs: int,
    duration: Optional[float],
    out: Optional[str],
    provider: Optional[str],
    branch_id: Optional[str],
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO runs (
            timestamp, cmd, count, attachments, attachment_bytes,
            tokens, skipped, pruned, diffs, duration, out, provider,
            branch_id, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp,
            cmd,
            count,
            attachments,
            attachment_bytes,
            tokens,
            skipped,
            pruned,
            diffs,
            duration,
            out,
            provider,
            branch_id,
            json.dumps(metadata) if metadata else None,
        ),
    )


def upsert_raw_import(
    conn: sqlite3.Connection,
    *,
    hash: str,
    provider: str,
    source_path: Optional[str],
    blob: bytes,
    parser_version: str,
    parse_status: str = "pending",
    error_message: Optional[str] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    """Insert or update a raw import record."""
    conn.execute(
        """
        INSERT INTO raw_imports (
            hash, provider, source_path, blob, parser_version,
            parse_status, error_message, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(hash) DO UPDATE SET
            parse_status=excluded.parse_status,
            error_message=excluded.error_message,
            metadata_json=excluded.metadata_json
        """,
        (
            hash,
            provider,
            source_path,
            blob,
            parser_version,
            parse_status,
            error_message,
            json.dumps(metadata) if metadata else None,
        ),
    )


def get_raw_import(conn: sqlite3.Connection, hash: str) -> Optional[sqlite3.Row]:
    """Retrieve a raw import by hash."""
    return conn.execute(
        "SELECT * FROM raw_imports WHERE hash = ?",
        (hash,),
    ).fetchone()


def list_failed_imports(conn: sqlite3.Connection, provider: Optional[str] = None) -> List[sqlite3.Row]:
    """List all imports that failed to parse."""
    if provider:
        return conn.execute(
            "SELECT * FROM raw_imports WHERE parse_status = 'failed' AND provider = ? ORDER BY imported_at DESC",
            (provider,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM raw_imports WHERE parse_status = 'failed' ORDER BY imported_at DESC"
    ).fetchall()
