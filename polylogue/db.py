from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .util import STATE_HOME

DB_PATH = STATE_HOME / "polylogue.db"

SCHEMA_VERSION = 1


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    current_version = conn.execute("PRAGMA user_version").fetchone()[0]
    if current_version >= SCHEMA_VERSION:
        return

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            slug TEXT NOT NULL,
            title TEXT,
            current_branch TEXT,
            last_updated TEXT,
            content_hash TEXT,
            token_count INTEGER DEFAULT 0,
            word_count INTEGER DEFAULT 0,
            attachment_bytes INTEGER DEFAULT 0,
            dirty INTEGER DEFAULT 0,
            metadata_json TEXT,
            PRIMARY KEY (provider, conversation_id)
        );

        CREATE TABLE IF NOT EXISTS branches (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            parent_branch_id TEXT,
            is_canonical INTEGER DEFAULT 0,
            depth INTEGER DEFAULT 0,
            message_count INTEGER DEFAULT 0,
            token_count INTEGER DEFAULT 0,
            word_count INTEGER DEFAULT 0,
            first_message_id TEXT,
            last_message_id TEXT,
            first_timestamp TEXT,
            last_timestamp TEXT,
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
            role TEXT,
            position INTEGER NOT NULL,
            timestamp TEXT,
            token_count INTEGER DEFAULT 0,
            word_count INTEGER DEFAULT 0,
            attachment_count INTEGER DEFAULT 0,
            body TEXT,
            metadata_json TEXT,
            PRIMARY KEY (provider, conversation_id, branch_id, position),
            FOREIGN KEY (provider, conversation_id, branch_id) REFERENCES branches(provider, conversation_id, branch_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation_message
            ON messages(provider, conversation_id, message_id);

        CREATE INDEX IF NOT EXISTS idx_branches_conversation
            ON branches(provider, conversation_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            provider,
            conversation_id,
            branch_id,
            message_id,
            body,
            tokenize='unicode61'
        );
        """
    )

    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


@contextmanager
def open_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    target_path = Path(db_path) if db_path is not None else DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    _apply_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def upsert_conversation(
    conn: sqlite3.Connection,
    *,
    provider: str,
    conversation_id: str,
    slug: str,
    title: Optional[str],
    current_branch: Optional[str],
    last_updated: Optional[str],
    content_hash: Optional[str],
    token_count: int,
    word_count: int,
    attachment_bytes: int,
    dirty: bool,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO conversations (
            provider, conversation_id, slug, title, current_branch,
            last_updated, content_hash, token_count, word_count,
            attachment_bytes, dirty, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, conversation_id) DO UPDATE SET
            slug=excluded.slug,
            title=excluded.title,
            current_branch=excluded.current_branch,
            last_updated=excluded.last_updated,
            content_hash=excluded.content_hash,
            token_count=excluded.token_count,
            word_count=excluded.word_count,
            attachment_bytes=excluded.attachment_bytes,
            dirty=excluded.dirty,
            metadata_json=excluded.metadata_json;
        """,
        (
            provider,
            conversation_id,
            slug,
            title,
            current_branch,
            last_updated,
            content_hash,
            token_count,
            word_count,
            attachment_bytes,
            int(dirty),
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
            provider, conversation_id, branch_id, parent_branch_id, is_canonical,
            depth, message_count, token_count, word_count, first_message_id,
            last_message_id, first_timestamp, last_timestamp, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                provider,
                conversation_id,
                b["branch_id"],
                b.get("parent_branch_id"),
                int(b.get("is_canonical", False)),
                b.get("depth", 0),
                b.get("message_count", 0),
                b.get("token_count", 0),
                b.get("word_count", 0),
                b.get("first_message_id"),
                b.get("last_message_id"),
                b.get("first_timestamp"),
                b.get("last_timestamp"),
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
    conn.execute(
        """
        DELETE FROM messages
        WHERE provider = ? AND conversation_id = ? AND branch_id = ?
        """,
        (provider, conversation_id, branch_id),
    )
    conn.execute(
        """
        DELETE FROM messages_fts
        WHERE provider = ? AND conversation_id = ? AND branch_id = ?
        """,
        (provider, conversation_id, branch_id),
    )
    if not messages:
        return
    conn.executemany(
        """
        INSERT INTO messages (
            provider, conversation_id, branch_id, message_id, parent_id,
            role, position, timestamp, token_count, word_count, attachment_count,
            body, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                provider,
                conversation_id,
                branch_id,
                msg["message_id"],
                msg.get("parent_id"),
                msg.get("role"),
                msg.get("position"),
                msg.get("timestamp"),
                msg.get("token_count", 0),
                msg.get("word_count", 0),
                msg.get("attachment_count", 0),
                msg.get("body"),
                json.dumps(msg.get("metadata")) if msg.get("metadata") else None,
            )
            for msg in messages
        ],
    )
    conn.execute(
        """
        INSERT INTO messages_fts
            (provider, conversation_id, branch_id, message_id, body)
        SELECT provider, conversation_id, branch_id, message_id, body
        FROM messages
        WHERE provider = ? AND conversation_id = ? AND branch_id = ?
        """,
        (provider, conversation_id, branch_id),
    )
