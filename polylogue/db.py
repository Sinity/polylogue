from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .paths import STATE_HOME

DB_PATH = STATE_HOME / "polylogue.db"

SCHEMA_VERSION = 2


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    version_row = conn.execute("PRAGMA user_version").fetchone()
    current_version = version_row[0] if version_row else 0
    if current_version >= SCHEMA_VERSION:
        return

    conn.executescript(
        """
        DROP TABLE IF EXISTS messages_fts;
        DROP TABLE IF EXISTS messages;
        DROP TABLE IF EXISTS branches;
        DROP TABLE IF EXISTS conversations;
        DROP TABLE IF EXISTS runs;
        DROP TABLE IF EXISTS state_meta;

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
            content_hash TEXT,
            rendered_text TEXT,
            raw_json TEXT,
            token_count INTEGER DEFAULT 0,
            word_count INTEGER DEFAULT 0,
            attachment_count INTEGER DEFAULT 0,
            metadata_json TEXT,
            PRIMARY KEY (provider, conversation_id, branch_id, position),
            FOREIGN KEY (provider, conversation_id, branch_id) REFERENCES branches(provider, conversation_id, branch_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_branch_order
            ON messages(provider, conversation_id, branch_id, position);

        CREATE INDEX IF NOT EXISTS idx_messages_conversation_message
            ON messages(provider, conversation_id, message_id);

        CREATE INDEX IF NOT EXISTS idx_branches_conversation
            ON branches(provider, conversation_id);

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
    conn.execute(
        "DELETE FROM messages WHERE provider = ? AND conversation_id = ? AND branch_id = ?",
        (provider, conversation_id, branch_id),
    )
    conn.execute(
        "DELETE FROM messages_fts WHERE provider = ? AND conversation_id = ? AND branch_id = ?",
        (provider, conversation_id, branch_id),
    )
    if not messages:
        return
    conn.executemany(
        """
        INSERT INTO messages (
            provider, conversation_id, branch_id, message_id, parent_id,
            position, timestamp, role, content_hash, rendered_text,
            raw_json, token_count, word_count, attachment_count, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                msg.get("content_hash"),
                msg.get("rendered_text"),
                msg.get("raw_json"),
                msg.get("token_count", 0),
                msg.get("word_count", 0),
                msg.get("attachment_count", 0),
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
