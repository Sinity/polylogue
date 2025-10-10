from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

from .render import MarkdownDocument
from .util import STATE_HOME


def _db_path() -> Path:
    return STATE_HOME / "index.sqlite"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            slug TEXT NOT NULL,
            path TEXT NOT NULL,
            title TEXT,
            updated_at TEXT,
            content_hash TEXT,
            tokens INTEGER,
            dirty INTEGER,
            metadata_json TEXT,
            PRIMARY KEY(provider, conversation_id)
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
        USING fts5(provider, conversation_id, title, content)
        """
    )


def update_sqlite_index(
    *,
    provider: str,
    conversation_id: str,
    slug: str,
    path: Path,
    document: MarkdownDocument,
    metadata: Dict[str, Any],
) -> None:
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        content = document.body
        title = document.metadata.get("title")
        tokens = document.stats.get("totalTokensApprox") or 0
        dirty = bool(metadata.get("dirty"))
        content_hash = metadata.get("contentHash")
        updated_at = metadata.get("updatedAt")
        metadata_json = json.dumps(metadata, ensure_ascii=False)

        conn.execute(
            """
            INSERT INTO conversations (provider, conversation_id, slug, path, title, updated_at, content_hash, tokens, dirty, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, conversation_id) DO UPDATE SET
                slug=excluded.slug,
                path=excluded.path,
                title=excluded.title,
                updated_at=excluded.updated_at,
                content_hash=excluded.content_hash,
                tokens=excluded.tokens,
                dirty=excluded.dirty,
                metadata_json=excluded.metadata_json
            """,
            (
                provider,
                conversation_id,
                slug,
                str(path),
                title,
                updated_at,
                content_hash,
                int(tokens),
                int(dirty),
                metadata_json,
            ),
        )
        conn.execute(
            "DELETE FROM conversations_fts WHERE provider = ? AND conversation_id = ?",
            (provider, conversation_id),
        )
        conn.execute(
            "INSERT INTO conversations_fts(provider, conversation_id, title, content) VALUES (?, ?, ?, ?)",
            (provider, conversation_id, title or "", content),
        )
        conn.commit()
    finally:
        conn.close()
