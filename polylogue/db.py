from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


SCHEMA_VERSION = 2
_LOCAL = threading.local()


def default_db_path() -> Path:
    raw_state_root = os.environ.get("XDG_STATE_HOME")
    state_root = Path(raw_state_root).expanduser() if raw_state_root else Path.home() / ".local/state"
    return state_root / "polylogue" / "polylogue.db"


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
            version INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_provider
        ON conversations(provider_name, provider_conversation_id);

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
            duration_ms INTEGER,
            profile TEXT
        );
        """
    )
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def _make_ref_id(attachment_id: str, conversation_id: str, message_id: Optional[str]) -> str:
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
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.commit()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    row = conn.execute("PRAGMA user_version").fetchone()
    version = row[0] if row else 0
    if version == 0:
        _apply_schema(conn)
        return
    if version == 1:
        _migrate_v1_to_v2(conn)
        return
    if version != SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported DB schema version {version} (expected {SCHEMA_VERSION})")


def _get_state() -> dict:
    state = getattr(_LOCAL, "state", None)
    if state is None:
        state = {"conn": None, "path": None, "depth": 0}
        _LOCAL.state = state
    return state


@contextmanager
def open_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    target_path = Path(db_path) if db_path is not None else default_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    state = _get_state()
    if state["conn"] is not None and state["path"] != target_path:
        raise RuntimeError(f"Existing connection opened for {state['path']}, cannot open {target_path}")

    created_here = False
    if state["conn"] is None:
        conn = sqlite3.connect(target_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode=WAL;")
        _ensure_schema(conn)
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


__all__ = ["open_connection", "default_db_path", "SCHEMA_VERSION"]
