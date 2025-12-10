from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
import threading
from typing import Dict, Iterable, List, Optional, Sequence

from .paths import STATE_HOME

DB_PATH = STATE_HOME / "polylogue.db"

_LOCAL = threading.local()

SCHEMA_VERSION = 5


def _run_migrations(db_path: Path) -> None:
    """Run Alembic migrations to ensure database schema is up to date.

    This replaces the old _apply_schema() function with proper Alembic migrations.
    Migrations are defined in polylogue/alembic/versions/.
    """
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Get project root (where alembic.ini is located)
    project_root = Path(__file__).parent.parent

    # Run alembic upgrade head
    try:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            # If alembic fails, it might not be installed - fall back to manual check
            # This allows the code to work even if alembic isn't available at runtime
            conn = sqlite3.connect(db_path)
            try:
                # Check if tables exist
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
                )
                if not cursor.fetchone():
                    # No tables exist - need migration but alembic failed
                    raise RuntimeError(
                        f"Database migration failed and no tables exist. "
                        f"Install alembic or run migrations manually.\n"
                        f"Error: {result.stderr}"
                    )
            finally:
                conn.close()
    except FileNotFoundError:
        # Alembic not installed - warn but don't fail if DB already exists
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
            )
            if not cursor.fetchone():
                raise RuntimeError(
                    "Alembic is not installed and database is not initialized. "
                    "Install alembic with: pip install alembic"
                )
        finally:
            conn.close()


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
        # Run Alembic migrations to ensure schema is up to date
        _run_migrations(target_path)

        # Open connection
        conn = sqlite3.connect(target_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
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
