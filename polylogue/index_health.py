from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import List

from .db import DB_PATH


def verify_sqlite_indexes(db_path: Path = DB_PATH, *, attempt_rebuild: bool = True) -> List[str]:
    issues: List[str] = []
    conn = sqlite3.connect(db_path)
    try:
        status = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if status != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {status}")
        try:
            conn.execute("SELECT count(*) FROM messages_fts")
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "no such table" in message:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                        provider,
                        conversation_id,
                        branch_id,
                        message_id,
                        content,
                        tokenize='unicode61'
                    )
                    """
                )
                issues.append("messages_fts table was missing; recreated")
            elif attempt_rebuild:
                conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
                issues.append("Rebuilt messages_fts index")
            else:
                raise
            conn.commit()
    finally:
        conn.close()
    return issues


def verify_qdrant_collection() -> List[str]:
    backend = os.environ.get("POLYLOGUE_INDEX_BACKEND", "sqlite").strip().lower()
    if backend != "qdrant":
        return []
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("qdrant-client package is required for POLYLOGUE_INDEX_BACKEND=qdrant") from exc

    url = os.environ.get("POLYLOGUE_QDRANT_URL")
    api_key = os.environ.get("POLYLOGUE_QDRANT_API_KEY")
    collection = os.environ.get("POLYLOGUE_QDRANT_COLLECTION", "polylogue")
    if not url:
        raise RuntimeError("POLYLOGUE_QDRANT_URL must be set when using the qdrant index backend")

    client = QdrantClient(url=url, api_key=api_key)
    if not client.collection_exists(collection_name=collection):
        raise RuntimeError(f"Qdrant collection '{collection}' not found at {url}")
    return [f"Qdrant collection '{collection}' reachable"]
