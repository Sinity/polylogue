"""Auxiliary schema DDL fragments."""

from __future__ import annotations

VEC0_DDL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
        message_id TEXT PRIMARY KEY,
        embedding float[1024],
        +source_name TEXT,
        +session_id TEXT
    )
"""


__all__ = [
    "VEC0_DDL",
]
