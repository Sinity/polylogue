"""FTS5 full-text search provider implementation.

This module provides a SearchProvider implementation using SQLite's FTS5
extension for full-text search capabilities.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.backends.sqlite import connection_context, open_connection
from polylogue.storage.store import MessageRecord


class FTS5Provider:
    """SearchProvider implementation using SQLite FTS5.

    This provider uses SQLite's built-in FTS5 (Full-Text Search) extension
    to index and search message content. Messages are stored in a virtual
    table optimized for text search with relevance ranking.

    Attributes:
        db_path: Path to SQLite database. If None, uses default from config.
    """

    def __init__(self, db_path: Path | None = None):
        """Initialize FTS5 search provider.

        Args:
            db_path: Optional path to SQLite database. If None, uses the
                    default database path from the application config.
        """
        self.db_path = db_path

    def _ensure_index(self, conn: sqlite3.Connection) -> None:
        """Create FTS5 virtual table if it doesn't exist.

        Args:
            conn: Active SQLite database connection
        """
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                message_id UNINDEXED,
                conversation_id UNINDEXED,
                content
            );
            """
        )

    def index(self, messages: list[MessageRecord]) -> None:
        """Index a list of messages for full-text search.

        This operation is incremental - it deletes existing entries for the
        affected conversations and re-inserts them. This ensures idempotency.

        Args:
            messages: Messages to index. Must include message_id, conversation_id,
                     and text content.

        Raises:
            DatabaseError: If indexing fails
        """
        if not messages:
            return

        # Extract unique conversation IDs
        conversation_ids = list({msg.conversation_id for msg in messages})

        with connection_context(self.db_path) as conn:
            self._ensure_index(conn)

            # Batch delete existing entries for these conversations
            if conversation_ids:
                placeholders = ",".join("?" * len(conversation_ids))
                conn.execute(f"DELETE FROM messages_fts WHERE conversation_id IN ({placeholders})", conversation_ids)

            # Prepare batch insert with provider_name lookup
            # Filter messages with text content
            messages_to_index = [msg for msg in messages if msg.text]

            if messages_to_index:
                # Build lookup map of conversation_id -> provider_name
                # Prepare batch insert data
                insert_data = [(msg.message_id, msg.conversation_id, msg.text) for msg in messages_to_index]

                # Batch insert using executemany
                if insert_data:
                    conn.executemany(
                        """
                        INSERT INTO messages_fts (message_id, conversation_id, content)
                        VALUES (?, ?, ?)
                        """,
                        insert_data,
                    )

            conn.commit()

    def search(self, query: str) -> list[str]:
        """Execute a full-text search query.

        Uses FTS5's MATCH syntax with relevance ranking. Results are ordered
        by best match first.

        Args:
            query: Search query string (FTS5 syntax)

        Returns:
            List of message IDs matching the query, ordered by relevance

        Raises:
            DatabaseError: If query is malformed or search fails
        """
        from polylogue.storage.search import escape_fts5_query

        # Escape the query to prevent syntax errors
        fts_query = escape_fts5_query(query)

        with open_connection(self.db_path) as conn:
            # Check if index exists
            row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()

            if not row:
                return []

            # Search with relevance ranking
            rows = conn.execute(
                """
                SELECT message_id
                FROM messages_fts
                WHERE messages_fts MATCH ?
                ORDER BY rank
                """,
                (fts_query,),
            ).fetchall()

            return [row["message_id"] for row in rows]


__all__ = ["FTS5Provider"]
