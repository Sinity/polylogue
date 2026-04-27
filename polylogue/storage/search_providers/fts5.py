"""FTS5 full-text search provider implementation.

This module provides a SearchProvider implementation using SQLite's FTS5
extension for full-text search capabilities.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.storage.backends.connection import connection_context, open_read_connection
from polylogue.storage.fts.fts_lifecycle import replace_fts_rows_for_messages_sync
from polylogue.storage.index import ensure_index
from polylogue.storage.runtime import MessageRecord
from polylogue.storage.search.cache import invalidate_search_cache


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

    def index(self, messages: list[MessageRecord]) -> None:
        """Repair index rows for the supplied conversations from persisted messages.

        Args:
            messages: Messages whose persisted conversations should be re-indexed.
                Messages must already exist in the SQLite archive.

        Raises:
            DatabaseError: If indexing fails
        """
        rows = [(m.message_id, m.conversation_id, m.text) for m in messages]
        with connection_context(self.db_path) as conn:
            ensure_index(conn)
            replace_fts_rows_for_messages_sync(conn, rows)
            conn.commit()
        if messages:
            invalidate_search_cache()

    def search(self, query: str, limit: int | None = None) -> list[str]:
        """Execute a full-text search query.

        Uses FTS5's MATCH syntax with relevance ranking. Results are ordered
        by best match first.

        Args:
            query: Search query string (FTS5 syntax)
            limit: Optional maximum number of message IDs to return

        Returns:
            List of message IDs matching the query, ordered by relevance

        Raises:
            DatabaseError: If query is malformed or search fails
        """
        from polylogue.storage.search import normalize_fts5_query

        # Escape the query to prevent syntax errors
        fts_query = normalize_fts5_query(query)
        if fts_query is None:
            return []

        with open_read_connection(self.db_path) as conn:
            # Check if index exists
            row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()

            if not row:
                return []

            # Search with relevance ranking
            sql = """
                SELECT message_id
                FROM messages_fts
                WHERE messages_fts MATCH ?
                ORDER BY bm25(messages_fts), message_id
            """
            params: list[object] = [fts_query]
            if limit is not None:
                sql += "\nLIMIT ?"
                params.append(limit)
            rows = conn.execute(sql, tuple(params)).fetchall()

            return [row["message_id"] for row in rows]


__all__ = ["FTS5Provider"]
