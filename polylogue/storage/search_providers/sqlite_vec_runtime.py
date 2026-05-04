"""Runtime/capability helpers for the sqlite-vec provider."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError, logger
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec


class SqliteVecRuntimeMixin:
    """Connection, capability, and table-management helpers."""

    if TYPE_CHECKING:
        db_path: Path
        _vec_available: bool | None
        _tables_ensured: bool

    def _get_connection(self) -> sqlite3.Connection:
        """Get connection with sqlite-vec extension loaded if available."""
        conn = open_connection(self.db_path)
        conn.row_factory = sqlite3.Row

        if self._vec_available is None:
            loaded, error = try_load_sqlite_vec(conn)
            if loaded:
                self._vec_available = True
            elif isinstance(error, ImportError):
                logger.warning("sqlite-vec not installed")
                self._vec_available = False
            else:
                logger.warning("sqlite-vec load failed: %s", error)
                self._vec_available = False
        elif self._vec_available:
            loaded, error = try_load_sqlite_vec(conn)
            if not loaded:
                conn.close()
                if error is None:
                    raise SqliteVecError("sqlite-vec extension failed to load on connection: unknown error")
                raise SqliteVecError(f"sqlite-vec extension failed to load on connection: {error}") from error

        return conn

    def _ensure_vec_available(self) -> None:
        """Ensure sqlite-vec is available, raising error if not."""
        if self._vec_available is None:
            conn = self._get_connection()
            conn.close()
        if not self._vec_available:
            raise SqliteVecError("sqlite-vec extension not available. Install with: pip install sqlite-vec")

    def _ensure_tables(self) -> None:
        """Create required vector and metadata tables if they don't exist."""
        if self._tables_ensured:
            return

        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
                    message_id TEXT PRIMARY KEY,
                    embedding float[1024],
                    +provider_name TEXT,
                    +conversation_id TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_meta (
                    target_id TEXT PRIMARY KEY,
                    target_type TEXT NOT NULL CHECK (target_type IN ('message', 'conversation')),
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    embedded_at TEXT NOT NULL,
                    content_hash TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_meta_type
                ON embeddings_meta(target_type)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_status (
                    conversation_id TEXT PRIMARY KEY,
                    message_count_embedded INTEGER DEFAULT 0,
                    last_embedded_at TEXT,
                    needs_reindex INTEGER DEFAULT 0,
                    error_message TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embedding_status_needs
                ON embedding_status(needs_reindex) WHERE needs_reindex = 1
            """)
            conn.commit()
            self._tables_ensured = True
        finally:
            conn.close()


__all__ = ["SqliteVecRuntimeMixin"]
