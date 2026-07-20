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
        model: str
        dimension: int
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
        """Create required vector and metadata tables if they don't exist.

        Detects dimension mismatches between the configured dimension and the
        existing vec0 table. Drops and recreates the vec0 table when the
        dimension has changed.

        Uses the canonical archive_tiers DDL (:mod:`polylogue.storage.sqlite.
        archive_tiers.embeddings`) rather than a duplicate hand-rolled schema
        -- a second, drifted declaration here previously created ``+source_name``
        / message_id-keyed shapes that mismatched what the archive_tiers
        bootstrap (and the daemon catch-up path) actually writes, silently
        breaking this provider's own :meth:`SqliteVecQueryMixin.upsert` when
        both ran against the same ``embeddings.db``.
        """
        conn = self._get_connection()
        try:
            # Detect and handle dimension mismatch before creating tables
            _reconcile_vec0_dimension(conn, self.dimension)

            from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_DDL

            conn.executescript(EMBEDDINGS_DDL)
            conn.commit()
            self._tables_ensured = True
        finally:
            conn.close()

    def _stored_embedding_dimension(self) -> int | None:
        """Return the dimension stored in message_embeddings_meta, if any."""
        conn = self._get_connection()
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='message_embeddings_meta'"
            ).fetchone()
            if has_table is None:
                return None
            row = conn.execute("SELECT dimension FROM message_embeddings_meta LIMIT 1").fetchone()
            return int(row["dimension"]) if row else None
        except (sqlite3.OperationalError, TypeError, ValueError):
            return None
        finally:
            conn.close()


def _vec0_table_dimension(conn: sqlite3.Connection) -> int | None:
    """Read the dimension of the existing vec0 table, if it exists."""
    try:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='message_embeddings'"
        ).fetchone()
        if has_table is None:
            return None
        # SQLite vec0 stores dimension in the CREATE VIRTUAL TABLE DDL.
        ddl_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='message_embeddings'"
        ).fetchone()
        if ddl_row is None or ddl_row["sql"] is None:
            return None
        import re

        match = re.search(r"float\[(\d+)\]", str(ddl_row["sql"]))
        return int(match.group(1)) if match else None
    except (sqlite3.OperationalError, TypeError, ValueError):
        return None


def _reconcile_vec0_dimension(conn: sqlite3.Connection, configured_dimension: int) -> None:
    """Drop vec0 table when its dimension differs from the configured dimension."""
    current = _vec0_table_dimension(conn)
    if current is not None and current != configured_dimension:
        logger.info(
            "vec0 dimension mismatch: stored=%d configured=%d — dropping message_embeddings",
            current,
            configured_dimension,
        )
        conn.execute("DROP TABLE IF EXISTS message_embeddings")
        conn.commit()


__all__ = ["SqliteVecRuntimeMixin", "_reconcile_vec0_dimension"]
