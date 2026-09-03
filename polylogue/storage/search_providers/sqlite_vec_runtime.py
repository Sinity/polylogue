"""Runtime/capability helpers for the sqlite-vec provider."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.embeddings.identity import (
    VECTOR_DERIVATION_HASH_SQL_FUNCTION,
    register_embedding_identity_sql,
)
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
        archive_root: Path | None
        _legacy_compatibility: bool
        _admitted_db_identity: tuple[int, int] | None

    def _assert_lifecycle_binding(self) -> None:
        if getattr(self, "_legacy_compatibility", False):
            return
        if self.archive_root is None:
            raise SqliteVecError("managed vector provider requires an archive root")
        root = self.archive_root.resolve(strict=True)
        db = self.db_path.resolve(strict=False)
        try:
            db.relative_to(root)
        except ValueError as exc:
            raise SqliteVecError("managed provider path is outside its trusted archive root") from exc
        if db.name != "embeddings.db":
            raise SqliteVecError("managed provider path must be archive-local embeddings.db")
        try:
            st = os.stat(db)
        except FileNotFoundError:
            raise SqliteVecError("managed embeddings database disappeared after admission") from None
        identity: tuple[int, int] = (st.st_dev, st.st_ino)
        bound = getattr(self, "_admitted_db_identity", None)
        if bound is not None and identity != bound:
            raise SqliteVecError("managed embeddings database changed after admission")
        self._admitted_db_identity = identity

    @contextmanager
    def _lifecycle_admission(self) -> Iterator[None]:
        """Bind managed provider use to the resolved archive path and inode."""
        if getattr(self, "_legacy_compatibility", False):
            yield
            return
        self._assert_lifecycle_binding()
        try:
            yield
        finally:
            self._admitted_db_identity = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get connection with sqlite-vec extension loaded if available."""
        self._assert_lifecycle_binding()
        conn = open_connection(self.db_path.resolve(strict=False))
        conn.row_factory = sqlite3.Row

        if getattr(self, "_legacy_compatibility", False):
            try:
                conn.executescript(
                    """
                    CREATE TEMP TABLE current_embedding_messages (
                        message_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        origin TEXT NOT NULL,
                        vector_derivation_hash BLOB NOT NULL
                    );
                    INSERT INTO current_embedding_messages
                    SELECT message_id, session_id, origin, vector_derivation_hash
                    FROM message_embedding_refs;
                    """
                )
            except sqlite3.Error:
                conn.execute("DROP TABLE IF EXISTS temp.current_embedding_messages")
        elif self.archive_root is not None:
            register_embedding_identity_sql(conn)
            index_path = resolve_active_index_path(self.archive_root).resolve(strict=False)
            if index_path != self.db_path.resolve(strict=False):
                # ATTACH creates the file when it does not exist, so a missing
                # or mistyped index would silently become an empty database
                # and the projection below would report zero eligible messages
                # as though the archive held none.
                if not index_path.is_file():
                    conn.close()
                    raise SqliteVecError(f"managed embedding projection found no active index at {index_path}")
                try:
                    conn.execute("ATTACH DATABASE ? AS archive_index", (str(index_path),))
                    conn.executescript(
                        """
                        CREATE TEMP TABLE current_embedding_messages (
                            message_id TEXT PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            origin TEXT NOT NULL,
                            vector_derivation_hash BLOB NOT NULL
                        );
                        """
                    )
                    conn.execute(
                        f"""
                        INSERT INTO current_embedding_messages (
                            message_id, session_id, origin, vector_derivation_hash
                        )
                        SELECT eligible.message_id, eligible.session_id, eligible.origin,
                               {VECTOR_DERIVATION_HASH_SQL_FUNCTION}(?, eligible.text)
                        FROM (
                            SELECT m.message_id, m.session_id, s.origin,
                                   (
                                       SELECT GROUP_CONCAT(prose.text, char(10) || char(10))
                                       FROM (
                                           SELECT b.text
                                           FROM archive_index.blocks b
                                           WHERE b.message_id = m.message_id
                                             AND b.block_type = 'text'
                                             AND b.text IS NOT NULL
                                           ORDER BY b.position
                                       ) AS prose
                                   ) AS text
                            FROM archive_index.messages m
                            JOIN archive_index.sessions s ON s.session_id = m.session_id
                            WHERE m.message_type = 'message'
                              AND m.role IN ('user', 'assistant')
                              AND m.material_origin IN ('human_authored', 'assistant_authored')
                              AND m.word_count > 0
                        ) AS eligible
                        WHERE LENGTH(TRIM(COALESCE(eligible.text, ''))) >= 20
                        """,
                        (self.model,),
                    )
                except sqlite3.Error as exc:
                    conn.close()
                    raise SqliteVecError("managed embedding projection could not bind the active index") from exc

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
        """Create required tables under lifecycle admission for managed tiers."""
        if getattr(self, "_legacy_compatibility", False) or self.db_path.name != "embeddings.db":
            self._ensure_tables_unlocked()
            return
        self._ensure_tables_unlocked()

    def _ensure_tables_unlocked(self) -> None:
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
