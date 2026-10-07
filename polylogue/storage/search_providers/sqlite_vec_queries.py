"""Query operations for the sqlite-vec provider."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, closing, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.core.errors import EmbeddingRetrievalNotReadyError, SessionNotFoundError
from polylogue.core.protocols import ScopedVectorQuery
from polylogue.storage.search_providers.sqlite_vec_runtime import _assert_vec0_dimension
from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError, _serialize_f32
from polylogue.storage.sqlite.connection_profile import readonly_temp_staging


class SqliteVecQueryMixin:
    """Vector query/stat operations."""

    if TYPE_CHECKING:
        db_path: Path
        archive_root: Path | None
        model: str
        dimension: int
        voyage_key: str | None

        def _lifecycle_admission(self) -> AbstractContextManager[None]: ...

        def _ensure_vec_available(self) -> None: ...

        def _ensure_tables(self) -> None: ...

        def _get_embeddings(
            self,
            texts: list[str],
            input_type: str = "document",
        ) -> list[list[float]]: ...

        def _get_connection(self) -> sqlite3.Connection: ...

        def _get_read_connection(
            self,
            *,
            index_path: Path | None = None,
            index_connection: sqlite3.Connection | None = None,
            configure_connection: Callable[[sqlite3.Connection], None] | None = None,
        ) -> sqlite3.Connection: ...

        def _release_connection(self, conn: sqlite3.Connection) -> None: ...

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Run the provider route under managed lifecycle admission."""
        if not self.voyage_key:
            raise EmbeddingRetrievalNotReadyError(
                "text retrieval requires embedding acquisition credentials", readiness_status="disabled"
            )
        with self._lifecycle_admission():
            return self._query_unlocked(text, limit)

    def _query_unlocked(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        if limit <= 0:
            return []
        conn = self._get_read_connection()
        try:
            embedding = self._query_vector(conn, text)
            if embedding is None:
                return []
            with closing(conn.cursor()) as cursor:
                cursor.execute(self._distance_sql(session_grain=False), (embedding, max(limit, 0)))
                return [(str(row["message_id"]), float(row["distance"])) for row in cursor]
        finally:
            self._release_connection(conn)

    def query_by_session(self, session_id: str, limit: int = 10) -> list[tuple[str, float]]:
        """Rank occurrences by their closest distance to any stored seed output."""
        with self._lifecycle_admission():
            return self._query_by_session_unlocked(session_id, limit)

    def _query_by_session_unlocked(self, session_id: str, limit: int = 10) -> list[tuple[str, float]]:
        conn = self._get_read_connection()
        try:
            _assert_vec0_dimension(conn, self.dimension)
            self._require_seed(conn, session_id)
            with closing(conn.cursor()) as cursor:
                cursor.execute(
                    self._distance_sql(session_grain=False, session_seed=True),
                    (session_id, session_id, session_id, max(limit, 0)),
                )
                return [(str(row["message_id"]), float(row["distance"])) for row in cursor]
        except sqlite3.Error as exc:
            raise SqliteVecError("stored session vectors could not be read") from exc
        finally:
            self._release_connection(conn)

    def _query_vector(self, conn: sqlite3.Connection, text: str, *, scoped: bool = False) -> bytes | None:
        _assert_vec0_dimension(conn, self.dimension)
        scope = "JOIN scoped_vector_sessions USING (session_id)" if scoped else ""
        with closing(conn.cursor()) as cursor:
            current = cursor.execute(f"SELECT 1 FROM current_embedding_messages {scope} LIMIT 1").fetchone()
        if current is None:
            raise EmbeddingRetrievalNotReadyError(
                "semantic retrieval has no vectors for the current archive messages and recipe",
                readiness_status="empty",
            )
        embeddings = self._get_embeddings([text], input_type="query")
        if not embeddings:
            return None
        return _serialize_f32(embeddings[0])

    @staticmethod
    def _require_seed(conn: sqlite3.Connection, session_id: str) -> None:
        with closing(conn.cursor()) as cursor:
            found = cursor.execute(
                "SELECT 1 FROM current_embedding_messages WHERE session_id = ? LIMIT 1", (session_id,)
            ).fetchone()
        if found is None:
            raise SqliteVecError(
                f"session {session_id!r} has no stored message embeddings; cannot run session-seeded similarity"
            )

    @staticmethod
    def _distance_sql(
        *,
        session_grain: bool,
        session_seed: bool = False,
        scoped: bool = False,
    ) -> str:
        # Scope belongs to occurrences. Shared output addresses are scored once,
        # then restored to every qualifying occurrence before session reduction.
        scope = "JOIN scoped_vector_sessions AS scope ON scope.session_id = r.session_id" if scoped else ""
        seed = (
            """seed_outputs AS MATERIALIZED (
            SELECT DISTINCT me.embedding
            FROM current_embedding_messages r
            JOIN message_embeddings me ON me.vector_derivation_hash = lower(hex(r.vector_derivation_hash))
            WHERE r.session_id = ?
        ),"""
            if session_seed
            else ""
        )
        exclude = "WHERE r.session_id != ?" if session_seed else ""
        score = (
            "MIN(vec_distance_L2(me.embedding, seed.embedding))" if session_seed else "vec_distance_L2(me.embedding, ?)"
        )
        seed_join = "CROSS JOIN seed_outputs seed" if session_seed else ""
        group = "GROUP BY me.vector_derivation_hash" if session_seed else ""
        final = (
            """SELECT message_id, distance FROM witnesses WHERE witness_rank = 1
            ORDER BY distance, session_id, message_id"""
            if session_grain
            else """SELECT message_id, distance FROM occurrences
            ORDER BY distance, message_id LIMIT ?"""
        )
        return f"""WITH {seed}
            eligible_outputs AS MATERIALIZED (
                SELECT DISTINCT r.vector_derivation_hash FROM current_embedding_messages r {scope} {exclude}
            ), distances AS MATERIALIZED (
                SELECT me.vector_derivation_hash, {score} AS distance
                FROM eligible_outputs output
                JOIN message_embeddings me ON me.vector_derivation_hash = lower(hex(output.vector_derivation_hash))
                {seed_join} {group}
            ), occurrences AS (
                SELECT r.message_id, r.session_id, distances.distance
                FROM current_embedding_messages r {scope}
                JOIN distances ON distances.vector_derivation_hash = lower(hex(r.vector_derivation_hash))
                {exclude}
            ), witnesses AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY distance, message_id) AS witness_rank
                FROM occurrences
            ) {final}"""

    @contextmanager
    def scoped_query(
        self,
        session_ids: Iterable[str],
        *,
        index_connection: sqlite3.Connection,
        configure_connection: Callable[[sqlite3.Connection], None],
        check_cancelled: Callable[[], None],
        text: str | None = None,
        seed_session_id: str | None = None,
    ) -> Iterator[ScopedVectorQuery]:
        """Traverse an exact full-scope session ranking under one owned cursor."""
        if (text is None) == (seed_session_id is None):
            raise ValueError("scoped vector query requires exactly one seed")
        if text is not None and not self.voyage_key:
            raise EmbeddingRetrievalNotReadyError(
                "text retrieval requires embedding acquisition credentials", readiness_status="disabled"
            )
        check_cancelled()
        with self._lifecycle_admission():
            conn = self._get_read_connection(
                index_connection=index_connection,
                configure_connection=configure_connection,
            )
            try:
                with readonly_temp_staging(conn), closing(conn.cursor()) as staging:
                    staging.execute("DROP TABLE IF EXISTS temp.scoped_vector_sessions")
                    staging.execute("CREATE TEMP TABLE scoped_vector_sessions (session_id TEXT PRIMARY KEY)")
                    staging.executemany(
                        "INSERT OR IGNORE INTO scoped_vector_sessions VALUES (?)", ((sid,) for sid in session_ids)
                    )
                check_cancelled()
                with closing(conn.cursor()) as cursor:
                    if seed_session_id is not None:
                        with closing(index_connection.cursor()) as seed_cursor:
                            if (
                                seed_cursor.execute(
                                    "SELECT 1 FROM sessions WHERE session_id = ?", (seed_session_id,)
                                ).fetchone()
                                is None
                            ):
                                raise SessionNotFoundError(seed_session_id)
                        _assert_vec0_dimension(conn, self.dimension)
                        self._require_seed(conn, seed_session_id)
                    if cursor.execute("SELECT 1 FROM scoped_vector_sessions LIMIT 1").fetchone() is None:
                        yield ScopedVectorQuery(rows=iter(()))
                        return
                    args: tuple[object, ...]
                    if seed_session_id is not None:
                        args = (seed_session_id, seed_session_id, seed_session_id)
                    else:
                        assert text is not None
                        query_vector = self._query_vector(conn, text, scoped=True)
                        if query_vector is None:
                            raise SqliteVecError("query embedding provider returned no vector")
                        args = (query_vector,)
                    cursor.execute(
                        self._distance_sql(session_grain=True, session_seed=seed_session_id is not None, scoped=True),
                        args,
                    )

                    def rows() -> Iterator[tuple[str, float]]:
                        while batch := cursor.fetchmany(200):
                            check_cancelled()
                            yield from ((str(row["message_id"]), float(row["distance"])) for row in batch)

                    yield ScopedVectorQuery(rows=rows())
            finally:
                # Cleanup is not query work. Suspend only this connection's
                # handler while dropping our TEMP scope, then restore the
                # caller's existing guard before releasing its borrowed frame.
                conn.set_progress_handler(None, 0)
                try:
                    with readonly_temp_staging(conn), closing(conn.cursor()) as cursor:
                        cursor.execute("DROP TABLE IF EXISTS temp.scoped_vector_sessions")
                finally:
                    try:
                        configure_connection(conn)
                    finally:
                        self._release_connection(conn)

    def count_session_embeddings(self, session_id: str) -> int:
        """Run the provider route under managed lifecycle admission."""
        with self._lifecycle_admission():
            return self._count_session_embeddings_unlocked(session_id)

    def _count_session_embeddings_unlocked(self, session_id: str) -> int:
        """Return the number of distinct stored vectors for ``session_id``.

        Counted from the stored refs joined to vector metadata, not from the
        eligible-message projection: eligibility says a message *should* be
        embedded, and a session mid-materialization or with failed embeds
        would otherwise report vectors it does not have.
        """
        conn = self._get_read_connection()
        try:
            try:
                with closing(conn.cursor()) as cursor:
                    row = cursor.execute(
                        """
                    SELECT COUNT(DISTINCT r.vector_derivation_hash) AS count
                    FROM message_embedding_refs AS r
                    JOIN message_embeddings_meta AS m
                      ON m.vector_derivation_hash = r.vector_derivation_hash
                    WHERE r.session_id = ?
                    """,
                        (session_id,),
                    ).fetchone()
            except sqlite3.OperationalError as exc:
                raise SqliteVecError("stored session vectors could not be read") from exc
            return int(row["count"]) if row is not None else 0
        except sqlite3.Error as exc:
            raise SqliteVecError("stored session vectors could not be read") from exc
        finally:
            self._release_connection(conn)


__all__ = ["SqliteVecQueryMixin"]
