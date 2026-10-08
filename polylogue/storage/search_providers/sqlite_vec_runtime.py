"""Runtime/capability helpers for the sqlite-vec provider."""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.core.compute import current_cancellation
from polylogue.core.errors import SchemaSkewError, SessionNotFoundError
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.embeddings.identity import (
    EmbeddingRecipe,
    register_embedding_identity_sql,
    retained_embedding_predicate,
)
from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError, SqliteVecUnavailableError, logger
from polylogue.storage.sqlite.connection_profile import (
    READ_CONNECTION_PROFILE,
    GenerationToken,
    attach_readonly_database,
    open_connection,
    open_readonly_connection,
    readonly_temp_staging,
)
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec


@contextmanager
def _vector_projection_errors() -> Iterator[None]:
    """Translate SQLite acquisition/projection failure at the provider seam."""
    try:
        yield
    except sqlite3.Error as exc:
        raise SqliteVecError("embedding projection could not bind its declared databases") from exc


def _configure_current_embedding_messages(
    conn: sqlite3.Connection,
    *,
    index_path: Path | None = None,
    recipe: EmbeddingRecipe,
    attach_index: bool = True,
    register_identity: bool = True,
    index_connection: sqlite3.Connection | None = None,
) -> None:
    """Bind the current-index projection on an already-open vector reader.

    Current occurrences can use proven retained producers in the selected
    retrieval space, or an exact selected-request output awaiting a binding.
    Neither route rewrites a producer identity or certifies binding completion.
    """

    if index_connection is not None:
        with closing(conn.cursor()) as cursor:
            cursor.execute("DROP TABLE IF EXISTS temp.current_embedding_messages")
            cursor.execute("DROP TABLE IF EXISTS temp.current_embedding_inputs")
    if register_identity:
        register_embedding_identity_sql(conn, recipe=recipe)
    if attach_index:
        if index_path is None:
            raise ValueError("embedding projection attachment requires an explicit index path")
        with closing(conn.cursor()) as cursor:
            cursor.execute("ATTACH DATABASE ? AS archive_index", (str(index_path),))
    with closing(conn.cursor()) as cursor:
        cursor.execute(
            """
        CREATE TEMP TABLE current_embedding_messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            origin TEXT NOT NULL,
            vector_derivation_hash BLOB NOT NULL
        );
        """
        )
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        raise SqliteVecUnavailableError(f"sqlite-vec extension failed to load: {error}")
    valid = retained_embedding_predicate(
        recipe=recipe,
        source="eligible",
        refs="r",
        meta="meta",
        vectors_table="message_embeddings",
    )
    address = f"CASE WHEN COALESCE({valid}, 0) THEN r.vector_derivation_hash ELSE eligible.vector_derivation_hash END"
    index_schema = "archive_index." if index_connection is None else ""
    input_sql = f"""
SELECT m.message_id, m.session_id, s.origin, m.content_hash,
                   (
                       SELECT GROUP_CONCAT(prose.text, char(10) || char(10))
                       FROM (
                           SELECT b.text
                           FROM {index_schema}blocks b
                           WHERE b.message_id = m.message_id
                             AND b.block_type = 'text'
                             AND b.text IS NOT NULL
                           ORDER BY b.position
                       ) AS prose
                   ) AS text
            FROM {index_schema}messages m
            JOIN {index_schema}sessions s ON s.session_id = m.session_id
            WHERE m.message_type = 'message'
              AND m.role IN ('user', 'assistant')
              AND m.material_origin IN ('human_authored', 'assistant_authored')
              AND m.word_count > 0
    """
    prose_relation = f"FROM ({input_sql}) AS prose_source"
    if index_connection is not None:
        # The archive owner lends the exact already-pinned evidence frame.
        # Only this provider's TEMP copy and cursors belong to this operation.
        with closing(conn.cursor()) as destination, closing(index_connection.cursor()) as source:
            destination.execute(
                "CREATE TEMP TABLE current_embedding_inputs ("
                "message_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, origin TEXT NOT NULL, "
                "content_hash BLOB, text TEXT)"
            )
            source.execute(input_sql)
            destination.executemany("INSERT INTO current_embedding_inputs VALUES (?, ?, ?, ?, ?)", source)
        prose_relation = "FROM current_embedding_inputs AS prose_source"
    with closing(conn.cursor()) as cursor:
        cursor.execute(
            f"""
        INSERT INTO current_embedding_messages (message_id, session_id, origin, vector_derivation_hash)
        SELECT eligible.message_id, eligible.session_id, eligible.origin, {address}
        FROM (
          SELECT prose_source.*, polylogue_vector_derivation_hash(X'{recipe.recipe_hash.hex()}', prose_source.text) AS vector_derivation_hash
          {prose_relation}
        ) AS eligible
        LEFT JOIN message_embedding_refs AS r ON r.message_id = eligible.message_id
        LEFT JOIN message_embeddings_meta AS meta ON meta.vector_derivation_hash = r.vector_derivation_hash
        JOIN message_embeddings_meta AS chosen_meta ON chosen_meta.vector_derivation_hash = {address}
        JOIN message_embeddings AS chosen_vector ON chosen_vector.vector_derivation_hash = lower(hex(chosen_meta.vector_derivation_hash))
        WHERE LENGTH(TRIM(COALESCE(eligible.text, ''))) >= 20
          AND polylogue_embedding_output_matches(
              X'{recipe.recipe_hash.hex()}', chosen_meta.model, chosen_meta.dimension,
              chosen_meta.recipe_hash, chosen_meta.output_contract_hash,
              chosen_meta.vector_derivation_hash, eligible.text
          )
        """
        )


def require_vector_seed_session(connection: sqlite3.Connection, session_id: str) -> None:
    """Require the seed on the same selected index handle used for vector reads."""
    with closing(connection.cursor()) as cursor:
        if (
            cursor.execute("SELECT 1 FROM archive_index.sessions WHERE session_id = ?", (session_id,)).fetchone()
            is None
        ):
            raise SessionNotFoundError(session_id)


def _vector_snapshot_binding(connection: sqlite3.Connection) -> tuple[Path, GenerationToken, int]:
    """Read index and thread proof retained by the existing snapshot owner."""
    binding = getattr(connection, "_polylogue_vector_read_snapshot_binding", None)
    if (
        not isinstance(binding, tuple)
        or len(binding) != 3
        or not isinstance(binding[0], Path)
        or not isinstance(binding[1], GenerationToken)
        or not isinstance(binding[2], int)
    ):
        raise SqliteVecError("vector snapshot lacks its owner's selected index proof")
    return binding[0], binding[1], binding[2]


def open_vector_read_snapshot(
    *,
    embeddings_path: Path,
    index_path: Path,
    recipe: EmbeddingRecipe,
    configure_connection: Callable[[sqlite3.Connection], None] | None = None,
    defer_projection: bool = False,
    index_connection: sqlite3.Connection | None = None,
) -> sqlite3.Connection:
    """Open one explicit, read-only vector snapshot for a pinned operation.

    Callers choose and hold both paths under their publication barrier before
    calling this function.  No configured root, active-generation resolver,
    or provider default participates here. The returned handle retains the
    selected index path, generation and creating thread; provider construction consumes
    that proof instead of certifying a held handle by its later pathname.
    With a lent Index frame, only Embeddings is opened and pinned; the transient
    handle cannot be exported as a standalone Index-bound snapshot.
    """

    selected_index = index_path.absolute()
    selected_generation: GenerationToken | None = None
    if index_connection is None:
        if not index_path.is_file():
            raise SqliteVecError(f"pinned vector snapshot found no index at {index_path}")
        selected_index = index_path.resolve(strict=True)
        selected_stat = selected_index.stat()
        selected_generation = GenerationToken(device=selected_stat.st_dev, inode=selected_stat.st_ino)
    opening_thread_id = threading.get_ident()
    with _vector_projection_errors():
        conn = open_readonly_connection(
            embeddings_path,
            validate_schema=False,
            profile=replace(READ_CONNECTION_PROFILE, temp_store="FILE"),
        )
    conn.row_factory = sqlite3.Row
    try:
        if configure_connection is not None:
            configure_connection(conn)
        with _vector_projection_errors():
            loaded, error = try_load_sqlite_vec(conn)
            if not loaded:
                raise SqliteVecUnavailableError(f"sqlite-vec extension failed to load: {error or 'unknown error'}")
            register_embedding_identity_sql(conn, recipe=recipe)
            # A lent canonical frame is the only Index evidence source. An
            # ordinary scoped reader cannot certify a later pathname or export
            # a standalone Index binding; constructor admission therefore
            # refuses this transient handle as a supplied operation snapshot.
            if index_connection is None:
                attach_readonly_database(conn, selected_index, alias="archive_index")
            attached: list[sqlite3.Row] = []
            with closing(conn.cursor()) as cursor:
                cursor.execute("BEGIN")
                cursor.execute("SELECT rootpage FROM main.sqlite_schema LIMIT 1").fetchone()
                if index_connection is None:
                    cursor.execute("SELECT rootpage FROM archive_index.sqlite_schema LIMIT 1").fetchone()
                    attached = cursor.execute("PRAGMA database_list").fetchall()
            if index_connection is None:
                attached_index = next((row[2] for row in attached if row[1] == "archive_index"), None)
                current_stat = selected_index.stat()
                current_generation = GenerationToken(device=current_stat.st_dev, inode=current_stat.st_ino)
                if (
                    not attached_index
                    or Path(attached_index).absolute() != selected_index
                    or current_generation != selected_generation
                ):
                    raise SqliteVecError("selected archive index changed while pinning the vector snapshot; retry")
                vars(conn)["_polylogue_vector_read_snapshot_binding"] = (
                    selected_index,
                    selected_generation,
                    opening_thread_id,
                )
            if not defer_projection:
                prepare_vector_read_projection(conn, recipe=recipe, index_connection=index_connection)
        return conn
    except BaseException:
        conn.close()
        raise


def prepare_vector_read_projection(
    connection: sqlite3.Connection,
    *,
    recipe: EmbeddingRecipe,
    index_connection: sqlite3.Connection | None = None,
) -> None:
    """Build the derived lookup after publication exclusion has been released."""
    if not connection.in_transaction:
        raise SqliteVecError("semantic projection requires an already pinned read transaction")
    # Persistent databases remain mode=ro. Only TEMP needs write permission;
    # single-statement execute preserves both pinned read transactions.
    with _vector_projection_errors(), readonly_temp_staging(connection):
        _configure_current_embedding_messages(
            connection,
            recipe=recipe,
            attach_index=False,
            register_identity=False,
            index_connection=index_connection,
        )


class SqliteVecRuntimeMixin:
    """Connection, capability, and table-management helpers."""

    if TYPE_CHECKING:
        db_path: Path
        model: str
        dimension: int
        _vec_available: bool | None
        _tables_ensured: bool
        archive_root: Path | None
        _admitted_db_identity: tuple[int, int] | None
        _snapshot_connection: sqlite3.Connection | None
        _snapshot_index_path: Path
        _snapshot_index_identity: GenerationToken

    def _assert_lifecycle_binding(self) -> None:
        if self._snapshot_connection is not None:
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
        if self._snapshot_connection is not None:
            yield
            return
        self._assert_lifecycle_binding()
        try:
            yield
        finally:
            self._admitted_db_identity = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get connection with sqlite-vec extension loaded if available."""
        if self._snapshot_connection is not None:
            return self._snapshot_connection
        self._assert_lifecycle_binding()
        db = self.db_path.resolve(strict=False)
        conn = open_connection(db, archive_root=db.parent)
        conn.row_factory = sqlite3.Row

        if self.archive_root is not None:
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
                    with _vector_projection_errors():
                        _configure_current_embedding_messages(
                            conn,
                            index_path=index_path,
                            recipe=EmbeddingRecipe.current(model=self.model, dimensions=self.dimension),
                        )
                except BaseException:
                    conn.close()
                    raise

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
                    raise SqliteVecUnavailableError("sqlite-vec extension failed to load on connection: unknown error")
                raise SqliteVecUnavailableError(
                    f"sqlite-vec extension failed to load on connection: {error}"
                ) from error

        return conn

    def _get_read_connection(
        self,
        *,
        index_path: Path | None = None,
        index_connection: sqlite3.Connection | None = None,
        configure_connection: Callable[[sqlite3.Connection], None] | None = None,
    ) -> sqlite3.Connection:
        """Read retained vectors without acquiring a writable tier handle."""
        if self._snapshot_connection is not None:
            if index_path is not None:
                selected = index_path.resolve(strict=True)
                stat = selected.stat()
                if (
                    selected != self._snapshot_index_path
                    or GenerationToken(device=stat.st_dev, inode=stat.st_ino) != self._snapshot_index_identity
                ):
                    raise SqliteVecError("operation vector snapshot does not match the requested archive index")
            if index_connection is not None:
                if (
                    getattr(self._snapshot_connection, "_polylogue_vector_read_canonical_connection", None)
                    is not index_connection
                ):
                    raise SqliteVecError("canonical archive frame does not match the operation vector snapshot")
                prepare_vector_read_projection(
                    self._snapshot_connection,
                    recipe=EmbeddingRecipe.current(model=self.model, dimensions=self.dimension),
                    index_connection=index_connection,
                )
            return self._snapshot_connection
        self._assert_lifecycle_binding()
        assert self.archive_root is not None
        if index_path is None and index_connection is not None:
            with closing(index_connection.cursor()) as cursor:
                selected_file = next(
                    (str(row[2]) for row in cursor.execute("PRAGMA database_list") if row[1] == "main"),
                    None,
                )
            if not selected_file:
                raise SqliteVecError("canonical archive frame has no selected index file")
            selected = Path(selected_file)
        else:
            selected = index_path if index_path is not None else resolve_active_index_path(self.archive_root)
        try:
            # SQLite reports the canonical frame's selected filename. Namespace
            # admission uses that retained name without re-resolving the Index
            # pathname to certify a different incarnation after publication.
            admitted_name = selected.absolute() if index_connection is not None else selected.resolve(strict=True)
            admitted_name.relative_to(self.archive_root.resolve(strict=True))
        except ValueError as exc:
            raise SqliteVecError("selected index is outside the provider's trusted archive root") from exc
        cancellation = current_cancellation()

        def configure_owned(connection: sqlite3.Connection) -> None:
            if cancellation is not None:
                cancellation.register_connection(connection)
            if configure_connection is not None:
                configure_connection(connection)

        return open_vector_read_snapshot(
            configure_connection=configure_owned,
            embeddings_path=self.db_path,
            index_path=selected,
            recipe=EmbeddingRecipe.current(model=self.model, dimensions=self.dimension),
            index_connection=index_connection,
        )

    def _release_connection(self, conn: sqlite3.Connection) -> None:
        """Release ordinary provider handles without closing an operation snapshot."""

        if conn is not self._snapshot_connection:
            conn.close()
            cancellation = current_cancellation()
            if cancellation is not None:
                cancellation.unregister_connection(conn)

    def _ensure_vec_available(self) -> None:
        """Ensure sqlite-vec is available, raising error if not."""
        if self._vec_available is None:
            conn = self._get_connection()
            self._release_connection(conn)
        if not self._vec_available:
            raise SqliteVecUnavailableError("sqlite-vec extension not available. Install with: pip install sqlite-vec")

    def _ensure_tables(self) -> None:
        """Create required tables under lifecycle admission for managed tiers."""
        if self._snapshot_connection is not None:
            return
        self._ensure_tables_unlocked()

    def _ensure_tables_unlocked(self) -> None:
        """Create required vector and metadata tables if they don't exist.

        Detects dimension mismatches between the configured dimension and the
        existing vec0 table and refuses with a typed
        :class:`~polylogue.core.errors.SchemaSkewError`. This runs on every
        semantic query, so it must never discard the stored vectors.

        Uses the canonical archive_tiers DDL (:mod:`polylogue.storage.sqlite.
        archive_tiers.embeddings`) rather than a duplicate hand-rolled schema
        so the tables match what the archive_tiers bootstrap and the session
        embedding route write to the same ``embeddings.db``.
        """
        conn = self._get_connection()
        try:
            # Detect and handle dimension mismatch before creating tables
            _assert_vec0_dimension(conn, self.dimension)

            from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_DDL

            conn.executescript(EMBEDDINGS_DDL)
            conn.commit()
            self._tables_ensured = True
        finally:
            self._release_connection(conn)


def _vec0_table_dimension(conn: sqlite3.Connection) -> int | None:
    """Read the dimension of the existing vec0 table, if it exists."""
    try:
        with closing(conn.cursor()) as cursor:
            has_table = cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='message_embeddings'"
            ).fetchone()
            if has_table is None:
                return None
            # SQLite vec0 stores dimension in the CREATE VIRTUAL TABLE DDL.
            ddl_row = cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='message_embeddings'"
            ).fetchone()
        if ddl_row is None or ddl_row["sql"] is None:
            return None
        import re

        match = re.search(r"float\[(\d+)\]", str(ddl_row["sql"]))
        return int(match.group(1)) if match else None
    except (TypeError, ValueError):
        return None


def _assert_vec0_dimension(conn: sqlite3.Connection, configured_dimension: int) -> None:
    """Refuse to serve a vec0 table whose dimension differs from the configured one.

    ``embeddings.db`` is the expensive-to-rebuild tier, and this runs from
    :meth:`SqliteVecQueryMixin.query` -- a read. A read must never destroy the
    stored vectors, so a dimension mismatch is a typed refusal naming both
    dimensions, not a silent ``DROP TABLE``. Discarding the vectors remains
    available, but only through the explicit operator route
    :func:`drop_vec0_for_dimension_change`.
    """

    current = _vec0_table_dimension(conn)
    if current is not None and current != configured_dimension:
        logger.warning(
            "vec0 dimension mismatch: stored=%d configured=%d — refusing to serve",
            current,
            configured_dimension,
        )
        raise SchemaSkewError(
            "embeddings",
            configured_dimension,
            current,
            remedy=(
                f"stored vectors are {current}-dimensional but this runtime is configured for "
                f"{configured_dimension}. Restore the configured embedding_dimension to {current} to keep "
                "the existing vectors, or discard them deliberately with "
                "`polylogue.storage.search_providers.sqlite_vec_runtime.drop_vec0_for_dimension_change`."
            ),
        )


def drop_vec0_for_dimension_change(conn: sqlite3.Connection, configured_dimension: int) -> int:
    """Discard stored vectors so the tier can be rebuilt at a new dimension.

    Destructive and explicit: only an operator-initiated maintenance route may
    call this. It also clears the ``message_embeddings_meta`` rows recorded at
    the outgoing dimension, because leaving them behind strands the archive
    with neither vectors nor a working path to recreate them -- the meta table
    carries ``CHECK(dimension = ...)`` at the old value and re-embedding then
    fails outright.

    Returns the dimension that was discarded, or ``configured_dimension`` when
    there was nothing to discard.
    """

    current = _vec0_table_dimension(conn)
    if current is None or current == configured_dimension:
        return configured_dimension
    logger.warning(
        "discarding %d-dimensional vectors for reconfiguration to %d",
        current,
        configured_dimension,
    )
    conn.execute("DROP TABLE IF EXISTS message_embeddings")
    # No metadata table to clear leaves the vector drop standing on its own.
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("DELETE FROM message_embeddings_meta WHERE dimension = ?", (current,))
    conn.commit()
    return current


__all__ = [
    "SqliteVecRuntimeMixin",
    "_assert_vec0_dimension",
    "drop_vec0_for_dimension_change",
    "open_vector_read_snapshot",
]
