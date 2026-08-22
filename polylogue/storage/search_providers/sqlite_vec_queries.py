"""Query and upsert operations for the sqlite-vec provider."""

from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from polylogue.storage.embeddings.embedding_stats import read_embedding_stats_sync
from polylogue.storage.runtime import MessageRecord
from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError, _serialize_f32, logger

# Per-seed-message neighbor fanout used to grow the candidate pool before
# deduplicating to messages. Bounds the number of MATCH queries issued for a
# large seed session to a fixed, representative sample.
_SESSION_SEED_FANOUT = 20


class SqliteVecQueryMixin:
    """Vector upsert/query/stat operations."""

    if TYPE_CHECKING:
        db_path: Path
        archive_root: Path | None
        _legacy_compatibility: bool
        model: str
        dimension: int

        def _lifecycle_admission(self) -> AbstractContextManager[None]: ...

        def _ensure_vec_available(self) -> None: ...

        def _ensure_tables(self) -> None: ...

        def _should_embed_message(self, msg: MessageRecord) -> bool: ...

        def _get_embeddings(
            self,
            texts: list[str],
            input_type: str = "document",
        ) -> list[list[float]]: ...

        def _get_connection(self) -> sqlite3.Connection: ...

    def upsert(self, session_id: str, messages: list[MessageRecord], *, origin: str | None = None) -> None:
        """Upsert embeddings while holding managed lifecycle admission.

        ``embeddings.db`` is a split tier and intentionally has no ``sessions``
        table.  Callers that know the archive origin should supply it; the
        message source is the compatibility fallback for older callers.
        """
        if getattr(self, "_legacy_compatibility", False) or self.db_path.name != "embeddings.db":
            self._upsert_unlocked(session_id, messages, origin=origin)
            return
        with self._lifecycle_admission():
            self._upsert_unlocked(session_id, messages, origin=origin)

    def _upsert_unlocked(self, session_id: str, messages: list[MessageRecord], *, origin: str | None = None) -> None:
        """Upsert message embeddings into the vector store.

        Delegates the actual write to the canonical content-addressed
        primitives (:mod:`polylogue.storage.sqlite.archive_tiers.
        embedding_write`) instead of hand-rolled SQL, so this provider's
        vectors are keyed and deduped exactly like the daemon catch-up path
        (``embed_archive_session_sync``) that shares the same ``embeddings.db``
        file (polylogue-q88p).
        """
        if not messages:
            return

        self._ensure_vec_available()
        self._ensure_tables()

        embeddable = [msg for msg in messages if self._should_embed_message(msg)]
        if not embeddable:
            return

        texts = [msg.text for msg in embeddable if msg.text]

        try:
            embeddings = self._get_embeddings(texts, input_type="document")
        except (SqliteVecError, httpx.HTTPError) as exc:
            logger.error("Failed to generate embeddings for %s: %s", session_id, exc)
            raise

        from datetime import UTC, datetime

        from polylogue.storage.embeddings.identity import embedding_input_hash
        from polylogue.storage.sqlite.archive_tiers.embedding_write import (
            ArchiveEmbeddingWrite,
            upsert_message_embeddings,
        )

        conn = self._get_connection()
        try:
            message_origin = origin
            if message_origin is None:
                message_origin = next(
                    (msg.source_name.strip() for msg in embeddable if msg.source_name and msg.source_name.strip()),
                    None,
                )
            # Legacy arbitrary-path providers historically carried a sessions
            # table beside vectors. Keep that compatibility fallback isolated;
            # canonical split embeddings.db never queries its own absent table.
            if message_origin is None and self.db_path.name != "embeddings.db":
                row = conn.execute(
                    "SELECT origin FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row:
                    message_origin = row[0] or message_origin
            message_origin = message_origin or "unknown"
            now_ms = int(datetime.now(UTC).timestamp() * 1000)
            writes = [
                ArchiveEmbeddingWrite(
                    message_id=msg.message_id,
                    session_id=msg.session_id,
                    origin=message_origin,
                    embedding=embedding,
                    model=self.model,
                    embedded_at_ms=now_ms,
                    embedding_input_hash=embedding_input_hash(model=self.model, input_text=str(msg.text)),
                )
                for msg, embedding in zip(embeddable, embeddings, strict=True)
            ]
            upsert_message_embeddings(conn, writes)

            conn.execute(
                """
                INSERT INTO embedding_status (
                    session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
                ) VALUES (?, ?, ?, ?, 0, NULL)
                ON CONFLICT(session_id) DO UPDATE SET
                    origin = excluded.origin,
                    message_count_embedded = excluded.message_count_embedded,
                    last_embedded_at_ms = excluded.last_embedded_at_ms,
                    needs_reindex = 0,
                    error_message = NULL
                """,
                (session_id, message_origin, len(embeddable), now_ms),
            )
            conn.commit()
        finally:
            conn.close()

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Run the provider route under managed lifecycle admission."""
        with self._lifecycle_admission():
            return self._query_unlocked(text, limit)

    def _query_unlocked(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Find semantically similar messages.

        ``message_embeddings`` is keyed by ``embedding_input_hash`` (content-
        addressed, deduped), not ``message_id``: a MATCH hit is resolved back
        to every message currently referencing that hash via
        ``message_embedding_refs``, so identical content shared across
        sessions surfaces every one of its messages, not just one arbitrary
        representative.
        """
        self._ensure_vec_available()
        self._ensure_tables()

        embeddings = self._get_embeddings([text], input_type="query")
        if not embeddings:
            return []

        query_embedding = _serialize_f32(embeddings[0])

        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT r.message_id AS message_id, hits.distance AS distance
                FROM (
                    SELECT embedding_input_hash, distance
                    FROM message_embeddings
                    WHERE embedding MATCH ?
                      AND k = ?
                ) AS hits
                JOIN message_embedding_refs AS r
                  ON lower(hex(r.embedding_input_hash)) = hits.embedding_input_hash
                ORDER BY hits.distance
                """,
                (query_embedding, limit),
            ).fetchall()
            return [(row["message_id"], row["distance"]) for row in rows]
        finally:
            conn.close()

    def query_by_session(self, session_id: str, limit: int = 10) -> list[tuple[str, float]]:
        """Run the provider route under managed lifecycle admission."""
        with self._lifecycle_admission():
            return self._query_by_session_unlocked(session_id, limit)

    def _query_by_session_unlocked(self, session_id: str, limit: int = 10) -> list[tuple[str, float]]:
        """Rank messages by similarity to a stored session's own embeddings.

        Fetches every stored vector for ``session_id`` and KNN-searches a bounded,
        representative sample of them against the store. Hits belonging to the seed
        session itself are dropped so the seed never ranks against its own messages;
        each surviving message keeps its closest (smallest) distance across the seed
        vectors. Returns ranked ``(message_id, distance)`` ascending by distance.

        Raises :class:`SqliteVecError` when the seed session has no stored
        embeddings (including when the vector table does not exist yet) so the caller
        fails typed rather than returning an empty/unfiltered listing.
        """
        self._ensure_vec_available()

        conn = self._get_connection()
        try:
            seed_rows: list[sqlite3.Row] = []
            try:
                seed_rows = conn.execute(
                    """
                    SELECT DISTINCT me.embedding_input_hash AS embedding_input_hash, me.embedding AS embedding
                    FROM message_embedding_refs AS r
                    JOIN message_embeddings AS me
                      ON lower(hex(r.embedding_input_hash)) = me.embedding_input_hash
                    WHERE r.session_id = ?
                    """,
                    (session_id,),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                raise SqliteVecError(f"session {session_id!r} has no stored embeddings: {exc}") from exc
            if not seed_rows:
                raise SqliteVecError(
                    f"session {session_id!r} has no stored message embeddings; cannot run session-seeded similarity"
                )

            k = max(limit, 1)
            best_distance: dict[str, float] = {}
            for seed_row in seed_rows[:_SESSION_SEED_FANOUT]:
                embedding_blob = bytes(seed_row["embedding"])
                neighbors = conn.execute(
                    """
                    SELECT r.message_id AS message_id, r.session_id AS session_id, hits.distance AS distance
                    FROM (
                        SELECT embedding_input_hash, distance
                        FROM message_embeddings
                        WHERE embedding MATCH ?
                          AND k = ?
                    ) AS hits
                    JOIN message_embedding_refs AS r
                      ON lower(hex(r.embedding_input_hash)) = hits.embedding_input_hash
                    ORDER BY hits.distance
                    """,
                    (embedding_blob, k),
                ).fetchall()
                for neighbor in neighbors:
                    if str(neighbor["session_id"]) == session_id:
                        continue
                    message_id = str(neighbor["message_id"])
                    distance = float(neighbor["distance"])
                    if message_id not in best_distance or distance < best_distance[message_id]:
                        best_distance[message_id] = distance

            ranked = sorted(best_distance.items(), key=lambda item: (item[1], item[0]))
            return ranked[:limit]
        finally:
            conn.close()

    def count_session_embeddings(self, session_id: str) -> int:
        """Run the provider route under managed lifecycle admission."""
        with self._lifecycle_admission():
            return self._count_session_embeddings_unlocked(session_id)

    def _count_session_embeddings_unlocked(self, session_id: str) -> int:
        """Return the number of distinct stored vectors for ``session_id``."""
        self._ensure_vec_available()
        conn = self._get_connection()
        try:
            try:
                row = conn.execute(
                    """
                    SELECT COUNT(DISTINCT embedding_input_hash) AS count
                    FROM message_embedding_refs
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
            except sqlite3.OperationalError as exc:
                raise SqliteVecError(f"session {session_id!r} has no stored embeddings: {exc}") from exc
            return int(row["count"]) if row is not None else 0
        finally:
            conn.close()

    def query_by_provider(
        self,
        text: str,
        provider: str,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Run the provider route under managed lifecycle admission."""
        with self._lifecycle_admission():
            return self._query_by_provider_unlocked(text, provider, limit)

    def _query_by_provider_unlocked(
        self,
        text: str,
        provider: str,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Find semantically similar messages filtered by provider (origin).

        ``message_embeddings`` no longer carries a per-vector origin column
        (it is content-addressed and shared across origins); the filter is
        applied post-join against ``message_embedding_refs.origin`` instead.
        Over-fetches the KNN candidate pool so a narrow origin filter still
        has enough candidates to fill ``limit`` after filtering.
        """
        self._ensure_vec_available()

        embeddings = self._get_embeddings([text], input_type="query")
        if not embeddings:
            return []

        query_embedding = _serialize_f32(embeddings[0])
        fanout_k = max(limit * 5, limit, 1)

        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT r.message_id AS message_id, hits.distance AS distance
                FROM (
                    SELECT embedding_input_hash, distance
                    FROM message_embeddings
                    WHERE embedding MATCH ?
                      AND k = ?
                ) AS hits
                JOIN message_embedding_refs AS r
                  ON lower(hex(r.embedding_input_hash)) = hits.embedding_input_hash
                WHERE r.origin = ?
                ORDER BY hits.distance
                LIMIT ?
                """,
                (query_embedding, fanout_k, provider, limit),
            ).fetchall()
            return [(row["message_id"], row["distance"]) for row in rows]
        finally:
            conn.close()

    def get_embedding_stats(self) -> dict[str, int]:
        """Run the provider route under managed lifecycle admission."""
        with self._lifecycle_admission():
            return self._get_embedding_stats_unlocked()

    def _get_embedding_stats_unlocked(self) -> dict[str, int]:
        """Get embedding statistics."""
        conn = self._get_connection()
        try:
            embedding_stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)
            return {
                "embedded_messages": embedding_stats.embedded_messages,
                "pending_sessions": embedding_stats.pending_sessions,
            }
        finally:
            conn.close()


__all__ = ["SqliteVecQueryMixin"]
