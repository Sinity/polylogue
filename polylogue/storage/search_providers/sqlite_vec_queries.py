"""Query and upsert operations for the sqlite-vec provider."""

from __future__ import annotations

import httpx

from polylogue.storage.embedding_stats import read_embedding_stats_sync
from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError, _serialize_f32, logger


class SqliteVecQueryMixin:
    """Vector upsert/query/stat operations."""

    def upsert(self, conversation_id: str, messages) -> None:
        """Upsert message embeddings into the vector store."""
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
            logger.error("Failed to generate embeddings for %s: %s", conversation_id, exc)
            raise

        conn = self._get_connection()
        try:
            provider_name = "unknown"
            row = conn.execute(
                "SELECT provider_name FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row:
                provider_name = row[0] or "unknown"

            for msg, embedding in zip(embeddable, embeddings, strict=True):
                embedding_blob = _serialize_f32(embedding)
                conn.execute(
                    "DELETE FROM message_embeddings WHERE message_id = ?",
                    (msg.message_id,),
                )
                conn.execute(
                    """
                    INSERT INTO message_embeddings (message_id, embedding, provider_name, conversation_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (msg.message_id, embedding_blob, provider_name, msg.conversation_id),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings_meta (
                        target_id, target_type, model, dimension, embedded_at, content_hash
                    ) VALUES (?, 'message', ?, ?, datetime('now'), ?)
                    """,
                    (
                        msg.message_id,
                        self.model,
                        self.dimension,
                        msg.content_hash if hasattr(msg, "content_hash") else None,
                    ),
                )

            conn.execute(
                """
                INSERT INTO embedding_status (
                    conversation_id, message_count_embedded, last_embedded_at, needs_reindex
                ) VALUES (?, ?, datetime('now'), 0)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    message_count_embedded = excluded.message_count_embedded,
                    last_embedded_at = excluded.last_embedded_at,
                    needs_reindex = 0,
                    error_message = NULL
                """,
                (conversation_id, len(embeddable)),
            )
            conn.commit()
        finally:
            conn.close()

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Find semantically similar messages."""
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
                SELECT message_id, distance
                FROM message_embeddings
                WHERE embedding MATCH ?
                  AND k = ?
                ORDER BY distance
                """,
                (query_embedding, limit),
            ).fetchall()
            return [(row["message_id"], row["distance"]) for row in rows]
        finally:
            conn.close()

    def query_by_provider(
        self,
        text: str,
        provider: str,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Find semantically similar messages filtered by provider."""
        self._ensure_vec_available()

        embeddings = self._get_embeddings([text], input_type="query")
        if not embeddings:
            return []

        query_embedding = _serialize_f32(embeddings[0])

        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT message_id, distance
                FROM message_embeddings
                WHERE embedding MATCH ?
                  AND k = ?
                  AND provider_name = ?
                ORDER BY distance
                """,
                (query_embedding, limit, provider),
            ).fetchall()
            return [(row["message_id"], row["distance"]) for row in rows]
        finally:
            conn.close()

    def get_embedding_stats(self) -> dict[str, int]:
        """Get embedding statistics."""
        conn = self._get_connection()
        try:
            embedding_stats = read_embedding_stats_sync(conn)
            return {
                "embedded_messages": embedding_stats.embedded_messages,
                "pending_conversations": embedding_stats.pending_conversations,
            }
        finally:
            conn.close()


__all__ = ["SqliteVecQueryMixin"]
