"""sqlite-vec vector search provider implementation.

This module provides a VectorProvider implementation using sqlite-vec for
self-contained semantic similarity search with Voyage AI embeddings.

No external server required - vectors stored directly in SQLite.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from polylogue.storage.store import MessageRecord

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
DEFAULT_MODEL = "voyage-4"
DEFAULT_DIMENSION = 1024
BATCH_SIZE = 128  # Voyage API limit per request


class SqliteVecError(RuntimeError):
    """Raised when sqlite-vec operations fail."""

    pass


def _serialize_f32(vector: list[float]) -> bytes:
    """Serialize float vector to binary format for sqlite-vec.

    sqlite-vec expects vectors as little-endian float32 arrays.
    """
    return struct.pack(f"<{len(vector)}f", *vector)


class SqliteVecProvider:
    """VectorProvider implementation using sqlite-vec + Voyage AI embeddings.

    This provider generates semantic embeddings using Voyage AI's API and
    stores them in SQLite using the sqlite-vec extension for KNN search.

    Key features:
    - Self-contained: No external vector database required
    - Portable: Embeddings stored in same SQLite file as conversations
    - Cost-optimized: Skip short messages, batch API calls
    - Configurable: Model selection, dimension reduction via Matryoshka

    Attributes:
        db_path: Path to SQLite database
        voyage_key: Voyage AI API key
        model: Voyage model name (default: voyage-4)
        dimension: Output embedding dimension (default: 1024)
    """

    def __init__(
        self,
        voyage_key: str,
        db_path: Path | None = None,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
    ) -> None:
        """Initialize sqlite-vec provider.

        Args:
            voyage_key: Voyage AI API key
            db_path: Path to SQLite database (uses default if None)
            model: Voyage model name
            dimension: Embedding dimension (for Matryoshka reduction)

        Raises:
            SqliteVecError: If sqlite-vec extension cannot be loaded
        """
        from polylogue.storage.backends.sqlite import default_db_path

        self.db_path = db_path or default_db_path()
        self.voyage_key = voyage_key
        self.model = model
        self.dimension = dimension
        self._vec_available: bool | None = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get connection with sqlite-vec extension loaded if available."""
        import sqlite3

        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row

        # Try to load sqlite-vec
        if self._vec_available is None:
            try:
                import sqlite_vec
                sqlite_vec.load(conn)
                self._vec_available = True
            except ImportError:
                logger.warning("sqlite-vec not installed")
                self._vec_available = False
            except Exception as exc:
                logger.warning("sqlite-vec load failed: %s", exc)
                self._vec_available = False
        elif self._vec_available:
            try:
                import sqlite_vec
                sqlite_vec.load(conn)
            except Exception:
                pass

        return conn

    def _ensure_vec_available(self) -> None:
        """Ensure sqlite-vec is available, raising error if not."""
        conn = self._get_connection()
        try:
            if not self._vec_available:
                raise SqliteVecError(
                    "sqlite-vec extension not available. Install with: pip install sqlite-vec"
                )
        finally:
            conn.close()

    def _get_embeddings(
        self,
        texts: list[str],
        input_type: str = "document",
    ) -> list[list[float]]:
        """Get embeddings from Voyage AI.

        Args:
            texts: List of text strings to embed
            input_type: "query" for search queries, "document" for indexing

        Returns:
            List of embedding vectors

        Raises:
            SqliteVecError: If embedding generation fails
        """
        if not texts:
            return []

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
            reraise=True,
        )
        def _do_request(batch: list[str]) -> list[list[float]]:
            with httpx.Client(timeout=60.0) as client:
                payload: dict[str, object] = {
                    "input": batch,
                    "model": self.model,
                    "input_type": input_type,
                }

                # Add dimension reduction if not default
                if self.dimension != DEFAULT_DIMENSION:
                    payload["output_dimension"] = self.dimension

                response = client.post(
                    VOYAGE_API_URL,
                    headers={"Authorization": f"Bearer {self.voyage_key}"},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]

        # Process in batches
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            try:
                embeddings = _do_request(batch)
                all_embeddings.extend(embeddings)

                # Rate limiting: small sleep between batches
                if i + BATCH_SIZE < len(texts):
                    import time
                    time.sleep(0.1)

            except httpx.HTTPError as exc:
                raise SqliteVecError(f"Embedding generation failed: {exc}") from exc

        return all_embeddings

    def _should_embed_message(self, msg: MessageRecord) -> bool:
        """Determine if a message should be embedded.

        Cost optimization: Skip messages that add noise without semantic value.

        Args:
            msg: Message record to evaluate

        Returns:
            True if message should be embedded
        """
        # Skip empty messages
        if not msg.text or not msg.text.strip():
            return False

        # Skip very short messages (< 20 chars)
        if len(msg.text.strip()) < 20:
            return False

        # Skip system messages (usually boilerplate)
        if msg.role == "system":
            return False

        # Skip tool results that are just status messages
        if msg.role == "tool_result":
            text = msg.text.strip().lower()
            if text in ("ok", "success", "done", "error", "failed"):
                return False

        return True

    def upsert(self, conversation_id: str, messages: list[MessageRecord]) -> None:
        """Upsert message embeddings into the vector store.

        Args:
            conversation_id: ID of the conversation
            messages: Messages to embed and store

        Raises:
            SqliteVecError: If embedding or storage fails
        """
        if not messages:
            return

        self._ensure_vec_available()

        # Filter to embeddable messages
        embeddable = [msg for msg in messages if self._should_embed_message(msg)]
        if not embeddable:
            return

        texts = [msg.text for msg in embeddable if msg.text]

        try:
            embeddings = self._get_embeddings(texts, input_type="document")
        except Exception as exc:
            logger.error("Failed to generate embeddings for %s: %s", conversation_id, exc)
            raise

        conn = self._get_connection()
        try:
            # Get provider name from first message metadata
            provider_name = "unknown"
            if embeddable[0].provider_meta:
                pname = embeddable[0].provider_meta.get("provider_name")
                if isinstance(pname, str):
                    provider_name = pname

            for msg, embedding in zip(embeddable, embeddings, strict=True):
                # Serialize embedding to binary format for vec0
                embedding_blob = _serialize_f32(embedding)

                # Upsert into vec0 table using rowid-based approach
                # First, try to delete existing entry
                conn.execute(
                    "DELETE FROM message_embeddings WHERE message_id = ?",
                    (msg.message_id,),
                )

                # Insert new entry
                conn.execute(
                    """
                    INSERT INTO message_embeddings (message_id, embedding, provider_name, conversation_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (msg.message_id, embedding_blob, provider_name, msg.conversation_id),
                )

                # Record metadata
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

            conn.commit()

            # Update embedding status
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
        """Find semantically similar messages.

        Args:
            text: Query text
            limit: Maximum results

        Returns:
            List of (message_id, distance) tuples, ordered by similarity.
            Lower distance = more similar.

        Raises:
            SqliteVecError: If search fails
        """
        self._ensure_vec_available()

        # Generate query embedding
        embeddings = self._get_embeddings([text], input_type="query")
        if not embeddings:
            return []

        query_embedding = _serialize_f32(embeddings[0])

        conn = self._get_connection()
        try:
            # KNN query using vec0
            # sqlite-vec uses distance (L2 by default), lower = more similar
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
        """Find semantically similar messages filtered by provider.

        Args:
            text: Query text
            provider: Provider name to filter by
            limit: Maximum results

        Returns:
            List of (message_id, distance) tuples
        """
        self._ensure_vec_available()

        embeddings = self._get_embeddings([text], input_type="query")
        if not embeddings:
            return []

        query_embedding = _serialize_f32(embeddings[0])

        conn = self._get_connection()
        try:
            # KNN query with provider filter
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
        """Get embedding statistics.

        Returns:
            Dict with counts: embedded_messages, pending_conversations
        """
        conn = self._get_connection()
        try:
            msg_count = 0
            pending = 0

            try:
                msg_count = conn.execute(
                    "SELECT COUNT(*) FROM message_embeddings"
                ).fetchone()[0]
            except Exception:
                pass

            try:
                pending = conn.execute(
                    "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 1"
                ).fetchone()[0]
            except Exception:
                pass

            return {
                "embedded_messages": msg_count,
                "pending_conversations": pending,
            }
        finally:
            conn.close()


__all__ = ["SqliteVecProvider", "SqliteVecError"]
