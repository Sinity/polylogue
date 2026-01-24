"""Qdrant vector search provider implementation.

This module provides a VectorProvider implementation using Qdrant for
semantic similarity search with Voyage AI embeddings.

Note: Imports are lazy to avoid loading heavy dependencies unless needed.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

from polylogue.storage.store import MessageRecord

logger = logging.getLogger(__name__)

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
DEFAULT_COLLECTION = "polylogue_messages"
DEFAULT_VECTOR_SIZE = 1024  # Voyage-2 embedding dimension


class QdrantError(RuntimeError):
    """Raised when Qdrant operations fail."""
    pass


def _retry_decorator() -> Any:
    """Lazy import for tenacity retry decorator."""
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    def decorator(stop_attempts: int, min_wait: int, max_wait: int, retry_on: type[Exception]) -> Any:
        return retry(
            stop=stop_after_attempt(stop_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(retry_on),
            reraise=True,
        )
    return decorator


class QdrantProvider:
    """VectorProvider implementation using Qdrant + Voyage AI embeddings.

    This provider generates semantic embeddings using Voyage AI's API and
    stores them in a Qdrant vector database for similarity search.

    Attributes:
        qdrant_url: URL of the Qdrant server
        api_key: Optional Qdrant API key for authentication
        voyage_key: Voyage AI API key for embedding generation
        collection: Name of the Qdrant collection to use
    """

    def __init__(
        self,
        qdrant_url: str,
        api_key: str | None,
        voyage_key: str,
        collection: str = DEFAULT_COLLECTION
    ) -> None:
        """Initialize Qdrant vector search provider.

        Args:
            qdrant_url: URL of Qdrant server (e.g., "http://localhost:6333")
            api_key: Optional Qdrant API key for authentication
            voyage_key: Voyage AI API key for generating embeddings
            collection: Name of collection to use (default: "polylogue_messages")

        Raises:
            QdrantError: If connection to Qdrant fails
        """
        self.qdrant_url = qdrant_url
        self.api_key = api_key
        self.voyage_key = voyage_key
        self.collection = collection
        self._client: QdrantClient | None = None

        # Ensure collection exists
        self._ensure_collection()

    @property
    def client(self) -> QdrantClient:
        """Lazy-load Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self.qdrant_url, api_key=self.api_key)
        return self._client

    def _ensure_collection(self, vector_size: int = DEFAULT_VECTOR_SIZE) -> None:
        """Create Qdrant collection if it doesn't exist.

        Args:
            vector_size: Dimension of embedding vectors (default: 1024 for Voyage-2)
        """
        from qdrant_client.http import models
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _do_ensure() -> None:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    ),
                )

        _do_ensure()

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings from Voyage AI.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            QdrantError: If embedding generation fails
        """
        if not texts:
            return []

        import httpx
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
            reraise=True,
        )
        def _do_get() -> list[list[float]]:
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        VOYAGE_API_URL,
                        headers={"Authorization": f"Bearer {self.voyage_key}"},
                        json={
                            "input": texts,
                            "model": "voyage-2",
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    return [item["embedding"] for item in data["data"]]
            except httpx.HTTPError as exc:
                raise QdrantError(f"Failed to generate embeddings: {exc}") from exc

        return _do_get()

    def upsert(self, conversation_id: str, messages: list[MessageRecord]) -> None:
        """Upsert message embeddings into the vector store.

        This operation is idempotent - repeated calls with the same messages
        will result in the same final state. Uses deterministic UUIDs based
        on message_id to ensure idempotency.

        Args:
            conversation_id: ID of the conversation containing these messages
            messages: Messages to embed and store. Must include message_id and text.

        Raises:
            QdrantError: If embedding generation or storage fails
        """
        if not messages:
            return

        from qdrant_client.http import models
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        # Filter messages with text content
        text_messages = [msg for msg in messages if msg.text]
        if not text_messages:
            return

        # Generate embeddings
        texts = [msg.text for msg in text_messages if msg.text]
        embeddings = self._get_embeddings(texts)

        # Prepare points for upsert
        points = []
        for msg, vector in zip(text_messages, embeddings, strict=True):
            # Use deterministic UUID based on message_id for idempotency
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, msg.message_id))

            # Need provider_name - fetch from message metadata or use placeholder
            # In practice, this should be passed in or fetched from DB
            provider_name: str = "unknown"
            if msg.provider_meta:
                pname = msg.provider_meta.get("provider_name", "unknown")
                if isinstance(pname, str):
                    provider_name = pname

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "message_id": msg.message_id,
                        "conversation_id": msg.conversation_id,
                        "provider_name": provider_name,
                        "content": msg.text,
                    },
                )
            )

        # Upsert to Qdrant with retry
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _do_upsert() -> None:
            try:
                self.client.upsert(collection_name=self.collection, points=points)
            except Exception as exc:
                raise QdrantError(f"Failed to upsert vectors: {exc}") from exc

        _do_upsert()

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Find semantically similar messages.

        Generates an embedding for the query text and searches for the most
        similar message embeddings in the vector store.

        Args:
            text: Query text to search for
            limit: Maximum number of results to return (default: 10)

        Returns:
            List of (message_id, similarity_score) tuples, ordered by descending
            similarity score. Scores are cosine similarity values (0.0 to 1.0).

        Raises:
            QdrantError: If query fails or embedding generation fails
        """
        # Generate embedding for query
        embeddings = self._get_embeddings([text])
        if not embeddings:
            return []

        query_vector = embeddings[0]

        # Search Qdrant
        try:
            results = self.client.search(  # type: ignore[attr-defined]
                collection_name=self.collection,
                query_vector=query_vector,
                limit=limit,
            )

            return [
                (result.payload["message_id"], result.score)
                for result in results
            ]
        except Exception as exc:
            raise QdrantError(f"Vector search failed: {exc}") from exc


__all__ = ["QdrantProvider", "QdrantError"]
