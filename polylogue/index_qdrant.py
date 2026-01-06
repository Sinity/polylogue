from __future__ import annotations

import json
import logging
import os
import uuid
from typing import List, Optional, Dict, Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

LOGGER = logging.getLogger(__name__)

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
DEFAULT_COLLECTION = "polylogue_messages"
DEFAULT_VECTOR_SIZE = 1024 # Voyage-2 size

class QdrantError(RuntimeError):
    pass

def _get_voyage_key() -> str:
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        raise QdrantError("VOYAGE_API_KEY environment variable not set")
    return key

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from Voyage AI using httpx."""
    key = _get_voyage_key()
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            VOYAGE_API_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "input": texts,
                "model": "voyage-2",
            }
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

class VectorStore:
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection: str = DEFAULT_COLLECTION,
    ):
        self.client = QdrantClient(
            url=url or os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=api_key or os.environ.get("QDRANT_API_KEY"),
        )
        self.collection = collection

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def ensure_collection(self, vector_size: int = DEFAULT_VECTOR_SIZE):
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def upsert_messages(self, messages: List[Dict[str, Any]]):
        """Upsert messages to Qdrant. Idempotency is ensured by UUID v5 of message_id."""
        if not messages:
            return

        texts = [m["content"] for m in messages]
        embeddings = get_embeddings(texts)

        points = []
        for msg, vector in zip(messages, embeddings):
            # Ensure idempotency by using a deterministic UUID based on message_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, msg["message_id"]))
            points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "message_id": msg["message_id"],
                    "conversation_id": msg["conversation_id"],
                    "provider_name": msg["provider_name"],
                    "content": msg["content"],
                }
            ))

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

def update_qdrant_for_conversations(conversation_ids: List[str], conn):
    """Bridge function to index conversations from SQLite to Qdrant."""
    if not conversation_ids:
        return
    
    store = VectorStore()
    store.ensure_collection()

    for i in range(0, len(conversation_ids), 100):
        chunk = conversation_ids[i:i+100]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(f"""
            SELECT messages.message_id, messages.conversation_id, conversations.provider_name, messages.text
            FROM messages
            JOIN conversations ON conversations.conversation_id = messages.conversation_id
            WHERE messages.text IS NOT NULL AND messages.conversation_id IN ({placeholders})
        """, tuple(chunk)).fetchall()

        messages = [
            {
                "message_id": row["message_id"],
                "conversation_id": row["conversation_id"],
                "provider_name": row["provider_name"],
                "content": row["text"],
            }
            for row in rows
        ]
        store.upsert_messages(messages)
