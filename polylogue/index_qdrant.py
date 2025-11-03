from __future__ import annotations

import os
from typing import Any, Dict

from .render import MarkdownDocument


def update_qdrant_index(
    *,
    provider: str,
    conversation_id: str,
    slug: str,
    path,
    document: MarkdownDocument,
    metadata: Dict[str, Any],
) -> None:
    url = os.environ.get("POLYLOGUE_QDRANT_URL")
    if not url:  # pragma: no cover - requires external service
        return
    collection = os.environ.get("POLYLOGUE_QDRANT_COLLECTION", "polylogue")
    size = int(os.environ.get("POLYLOGUE_QDRANT_VECTOR_SIZE", "1"))
    api_key = os.environ.get("POLYLOGUE_QDRANT_API_KEY")

    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    from qdrant_client.http.exceptions import UnexpectedResponse

    client = QdrantClient(url=url, api_key=api_key)
    try:
        client.get_collection(collection)
    except UnexpectedResponse as exc:
        status = getattr(exc, "status_code", None)
        if status == 404:
            client.recreate_collection(
                collection_name=collection,
                vectors_config=rest.VectorParams(size=size, distance=rest.Distance.COSINE),
            )
        else:
            raise

    metric = float(document.stats.get("totalTokensApprox", 0) or len(document.body))
    vector = [metric] + [0.0] * max(0, size - 1)

    payload = {
        "provider": provider,
        "conversation_id": conversation_id,
        "slug": slug,
        "path": str(path),
        "title": document.metadata.get("title"),
        "updated_at": metadata.get("updatedAt"),
        "tokens": document.stats.get("totalTokensApprox"),
        "content": document.body,
    }

    point_id = f"{provider}:{conversation_id}"
    client.upsert(
        collection_name=collection,
        points=[
            rest.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        ],
    )
