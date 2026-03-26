"""Embedding generation helpers for the sqlite-vec provider."""

from __future__ import annotations

import time

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from polylogue.storage.search_providers.sqlite_vec_support import (
    BATCH_SIZE,
    DEFAULT_DIMENSION,
    VOYAGE_API_URL,
    SqliteVecError,
)
from polylogue.storage.store import MessageRecord


class SqliteVecEmbeddingMixin:
    """Embedding generation and message-selection helpers."""

    def _get_embeddings(
        self,
        texts: list[str],
        input_type: str = "document",
    ) -> list[list[float]]:
        """Get embeddings from Voyage AI."""
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

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            try:
                embeddings = _do_request(batch)
                all_embeddings.extend(embeddings)
                if i + BATCH_SIZE < len(texts):
                    time.sleep(0.1)
            except httpx.HTTPError as exc:
                status = getattr(exc.response, "status_code", None) if hasattr(exc, "response") else None
                detail = f"HTTP {status}" if status else type(exc).__name__
                raise SqliteVecError(f"Embedding generation failed: {detail}") from exc

        return all_embeddings

    def _should_embed_message(self, msg: MessageRecord) -> bool:
        """Determine if a message should be embedded."""
        if not msg.text or not msg.text.strip():
            return False
        if len(msg.text.strip()) < 20:
            return False
        if msg.role == "system":
            return False
        if msg.role == "tool_result":
            text = msg.text.strip().lower()
            if text in ("ok", "success", "done", "error", "failed"):
                return False
        return True


__all__ = ["SqliteVecEmbeddingMixin"]
