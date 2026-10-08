"""Embedding generation helpers for the sqlite-vec provider."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from polylogue.storage.embeddings.identity import EmbeddingRecipe, EmbeddingRequestSpec
from polylogue.storage.search_providers.sqlite_vec_support import (
    BATCH_SIZE,
    VOYAGE_API_URL,
    SqliteVecError,
)


class SqliteVecEmbeddingMixin:
    """Embedding generation and message-selection helpers."""

    if TYPE_CHECKING:
        model: str
        dimension: int
        voyage_key: str | None

        @property
        def document_recipe(self) -> EmbeddingRecipe: ...

        @property
        def query_recipe(self) -> EmbeddingRecipe: ...

    def _get_embeddings(
        self,
        texts: list[str],
        input_type: str = "document",
    ) -> list[list[float]]:
        """Get embeddings from Voyage AI."""
        if input_type not in ("document", "query"):
            raise SqliteVecError("embedding input_type must be document or query")
        if not texts:
            return []
        if not self.voyage_key:
            raise SqliteVecError("embedding acquisition requires a Voyage API key")
        if not self.document_recipe.retrieval_compatible(self.query_recipe):
            raise SqliteVecError("query and document recipes do not declare compatible retrieval contracts")

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
            reraise=True,
        )
        def _do_request(batch: list[str]) -> list[list[float]]:
            with httpx.Client(timeout=60.0) as client:
                recipe = self.query_recipe if input_type == "query" else self.document_recipe
                payload = EmbeddingRequestSpec(recipe=recipe, input_text=batch[0]).provider_request
                payload["input"] = [__import__("unicodedata").normalize("NFC", text) for text in batch]

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


__all__ = ["SqliteVecEmbeddingMixin"]
