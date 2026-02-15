"""Protocol definitions for pluggable backends in Polylogue.

Only protocols with 2+ implementations earn their existence here:
- SearchProvider: FTS5, Hybrid
- VectorProvider: sqlite-vec (optional, requires VOYAGE_API_KEY)
- OutputRenderer: Markdown, HTML
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from polylogue.storage.store import MessageRecord


@runtime_checkable
class SearchProvider(Protocol):
    """Full-text search provider for message content.

    Implementations: FTS5 (polylogue.storage.index), Hybrid (polylogue.storage.search_providers.hybrid)
    """

    def index(self, messages: list[MessageRecord]) -> None:
        """Add messages to the search index."""
        ...

    def search(self, query: str) -> list[str]:
        """Search indexed messages, returning matching conversation IDs ranked by relevance."""
        ...


@runtime_checkable
class VectorProvider(Protocol):
    """Vector search provider for semantic similarity.

    Implementations: SqliteVecProvider (polylogue.storage.search_providers.sqlite_vec)
    Uses Voyage AI embeddings stored in sqlite-vec.
    """

    def upsert(self, conversation_id: str, messages: list[MessageRecord]) -> None:
        """Embed and store vectors for a conversation's messages."""
        ...

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Semantic search returning (conversation_id, similarity_score) tuples."""
        ...


@runtime_checkable
class OutputRenderer(Protocol):
    """Pluggable output renderer.

    Implementations: MarkdownRenderer, HTMLRenderer (polylogue.rendering.renderers)
    """

    async def render(self, conversation_id: str, output_path: Path) -> Path:
        """Render a conversation to the output path, returning the written file path."""
        ...

    def supports_format(self) -> str:
        """Return the format name this renderer handles (e.g. 'markdown', 'html')."""
        ...


__all__ = [
    "SearchProvider",
    "VectorProvider",
    "OutputRenderer",
]
