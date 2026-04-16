"""Protocol definitions for pluggable backends in Polylogue.

Only protocols with 2+ implementations earn their existence here:
- SearchProvider: FTS5, Hybrid
- VectorProvider: sqlite-vec (optional, requires `VOYAGE_API_KEY`)
- OutputRenderer: Markdown, HTML
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from polylogue.storage.store import MessageRecord

if TYPE_CHECKING:
    from polylogue.lib.conversation_models import Conversation, ConversationSummary
    from polylogue.lib.message_models import Message
    from polylogue.types import ConversationId


@runtime_checkable
class SearchProvider(Protocol):
    """Full-text search provider for message content.

    Implementations: FTS5 (polylogue.storage.index), Hybrid (polylogue.storage.search_providers.hybrid)
    """

    def index(self, messages: list[MessageRecord]) -> None:
        """Add messages to the search index."""
        ...

    def search(self, query: str) -> list[str]:
        """Search indexed messages, returning matching message IDs ranked by relevance."""
        ...


@runtime_checkable
class VectorProvider(Protocol):
    """Vector search provider for semantic similarity.

    Implementations: SqliteVecProvider (polylogue.storage.search_providers.sqlite_vec)
    Uses Voyage AI embeddings stored in sqlite-vec.
    """

    def upsert(self, conversation_id: str, messages: list[MessageRecord]) -> None:
        """Synchronously embed and store vectors for a conversation's messages."""
        ...

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Synchronously return ranked ``(message_id, distance)`` search results."""
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


@runtime_checkable
class ProgressCallback(Protocol):
    """Progress callback shared by pipeline stages and CLI observers."""

    def __call__(self, amount: int, desc: str | None = None) -> None:
        """Report incremental progress."""
        ...


@runtime_checkable
class ConversationReader(Protocol):
    """Read-only interface for conversation retrieval.

    Subset of ConversationRepository used by filters and query specs that
    only need to read conversations, not write or search them.
    """

    async def get(self, conversation_id: str) -> Conversation | None: ...

    async def get_eager(self, conversation_id: str) -> Conversation | None: ...

    async def list(self, **kwargs: object) -> list[Conversation]: ...

    async def list_summaries(self, **kwargs: object) -> list[ConversationSummary]: ...

    async def count(self, **kwargs: object) -> int: ...

    async def get_summary(self, conversation_id: str) -> ConversationSummary | None: ...

    async def resolve_id(self, id_prefix: str) -> ConversationId | None: ...

    def iter_messages(
        self,
        conversation_id: str,
        *,
        dialogue_only: bool = False,
        limit: int | None = None,
    ) -> AsyncIterator[Message]: ...


@runtime_checkable
class SearchStore(Protocol):
    """Search interface for conversation retrieval."""

    async def search(self, query: str, **kwargs: object) -> list[Conversation]: ...

    async def search_summaries(self, query: str, **kwargs: object) -> list[ConversationSummary]: ...

    async def search_similar(self, text: str, **kwargs: object) -> list[Conversation]: ...


@runtime_checkable
class TagStore(Protocol):
    """Tag and metadata management interface."""

    async def list_tags(self, **kwargs: object) -> dict[str, int]: ...

    async def get_metadata(self, conversation_id: str) -> dict[str, object]: ...

    async def update_metadata(self, conversation_id: str, key: str, value: object) -> None: ...

    async def delete_metadata(self, conversation_id: str, key: str) -> None: ...


__all__ = [
    "SearchProvider",
    "VectorProvider",
    "OutputRenderer",
    "ProgressCallback",
    "ConversationReader",
    "SearchStore",
    "TagStore",
]
