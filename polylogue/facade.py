"""High-level async library facade for Polylogue.

This module provides the `Polylogue` class for async/await operations.
Enables concurrent queries and parallel batch operations.

Performance gains:
- Batch retrieval: 5-10x faster via concurrent execution
- Parallel queries: Run multiple filters concurrently
- Non-blocking I/O: Doesn't block event loop during database operations

Example:
    async with Polylogue() as archive:
        # Concurrent queries
        stats, recent, claude = await asyncio.gather(
            archive.stats(),
            archive.filter().limit(10).list(),
            archive.filter().provider("claude").list()
        )

        # Parallel batch retrieval
        ids = ["id1", "id2", "id3", "id4", "id5"]
        convs = await archive.get_conversations(ids)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from polylogue.config import Config, Source
from polylogue.storage.backends.async_sqlite import SQLiteBackend

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.models import Conversation
    from polylogue.pipeline.services.parsing import ParseResult
    from polylogue.storage.search import SearchResult


class ArchiveStats:
    """Statistics about the archive (facade-level).

    Attributes:
        conversation_count: Total number of conversations
        message_count: Total number of messages
        word_count: Total word count across all messages
        providers: Provider name -> count mapping
        tags: Tag name -> count mapping
        last_sync: Timestamp of last sync operation
        recent: List of 5 most recent conversations
    """

    def __init__(
        self,
        conversation_count: int,
        message_count: int,
        word_count: int,
        providers: dict[str, int],
        tags: dict[str, int],
        last_sync: str | None,
        recent: list[Conversation],
    ):
        self.conversation_count = conversation_count
        self.message_count = message_count
        self.word_count = word_count
        self.providers = providers
        self.tags = tags
        self.last_sync = last_sync
        self.recent = recent

    def __repr__(self) -> str:
        return (
            f"ArchiveStats(conversations={self.conversation_count}, "
            f"messages={self.message_count}, providers={list(self.providers.keys())})"
        )


class Polylogue:
    """High-level async facade for the Polylogue library.

    Provides an async/await API for querying, filtering, and managing
    a conversation archive.

    Args:
        archive_root: Path to the archive directory
        db_path: Optional path to database file

    Example:
        # Context manager (recommended)
        async with Polylogue() as archive:
            convs = await archive.filter().list()

        # Manual lifecycle
        archive = Polylogue()
        try:
            convs = await archive.filter().list()
        finally:
            await archive.close()
    """

    def __init__(
        self,
        archive_root: str | Path | None = None,
        db_path: str | Path | None = None,
    ):
        """Initialize async Polylogue archive."""
        # Convert paths
        if archive_root is not None:
            archive_root = Path(archive_root).expanduser().resolve()
        if db_path is not None:
            db_path = Path(db_path).expanduser().resolve()

        # Create minimal config
        from polylogue.paths import archive_root as _archive_root
        from polylogue.paths import render_root

        if archive_root is None:
            archive_root = _archive_root()

        self._config = Config(
            archive_root=archive_root,
            render_root=render_root(),
            sources=[],
        )

        # Create async backend
        self._db_path = db_path
        self._backend = SQLiteBackend(db_path=db_path)

    async def __aenter__(self) -> Polylogue:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit async context manager."""
        await self.close()

    async def close(self) -> None:
        """Close database connections and release resources."""
        await self._backend.close()

    @property
    def config(self) -> Config:
        """Get the current configuration."""
        return self._config

    @property
    def archive_root(self) -> Path:
        """Get the archive root directory."""
        return self._config.archive_root

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID.

        Supports both full IDs and unambiguous prefixes (e.g. the first 8
        characters of a UUID). If the prefix matches multiple conversations,
        returns None.

        Args:
            conversation_id: Full or partial conversation ID

        Returns:
            Conversation object or None if not found

        Example:
            conv = await archive.get_conversation("claude:abc123")
            conv = await archive.get_conversation("abc12345")  # prefix match
        """
        full_id = await self._backend.resolve_id(conversation_id) or conversation_id
        conv_record = await self._backend.get_conversation(full_id)
        if not conv_record:
            return None

        # Get messages and attachments
        msg_records = await self._backend.get_messages(full_id)

        # Convert to Conversation model
        from polylogue.storage.repository import _records_to_conversation

        return _records_to_conversation(conv_record, msg_records, [])

    async def get_conversations(self, conversation_ids: list[str]) -> list[Conversation]:
        """Get multiple conversations by ID using batch queries.

        Uses 2 queries (conversations + messages) instead of 2*N individual
        queries, avoiding connection storms on large databases.

        Args:
            conversation_ids: List of conversation IDs

        Returns:
            List of Conversation objects (may be fewer than requested)

        Example:
            ids = ["id1", "id2", "id3", "id4", "id5"]
            convs = await archive.get_conversations(ids)
        """
        if not conversation_ids:
            return []

        from polylogue.storage.repository import _records_to_conversation

        records = await self._backend.get_conversations_batch(conversation_ids)
        if not records:
            return []

        by_id = {rec.conversation_id: rec for rec in records}
        present_ids = [cid for cid in conversation_ids if cid in by_id]
        msgs_by_id = await self._backend.get_messages_batch(present_ids)

        return [
            _records_to_conversation(by_id[cid], msgs_by_id.get(cid, []), [])
            for cid in present_ids
        ]

    async def list_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
    ) -> list[Conversation]:
        """List conversations with optional filtering.

        Uses batch queries to avoid N+1 connection storms.

        Args:
            provider: Filter by provider name
            limit: Maximum number of conversations

        Returns:
            List of Conversation objects

        Example:
            convs = await archive.list_conversations(provider="claude", limit=10)
        """
        conv_records = await self._backend.list_conversations(
            provider=provider,
            limit=limit,
        )
        if not conv_records:
            return []

        from polylogue.storage.repository import _records_to_conversation

        ids = [cr.conversation_id for cr in conv_records]
        msgs_by_id = await self._backend.get_messages_batch(ids)

        return [
            _records_to_conversation(cr, msgs_by_id.get(cr.conversation_id, []), [])
            for cr in conv_records
        ]

    async def search(
        self,
        query: str,
        *,
        limit: int = 100,
        source: str | None = None,
        since: str | None = None,
    ) -> SearchResult:
        """Search conversations using full-text search.

        Args:
            query: Search query string
            limit: Maximum number of results to return (default: 100)
            source: Optional source/provider filter
            since: Optional timestamp filter (ISO format)

        Returns:
            SearchResult with matching conversations

        Example:
            results = await archive.search("python error handling", limit=20)
            for hit in results.hits:
                print(f"{hit.title}: {hit.snippet}")
        """
        from polylogue.storage.search import search_messages

        return await asyncio.to_thread(
            search_messages,
            query=query,
            archive_root=self._config.archive_root,
            render_root_path=self._config.render_root,
            limit=limit,
            source=source,
            since=since,
        )

    async def parse_file(
        self,
        path: str | Path,
        *,
        source_name: str | None = None,
    ) -> ParseResult:
        """Parse a single file containing AI conversations.

        The provider (ChatGPT, Claude, Codex, etc.) is automatically detected
        from the file structure.

        Args:
            path: Path to the file to parse (.json, .jsonl, .zip)
            source_name: Optional source name for tracking (defaults to filename)

        Returns:
            ParseResult with counts of imported items

        Example:
            result = await archive.parse_file("chatgpt_export.json")
            print(f"Imported {result.counts['conversations']} conversations")
        """
        from polylogue.pipeline.services.parsing import ParsingService
        from polylogue.storage.repository import ConversationRepository

        repository = ConversationRepository(backend=self._backend)
        parsing_service = ParsingService(
            repository=repository,
            archive_root=self._config.archive_root,
            config=self._config,
        )

        file_path = Path(path).expanduser().resolve()
        if source_name is None:
            source_name = file_path.stem

        source = Source(name=source_name, path=file_path)
        return await parsing_service.parse_sources(
            sources=[source],
            ui=None,
            download_assets=False,
        )

    async def parse_sources(
        self,
        sources: list[Source] | None = None,
        *,
        download_assets: bool = True,
    ) -> ParseResult:
        """Parse conversations from configured sources.

        Args:
            sources: List of sources to parse. If None, uses all configured sources.
            download_assets: Whether to download attachments from Google Drive (default: True)

        Returns:
            ParseResult with counts of imported items

        Example:
            result = await archive.parse_sources()
        """
        from polylogue.pipeline.services.parsing import ParsingService
        from polylogue.storage.repository import ConversationRepository

        repository = ConversationRepository(backend=self._backend)
        parsing_service = ParsingService(
            repository=repository,
            archive_root=self._config.archive_root,
            config=self._config,
        )

        if sources is None:
            sources = self._config.sources

        return await parsing_service.parse_sources(
            sources=sources,
            ui=None,
            download_assets=download_assets,
        )

    async def rebuild_index(self) -> bool:
        """Rebuild the full-text search index.

        Returns:
            True if rebuild succeeded, False otherwise

        Example:
            success = await archive.rebuild_index()
        """
        from polylogue.pipeline.services.indexing import IndexService

        index_service = IndexService(config=self._config, backend=self._backend)
        return await index_service.rebuild_index()

    def filter(self) -> ConversationFilter:
        """Create a fluent filter builder for querying conversations.

        Terminal methods (list, first, count, etc.) are async and must be awaited.

        Returns:
            ConversationFilter for building queries

        Example:
            convs = await archive.filter().provider("claude").contains("error").limit(10).list()
        """
        from polylogue.lib.filters import ConversationFilter
        from polylogue.storage.repository import ConversationRepository

        repository = ConversationRepository(backend=self._backend)

        vector_provider = None
        try:
            from polylogue.storage.search_providers import create_vector_provider

            vector_provider = create_vector_provider(self._config)
        except (ValueError, ImportError):
            pass

        return ConversationFilter(repository, vector_provider=vector_provider)

    async def stats(self) -> ArchiveStats:
        """Get statistics about the archive.

        Returns:
            ArchiveStats with conversation count, message count, provider breakdown, etc.

        Example:
            stats = await archive.stats()
            print(f"Total: {stats.conversation_count} conversations")
            for provider, count in stats.providers.items():
                print(f"  {provider}: {count}")
        """
        conversations = await self.list_conversations(limit=10000)

        providers: dict[str, int] = {}
        tags: dict[str, int] = {}
        total_messages = 0
        total_words = 0

        for conv in conversations:
            providers[conv.provider] = providers.get(conv.provider, 0) + 1
            for tag in conv.tags:
                tags[tag] = tags.get(tag, 0) + 1
            total_messages += len(conv.messages)
            total_words += sum(m.word_count for m in conv.messages)

        _epoch = datetime.min.replace(tzinfo=timezone.utc)
        recent = sorted(
            conversations,
            key=lambda c: c.updated_at or c.created_at or _epoch,
            reverse=True,
        )[:5]

        last_sync = None
        try:
            last_sync = await self._backend.get_last_sync_timestamp()
        except Exception as exc:
            logger.debug("failed to query last sync timestamp", error=str(exc))

        return ArchiveStats(
            conversation_count=len(conversations),
            message_count=total_messages,
            word_count=total_words,
            providers=providers,
            tags=tags,
            last_sync=last_sync,
            recent=recent,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Polylogue(archive_root={self._config.archive_root!r})"


__all__ = ["ArchiveStats", "Polylogue"]
