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
            archive.filter().provider("claude-ai").list()
        )

        # Parallel batch retrieval
        ids = ["id1", "id2", "id3", "id4", "id5"]
        convs = await archive.get_conversations(ids)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.archive_products import (
    DaySessionSummaryProduct,
    DaySessionSummaryProductQuery,
    MaintenanceRunProduct,
    MaintenanceRunProductQuery,
    SessionProfileProduct,
    SessionProfileProductQuery,
    SessionTagRollupProduct,
    SessionTagRollupQuery,
    SessionWorkEventProduct,
    SessionWorkEventProductQuery,
    WeekSessionSummaryProduct,
    WeekSessionSummaryProductQuery,
    WorkThreadProduct,
    WorkThreadProductQuery,
)
from polylogue.config import Config, Source
from polylogue.operations import ArchiveOperations, ArchiveStats
from polylogue.services import build_runtime_services
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository

if TYPE_CHECKING:
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.models import Conversation
    from polylogue.pipeline.services.parsing import ParseResult
    from polylogue.storage.search import SearchResult


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
        self._services = build_runtime_services(config=self._config, db_path=db_path)
        self._operations = ArchiveOperations.from_services(self._services)

    async def __aenter__(self) -> Polylogue:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit async context manager."""
        await self.close()

    async def close(self) -> None:
        """Close database connections and release resources."""
        await self._services.close()

    @property
    def config(self) -> Config:
        """Get the current configuration."""
        return self._config

    @property
    def archive_root(self) -> Path:
        """Get the archive root directory."""
        return self._config.archive_root

    @property
    def backend(self) -> SQLiteBackend:
        """Get the archive backend."""
        return self._services.get_backend()

    @property
    def repository(self) -> ConversationRepository:
        """Get the archive repository."""
        return self._services.get_repository()

    @property
    def operations(self) -> ArchiveOperations:
        """Get canonical archive operations bound to this facade."""
        return self._operations

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
            conv = await archive.get_conversation("claude-ai:abc123")
            conv = await archive.get_conversation("abc12345")  # prefix match
        """
        return await self.operations.get_conversation(conversation_id)

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
        return await self.operations.get_conversations(conversation_ids)

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
            convs = await archive.list_conversations(provider="claude-ai", limit=10)
        """
        return await self.operations.list_conversations(
            provider=provider,
            limit=limit,
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = 100,
        source: str | None = None,
        since: str | None = None,
    ) -> SearchResult:
        """Search conversations through the canonical query/filter path.

        Args:
            query: Search query string
            limit: Maximum number of results to return (default: 100)
            source: Optional source/provider scope
            since: Optional timestamp filter (ISO format)

        Returns:
            SearchResult adapted from canonical conversation matches

        Example:
            results = await archive.search("python error handling", limit=20)
            for hit in results.hits:
                print(f"{hit.title}: {hit.snippet}")
        """
        return await self.operations.search(
            query,
            limit=limit,
            source=source,
            since=since,
        )

    async def get_session_product_status(self) -> dict[str, int | bool]:
        """Get durable session-product readiness counters."""
        return await self.operations.get_session_product_status()

    async def get_session_profile_product(self, conversation_id: str) -> SessionProfileProduct | None:
        """Get the versioned durable session-profile product for one conversation."""
        return await self.operations.get_session_profile_product(conversation_id)

    async def list_session_profile_products(
        self,
        query: SessionProfileProductQuery | None = None,
    ) -> list[SessionProfileProduct]:
        """List versioned durable session-profile products."""
        return await self.operations.list_session_profile_products(query)

    async def list_session_tag_rollup_products(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupProduct]:
        """List versioned durable session-tag rollup products."""
        return await self.operations.list_session_tag_rollup_products(query)

    async def get_session_work_event_products(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventProduct]:
        """Get versioned durable work-event products for one conversation."""
        return await self.operations.get_session_work_event_products(conversation_id)

    async def list_session_work_event_products(
        self,
        query: SessionWorkEventProductQuery | None = None,
    ) -> list[SessionWorkEventProduct]:
        """List versioned durable work-event products."""
        return await self.operations.list_session_work_event_products(query)

    async def get_work_thread_product(self, thread_id: str) -> WorkThreadProduct | None:
        """Get the versioned durable work-thread product for one thread."""
        return await self.operations.get_work_thread_product(thread_id)

    async def list_work_thread_products(
        self,
        query: WorkThreadProductQuery | None = None,
    ) -> list[WorkThreadProduct]:
        """List versioned durable work-thread products."""
        return await self.operations.list_work_thread_products(query)

    async def list_day_session_summary_products(
        self,
        query: DaySessionSummaryProductQuery | None = None,
    ) -> list[DaySessionSummaryProduct]:
        """List durable day-level session summary products."""
        return await self.operations.list_day_session_summary_products(query)

    async def list_week_session_summary_products(
        self,
        query: WeekSessionSummaryProductQuery | None = None,
    ) -> list[WeekSessionSummaryProduct]:
        """List durable week-level session summary products."""
        return await self.operations.list_week_session_summary_products(query)

    async def list_maintenance_run_products(
        self,
        query: MaintenanceRunProductQuery | None = None,
    ) -> list[MaintenanceRunProduct]:
        """List versioned maintenance-lineage products."""
        return await self.operations.list_maintenance_run_products(query)

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
        parsing_service = ParsingService(
            repository=self.repository,
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
        parsing_service = ParsingService(
            repository=self.repository,
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

        index_service = IndexService(config=self._config, backend=self.backend)
        return await index_service.rebuild_index()

    def filter(self) -> ConversationFilter:
        """Create a fluent filter builder for querying conversations.

        Terminal methods (list, first, count, etc.) are async and must be awaited.

        Returns:
            ConversationFilter for building queries

        Example:
            convs = await archive.filter().provider("claude-ai").contains("error").limit(10).list()
        """
        from polylogue.lib.filters import ConversationFilter

        vector_provider = None
        try:
            from polylogue.storage.search_providers import create_vector_provider

            vector_provider = create_vector_provider(self._config)
        except (ValueError, ImportError):
            pass

        return ConversationFilter(self.repository, vector_provider=vector_provider)

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
        return await self.operations.summary_stats()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Polylogue(archive_root={self._config.archive_root!r})"


__all__ = ["ArchiveStats", "Polylogue"]
