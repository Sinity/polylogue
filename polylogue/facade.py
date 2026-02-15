"""High-level library facade for Polylogue.

This module provides the `Polylogue` class, which is the primary interface
for using Polylogue as a library. It wraps the underlying repository, services,
and configuration in a simple, user-friendly API.

Example:
    from polylogue import Polylogue

    # Initialize
    archive = Polylogue(archive_root="~/.polylogue")

    # Parse files
    result = archive.parse_file("chatgpt_export.json")
    print(f"Imported {result.counts['conversations']} conversations")

    # Query with semantic projections
    conv = archive.get_conversation("claude:abc")
    if conv:
        for pair in conv.substantive_only().iter_pairs():
            print(f"Q: {pair.user.text[:50]}")
            print(f"A: {pair.assistant.text[:50]}")

    # Search
    results = archive.search("python error handling")
    for hit in results.hits:
        print(f"{hit.title}: {hit.snippet}")
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import Config, Source, get_config
from polylogue.lib.filters import ConversationFilter
from polylogue.lib.log import get_logger
from polylogue.storage.backends.sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.search import SearchResult, search_messages

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.pipeline.services.indexing import IndexService
    from polylogue.pipeline.services.parsing import ParseResult, ParsingService

logger = get_logger(__name__)


class ArchiveStats:
    """Statistics about the archive.

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
    """High-level facade for Polylogue library.

    This class provides a simple, user-friendly API for using Polylogue
    as a library. It manages services and provides convenient methods
    for common operations.

    Args:
        archive_root: Override archive directory. Defaults to ~/.local/share/polylogue
        db_path: Override database file. Defaults to ~/.local/share/polylogue/polylogue.db

    Example:
        archive = Polylogue()  # Uses XDG defaults
        result = archive.parse_file("chatgpt.json")
        conv = archive.get_conversation("claude:abc123")
    """

    def __init__(
        self,
        archive_root: str | Path | None = None,
        db_path: str | Path | None = None,
    ):
        """Initialize the Polylogue archive.

        Args:
            archive_root: Override archive directory (defaults to XDG data home)
            db_path: Override database path (defaults to XDG data home)
        """
        # Get hardcoded config (zero-config)
        self._config: Config = get_config()

        # Override archive_root if provided
        if archive_root is not None:
            self._config.archive_root = Path(archive_root).expanduser().resolve()

        # Create storage backend directly
        self._db_path = Path(db_path).expanduser().resolve() if db_path else None
        self._backend = SQLiteBackend(db_path=self._db_path)

        # Create repo using shared backend
        self._repository = ConversationRepository(backend=self._backend)

        # Services (lazy-initialized)
        self._parsing_service: ParsingService | None = None
        self._indexing_service: IndexService | None = None

    @property
    def config(self) -> Config:
        """Get the current configuration."""
        return self._config

    @property
    def repository(self) -> ConversationRepository:
        """Get the conversation repository for direct database access."""
        return self._repository

    @property
    def archive_root(self) -> Path:
        """Get the archive root directory."""
        return self._config.archive_root

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID with semantic projection support.

        Supports partial ID resolution - if a unique prefix is provided,
        it will be resolved to the full ID.

        Args:
            conversation_id: Full or partial conversation ID (e.g., "claude:abc" or "abc")

        Returns:
            A Conversation object with projection methods, or None if not found.

        Example:
            conv = archive.get_conversation("claude:abc123")
            if conv:
                for msg in conv.substantive_only():
                    print(msg.text)
        """
        return self._repository.view(conversation_id)

    def get_conversations(self, conversation_ids: list[str]) -> list[Conversation]:
        """Get multiple conversations by ID in a single database query.

        This is more efficient than calling get_conversation() in a loop.
        Non-existent IDs are silently skipped.

        Args:
            conversation_ids: List of full conversation IDs

        Returns:
            List of Conversation objects (may be fewer than requested if some don't exist)

        Example:
            # Efficient batch retrieval
            ids = ["claude:abc123", "chatgpt:def456", "claude:ghi789"]
            convs = archive.get_conversations(ids)
            for conv in convs:
                print(f"{conv.id}: {conv.display_title}")
        """
        return self._repository._get_many(conversation_ids)

    def search(
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
            results = archive.search("python error handling", limit=20)
            for hit in results.hits:
                print(f"{hit.title}: {hit.snippet}")
        """
        return search_messages(
            query=query,
            archive_root=self._config.archive_root,
            render_root_path=self._config.render_root,
            limit=limit,
            source=source,
            since=since,
        )

    def parse_file(
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
            result = archive.parse_file("chatgpt_export.json")
            print(f"Imported {result.counts['conversations']} conversations")
            print(f"Skipped {result.counts['skipped_conversations']} duplicates")
        """
        # Lazy-initialize parsing service
        if self._parsing_service is None:
            from polylogue.pipeline.services.parsing import ParsingService

            self._parsing_service = ParsingService(
                repository=self._repository,
                archive_root=self._config.archive_root,
                config=self._config,
            )

        # Convert path
        file_path = Path(path).expanduser().resolve()

        # Create temporary source
        if source_name is None:
            source_name = file_path.stem

        source = Source(name=source_name, path=file_path)

        # Parse
        return self._parsing_service.parse_sources(
            sources=[source],
            ui=None,
            download_assets=False,
        )

    def parse_sources(
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
            # Parse all configured sources
            result = archive.parse_sources()

            # Parse specific sources
            from polylogue.config import Source
            sources = [Source(name="chatgpt", path="/path/to/export.json")]
            result = archive.parse_sources(sources=sources)
        """
        # Lazy-initialize parsing service
        if self._parsing_service is None:
            from polylogue.pipeline.services.parsing import ParsingService

            self._parsing_service = ParsingService(
                repository=self._repository,
                archive_root=self._config.archive_root,
                config=self._config,
            )

        # Use configured sources if none provided
        if sources is None:
            sources = self._config.sources

        return self._parsing_service.parse_sources(
            sources=sources,
            ui=None,
            download_assets=download_assets,
        )

    def list_conversations(
        self,
        *,
        provider: str | None = None,
        source: str | None = None,
        limit: int | None = None,
    ) -> list[Conversation]:
        """List conversations with optional filtering.

        Args:
            provider: Filter by provider name (e.g., "claude", "chatgpt")
            source: Filter by source name
            limit: Maximum number of conversations to return

        Returns:
            List of Conversation objects

        Example:
            # List all Claude conversations
            convs = archive.list_conversations(provider="claude", limit=10)
            for conv in convs:
                print(f"{conv.id}: {conv.title}")
        """
        # When source filter is active, we need all candidates (source is in
        # provider_meta, not a SQL-pushable column).  Otherwise respect limit.
        fetch_limit = limit if not source else None
        all_conversations = self._repository.list(
            provider=provider,
            limit=fetch_limit,
            offset=0,
        )

        # Apply source filter if needed
        if source:
            filtered = []
            for conv in all_conversations:
                source_name = None
                if conv.provider_meta:
                    source_name = conv.provider_meta.get("source")
                if source_name == source:
                    filtered.append(conv)
                if limit and len(filtered) >= limit:
                    break
            return filtered

        return all_conversations

    def rebuild_index(self) -> None:
        """Rebuild the full-text search index.

        This is typically not needed as the index is updated incrementally
        during ingestion. Use this if the index becomes corrupted or after
        manual database modifications.

        Example:
            archive.rebuild_index()
        """
        # Lazy-initialize indexing service
        if self._indexing_service is None:
            from polylogue.pipeline.services.indexing import IndexService

            self._indexing_service = IndexService(
                config=self._config,
                conn=None,
            )

        self._indexing_service.rebuild_index()

    def filter(self) -> ConversationFilter:
        """Create a fluent filter builder for querying conversations.

        The filter builder allows chaining multiple filter criteria before
        executing the query.

        Returns:
            ConversationFilter for building queries

        Example:
            # Get recent Claude conversations about errors
            convs = archive.filter().provider("claude").contains("error").limit(10).list()

            # Count conversations with thinking blocks
            count = archive.filter().has("thinking").count()

            # Get first matching conversation
            conv = archive.filter().tag("important").first()
        """
        # Get vector provider if available (may be None)
        vector_provider = None
        try:
            from polylogue.storage.search_providers import create_vector_provider

            vector_provider = create_vector_provider(self._config)
        except (ValueError, ImportError):
            pass  # Vector provider not configured

        return ConversationFilter(self._repository, vector_provider=vector_provider)

    def __enter__(self) -> Polylogue:
        """Enter context manager - returns self for use in 'with' statements."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager - closes backend connections.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        self.close()

    def close(self) -> None:
        """Close database connections and release resources.

        This is called automatically when using Polylogue as a context manager.
        Manual calling is only needed when not using 'with' statements.

        Example:
            # Context manager (automatic close)
            with Polylogue() as archive:
                convs = archive.filter().list()

            # Manual close (if not using context manager)
            archive = Polylogue()
            try:
                convs = archive.filter().list()
            finally:
                archive.close()
        """
        if hasattr(self._backend, "close"):
            self._backend.close()

    def stats(self) -> ArchiveStats:
        """Get statistics about the archive.

        Returns:
            ArchiveStats with conversation count, message count, provider breakdown, etc.

        Example:
            stats = archive.stats()
            print(f"Total: {stats.conversation_count} conversations")
            for provider, count in stats.providers.items():
                print(f"  {provider}: {count}")
        """
        # Get all conversations (with reasonable limit for stats)
        conversations = self._repository.list(limit=10000)

        # Calculate stats
        providers: dict[str, int] = {}
        tags: dict[str, int] = {}
        total_messages = 0
        total_words = 0

        for conv in conversations:
            # Provider counts
            providers[conv.provider] = providers.get(conv.provider, 0) + 1

            # Tag counts
            for tag in conv.tags:
                tags[tag] = tags.get(tag, 0) + 1

            # Message and word counts
            total_messages += len(conv.messages)
            total_words += sum(m.word_count for m in conv.messages)

        # Get recent conversations (top 5 by date)
        # Use UTC-aware datetime.min as fallback — parse_timestamp() returns aware datetimes
        _epoch = datetime.min.replace(tzinfo=timezone.utc)
        recent = sorted(
            conversations,
            key=lambda c: c.updated_at or c.created_at or _epoch,
            reverse=True,
        )[:5]

        # Get last sync time from runs table if available
        last_sync = None
        try:
            # Check if backend has runs info
            conn = getattr(self._backend, "_get_connection", lambda: None)()
            if conn:
                row = conn.execute("SELECT MAX(ended_at) as last FROM runs").fetchone()
                if row and row[0]:
                    last_sync = row[0]
        except Exception as exc:
            logger.warning("Last sync lookup failed: %s", exc)

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
        return f"Polylogue(archive_root={self.archive_root!r})"
