"""High-level library facade for Polylogue.

This module provides the `Polylogue` class, which is the primary interface
for using Polylogue as a library. It wraps the underlying repository, services,
and configuration in a simple, user-friendly API.

Example:
    from polylogue import Polylogue

    # Initialize
    archive = Polylogue(archive_root="~/.polylogue")

    # Ingest files
    result = archive.ingest_file("chatgpt_export.json")
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

from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import Config, Source
from polylogue.container import ApplicationContainer
from polylogue.lib.filters import ConversationFilter
from polylogue.lib.repository import ConversationRepository
from polylogue.storage.search import SearchResult, search_messages

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.pipeline.services.indexing import IndexService
    from polylogue.pipeline.services.ingestion import IngestionService, IngestResult
    from polylogue.protocols import StorageBackend
    from polylogue.storage.repository import StorageRepository


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
    as a library. It manages configuration, services, and provides
    convenient methods for common operations.

    Args:
        archive_root: Path to the archive directory. Defaults to ~/.local/share/polylogue/archive
        config_path: Optional path to config file. If None, uses default location.
        db_path: Optional path to database file. If None, uses default location.

    Example:
        archive = Polylogue(archive_root="~/my-chats")
        result = archive.ingest_file("chatgpt.json")
        conv = archive.get_conversation("claude:abc123")
    """

    def __init__(
        self,
        archive_root: str | Path | None = None,
        config_path: str | Path | None = None,
        db_path: str | Path | None = None,
    ):
        """Initialize the Polylogue archive."""
        # Convert paths
        if archive_root is not None:
            archive_root = Path(archive_root).expanduser().resolve()
        if config_path is not None:
            config_path = Path(config_path).expanduser().resolve()
        if db_path is not None:
            db_path = Path(db_path).expanduser().resolve()

        # Initialize container
        self._container: ApplicationContainer | None

        # Try to load config, fall back to minimal config if not found
        if config_path is not None and config_path.exists():
            from dependency_injector import providers

            from polylogue.config import load_config

            self._container = ApplicationContainer()
            self._container.config.override(
                providers.Singleton(load_config, path=config_path)
            )
            self._config: Config = self._container.config()
        else:
            # Create minimal config for library use
            from polylogue.config import CONFIG_VERSION, DEFAULT_ARCHIVE_ROOT

            if archive_root is None:
                archive_root = DEFAULT_ARCHIVE_ROOT

            self._config = Config(
                version=CONFIG_VERSION,
                archive_root=archive_root,
                render_root=archive_root / "render",
                sources=[],
                path=config_path or (DEFAULT_ARCHIVE_ROOT.parent / "config.json"),
            )
            self._container = None

        # Override archive_root if provided
        if archive_root is not None:
            self._config.archive_root = archive_root

        # Create storage backend (single source of truth for database access)
        from polylogue.storage.backends.sqlite import SQLiteBackend

        self._db_path = db_path
        self._backend: StorageBackend = SQLiteBackend(db_path=db_path)

        # Create repositories using shared backend
        self._repository = ConversationRepository(backend=self._backend)
        self._storage_repository: StorageRepository | None = None  # Lazy-initialized

        # Services (lazy-initialized)
        self._ingestion_service: IngestionService | None = None
        self._indexing_service: IndexService | None = None
        self._rendering_service: object | None = None

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

    def ingest_file(
        self,
        path: str | Path,
        *,
        source_name: str | None = None,
    ) -> IngestResult:
        """Ingest a single file containing AI conversations.

        The provider (ChatGPT, Claude, Codex, etc.) is automatically detected
        from the file structure.

        Args:
            path: Path to the file to ingest (.json, .jsonl, .zip)
            source_name: Optional source name for tracking (defaults to filename)

        Returns:
            IngestResult with counts of imported items

        Example:
            result = archive.ingest_file("chatgpt_export.json")
            print(f"Imported {result.counts['conversations']} conversations")
            print(f"Skipped {result.counts['skipped_conversations']} duplicates")
        """
        # Lazy-initialize ingestion service
        if self._ingestion_service is None:
            if self._container is not None:
                self._ingestion_service = self._container.ingestion_service()
            else:
                from polylogue.pipeline.services.ingestion import IngestionService
                from polylogue.storage.repository import StorageRepository

                # Create storage repository for multi-threaded ingestion
                # Uses the same backend (SQLiteBackend is internally thread-safe)
                if self._storage_repository is None:
                    self._storage_repository = StorageRepository(backend=self._backend)

                self._ingestion_service = IngestionService(
                    repository=self._storage_repository,
                    archive_root=self._config.archive_root,
                    config=self._config,
                )

        # Convert path
        file_path = Path(path).expanduser().resolve()

        # Create temporary source
        if source_name is None:
            source_name = file_path.stem

        source = Source(name=source_name, path=file_path)

        # Ingest
        return self._ingestion_service.ingest_sources(
            sources=[source],
            ui=None,
            download_assets=False,
        )

    def ingest_sources(
        self,
        sources: list[Source] | None = None,
        *,
        download_assets: bool = True,
    ) -> IngestResult:
        """Ingest conversations from configured sources.

        Args:
            sources: List of sources to ingest. If None, uses all configured sources.
            download_assets: Whether to download attachments from Google Drive (default: True)

        Returns:
            IngestResult with counts of imported items

        Example:
            # Ingest all configured sources
            result = archive.ingest_sources()

            # Ingest specific sources
            from polylogue.config import Source
            sources = [Source(name="chatgpt", path="/path/to/export.json")]
            result = archive.ingest_sources(sources=sources)
        """
        # Lazy-initialize ingestion service
        if self._ingestion_service is None:
            if self._container is not None:
                self._ingestion_service = self._container.ingestion_service()
            else:
                from polylogue.pipeline.services.ingestion import IngestionService
                from polylogue.storage.repository import StorageRepository

                # Create storage repository for multi-threaded ingestion
                # Uses the same backend (SQLiteBackend is internally thread-safe)
                if self._storage_repository is None:
                    self._storage_repository = StorageRepository(backend=self._backend)

                self._ingestion_service = IngestionService(
                    repository=self._storage_repository,
                    archive_root=self._config.archive_root,
                    config=self._config,
                )

        # Use configured sources if none provided
        if sources is None:
            sources = self._config.sources

        return self._ingestion_service.ingest_sources(
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
        # Use repository's list method with filters
        all_conversations = self._repository.list(
            provider=provider,
            limit=limit or 50,
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
            if self._container is not None:
                self._indexing_service = self._container.indexing_service()
            else:
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
        if self._container is not None:
            try:
                vector_provider = self._container.vector_provider()
            except Exception:
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
        recent = sorted(
            conversations,
            key=lambda c: c.updated_at or c.created_at or c.id,
            reverse=True,
        )[:5]

        # Get last sync time from runs table if available
        last_sync = None
        try:
            # Check if backend has runs info
            conn = getattr(self._backend, "_get_connection", lambda: None)()
            if conn:
                row = conn.execute(
                    "SELECT MAX(ended_at) as last FROM runs"
                ).fetchone()
                if row and row[0]:
                    last_sync = row[0]
        except Exception:
            pass

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
