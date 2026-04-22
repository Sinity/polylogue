"""Protocol definitions for pluggable backends in Polylogue.

Only protocols with 2+ implementations earn their existence here:
- SearchProvider: FTS5, Hybrid
- VectorProvider: sqlite-vec (optional, requires `VOYAGE_API_KEY`)
- OutputRenderer: Markdown, HTML
"""

from __future__ import annotations

import builtins
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from polylogue.lib.json import JSONDocument, JSONValue
from polylogue.storage.store import (
    ArtifactObservationRecord,
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    RawConversationRecord,
)
from polylogue.types import Provider, ValidationMode, ValidationStatus

if TYPE_CHECKING:
    import aiosqlite

    from polylogue.lib.action_events import ActionEvent
    from polylogue.lib.conversation_models import Conversation, ConversationSummary
    from polylogue.lib.message_models import Message
    from polylogue.lib.message_roles import MessageRoleFilter
    from polylogue.lib.search_hits import ConversationSearchHit
    from polylogue.lib.session_profile import SessionProfile
    from polylogue.lib.stats import ArchiveStats
    from polylogue.storage.action_event_artifacts import ActionEventArtifactState
    from polylogue.storage.archive_views import ConversationRenderProjection
    from polylogue.storage.backends.queries.stats import AggregateMessageStats
    from polylogue.storage.query_models import ConversationRecordQuery
    from polylogue.storage.store import ConversationRecord
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

    model: str

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

    async def list(
        self,
        limit: int | None = 50,
        offset: int = 0,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        path_terms: builtins.list[str] | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
    ) -> builtins.list[Conversation]: ...

    async def list_summaries(
        self,
        limit: int | None = 50,
        offset: int = 0,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        source: str | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        path_terms: builtins.list[str] | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
    ) -> builtins.list[ConversationSummary]: ...

    async def count(
        self,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        path_terms: builtins.list[str] | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
    ) -> int: ...

    async def get_summary(self, conversation_id: str) -> ConversationSummary | None: ...

    async def resolve_id(self, id_prefix: str) -> ConversationId | None: ...

    def iter_messages(
        self,
        conversation_id: str,
        *,
        dialogue_only: bool = False,
        message_roles: MessageRoleFilter = (),
        limit: int | None = None,
    ) -> AsyncIterator[Message]: ...


@runtime_checkable
class SearchStore(Protocol):
    """Search interface for conversation retrieval."""

    async def search(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ) -> list[Conversation]: ...

    async def search_summaries(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ) -> list[ConversationSummary]: ...

    async def search_summary_hits(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
        since: str | None = None,
    ) -> list[ConversationSearchHit]: ...

    async def search_similar(
        self,
        text: str,
        limit: int = 10,
        vector_provider: VectorProvider | None = None,
    ) -> list[Conversation]: ...


@runtime_checkable
class ActionEventArtifactReader(Protocol):
    """Read-only action-event artifact readiness surface."""

    async def get_action_event_artifact_state(self) -> ActionEventArtifactState: ...


@runtime_checkable
class ConversationQueryRuntimeStore(
    ConversationReader,
    SearchStore,
    ActionEventArtifactReader,
    Protocol,
):
    """Repository/query runtime surface consumed by canonical query execution."""

    async def list_summaries_by_query(
        self,
        query: ConversationRecordQuery,
    ) -> list[ConversationSummary]: ...

    async def list_by_query(
        self,
        query: ConversationRecordQuery,
    ) -> list[Conversation]: ...

    async def count_by_query(self, query: ConversationRecordQuery) -> int: ...

    async def delete_conversation(self, conversation_id: str) -> bool: ...

    async def search_actions(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ) -> list[Conversation]: ...


@runtime_checkable
class ArchiveMessageQueryStore(Protocol):
    """Low-level archive/message query band used by CLI output and stats helpers."""

    async def get_messages(self, conversation_id: str) -> list[MessageRecord]: ...

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, int]: ...

    async def get_message_counts_batch(self, conversation_ids: list[str]) -> dict[str, int]: ...

    async def aggregate_message_stats(
        self,
        conversation_ids: list[str] | None = None,
    ) -> AggregateMessageStats: ...

    async def get_stats_by(self, group_by: str = "provider") -> dict[str, int]: ...


@runtime_checkable
class SemanticArchiveQueryStore(ArchiveMessageQueryStore, Protocol):
    """Archive query band needed to hydrate semantic facts in batch."""

    async def get_conversations_batch(self, ids: list[str]) -> list[ConversationRecord]: ...

    async def get_messages_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[MessageRecord]]: ...

    async def get_attachments_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[AttachmentRecord]]: ...


@runtime_checkable
class ConversationOutputStore(ConversationReader, Protocol):
    """Conversation output surface used by streaming and summary display helpers."""

    async def get_render_projection(
        self,
        conversation_id: str,
    ) -> ConversationRenderProjection | None: ...

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, int]: ...

    async def get_message_counts_batch(self, conversation_ids: list[str]) -> dict[str, int]: ...


@runtime_checkable
class ConversationSemanticStatsStore(ActionEventArtifactReader, Protocol):
    """Semantic stats surface for action-event-backed grouped output."""

    async def get_conversations_batch(self, ids: list[str]) -> list[ConversationRecord]: ...

    async def get_messages_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[MessageRecord]]: ...

    async def get_attachments_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[AttachmentRecord]]: ...

    async def get_action_events_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, tuple[ActionEvent, ...]]: ...


@runtime_checkable
class ConversationArchiveStatsStore(
    ConversationOutputStore,
    ConversationSemanticStatsStore,
    Protocol,
):
    """Archive stats/profile surface consumed by grouped CLI output helpers."""

    async def aggregate_message_stats(
        self,
        conversation_ids: list[str] | None = None,
    ) -> AggregateMessageStats: ...

    async def get_archive_stats(
        self,
        *,
        conn: aiosqlite.Connection | None = None,
    ) -> ArchiveStats: ...

    async def get_stats_by(self, group_by: str = "provider") -> dict[str, int]: ...

    async def get_session_profiles_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, SessionProfile]: ...

    async def get_many(self, conversation_ids: list[str]) -> list[Conversation]: ...


@runtime_checkable
class TagStore(Protocol):
    """Tag and metadata management interface."""

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]: ...

    async def get_metadata(self, conversation_id: str) -> JSONDocument: ...

    async def update_metadata(self, conversation_id: str, key: str, value: JSONValue) -> None: ...

    async def delete_metadata(self, conversation_id: str, key: str) -> None: ...

    async def add_tag(self, conversation_id: str, tag: str) -> None: ...

    async def remove_tag(self, conversation_id: str, tag: str) -> None: ...


@runtime_checkable
class ConversationArchiveReadStore(ConversationReader, SearchStore, Protocol):
    """Small read-side surface used by UI and resource adapters."""

    async def get_archive_stats(
        self,
        *,
        conn: aiosqlite.Connection | None = None,
    ) -> ArchiveStats: ...


@runtime_checkable
class RawPersistenceStore(Protocol):
    """Minimal raw-persistence surface used during acquisition."""

    async def save_raw_conversation(self, record: RawConversationRecord) -> bool: ...

    async def save_artifact_observation(self, record: ArtifactObservationRecord) -> bool: ...


@runtime_checkable
class RawValidationStore(Protocol):
    """Minimal raw-validation surface used by validation flows."""

    async def get_raw_conversations_batch(
        self,
        raw_ids: builtins.list[str],
    ) -> builtins.list[RawConversationRecord]: ...

    async def mark_raw_validated(
        self,
        raw_id: str,
        *,
        status: ValidationStatus | str,
        error: str | None = None,
        drift_count: int = 0,
        provider: Provider | str | None = None,
        mode: ValidationMode | str | None = None,
        payload_provider: Provider | str | None = None,
    ) -> None: ...

    async def mark_raw_parsed(
        self,
        raw_id: str,
        *,
        error: str | None = None,
        payload_provider: Provider | str | None = None,
    ) -> None: ...


__all__ = [
    "SearchProvider",
    "VectorProvider",
    "OutputRenderer",
    "ProgressCallback",
    "ConversationReader",
    "SearchStore",
    "ActionEventArtifactReader",
    "ConversationQueryRuntimeStore",
    "ArchiveMessageQueryStore",
    "SemanticArchiveQueryStore",
    "ConversationOutputStore",
    "ConversationSemanticStatsStore",
    "ConversationArchiveStatsStore",
    "TagStore",
    "RawPersistenceStore",
    "RawValidationStore",
]
