"""Protocol definitions for pluggable backends in Polylogue.

Only protocols with 2+ implementations earn their existence here:
- VectorProvider: sqlite-vec (optional, requires `VOYAGE_API_KEY`)

``SearchProvider`` (FTS5, Hybrid) was removed (polylogue-a7xr.10): both
implementations had zero production consumers — production full-text and
hybrid retrieval has always queried FTS5/vector tables and fused results
inline rather than through a swappable provider abstraction.
"""

from __future__ import annotations

import builtins
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.core.json import JSONDocument, JSONValue

if TYPE_CHECKING:
    import aiosqlite

    from polylogue.archive.actions.actions import Action
    from polylogue.archive.message.models import Message
    from polylogue.archive.message.roles import MessageRoleFilter
    from polylogue.archive.query.search_hits import SessionSearchHit
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.archive.session.session_profile import SessionProfile
    from polylogue.archive.stats import ArchiveStats
    from polylogue.core.enums import MaterialOrigin
    from polylogue.core.types import SessionId
    from polylogue.storage.archive_views import SessionRenderProjection
    from polylogue.storage.query_models import SessionRecordQuery
    from polylogue.storage.runtime import (
        ArtifactObservationRecord,
        AttachmentRecord,
        MessageRecord,
        RawSessionRecord,
        SessionRecord,
    )
    from polylogue.storage.sqlite.queries.messages import MessageTypeName
    from polylogue.storage.sqlite.queries.stats import AggregateMessageStats


@runtime_checkable
class VectorProvider(Protocol):
    """Vector search provider for semantic similarity.

    Implementations: SqliteVecProvider (polylogue.storage.search_providers.sqlite_vec)
    Uses Voyage AI embeddings stored in sqlite-vec.
    """

    model: str

    def upsert(self, session_id: str, messages: list[MessageRecord]) -> None:
        """Synchronously embed and store vectors for a session's messages."""
        ...

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Synchronously return ranked ``(message_id, distance)`` search results."""
        ...

    def query_by_session(self, session_id: str, limit: int = 10) -> list[tuple[str, float]]:
        """Rank messages by similarity to a stored session's own embeddings.

        Reads ``session_id``'s already-materialized message vectors and KNN-searches
        them against the store, returning ranked ``(message_id, distance)`` hits with
        the seed session's own messages excluded (so the seed never ranks against
        itself). No re-embedding occurs — only stored vectors are read. Raises a typed
        error when the seed session has no stored embeddings, never silently returning
        an empty or unfiltered result.
        """
        ...


@runtime_checkable
class ProgressCallback(Protocol):
    """Progress callback shared by pipeline stages and CLI observers."""

    def __call__(self, amount: int, desc: str | None = None) -> None:
        """Report incremental progress."""
        ...


@runtime_checkable
class SessionReader(Protocol):
    """Read-only interface for session retrieval.

    Subset of SessionRepository used by filters and query specs that
    only need to read sessions, not write or search them.
    """

    async def get(self, session_id: str) -> Session | None: ...

    async def get_eager(self, session_id: str) -> Session | None: ...

    async def list(
        self,
        limit: int | None = 50,
        offset: int = 0,
        origin: str | None = None,
        origins: builtins.list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        referenced_path: builtins.list[str] | None = None,
        cwd_prefix: str | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        message_type: str | None = None,
    ) -> builtins.list[Session]: ...

    async def list_summaries(
        self,
        limit: int | None = 50,
        offset: int = 0,
        origin: str | None = None,
        origins: builtins.list[str] | None = None,
        source: str | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        referenced_path: builtins.list[str] | None = None,
        cwd_prefix: str | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        message_type: str | None = None,
    ) -> builtins.list[SessionSummary]: ...

    async def count(
        self,
        origin: str | None = None,
        origins: builtins.list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        referenced_path: builtins.list[str] | None = None,
        cwd_prefix: str | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        message_type: str | None = None,
    ) -> int: ...

    async def get_summary(self, session_id: str) -> SessionSummary | None: ...

    async def resolve_id(self, id_prefix: str, *, strict: bool = False) -> SessionId | None: ...

    def iter_messages(
        self,
        session_id: str,
        *,
        message_roles: MessageRoleFilter = (),
        material_origin: tuple[MaterialOrigin, ...] = (),
        limit: int | None = None,
    ) -> AsyncIterator[Message]: ...


@runtime_checkable
class SearchStore(Protocol):
    """Search interface for session retrieval."""

    async def search(
        self,
        query: str,
        limit: int = 20,
        origins: builtins.list[str] | None = None,
    ) -> list[Session]: ...

    async def search_summaries(
        self,
        query: str,
        limit: int = 20,
        origins: builtins.list[str] | None = None,
    ) -> list[SessionSummary]: ...

    async def search_summary_hits(
        self,
        query: str,
        limit: int = 20,
        origins: builtins.list[str] | None = None,
        since: str | None = None,
    ) -> list[SessionSearchHit]: ...

    async def search_similar(
        self,
        text: str,
        limit: int = 10,
        vector_provider: VectorProvider | None = None,
    ) -> list[Session]: ...


@runtime_checkable
class NeighborStore(Protocol):
    """Minimal session access protocol for neighboring-session discovery.

    Consumed by: polylogue.archive.session.neighbor_candidates.discover_neighbor_candidates
    """

    async def resolve_id(self, id_prefix: str, *, strict: bool = False) -> SessionId | None: ...

    async def get(self, session_id: str) -> Session | None: ...

    async def list_summaries_by_query(
        self,
        query: SessionRecordQuery,
    ) -> list[SessionSummary]: ...

    async def search_summary_hits(
        self,
        query: str,
        limit: int = 20,
        origins: builtins.list[str] | None = None,
        since: str | None = None,
    ) -> list[SessionSearchHit]: ...


@runtime_checkable
class SessionQueryRuntimeStore(
    SessionReader,
    SearchStore,
    Protocol,
):
    """Repository/query runtime surface consumed by canonical query execution."""

    async def list_summaries_by_query(
        self,
        query: SessionRecordQuery,
    ) -> list[SessionSummary]: ...

    async def list_by_query(
        self,
        query: SessionRecordQuery,
    ) -> list[Session]: ...

    async def count_by_query(self, query: SessionRecordQuery) -> int: ...

    async def delete_session(self, session_id: str) -> bool: ...

    async def search_actions(
        self,
        query: str,
        limit: int = 20,
        origins: builtins.list[str] | None = None,
    ) -> list[Session]: ...


@runtime_checkable
class ArchiveMessageQueryStore(Protocol):
    """Low-level archive/message query band used by CLI output and stats helpers."""

    async def get_messages(self, session_id: str) -> list[MessageRecord]: ...

    async def get_messages_paginated(
        self,
        session_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        material_origin: tuple[MaterialOrigin, ...] = (),
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[MessageRecord], int]: ...

    async def get_session_stats(self, session_id: str) -> dict[str, int]: ...

    async def get_message_counts_batch(self, session_ids: list[str]) -> dict[str, int]: ...

    async def aggregate_message_stats(
        self,
        session_ids: list[str] | None = None,
    ) -> AggregateMessageStats: ...

    async def get_stats_by(self, group_by: str = "origin") -> dict[str, int]: ...


@runtime_checkable
class SemanticArchiveQueryStore(ArchiveMessageQueryStore, Protocol):
    """Archive query band needed to hydrate semantic facts in batch."""

    async def get_sessions_batch(self, ids: list[str]) -> list[SessionRecord]: ...

    async def get_messages_batch(
        self,
        session_ids: list[str],
        *,
        sort_key_since: float | None = None,
        sort_key_until: float | None = None,
        message_role: MessageRoleFilter = (),
    ) -> dict[str, list[MessageRecord]]: ...

    async def get_attachments_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[AttachmentRecord]]: ...


@runtime_checkable
class SessionOutputStore(SessionReader, Protocol):
    """Session output surface used by streaming and summary display helpers."""

    async def get_render_projection(
        self,
        session_id: str,
    ) -> SessionRenderProjection | None: ...

    async def get_session_stats(self, session_id: str) -> dict[str, int]: ...

    async def get_message_counts_batch(self, session_ids: list[str]) -> dict[str, int]: ...


@runtime_checkable
class SessionSemanticStatsStore(Protocol):
    """Semantic stats surface derived from sessions, messages, and blocks."""

    async def get_sessions_batch(self, ids: list[str]) -> list[SessionRecord]: ...

    async def get_messages_batch(
        self,
        session_ids: list[str],
        *,
        sort_key_since: float | None = None,
        sort_key_until: float | None = None,
        message_role: MessageRoleFilter = (),
    ) -> dict[str, list[MessageRecord]]: ...

    async def get_attachments_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[AttachmentRecord]]: ...

    async def get_actions_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, tuple[Action, ...]]: ...


@runtime_checkable
class SessionArchiveStatsStore(
    SessionOutputStore,
    SessionSemanticStatsStore,
    Protocol,
):
    """Archive stats/profile surface consumed by grouped CLI output helpers."""

    async def aggregate_message_stats(
        self,
        session_ids: list[str] | None = None,
    ) -> AggregateMessageStats: ...

    async def get_archive_stats(
        self,
        *,
        conn: aiosqlite.Connection | None = None,
    ) -> ArchiveStats: ...

    async def get_stats_by(self, group_by: str = "origin") -> dict[str, int]: ...

    async def get_session_profiles_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, SessionProfile]: ...

    async def get_many(self, session_ids: list[str]) -> list[Session]: ...


@runtime_checkable
class TagStore(Protocol):
    """Tag and metadata management interface."""

    async def list_tags(self, *, origin: str | None = None) -> dict[str, int]: ...

    async def get_metadata(self, session_id: str) -> JSONDocument: ...

    async def update_metadata(self, session_id: str, key: str, value: JSONValue) -> bool: ...

    async def delete_metadata(self, session_id: str, key: str) -> bool: ...

    async def add_tag(
        self,
        session_id: str,
        tag: str,
        *,
        author_ref: str | None = None,
        author_kind: str | None = None,
    ) -> bool: ...

    async def bulk_add_tags(self, session_ids: list[str], tags: list[str]) -> int: ...

    async def remove_tag(self, session_id: str, tag: str) -> bool: ...


@runtime_checkable
class SessionArchiveReadStore(SessionReader, SearchStore, Protocol):
    """Small read-side surface used by UI and resource adapters."""

    async def get_archive_stats(
        self,
        *,
        conn: aiosqlite.Connection | None = None,
    ) -> ArchiveStats: ...


@runtime_checkable
class RawPersistenceStore(Protocol):
    """Minimal raw-persistence surface used during acquisition."""

    async def save_raw_session(self, record: RawSessionRecord) -> bool: ...

    async def save_artifact_observation(self, record: ArtifactObservationRecord) -> bool: ...


@runtime_checkable
class RawValidationStore(Protocol):
    """Minimal raw-validation surface used by validation flows."""

    async def get_raw_sessions_batch(
        self,
        raw_ids: builtins.list[str],
    ) -> builtins.list[RawSessionRecord]: ...

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
    "VectorProvider",
    "ProgressCallback",
    "SessionReader",
    "SearchStore",
    "SessionQueryRuntimeStore",
    "ArchiveMessageQueryStore",
    "SemanticArchiveQueryStore",
    "SessionOutputStore",
    "SessionSemanticStatsStore",
    "SessionArchiveStatsStore",
    "TagStore",
    "RawPersistenceStore",
    "RawValidationStore",
]
