"""Protocol definitions for pluggable backends in Polylogue.

Only protocols with 2+ implementations earn their existence here:
- VectorProvider: sqlite-vec (local retained reads; credentials for acquisition)

``SearchProvider`` (FTS5, Hybrid) was removed (polylogue-a7xr.10): both
implementations had zero production consumers — production full-text and
hybrid retrieval has always queried FTS5/vector tables and fused results
inline rather than through a swappable provider abstraction.

``SessionReader``, ``SearchStore``, ``ArchiveMessageQueryStore``,
``SemanticArchiveQueryStore``, ``SessionSemanticStatsStore``, and
``SessionArchiveReadStore`` were removed (polylogue-a7xr.11): each had zero
consumers anywhere in the tree. Their methods that surviving protocols
actually needed (``SessionOutputStore``, ``SessionArchiveStatsStore``) are
inlined directly rather than inherited from a now-deleted shared base.
"""

from __future__ import annotations

import builtins
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    import sqlite3

    import aiosqlite

    from polylogue.archive.actions.actions import Action
    from polylogue.archive.message.models import Message
    from polylogue.archive.message.roles import MessageRoleFilter
    from polylogue.archive.query.search_hits import SessionSearchHit
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.archive.session.session_profile import SessionProfile
    from polylogue.archive.stats import ArchiveStats
    from polylogue.core.enums import MaterialOrigin, Provider, ValidationMode, ValidationStatus
    from polylogue.core.json import JSONDocument, JSONValue
    from polylogue.core.types import SessionId
    from polylogue.storage.query_models import SessionRecordQuery
    from polylogue.storage.runtime import (
        ArtifactObservationRecord,
        AttachmentRecord,
        MessageRecord,
        RawSessionRecord,
        SessionRecord,
    )
    from polylogue.storage.sqlite.archive_tiers.raw_admission import (
        RawAdmissionExecution,
        RawAdmissionRequest,
    )
    from polylogue.storage.sqlite.queries.stats import AggregateMessageStats


class ArchiveRootOwner(Protocol):
    """Minimal archive handle required by live-ingest coordination."""

    @property
    def archive_root(self) -> Path: ...


_SimilarityT = TypeVar("_SimilarityT")


@dataclass(frozen=True, slots=True)
class ScopedVectorQuery:
    """Exact full-scope session order, with one actual message witness per row.

    Rows carry message IDs and ascending minimum L2 distance. The complete
    ranking relation is evaluated before traversal; the consumer still owns
    recording successful full consumption separately from this exactness proof.
    """

    rows: Iterator[tuple[str, float]]
    exact: bool = True
    population: str = "eligible-sessions"


@runtime_checkable
class VectorProvider(Protocol):
    """Vector search provider for semantic similarity.

    Implementations: SqliteVecProvider (polylogue.storage.search_providers.sqlite_vec)
    Uses Voyage AI embeddings stored in sqlite-vec.
    """

    model: str

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Synchronously return ranked ``(message_id, distance)`` search results."""
        ...

    def query_by_session(self, session_id: str, limit: int = 10) -> list[tuple[str, float]]:
        """Rank messages by similarity to a stored session's own embeddings.

        Reads ``session_id``'s already-materialized message vectors and scores
        each occurrence by its minimum L2 distance over all seed outputs, returning ranked ``(message_id, distance)`` hits with
        the seed session's own messages excluded (so the seed never ranks against
        itself). No re-embedding occurs — only stored vectors are read. Raises a typed
        error when the seed session has no stored embeddings, never silently returning
        an empty or unfiltered result.
        """
        ...

    def scoped_query(
        self,
        session_ids: Iterable[str],
        *,
        index_connection: sqlite3.Connection,
        configure_connection: Callable[[sqlite3.Connection], None],
        check_cancelled: Callable[[], None],
        text: str | None = None,
        seed_session_id: str | None = None,
    ) -> AbstractContextManager[ScopedVectorQuery]:
        """Rank all supplied eligible sessions, then traverse their witnesses.

        Exactly one seed is required. Session seeds use all distinct retained
        outputs and exclude their own session; text seeds acquire one query
        embedding. The borrowed canonical index frame remains caller-owned.
        The context settles provider cursors and owned handles on every exit.
        """
        ...

    async def read_similarity(
        self,
        *,
        index_path: Path,
        project: Callable[[sqlite3.Connection, int, list[tuple[str, float]]], _SimilarityT],
        text: str | None = None,
        seed_session_id: str | None = None,
        limit: int = 10,
    ) -> _SimilarityT:
        """Rank sessions and project them on the same selected index snapshot.

        Exactly one seed is required. Text acquires one query embedding; session
        seeds use all distinct retained outputs without acquisition. Select one
        best actual occurrence per session before the page limit. The provider
        dispatches owned snapshots to a worker and keeps borrowed snapshots on
        their creating thread. Projection finishes before owned handles close.
        """
        ...


@runtime_checkable
class ProgressCallback(Protocol):
    """Progress callback shared by pipeline stages and CLI observers."""

    def __call__(self, amount: int, desc: str | None = None) -> None:
        """Report incremental progress."""
        ...


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
class SessionOutputStore(Protocol):
    """Session output surface used by streaming and summary display helpers."""

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

    async def get_session_stats(self, session_id: str) -> dict[str, int]: ...

    async def get_message_counts_batch(self, session_ids: builtins.list[str]) -> dict[str, int]: ...


@runtime_checkable
class SessionArchiveStatsStore(
    SessionOutputStore,
    Protocol,
):
    """Archive stats/profile surface consumed by grouped CLI output helpers."""

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
class RawPersistenceStore(Protocol):
    """Minimal raw-persistence surface used during acquisition."""

    async def admit_raw(self, request: RawAdmissionRequest) -> RawAdmissionExecution: ...

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
    "ScopedVectorQuery",
    "ProgressCallback",
    "SessionOutputStore",
    "SessionArchiveStatsStore",
    "TagStore",
    "RawPersistenceStore",
    "RawValidationStore",
]
