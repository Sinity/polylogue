"""Typed MCP payload models shared by server tools and resources."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from pydantic import RootModel
from typing_extensions import TypedDict

from polylogue.core.json import JSONDocument
from polylogue.mcp.context_pack import (
    ContextPackActionSummary as MCPContextPackActionSummary,
)
from polylogue.mcp.context_pack import (
    ContextPackDateRange as MCPContextPackDateRange,
)
from polylogue.mcp.context_pack import (
    ContextPackMessage as MCPContextPackMessage,
)
from polylogue.mcp.context_pack import (
    ContextPackPayload as MCPContextPackPayload,
)
from polylogue.mcp.context_pack import (
    ContextPackProject as MCPContextPackProject,
)
from polylogue.mcp.context_pack import (
    ContextPackProvenance as MCPContextPackProvenance,
)
from polylogue.mcp.context_pack import (
    ContextPackQueryContext as MCPContextPackQueryContext,
)
from polylogue.mcp.context_pack import (
    ContextPackSession as MCPContextPackSession,
)
from polylogue.mcp.context_pack import (
    ContextPackUnresolvedWork as MCPContextPackUnresolvedWork,
)
from polylogue.surfaces.payloads import (
    MutationResultPayload,
    SearchCursor,
    SearchEnvelope,
    SurfacePayloadModel,
    build_search_envelope,
    model_json_document,
    normalize_role,
)
from polylogue.surfaces.payloads import (
    ReaderActionAvailabilityPayload as MCPReaderActionAvailabilityPayload,
)
from polylogue.surfaces.payloads import (
    SessionDetailPayload as MCPSessionDetailPayload,
)
from polylogue.surfaces.payloads import (
    SessionMessagePayload as MCPMessagePayload,
)
from polylogue.surfaces.payloads import (
    SessionNeighborCandidatePayload as MCPSessionNeighborCandidatePayload,
)
from polylogue.surfaces.payloads import (
    SessionSearchHitPayload as MCPSessionSearchHitPayload,
)
from polylogue.surfaces.payloads import (
    SessionSummaryPayload as MCPSessionSummaryPayload,
)
from polylogue.surfaces.payloads import (
    TargetRefPayload as MCPTargetRefPayload,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polylogue.archive.models import Session
    from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics, QueryMissReason
    from polylogue.archive.query.search_hits import SessionSearchHit
    from polylogue.archive.session.neighbor_candidates import SessionNeighborCandidate
    from polylogue.archive.stats import ArchiveStats
    from polylogue.readiness import ReadinessCheck, ReadinessReport
    from polylogue.storage.runtime import RawSessionRecord
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSearchHit, ArchiveSessionSummary
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveBlockRow, ArchiveMessageRow, ArchiveSessionEnvelope

TRoot = TypeVar("TRoot")


class MCPRootPayload(RootModel[TRoot], Generic[TRoot]):
    """Root-model variant for list/map payloads."""

    def to_json(self, *, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=2, exclude_none=exclude_none)


class MCPErrorPayload(SurfacePayloadModel):
    error: str
    code: int | str | None = None
    detail: str | None = None
    tool: str | None = None
    session_id: str | None = None
    # Schema-mismatch surface (#1611): when an MCP tool body raises
    # ``SchemaVersionMismatchError`` the typed payload exposes both versions
    # so clients can render the actionable operator message without
    # re-parsing the error string.
    current_version: int | None = None
    expected_version: int | None = None
    # Embedding-readiness surface (#1503 AC4): when an MCP tool body
    # raises ``EmbeddingRetrievalNotReadyError`` the readiness status
    # enum value is exposed so clients can render the same actionable
    # operator message ``polylogue embed status`` does.
    readiness_status: str | None = None
    is_error: Literal[True] = True


class MCPFencedCodeBlock(TypedDict):
    language: str
    code: str


class MCPSessionSummaryListPayload(MCPRootPayload[list[MCPSessionSummaryPayload]]):
    root: list[MCPSessionSummaryPayload]


class MCPSessionSearchHitListPayload(MCPRootPayload[list[MCPSessionSearchHitPayload]]):
    root: list[MCPSessionSearchHitPayload]


class MCPSessionNeighborCandidateListPayload(MCPRootPayload[list[MCPSessionNeighborCandidatePayload]]):
    root: list[MCPSessionNeighborCandidatePayload]


class MCPQueryMissReasonPayload(SurfacePayloadModel):
    code: str
    severity: str
    summary: str
    detail: str | None = None
    count: int | None = None

    @classmethod
    def from_reason(cls, reason: QueryMissReason) -> MCPQueryMissReasonPayload:
        return cls(
            code=reason.code,
            severity=reason.severity,
            summary=reason.summary,
            detail=reason.detail,
            count=reason.count,
        )


class MCPQueryMissDiagnosticsPayload(SurfacePayloadModel):
    message: str
    filters: tuple[str, ...]
    reasons: tuple[MCPQueryMissReasonPayload, ...]
    archive_session_count: int | None = None
    raw_session_count: int | None = None

    @classmethod
    def from_diagnostics(cls, diagnostics: QueryMissDiagnostics) -> MCPQueryMissDiagnosticsPayload:
        return cls(
            message=diagnostics.message,
            filters=diagnostics.filters,
            reasons=tuple(MCPQueryMissReasonPayload.from_reason(reason) for reason in diagnostics.reasons),
            archive_session_count=diagnostics.archive_session_count,
            raw_session_count=diagnostics.raw_session_count,
        )


class MCPSessionQueryNoResultsPayload(SurfacePayloadModel):
    results: tuple[MCPSessionSummaryPayload, ...] = ()
    diagnostics: MCPQueryMissDiagnosticsPayload


class MCPSessionSearchNoResultsPayload(SurfacePayloadModel):
    results: tuple[MCPSessionSearchHitPayload, ...] = ()
    diagnostics: MCPQueryMissDiagnosticsPayload


class MCPPaginatedQueryResultPayload(SurfacePayloadModel):
    """Paginated query result envelope for list_sessions."""

    items: tuple[MCPSessionSummaryPayload, ...]
    total: int
    limit: int
    offset: int
    next_offset: int | None = None
    diagnostics: MCPQueryMissDiagnosticsPayload | None = None


#: MCP search uses the canonical :class:`~polylogue.surfaces.payloads.SearchEnvelope`
#: shape (#1266). The compatibility alias is retained so existing call sites keep
#: working; new code should import :class:`SearchEnvelope` directly from
#: ``polylogue.surfaces.payloads``.
MCPPaginatedSearchResultPayload = SearchEnvelope


class MCPSessionTreePayload(SurfacePayloadModel):
    """Bounded envelope for ``get_session_tree``.

    The tree of related sessions can be unbounded in principle; the
    envelope makes the size visible to callers and preserves room for
    future ``limit``/``offset`` pagination without breaking the response
    shape.
    """

    items: tuple[MCPSessionSummaryPayload, ...]
    total: int


class MCPSessionRefPayload(SurfacePayloadModel):
    """One session reference inside a topology payload (#1261)."""

    session_id: str
    source_name: str = ""
    title: str | None = None
    depth: int = 0


class MCPTopologyEdgePayload(SurfacePayloadModel):
    """One resolved or unresolved-native edge inside a topology payload."""

    child_id: str
    parent_id: str | None
    parent_native_id: str | None
    kind: str
    resolved: bool


class MCPSessionTopologyPayload(SurfacePayloadModel):
    """Typed envelope for ``get_session_topology`` (#1261 / #866 slice D).

    The four ref lists mirror the helper methods on
    :class:`~polylogue.insights.topology.SessionTopology` so callers do
    not have to re-derive lineage from the raw node/edge tuples.
    """

    target_id: str
    root_id: str
    cycle_detected: bool
    nodes: tuple[MCPSessionRefPayload, ...]
    edges: tuple[MCPTopologyEdgePayload, ...]
    ancestors: tuple[MCPSessionRefPayload, ...]
    descendants: tuple[MCPSessionRefPayload, ...]
    siblings: tuple[MCPSessionRefPayload, ...]
    thread: tuple[MCPSessionRefPayload, ...]


class MCPLogicalSessionPayload(SurfacePayloadModel):
    """Compact envelope for ``get_logical_session`` (#866)."""

    session_id: str
    root_id: str
    thread: tuple[MCPSessionRefPayload, ...]
    siblings: tuple[MCPSessionRefPayload, ...]
    descendants: tuple[MCPSessionRefPayload, ...]
    cycle_detected: bool = False


class MCPNeighborCandidatesPayload(SurfacePayloadModel):
    """Bounded envelope for ``neighbor_candidates``.

    Records the ``limit`` actually applied so the caller can recognise
    truncation and decide whether to widen the request.
    """

    items: tuple[MCPSessionNeighborCandidatePayload, ...]
    total: int
    limit: int


class MCPArchiveSessionSummaryPayload(SurfacePayloadModel):
    """Archive session summary payload."""

    session_id: str
    native_id: str
    origin: str
    source: str
    title: str | None
    created_at: str | None
    updated_at: str | None
    message_count: int
    word_count: int
    tags: tuple[str, ...]

    @classmethod
    def from_summary(cls, summary: ArchiveSessionSummary) -> MCPArchiveSessionSummaryPayload:
        return cls(
            session_id=summary.session_id,
            native_id=summary.native_id,
            origin=summary.origin,
            source=summary.origin,
            title=summary.title,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
            message_count=summary.message_count,
            word_count=summary.word_count,
            tags=summary.tags,
        )


class MCPArchiveSessionListPayload(SurfacePayloadModel):
    """Paginated envelope for archive session summaries."""

    items: tuple[MCPArchiveSessionSummaryPayload, ...]
    total: int
    limit: int
    offset: int
    origin: str | None = None


class MCPArchiveSearchHitPayload(SurfacePayloadModel):
    """Archive block-search hit payload."""

    rank: int
    session_id: str
    block_id: str
    message_id: str
    origin: str
    source: str
    title: str | None
    snippet: str

    @classmethod
    def from_hit(cls, hit: ArchiveSessionSearchHit) -> MCPArchiveSearchHitPayload:
        return cls(
            rank=hit.rank,
            session_id=hit.session_id,
            block_id=hit.block_id,
            message_id=hit.message_id,
            origin=hit.origin,
            source=hit.origin,
            title=hit.title,
            snippet=hit.snippet,
        )


class MCPArchiveSearchPayload(SurfacePayloadModel):
    """Paginated envelope for archive block-search hits."""

    items: tuple[MCPArchiveSearchHitPayload, ...]
    total: int
    limit: int
    query: str
    origin: str | None = None


class MCPArchiveBlockPayload(SurfacePayloadModel):
    """Archive message block payload."""

    block_id: str
    message_id: str
    block_type: str
    text: str | None

    @classmethod
    def from_block(cls, block: ArchiveBlockRow) -> MCPArchiveBlockPayload:
        return cls(
            block_id=block.block_id,
            message_id=block.message_id,
            block_type=block.block_type,
            text=block.text,
        )


class MCPArchiveMessagePayload(SurfacePayloadModel):
    """Archive message payload."""

    message_id: str
    native_id: str | None
    role: str
    position: int
    variant_index: int
    is_active_path: bool
    is_active_leaf: bool
    blocks: tuple[MCPArchiveBlockPayload, ...]

    @classmethod
    def from_message(cls, message: ArchiveMessageRow) -> MCPArchiveMessagePayload:
        return cls(
            message_id=message.message_id,
            native_id=message.native_id,
            role=message.role,
            position=message.position,
            variant_index=message.variant_index,
            is_active_path=message.is_active_path,
            is_active_leaf=message.is_active_leaf,
            blocks=tuple(MCPArchiveBlockPayload.from_block(block) for block in message.blocks),
        )


class MCPArchiveSessionPayload(SurfacePayloadModel):
    """Archive full session envelope."""

    session_id: str
    native_id: str
    origin: str
    source: str
    title: str | None
    active_leaf_message_id: str | None
    messages: tuple[MCPArchiveMessagePayload, ...]

    @classmethod
    def from_session(cls, session: ArchiveSessionEnvelope) -> MCPArchiveSessionPayload:
        return cls(
            session_id=session.session_id,
            native_id=session.native_id,
            origin=session.origin,
            source=session.origin,
            title=session.title,
            active_leaf_message_id=session.active_leaf_message_id,
            messages=tuple(MCPArchiveMessagePayload.from_message(message) for message in session.messages),
        )


def session_summary_list_payload(
    sessions: Sequence[Session],
) -> MCPSessionSummaryListPayload:
    return MCPSessionSummaryListPayload(root=[MCPSessionSummaryPayload.from_session(conv) for conv in sessions])


def session_query_result_payload(
    sessions: Sequence[Session],
    *,
    total: int,
    limit: int,
    offset: int,
    diagnostics: QueryMissDiagnostics | None = None,
) -> MCPPaginatedQueryResultPayload:
    next_offset = offset + len(sessions) if len(sessions) == limit and offset + limit < total else None
    return MCPPaginatedQueryResultPayload(
        items=tuple(MCPSessionSummaryPayload.from_session(conv) for conv in sessions),
        total=total,
        limit=limit,
        offset=offset,
        next_offset=next_offset,
        diagnostics=(MCPQueryMissDiagnosticsPayload.from_diagnostics(diagnostics) if diagnostics else None),
    )


def session_search_hit_list_payload(
    hits: Sequence[SessionSearchHit],
) -> MCPSessionSearchHitListPayload:
    return MCPSessionSearchHitListPayload(
        root=[
            MCPSessionSearchHitPayload.from_search_hit(
                hit,
                message_count=hit.summary.message_count,
            )
            for hit in hits
        ]
    )


def session_neighbor_candidate_list_payload(
    candidates: Sequence[SessionNeighborCandidate],
) -> MCPSessionNeighborCandidateListPayload:
    return MCPSessionNeighborCandidateListPayload(
        root=[MCPSessionNeighborCandidatePayload.from_candidate(candidate) for candidate in candidates]
    )


def session_tree_payload(
    sessions: Sequence[Session],
) -> MCPSessionTreePayload:
    items = tuple(MCPSessionSummaryPayload.from_session(conv) for conv in sessions)
    return MCPSessionTreePayload(items=items, total=len(items))


def _ref_payload(ref: object) -> MCPSessionRefPayload:
    # Imported lazily to avoid pulling insights/topology into the module
    # import graph at module load time.
    from polylogue.insights.topology import SessionRef

    assert isinstance(ref, SessionRef)
    return MCPSessionRefPayload(
        session_id=str(ref.session_id),
        source_name=ref.source_name,
        title=ref.title,
        depth=ref.depth,
    )


def session_topology_payload(topology: object, *, session_id: str) -> MCPSessionTopologyPayload:
    """Build the typed MCP payload for ``get_session_topology`` (#1261)."""
    from polylogue.insights.topology import SessionTopology

    assert isinstance(topology, SessionTopology)
    nodes = tuple(_ref_payload(node.as_ref()) for node in topology.nodes)
    edges = tuple(
        MCPTopologyEdgePayload(
            child_id=str(edge.child_id),
            parent_id=str(edge.parent_id) if edge.parent_id is not None else None,
            parent_native_id=edge.parent_native_id,
            kind=str(edge.kind.value),
            resolved=edge.resolved,
        )
        for edge in topology.edges
    )
    return MCPSessionTopologyPayload(
        target_id=str(topology.target_id),
        root_id=str(topology.root_id),
        cycle_detected=topology.cycle_detected,
        nodes=nodes,
        edges=edges,
        ancestors=tuple(_ref_payload(ref) for ref in topology.ancestor_refs(session_id)),
        descendants=tuple(_ref_payload(ref) for ref in topology.descendant_refs(session_id)),
        siblings=tuple(_ref_payload(ref) for ref in topology.sibling_refs(session_id)),
        thread=tuple(_ref_payload(ref) for ref in topology.thread_refs(session_id)),
    )


def logical_session_payload(logical_session: object) -> MCPLogicalSessionPayload:
    """Build the typed MCP payload for ``get_logical_session`` (#866)."""
    from polylogue.insights.topology import LogicalSession

    assert isinstance(logical_session, LogicalSession)
    return MCPLogicalSessionPayload(
        session_id=str(logical_session.session_id),
        root_id=str(logical_session.root_id),
        thread=tuple(_ref_payload(ref) for ref in logical_session.thread),
        siblings=tuple(_ref_payload(ref) for ref in logical_session.siblings),
        descendants=tuple(_ref_payload(ref) for ref in logical_session.descendants),
        cycle_detected=logical_session.cycle_detected,
    )


def neighbor_candidates_payload(
    candidates: Sequence[SessionNeighborCandidate],
    *,
    limit: int,
) -> MCPNeighborCandidatesPayload:
    items = tuple(MCPSessionNeighborCandidatePayload.from_candidate(candidate) for candidate in candidates)
    return MCPNeighborCandidatesPayload(items=items, total=len(items), limit=limit)


def session_search_result_payload(
    hits: Sequence[SessionSearchHit],
    *,
    total: int,
    limit: int,
    offset: int,
    diagnostics: QueryMissDiagnostics | None = None,
    query: str = "",
    retrieval_lane: str = "auto",
    sort: str | None = None,
    cursor: SearchCursor | None = None,
) -> SearchEnvelope:
    """Build the canonical :class:`SearchEnvelope` for an MCP search call.

    Delegates to :func:`polylogue.surfaces.payloads.build_search_envelope`
    so the cursor/next_offset/ranking-policy fields match CLI, daemon HTTP,
    and the Python API. ``retrieval_lane`` falls back to ``"auto"`` when the
    caller does not know which lane ran; downstream surfaces SHOULD pass the
    resolved lane (from the first hit, or from the query spec).
    """
    from polylogue.surfaces.payloads import QueryMissDiagnosticsPayload

    resolved_lane = retrieval_lane
    if resolved_lane in {"", "auto"} and hits:
        resolved_lane = hits[0].retrieval_lane
    hit_payloads = [
        MCPSessionSearchHitPayload.from_search_hit(
            hit,
            message_count=hit.summary.message_count,
        )
        for hit in hits
    ]
    diag_payload = QueryMissDiagnosticsPayload.from_diagnostics(diagnostics) if diagnostics else None
    return build_search_envelope(
        hit_payloads,
        total=total,
        limit=limit,
        offset=offset,
        query=query,
        retrieval_lane=resolved_lane,
        sort=sort,
        diagnostics=diag_payload,
        cursor=cursor,
    )


class MCPArchiveStatsPayload(SurfacePayloadModel):
    total_sessions: int
    total_messages: int
    origins: dict[str, int]
    embedded_sessions: int | None = None
    embedded_messages: int | None = None
    pending_embedding_sessions: int | None = None
    embedding_coverage_percent: float | None = None
    stale_embedding_messages: int | None = None
    messages_missing_embedding_provenance: int | None = None
    embedding_readiness_status: str | None = None
    embedding_models: dict[str, int] | None = None
    embedding_dimensions: dict[int, int] | None = None
    embedding_oldest_at: str | None = None
    embedding_newest_at: str | None = None
    db_size_mb: float | int | None = None

    @classmethod
    def from_archive_stats(
        cls,
        archive_stats: ArchiveStats,
        *,
        include_embedded: bool,
        include_db_size: bool,
    ) -> MCPArchiveStatsPayload:
        return cls(
            total_sessions=archive_stats.total_sessions,
            total_messages=archive_stats.total_messages,
            origins=archive_stats.origins,
            embedded_sessions=archive_stats.embedded_sessions if include_embedded else None,
            embedded_messages=archive_stats.embedded_messages if include_embedded else None,
            pending_embedding_sessions=(archive_stats.pending_embedding_sessions if include_embedded else None),
            embedding_coverage_percent=(
                round(float(archive_stats.embedding_coverage), 1) if include_embedded else None
            ),
            stale_embedding_messages=archive_stats.stale_embedding_messages if include_embedded else None,
            messages_missing_embedding_provenance=(
                archive_stats.messages_missing_embedding_provenance if include_embedded else None
            ),
            embedding_readiness_status=archive_stats.embedding_readiness_status if include_embedded else None,
            embedding_models=archive_stats.embedding_models if include_embedded else None,
            embedding_dimensions=archive_stats.embedding_dimensions if include_embedded else None,
            embedding_oldest_at=archive_stats.embedding_oldest_at if include_embedded else None,
            embedding_newest_at=archive_stats.embedding_newest_at if include_embedded else None,
            db_size_mb=(
                round(archive_stats.db_size_bytes / 1_048_576, 1)
                if include_db_size and archive_stats.db_size_bytes
                else 0
                if include_db_size
                else None
            ),
        )


class MCPMutationStatusPayload(SurfacePayloadModel):
    status: str
    session_id: str | None = None
    tag: str | None = None
    key: str | None = None
    index_exists: bool | None = None
    indexed_messages: int | None = None
    session_count: int | None = None
    outcome: str | None = None
    """Tag idempotency outcome: ``added``, ``no_op``, ``removed``, or ``not_present``."""


class MCPTagCountsPayload(MCPRootPayload[dict[str, int]]):
    root: dict[str, int]


class MCPMetadataPayload(SurfacePayloadModel):
    root: dict[str, object]

    @classmethod
    def from_document(cls, document: JSONDocument) -> MCPMetadataPayload:
        return cls(root=dict(document))

    def to_json(self, *, exclude_none: bool = False) -> str:
        del exclude_none
        return json.dumps(self.root, indent=2)


class MCPEmbeddingStatusPayload(SurfacePayloadModel):
    """Canonical embedding readiness payload exposed over MCP (#1503)."""

    root: dict[str, object]

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> MCPEmbeddingStatusPayload:
        return cls(root=dict(payload))

    def to_json(self, *, exclude_none: bool = False) -> str:
        del exclude_none
        return json.dumps(self.root, indent=2)


class MCPEmbeddingPreflightPayload(SurfacePayloadModel):
    """Canonical embedding catch-up preflight payload exposed over MCP (#1503)."""

    root: dict[str, object]

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> MCPEmbeddingPreflightPayload:
        return cls(root=dict(payload))

    def to_json(self, *, exclude_none: bool = False) -> str:
        del exclude_none
        return json.dumps(self.root, indent=2)


class MCPUserMarkPayload(SurfacePayloadModel):
    target_type: str = "session"
    target_id: str
    session_id: str
    message_id: str | None = None
    mark_type: str
    created_at: str


class MCPUserMarkListPayload(SurfacePayloadModel):
    items: tuple[MCPUserMarkPayload, ...]
    total: int


class MCPUserAnnotationPayload(SurfacePayloadModel):
    annotation_id: str
    target_type: str
    target_id: str
    session_id: str
    message_id: str | None = None
    note_text: str
    created_at: str
    updated_at: str


class MCPUserAnnotationListPayload(SurfacePayloadModel):
    items: tuple[MCPUserAnnotationPayload, ...]
    total: int


class MCPSavedViewPayload(SurfacePayloadModel):
    view_id: str
    name: str
    query: dict[str, object]
    created_at: str


class MCPSavedViewListPayload(SurfacePayloadModel):
    items: tuple[MCPSavedViewPayload, ...]
    total: int


class MCPRecallPackPayload(SurfacePayloadModel):
    pack_id: str
    label: str
    session_ids: tuple[str, ...]
    payload: dict[str, object]
    created_at: str


class MCPRecallPackListPayload(SurfacePayloadModel):
    items: tuple[MCPRecallPackPayload, ...]
    total: int


class MCPReaderWorkspacePayload(SurfacePayloadModel):
    workspace_id: str
    name: str
    mode: str
    open_targets: tuple[dict[str, object], ...]
    layout: dict[str, object]
    active_target: dict[str, object]
    created_at: str
    updated_at: str


class MCPReaderWorkspaceListPayload(SurfacePayloadModel):
    items: tuple[MCPReaderWorkspacePayload, ...]
    total: int


class MCPStatsByPayload(MCPRootPayload[dict[str, int]]):
    root: dict[str, int]


class MCPMessagesListPayload(SurfacePayloadModel):
    """Paginated message list response for get_messages tool."""

    session_id: str
    messages: tuple[MCPMessagePayload, ...]
    total: int
    limit: int
    offset: int


class MCPRawArtifactPayload(SurfacePayloadModel):
    """One raw archive artifact for the raw_artifacts tool."""

    raw_id: str
    source_name: str | None = None
    source_path: str
    blob_size: int
    acquired_at: str
    parsed_at: str | None = None
    parse_error: str | None = None
    validated_at: str | None = None
    validation_status: str | None = None
    validation_error: str | None = None

    @classmethod
    def from_record(cls, record: RawSessionRecord) -> MCPRawArtifactPayload:
        return cls(
            raw_id=record.raw_id,
            source_name=record.source_name,
            source_path=record.source_path,
            blob_size=record.blob_size,
            acquired_at=record.acquired_at,
            parsed_at=record.parsed_at,
            parse_error=record.parse_error,
            validated_at=record.validated_at,
            validation_status=str(record.validation_status) if record.validation_status else None,
            validation_error=record.validation_error,
        )


class MCPRawArtifactsListPayload(SurfacePayloadModel):
    """Paginated raw archive artifact response for the raw_artifacts tool."""

    session_id: str
    raw_artifacts: tuple[MCPRawArtifactPayload, ...]
    total: int
    limit: int
    offset: int


class MCPReadinessCheckPayload(SurfacePayloadModel):
    name: str
    status: str
    count: int | None = None
    detail: str | None = None

    @classmethod
    def from_check(
        cls,
        check: ReadinessCheck,
        *,
        include_counts: bool,
        include_detail: bool,
    ) -> MCPReadinessCheckPayload:
        return cls(
            name=check.name,
            status=check.status.value,
            count=check.count if include_counts else None,
            detail=check.detail if include_detail else None,
        )


def _extract_readiness_source(report: ReadinessReport) -> str | None:
    provenance = report.provenance
    if provenance.source is None:
        return None
    return provenance.source


class MCPReadinessReportPayload(SurfacePayloadModel):
    checks: list[MCPReadinessCheckPayload]
    summary: str | dict[str, int]
    source: str | None = None

    @classmethod
    def from_report(
        cls,
        report: ReadinessReport,
        *,
        include_counts: bool,
        include_detail: bool,
        include_cached: bool,
    ) -> MCPReadinessReportPayload:
        return cls(
            checks=[
                MCPReadinessCheckPayload.from_check(
                    check,
                    include_counts=include_counts,
                    include_detail=include_detail,
                )
                for check in report.checks
            ],
            summary=report.summary,
            source=_extract_readiness_source(report) if include_cached else None,
        )


__all__ = [
    "MCPArchiveStatsPayload",
    "MCPContextPackActionSummary",
    "MCPContextPackSession",
    "MCPContextPackDateRange",
    "MCPContextPackMessage",
    "MCPContextPackPayload",
    "MCPContextPackProject",
    "MCPContextPackProvenance",
    "MCPContextPackQueryContext",
    "MCPContextPackUnresolvedWork",
    "MCPSessionDetailPayload",
    "MCPSessionNeighborCandidateListPayload",
    "MCPSessionNeighborCandidatePayload",
    "MCPSessionQueryNoResultsPayload",
    "MCPSessionSearchHitListPayload",
    "MCPSessionSearchHitPayload",
    "MCPSessionSearchNoResultsPayload",
    "MCPSessionSummaryListPayload",
    "MCPSessionSummaryPayload",
    "MCPErrorPayload",
    "MCPFencedCodeBlock",
    "MCPMessagePayload",
    "MCPMessagesListPayload",
    "MCPMetadataPayload",
    "MCPLogicalSessionPayload",
    "MCPMutationStatusPayload",
    "MCPReaderActionAvailabilityPayload",
    "MutationResultPayload",
    "MCPNeighborCandidatesPayload",
    "MCPPaginatedQueryResultPayload",
    "MCPPaginatedSearchResultPayload",
    "MCPQueryMissDiagnosticsPayload",
    "MCPQueryMissReasonPayload",
    "MCPRawArtifactPayload",
    "MCPRawArtifactsListPayload",
    "MCPReadinessCheckPayload",
    "MCPReadinessReportPayload",
    "MCPRootPayload",
    "MCPArchiveBlockPayload",
    "MCPArchiveMessagePayload",
    "MCPArchiveSearchHitPayload",
    "MCPArchiveSearchPayload",
    "MCPArchiveSessionListPayload",
    "MCPArchiveSessionPayload",
    "MCPArchiveSessionSummaryPayload",
    "MCPSessionTreePayload",
    "MCPStatsByPayload",
    "MCPTagCountsPayload",
    "MCPSavedViewListPayload",
    "MCPSavedViewPayload",
    "MCPRecallPackListPayload",
    "MCPRecallPackPayload",
    "MCPUserMarkListPayload",
    "MCPUserMarkPayload",
    "MCPUserAnnotationListPayload",
    "MCPUserAnnotationPayload",
    "MCPTargetRefPayload",
    "session_neighbor_candidate_list_payload",
    "session_query_result_payload",
    "session_search_hit_list_payload",
    "session_search_result_payload",
    "session_summary_list_payload",
    "model_json_document",
    "neighbor_candidates_payload",
    "logical_session_payload",
    "normalize_role",
    "session_tree_payload",
    "session_topology_payload",
]
