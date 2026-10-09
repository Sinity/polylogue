"""Archive/query domain methods for the async Polylogue facade."""

from __future__ import annotations

import builtins
import itertools
import json
import random
import sqlite3
from collections.abc import AsyncGenerator, Callable, Collection, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import aclosing, closing, contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, TypeVar, cast

from polylogue.analysis.archive import (
    SessionProfileInsight,
    SessionProfileInsightQuery,
)
from polylogue.analysis.feedback import LearningCorrection, parse_correction_kind
from polylogue.analysis.insight_reads import read_insight_page
from polylogue.api.archive_reads import ArchiveReadCapability
from polylogue.api.facade_client import submit_facade_product
from polylogue.archive.actions.actions import Action
from polylogue.archive.blackboard import BlackboardNote, BlackboardPage
from polylogue.archive.context_models import (
    DEFAULT_CONTEXT_IMAGE_MAX_CHARS_PER_MESSAGE,
    DEFAULT_CONTEXT_IMAGE_MAX_MESSAGES_PER_SESSION,
)
from polylogue.archive.hydration import archive_envelope_to_session, archive_summary_to_domain
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import MessageRoleFilter
from polylogue.archive.message.types import MessageType, validate_message_type_filter
from polylogue.archive.query.spec import (
    DEFAULT_SESSION_LIST_LIMIT,
    normalize_action_sequence,
    normalize_action_terms,
    parse_query_date,
    resolve_default_root_filter,
)
from polylogue.archive.query.transaction import archive_read_context, run_archive_read
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.config import active_archive_root as _active_archive_root
from polylogue.context.scheduler import (
    ContextLedgerRecord,
    read_context_ledger,
)
from polylogue.core.enums import AssertionKind, AssertionStatus, MaterialOrigin, Origin
from polylogue.core.errors import ArchiveTierUnavailableError, DatabaseError, SchemaRefusalError
from polylogue.core.json import JSONDocument
from polylogue.core.timestamps import parse_archive_datetime
from polylogue.core.types import SessionId
from polylogue.core.user_state_targets import TARGET_MESSAGE, TARGET_SESSION
from polylogue.logging import WARNING, emit
from polylogue.operations.user_state_resolution import (
    resolve_durable_user_state_session_id as _resolve_durable_user_state_session_id,
)
from polylogue.storage.derived.session.records import SessionProfileRecord
from polylogue.storage.derived.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.query_models import SessionRecordQuery
from polylogue.storage.runtime import LineageCompleteness, LineageTruncationReason
from polylogue.storage.search.models import SearchHit, SearchResult
from polylogue.storage.search.query_builders import session_web_url
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionIdentity, ArchiveSessionSummary, IndexStatus
from polylogue.storage.sqlite.archive_tiers.context_delivery_write import (
    ArchiveContextDeliveryEnvelope,
    ArchiveContextDeliveryPage,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveSessionEnvelope,
)
from polylogue.storage.sqlite.connection_profile import (
    ReadFrameExpiredError,
    StaleContinuationError,
)
from polylogue.storage.sqlite.connection_profile import (
    attach_readonly_database as attach_readonly_database,
)
from polylogue.storage.sqlite.connection_profile import (
    open_readonly_connection as open_readonly_connection,
)
from polylogue.storage.sqlite.connection_profile import read_frame as read_frame
from polylogue.storage.sqlite.queries.message_query_reads import MessageTypeName
from polylogue.surfaces.chronicle import (
    ChronicleProjectionPayload,
)
from polylogue.surfaces.temporal_evidence import (
    TemporalEvidenceWindow,
)

if TYPE_CHECKING:
    from polylogue.analysis.audit import InsightRigorAuditQuery, InsightRigorAuditReport
    from polylogue.analysis.export_bundle_contracts import InsightExportBundleRequest, InsightExportBundleResult
    from polylogue.analysis.fable_packet_contracts import FableDelegationPacket
    from polylogue.analysis.hermes_health_contracts import HermesIntegrationHealth
    from polylogue.analysis.judgment.types import ComparativeJudgment
    from polylogue.analysis.orchestration_evidence import SessionOrchestrationEvidence
    from polylogue.analysis.pathology import PathologyReport
    from polylogue.analysis.portfolio import PortfolioBundle
    from polylogue.analysis.postmortem import PostmortemBundle
    from polylogue.analysis.readiness import InsightReadinessQuery, InsightReadinessReport
    from polylogue.analysis.resume import ResumeBrief
    from polylogue.analysis.resume_contracts import ResumeCandidate
    from polylogue.analysis.transforms import SessionDigest
    from polylogue.annotations.importer import AnnotationBatchImportRequest, AnnotationBatchImportResult
    from polylogue.annotations.join_contracts import AnnotationStructuralJoinResult
    from polylogue.annotations.schema import AnnotationSchemaRegistry
    from polylogue.api import Polylogue
    from polylogue.archive.context_models import ContextImage, ContextOmission, ContextSpec
    from polylogue.archive.filter.filters import SessionFilter
    from polylogue.archive.message.models import Message
    from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics
    from polylogue.archive.query.search_hits import SearchHitResults, SessionSearchHit
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.archive.session.neighbor_candidates import SessionNeighborCandidate
    from polylogue.archive.stats import ArchiveStats as StorageArchiveStats
    from polylogue.config import Config
    from polylogue.context.claude_agent_dispatch_correlation import ClaudeAgentDispatchCorrelation
    from polylogue.context.codex_spawn_edge_correlation import CodexSpawnEdgeReconciliation
    from polylogue.context.hermes_delivery_correlation import HermesContextDeliveryCorrelation
    from polylogue.core.protocols import ProgressCallback
    from polylogue.operations import ArchiveStats
    from polylogue.operations.transcript_window import TranscriptWindow
    from polylogue.readiness import ReadinessReport
    from polylogue.sources.parsers.hermes_lifecycle import HermesLifecycleReconciliation
    from polylogue.storage.derived.session.runtime import SessionInsightCounts
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.search.models import SearchResult
    from polylogue.storage.sqlite.archive_tiers.archive import (
        ArchiveSessionSearchHit,
        ArchiveSessionSummary,
        ArchiveStore,
    )
    from polylogue.storage.sqlite.archive_tiers.user_settings_write import ArchiveUserSettingEnvelope
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        ArchiveAssertionCandidateReviewEnvelope,
        ArchiveAssertionEnvelope,
    )
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveSessionEnvelope
    from polylogue.storage.usage import ProviderUsageReport, SessionUsageReconciliation
    from polylogue.surfaces.payloads import (
        ArchiveDebtListPayload,
        AssertionBulkJudgmentPayload,
        AssertionCandidateQueueHealthPayload,
        AssertionCandidateQueueState,
        AssertionCandidateReviewListPayload,
        AssertionClaimPayload,
        AssertionEvidenceResolutionState,
        AssertionJudgmentResultPayload,
        BulkTagMutationResult,
        DeleteSessionPreview,
        DeleteSessionResult,
        FacetsResponse,
        ImportExplainPayload,
        MetadataMutationResult,
        OtelProjectionPayload,
        PublicRefResolutionPayload,
        QueryUnitResultEnvelope,
        SearchEnvelope,
        SessionSearchHitPayload,
        TagMutationResult,
    )


_FACET_CORE_FAMILIES = (
    "total_counts",
    "origins",
    "tags",
)

_FACET_DEFERRED_FAMILIES = (
    "repos",
    "role_counts",
    "material_origins",
    "message_types",
    "action_types",
    "has_flags",
)

_FACET_COMPLETE_FAMILIES = _FACET_CORE_FAMILIES + _FACET_DEFERRED_FAMILIES

_ReadResultT = TypeVar("_ReadResultT")
_T = TypeVar("_T")

_CANDIDATE_CAPTURE_KIND_MAP: dict[str, AssertionKind] = {
    "note": AssertionKind.NOTE,
    "claim": AssertionKind.DECISION,
    "correction": AssertionKind.CORRECTION,
    "lesson": AssertionKind.LESSON,
    "caveat": AssertionKind.CAVEAT,
    "highlight": AssertionKind.HIGHLIGHT,
    "prompt_eval": AssertionKind.PROMPT_EVAL,
}


def candidate_capture_kind(value: str) -> AssertionKind:
    """Resolve the stable terminal/MCP candidate-capture kind vocabulary."""

    try:
        return _CANDIDATE_CAPTURE_KIND_MAP[value]
    except KeyError as exc:
        choices = ", ".join(_CANDIDATE_CAPTURE_KIND_MAP)
        raise ValueError(f"candidate kind must be one of: {choices}") from exc


_FACET_FAMILY_METADATA: dict[str, dict[str, object]] = {
    "total_counts": {
        "label": "Total counts",
        "source": "session summaries",
        "canonicalization": "unique sessions plus stored message counts",
        "expensive": False,
    },
    "origins": {
        "label": "Provider origins",
        "source": "session summaries",
        "canonicalization": "provider/archive origin; not a repo or authoredness signal",
        "expensive": False,
    },
    "tags": {
        "label": "User tags",
        "source": "session summaries",
        "canonicalization": "session tags de-duplicated within each session",
        "expensive": False,
    },
    "repos": {
        "label": "Canonical repositories",
        "source": "session_repos + repos",
        "canonicalization": "prefer repo_name or origin_url; omit archive/path tokens that are not product repo identities",
        "expensive": True,
    },
    "role_counts": {
        "label": "Provider-role counts",
        "source": "messages.role",
        "canonicalization": "provider-reported message role; not authoredness",
        "expensive": True,
    },
    "material_origins": {
        "label": "Material origins",
        "source": "messages.material_origin",
        "canonicalization": "authoredness/protocol provenance; separates human text from runtime or assistant material",
        "expensive": True,
    },
    "message_types": {
        "label": "Message content types",
        "source": "messages.message_type",
        "canonicalization": "normalized message content kind",
        "expensive": True,
    },
    "action_types": {
        "label": "Action types",
        "source": "actions.semantic_type",
        "canonicalization": "normalized semantic action kind",
        "expensive": True,
    },
    "has_flags": {
        "label": "Content flags",
        "source": "messages.has_*",
        "canonicalization": "boolean message feature counters",
        "expensive": True,
    },
}

_NOISY_REPO_LABELS = {
    "",
    ".agent",
    ".cache",
    ".claude",
    ".config",
    ".git",
    ".local",
    "archive",
    "archives",
    "browser-capture",
    "captures",
    "chatlog",
    "codex",
    "data",
    "download",
    "downloads",
    "exports",
    "home",
    "inbox",
    "logs",
    "misc",
    "raw",
    "sessions",
    "source",
    "tmp",
    "var",
}


@dataclass(frozen=True, slots=True)
class SessionTranscriptPage:
    """A session header plus a declared ``[offset, offset + limit)`` window.

    ``session.messages`` holds only the window, so the session's true totals
    are carried here instead of being recomputed from the rows served:
    ``Session.word_count`` and ``len(session.messages)`` count what was read,
    which under a declared window is the window, not the session. A caller
    that reports either as a session length turns a bounded read into a wrong
    answer, so the bound and the truth travel together (polylogue-2go3o).

    ``limit is None`` is the whole transcript -- a declared request for every
    row, not a cap that happened not to bite.
    """

    session: Session
    total_message_count: int
    word_count: int
    limit: int | None
    offset: int


from polylogue.core.errors import SessionNotFoundError as SessionNotFoundError  # noqa: E402
from polylogue.operations.archive_mutation import MutationBlockedError as MutationBlockedError  # noqa: E402
from polylogue.operations.archive_mutation import (  # noqa: E402
    MutationTargetVanishedError as MutationTargetVanishedError,
)


def _read_session_transcript_page(
    archive: ArchiveStore,
    session_id: str,
    *,
    limit: int | None,
    offset: int,
    content_projection: ContentProjectionSpec | None,
) -> SessionTranscriptPage | None:
    """Read one session header plus the requested transcript window.

    One reader for both the whole-transcript and the bounded shape, so the
    two cannot drift about hydration, display labels or content projection --
    they differ only in which storage read composes the rows.
    """

    normalized_offset = max(int(offset), 0)
    try:
        resolved_id = archive.resolve_session_id(session_id)
    except KeyError:
        return None
    summary = archive.read_summary(resolved_id)
    envelope = (
        archive.read_session_page(resolved_id, limit=limit, offset=normalized_offset)
        if limit is not None
        else archive.read_session(resolved_id)
    )
    session = archive_envelope_to_session(
        envelope,
        display_label=summary.display_label,
        display_label_source=summary.display_label_source,
    )
    if content_projection is not None and content_projection.filters_content():
        session = session.with_content_projection(content_projection)
    return SessionTranscriptPage(
        session=session,
        total_message_count=(
            envelope.total_message_count if envelope.total_message_count is not None else len(session.messages)
        ),
        word_count=summary.word_count,
        limit=limit,
        offset=normalized_offset,
    )


def _archive_query_date_ms(field: str, value: str | None) -> int | None:
    parsed = parse_query_date(field, value)
    if parsed is None:
        return None
    return int(parsed.timestamp() * 1000)


def _archive_message_type(value: str | None) -> str | None:
    if value is None:
        return None
    return validate_message_type_filter(value).value


def _archive_action_terms(field: str, values: Sequence[str]) -> tuple[str, ...]:
    return normalize_action_terms(field, tuple(values))


def _archive_action_sequence(values: Sequence[str]) -> tuple[str, ...]:
    return normalize_action_sequence("action_sequence", ",".join(values))


def _archive_context_temporal_window(config: Config, summary: SessionSummary) -> TemporalEvidenceWindow:
    """Build the temporal context excerpt for one selected session."""
    from polylogue.operations.context_image_product import CONTEXT_TEMPORAL_MESSAGE_EVENTS, context_temporal_window

    with archive_read_context(
        _active_archive_root(config),
        operation="archive.context.temporal_window",
        arguments={"session_id": str(summary.id)},
        page_size=CONTEXT_TEMPORAL_MESSAGE_EVENTS,
        projection="temporal-window",
        stable_order="time,message_id",
    ) as archive:
        return context_temporal_window(archive, summary)


async def _archive_context_chronicle_payload(config: Config, summary: SessionSummary) -> ChronicleProjectionPayload:
    """Build the chronicle context excerpt for one selected session."""
    from polylogue.operations.context_image_product import context_chronicle_payload

    with archive_read_context(
        _active_archive_root(config),
        operation="archive.context.chronicle",
        arguments={"session_id": str(summary.id)},
        projection="chronicle",
    ) as archive:
        return context_chronicle_payload(archive, summary)


def _archive_query_kwargs(spec: SessionQuerySpec, *, default_limit: int | None) -> dict[str, object]:
    limit = spec.limit if spec.limit is not None else default_limit
    kwargs: dict[str, object] = {
        "offset": spec.offset,
        "origins": spec.origins,
        "excluded_origins": spec.excluded_origins,
        "tags": spec.tags,
        "excluded_tags": spec.excluded_tags,
        "repo_names": spec.repo_names,
        "project_refs": spec.project_refs,
        "has_types": spec.has_types,
        "has_tool_use": spec.filter_has_tool_use,
        "has_thinking": spec.filter_has_thinking,
        "has_paste": spec.filter_has_paste,
        "tool_terms": spec.tool_terms,
        "excluded_tool_terms": spec.excluded_tool_terms,
        "action_terms": _archive_action_terms("action", spec.action_terms),
        "excluded_action_terms": _archive_action_terms("exclude_action", spec.excluded_action_terms),
        "action_sequence": _archive_action_sequence(spec.action_sequence),
        "action_text_terms": spec.action_text_terms,
        "referenced_paths": spec.referenced_path,
        "cwd_prefix": spec.cwd_prefix,
        "typed_only": spec.typed_only,
        "message_type": _archive_message_type(spec.message_type),
        "title": spec.title,
        "session_id": spec.session_id,
        "min_messages": spec.min_messages,
        "max_messages": spec.max_messages,
        "min_words": spec.min_words,
        "max_words": spec.max_words,
        "since_ms": _archive_query_date_ms("since", spec.since),
        "until_ms": _archive_query_date_ms("until", spec.until),
        "since_session_id": spec.since_session_id,
        "boolean_predicate": spec.boolean_predicate,
        "root": resolve_default_root_filter(
            spec.root,
            boolean_predicate=spec.boolean_predicate,
        ),
    }
    if limit is not None:
        kwargs["limit"] = limit
    if spec.sort is not None:
        kwargs["sort"] = spec.sort
    if spec.reverse:
        kwargs["reverse"] = True
    if spec.sample is not None:
        # ``spec.sample`` is the requested page size; ``list_summaries(sample=...)``
        # is the boolean "order randomly" switch. Sampling starts at offset
        # zero and its requested size replaces the ordinary page limit.
        kwargs["sample"] = True
        kwargs["limit"] = spec.sample
        kwargs["offset"] = 0
    elif spec.latest:
        # Keep the low-level summary route aligned with query_spec_to_plan.
        # The plan's latest expansion is an updated-date ordering and one row.
        latest = spec.to_plan()
        kwargs["sort"] = latest.sort
        kwargs["limit"] = latest.limit
    return kwargs


def _archive_text_query(spec: SessionQuerySpec) -> str | None:
    terms = (*spec.query_terms, *spec.contains_terms)
    if not terms:
        return None
    return " ".join(term for term in terms if term).strip() or None


def _archive_list_summaries_for_spec(
    archive: Any,
    spec: SessionQuerySpec,
    *,
    default_limit: int,
    limit: int | None = None,
    offset: int | None = None,
) -> list[ArchiveSessionSummary]:
    query_text = _archive_text_query(spec)
    query_kwargs = _archive_query_kwargs(spec, default_limit=default_limit)
    if spec.exclude_text_terms:
        return _archive_list_summaries_with_post_filters(
            archive,
            spec,
            query_text=query_text,
            query_kwargs=query_kwargs,
            limit=limit,
            offset=offset,
        )
    if limit is not None:
        query_kwargs["limit"] = limit
    if offset is not None:
        query_kwargs["offset"] = offset
    if query_text is not None:
        query_kwargs.pop("sample", None)
        return [archive.read_summary(hit.session_id) for hit in archive.search_summaries(query_text, **query_kwargs)]
    return cast(list[ArchiveSessionSummary], archive.list_summaries(**query_kwargs))


#: Candidate sessions hydrated per post-filter chunk. Each chunk's ``Session``
#: objects are dropped before the next chunk is built, so peak memory is one
#: chunk rather than the whole candidate set.
POST_FILTER_HYDRATION_CHUNK = 200


def _post_filter_candidates(
    archive: Any,
    *,
    query_text: str | None,
    query_kwargs: dict[str, object],
) -> Iterator[ArchiveSessionSummary]:
    """Stream the SQL candidate set for a post-filtered spec in one forward pass.

    ``exclude_text`` has no SQL reduction, so every candidate may need to be
    hydrated to be tested. One cursor (``iter_summaries``/``iter_search_summaries``
    with ``limit=None``) keeps that bounded in memory without refusing a large
    scope, and never re-walks earlier rows the way a growing ``OFFSET`` does;
    the caller stops reading once its page is full.
    """

    query_kwargs = dict(query_kwargs)
    query_kwargs["limit"] = None
    query_kwargs.pop("offset", None)
    # Randomization applies to the survivors, not the candidates.
    query_kwargs.pop("sample", None)
    if query_kwargs.get("sort") == "random":
        query_kwargs.pop("sort")
    if query_text is not None:
        # A search yields one hit per matching block; each session is a
        # single candidate, so it is hydrated and sampled once.
        with _DistinctSessions() as seen:
            for hit in cast(Iterator[Any], archive.iter_search_summaries(query_text, **query_kwargs)):
                if seen.add(str(hit.session_id)):
                    yield archive.read_summary(hit.session_id)
        return
    yield from cast(Iterator[ArchiveSessionSummary], archive.iter_summaries(**query_kwargs))


def _iter_post_filtered_summaries(
    archive: Any,
    spec: SessionQuerySpec,
    candidates: Iterable[ArchiveSessionSummary],
    *,
    needed: int | None,
) -> Iterator[ArchiveSessionSummary]:
    """Yield post-filter survivors, hydrating one bounded chunk at a time.

    Follows the sibling executor's bounded post-filter fetch
    (``archive_execution._fetch_limit``/``post_filter_fetch``): hydrate a batch,
    keep its survivors, drop the batch, and stop as soon as the requested page
    is full.
    """

    plan = spec.to_plan()
    produced = 0
    source = iter(candidates)
    while chunk := list(itertools.islice(source, POST_FILTER_HYDRATION_CHUNK)):
        sessions = [
            archive_envelope_to_session(
                archive.read_session(summary.session_id),
                display_label=summary.display_label,
                display_label_source=summary.display_label_source,
            )
            for summary in chunk
        ]
        matched_ids = {str(session.id) for session in plan._apply_full_filters(sessions, sql_pushed=True)}
        del sessions
        for summary in chunk:
            if summary.session_id not in matched_ids:
                continue
            yield summary
            produced += 1
            if needed is not None and produced >= needed:
                return


def _archive_list_summaries_with_post_filters(
    archive: Any,
    spec: SessionQuerySpec,
    *,
    query_text: str | None,
    query_kwargs: dict[str, object],
    limit: int | None,
    offset: int | None,
) -> list[ArchiveSessionSummary]:
    """Apply content-dependent spec filters after the SQL candidate query.

    A random sample is drawn from the survivors, never from the candidates:
    sampling before the filter would shrink the sample by the excluded rows.
    """
    candidates = _post_filter_candidates(archive, query_text=query_text, query_kwargs=query_kwargs)
    survivors = _iter_post_filtered_summaries(archive, spec, candidates, needed=None)
    return _post_filter_window(survivors, query_kwargs=query_kwargs, limit=limit, offset=offset)


def _post_filter_window(
    survivors: Iterable[_T], *, query_kwargs: dict[str, object], limit: int | None, offset: int | None
) -> list[_T]:
    """Apply the canonical post-filter page or sample to supplied survivors."""
    # The spec's own page is authoritative unless an adapter explicitly
    # supplies a replacement. Candidate widening above removes SQL paging;
    # restore the effective page after content filtering.
    raw_offset = query_kwargs.get("offset", 0)
    effective_offset = offset if offset is not None else int(raw_offset) if isinstance(raw_offset, (int, str)) else 0
    raw_limit = limit if limit is not None else query_kwargs.get("limit")
    effective_limit = None if raw_limit is None else int(raw_limit) if isinstance(raw_limit, (int, str)) else None
    # A negative window reads as the SQL route reads it: from the start, empty.
    effective_offset = max(effective_offset, 0)
    if effective_limit is not None:
        effective_limit = max(effective_limit, 0)
    if query_kwargs.get("sample") or query_kwargs.get("sort") == "random":
        start = 0 if query_kwargs.get("sample") else effective_offset
        if effective_limit is None:
            everything = list(survivors)
            random.shuffle(everything)
            return everything[start:]
        # Reservoir sampling keeps memory at the returned window while every
        # survivor still has an equal chance of selection.
        chosen = _reservoir_sample(survivors, start + max(effective_limit, 0))
        random.shuffle(chosen)
        return chosen[start:]
    start = effective_offset
    end = None if effective_limit is None else start + effective_limit
    # ``islice`` skips the offset without retaining it.
    return list(itertools.islice(survivors, start, end))


def _reservoir_sample(items: Iterable[_T], size: int) -> list[_T]:
    """A uniform random sample of ``size`` items in one pass and O(size) memory."""

    reservoir: list[_T] = []
    if size <= 0:
        return reservoir
    for seen, item in enumerate(items):
        if seen < size:
            reservoir.append(item)
            continue
        slot = random.randint(0, seen)
        if slot < size:
            reservoir[slot] = item
    return reservoir


def _archive_search_hits_for_spec(
    archive: Any,
    spec: SessionQuerySpec,
    query_text: str,
    *,
    limit: int,
    offset: int,
) -> list[Any]:
    query_kwargs = _archive_query_kwargs(spec, default_limit=None)
    query_kwargs.pop("sample", None)
    query_kwargs["limit"] = limit
    query_kwargs["offset"] = offset
    return cast(list[Any], archive.search_summaries(query_text, **query_kwargs))


def _iter_archive_session_identity_candidates(
    archive: Any, spec: SessionQuerySpec | None
) -> Generator[ArchiveSessionIdentity, None, None]:
    """Yield canonical scalar scope candidates or content-filter survivors."""
    from polylogue.archive.query.runtime_filters import has_negative_message_term

    if spec is None:
        with closing(archive.iter_session_identities()) as selected:
            yield from selected
            return
    query_text = _archive_text_query(spec)
    query_kwargs = _archive_query_kwargs(spec, default_limit=None)

    def candidates(kwargs: dict[str, object]) -> Generator[ArchiveSessionIdentity, None, None]:
        if query_text is None:
            with closing(archive.iter_session_identities(**kwargs)) as selected:
                yield from selected
        else:
            with closing(
                archive.iter_session_identities(
                    query=query_text, actions_only=spec.retrieval_lane == "actions", **kwargs
                )
            ) as selected:
                yield from selected

    if not spec.exclude_text_terms:
        yield from candidates(query_kwargs)
        return
    candidate_kwargs = dict(query_kwargs)
    candidate_kwargs["limit"] = None
    candidate_kwargs.pop("offset", None)
    candidate_kwargs.pop("sample", None)
    if candidate_kwargs.get("sort") == "random":
        candidate_kwargs.pop("sort")

    def messages(session_id: str) -> Generator[Message, None, None]:
        from polylogue.operations.read_contracts import ReadPageUnavailableError

        offset = 0
        while True:
            page = archive.read_session_page(session_id, limit=100, offset=offset)
            yield from archive_envelope_to_session(page).messages
            offset += len(page.messages)
            if page.total_message_count is None:
                raise ReadPageUnavailableError("message-only scope page omitted its physical total")
            if offset >= page.total_message_count:
                return
            if not page.messages:
                raise ReadPageUnavailableError("message-only scope page did not advance")

    def survivors() -> Generator[ArchiveSessionIdentity, None, None]:
        terms = tuple(term.lower() for term in spec.exclude_text_terms)
        with closing(candidates(candidate_kwargs)) as selected:
            for row in selected:
                with closing(messages(row.session_id)) as records:
                    if not has_negative_message_term(records, terms):
                        yield row

    yield from survivors()


def _archive_session_identities_for_spec(archive: Any, spec: SessionQuerySpec | None) -> list[ArchiveSessionIdentity]:
    """Resolve the explicit full-return analysis scope without unused metadata."""
    with closing(_iter_archive_session_identity_candidates(archive, spec)) as selected:
        if spec is None or not spec.exclude_text_terms:
            return list(selected)
        return _post_filter_window(
            selected, query_kwargs=_archive_query_kwargs(spec, default_limit=None), limit=None, offset=None
        )


def _archive_selected_session_count_for_spec(archive: Any, spec: SessionQuerySpec) -> int:
    """Count the same survivor window without retaining its scalar identities."""
    with closing(_iter_archive_session_identity_candidates(archive, spec)) as selected:
        if not spec.exclude_text_terms:
            return sum(1 for _ in selected)
        limit = spec.sample if spec.sample is not None else spec.limit
        offset = 0 if spec.sample is not None else max(spec.offset, 0)
        if spec.sample is not None or spec.sort == "random":
            # Random ordering changes which identities survive the requested
            # window, never its cardinality. No sample collection is needed.
            count = max(sum(1 for _ in selected) - offset, 0)
            return count if limit is None else min(count, max(limit, 0))
        end = None if limit is None else offset + max(limit, 0)
        return sum(1 for _ in itertools.islice(selected, offset, end))


def _archive_count_sessions_for_spec(archive: Any, spec: SessionQuerySpec) -> int:
    if spec.exclude_text_terms:
        # A list total intentionally ignores its presentation window, while
        # sharing the scalar survivor walk with requested-window aggregates.
        with closing(_iter_archive_session_identity_candidates(archive, spec)) as selected:
            return sum(1 for _ in selected)
    query_kwargs = _archive_query_kwargs(spec, default_limit=None)
    for key in ("limit", "offset", "sort", "reverse", "sample"):
        query_kwargs.pop(key, None)
    query_text = _archive_text_query(spec)
    if query_text is not None:
        # An actions-lane count counts only sessions its search can return.
        return int(
            archive.count_search_sessions(query_text, actions_only=spec.retrieval_lane == "actions", **query_kwargs)
        )
    return int(archive.count_sessions(**query_kwargs))


def build_facets_response(
    *,
    global_buckets: Any,
    scoped_buckets: Any,
    scoped_to_query: bool,
    include_deferred: bool,
    elapsed_s: float | None,
    include_idf: bool,
    scope_gaps: Sequence[str] = (),
) -> FacetsResponse:
    """Assemble the one canonical facets envelope.

    Every surface that answers facets builds it here. The daemon read
    operation used to restate this assembly over the same buckets, and the
    restatement drifted: it dropped ``availability``, ``deadline_s``,
    ``elapsed_s``, ``stale_age_s`` and ``family_status`` entirely, hard-coded
    ``budget_exceeded``/``cost_class``, carried its own divergent family lists,
    and classified no gaps. Adding those keys back by hand would only restart
    the drift, so there is now one assembly and two callers.
    """

    from polylogue.analysis.projection_contracts import facets_availability
    from polylogue.archive.query.facets import compute_idf
    from polylogue.surfaces.outcome import decide_outcome
    from polylogue.surfaces.payloads import FacetBucketsPayload, FacetsResponse

    def _payload(b: Any) -> FacetBucketsPayload:
        return FacetBucketsPayload(
            origins=dict(b.origins),
            tags=dict(b.tags),
            repos=dict(b.repos),
            role_counts=dict(b.role_counts),
            material_origins=dict(b.material_origins),
            message_types=dict(b.message_types),
            action_types=dict(b.action_types),
            has_flags=dict(b.has_flags),
            omitted=dict(b.omitted),
            total_sessions=b.total_sessions,
            total_messages=b.total_messages,
        )

    def _family_status_payload(family: str, *, state: str, reason: str | None = None) -> dict[str, object]:
        return {
            "state": state,
            "reason": reason,
            "stale": False,
            **_FACET_FAMILY_METADATA.get(family, {}),
        }

    availability = facets_availability(include_deferred=include_deferred, elapsed_s=elapsed_s)
    active = scoped_buckets if scoped_to_query else global_buckets
    complete_families: tuple[str, ...] = _FACET_COMPLETE_FAMILIES if include_deferred else _FACET_CORE_FAMILIES
    deferred_families = {} if include_deferred else dict.fromkeys(_FACET_DEFERRED_FAMILIES, "deferred_by_default")
    # A scope that hit its session cap produced buckets over a truncated
    # denominator. Every family rolled from that scope is then partial, so it
    # must leave ``complete_families`` -- a truncated count reported as
    # complete is an unmeasured value rendered as a measured one.
    truncated_families: dict[str, str] = {}
    if scope_gaps:
        truncated_families = dict.fromkeys(complete_families, scope_gaps[0])
        complete_families = ()
    # A projection that missed its budget or lost a prerequisite is a named
    # gap: without it, zero facet rows at live scale reads identically to a
    # genuinely empty archive. Deferral is declared scope, not a gap.
    facet_gaps: list[str] = []
    if availability.state != "ready":
        facet_gaps.append(f"facets_{availability.state}")
    facet_gaps.extend(scope_gaps)
    return FacetsResponse.model_validate(
        {
            "outcome": decide_outcome(matched=active.total_sessions, degraded=facet_gaps),
            "scoped_to_query": scoped_to_query,
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "stale": False,
            "stale_age_s": None,
            "budget_exceeded": availability.budget_exceeded,
            "cost_class": availability.cost_class,
            "deadline_s": availability.deadline_s,
            "elapsed_s": availability.elapsed_s,
            "availability": availability,
            "complete_families": complete_families,
            "deferred_families": deferred_families,
            "family_errors": dict(truncated_families),
            "family_status": {
                **{family: _family_status_payload(family, state="complete") for family in complete_families},
                **{
                    family: _family_status_payload(family, state="deferred", reason=reason)
                    for family, reason in deferred_families.items()
                },
                **{
                    family: {
                        **_family_status_payload(family, state="error", reason=reason),
                        "error": reason,
                    }
                    for family, reason in truncated_families.items()
                },
            },
            "origins": dict(active.origins),
            "tags": dict(active.tags),
            "repos": dict(active.repos),
            "role_counts": dict(active.role_counts),
            "material_origins": dict(active.material_origins),
            "message_types": dict(active.message_types),
            "action_types": dict(active.action_types),
            "has_flags": dict(active.has_flags),
            "omitted_facet_counts": dict(active.omitted),
            "total_sessions": active.total_sessions,
            "total_messages": active.total_messages,
            "scoped": _payload(scoped_buckets),
            "global": _payload(global_buckets),
            "idf": compute_idf(global_buckets) if include_idf else {},
        }
    )


def _iter_facet_scope(archive: Any, spec: SessionQuerySpec | None) -> Iterator[ArchiveSessionSummary]:
    """Every session in the matched scope, streamed in one forward pass.

    Offset paging over an ordered/grouped scan redoes O(N) work per page
    (page k re-walks all k-1 earlier pages), so a facet aggregation over a
    large archive used to approach the read deadline. Each branch below now
    drives a single cursor -- ``archive.iter_summaries``/``iter_search_summaries``
    with ``limit=None`` -- instead of paging with a growing ``offset``.
    """
    from dataclasses import replace

    # Order and sampling are display choices like ``limit``; a sampled read
    # ignores ``offset`` and a random sort reshuffles between pages, so the
    # scope is walked in the default order.
    scope_spec = None if spec is None else replace(spec, limit=None, offset=0, sample=None, sort=None, reverse=False)
    if scope_spec is not None and scope_spec.exclude_text_terms:
        # One post-filter pass over the candidates: restarting it per facet
        # page would re-hydrate every earlier survivor on each page.
        candidates = _post_filter_candidates(
            archive,
            query_text=_archive_text_query(scope_spec),
            query_kwargs=_archive_query_kwargs(scope_spec, default_limit=None),
        )
        yield from _iter_post_filtered_summaries(archive, scope_spec, candidates, needed=None)
        return
    if scope_spec is None:
        yield from cast(Iterator[ArchiveSessionSummary], archive.iter_summaries(limit=None))
        return
    query_text = _archive_text_query(scope_spec)
    query_kwargs = _archive_query_kwargs(scope_spec, default_limit=None)
    query_kwargs["limit"] = None
    query_kwargs.pop("offset", None)
    if query_text is not None:
        query_kwargs.pop("sample", None)
        # One hit per matching block: each session is hydrated once.
        with _DistinctSessions() as seen:
            for hit in cast(Iterator[Any], archive.iter_search_summaries(query_text, **query_kwargs)):
                if seen.add(str(hit.session_id)):
                    yield archive.read_summary(hit.session_id)
        return
    yield from cast(Iterator[ArchiveSessionSummary], archive.iter_summaries(**query_kwargs))


def _archive_facet_buckets(
    archive: Any,
    spec: SessionQuerySpec | None,
    *,
    include_deferred: bool = True,
    scope_gaps: list[str] | None = None,
) -> Any:
    """Roll facet buckets over the whole matched scope.

    ``spec.limit``/``spec.offset`` are stripped before the scope query: a
    caller's page size is a display bound, not a denominator. The scope is
    streamed in one pass, so its size never truncates the buckets.
    """
    from polylogue.archive.query.facets import FacetBuckets

    del scope_gaps
    origins: dict[str, int] = {}
    tags: dict[str, int] = {}
    total_messages = 0
    total_sessions = 0
    sql_buckets = _empty_facet_families()
    # A scoped aggregation reads its SQL families one bounded chunk of
    # sessions at a time; the global one needs no session list at all.
    chunk: list[str] = []
    scoped = include_deferred and spec is not None
    # The scope yields each session once (search hits are deduplicated
    # before hydration), so the counts need no set of their own.
    for summary in _iter_facet_scope(archive, spec):
        total_sessions += 1
        total_messages += summary.message_count
        origins[summary.origin] = origins.get(summary.origin, 0) + 1
        for tag in set(summary.tags):
            tags[tag] = tags.get(tag, 0) + 1
        if scoped:
            chunk.append(summary.session_id)
            if len(chunk) >= _FACET_FAMILY_CHUNK:
                _merge_facet_families(sql_buckets, _archive_aggregate_facet_families(archive._conn, session_ids=chunk))
                chunk = []
    if scoped and chunk:
        _merge_facet_families(sql_buckets, _archive_aggregate_facet_families(archive._conn, session_ids=chunk))
    elif include_deferred and spec is None:
        sql_buckets = _archive_aggregate_facet_families(archive._conn, session_ids=None)
    return FacetBuckets(
        origins=origins,
        tags=tags,
        repos=sql_buckets["repos"],
        role_counts=sql_buckets["role_counts"],
        material_origins=sql_buckets["material_origins"],
        message_types=sql_buckets["message_types"],
        action_types=sql_buckets["action_types"],
        has_flags=sql_buckets["has_flags"],
        omitted=sql_buckets["omitted"],
        total_sessions=total_sessions,
        total_messages=total_messages,
    )


#: Sessions whose SQL facet families are aggregated per query; below SQLite's
#: bound-parameter limit.
_FACET_FAMILY_CHUNK = 900


class _DistinctSessions:
    """Membership of the session ids already counted, held on scratch disk.

    A search scope yields one hit per matching block, so a session repeats;
    remembering the ids in a set would grow with the scope.
    """

    def __enter__(self) -> _DistinctSessions:
        import tempfile

        self._scratch = tempfile.TemporaryDirectory(prefix="polylogue-facet-scope-")
        self._conn = sqlite3.connect(Path(self._scratch.name) / "seen.db")
        self._conn.execute("PRAGMA journal_mode=OFF")
        self._conn.execute("PRAGMA synchronous=OFF")
        self._conn.execute("CREATE TABLE seen (session_id TEXT PRIMARY KEY) WITHOUT ROWID")
        return self

    def add(self, session_id: str) -> bool:
        """Record ``session_id``; return whether it was new."""
        return self._conn.execute("INSERT OR IGNORE INTO seen VALUES (?)", (session_id,)).rowcount == 1

    def __exit__(self, *exc: object) -> None:
        self._conn.close()
        self._scratch.cleanup()


def _empty_facet_families() -> dict[str, dict[str, int]]:
    return {
        "repos": {},
        "role_counts": {},
        "material_origins": {},
        "message_types": {},
        "action_types": {},
        "has_flags": {},
        "omitted": {},
    }


def _merge_facet_families(total: dict[str, dict[str, int]], part: dict[str, dict[str, int]]) -> None:
    """Add one disjoint session chunk's family counts into ``total``."""
    for family, counts in part.items():
        merged = total.setdefault(family, {})
        for key, count in counts.items():
            merged[key] = merged.get(key, 0) + count


def _archive_aggregate_facet_families(
    conn: Any,
    *,
    session_ids: list[str] | None,
) -> dict[str, dict[str, int]]:
    result = _empty_facet_families()
    if session_ids is not None and not session_ids:
        return result

    def scoped_rows(scoped_sql: str, global_sql: str) -> list[Any]:
        if session_ids is None:
            return list(conn.execute(global_sql).fetchall())
        rows: list[Any] = []
        for start in range(0, len(session_ids), 900):
            chunk = session_ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(conn.execute(scoped_sql.format(placeholders), chunk).fetchall())
        return rows

    def keyed(rows: list[Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            if row[0]:
                key = str(row[0])
                counts[key] = counts.get(key, 0) + int(row[1] or 0)
        return counts

    repo_rows = scoped_rows(
        """
        SELECT sr.session_id, r.repo_name, r.root_path, r.origin_url
        FROM session_repos sr
        JOIN repos r ON r.repo_id = sr.repo_id
        WHERE sr.session_id IN ({})
        """,
        """
        SELECT sr.session_id, r.repo_name, r.root_path, r.origin_url
        FROM session_repos sr
        JOIN repos r ON r.repo_id = sr.repo_id
        """,
    )
    repo_sessions: dict[str, set[str]] = {}
    omitted_repo_sessions: set[str] = set()
    for row in repo_rows:
        session_id = str(row[0])
        label = _canonical_repo_facet_label(repo_name=row[1], root_path=row[2], origin_url=row[3])
        if label is None:
            omitted_repo_sessions.add(session_id)
            continue
        repo_sessions.setdefault(label, set()).add(session_id)
    result["repos"] = {label: len(sessions) for label, sessions in repo_sessions.items()}
    if omitted_repo_sessions:
        result["omitted"]["repos"] = len(omitted_repo_sessions)
    result["role_counts"] = keyed(
        scoped_rows(
            """
            SELECT COALESCE(NULLIF(role, ''), 'unknown') AS role_key, COUNT(*) AS n
            FROM messages
            WHERE session_id IN ({})
            GROUP BY role_key
            """,
            """
            SELECT COALESCE(NULLIF(role, ''), 'unknown') AS role_key, COUNT(*) AS n
            FROM messages
            GROUP BY role_key
            """,
        )
    )
    result["material_origins"] = keyed(
        scoped_rows(
            """
            SELECT COALESCE(NULLIF(material_origin, ''), 'unknown') AS material_key, COUNT(*) AS n
            FROM messages
            WHERE session_id IN ({})
            GROUP BY material_key
            """,
            """
            SELECT COALESCE(NULLIF(material_origin, ''), 'unknown') AS material_key, COUNT(*) AS n
            FROM messages
            GROUP BY material_key
            """,
        )
    )
    result["message_types"] = keyed(
        scoped_rows(
            "SELECT message_type, COUNT(*) AS n FROM messages WHERE session_id IN ({}) GROUP BY message_type",
            "SELECT message_type, COUNT(*) AS n FROM messages GROUP BY message_type",
        )
    )
    result["action_types"] = keyed(
        scoped_rows(
            "SELECT semantic_type, COUNT(*) AS n FROM actions WHERE session_id IN ({}) GROUP BY semantic_type",
            "SELECT semantic_type, COUNT(*) AS n FROM actions GROUP BY semantic_type",
        )
    )
    flag_rows = scoped_rows(
        """
        SELECT COALESCE(SUM(has_tool_use), 0), COALESCE(SUM(has_thinking), 0), COALESCE(SUM(has_paste), 0)
        FROM messages
        WHERE session_id IN ({})
        """,
        """
        SELECT COALESCE(SUM(has_tool_use), 0), COALESCE(SUM(has_thinking), 0), COALESCE(SUM(has_paste), 0)
        FROM messages
        """,
    )
    result["has_flags"] = {
        "has_tool_use": sum(int(row[0] or 0) for row in flag_rows),
        "has_thinking": sum(int(row[1] or 0) for row in flag_rows),
        "has_paste": sum(int(row[2] or 0) for row in flag_rows),
    }
    return result


def _canonical_repo_facet_label(*, repo_name: object, root_path: object, origin_url: object) -> str | None:
    """Return a product-level repo facet label or ``None`` for path noise."""

    repo = repo_name if isinstance(repo_name, str) and repo_name else None
    if repo and not _is_noisy_repo_label(repo):
        return repo
    url_label = _repo_label_from_url(origin_url)
    if url_label and not _is_noisy_repo_label(url_label):
        return url_label
    root = root_path if isinstance(root_path, str) and root_path else None
    if root is None:
        return None
    basename = root.rstrip("/").rsplit("/", maxsplit=1)[-1]
    if _is_noisy_repo_label(basename):
        return None
    if "/" not in root:
        return basename
    if root.endswith(f"/{basename}") and (root.endswith("/project/" + basename) or root.endswith("/repo/" + basename)):
        return basename
    return None


def _clean_repo_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _repo_label_from_url(value: object) -> str | None:
    cleaned = _clean_repo_label(value)
    if cleaned is None:
        return None
    label = cleaned.rstrip("/").rsplit("/", maxsplit=1)[-1]
    if label.endswith(".git"):
        label = label[:-4]
    return label or None


def _is_noisy_repo_label(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _NOISY_REPO_LABELS:
        return True
    return normalized.isdigit()


def _archive_health_report(config: Config) -> ReadinessReport:
    from polylogue.archive.query.transaction import archive_read_context
    from polylogue.readiness import ReadinessCheck, ReadinessReport, VerifyStatus

    checks: list[ReadinessCheck] = []
    root = config.archive_root
    checks.append(
        ReadinessCheck(
            "archive_root",
            VerifyStatus.OK if root.exists() else VerifyStatus.WARNING,
            summary=str(root),
        )
    )

    tier_paths = {
        ArchiveTier.SOURCE: root / "source.db",
        ArchiveTier.INDEX: root / "index.db",
        ArchiveTier.EMBEDDINGS: root / "embeddings.db",
        ArchiveTier.USER: root / "user.db",
        ArchiveTier.AUDIT: root / "audit.db",
        ArchiveTier.OPS: root / "ops.db",
    }
    for tier, path in tier_paths.items():
        checks.append(_archive_tier_readiness_check(tier, path))

    try:
        with archive_read_context(
            root,
            operation="archive.health_check",
            arguments={},
            projection="health",
        ) as archive:
            stats = archive.stats()
            checks.append(
                ReadinessCheck(
                    "archive_index_rows",
                    VerifyStatus.OK,
                    count=stats.total_sessions,
                    summary=f"{stats.total_sessions:,} sessions / {stats.total_messages:,} messages",
                )
            )
            fts_count = _archive_count_table_rows(archive._conn, "messages_fts")
            checks.append(
                ReadinessCheck(
                    "archive_search",
                    VerifyStatus.OK if fts_count is not None else VerifyStatus.WARNING,
                    count=fts_count or 0,
                    summary="messages_fts present" if fts_count is not None else "messages_fts missing",
                )
            )
            insight_status = archive.session_insight_status()
            insights_ready = all(
                value == 0
                for value in (
                    insight_status.missing_profile_row_count,
                    insight_status.stale_profile_row_count,
                    insight_status.orphan_profile_row_count,
                    insight_status.stale_thread_count,
                    insight_status.orphan_thread_count,
                )
            ) and (
                insight_status.profile_row_count == insight_status.total_sessions
                and insight_status.thread_count == insight_status.root_threads
            )
            checks.append(
                ReadinessCheck(
                    "archive_session_insights",
                    VerifyStatus.OK if insights_ready else VerifyStatus.WARNING,
                    count=insight_status.profile_row_count,
                    summary=(
                        "session insight rows ready"
                        if insights_ready
                        else "session insight rows missing, stale, or inconsistent; run rebuild_insights"
                    ),
                )
            )
    except Exception as exc:
        checks.append(ReadinessCheck("archive_index", VerifyStatus.ERROR, summary=str(exc)))

    return ReadinessReport(checks=checks)


def _archive_tier_readiness_check(tier: ArchiveTier, path: Any) -> Any:
    from polylogue.readiness import ReadinessCheck, VerifyStatus

    name = f"archive_{tier.value}"
    if not path.exists():
        return ReadinessCheck(name, VerifyStatus.WARNING, summary=f"missing: {path}")
    try:
        conn = open_readonly_connection(path, tier=tier, timeout_class="interactive-read")
        try:
            row = conn.execute("PRAGMA user_version").fetchone()
            version = int(row[0] or 0) if row is not None else 0
        finally:
            conn.close()
    except SchemaRefusalError as exc:
        # The open's own admission check is the diagnosis: it covers both a
        # version skew and a derived-identity mismatch at the same version.
        return ReadinessCheck(name, VerifyStatus.ERROR, summary=f"{exc}: {path}")
    except (OSError, sqlite3.Error) as exc:
        return ReadinessCheck(name, VerifyStatus.ERROR, summary=str(exc))

    expected = ARCHIVE_VERSION_BY_TIER[tier]
    return ReadinessCheck(
        name,
        VerifyStatus.OK if version == expected else VerifyStatus.ERROR,
        summary=f"v{version}/{expected}: {path}",
    )


def _archive_get_context_delivery(
    config: Config,
    *,
    snapshot_ref: str,
    recipient_ref: str,
) -> ArchiveContextDeliveryEnvelope | None:
    """Read one delivery receipt only when it belongs to its recorded recipient."""

    from polylogue.storage.sqlite.archive_tiers.context_delivery_write import read_context_delivery

    with _readable_user_tier(config) as conn:
        receipt = read_context_delivery(conn, snapshot_ref)
        return receipt if receipt is not None and receipt.recipient_ref == recipient_ref else None


def _archive_list_context_deliveries(
    config: Config,
    *,
    recipient_ref: str | None,
    assertion_ref: str | None,
    limit: int,
    offset: int,
) -> ArchiveContextDeliveryPage:
    """Read one receipt-summary page, most recent first."""

    from polylogue.storage.sqlite.archive_tiers.context_delivery_write import list_context_deliveries

    with _readable_user_tier(config) as conn:
        return list_context_deliveries(
            conn, recipient_ref=recipient_ref, assertion_ref=assertion_ref, limit=limit, offset=offset
        )


def _archive_list_context_injection_ledger(
    config: Config,
    *,
    target_session: str | None,
    execution_context_ref: str | None,
    limit: int,
) -> list[ContextLedgerRecord]:
    """Read scheduler decisions from the disposable ops tier."""

    with _readable_required_tier(config, ArchiveTier.OPS) as conn:
        return list(
            read_context_ledger(
                conn, target_session=target_session, execution_context_ref=execution_context_ref, limit=limit
            )
        )


def _archive_correlate_hermes_context_deliveries(
    config: Config,
    *,
    hermes_session_native_id: str,
) -> tuple[HermesContextDeliveryCorrelation, ...]:
    """Correlate drained Hermes events with receipts, refusing unavailable authority."""

    from polylogue.context.hermes_delivery_correlation import correlate_hermes_context_deliveries

    with _readable_required_tier(config, ArchiveTier.SOURCE) as source_conn:
        source_conn.execute("SELECT 1 FROM raw_hook_events LIMIT 0")
        with _readable_user_tier(config) as user_conn:
            user_conn.execute("SELECT snapshot_ref FROM context_deliveries LIMIT 0")
            return correlate_hermes_context_deliveries(
                source_conn, user_conn, hermes_session_native_id=hermes_session_native_id
            )


def _archive_get_setting(config: Config, setting_key: str) -> ArchiveUserSettingEnvelope | None:
    """Read one durable ``user_settings`` row, or ``None`` when unset (polylogue-at44)."""

    from polylogue.storage.sqlite.archive_tiers.user_settings_write import get_user_setting

    with _readable_user_tier(config) as conn:
        return get_user_setting(conn, setting_key)


def _archive_list_settings(config: Config) -> list[ArchiveUserSettingEnvelope]:
    """List every stored ``user_settings`` row, ordered by key."""

    from polylogue.storage.sqlite.archive_tiers.user_settings_write import list_user_settings

    with _readable_user_tier(config) as conn:
        return list_user_settings(conn)


def _read_source_and_index(
    config: Config,
    work: Callable[[sqlite3.Connection, sqlite3.Connection], _ReadResultT],
    *,
    seam: str,
) -> _ReadResultT | None:
    """Run one read-only join across source.db and index.db, or return ``None``.

    The shared shape of the archive's read-only audit seams: both tiers are
    opened read-only, neither is mutated, and ``None`` covers two distinct
    cases the facade contract does not separate -- the archive is not
    initialized yet (no tier file), or it is present and unreadable. Only the
    second logs, so the two stay distinguishable in the record.
    """

    archive_root = _active_archive_root(config)
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        return None
    try:
        # ``work`` is caller-supplied and its duration is not this seam's to
        # know, so both readers are frames: each is bound to the generation it
        # opened on and refuses to serve past the declared snapshot age rather
        # than pinning WAL frames for an unbounded audit.
        with (
            read_frame(source_db, timeout_class="background-read") as source_frame,
            read_frame(index_db, timeout_class="background-read") as index_frame,
        ):
            return work(source_frame.connection, index_frame.connection)
    except (ReadFrameExpiredError, StaleContinuationError) as exc:
        emit(
            "archive.read.frame_expired",
            level=WARNING,
            outcome="degraded",
            route=seam,
            reason="declared_frame_exceeded",
            db_path=index_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return None
    except sqlite3.Error as exc:
        emit(
            "archive.read.unreadable",
            level=WARNING,
            outcome="degraded",
            route=seam,
            reason="archive_present_but_unreadable",
            db_path=index_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return None


def _archive_reconcile_hermes_session_lifecycle(
    config: Config,
    *,
    hermes_session_native_id: str,
) -> HermesLifecycleReconciliation | None:
    """Reconcile a Hermes session's drained lifecycle-event stream (fs1.7 AC).

    Read-only audit seam over two durable tiers (source.db lifecycle spool +
    index.db ingested snapshot); see
    ``context.hermes_lifecycle_reconciliation`` for the join semantics and
    ``_read_source_and_index`` for the ``None`` contract. A caller
    distinguishes "not available yet" (``None``) from "reconciled, zero events
    observed" (``total_events == 0``).
    """

    from polylogue.context.hermes_lifecycle_reconciliation import reconcile_hermes_session_lifecycle

    return _read_source_and_index(
        config,
        lambda source_conn, index_conn: reconcile_hermes_session_lifecycle(
            source_conn,
            index_conn,
            hermes_session_native_id=hermes_session_native_id,
        ),
        seam="hermes_session_lifecycle_reconciliation",
    )


def _archive_correlate_claude_agent_dispatches(config: Config) -> ClaudeAgentDispatchCorrelation | None:
    """Resolve Claude Code hook ``agent_id``/``agent_type`` onto archived tool calls
    (bd polylogue-xo9gq).

    Read-only audit seam over ``source.db``'s hook-event spool and ``index.db``'s
    block tree; see ``context.claude_agent_dispatch_correlation`` for the join
    semantics and ``_read_source_and_index`` for the ``None`` contract.
    """

    from polylogue.context.claude_agent_dispatch_correlation import correlate_claude_agent_dispatches

    return _read_source_and_index(config, correlate_claude_agent_dispatches, seam="claude_agent_dispatch_correlation")


def _archive_reconcile_codex_spawn_edges(config: Config) -> CodexSpawnEdgeReconciliation | None:
    """Reconcile projected Codex spawn edges against transcript-inferred topology.

    Read-only audit seam over ``index.db``; see
    ``context.codex_spawn_edge_correlation`` for the join semantics. ``None``
    covers the archive not being initialized yet or being unreadable.
    """

    from polylogue.context.codex_spawn_edge_correlation import reconcile_codex_spawn_edges

    archive_root = _active_archive_root(config)
    index_db = archive_root / "index.db"
    if not index_db.exists():
        return None
    try:
        index_conn = open_readonly_connection(index_db, timeout_class="background-read")
        index_conn.row_factory = sqlite3.Row
        try:
            return reconcile_codex_spawn_edges(index_conn)
        finally:
            index_conn.close()
    except sqlite3.Error as exc:
        emit(
            "archive.read.unreadable",
            level=WARNING,
            outcome="degraded",
            route="codex_spawn_edge_reconciliation",
            reason="archive_present_but_unreadable",
            db_path=index_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return None


@contextmanager
def _readable_required_tier(config: Config, tier: ArchiveTier) -> Iterator[sqlite3.Connection]:
    from polylogue.operations.user_overlay_reads import readable_required_tier

    with readable_required_tier(_active_archive_root(config) / f"{tier.value}.db", tier) as conn:
        yield conn


@contextmanager
def _readable_user_tier(config: Config) -> Iterator[sqlite3.Connection]:
    """Read the durable user authority, refusing unavailable state."""
    with _readable_required_tier(config, ArchiveTier.USER) as conn:
        yield conn


def _archive_list_assertion_candidate_reviews(
    config: Config,
    *,
    target_ref: str | None = None,
    kinds: Sequence[str | AssertionKind] | None = None,
    statuses: Sequence[str | AssertionStatus] | None = None,
    limit: int | None = None,
) -> tuple[list[ArchiveAssertionCandidateReviewEnvelope], int]:
    """Read one review page and its unpaginated count from the same snapshot."""

    from polylogue.storage.sqlite.archive_tiers.user_write import (
        ASSERTION_CANDIDATE_JUDGMENT_KINDS,
        count_assertion_claims,
        list_assertion_candidate_reviews,
    )

    with _readable_user_tier(config) as conn:
        conn.execute("BEGIN")
        # Reviews deliberately include expired claims; this scalar count has
        # the same kind, status and target selection without hydrating every
        # claim or fetching every claim's latest judgment.
        matched = count_assertion_claims(
            conn,
            kinds=ASSERTION_CANDIDATE_JUDGMENT_KINDS if kinds is None else kinds,
            target_ref=target_ref,
            statuses=statuses,
            include_expired=True,
        )
        rows = list_assertion_candidate_reviews(
            conn,
            target_ref=target_ref,
            kinds=kinds,
            statuses=statuses,
            limit=limit,
        )
        return rows, matched


def _archive_list_assertion_candidates(
    config: Config,
    *,
    target_ref: str | None = None,
    kinds: Sequence[str | AssertionKind] | None = None,
    limit: int | None = None,
) -> list[Any]:
    """Read every pending candidate kind from the durable judgment queue.

    The actionable queue excludes expired claims. The sibling review read model
    retains them on purpose, so it cannot stand in for this one.
    """

    from polylogue.storage.sqlite.archive_tiers.user_write import list_assertion_candidates

    with _readable_user_tier(config) as conn:
        return list_assertion_candidates(conn, target_ref=target_ref, kinds=kinds, limit=limit)


def _archive_assertion_candidate_queue_health(
    config: Config,
    *,
    now_ms: int | None = None,
) -> AssertionCandidateQueueHealthPayload:
    """Project queue depth, retention, producer telemetry, and scheduler health."""

    from polylogue.daemon.judgment_automation import (
        is_valid_judgment_automation_receipt_payload,
        judgment_automation_receipt_freshness_window_ms,
    )
    from polylogue.daemon.lifecycle import DAEMON_HEARTBEAT_STALE_AFTER_SECONDS
    from polylogue.operations.judgment_scheduler import read_latest_judgment_scheduler_receipt
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        ASSERTION_CANDIDATE_JUDGMENT_KINDS,
        ASSERTION_CANDIDATE_REVIEW_STATUSES,
    )
    from polylogue.surfaces.payloads import AssertionCandidateQueueHealthPayload

    observed_at_ms = int(datetime.now(UTC).timestamp() * 1000) if now_ms is None else now_ms
    interval_s = getattr(config, "judgment_automation_interval_s", None)
    if isinstance(interval_s, bool) or not isinstance(interval_s, int):
        return AssertionCandidateQueueHealthPayload(
            state="unavailable",
            observed_at_ms=observed_at_ms,
            pending_count=0,
            caveats=("judgment scheduler interval authority is unavailable; freshness is unverified",),
        )
    archive_root = _active_archive_root(config)
    user_db = archive_root / "user.db"
    if not user_db.exists():
        return AssertionCandidateQueueHealthPayload(
            state="unavailable",
            observed_at_ms=observed_at_ms,
            pending_count=0,
            caveats=("user.db is not initialized",),
        )

    kind_values = tuple(kind.value for kind in ASSERTION_CANDIDATE_JUDGMENT_KINDS)
    review_status_values = tuple(status.value for status in ASSERTION_CANDIDATE_REVIEW_STATUSES)
    kind_placeholders = ",".join("?" for _ in kind_values)
    status_placeholders = ",".join("?" for _ in review_status_values)
    try:
        conn = open_readonly_connection(user_db)
        conn.row_factory = sqlite3.Row
        try:
            table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='assertions'").fetchone()
            if table is None:
                raise sqlite3.OperationalError("assertions table is unavailable")
            status_counts = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    f"""
                    SELECT status, COUNT(*)
                    FROM assertions
                    WHERE kind IN ({kind_placeholders})
                      AND status IN ({status_placeholders})
                    GROUP BY status
                    ORDER BY status
                    """,
                    (*kind_values, *review_status_values),
                )
            }
            kind_counts = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    f"""
                    SELECT kind, COUNT(*)
                    FROM assertions
                    WHERE kind IN ({kind_placeholders}) AND status = ?
                    GROUP BY kind
                    ORDER BY kind
                    """,
                    (*kind_values, AssertionStatus.CANDIDATE.value),
                )
            }
            source_counts = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    f"""
                    SELECT COALESCE(author_kind, 'unknown') || ':' || COALESCE(author_ref, 'unknown'), COUNT(*)
                    FROM assertions
                    WHERE kind IN ({kind_placeholders}) AND status = ?
                    GROUP BY author_kind, author_ref
                    ORDER BY COUNT(*) DESC, author_kind, author_ref
                    """,
                    (*kind_values, AssertionStatus.CANDIDATE.value),
                )
            }
            age_cutoff_ms = observed_at_ms - 60 * 24 * 60 * 60 * 1000
            aggregate = conn.execute(
                f"""
                SELECT COUNT(*), MIN(created_at_ms), MAX(created_at_ms),
                       SUM(CASE WHEN created_at_ms < ? THEN 1 ELSE 0 END)
                FROM assertions
                WHERE kind IN ({kind_placeholders}) AND status = ?
                """,
                (age_cutoff_ms, *kind_values, AssertionStatus.CANDIDATE.value),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return AssertionCandidateQueueHealthPayload(
            state="unavailable",
            observed_at_ms=observed_at_ms,
            pending_count=0,
            caveats=(f"queue read failed: {exc}",),
        )

    pending_count = int(aggregate[0] or 0) if aggregate is not None else 0
    oldest_pending_at_ms = int(aggregate[1]) if aggregate is not None and aggregate[1] is not None else None
    newest_pending_at_ms = int(aggregate[2]) if aggregate is not None and aggregate[2] is not None else None
    stale_pending_count = int(aggregate[3] or 0) if aggregate is not None else 0

    producer_status: str | None = None
    producer_observed_at_ms: int | None = None
    producer_debt_count = 0
    scheduler_state: Literal["fresh", "stale", "stopped", "unknown"] = "unknown"
    scheduler_heartbeat_at_ms: int | None = None
    judgment_scheduler_receipt_status: Literal["completed", "parked", "failed", "unknown"] = "unknown"
    judgment_scheduler_receipt_at_ms: int | None = None
    judgment_scheduler_receipt_reason: str | None = None
    judgment_scheduler_receipt_retryable: bool | None = None
    judgment_scheduler_receipt_retry_route: str | None = None
    judgment_scheduler_receipt_batch_limit: int | None = None
    judgment_scheduler_receipt_considered: int | None = None
    judgment_scheduler_receipt_accepted: int | None = None
    judgment_scheduler_receipt_rejected: int | None = None
    judgment_scheduler_receipt_escalated: int | None = None
    judgment_scheduler_receipt_idempotent: int | None = None
    judgment_scheduler_receipt_failed: int | None = None
    judgment_scheduler_receipt_persistence_degraded: bool | None = None
    judgment_scheduler_receipt_persistence_recovered: bool | None = None
    caveats: list[str] = []
    ops_db = archive_root / "ops.db"
    if not ops_db.exists():
        caveats.append("ops.db is unavailable; producer and scheduler health are unverified")
    else:
        try:
            ops_conn = open_readonly_connection(ops_db)
            try:
                if ops_conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='daemon_stage_events'"
                ).fetchone():
                    producer_row = ops_conn.execute(
                        """
                        SELECT status, observed_at_ms
                        FROM daemon_stage_events
                        WHERE stage = 'standing-queries'
                        ORDER BY observed_at_ms DESC, rowid DESC
                        LIMIT 1
                        """
                    ).fetchone()
                    if producer_row is not None:
                        producer_status = str(producer_row[0])
                        producer_observed_at_ms = int(producer_row[1])
                if ops_conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='convergence_debt'"
                ).fetchone():
                    debt_row = ops_conn.execute(
                        """
                        SELECT COUNT(*) FROM convergence_debt
                        WHERE stage = 'standing-queries' AND status IN ('failed', 'deferred')
                        """
                    ).fetchone()
                    producer_debt_count = int(debt_row[0] or 0) if debt_row is not None else 0
                if ops_conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='daemon_lifecycle'"
                ).fetchone():
                    lifecycle_row = ops_conn.execute(
                        """
                        SELECT stopped_at_ms, last_heartbeat_at_ms
                        FROM daemon_lifecycle
                        ORDER BY started_at_ms DESC
                        LIMIT 1
                        """
                    ).fetchone()
                    if lifecycle_row is not None:
                        scheduler_heartbeat_at_ms = int(lifecycle_row[1])
                        if lifecycle_row[0] is not None:
                            scheduler_state = "stopped"
                        elif observed_at_ms - scheduler_heartbeat_at_ms <= (
                            DAEMON_HEARTBEAT_STALE_AFTER_SECONDS * 1000
                        ):
                            scheduler_state = "fresh"
                        else:
                            scheduler_state = "stale"
                typed_receipt = read_latest_judgment_scheduler_receipt(ops_conn)
                if typed_receipt is not None:
                    judgment_scheduler_receipt_status = cast(
                        Literal["completed", "parked", "failed"], typed_receipt.status
                    )
                    judgment_scheduler_receipt_at_ms = typed_receipt.observed_at_ms
                    judgment_scheduler_receipt_reason = typed_receipt.reason
                    judgment_scheduler_receipt_retryable = typed_receipt.retryable
                    judgment_scheduler_receipt_retry_route = typed_receipt.retry_route
                    judgment_scheduler_receipt_batch_limit = typed_receipt.batch_limit
                    judgment_scheduler_receipt_considered = typed_receipt.considered
                    judgment_scheduler_receipt_accepted = typed_receipt.accepted
                    judgment_scheduler_receipt_rejected = typed_receipt.rejected
                    judgment_scheduler_receipt_escalated = typed_receipt.escalated
                    judgment_scheduler_receipt_idempotent = typed_receipt.idempotent
                    judgment_scheduler_receipt_failed = typed_receipt.failed
                    judgment_scheduler_receipt_persistence_degraded = typed_receipt.receipt_persistence_degraded
                    judgment_scheduler_receipt_persistence_recovered = typed_receipt.receipt_persistence_recovered
                elif ops_conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='daemon_events'"
                ).fetchone():
                    # Compatibility for ops.db files created before the typed
                    # receipt table. Once a typed row exists it is authoritative,
                    # even if a newer legacy event is malformed or stale.
                    receipt_row = ops_conn.execute(
                        """
                        SELECT ts_ms, payload_json
                        FROM daemon_events
                        WHERE kind = 'judgment-automation'
                        ORDER BY id DESC
                        LIMIT 1
                        """
                    ).fetchone()
                    if receipt_row is not None:
                        try:
                            receipt_payload = json.loads(str(receipt_row[1]))
                        except (TypeError, ValueError):
                            receipt_payload = None
                        receipt_is_valid = is_valid_judgment_automation_receipt_payload(receipt_payload)
                        raw_status = (
                            str(receipt_payload.get("status", "unknown"))
                            if receipt_is_valid and isinstance(receipt_payload, dict)
                            else "unknown"
                        )
                        if not receipt_is_valid:
                            caveats.append("latest judgment scheduler receipt is malformed")
                        if raw_status in {"completed", "parked", "failed"}:
                            judgment_scheduler_receipt_status = cast(
                                Literal["completed", "parked", "failed"], raw_status
                            )
                        judgment_scheduler_receipt_at_ms = int(receipt_row[0])
                        if receipt_is_valid and isinstance(receipt_payload, dict):
                            judgment_scheduler_receipt_reason = str(receipt_payload["reason"])
                            judgment_scheduler_receipt_retryable = bool(receipt_payload["retryable"])
                            judgment_scheduler_receipt_retry_route = str(receipt_payload["retry_route"])
                            judgment_scheduler_receipt_batch_limit = int(receipt_payload["batch_limit"])
                            counter_names = (
                                "considered",
                                "accepted",
                                "rejected",
                                "escalated",
                                "idempotent",
                                "failed",
                            )
                            if all(name in receipt_payload for name in counter_names):
                                for name in counter_names:
                                    value = receipt_payload[name]
                                    if isinstance(value, int) and not isinstance(value, bool):
                                        if name == "considered":
                                            judgment_scheduler_receipt_considered = value
                                        elif name == "accepted":
                                            judgment_scheduler_receipt_accepted = value
                                        elif name == "rejected":
                                            judgment_scheduler_receipt_rejected = value
                                        elif name == "escalated":
                                            judgment_scheduler_receipt_escalated = value
                                        elif name == "idempotent":
                                            judgment_scheduler_receipt_idempotent = value
                                        elif name == "failed":
                                            judgment_scheduler_receipt_failed = value
                            judgment_scheduler_receipt_persistence_degraded = bool(
                                receipt_payload["receipt_persistence_degraded"]
                            )
                            judgment_scheduler_receipt_persistence_recovered = bool(
                                receipt_payload["receipt_persistence_recovered"]
                            )
            finally:
                ops_conn.close()
        except sqlite3.Error as exc:
            caveats.append(f"ops telemetry read failed: {exc}")

    producer_age_ms = None if producer_observed_at_ms is None else max(0, observed_at_ms - producer_observed_at_ms)
    heartbeat_age_ms = None if scheduler_heartbeat_at_ms is None else max(0, observed_at_ms - scheduler_heartbeat_at_ms)
    successful_producer_statuses = {"completed", "done", "ok", "success", "succeeded"}
    failed_producer_statuses = {"error", "failed", "interrupted"}
    producer_fresh = (
        producer_status in successful_producer_statuses
        and producer_age_ms is not None
        and producer_age_ms <= 24 * 60 * 60 * 1000
    )
    judgment_receipt_age_ms = (
        None if judgment_scheduler_receipt_at_ms is None else max(0, observed_at_ms - judgment_scheduler_receipt_at_ms)
    )
    receipt_freshness_window_ms = judgment_automation_receipt_freshness_window_ms(interval_s)
    parked_receipt_freshness_window_ms = judgment_automation_receipt_freshness_window_ms(interval_s, parked=True)
    judgment_receipt_fresh = (
        judgment_scheduler_receipt_status == "completed"
        and judgment_receipt_age_ms is not None
        and judgment_receipt_age_ms <= receipt_freshness_window_ms
    )
    judgment_parked_receipt_fresh = (
        judgment_scheduler_receipt_status == "parked"
        and judgment_receipt_age_ms is not None
        and judgment_receipt_age_ms <= parked_receipt_freshness_window_ms
    )

    state: AssertionCandidateQueueState
    if producer_debt_count or producer_status in failed_producer_statuses or scheduler_state in {"stale", "stopped"}:
        state = "producer-stalled"
    elif stale_pending_count:
        state = "stale-pending"
    elif pending_count and judgment_scheduler_receipt_status == "parked" and not judgment_parked_receipt_fresh:
        state = "scheduler-stalled"
        caveats.append(
            "judgment scheduler has no fresh parked receipt; the bounded retry route is the next daemon tick"
        )
    elif pending_count and judgment_scheduler_receipt_status in {"parked", "unknown"}:
        state = "parked-pending"
        if judgment_scheduler_receipt_status == "parked":
            caveats.append("judgment scheduler is parked; the bounded retry route is the next enabled daemon tick")
        else:
            caveats.append("no judgment scheduler receipt is observable; pending candidates are not converged")
    elif pending_count and (judgment_scheduler_receipt_status == "failed" or not judgment_receipt_fresh):
        state = "scheduler-stalled"
        caveats.append(
            "judgment scheduler has no fresh successful receipt; the bounded retry route is the next daemon tick"
        )
    elif pending_count:
        state = "pending"
    elif producer_fresh and scheduler_state == "fresh":
        state = "healthy-empty"
    else:
        state = "empty-unverified"
        if producer_status is None:
            caveats.append("no successful standing-queries producer event is observable")
        if scheduler_state != "fresh":
            caveats.append("a fresh scheduler heartbeat is not observable")

    return AssertionCandidateQueueHealthPayload(
        state=state,
        observed_at_ms=observed_at_ms,
        pending_count=pending_count,
        status_counts=status_counts,
        kind_counts=kind_counts,
        source_counts=source_counts,
        oldest_pending_at_ms=oldest_pending_at_ms,
        newest_pending_at_ms=newest_pending_at_ms,
        oldest_pending_age_ms=None if oldest_pending_at_ms is None else max(0, observed_at_ms - oldest_pending_at_ms),
        stale_pending_count=stale_pending_count,
        retention_outcome="retained-visible" if stale_pending_count else "none",
        producer_status=producer_status,
        producer_observed_at_ms=producer_observed_at_ms,
        producer_age_ms=producer_age_ms,
        scheduler_state=scheduler_state,
        scheduler_heartbeat_at_ms=scheduler_heartbeat_at_ms,
        scheduler_heartbeat_age_ms=heartbeat_age_ms,
        judgment_scheduler_receipt_status=judgment_scheduler_receipt_status,
        judgment_scheduler_receipt_at_ms=judgment_scheduler_receipt_at_ms,
        judgment_scheduler_receipt_age_ms=judgment_receipt_age_ms,
        judgment_scheduler_receipt_reason=judgment_scheduler_receipt_reason,
        judgment_scheduler_receipt_retryable=judgment_scheduler_receipt_retryable,
        judgment_scheduler_receipt_retry_route=judgment_scheduler_receipt_retry_route,
        judgment_scheduler_receipt_batch_limit=judgment_scheduler_receipt_batch_limit,
        judgment_scheduler_receipt_considered=judgment_scheduler_receipt_considered,
        judgment_scheduler_receipt_accepted=judgment_scheduler_receipt_accepted,
        judgment_scheduler_receipt_rejected=judgment_scheduler_receipt_rejected,
        judgment_scheduler_receipt_escalated=judgment_scheduler_receipt_escalated,
        judgment_scheduler_receipt_idempotent=judgment_scheduler_receipt_idempotent,
        judgment_scheduler_receipt_failed=judgment_scheduler_receipt_failed,
        judgment_scheduler_receipt_persistence_degraded=judgment_scheduler_receipt_persistence_degraded,
        judgment_scheduler_receipt_persistence_recovered=judgment_scheduler_receipt_persistence_recovered,
        producer_debt_count=producer_debt_count,
        caveats=tuple(dict.fromkeys(caveats)),
    )


def _archive_list_comparative_judgments(config: Config) -> Any:
    """Read back every live comparative-judgment assertion row."""

    from polylogue.storage.sqlite.archive_tiers.user_write import list_comparative_judgments

    user_db = _active_archive_root(config) / "user.db"
    if not user_db.exists():
        return []
    try:
        conn = open_readonly_connection(user_db)
        try:
            return list_comparative_judgments(conn)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"failed to list comparative judgments: {exc}") from exc


def _archive_count_table_rows(conn: Any, table_name: str) -> int | None:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    if row is None:
        return None
    count_row = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(count_row[0] or 0) if count_row is not None else 0


def _archive_search_hit_to_domain(hit: ArchiveSessionSearchHit) -> SearchHit:
    return SearchHit(
        session_id=hit.session_id,
        source_name=hit.origin,
        message_id=hit.message_id,
        title=hit.title,
        timestamp=None,
        snippet=hit.snippet,
        session_url=session_web_url(hit.session_id),
    )


def _archive_search_hit_to_payload(
    hit: ArchiveSessionSearchHit, summary: ArchiveSessionSummary
) -> SessionSearchHitPayload:
    from polylogue.surfaces.payloads import (
        SessionSearchHitPayload,
        SessionSearchMatchPayload,
        TargetRefPayload,
        reader_anchor,
        reader_message_actions,
        session_summary_envelope_from_summary,
    )

    return SessionSearchHitPayload(
        session=session_summary_envelope_from_summary(
            archive_summary_to_domain(summary),
            message_count=summary.message_count,
        ),
        match=SessionSearchMatchPayload(
            rank=hit.rank,
            retrieval_lane="dialogue",
            match_surface="message",
            target_ref=TargetRefPayload.message(session_id=hit.session_id, message_id=hit.message_id),
            anchor=reader_anchor("message", hit.message_id),
            actions=reader_message_actions(),
            message_id=hit.message_id,
            snippet=hit.snippet,
            score=None,
        ),
    )


class _ArchiveNeighborRuntime:
    """Minimal neighbor discovery store adapter for archive neighbor discovery.

    Implements: NeighborStore protocol (resolve_id, get, list_summaries_by_query, search_summary_hits)
    """

    def __init__(self, archive: Any) -> None:
        self._archive = archive

    async def resolve_id(self, id_prefix: str, *, strict: bool = False) -> SessionId | None:
        del strict
        try:
            return SessionId(self._archive.resolve_session_id(id_prefix))
        except KeyError:
            return None

    async def get(self, session_id: str) -> Session | None:
        try:
            resolved = self._archive.resolve_session_id(session_id)
            summary = self._archive.read_summary(resolved)
            return archive_envelope_to_session(
                self._archive.read_session(resolved),
                display_label=summary.display_label,
                display_label_source=summary.display_label_source,
            )
        except KeyError:
            return None

    async def list_summaries_by_query(self, query: SessionRecordQuery) -> builtins.list[SessionSummary]:
        origin, origins = self._origin_filters(
            origin=query.origin,
            origins=builtins.list(query.origins) if query.origins else [],
        )
        return [
            archive_summary_to_domain(summary)
            for summary in self._archive.list_summaries(
                limit=query.limit or 50,
                offset=query.offset or 0,
                origin=origin,
                origins=origins,
                referenced_paths=tuple(query.referenced_path or ()),
                cwd_prefix=query.cwd_prefix,
                action_terms=tuple(query.action_terms or ()),
                excluded_action_terms=tuple(query.excluded_action_terms or ()),
                tool_terms=tuple(query.tool_terms or ()),
                excluded_tool_terms=tuple(query.excluded_tool_terms or ()),
                has_tool_use=query.has_tool_use or False,
                has_thinking=query.has_thinking or False,
                message_type=_archive_message_type(query.message_type),
                title=query.title_contains,
                min_messages=query.min_messages,
                max_messages=query.max_messages,
                min_words=query.min_words,
                max_words=query.max_words,
                since_ms=_archive_query_date_ms("since", query.since),
                until_ms=_archive_query_date_ms("until", query.until),
            )
        ]

    async def search_summary_hits(
        self,
        query: str,
        limit: int = 20,
        origins: builtins.list[str] | None = None,
        since: str | None = None,
    ) -> builtins.list[SessionSearchHit]:
        from polylogue.archive.query.search_hits import session_search_hit_from_summary

        _origin, filter_origins = self._origin_filters(origin=None, origins=origins if origins else [])
        hits = self._archive.search_summaries(
            query,
            limit=limit,
            origins=filter_origins,
            since_ms=_archive_query_date_ms("since", since),
        )
        results: builtins.list[SessionSearchHit] = []
        for hit in hits:
            try:
                summary = archive_summary_to_domain(self._archive.read_summary(hit.session_id))
            except KeyError:
                continue
            results.append(
                session_search_hit_from_summary(
                    summary,
                    rank=hit.rank,
                    retrieval_lane="dialogue",
                    match_surface="message",
                    message_id=hit.message_id,
                    snippet=hit.snippet,
                    score=None,
                )
            )
        return results

    def _origin_filters(
        self,
        *,
        origin: str | None,
        origins: builtins.list[str],
    ) -> tuple[str | None, tuple[str, ...]]:
        validated_origin = Origin(origin).value if origin is not None else None
        validated_origins = tuple(Origin(value).value for value in origins)
        return validated_origin, validated_origins


def _actions_for_session(session: Session) -> tuple[Action, ...]:
    """Derive ordered actions from an archive session's tool blocks.

    Each message's content blocks are parsed into tool calls, then promoted
    to ``Action`` records. No storage round-trip — the domain
    session already carries the content blocks the actions are built from.
    """
    from polylogue.archive.actions.actions import build_actions, build_tool_calls_from_content_blocks

    # Keep pairing aligned with the canonical action_pairs view: rank uses and
    # results independently by (message position, variant index, block
    # position), then join equal ranks for each (session, tool_id). This is
    # intentionally session-wide because providers commonly emit a tool use in
    # one message and its result in a later message.
    ordered_messages = [
        message
        for _input_index, message in sorted(
            enumerate(session.messages),
            key=lambda item: (item[1].position, item[1].branch_index, item[0]),
        )
    ]
    uses_by_tool_id: dict[str, builtins.list[Mapping[str, object]]] = {}
    results_by_tool_id: dict[str, builtins.list[Mapping[str, object]]] = {}
    for message in ordered_messages:
        for block in message.blocks:
            block_type = str(block.get("type"))
            tool_id = block.get("tool_id")
            if not isinstance(tool_id, str) or not tool_id:
                continue
            if block_type == "tool_use":
                uses_by_tool_id.setdefault(tool_id, []).append(block)
            elif block_type == "tool_result":
                results_by_tool_id.setdefault(tool_id, []).append(block)

    result_block_by_use_block_id: dict[int, Mapping[str, object] | None] = {}
    for tool_id, use_blocks in uses_by_tool_id.items():
        result_blocks = results_by_tool_id.get(tool_id, ())
        for rank, use_block in enumerate(use_blocks):
            result_block_by_use_block_id[id(use_block)] = result_blocks[rank] if rank < len(result_blocks) else None

    events: builtins.list[Action] = []
    for message in ordered_messages:
        calls = build_tool_calls_from_content_blocks(
            origin=session.origin,
            content_blocks=message.blocks,
            result_block_by_use_block_id=result_block_by_use_block_id,
        )
        events.extend(build_actions(message, calls))
    return tuple(events)


def _archive_message_matches(
    message: Message,
    *,
    message_role: MessageRoleFilter,
    message_type: MessageTypeName | None,
    material_origin: tuple[MaterialOrigin, ...] = (),
    since_ms: int | None = None,
    until_ms: int | None = None,
) -> bool:
    if message_role and message.role not in message_role:
        return False
    if message_type is not None and message.message_type != MessageType.normalize(message_type):
        return False
    if material_origin and message.material_origin not in material_origin:
        return False
    occurred_ms = int(message.timestamp.timestamp() * 1000) if message.timestamp is not None else None
    if since_ms is not None and (occurred_ms is None or occurred_ms < since_ms):
        return False
    return not (until_ms is not None and (occurred_ms is None or occurred_ms > until_ms))


class PolylogueArchiveMixin(ArchiveReadCapability):
    if TYPE_CHECKING:

        @property
        def config(self) -> Config: ...

        @property
        def repository(self) -> SessionRepository: ...

    async def import_annotation_batch(
        self,
        request: AnnotationBatchImportRequest,
        *,
        input: BinaryIO,
        registry: AnnotationSchemaRegistry | None = None,
    ) -> AnnotationBatchImportResult:
        """Import annotation candidates and return the committed batch summary.

        This is the library binding for the shared annotation-import product
        operation used by the CLI and MCP surfaces. ``registry`` lets callers
        use a deliberately constructed schema registry without bypassing the
        facade. Exact assertions and validation errors are paged by resolving
        the returned batch_ref with limit/offset. A readback failure does not
        undo the completed import.
        """
        from polylogue.annotations.importer import AnnotationBatchImportResult
        from polylogue.annotations.schema import ANNOTATION_SCHEMA_REGISTRY
        from polylogue.api.facade_client import submit_facade_operation

        payload = request.model_dump(mode="json")
        schema_registry = registry if registry is not None else ANNOTATION_SCHEMA_REGISTRY
        try:
            schema = schema_registry.get(request.schema_id, request.schema_version)
        except KeyError:
            if registry is not None:
                raise
        else:
            payload["schema_definition_json"] = schema.canonical_definition_json()
        state = await submit_facade_operation(self.config, "mutation.annotation.import_batch", payload, input=input)
        return AnnotationBatchImportResult.model_validate(state["result"])

    async def get_session(
        self,
        session_id: str,
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> Session | None:
        """Read one session with its WHOLE composed transcript.

        A caller that renders a window wants :meth:`get_session_page`: this
        method composes every message before returning, so its cost is the
        session's length by construction.
        """

        page = await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.session.get",
            arguments={"session_id": session_id, "content_projection": content_projection},
            work=lambda archive: _read_session_transcript_page(
                archive,
                session_id,
                limit=None,
                offset=0,
                content_projection=content_projection,
            ),
            projection="session",
            stable_order="session,message,block",
        )
        return None if page is None else page.session

    async def get_session_page(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        content_projection: ContentProjectionSpec | None = None,
    ) -> SessionTranscriptPage | None:
        """Read one session bounded to ``[offset, offset + limit)`` messages.

        The window is composed at the storage layer
        (``ArchiveStore.read_session_page``), so a page of a long session --
        or of a deep prefix-sharing lineage child -- costs the page, not the
        transcript (polylogue-2go3o). ``limit=None`` is the declared whole
        transcript, the one shape that legitimately composes everything.

        The returned page carries the session's true totals; see
        :class:`SessionTranscriptPage` for why they cannot be recomputed from
        the rows it serves.
        """

        normalized_offset = max(int(offset), 0)
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.session.page",
            arguments={
                "session_id": session_id,
                "limit": limit,
                "offset": normalized_offset,
                "content_projection": content_projection,
            },
            work=lambda archive: _read_session_transcript_page(
                archive,
                session_id,
                limit=limit,
                offset=normalized_offset,
                content_projection=content_projection,
            ),
            page_size=limit,
            projection="session",
            stable_order="session,message,block",
        )

    async def explain_import(
        self,
        path: str | Path | None = None,
        *,
        raw_ref: str | None = None,
        source_path: str | None = None,
        source_name: str = "unknown",
        limit: int = 100,
        redact_paths: bool = True,
    ) -> ImportExplainPayload:
        """Explain detector/parser decisions for local or archived import evidence."""
        from polylogue.sources.import_explain import explain_import_archive, explain_import_path

        if raw_ref is not None or source_path is not None:
            return explain_import_archive(
                _active_archive_root(self.config),
                raw_ref=raw_ref,
                source_path=source_path,
                limit=limit,
                redact_paths=redact_paths,
            )
        if path is None:
            raise ValueError("path is required unless raw_ref or source_path is provided")
        return explain_import_path(Path(path), source_name=source_name, limit=limit)

    async def _session_digest(self, session_id: str) -> SessionDigest | None:
        """Compile one resolved session and its child links into a session digest."""
        from polylogue.analysis.transforms import compile_session_digest
        from polylogue.storage.query_models import SessionRecordQuery

        session = await self.get_session(session_id)
        if session is None:
            return None
        resolved_session_id = str(session.id)
        # The envelope read that backs get_session does not carry session_events,
        # and the digest's compaction geometry (polylogue-4ts.5) is stored there.
        events = await self.repository.get_session_event_models(resolved_session_id)
        session = session.model_copy(update={"session_events": tuple(events)})
        session_links: list[dict[str, object]] = await self.repository.queries.list_session_links_for_session(
            resolved_session_id
        )
        children = await self.repository.queries.list_sessions(SessionRecordQuery(parent_id=resolved_session_id))
        session_links.extend(
            {
                "dst_origin": child.origin.value,
                "dst_native_id": child.native_id,
                "resolved_dst_session_id": str(child.session_id),
                "status": "resolved",
                "link_type": child.branch_type.value if child.branch_type is not None else "child",
            }
            for child in children
        )
        return compile_session_digest(session, session_links=session_links)

    async def postmortem_bundle(
        self,
        spec: SessionQuerySpec | None = None,
        *,
        limit: int | None = None,
    ) -> PostmortemBundle:
        """Compile a distilled postmortem bundle over a matched session scope (#2380).

        Resolves the matched session set from ``spec`` (the same summary path
        ``facets`` uses), batch-fetches profiles, compiles per-session
        digests for a bounded set, and delegates aggregation to the pure
        :func:`compile_postmortem_bundle`. The analysis cap defaults to 200
        sessions; when more match, the bundle is marked ``truncated`` and the
        drop is logged rather than silently capped.
        """
        from polylogue.analysis.postmortem import (
            PostmortemBundle as _PostmortemBundle,
        )
        from polylogue.analysis.postmortem import (
            PostmortemScope,
            compile_postmortem_bundle,
        )

        if limit is not None and limit <= 0:
            raise ValueError("limit must be a positive integer")
        cap = limit if limit is not None else 200
        summaries = await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.postmortem.scope",
            arguments={"spec": spec, "limit": cap},
            work=lambda archive: (
                _archive_list_summaries_for_spec(archive, replace(spec, limit=None, offset=0), default_limit=1_000_000)
                if spec is not None and spec.has_filters()
                else archive.list_summaries(limit=1_000_000)
            ),
            # The analysis cap applies after the authoritative scope is
            # resolved. Reading only ``cap`` summaries would make a bounded
            # analysis look complete and hide the matched denominator.
            page_size=1_000_000,
            projection="session-scope",
            workload_class="scan",
        )

        session_ids: list[str] = []
        seen: set[str] = set()
        for summary in summaries:
            if summary.session_id in seen:
                continue
            seen.add(summary.session_id)
            session_ids.append(summary.session_id)

        matched = len(session_ids)
        truncated = matched > cap
        dropped = matched - cap if truncated else 0
        analyzed_ids = session_ids[:cap]
        if truncated:
            emit(
                "archive.postmortem_bundle.truncated",
                level=WARNING,
                outcome="degraded",
                reason="match_cap_exceeded",
                considered=matched,
                limit=cap,
                skipped=dropped,
            )

        profiles_map = await self.repository.get_session_profiles_batch(analyzed_ids)
        profiles = [profiles_map[sid] for sid in analyzed_ids if sid in profiles_map]

        digests: dict[str, SessionDigest] = {}
        for sid in analyzed_ids:
            digest = await self._session_digest(sid)
            if digest is not None:
                digests[sid] = digest

        scope = PostmortemScope(
            since=spec.since if spec is not None else None,
            until=spec.until if spec is not None else None,
            query=_archive_text_query(spec) if spec is not None else None,
            matched_session_count=matched,
            analyzed_session_count=len(profiles),
            truncated=truncated,
            dropped_session_count=dropped,
        )
        bundle = compile_postmortem_bundle(profiles, digests, scope=scope)
        assert isinstance(bundle, _PostmortemBundle)
        return bundle

    async def pathology_report(
        self,
        spec: SessionQuerySpec | None = None,
        *,
        limit: int | None = None,
    ) -> PathologyReport:
        """Mine agent-workflow pathologies across a matched session scope (#2383).

        Resolves the matched session set (the same summary path
        ``postmortem_bundle`` uses), fetches each session's session digest, and
        runs the deterministic detectors in :mod:`polylogue.analysis.pathology`
        over the typed run projections. Returns the aggregate
        :class:`PathologyReport` (findings + per-kind distribution), the
        queryable distribution/summary view for #2383. The analysis cap defaults
        to 200 sessions.
        """
        from polylogue.analysis.pathology import compile_pathology_report

        if limit is not None and limit <= 0:
            raise ValueError("limit must be a positive integer")
        cap = limit if limit is not None else 200
        summaries = await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.pathology.scope",
            arguments={"spec": spec, "limit": cap},
            work=lambda archive: (
                _archive_list_summaries_for_spec(archive, replace(spec, limit=None, offset=0), default_limit=1_000_000)
                if spec is not None and spec.has_filters()
                else archive.list_summaries(limit=1_000_000)
            ),
            page_size=cap,
            projection="session-scope",
            workload_class="scan",
        )

        session_ids: list[str] = []
        seen: set[str] = set()
        for summary in summaries:
            if summary.session_id in seen:
                continue
            seen.add(summary.session_id)
            session_ids.append(summary.session_id)

        matched = len(session_ids)
        analyzed_ids = session_ids[:cap]
        if matched > cap:
            emit(
                "archive.pathology_report.truncated",
                level=WARNING,
                outcome="degraded",
                reason="match_cap_exceeded",
                considered=matched,
                limit=cap,
                skipped=matched - cap,
            )
        projections = []
        failed = 0
        for sid in analyzed_ids:
            digest = await self._session_digest(sid)
            if digest is None:
                # A session inside the analyzed slice whose digest is missing
                # was never examined. Counting it as analyzed would report an
                # unmeasured session as a measured pathology-free one.
                failed += 1
                continue
            projections.append(digest.run_projection)
        if failed:
            emit(
                "archive.pathology_report.digest_unavailable",
                level=WARNING,
                outcome="degraded",
                reason="session_digest_unavailable",
                considered=len(analyzed_ids),
                failed=failed,
            )
        report = compile_pathology_report(projections)
        return report.model_copy(
            update={
                "matched_session_count": matched,
                "analyzed_session_count": len(projections),
                "truncated": matched > cap,
                "dropped_session_count": max(0, matched - cap),
                "failed_session_count": failed,
            }
        )

    async def portfolio_bundle(
        self,
        spec: SessionQuerySpec | None = None,
        *,
        limit: int | None = None,
        top_n: int = 10,
    ) -> PortfolioBundle:
        """Compile a corpus-wide portfolio report (#2437).

        Resolves the matched session set (the same summary path
        ``postmortem_bundle`` uses), batch-fetches profiles + session digests,
        and delegates aggregation to the pure :func:`compile_portfolio_bundle`
        (session/repo/origin counts, cost + wall-clock distributions, pathology
        and context-loss distribution). The analysis cap defaults to 200 sessions.
        """
        from polylogue.analysis.portfolio import (
            compile_portfolio_bundle,
        )
        from polylogue.analysis.postmortem import PostmortemScope

        if limit is not None and limit <= 0:
            raise ValueError("limit must be a positive integer")
        cap = limit if limit is not None else 200
        summaries = await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.portfolio.scope",
            arguments={"spec": spec, "limit": cap},
            work=lambda archive: (
                _archive_list_summaries_for_spec(archive, spec, default_limit=1_000_000)
                if spec is not None and spec.has_filters()
                else archive.list_summaries(limit=1_000_000)
            ),
            page_size=cap,
            projection="session-scope",
            workload_class="scan",
        )

        session_ids: list[str] = []
        seen: set[str] = set()
        for summary in summaries:
            if summary.session_id in seen:
                continue
            seen.add(summary.session_id)
            session_ids.append(summary.session_id)

        matched = len(session_ids)
        truncated = matched > cap
        dropped = matched - cap if truncated else 0
        analyzed_ids = session_ids[:cap]
        if truncated:
            emit(
                "archive.portfolio_bundle.truncated",
                level=WARNING,
                outcome="degraded",
                reason="match_cap_exceeded",
                considered=matched,
                limit=cap,
                skipped=dropped,
            )

        profiles_map = await self.repository.get_session_profiles_batch(analyzed_ids)
        profiles = [profiles_map[sid] for sid in analyzed_ids if sid in profiles_map]

        digests: dict[str, SessionDigest] = {}
        for sid in analyzed_ids:
            digest = await self._session_digest(sid)
            if digest is not None:
                digests[sid] = digest

        scope = PostmortemScope(
            since=spec.since if spec is not None else None,
            until=spec.until if spec is not None else None,
            query=_archive_text_query(spec) if spec is not None else None,
            matched_session_count=matched,
            analyzed_session_count=len(profiles),
            truncated=truncated,
            dropped_session_count=dropped,
        )
        return compile_portfolio_bundle(profiles, digests, scope=scope, top_n=top_n)

    async def list_assertion_claims(
        self,
        *,
        kinds: Sequence[str | AssertionKind] | None = None,
        target_ref: str | None = None,
        session_id: str | None = None,
        target_refs: Collection[str] | None = None,
        scope_ref: str | None = None,
        statuses: Sequence[str | AssertionStatus] | None = ("active", "candidate"),
        context_inject: bool | None = None,
        limit: int | None = None,
    ) -> list[ArchiveAssertionEnvelope]:
        """List assertion-backed lifecycle claims for read-surface consumers.

        ``target_refs`` narrows the read to any of several targets inside the
        storage query. ``session_id`` includes exact session and composed-message
        targets, refusing incomplete lineage rather than hiding missing evidence.
        """

        from polylogue.storage.sqlite.archive_tiers.user_write import ASSERTION_CLAIM_KINDS, list_assertion_claims

        root = _active_archive_root(self.config)

        def read(archive: ArchiveStore) -> list[ArchiveAssertionEnvelope]:
            archive.require_user_tier()
            try:
                return list_assertion_claims(
                    archive._conn,
                    schema="user_tier",
                    kinds=ASSERTION_CLAIM_KINDS if kinds is None else kinds,
                    target_ref=target_ref,
                    session_id=session_id,
                    target_refs=target_refs,
                    scope_ref=scope_ref,
                    statuses=statuses,
                    context_inject=context_inject,
                    limit=limit,
                )
            except sqlite3.Error as exc:
                # A durable read failure is a typed refusal, never an empty
                # (and therefore clean-looking) claim list.
                raise ArchiveTierUnavailableError(
                    tier="user.db",
                    path=str(archive.user_db_path.resolve(strict=False)),
                    reason=f"cannot read assertions ({exc})",
                    guidance="restore the durable user tier at this path, then retry",
                ) from exc

        return await run_archive_read(
            root,
            operation="archive.assertion.claims",
            arguments={
                "kinds": tuple(str(kind) for kind in kinds) if kinds is not None else None,
                "target_ref": target_ref,
                "session_id": session_id,
                "target_refs": tuple(sorted(target_refs)) if target_refs is not None else None,
                "scope_ref": scope_ref,
                "statuses": tuple(str(status) for status in statuses) if statuses is not None else None,
                "context_inject": context_inject,
                "limit": limit,
            },
            work=read,
            projection="assertion-claims",
            stable_order="updated_at_ms,assertion_id",
        )

    async def get_context_delivery(
        self, snapshot_ref: str, *, recipient_ref: str
    ) -> ArchiveContextDeliveryEnvelope | None:
        """Return an exact delivery receipt only for its recorded recipient.

        This is a read-only audit seam over the durable user-tier receipt. It
        does not schedule recall or grant a caller broader archive access.
        """

        return _archive_get_context_delivery(
            self.config,
            snapshot_ref=snapshot_ref,
            recipient_ref=recipient_ref,
        )

    async def list_context_deliveries(
        self,
        *,
        recipient_ref: str | None = None,
        assertion_ref: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ArchiveContextDeliveryPage:
        """Read one counted receipt-summary page from durable user authority.

        Count and rows come from one SQLite snapshot. Summaries never read the
        context image; exact recipient-scoped disclosure uses get_context_delivery.
        Follow next_offset until it is null to enumerate the requested scope.
        """

        return _archive_list_context_deliveries(
            self.config,
            recipient_ref=recipient_ref,
            assertion_ref=assertion_ref,
            limit=limit,
            offset=offset,
        )

    async def list_context_injection_ledger(
        self,
        *,
        target_session: str | None = None,
        execution_context_ref: str | None = None,
        limit: int = 100,
    ) -> list[ContextLedgerRecord]:
        """Read bounded scheduler admission receipts from ``ops.db``."""

        return _archive_list_context_injection_ledger(
            self.config,
            target_session=target_session,
            execution_context_ref=execution_context_ref,
            limit=max(1, min(limit, 500)),
        )

    async def record_context_delivery(
        self,
        *,
        image: ContextImage,
        boundary: str,
        recipient_ref: str,
        delivered_by_ref: str,
        run_ref: str | None = None,
        inheritance_mode: str = "explicit",
    ) -> ArchiveContextDeliveryEnvelope:
        """Persist one exact delivery receipt for an already-compiled context image.

        This is the low-level delivery boundary: it records exactly the
        ``image`` passed in, scoped to ``recipient_ref``. Exact retries
        (identical image + identity fields) are idempotent; any drift in the
        immutable delivery identity is rejected. Most callers should use
        :meth:`compile_and_record_context`, which also performs the
        compilation and returns the exact image alongside the receipt.
        """

        from polylogue.api.facade_client import submit_facade_writer

        value = await submit_facade_writer(
            self.config,
            "record_context_delivery",
            {
                "image": image.model_dump(mode="json"),
                "boundary": boundary,
                "recipient_ref": recipient_ref,
                "delivered_by_ref": delivered_by_ref,
                "run_ref": run_ref,
                "inheritance_mode": inheritance_mode,
            },
        )
        return ArchiveContextDeliveryEnvelope(
            **{
                **value,
                "context_image": image,
                "segment_refs": tuple(value["segment_refs"]),
                "evidence_refs": tuple(value["evidence_refs"]),
                "assertion_refs": tuple(value["assertion_refs"]),
                "omissions": tuple(value["omissions"]),
                "caveats": tuple(value["caveats"]),
            }
        )

    async def compile_and_record_context(
        self,
        *,
        recipient_ref: str,
        delivered_by_ref: str,
        boundary: str,
        query: str | None = None,
        max_sessions: int = 5,
        max_tokens: int | None = None,
        include_messages: bool = True,
        include_assertions: bool = True,
        redact_paths: bool = True,
        seed_session_id: str | None = None,
        segment_profile: Literal["default", "prose_with_refs"] = "default",
        run_ref: str | None = None,
        inheritance_mode: str = "explicit",
    ) -> ArchiveContextDeliveryEnvelope:
        """Compile one bounded context image and record its exact delivery receipt.

        This is the named delivery boundary: it compiles through
        :meth:`context_image_payload` (the same engine every other context
        surface uses -- no parallel compiler) and the returned receipt's
        ``context_image`` is exactly the image that was compiled, so the call
        itself is evidence of what crossed the boundary, not merely that
        compilation happened.
        """

        image = await self.context_image_payload(
            query=query,
            max_sessions=max_sessions,
            max_tokens=max_tokens,
            include_messages=include_messages,
            include_assertions=include_assertions,
            redact_paths=redact_paths,
            seed_session_id=seed_session_id,
            segment_profile=segment_profile,
        )
        return await self.record_context_delivery(
            image=image,
            boundary=boundary,
            recipient_ref=recipient_ref,
            delivered_by_ref=delivered_by_ref,
            run_ref=run_ref,
            inheritance_mode=inheritance_mode,
        )

    async def correlate_hermes_context_deliveries(
        self, hermes_session_native_id: str
    ) -> tuple[HermesContextDeliveryCorrelation, ...]:
        """Correlate a Hermes session's ``context_injected`` events with delivery receipts.

        Read-only audit seam (fs1.11 x fs1.7): for every durable
        ``context_injected`` lifecycle event drained for this Hermes session,
        resolves the exact delivered context-image bytes, token budget, and
        rendered-token estimate from the existing delivery ledger. An event
        with no resolvable receipt (the write has not committed yet) is returned with ``available=False`` and an explicit
        caveat rather than omitted.
        """

        return _archive_correlate_hermes_context_deliveries(
            self.config,
            hermes_session_native_id=hermes_session_native_id,
        )

    async def reconcile_hermes_session_lifecycle(
        self, hermes_session_native_id: str
    ) -> HermesLifecycleReconciliation | None:
        """Reconcile a Hermes session's drained lifecycle-event stream (fs1.7 AC).

        Read-only audit seam over the durable spool (source.db
        ``raw_hook_events``) and the ingested session snapshot (index.db
        ``messages``): renders unpaired start/finish events, per-turn-end
        without durable finalization, and events referencing a message id
        the snapshot does not retain -- all *visible* in one report rather
        than left as an assumption. Returns ``None`` only when the archive
        itself is not yet initialized; a session with zero drained events
        still returns a well-formed report (``total_events == 0``).
        """

        return _archive_reconcile_hermes_session_lifecycle(
            self.config,
            hermes_session_native_id=hermes_session_native_id,
        )

    async def correlate_claude_agent_dispatches(self) -> ClaudeAgentDispatchCorrelation | None:
        """Resolve Claude Code hook-asserted agent identity onto archived tool calls (bd polylogue-xo9gq).

        Read-only audit seam over the durable spool (source.db
        ``raw_hook_events``: ``PreToolUse``/``PostToolUse`` payloads carrying
        ``agent_id``/``agent_type``) and the ingested block tree (index.db
        ``blocks`` of type ``tool_use``, joined on ``tool_id``). Reports which
        calls a dispatched agent instance is attributed to, which asserted
        calls are not archived yet, and which ``tool_use_id`` values two
        different agent instances both claim. Returns ``None`` only when the archive
        itself is not yet initialized.
        """

        return _archive_correlate_claude_agent_dispatches(self.config)

    async def reconcile_codex_spawn_edges(self) -> CodexSpawnEdgeReconciliation | None:
        """Reconcile projected Codex spawn edges against inferred topology.

        Read-only audit seam over index.db: the thread-state graph
        (``work_evidence_edges``) projected from the retained state export, and the ingested topology
        (``session_links``, ``BranchType.SUBAGENT`` edges
        ``sources/parsers/codex.py`` infers structurally from each child
        session's own transcript). Reports how many transcript-inferred
        edges are backed by Codex's own orchestration-level record, and how
        many edges exist in each source but not the other (Codex's own
        record can carry edges from a crashed or still-running child the
        transcript never proves). Returns ``None`` only when the archive
        itself is not yet initialized.
        """

        return _archive_reconcile_codex_spawn_edges(self.config)

    async def hermes_integration_health(self) -> HermesIntegrationHealth:
        """Return the bounded Hermes-to-Polylogue integration health rollup (fs1.15).

        Composes existing evidence only: per-source freshness/cursor
        position (polylogue-1xc.13), a bounded dry-run parser/fidelity pass
        over the Hermes runtime root, convergence debt bucketed to the
        Hermes source family, lifecycle-event pairing debt (fs1.7), and
        context-delivery correlation state (fs1.11). Reports an explicit
        ``disabled``/``unavailable``/``degraded``/``healthy`` verdict rather
        than a silent zero when a producer is absent, a source is stale, an
        event is malformed, or the archive itself is unavailable. Never
        includes raw transcript text, credentials, or absolute filesystem
        paths.
        """

        from polylogue.operations.hermes_health import configured_hermes_health

        return configured_hermes_health(self.config)

    async def list_assertion_claim_payloads(
        self,
        *,
        kinds: Sequence[str | AssertionKind] | None = None,
        target_ref: str | None = None,
        session_id: str | None = None,
        scope_ref: str | None = None,
        statuses: Sequence[str | AssertionStatus] | None = ("active", "candidate"),
        context_inject: bool | None = None,
        limit: int | None = None,
    ) -> list[AssertionClaimPayload]:
        """List assertion claims using the shared API payload shape.

        Daemon/MCP/web adapters should consume this method instead of
        importing storage-tier assertion helpers directly. The storage
        tier remains the durable owner of assertion rows; this API method
        owns the cross-surface JSON boundary for read consumers.
        """

        from polylogue.surfaces.payloads import AssertionClaimPayload

        claims = await self.list_assertion_claims(
            kinds=kinds,
            target_ref=target_ref,
            session_id=session_id,
            scope_ref=scope_ref,
            statuses=statuses,
            context_inject=context_inject,
            limit=limit,
        )
        return [AssertionClaimPayload.from_envelope(claim) for claim in claims]

    async def list_assertion_candidates(
        self,
        *,
        target_ref: str | None = None,
        kinds: Sequence[str | AssertionKind] | None = None,
        limit: int | None = None,
    ) -> list[AssertionClaimPayload]:
        """List candidate assertion claims awaiting explicit judgment."""

        from polylogue.surfaces.payloads import AssertionClaimPayload

        candidates = cast(
            list["ArchiveAssertionEnvelope"],
            _archive_list_assertion_candidates(self.config, target_ref=target_ref, kinds=kinds, limit=limit),
        )
        return [AssertionClaimPayload.from_envelope(candidate) for candidate in candidates]

    async def list_assertion_candidate_reviews(
        self,
        *,
        target_ref: str | None = None,
        kinds: Sequence[str | AssertionKind] | None = None,
        statuses: Sequence[str | AssertionStatus] | None = None,
        limit: int | None = None,
    ) -> AssertionCandidateReviewListPayload:
        """List candidate assertion review state separately from active claims."""

        from polylogue.storage.sqlite.archive_tiers.user_write import ASSERTION_CANDIDATE_REVIEW_STATUSES
        from polylogue.surfaces.payloads import (
            AssertionCandidateReviewListPayload,
            AssertionEvidencePreviewPayload,
        )

        candidate_statuses = ASSERTION_CANDIDATE_REVIEW_STATUSES if statuses is None else statuses
        review_rows, matched = _archive_list_assertion_candidate_reviews(
            self.config,
            target_ref=target_ref,
            kinds=kinds,
            statuses=candidate_statuses,
            limit=limit,
        )
        evidence_previews: dict[str, tuple[AssertionEvidencePreviewPayload, ...]] = {}
        for review in review_rows:
            previews: list[AssertionEvidencePreviewPayload] = []
            for evidence_ref in review.candidate.evidence_refs[:5]:
                try:
                    resolution = await self.resolve_ref(evidence_ref)
                except Exception as exc:
                    previews.append(
                        AssertionEvidencePreviewPayload(
                            ref=evidence_ref,
                            state="error",
                            reason=str(exc)[:300],
                        )
                    )
                    continue
                caveat = next(iter(resolution.caveats), None)
                diagnostic = " ".join(
                    value for value in (caveat, resolution.summary, resolution.payload_kind) if value
                ).lower()
                evidence_state: AssertionEvidenceResolutionState
                if resolution.resolved:
                    evidence_state = "resolved"
                elif "unsupported" in diagnostic:
                    evidence_state = "unsupported"
                elif "pending" in diagnostic or resolution.payload_kind == "pending":
                    evidence_state = "pending"
                else:
                    evidence_state = "missing"
                previews.append(
                    AssertionEvidencePreviewPayload(
                        ref=evidence_ref,
                        state=evidence_state,
                        kind=resolution.kind,
                        title=None if resolution.title is None else resolution.title[:200],
                        excerpt=None if resolution.summary is None else resolution.summary[:600],
                        reason=None if resolution.resolved else (caveat or "reference did not resolve")[:300],
                        open_commands=tuple(
                            action.command
                            for action in resolution.actions
                            if action.enabled and action.command is not None
                        )[:3],
                        open_hrefs=tuple(
                            action.href for action in resolution.actions if action.enabled and action.href is not None
                        )[:3],
                    )
                )
            evidence_previews[review.candidate.assertion_id] = tuple(previews)
        return AssertionCandidateReviewListPayload.from_envelopes(
            review_rows,
            limit=limit if limit is not None else len(review_rows),
            target_ref=target_ref,
            candidate_statuses=candidate_statuses,
            evidence_previews=evidence_previews,
            matched=matched,
        )

    async def assertion_candidate_queue_health(self) -> AssertionCandidateQueueHealthPayload:
        """Return a non-destructive health projection for the judgment queue."""

        return _archive_assertion_candidate_queue_health(self.config)

    async def judge_assertion_candidate(
        self,
        *,
        candidate_ref: str,
        decision: str,
        reason: str | None = None,
        actor_ref: str = "user:local",
        inject: bool = False,
        replacement_kind: str | None = None,
        replacement_body_text: str | None = None,
        replacement_value: object | None = None,
    ) -> AssertionJudgmentResultPayload:
        """Record an explicit judgment for one candidate assertion."""

        from polylogue.api.facade_client import submit_facade_writer
        from polylogue.surfaces.payloads import AssertionJudgmentResultPayload

        value = await submit_facade_writer(
            self.config,
            "judge_assertion_candidate",
            {
                "candidate_ref": candidate_ref,
                "decision": decision,
                "reason": reason,
                "actor_ref": actor_ref,
                "inject": inject,
                "replacement_kind": replacement_kind,
                "replacement_body_text": replacement_body_text,
                "replacement_value": replacement_value,
            },
        )
        return AssertionJudgmentResultPayload.model_validate(value)

    async def capture_assertion_candidate(
        self,
        *,
        body_text: str,
        kind: AssertionKind,
        refs: Sequence[str] = (),
        scope_refs: Sequence[str] = (),
        cwd: Path | None = None,
        author_ref: str = "user:local",
        author_kind: str = "user",
        idempotency_key: str | None = None,
        ttl_seconds: int | None = None,
    ) -> AssertionClaimPayload:
        """Capture a terminal assertion as a non-injected candidate for review.

        ``ttl_seconds`` stamps an expiry on the written row (polylogue-37t.1);
        the executor actuator preserves the existing user-tier admission and
        idempotency semantics.
        """

        from polylogue.surfaces.payloads import AssertionClaimPayload

        receipt, _plan = await submit_facade_product(
            self.config,
            "capture_assertion_candidate",
            body_text=body_text,
            kind=kind,
            refs=tuple(refs),
            scope_refs=tuple(scope_refs),
            cwd=cwd,
            author_ref=author_ref,
            author_kind=author_kind,
            idempotency_key=idempotency_key,
            ttl_seconds=ttl_seconds,
        )
        return AssertionClaimPayload.model_validate(receipt.domain_receipt["claim"])

    async def judge_assertion_candidates(
        self,
        *,
        items: Sequence[Any],
    ) -> AssertionBulkJudgmentPayload:
        """Apply a review batch with per-candidate partial-success outcomes."""

        from dataclasses import asdict, is_dataclass

        from polylogue.api.facade_client import submit_facade_operation
        from polylogue.surfaces.payloads import AssertionBulkJudgmentPayload

        reviews = [asdict(item) if is_dataclass(item) and not isinstance(item, type) else dict(item) for item in items]
        state = await submit_facade_operation(
            self.config,
            "mutation.judgment.record",
            {"judgment_kind": "assertion-review", "reviews": reviews},
        )
        return AssertionBulkJudgmentPayload.model_validate(state["result"])

    async def record_comparative_judgment(
        self,
        judgment: ComparativeJudgment,
        *,
        author_kind: str = "user",
    ) -> ArchiveAssertionEnvelope:
        """Persist one blind pairwise/n-wise comparative judgment (rxdo.9.6/.9.7/.9.11/.9.12).

        ``author_kind`` follows the existing promotion gate: a non-``"user"``
        author (an agent judge) is coerced to a non-injected ``CANDIDATE``
        row regardless of the caller's request, per the recursive-safety
        spine. Reuses the storage layer's fully-tested write chokepoint
        (:func:`~polylogue.storage.sqlite.archive_tiers.user_write.upsert_comparative_judgment_assertion`),
        which previously had no production caller.
        """
        from polylogue.api.facade_client import submit_facade_writer
        from polylogue.core.enums import AssertionVisibility
        from polylogue.operations.judgment_wire import comparative_judgment_wire_form
        from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionEnvelope

        value = await submit_facade_writer(
            self.config,
            "record_comparative_judgment",
            {"judgment": comparative_judgment_wire_form(judgment), "author_kind": author_kind},
        )
        return ArchiveAssertionEnvelope(
            **{
                **value,
                "kind": AssertionKind.from_string(value["kind"]),
                "status": AssertionStatus.from_string(value["status"]),
                "visibility": AssertionVisibility.from_string(value["visibility"]),
            }
        )

    async def list_comparative_judgments(self) -> list[ComparativeJudgment]:
        """Read back every live comparative-judgment assertion row."""
        return cast("list[ComparativeJudgment]", _archive_list_comparative_judgments(self.config))

    async def join_typed_annotations(
        self,
        *,
        schema_id: str,
        schema_version: int,
        statuses: Sequence[str | AssertionStatus],
        target_kind: str | None = None,
        group_by: Sequence[Literal["repo", "model", "time", "origin"]] = (),
        limit: int = 500,
        offset: int = 0,
    ) -> AnnotationStructuralJoinResult:
        """Join selected typed annotations to exact structural targets."""

        from polylogue.annotations.join_contracts import AnnotationJoinOperationResult, AnnotationStructuralJoinRequest
        from polylogue.operations.annotation_join import execute_annotation_join, open_annotation_join_read

        request = AnnotationStructuralJoinRequest(
            schema_id=schema_id,
            schema_version=schema_version,
            statuses=tuple(AssertionStatus.from_string(status) for status in statuses),
            target_kind=target_kind,
            group_by=tuple(group_by),
            limit=limit,
            offset=offset,
        )
        payload = request.model_dump(mode="json")

        def read_join(archive: ArchiveStore) -> dict[str, object]:
            return execute_annotation_join(payload, archive=archive, checkpoint=archive.check_operation_read)

        response = await run_archive_read(
            _active_archive_root(self.config),
            operation="annotation.join",
            arguments=payload,
            work=read_join,
            page_size=limit,
            offset=offset,
            projection="annotation-join",
            read_owner=lambda context: open_annotation_join_read(_active_archive_root(self.config), context),
        )
        return AnnotationJoinOperationResult.model_validate(response).result

    async def _compile_context_seed_query(
        self,
        spec: ContextSpec,
    ) -> tuple[list[str], dict[str, str], list[ContextOmission]]:
        """Resolve ContextSpec query/filter seed selection into session ids."""
        from polylogue.archive.context_models import ContextOmission
        from polylogue.context.selection import clamp_context_image_limit, select_context_image_sessions

        session_ids: list[str] = []
        message_anchor_by_session: dict[str, str] = {}
        omitted: list[ContextOmission] = []
        has_filters = any(
            (
                spec.seed_project_path,
                spec.seed_project_repo,
                spec.seed_since,
                spec.seed_until,
                spec.seed_origin,
            )
        )
        if has_filters or spec.seed_query == "":
            selection = await select_context_image_sessions(
                self.list_sessions_for_spec,
                clamp_context_image_limit,
                project_path=spec.seed_project_path,
                project_repo=spec.seed_project_repo,
                since=spec.seed_since,
                until=spec.seed_until,
                origin=spec.seed_origin,
                query=spec.seed_query or None,
                limit=spec.seed_query_limit,
            )
            if not selection.sessions:
                omitted.append(
                    ContextOmission(
                        query=spec.seed_query or None,
                        reason="not_found",
                        detail="seed selection matched no sessions",
                    )
                )
            else:
                session_ids.extend(str(summary.id) for summary in selection.sessions[: spec.seed_query_limit])
            return session_ids, message_anchor_by_session, omitted

        if spec.seed_query is None:
            return session_ids, message_anchor_by_session, omitted

        result = await self.search(spec.seed_query, limit=spec.seed_query_limit)
        if not result.hits:
            omitted.append(
                ContextOmission(
                    query=spec.seed_query,
                    reason="not_found",
                    detail="seed query matched no sessions",
                )
            )
        else:
            for hit in result.hits:
                session_ids.append(hit.session_id)
                if hit.session_id not in message_anchor_by_session and hit.message_id:
                    message_anchor_by_session[hit.session_id] = hit.message_id
        return session_ids, message_anchor_by_session, omitted

    async def record_manual_continuation(self, child_session_id: str, parent_session_id: str) -> None:
        """Commit a durable handoff parent and derive its spawned-fresh continuation."""
        from polylogue.api.facade_client import submit_facade_writer

        child = str(child_session_id).strip()
        parent = str(parent_session_id).strip()
        if not child or not parent or ":" not in child or ":" not in parent:
            raise ValueError("manual continuation requires origin-prefixed child and parent session ids")
        await submit_facade_writer(
            self.config,
            "record_manual_continuation",
            {"child_session_id": child, "parent_session_id": parent},
        )

    async def compile_context(self, spec: ContextSpec) -> ContextImage:
        """Compile an image and submit its disposable scheduler receipt."""
        from polylogue.api.facade_client import FacadeDaemonRequiredError, submit_facade_writer
        from polylogue.context.product_image import compile_context_image

        observed_at_ms = int(datetime.now(UTC).timestamp() * 1000)
        image = await compile_context_image(self, spec)
        with suppress(OSError, sqlite3.Error, DatabaseError, FacadeDaemonRequiredError):
            await submit_facade_writer(
                self.config,
                "context_ledger",
                {
                    "build_ref": image.build_ref,
                    "ledger_rows": [cast(ContextLedgerRecord, row).as_dict() for row in image.ledger],
                    "observed_at_ms": observed_at_ms,
                },
            )
        return image

    async def record_context_ledger(self, assembly: Any, *, observed_at_ms: int) -> None:
        """Submit preamble scheduling receipts through the archive writer."""
        from polylogue.api.facade_client import submit_facade_writer

        await submit_facade_writer(
            self.config,
            "context_ledger",
            {
                "build_ref": assembly.build_ref,
                "ledger_rows": [row.as_dict() for row in assembly.ledger],
                "observed_at_ms": observed_at_ms,
            },
        )

    def _context_temporal_window(self, summary: SessionSummary) -> TemporalEvidenceWindow:
        return _archive_context_temporal_window(self.config, summary)

    async def _context_chronicle_payload(self, summary: SessionSummary) -> ChronicleProjectionPayload:
        return await _archive_context_chronicle_payload(self.config, summary)

    async def list_read_view_profiles(self) -> list[JSONDocument]:
        """List executable read-view profile metadata."""
        from polylogue.archive.viewport import read_view_profile_payloads

        return list(read_view_profile_payloads())

    async def context_image_payload(
        self,
        *,
        project_path: str | None = None,
        project_repo: str | None = None,
        since: str | None = None,
        until: str | None = None,
        origin: str | None = None,
        query: str | None = None,
        max_sessions: int = 5,
        max_tokens: int | None = None,
        max_messages_per_session: int | None = DEFAULT_CONTEXT_IMAGE_MAX_MESSAGES_PER_SESSION,
        max_chars_per_message: int | None = DEFAULT_CONTEXT_IMAGE_MAX_CHARS_PER_MESSAGE,
        include_messages: bool = True,
        include_assertions: bool = True,
        redact_paths: bool = True,
        seed_session_id: str | None = None,
        segment_profile: Literal["default", "prose_with_refs"] = "default",
    ) -> ContextImage:
        """Compile a multi-session context image through ``compile_context``.

        This is a thin lens over the shared context engine, not a parallel
        assembler. Session selection runs through the query algebra (a seed ref
        or the context-image selection filters); compilation, token-budgeted
        accumulation, omission accounting, and assertion inclusion are all
        delegated to :meth:`compile_context`.
        """
        from polylogue.archive.context_models import ContextSpec
        from polylogue.surfaces.projection_spec import projection_from_views

        views: tuple[str, ...] = ("messages",) if include_messages else ()
        redaction: Literal["default", "raw-opt-in"] = "raw-opt-in" if not redact_paths else "default"
        limit = max(1, min(max_sessions, 20))

        seed_refs: tuple[str, ...] = (f"session:{seed_session_id}",) if seed_session_id is not None else ()

        spec = ContextSpec(
            purpose="handoff",
            seed_refs=seed_refs,
            seed_query=query if query is not None else ("" if not seed_refs else None),
            seed_query_limit=limit,
            seed_project_path=project_path,
            seed_project_repo=project_repo,
            seed_since=since,
            seed_until=until,
            seed_origin=origin,
            read_views=views,
            max_tokens=max_tokens,
            max_messages_per_session=max_messages_per_session,
            max_chars_per_message=max_chars_per_message,
            include_assertions=include_assertions,
            redaction_policy=redaction,
            segment_profile=segment_profile,
        )
        image = await self.compile_context(spec)
        projection_spec = projection_from_views(
            ("context-image",),
            format="json",
            destination="stdout",
            layout="context-image",
            max_tokens=max_tokens,
            query=query,
            origin=origin,
            since=since,
            until=until,
            project_path=project_path,
            project_repo=project_repo,
            limit=limit,
        )
        if seed_session_id is not None:
            selection = projection_spec.selection.model_copy(update={"refs": (f"session:{seed_session_id}",)})
            projection_spec = projection_spec.model_copy(update={"selection": selection})
        return image.model_copy(update={"projection_spec": projection_spec})

    async def context_preamble_payload(
        self,
        session_id: str,
        *,
        related_limit: int = 5,
        boundary: str = "session_start",
        token_budget: int | None = None,
        repo_path: str | None = None,
        cwd: str | None = None,
        recent_files: tuple[str, ...] = (),
        require_session: bool = True,
        source_tool_calls: dict[str, str] | None = None,
    ) -> Any:
        """Build a scheduler-admitted context preamble for one boundary."""
        from contextlib import suppress
        from datetime import datetime, timezone

        from polylogue.api.facade_client import FacadeDaemonRequiredError, submit_facade_writer
        from polylogue.context.preamble import _git_project_state
        from polylogue.operations.context_preamble import execute_context_preamble

        observed_at = datetime.now(timezone.utc)
        project_state = _git_project_state(cwd)
        result = await run_archive_read(
            _active_archive_root(self.config),
            operation="context.preamble",
            arguments={"session_id": session_id, "related_limit": related_limit, "boundary": boundary},
            work=lambda archive: execute_context_preamble(
                archive,
                session_id=session_id,
                related_limit=related_limit,
                repo_path=repo_path,
                cwd=cwd,
                recent_files=recent_files,
                source_tool_calls=source_tool_calls or {"context_preamble_payload": "polylogue-api"},
                require_session=require_session,
                boundary=boundary,
                token_budget=token_budget,
                observed_project_state=project_state,
                observed_at=observed_at,
            ),
        )
        if result.ledger is not None:
            with suppress(OSError, sqlite3.Error, DatabaseError, FacadeDaemonRequiredError):
                await submit_facade_writer(
                    self.config,
                    "context_ledger",
                    {
                        "build_ref": result.ledger.build_ref,
                        "ledger_rows": [row.as_dict() for row in result.ledger.ledger],
                        "observed_at_ms": result.observed_at_ms,
                    },
                )
        return result.payload

    async def explain_query_expression(self, expression: str) -> JSONDocument:
        """Explain query DSL parsing, AST metadata, and lowering details."""
        from polylogue.archive.query.expression import explain_expression

        return cast(JSONDocument, explain_expression(expression).to_payload())

    async def query_units(
        self,
        expression: str | None = None,
        *,
        limit: int | None = None,
        offset: int | None = None,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tag: str | None = None,
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo: str | None = None,
        repo_names: tuple[str, ...] = (),
        project: str | None = None,
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        title: str | None = None,
        since: str | None = None,
        until: str | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        typed_only: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        message_type: str | None = None,
        continuation: str | None = None,
    ) -> QueryUnitResultEnvelope:
        """Execute a terminal unit-source query."""
        from polylogue.archive.query.execution_control import classify_unit_expression_workload
        from polylogue.archive.query.transaction import (
            QueryContinuationInvalidError,
            QueryTransaction,
            decode_query_units_continuation,
            query_units_transaction_request,
        )
        from polylogue.archive.query.unit_results import query_unit_envelope, query_unit_request

        supplied_filters = any(
            (
                origin is not None,
                origins,
                excluded_origins,
                tag is not None,
                tags,
                excluded_tags,
                repo is not None,
                repo_names,
                project is not None,
                project_refs,
                has_types,
                tool_terms,
                excluded_tool_terms,
                action_terms,
                excluded_action_terms,
                action_sequence,
                action_text_terms,
                referenced_paths,
                cwd_prefix is not None,
                title is not None,
                since is not None,
                until is not None,
                has_tool_use,
                has_thinking,
                has_paste,
                typed_only,
                min_messages is not None,
                max_messages is not None,
                min_words is not None,
                max_words is not None,
                message_type is not None,
            )
        )
        continuation_request = None
        if continuation is not None:
            if expression is not None or limit is not None or offset is not None or supplied_filters:
                raise QueryContinuationInvalidError(
                    "continuation requests must not override the original query parameters"
                )
            continuation_request = decode_query_units_continuation(continuation).request
            continuation_arguments = continuation_request.arguments
            expression = str(continuation_arguments["expression"])
            session_filters = continuation_arguments["session_filters"]
            assert isinstance(session_filters, Mapping)
            effective_limit = continuation_request.page_size
            effective_offset = continuation_request.offset
        else:
            if expression is None or not expression.strip():
                raise QueryContinuationInvalidError("initial query requires a non-empty expression")
            session_filters = None
            effective_limit = max(1, limit if limit is not None else 50)
            effective_offset = max(0, offset if offset is not None else 0)
        assert expression is not None

        request = query_unit_request(
            expression=expression,
            limit=effective_limit,
            offset=effective_offset,
            session_filters=session_filters,
            origin=origin,
            origins=origins,
            tags=(tag,) if tag else tags,
            excluded_tags=excluded_tags,
            excluded_origins=excluded_origins,
            repo=repo,
            repo_names=repo_names,
            project=project,
            project_refs=project_refs,
            has_types=has_types,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            title=title,
            since=since,
            until=until,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            typed_only=typed_only,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            message_type=message_type,
        )
        archive_root = _active_archive_root(self.config)
        transaction = QueryTransaction(
            archive_root,
            continuation_request
            or query_units_transaction_request(
                expression=expression,
                session_filters=request.session_filters or {},
                page_size=effective_limit,
                offset=effective_offset,
            ),
            workload_class=classify_unit_expression_workload(expression),
        )
        return await transaction.run(
            lambda archive: query_unit_envelope(
                archive,
                request,
                execution_context=transaction.context,
                transaction_request=transaction.request,
            )
        )

    async def export_otel(
        self,
        *,
        source_ref: str,
        expressions: Sequence[str],
        limit: int = 50,
        include_message_text: bool = False,
    ) -> OtelProjectionPayload:
        """Project bounded query-unit evidence into an OTel-like JSON payload."""
        from polylogue.surfaces.payloads import MessageQueryRowPayload, QueryUnitEnvelope
        from polylogue.telemetry.otel_projection import OtelProjectionInputError, project_query_unit_rows_to_otel

        rows: list[Any] = []
        for expression in expressions:
            envelope = await self.query_units(expression, limit=limit)
            if not isinstance(envelope, QueryUnitEnvelope):
                raise ValueError("OTel export does not support aggregate query rows")
            if envelope.unit == "message" and envelope.projected_items:
                required_fields = {
                    name for name, field in MessageQueryRowPayload.model_fields.items() if field.is_required()
                }
                for item in envelope.projected_items:
                    missing_fields = required_fields - item.root.keys()
                    if missing_fields:
                        raise OtelProjectionInputError(unit="message", missing_fields=missing_fields)
                    rows.append(MessageQueryRowPayload.model_validate(item.root))
            elif envelope.items:
                rows.extend(envelope.items)
            elif envelope.projected_items:
                raise ValueError(f"OTel export does not support projected {envelope.unit} rows")
        return project_query_unit_rows_to_otel(
            source_ref,
            rows,
            include_message_text=include_message_text,
        )

    async def resolve_ref(
        self, ref: str, *, limit: int = 50, offset: int = 0, continuation: str | None = None
    ) -> PublicRefResolutionPayload:
        """Resolve one public object/evidence ref into a bounded read payload.

        The resolution itself is ``polylogue/operations/ref_resolution.py``,
        not this method: ``annotations import`` admits or rejects every durable
        ``user.db`` candidate row on this answer, and a daemon handler cannot
        import a surface, so a second implementation would decide admission
        differently with no failure at the point of divergence
        (polylogue-j5u2b).  This binding frames the plan's declared read as a
        transaction; the daemon frames the same plan against its pinned reader.
        """
        from polylogue.operations.ref_resolution import plan_ref_resolution

        archive_root = _active_archive_root(self.config)
        plan = plan_ref_resolution(
            ref, archive_root=archive_root, limit=limit, offset=offset, continuation=continuation
        )
        if plan.payload is not None:
            return plan.payload
        assert plan.read is not None
        return await run_archive_read(
            archive_root,
            operation=plan.operation,
            arguments=plan.arguments,
            work=plan.read,
            projection=plan.projection,
            stable_order=plan.stable_order,
        )

    async def query_completions(
        self,
        kind: str,
        *,
        incomplete: str = "",
        unit: str | None = None,
        field: str | None = None,
    ) -> JSONDocument:
        """Return shared query/action completion metadata for adapters."""
        from polylogue.archive.query.completions import query_completion_payload

        return cast(
            JSONDocument,
            query_completion_payload(kind, incomplete=incomplete, unit=unit, field=field),
        )

    async def get_sessions(
        self,
        session_ids: list[str],
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> list[Session]:
        rows: list[Session] = []
        for session_id in session_ids:
            row = await self.get_session(session_id, content_projection=content_projection)
            if row is not None:
                rows.append(row)
        return rows

    async def get_actions_batch(
        self,
        session_ids: builtins.list[str],
    ) -> dict[str, tuple[Action, ...]]:
        """Derive actions for a batch of sessions from their content blocks.

        ``index.db`` exposes an ``actions`` view; these actions are derived on
        read from each session's tool-use/tool-result blocks — the same
        source the archive materializer hashed into durable rows. Missing
        sessions are omitted from the result mapping, mirroring the archive
        repository batch reader.
        """
        sessions = await self.get_sessions(session_ids)
        return {str(session.id): _actions_for_session(session) for session in sessions}

    async def list_sessions(
        self,
        origin: str | None = None,
        limit: int | None = None,
        content_projection: ContentProjectionSpec | None = None,
    ) -> list[Session]:
        def read(archive: ArchiveStore) -> list[Session]:
            summaries = archive.list_summaries(
                origin=origin,
                limit=DEFAULT_SESSION_LIST_LIMIT if limit is None else limit,
            )
            sessions = [
                archive_envelope_to_session(
                    archive.read_session(summary.session_id),
                    display_label=summary.display_label,
                    display_label_source=summary.display_label_source,
                )
                for summary in summaries
            ]
            if content_projection is None or not content_projection.filters_content():
                return sessions
            return [session.with_content_projection(content_projection) for session in sessions]

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.sessions.list",
            arguments={"origin": origin, "content_projection": content_projection},
            work=read,
            page_size=limit,
            projection="session",
            stable_order="date,session_id",
        )

    async def list_summaries(
        self,
        *,
        limit: int | None = DEFAULT_SESSION_LIST_LIMIT,
        offset: int = 0,
        origin: str | None = None,
    ) -> builtins.list[SessionSummary]:
        """List archive session summaries without hydrating full sessions.

        The cheap read path for callers that only need summary fields
        (title, timestamps, origin, model, counts). Use
        :meth:`list_sessions` when full message bodies are required.
        """
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.summaries.list",
            arguments={"origin": origin},
            work=lambda archive: [
                archive_summary_to_domain(summary)
                for summary in archive.list_summaries(
                    origin=origin,
                    limit=DEFAULT_SESSION_LIST_LIMIT if limit is None else limit,
                    offset=offset,
                )
            ],
            page_size=limit,
            offset=offset,
            projection="session-summary",
            stable_order="date,session_id",
        )

    async def list_sessions_for_spec(
        self,
        spec: SessionQuerySpec,
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> list[Session]:
        """Run a ``SessionQuerySpec`` directly, returning full sessions.

        The spec-based counterpart of :meth:`list_sessions` (which only
        takes origin/limit). A vector provider is resolved only for explicit
        semantic (``similar_text``) or ``hybrid`` specs; plain filter specs run
        without touching the embeddings tier.
        """
        vector_provider = None
        if spec.similar_text or spec.retrieval_lane == "hybrid":
            from polylogue.storage.search_providers import create_vector_provider

            archive_root = _active_archive_root(self.config)
            vector_provider = create_vector_provider(
                self.config, db_path=archive_root / "embeddings.db", archive_root=archive_root
            )
        sessions = await spec.list(self.config, vector_provider=vector_provider)
        if content_projection is None or not content_projection.filters_content():
            return sessions
        return [session.with_content_projection(content_projection) for session in sessions]

    async def search_session_hits(self, spec: SessionQuerySpec) -> SearchHitResults:
        """Return archive FTS/hybrid search-hit projections for a query spec.

        The hit projection carries match snippets and ranking metadata the
        :class:`SearchEnvelope` builder needs, distinct from the full
        session hydration of :meth:`list_sessions_for_spec`.
        """
        from polylogue.archive.query.search_hits import search_hits_for_plan

        # The executor resolves construction failures into its typed lane
        # outcome; keep facade setup outside that failure-classifying path.
        return await search_hits_for_plan(spec.to_plan(), self.config)

    async def diagnose_query_miss(self, spec: SessionQuerySpec, *, full: bool = False) -> QueryMissDiagnostics:
        """Best-effort explanation for an empty archive query result.

        The diagnostic is duck-typed over this facade: it reads whatever
        archive count/stats methods are available and degrades gracefully when
        a probe is absent. ``full=True`` requests the complete breakdown
        (clause-drop attribution plus since/until relaxation and
        FTS-vs-structured disagreement probes, polylogue-jnj.12); the default
        still attributes which predicate(s) zeroed the result but skips those
        extra probes.
        """
        from polylogue.archive.query.miss_diagnostics import diagnose_query_miss

        return await diagnose_query_miss(self, spec, config=self.config, full=full)

    async def storage_counts(self) -> dict[str, int]:
        """Read canonical session and message counts without full statistics."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.storage_counts",
            arguments={},
            work=lambda archive: archive.counts(),
            projection="counts",
        )

    async def query_capability_readiness(self) -> dict[str, object]:
        """Read standing query-binding metadata and aggregate-only debt evidence.

        This certifies only that measured scope. Missing binding evidence stays
        unknown instead of triggering an archive-wide inspection.
        """
        from polylogue.daemon.convergence_debt_status import convergence_debt_stage_counts_info
        from polylogue.operations.fts_derivation import bound_archive_fts_surface
        from polylogue.readiness.capability import component_from_query_binding

        root = _active_archive_root(self.config)
        binding = await run_archive_read(
            root,
            operation="archive.query_capability_readiness",
            arguments={},
            work=lambda archive: bound_archive_fts_surface(archive._conn),
            projection="query-binding",
        )
        debt = convergence_debt_stage_counts_info(root / "index.db", ops_db=root / "ops.db")
        return dict(
            component_from_query_binding(
                binding,
                debt_available=debt.available,
                debt_count=sum(count for _, _, count in debt.counts) if debt.available else None,
            ).to_dict()
        )

    async def storage_stats(self) -> StorageArchiveStats:
        """Lightweight archive stats without recent-session hydration.

        The cheap counterpart of :meth:`stats`: counts and provider/tag
        breakdowns straight from ``index.db`` for status surfaces.
        """
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.storage_stats",
            arguments={},
            work=lambda archive: archive.stats(),
            projection="stats",
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = 100,
        source: str | None = None,
        since: str | None = None,
    ) -> SearchResult:
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.search",
            arguments={"query": query, "source": source, "since": since},
            work=lambda archive: SearchResult(
                hits=tuple(
                    _archive_search_hit_to_domain(hit)
                    for hit in archive.search_summaries(
                        query,
                        limit=limit,
                        origin=source,
                        since_ms=_archive_query_date_ms("since", since),
                    )
                )
            ),
            page_size=limit,
            projection="search-hits",
            stable_order="rank,session_id,message_id",
        )

    async def search_envelope(
        self,
        query: str,
        *,
        limit: int = 50,
        offset: int = 0,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
        retrieval_lane: str = "auto",
        sort: str | None = None,
        cursor: str | None = None,
    ) -> SearchEnvelope:
        """Return the canonical :class:`SearchEnvelope` for a query (#1266).

        Pass ``cursor`` (an opaque token previously returned as
        :attr:`SearchEnvelope.next_cursor`) to fetch the next page
        without losing or duplicating hits even when the archive grew
        between requests (#1268).
        """
        from polylogue.api.search_envelope_builder import build_search_envelope_for_spec
        from polylogue.archive.query.expression import compile_expression_into
        from polylogue.archive.query.spec import SessionQuerySpec

        base_spec = SessionQuerySpec.from_params(
            {
                "query": "",
                "origin": origin,
                "since": since,
                "until": until,
                "retrieval_lane": retrieval_lane,
                "sort": sort,
                "limit": limit,
                "offset": offset,
                "cursor": cursor,
            },
            strict=True,
        )
        spec = compile_expression_into(query, base_spec) if query.strip() else base_spec
        return await build_search_envelope_for_spec(
            cast("Polylogue", self), spec, limit=limit, offset=offset, query=query
        )

    async def archive_count_sessions(
        self,
        *,
        origin: str | None = None,
        excluded_origins: Sequence[str] = (),
        tags: Sequence[str] = (),
        excluded_tags: Sequence[str] = (),
        repo_names: Sequence[str] = (),
        project_refs: Sequence[str] = (),
        has_types: Sequence[str] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: Sequence[str] = (),
        excluded_tool_terms: Sequence[str] = (),
        action_terms: Sequence[str] = (),
        excluded_action_terms: Sequence[str] = (),
        action_sequence: Sequence[str] = (),
        action_text_terms: Sequence[str] = (),
        referenced_paths: Sequence[str] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> int:
        """Count sessions in the index tier."""
        from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest

        arguments = {
            "origin": origin,
            "excluded_origins": tuple(excluded_origins),
            "tags": tuple(tags),
            "excluded_tags": tuple(excluded_tags),
            "repo_names": tuple(repo_names),
            "project_refs": tuple(project_refs),
            "has_types": tuple(has_types),
            "has_tool_use": has_tool_use,
            "has_thinking": has_thinking,
            "has_paste": has_paste,
            "tool_terms": tuple(tool_terms),
            "excluded_tool_terms": tuple(excluded_tool_terms),
            "action_terms": tuple(action_terms),
            "excluded_action_terms": tuple(excluded_action_terms),
            "action_sequence": tuple(action_sequence),
            "action_text_terms": tuple(action_text_terms),
            "referenced_paths": tuple(referenced_paths),
            "cwd_prefix": cwd_prefix,
            "typed_only": typed_only,
            "message_type": message_type,
            "title": title,
            "min_messages": min_messages,
            "max_messages": max_messages,
            "min_words": min_words,
            "max_words": max_words,
            "since": since,
            "until": until,
        }
        transaction = QueryTransaction(
            _active_archive_root(self.config),
            QueryTransactionRequest(
                operation="archive_count_sessions",
                arguments=arguments,
                page_size=1,
                projection="count",
                stable_order="canonical",
            ),
        )
        return await transaction.run(
            lambda archive: archive.count_sessions(
                origin=origin,
                excluded_origins=tuple(excluded_origins),
                tags=tuple(tags),
                excluded_tags=tuple(excluded_tags),
                repo_names=tuple(repo_names),
                project_refs=tuple(project_refs),
                has_types=tuple(has_types),
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                has_paste=has_paste,
                tool_terms=tuple(tool_terms),
                excluded_tool_terms=tuple(excluded_tool_terms),
                action_terms=_archive_action_terms("action", action_terms),
                excluded_action_terms=_archive_action_terms("exclude_action", excluded_action_terms),
                action_sequence=_archive_action_sequence(action_sequence),
                action_text_terms=tuple(action_text_terms),
                referenced_paths=tuple(referenced_paths),
                cwd_prefix=cwd_prefix,
                typed_only=typed_only,
                message_type=_archive_message_type(message_type),
                title=title,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                max_words=max_words,
                since_ms=_archive_query_date_ms("since", since),
                until_ms=_archive_query_date_ms("until", until),
            )
        )

    async def archive_get_session(self, session_id: str) -> ArchiveSessionEnvelope | None:
        """Read a full session envelope by exact id or prefix."""
        from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest

        transaction = QueryTransaction(
            _active_archive_root(self.config),
            QueryTransactionRequest(
                operation="archive_get_session",
                arguments={"session_id": session_id},
                page_size=1,
                projection="session-envelope",
                stable_order="session,message,block",
            ),
        )

        def read(archive: ArchiveStore) -> ArchiveSessionEnvelope | None:
            try:
                resolved_id = archive.resolve_session_id(session_id)
            except KeyError:
                return None
            return archive.read_session(resolved_id)

        return await transaction.run(read)

    async def get_session_insight_status(self) -> SessionInsightStatusSnapshot:
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.session_status",
            arguments={},
            work=lambda archive: archive.session_insight_status(),
            projection="insight-status",
        )

    async def get_session_profile_insight(
        self,
        session_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileInsight | None:
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.session_profile.get",
            arguments={"session_id": session_id, "tier": tier},
            work=lambda archive: archive.get_session_profile_insight(session_id, tier=tier),
            projection="session-profile",
        )

    async def get_session_profile_record(
        self,
        session_id: str,
    ) -> SessionProfileRecord | None:
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.session_profile_record.get",
            arguments={"session_id": session_id},
            work=lambda archive: archive.get_session_profile_record(session_id),
            projection="session-profile-record",
        )

    async def list_session_profile_insights(
        self,
        query: SessionProfileInsightQuery | None = None,
    ) -> list[SessionProfileInsight]:
        request = query or SessionProfileInsightQuery()
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.session_profile.list",
            arguments={
                "origin": request.origin,
                "workflow_shape": request.workflow_shape,
                "terminal_state": request.terminal_state,
                "tag": request.tag,
                "repo": request.repo,
                "since": request.since,
                "until": request.until,
                "first_message_since": request.first_message_since,
                "first_message_until": request.first_message_until,
                "session_date_since": request.session_date_since,
                "session_date_until": request.session_date_until,
                "tier": request.tier,
                "query": request.query,
            },
            work=lambda archive: cast(list[SessionProfileInsight], read_insight_page(archive, request)),
            page_size=request.limit,
            offset=request.offset,
            projection="session-profile",
            stable_order="time,session_id",
        )

    def filter(self) -> SessionFilter:
        from polylogue.archive.filter.filters import SessionFilter

        archive_root = _active_archive_root(self.config)
        return SessionFilter(
            archive_root=archive_root,
            config=self.config,
        )

    async def origin_usage_report(
        self,
        *,
        origin: str | None = None,
        limit: int | None = 25,
        detail: str = "full",
    ) -> ProviderUsageReport:
        """Return provider usage accounting diagnostics for the active archive."""
        from polylogue.storage.usage import origin_usage_report_from_connection

        if detail not in {"headline", "full"}:
            raise ValueError("detail must be 'headline' or 'full'")
        usage_detail = cast(Literal["headline", "full"], detail)
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.provider_usage",
            arguments={"origin": origin, "limit": limit, "detail": detail},
            work=lambda archive: origin_usage_report_from_connection(
                archive._conn,
                archive_root=_active_archive_root(self.config),
                origin=origin,
                limit=limit,
                detail=usage_detail,
            ),
            page_size=limit,
            projection="provider-usage",
            workload_class="scan",
        )

    async def session_usage_reconciliation(self, session_id: str) -> SessionUsageReconciliation:
        """Return the fast, session-scoped usage/cost reconciliation for one session.

        Reads only ``session_id``-indexed rows (``session_model_usage``,
        ``session_profiles`` PK lookup) instead of the archive-wide
        ``origin_usage_report`` audit, so it stays cheap regardless of
        archive size (polylogue-zumd).
        """
        from polylogue.storage.usage import session_usage_reconciliation_for_connection

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.session_usage_reconciliation",
            arguments={"session_id": session_id},
            work=lambda archive: session_usage_reconciliation_for_connection(
                archive._conn,
                session_id=session_id,
            ),
            projection="session-usage-reconciliation",
        )

    async def stats(self) -> ArchiveStats:
        from polylogue.operations import ArchiveStats as PublicArchiveStats

        def read(archive: ArchiveStore) -> PublicArchiveStats:
            stats = archive.stats()
            word_row = archive._conn.execute("SELECT COALESCE(SUM(word_count), 0) FROM sessions").fetchone()
            recent = [
                archive_envelope_to_session(
                    archive.read_session(summary.session_id),
                    display_label=summary.display_label,
                    display_label_source=summary.display_label_source,
                )
                for summary in archive.list_summaries(limit=5)
            ]
            return PublicArchiveStats(
                session_count=stats.total_sessions,
                message_count=stats.total_messages,
                word_count=int(word_row[0] or 0) if word_row is not None else 0,
                origins=stats.origins,
                tags={},
                last_sync=None,
                recent=recent,
            )

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.stats",
            arguments={},
            work=read,
            projection="stats",
        )

    async def facets(
        self,
        spec: SessionQuerySpec | None = None,
        *,
        include_idf: bool = True,
        include_deferred: bool = True,
    ) -> FacetsResponse:
        """Compute scoped + global facet aggregates over the archive.

        When ``spec`` carries any active filter, the scoped buckets are
        rolled from that filter's summary list and ``scoped_to_query``
        becomes true.  The global buckets always reflect the
        unfiltered archive.  Surfaces (daemon HTTP, MCP, CLI) call into
        this method so the scope vocabulary stays in one place
        (#1269 / slice D of #873).
        """
        import time

        from polylogue.archive.query.facets import (
            FacetBuckets as _FacetBuckets,
        )
        from polylogue.surfaces.payloads import (
            FacetBucketsPayload,
        )

        def _payload(b: _FacetBuckets) -> FacetBucketsPayload:
            return FacetBucketsPayload(
                origins=dict(b.origins),
                tags=dict(b.tags),
                repos=dict(b.repos),
                role_counts=dict(b.role_counts),
                material_origins=dict(b.material_origins),
                message_types=dict(b.message_types),
                action_types=dict(b.action_types),
                has_flags=dict(b.has_flags),
                omitted=dict(b.omitted),
                total_sessions=b.total_sessions,
                total_messages=b.total_messages,
            )

        def _family_status_payload(family: str, *, state: str, reason: str | None = None) -> dict[str, object]:
            return {
                "state": state,
                "reason": reason,
                "stale": False,
                **_FACET_FAMILY_METADATA.get(family, {}),
            }

        scoped_to_query = spec is not None and spec.has_filters()
        started_at = time.perf_counter()

        def _facet_work(archive: Any) -> tuple[Any, Any, list[str]]:
            scope_gaps: list[str] = []
            global_b = _archive_facet_buckets(archive, None, include_deferred=include_deferred, scope_gaps=scope_gaps)
            if not scoped_to_query:
                return global_b, global_b, scope_gaps
            scoped_b = _archive_facet_buckets(archive, spec, include_deferred=include_deferred, scope_gaps=scope_gaps)
            return global_b, scoped_b, scope_gaps

        global_buckets, scoped_buckets, scope_gaps = await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.facets",
            arguments={"spec": spec, "include_deferred": include_deferred},
            work=_facet_work,
            projection="facets",
            workload_class="scan",
        )
        return build_facets_response(
            global_buckets=global_buckets,
            scoped_buckets=scoped_buckets,
            scoped_to_query=scoped_to_query,
            include_deferred=include_deferred,
            elapsed_s=time.perf_counter() - started_at,
            include_idf=include_idf,
            scope_gaps=scope_gaps,
        )

    async def health_check(self) -> ReadinessReport:
        """Return the canonical archive readiness report."""
        return _archive_health_report(self.config)

    async def rebuild_insights(
        self,
        session_ids: Sequence[str] | None = None,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> SessionInsightCounts:
        """Refuse in-process insight maintenance and name its sealed owner.

        This method is deliberately not an executor route.  Session-insight
        maintenance is authorized as a sealed, page-bounded machine: the scope
        is frozen into a manifest, staged as immutable preview pages, sealed
        into accepted parts, and started one ordinal at a time through
        ``OperationExecutor.begin_accepted_insight_part``.  The sealed sequence
        needs durable audit authority, a pinned index generation and recipe
        binding, and the resident session-profile publication owner — a library
        process has none of them, and ``polylogued run`` is the required
        live-write owner for exactly this reason.

        The generic ``prepare -> authorize -> execute_bound`` path this method
        used to take is refused at two independent points in
        ``operations/mutation_transaction.py`` (``begin_bound`` and
        ``execute``), because consuming an unsealed staging page as execution
        authority would let ``session_ids=None`` become an accepted full sweep
        after the fact.  Raising a typed, actionable error here replaces an
        internal ``MutationTransactionError`` leaking to public callers.

        The sanctioned route is the daemon operation
        ``maintenance.insights.rebuild``
        (``operations/daemon_insights.py``).  ``session_ids`` and
        ``progress_callback`` are retained so existing call sites still type
        check; both are the daemon operation's to honour, not this method's.
        """
        from polylogue.core.errors import InsightMaintenanceRequiresDaemonError

        del session_ids, progress_callback
        raise InsightMaintenanceRequiresDaemonError

    async def resume_brief(
        self,
        session_id: str,
        *,
        related_limit: int = 6,
        repo_path: str | None = None,
        recent_files: Sequence[str] = (),
    ) -> ResumeBrief | None:
        """Build a compact handoff brief for an archived session."""
        from polylogue.analysis.resume import ResumeOperations, build_resume_brief

        return await build_resume_brief(
            cast(ResumeOperations, self),
            session_id,
            related_limit=related_limit,
            repo_path=repo_path,
            recent_files=recent_files,
        )

    async def find_resume_candidates(
        self, *, repo_path: str, cwd: str | None = None, recent_files: Sequence[str] = (), limit: int = 10
    ) -> tuple[ResumeCandidate, ...]:
        from polylogue.analysis.resume import ResumeOperations, find_resume_candidates

        return await find_resume_candidates(
            cast(ResumeOperations, self),
            repo_path=repo_path,
            cwd=cwd,
            recent_files=recent_files,
            limit=limit,
        )

    async def insight_readiness_report(
        self,
        query: InsightReadinessQuery | None = None,
    ) -> InsightReadinessReport:
        """Return insight materialization readiness for downstream consumers."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.readiness",
            arguments={"query": query},
            work=lambda archive: archive.insight_readiness_report(query),
            projection="insight-readiness",
        )

    async def archive_debt(
        self,
        *,
        kinds: Iterable[str] | None = None,
        only_actionable: bool = False,
        limit: int | None = None,
        exact_fts: bool = False,
    ) -> ArchiveDebtListPayload:
        """Return the unified archive debt payload used by CLI, MCP, and daemon surfaces."""
        from polylogue.operations.archive_debt import archive_debt_list

        return archive_debt_list(
            archive_root=_active_archive_root(self.config),
            kinds=kinds,
            only_actionable=only_actionable,
            limit=limit,
            exact_fts=exact_fts,
        )

    async def insight_rigor_audit(
        self,
        query: InsightRigorAuditQuery | None = None,
    ) -> InsightRigorAuditReport:
        """Per-product rigor profile across materialized insights (#1275)."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.rigor_audit",
            arguments={"query": query},
            work=lambda archive: archive.audit_insight_rigor(query),
            projection="insight-rigor",
            workload_class="scan",
        )

    async def regenerate_private_fable_packet(
        self,
        *,
        seed: str,
        requested_size: int,
        schema_id: str = "delegation.discourse",
        schema_version: int = 1,
        exact_template_cap: int = 1,
    ) -> FableDelegationPacket:
        """Cold-regenerate the private descriptive Fable packet from the archive."""
        from polylogue.analysis.fable_packet import regenerate_private_fable_packet

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.fable_packet.regenerate",
            arguments={
                "seed": seed,
                "requested_size": requested_size,
                "schema_id": schema_id,
                "schema_version": schema_version,
                "exact_template_cap": exact_template_cap,
            },
            work=lambda archive: regenerate_private_fable_packet(
                archive,
                seed=seed,
                requested_size=requested_size,
                schema_id=schema_id,
                schema_version=schema_version,
                exact_template_cap=exact_template_cap,
            ),
            projection="fable-delegation-packet",
            workload_class="scan",
        )

    async def get_messages_paginated(
        self,
        session_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        material_origin: tuple[MaterialOrigin, ...] = (),
        limit: int = 50,
        offset: int = 0,
        content_projection: ContentProjectionSpec | None = None,
    ) -> tuple[list[Message], int, LineageCompleteness]:
        """Return paginated ``Message`` objects for a session.

        Raises ``SessionNotFoundError`` if the session does not exist. The
        third element reports whether the composed transcript is the full
        logical transcript or was truncated by a dangling branch point,
        cycle, or depth-limited composition (polylogue-ppkj) -- the same
        read-time signal the MCP surface already carries.
        """
        from polylogue.operations.transcript_window import message_transcript_window, window_request

        window = await message_transcript_window(
            self,
            window_request(
                session_id,
                limit=limit,
                offset=offset,
                filters={
                    "message_role": tuple(message_role),
                    "message_type": message_type,
                    "material_origin": tuple(material_origin),
                },
            ),
            content_projection=content_projection,
        )
        messages = list(window.rows)
        total = window.total
        completeness = LineageCompleteness(
            complete=window.lineage_complete,
            truncation_reason=cast(
                "LineageTruncationReason | None",
                window.lineage_truncation_reason,
            ),
        )
        # message_transcript_window already applied content_projection before
        # pagination; projecting again would reclassify projected code as prose.
        return messages, total, completeness

    async def read_transcript_window(
        self,
        session_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        material_origin: tuple[MaterialOrigin, ...] = (),
        limit: int = 50,
        offset: int = 0,
        continuation: str | None = None,
        around: str | None = None,
    ) -> TranscriptWindow[Message]:
        """Read one snapshot-bound transcript window (polylogue-ijbwq).

        This is the Python API's entry to the single execution route every
        public surface shares. ``get_messages_paginated`` is the storage read
        inside it and answers rows only; this method additionally binds the
        archive snapshot, validates a resumed continuation's epoch and mints
        the next continuation, so a write landing between two pages is refused
        as stale here exactly as it is on the CLI, MCP and HTTP.

        ``around`` names a message whose window is wanted instead of a
        coordinate naming it (polylogue-idrej). It is resolved to an offset
        through the shared locator on the same pinned reader as the window.
        The continuation it mints is identical to asking for the
        offset this call reports back in ``TranscriptWindow.offset``.
        """

        from polylogue.operations.transcript_window import message_transcript_window, window_request

        if around is not None:
            if continuation is not None or offset:
                raise ValueError("around and an explicit window coordinate name two different windows")
            if message_role or message_type is not None or material_origin:
                raise ValueError("around cannot be combined with transcript filters")

        return await message_transcript_window(
            self,
            window_request(
                session_id,
                limit=limit,
                offset=offset,
                continuation=continuation,
                filters={
                    "message_role": tuple(message_role),
                    "message_type": message_type,
                    "material_origin": tuple(material_origin),
                },
            ),
            around=around,
        )

    def iter_messages(
        self,
        session_id: str,
        *,
        message_roles: MessageRoleFilter = (),
        material_origin: tuple[MaterialOrigin, ...] = (),
        limit: int | None = None,
    ) -> AsyncGenerator[Message, None]:
        async def _iter() -> AsyncGenerator[Message, None]:
            async with aclosing(
                self.repository.iter_messages(
                    session_id,
                    message_roles=message_roles,
                    material_origin=material_origin,
                    limit=limit,
                )
            ) as messages:
                async for message in messages:
                    yield message

        return _iter()

    async def bulk_get_messages(
        self,
        session_ids: Sequence[str],
        *,
        since: str | None = None,
        until: str | None = None,
        message_role: MessageRoleFilter = (),
        material_origin: tuple[MaterialOrigin, ...] = (),
        content_projection: ContentProjectionSpec | None = None,
    ) -> dict[str, list[Message]]:
        """Return messages for many sessions using one archive batch read."""
        since_ms = _archive_query_date_ms("since", since)
        until_ms = _archive_query_date_ms("until", until)
        rows: dict[str, list[Message]] = {}
        for session_id in session_ids:
            session = await self.get_session(session_id, content_projection=content_projection)
            if session is None:
                continue
            rows[str(session.id)] = [
                message
                for message in session.messages
                if _archive_message_matches(
                    message,
                    message_role=message_role,
                    message_type=None,
                    material_origin=material_origin,
                    since_ms=since_ms,
                    until_ms=until_ms,
                )
            ]
        return rows

    async def get_raw_artifacts_for_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, object]], int]:
        """Return paginated raw archive artifact rows for a session.

        Delegates to the archive layer rather than accessing
        the private ``_backend`` connection directly.
        """
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.raw_artifacts.list",
            arguments={"session_id": session_id},
            work=lambda archive: archive.raw_artifacts_for_session(session_id, limit=limit, offset=offset),
            page_size=limit,
            offset=offset,
            projection="raw-artifacts",
            stable_order="artifact_id",
        )

    async def get_hook_event_summary_for_session(self, session_id: str) -> dict[str, object] | None:
        """Return the per-event-type hook-event summary for a session.

        Delegates to the archive layer rather than accessing
        the private ``_backend`` connection directly.
        """
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.hook_events.summary",
            arguments={"session_id": session_id},
            work=lambda archive: archive.hook_event_summary_for_session(session_id),
            page_size=1,
            projection="hook-event-summary",
            stable_order="session_id",
        )

    async def get_effective_context(
        self,
        session_id: str,
        *,
        at_position: int | None = None,
    ) -> list[dict[str, object]] | None:
        """Return the provider-effective context at a message position."""

        def resolve_existing(archive: ArchiveStore) -> str | None:
            try:
                resolved_id = archive.resolve_session_id(session_id)
                archive.read_summary(resolved_id)
                return resolved_id
            except KeyError:
                return None

        target = await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.session.exists",
            arguments={"session_id": session_id},
            work=resolve_existing,
            projection="session-id",
        )
        if target is None:
            return None
        messages = await self.repository.get_effective_context(target, at_position)
        return [message.model_dump(mode="json", exclude_none=True) for message in messages]

    async def get_session_orchestration(self, session_id: str) -> SessionOrchestrationEvidence | None:
        """Return versioned orchestration evidence from the archive's stored records.

        The session's own messages, events and usage rows are streamed page by
        page into the bounded projection; the full session is never hydrated.
        """
        from polylogue.operations.orchestration import read_session_orchestration

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.orchestration.evidence",
            arguments={"session_id": session_id},
            work=lambda archive: read_session_orchestration(archive, session_id),
            projection="orchestration-evidence",
            stable_order="position",
        )

    async def get_session_events(
        self,
        session_id: str,
        *,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]] | None:
        """Return the raw session-timeline events for one session, newest last.

        ``Session.session_events`` is the substrate's schema-free evidence
        axis for artifacts that ride the session timeline rather than a
        dialogue message -- ``event_type`` has no CHECK vocabulary by design
        (docs/internals.md), so new provider evidence needs no schema bump to
        start landing here. Before this reader, nothing on any surface could
        reach it: it was populated on every full session read (for content
        hashing and a couple of insight materializers) but never rendered or
        queried. This is the generic reader that makes it visible -- Codex
        ``world_state``/``agent_policy``/``turn_context`` policy facts
        (sandbox/truncation policy, rate limits, ghost commits, developer
        instructions), every Claude Code sidecar event (``claude_todo_state``,
        ``claude_tool_result_sidecar``, ``claude_attachment_file``, ``micro_compaction``,
        etc.), Hermes ``hermes_tool_availability_span``/``hermes_*`` step
        telemetry, and ``claude_ai_conversation_summary`` all surface through
        the same one reader instead of a bespoke accessor per event type.

        Returns ``None`` when the session does not exist (distinct from an
        empty list, which means the session exists but has no matching
        events). ``event_type`` narrows to an exact match; an unrecognized
        value yields an empty list rather than an error, matching the
        substrate's schema-free vocabulary.
        """
        resolved = await self.repository.resolve_id(session_id)
        session = await self.repository.get(str(resolved) if resolved is not None else session_id)
        if session is None:
            return None
        events = session.session_events
        if event_type is not None:
            events = tuple(event for event in events if event.event_type == event_type)
        if limit is not None:
            events = events[:limit]
        return [
            {
                "event_id": str(event.id),
                "event_index": event.event_index,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat() if event.timestamp is not None else None,
                "payload": event.payload,
            }
            for event in events
        ]

    async def record_work_event(
        self,
        session_id: str,
        *,
        event_id: str,
        event_type: str,
        summary: str,
        payload: dict[str, object] | None = None,
        timestamp: str | None = None,
    ) -> dict[str, object]:
        """Record one typed live-agent event through the archive writer."""
        from polylogue.api.facade_client import submit_facade_writer

        value = await submit_facade_writer(
            self.config,
            "record_work_event",
            {
                "session_id": session_id,
                "event_id": event_id,
                "event_type": event_type,
                "summary": summary,
                "payload": dict(payload or {}),
                "timestamp": timestamp,
            },
        )
        return cast("dict[str, object]", value)

    async def emit_decision(
        self,
        session_id: str,
        *,
        event_id: str,
        decision: str,
        summary: str,
        evidence_refs: tuple[str, ...] = (),
        timestamp: str | None = None,
    ) -> dict[str, object]:
        """Record a decision as the shared typed work-event kind."""
        return await self.record_work_event(
            session_id,
            event_id=event_id,
            event_type="decision",
            summary=summary,
            payload={"decision": decision, "evidence_refs": list(evidence_refs)},
            timestamp=timestamp,
        )

    async def get_file_edits(self, session_id: str) -> list[dict[str, object]] | None:
        """Return file-edit tool-call evidence (structuredPatch/originalFile/...) for one session.

        polylogue-nua7: the writer materializes ``ParsedFileEdit`` evidence
        (Claude Code Edit/Write/MultiEdit tool calls -- structured unified
        diffs, pre-edit file content, old/new string pairs) into the
        dedicated ``file_edits`` index table on every ingest
        (``storage/repository/archive/sessions.py::get_file_edits``), but
        before this reader nothing above the storage layer could reach it.
        This is the read surface: what a "what did this session change"
        report needs instead of re-deriving edits from tool-call prose.

        Returns ``None`` when the session does not exist (distinct from an
        empty list, meaning the session exists but made no captured edits).
        """
        resolved = await self.repository.resolve_id(session_id)
        resolved_id = str(resolved) if resolved is not None else session_id
        session = await self.repository.get(resolved_id)
        if session is None:
            return None
        edits = await self.repository.get_file_edits(resolved_id)
        return [
            {
                "tool_use_block_id": edit.tool_use_block_id,
                "message_id": str(edit.message_id),
                "file_path": edit.file_path,
                "structured_patch": edit.structured_patch,
                "original_file": edit.original_file,
                "old_string": edit.old_string,
                "new_string": edit.new_string,
                "replace_all": edit.replace_all,
                "user_modified": edit.user_modified,
                "observed_at_ms": edit.observed_at_ms,
            }
            for edit in edits
        ]

    async def get_web_content_constructs(
        self,
        session_id: str,
        *,
        construct_type: str | None = None,
    ) -> list[dict[str, object]] | None:
        """Return typed web-export constructs (search results, canvas, ...) for one session.

        polylogue-kktg: ``web_content_constructs`` holds 155k+ rows written
        every ingest from the ChatGPT/Claude parsers (search queries/
        results, canvas documents, content references, image results, async
        tasks, selected sources, token budgets, voice notes) but before this
        reader had no production reader at all -- every existing SELECT
        against it exists only to DELETE orphans or as a demo smoke-probe
        COUNT(*). This is the read surface.

        ``construct_type`` optionally narrows to one
        ``core.enums.WebConstructType`` value (e.g. ``"search_result"``).

        Returns ``None`` when the session does not exist (distinct from an
        empty list, meaning the session exists but has no captured web
        constructs).
        """
        resolved = await self.repository.resolve_id(session_id)
        resolved_id = str(resolved) if resolved is not None else session_id
        session = await self.repository.get(resolved_id)
        if session is None:
            return None
        constructs = await self.repository.get_web_content_constructs(resolved_id, construct_type=construct_type)
        return [
            {
                "construct_id": construct.construct_id,
                "message_id": str(construct.message_id),
                "block_id": construct.block_id,
                "position": construct.position,
                "provider": construct.provider,
                "construct_type": construct.construct_type,
                "provider_key": construct.provider_key,
                "title": construct.title,
                "url": construct.url,
                "text": construct.text,
                "source_id": construct.source_id,
                "group_id": construct.group_id,
                "group_title": construct.group_title,
                "query": construct.query,
                "asset_pointer": construct.asset_pointer,
                "mime_type": construct.mime_type,
                "status": construct.status,
                "task_id": construct.task_id,
                "task_type": construct.task_type,
                "rank": construct.rank,
                "start_index": construct.start_index,
                "end_index": construct.end_index,
            }
            for construct in constructs
        ]

    async def get_session_materials(self, session_id: str) -> list[dict[str, object]] | None:
        """Return the source-tier materials retained for one session, with their content.

        Codex goals and memories are admitted only as materials, so this is
        where their objective, status and memory text are read back. Rows are
        the ones ``read --view materials`` pages, in the same order.

        Returns ``None`` when the session does not exist (distinct from an
        empty list, meaning it exists with no retained materials).
        """
        from polylogue.operations.session_evidence import read_session_materials

        def work(archive: ArchiveStore) -> list[dict[str, object]] | None:
            try:
                resolved = archive.resolve_session_id(session_id)
            except KeyError:
                return None
            return read_session_materials(archive, resolved)

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.session.materials",
            arguments={"session_id": session_id},
            work=work,
            projection="session-materials",
            stable_order="created_at_ms,material_id",
        )

    async def read_session_evidence_window(
        self,
        session_id: str,
        kind: str,
        *,
        limit: int = 50,
        offset: int = 0,
        continuation: str | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, object] | None:
        """Read one bounded page of a per-session evidence relation.

        ``kind`` names a windowed ``session.read`` evidence kind (``events``,
        ``raw``, ``file-edits``, ``web-content``, ``materials``). The page is
        the ``EvidenceWindowBody`` ``session.read`` returns: ``rows``, the
        relation's own ``total``, the page coordinates, and a ``continuation``
        that resumes it on any surface until ``complete``. A continuation
        whose archive frame or source relation has moved raises
        ``QueryContinuationStaleError``.

        Large file edits and web constructs carry ``row_fragment`` instead
        of a partial row. ``returned`` counts completed rows; field-byte
        progress is carried by the same cross-surface continuation.
        Returns ``None`` when the session does not exist.
        """
        from polylogue.operations.evidence_payloads import DEFAULT_EVIDENCE_PAGE_BYTES
        from polylogue.operations.session_evidence import SESSION_EVIDENCE_PAGE_READERS, read_session_evidence_window

        if kind not in SESSION_EVIDENCE_PAGE_READERS:
            raise ValueError(f"not a windowed session evidence kind: {kind!r}")
        ref = session_id if session_id.startswith("session:") else f"session:{session_id}"

        def work(archive: ArchiveStore) -> dict[str, object] | None:
            window = read_session_evidence_window(
                archive,
                kind,
                ref=ref,
                limit=limit,
                offset=offset,
                continuation=continuation,
                max_bytes=DEFAULT_EVIDENCE_PAGE_BYTES if max_bytes is None else max_bytes,
            )
            return None if window is None else dict(window)

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.session.evidence_window",
            arguments={"session_id": session_id, "kind": kind},
            work=work,
            page_size=limit,
            offset=offset,
            projection=f"session-evidence:{kind}",
        )

    async def get_agent_policies(self, session_id: str) -> list[dict[str, object]] | None:
        """Return sandbox/approval/network policy facts recorded for one session.

        polylogue-nua7: the writer diverts Codex ``agent_policy`` events out
        of ``session_events`` into the dedicated ``session_agent_policies``
        table (fully re-derivable, zero evidence loss -- see
        ``archive_tiers/write.py:_SESSION_EVENTS_REDUNDANT_TYPES``), but
        before this reader nothing above the storage layer could reach it
        back. This is the read surface.

        Returns ``None`` when the session does not exist (distinct from an
        empty list, meaning the session exists but reported no agent-policy
        facts -- expected for non-Codex origins).
        """
        resolved = await self.repository.resolve_id(session_id)
        resolved_id = str(resolved) if resolved is not None else session_id
        session = await self.repository.get(resolved_id)
        if session is None:
            return None
        policies = await self.repository.get_agent_policies(resolved_id)
        return [
            {
                "policy_id": policy.policy_id,
                "position": policy.position,
                "approval_policy": policy.approval_policy,
                "sandbox_policy": policy.sandbox_policy,
                "network_policy": policy.network_policy,
                "observed_at_ms": policy.observed_at_ms,
                "source_message_id": policy.source_message_id,
            }
            for policy in policies
        ]

    async def query_sessions(
        self,
        *,
        origin: str | None = None,
        tag: str | None = None,
        since: str | None = None,
        until: str | None = None,
        sort: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        typed_only: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        **kwargs: object,
    ) -> builtins.list[dict[str, object]]:
        """Query sessions with full filter support.

        Returns lightweight dicts suitable for the web reader and daemon API.
        For full ``Session`` objects use ``list_sessions``.
        """
        from polylogue.archive.query.spec import SessionQuerySpec

        spec = SessionQuerySpec.from_params(
            {
                "origin": origin,
                "tag": tag,
                "since": since,
                "until": until,
                "sort": sort,
                "limit": limit,
                "offset": offset,
                "filter_has_tool_use": has_tool_use,
                "filter_has_thinking": has_thinking,
                "filter_has_paste": has_paste,
                "typed_only": typed_only,
                "min_messages": min_messages,
                "max_messages": max_messages,
                "min_words": min_words,
                **kwargs,
            },
            strict=True,
        )
        archive_summaries = await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.sessions.query",
            arguments={"origin": origin, "tag": tag, "since": since, "until": until, "sort": sort, **kwargs},
            work=lambda archive: _archive_list_summaries_for_spec(
                archive, spec, default_limit=DEFAULT_SESSION_LIST_LIMIT
            ),
            page_size=limit,
            offset=offset,
            projection="session-summary",
            stable_order=sort or "date,session_id",
        )
        return [
            {
                "id": summary.session_id,
                "title": summary.display_label or summary.title,
                "title_is_synthesized": summary.display_label_source == "synthesized",
                "origin": summary.origin,
                "created_at": parse_archive_datetime(summary.created_at),
                "updated_at": parse_archive_datetime(summary.updated_at),
                "message_count": summary.message_count,
                "word_count": summary.word_count,
            }
            for summary in archive_summaries
        ]

    async def list_session_summaries_with_count(
        self,
        spec: SessionQuerySpec,
    ) -> tuple[builtins.list[SessionSummary], int]:
        """Read one session page and its total from the same archive snapshot."""

        def read(archive: Any) -> tuple[builtins.list[SessionSummary], int]:
            from polylogue.archive.query.archive_execution import _count_in_archive, _list_summaries_in_archive

            if spec.session_id is not None:
                try:
                    archive.resolve_session_id(spec.session_id)
                except KeyError:
                    return [], 0
            count_plan = spec.to_plan()
            plan = count_plan
            if spec.sample is not None:
                if spec.sample <= 0:
                    raise ValueError("sample must be positive")
                if spec.query_terms or spec.contains_terms:
                    raise ValueError("sample does not combine with search terms")
                if spec.cursor:
                    raise ValueError("sample does not combine with a cursor")
                plan = replace(plan, sort="random", limit=spec.sample, offset=0, sample=None)
            summaries = _list_summaries_in_archive(
                plan,
                archive,
                config=self.config,
                archive_root=_active_archive_root(self.config),
                default_limit=DEFAULT_SESSION_LIST_LIMIT,
            )
            total = _count_in_archive(
                count_plan,
                archive,
                config=self.config,
                archive_root=_active_archive_root(self.config),
            )
            if spec.latest:
                total = min(total, 1)
            return summaries, total

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.sessions.list-with-count",
            arguments={"spec": spec},
            work=read,
            page_size=spec.limit,
            offset=spec.offset,
            projection="session-summary-and-count",
            workload_class="scan" if spec.limit is None or spec.limit > 1000 else "interactive",
        )

    async def count_sessions(
        self,
        *,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
        **kwargs: object,
    ) -> int:
        """Count sessions matching the given filters."""
        from polylogue.archive.query.spec import SessionQuerySpec

        query_params = dict(kwargs)
        if "has_tool_use" in query_params and "filter_has_tool_use" not in query_params:
            query_params["filter_has_tool_use"] = query_params.pop("has_tool_use")
        if "has_thinking" in query_params and "filter_has_thinking" not in query_params:
            query_params["filter_has_thinking"] = query_params.pop("has_thinking")
        if "has_paste" in query_params and "filter_has_paste" not in query_params:
            query_params["filter_has_paste"] = query_params.pop("has_paste")
        spec = SessionQuerySpec.from_params(
            {"origin": origin, "since": since, "until": until, **query_params},
            strict=True,
        )
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.sessions.count",
            arguments={"origin": origin, "since": since, "until": until, **query_params},
            work=lambda archive: _archive_count_sessions_for_spec(archive, spec),
            page_size=1,
            projection="count",
            workload_class="scan",
        )

    async def export_insight_bundle(
        self,
        request: InsightExportBundleRequest,
    ) -> InsightExportBundleResult:
        """Write a versioned archive-insight export bundle."""
        from polylogue.analysis.export_bundles import export_insight_bundle

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.export_bundle",
            arguments={"request": request},
            work=lambda archive: export_insight_bundle(archive, request, checkpoint=archive.check_operation_read),
            page_size=1,
            projection="insight-export",
            workload_class="scan",
        )

    async def get_session_summary(self, session_id: str) -> SessionSummary | None:
        """Return a summary record for a single session, or ``None`` if not found."""

        def read(archive: ArchiveStore) -> SessionSummary | None:
            try:
                resolved_id = archive.resolve_session_id(session_id)
                return archive_summary_to_domain(archive.read_summary(resolved_id))
            except KeyError:
                return None

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.session_summary.get",
            arguments={"session_id": session_id},
            work=read,
            projection="session-summary",
        )

    async def get_session_summaries(self, session_ids: Sequence[str]) -> dict[str, SessionSummary]:
        """Return summaries for a bounded set of sessions in one archive read.

        Keys are the requested ids; ids that do not resolve are omitted.
        """
        requested = tuple(dict.fromkeys(session_ids))

        def read(archive: ArchiveStore) -> dict[str, SessionSummary]:
            summaries: dict[str, SessionSummary] = {}
            for session_id in requested:
                try:
                    resolved_id = archive.resolve_session_id(session_id)
                    summaries[session_id] = archive_summary_to_domain(archive.read_summary(resolved_id))
                except KeyError:
                    continue
            return summaries

        if not requested:
            return {}
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.session_summary.get_many",
            arguments={"session_ids": list(requested)},
            work=read,
            projection="session-summary",
        )

    async def _get_attachment_library_page(
        self, *, limit: int, offset: int, mime_filter: str = "", session_filter: str = "", state_filter: str = ""
    ) -> Sequence[tuple[object, str, str | None]]:
        """Return a bounded attachment page from the declared archive read."""
        return await self.repository.get_attachment_library_page(
            limit=limit,
            offset=offset,
            mime_filter=mime_filter,
            session_filter=session_filter,
            state_filter=state_filter,
        )

    async def get_session_stats(self, session_id: str) -> dict[str, int]:
        """Return message-count and word-count stats for a single session."""

        def read(archive: ArchiveStore) -> dict[str, int]:
            try:
                resolved_id = archive.resolve_session_id(session_id)
                summary = archive.read_summary(resolved_id)
            except KeyError:
                return {}
            return {
                "messages": summary.message_count,
                "words": summary.word_count,
                "attachments": 0,
            }

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.session_stats.get",
            arguments={"session_id": session_id},
            work=read,
            projection="session-stats",
        )

    async def get_stats_by(self, group_by: str = "origin") -> dict[str, int]:
        """Group session counts by origin/calendar dimensions."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.stats_by",
            arguments={"group_by": group_by},
            work=lambda archive: archive.stats_by(group_by),
            projection="stats-by",
            workload_class="scan",
        )

    async def get_index_status(self) -> IndexStatus:
        """Return archive block-FTS index existence and document count."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.index_status",
            arguments={},
            work=lambda archive: archive.index_status(),
            projection="index-status",
        )

    async def neighbor_candidates(
        self,
        *,
        session_id: str | None = None,
        query: str | None = None,
        origin: str | None = None,
        limit: int = 10,
        window_hours: int = 24,
    ) -> list[SessionNeighborCandidate]:
        """Discover explainable neighboring or near-duplicate candidates.

        At least one of ``session_id`` or ``query`` must be provided.
        """
        from polylogue.archive.session.neighbor_candidates import (
            NeighborDiscoveryRequest,
            discover_neighbor_candidates,
        )
        from polylogue.core.async_bridge import complete_without_suspension

        request = NeighborDiscoveryRequest(
            session_id=session_id,
            query=query,
            origin=origin,
            limit=limit,
            window_hours=window_hours,
        )
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.neighbor_candidates",
            arguments={
                "session_id": session_id,
                "query": query,
                "origin": origin,
                "limit": limit,
                "window_hours": window_hours,
            },
            # The admitted read may run nested on a compute worker driving an
            # event loop; the archive-backed runtime never suspends.
            work=lambda archive: complete_without_suspension(
                discover_neighbor_candidates(_ArchiveNeighborRuntime(archive), request)
            ),
            page_size=limit,
            projection="neighbor-candidates",
            stable_order="score,time,session_id",
            workload_class="scan",
        )

    async def neighbor_candidate_payloads(
        self,
        *,
        session_id: str | None = None,
        query: str | None = None,
        origin: str | None = None,
        limit: int = 10,
        window_hours: int = 24,
    ) -> list[JSONDocument]:
        """Return neighboring-session candidates as shared surface payloads."""

        from polylogue.surfaces.payloads import SessionNeighborCandidatePayload, model_json_document

        candidates = await self.neighbor_candidates(
            session_id=session_id,
            query=query,
            origin=origin,
            limit=limit,
            window_hours=window_hours,
        )
        return [
            model_json_document(SessionNeighborCandidatePayload.from_candidate(candidate), exclude_none=True)
            for candidate in candidates
        ]

    async def session_correlation_payload(
        self,
        session_id: str,
        *,
        repo_path: str | None = None,
        since_hours: int = 2,
        confidence_threshold: float = 0.3,
    ) -> JSONDocument | None:
        """Return git/GitHub correlation evidence as a JSON surface payload."""

        from polylogue.analysis.session_commit import (
            bridge_session_ids_from_events,
            build_correlation_result,
            correlation_result_to_payload,
            typed_refs_from_session_refs,
        )

        session = await self.get_session(session_id)
        if session is None:
            return None
        if session.created_at is None or session.updated_at is None:
            raise SessionNotFoundError("Session has no timestamp data.")

        repo = repo_path
        if not repo:
            repo_url = getattr(session, "git_repository_url", None)
            if isinstance(repo_url, str) and repo_url:
                repo = repo_url
            else:
                directories = getattr(session, "working_directories", ()) or ()
                repo = str(directories[0]) if directories else "."

        messages: list[dict[str, object]] = []
        for message in session.messages:
            content_blocks = list(message.blocks) if getattr(message, "blocks", ()) else []
            messages.append(
                {
                    "id": message.id,
                    "role": message.role.value if hasattr(message.role, "value") else str(message.role),
                    "text": message.text,
                    "content_blocks": content_blocks,
                }
            )

        session_refs = await self.repository.get_session_refs(session_id)
        typed_pr_refs, typed_issue_refs = typed_refs_from_session_refs(session_refs)
        bridge_session_ids = bridge_session_ids_from_events(session.session_events)

        result = build_correlation_result(
            session_id=session_id,
            messages=messages,
            session_created_at=session.created_at,
            session_updated_at=session.updated_at,
            repo_path=repo,
            before_hours=since_hours,
            after_hours=since_hours,
            confidence_threshold=confidence_threshold,
            typed_pr_refs=typed_pr_refs,
            typed_issue_refs=typed_issue_refs,
            bridge_session_ids=bridge_session_ids,
        )
        payload = correlation_result_to_payload(result)
        # polylogue-cijx.3 AC3: session_commits was a write-only table (the
        # parser-reported repo checkout HEAD at session capture, distinct
        # from the on-demand commit-authorship correlation above). Surface
        # it here, clearly separated from `commits` (which is
        # detect_session_commits' scored/heuristic list) rather than merged
        # into it.
        checkout_commits = await self.repository.get_session_commits(session_id)
        payload["checkout_commits"] = [
            {
                "commit_sha": record.commit_sha,
                "short_sha": record.commit_sha[:8],
                "repo_id": record.repo_id,
                "detection_type": record.detection_type,
                "method": record.method,
                "confidence": record.confidence,
                "evidence": record.evidence,
            }
            for record in checkout_commits
        ]
        return cast(JSONDocument, payload)

    async def get_session_tree(self, session_id: str) -> list[Session]:
        """Return the full session tree (parent + children) for a session."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.session_tree",
            arguments={"session_id": session_id},
            work=lambda archive: [
                archive_envelope_to_session(
                    session,
                    display_label=archive.read_summary(session.session_id).display_label,
                    display_label_source=archive.read_summary(session.session_id).display_label_source,
                )
                for session in archive.get_session_tree(session_id)
            ],
            projection="session-tree",
            stable_order="depth,session_id",
        )

    async def list_tags(self, *, origin: str | None = None) -> dict[str, int]:
        """List all tags with session counts, optionally filtered by origin."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.tags.list",
            arguments={"origin": origin},
            work=lambda archive: archive.list_user_tags(origin=origin),
            projection="tags",
            stable_order="tag",
        )

    async def prepare_delete_session(self, session_id: str) -> DeleteSessionPreview:
        """Prepare a permanent delete of one session and return its preview.

        The daemon records the preview under this caller's authenticated
        principal through the same ``mutation.session.delete.preview``
        operation the CLI ``delete`` verb uses. Nothing is deleted until the
        caller presents ``preview_ref`` to :meth:`delete_session_safe`.
        Returns ``outcome="not_found"`` with no reference for an unknown ID.
        """
        from polylogue.api.facade_client import submit_facade_operation
        from polylogue.operations.daemon_errors import DaemonOperationRejectedError
        from polylogue.surfaces.payloads import DeleteSessionPreview

        # The preview operation takes exact stored IDs; resolve the prefix and
        # provider-alias forms the delete itself accepts first.
        try:
            canonical = await self._archive_resolve_session_id(session_id)
        except SessionNotFoundError:
            return DeleteSessionPreview(outcome="not_found", session_id=session_id)
        try:
            state = await submit_facade_operation(
                self.config, "mutation.session.delete.preview", {"session_ids": [canonical or session_id]}
            )
        except DaemonOperationRejectedError as exc:
            if exc.outcome == "selection_is_stale":
                return DeleteSessionPreview(outcome="not_found", session_id=session_id)
            raise
        preview_ref = state.get("preview_ref")
        sample = state.get("session_ids_sample")
        expires_at_ms = state.get("expires_at_ms")
        if (
            not isinstance(preview_ref, str)
            or not preview_ref
            or not isinstance(sample, list)
            or len(sample) != 1
            or not isinstance(sample[0], str)
            or type(expires_at_ms) is not int
        ):
            raise ValueError("daemon returned an invalid single-session delete preview")
        return DeleteSessionPreview(
            outcome="prepared", session_id=sample[0], preview_ref=preview_ref, expires_at_ms=expires_at_ms
        )

    async def delete_session(self, session_id: str, *, preview_ref: str) -> bool:
        """Permanently delete a session under a presented delete preview.

        Returns ``True`` if something was deleted, ``False`` if the session
        was not found. See :meth:`delete_session_safe`.
        """
        result = await self.delete_session_safe(session_id, preview_ref=preview_ref)
        return result.outcome == "deleted"

    async def delete_session_safe(self, session_id: str, *, preview_ref: str) -> DeleteSessionResult:
        """Typed delete that returns ``outcome="deleted"`` or ``"not_found"``.

        ``preview_ref`` is the reference :meth:`prepare_delete_session`
        returned to this caller. The daemon authorizes that exact preview and
        consumes it once, so a missing, foreign, reused, expired, or stale
        reference is refused (``DaemonOperationRejectedError``) and deletes
        nothing. The audit receipt's ``bound_token`` strength therefore
        records a plan this caller prepared and presented.
        """
        from polylogue.api.facade_client import submit_facade_writer
        from polylogue.surfaces.payloads import DeleteSessionResult

        value = await submit_facade_writer(
            self.config, "delete_session", {"session_id": session_id, "preview_ref": preview_ref}
        )
        return DeleteSessionResult.model_validate(value)

    async def add_tag(
        self,
        session_id: str,
        tag: str,
        *,
        author_ref: str | None = None,
        author_kind: str | None = None,
    ) -> TagMutationResult:
        """Add a tag to a session.

        Returns a ``TagMutationResult`` with:
        - ``outcome="added"`` if the tag was newly added
        - ``outcome="no_op"`` if the tag was already present

        Routed through ``OperationExecutor``/``TagAddActuator`` (t46.9 phase
        2): the real mutation is still ``ArchiveStore.add_user_tags``, but
        every surface (this facade method, reached by CLI's
        ``apply_modifiers``/query-mutation path and MCP's
        ``write(operation='add_tag')``) now shares one preview/authorize/
        receipt contract instead of calling the primitive independently.
        """
        from polylogue.surfaces.payloads import TagMutationResult

        # The lookup window (``session_id=``) is what answers "session never
        # existed". A target that disappears after AUTHORIZE surfaces as
        # ``MutationTargetVanishedError``, not as a missing session.
        receipt, _plan = await submit_facade_product(
            self.config, "add_tag", session_id=session_id, tag=tag, author_ref=author_ref, author_kind=author_kind
        )
        changed = receipt.affected_count
        return TagMutationResult(
            outcome="added" if changed else "no_op",
            detail=None if changed else "already_present",
        )

    async def remove_tag(self, session_id: str, tag: str) -> TagMutationResult:
        """Remove a tag from a session.

        Returns a ``TagMutationResult`` with:
        - ``outcome="removed"`` if the tag was removed
        - ``outcome="not_present"`` if the tag was not present

        Routed through ``OperationExecutor``/``TagRemoveActuator`` (t46.9
        phase 2); see :meth:`add_tag` for the shared-contract rationale.
        """
        from polylogue.surfaces.payloads import TagMutationResult

        receipt, _plan = await submit_facade_product(self.config, "remove_tag", session_id=session_id, tag=tag)
        changed = receipt.affected_count
        return TagMutationResult(
            outcome="removed" if changed else "not_present",
            detail=None if changed else "tag_not_present",
        )

    async def get_metadata(self, session_id: str) -> dict[str, str]:
        """Return all metadata key-value pairs for a session."""

        def read(archive: ArchiveStore) -> dict[str, str]:
            try:
                doc = archive.read_user_metadata(session_id)
            except KeyError:
                return {}
            return {str(k): str(v) if not isinstance(v, str) else v for k, v in doc.items()}

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.metadata.get",
            arguments={"session_id": session_id},
            work=read,
            projection="metadata",
        )

    async def update_metadata(self, session_id: str, key: str, value: str) -> bool:
        """Set a metadata key on a session.

        Returns ``True`` if the value was changed, ``False`` if it was already set
        to the same value. This is the boolean wrapper over
        :meth:`set_metadata`, so it follows the active archive backend.
        """
        result = await self.set_metadata(session_id, key, value)
        return result.outcome == "set"

    async def set_metadata(self, session_id: str, key: str, value: object) -> MetadataMutationResult:
        """Typed metadata-set returning ``outcome="set"`` or ``"unchanged"``.

        Follows the centralized mutation contract (#862): the key is
        validated before any store call (raising
        :class:`~polylogue.surfaces.payloads.MetadataKeyValidationError`),
        and the ``unchanged`` detail token is the shared ``value_unchanged``.

        Routed through ``OperationExecutor``/``MetadataSetActuator`` (t46.9
        phase 2); see :meth:`add_tag` for the shared-contract rationale.
        """
        from polylogue.surfaces.payloads import (
            MetadataKeyValidationError,
            MetadataMutationResult,
            validate_metadata_key,
        )

        validation_error = validate_metadata_key(key)
        if validation_error is not None:
            raise MetadataKeyValidationError(validation_error)

        receipt, plan = await submit_facade_product(
            self.config, "set_metadata", session_id=session_id, key=key, value=value
        )
        changed = receipt.affected_count
        resolved = str(plan.context["session_id"])
        return MetadataMutationResult(
            outcome="set" if changed else "unchanged",
            session_id=resolved,
            key=key,
            detail=None if changed else "value_unchanged",
        )

    async def delete_metadata(self, session_id: str, key: str) -> MetadataMutationResult:
        """Typed metadata-delete returning ``outcome="deleted"`` or ``"not_found"``.

        Follows the centralized mutation contract (#862): the key is
        validated before any store call, and the missing-key detail token is
        the shared ``key_not_found``.

        Routed through ``OperationExecutor``/``MetadataDeleteActuator``
        (t46.9 phase 2); see :meth:`add_tag` for the shared-contract
        rationale.
        """
        from polylogue.surfaces.payloads import (
            MetadataKeyValidationError,
            MetadataMutationResult,
            validate_metadata_key,
        )

        validation_error = validate_metadata_key(key)
        if validation_error is not None:
            raise MetadataKeyValidationError(validation_error)

        receipt, plan = await submit_facade_product(self.config, "delete_metadata", session_id=session_id, key=key)
        changed = receipt.affected_count
        resolved = str(plan.context["session_id"])
        return MetadataMutationResult(
            outcome="deleted" if changed else "not_found",
            session_id=resolved,
            key=key,
            detail=None if changed else "key_not_found",
        )

    async def bulk_tag_sessions(
        self,
        session_ids: list[str],
        tags: list[str],
        *,
        author_ref: str | None = None,
        author_kind: str | None = None,
    ) -> BulkTagMutationResult:
        """Apply a bulk-tag operation across many sessions (#862).

        Reject empty selections and tags before submitting the audited daemon
        product.

        Routed through ``OperationExecutor``/``BulkTagActuator`` (t46.9 phase
        2); see :meth:`add_tag` for the shared-contract rationale.
        """
        from polylogue.surfaces.payloads import BulkTagMutationResult

        if not session_ids:
            raise ValueError("bulk_tag_sessions requires at least one session_id")
        if not tags:
            raise ValueError("bulk_tag_sessions requires at least one tag")

        receipt, _plan = await submit_facade_product(
            self.config,
            "bulk_tag_sessions",
            session_ids=tuple(session_ids),
            tags=tuple(tags),
            author_ref=author_ref,
            author_kind=author_kind,
        )
        domain_receipt = receipt.domain_receipt
        return BulkTagMutationResult(
            session_count=int(cast("int", domain_receipt["session_count"])),
            tag_count=int(cast("int", domain_receipt["tag_count"])),
            affected_count=int(cast("int", domain_receipt["affected_count"])),
            skipped_count=int(cast("int", domain_receipt["skipped_count"])),
        )

    # ------------------------------------------------------------------
    # Marks
    # ------------------------------------------------------------------

    @staticmethod
    def _user_state_mutation_input(
        session_id: str,
        *,
        target_type: str,
        target_id: str | None,
        message_id: str | None,
    ) -> dict[str, str | None]:
        """Preserve target selectors for the daemon's pinned Source admission.

        The daemon resolves positional input under its existing Source-bound
        seal. Resolving it here would consult a separate Index snapshot and
        could persist whichever block currently occupies the requested slot.
        """
        from polylogue.core.user_state_targets import validate_target_kind

        if not session_id:
            raise ValueError("session_id must not be empty")
        validate_target_kind(target_type)
        if target_type == TARGET_MESSAGE and target_id and message_id and target_id != message_id:
            raise ValueError("message target_id must match message_id")
        selected_id = target_id
        if target_type == TARGET_SESSION:
            selected_id = target_id or session_id
        elif target_type == TARGET_MESSAGE:
            selected_id = target_id or message_id
        elif not selected_id:
            raise ValueError(f"{target_type} target requires target_id")
        return {
            "target_type": target_type,
            "target_id": selected_id,
            "session_id": session_id,
            "message_id": message_id,
        }

    async def _resolve_user_state_target(
        self,
        session_id: str,
        *,
        target_type: str = TARGET_SESSION,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> dict[str, str | None]:
        from polylogue.api.user_state_resolver import resolve_insight_target
        from polylogue.core.user_state_targets import validate_target_kind

        resolved_session_id = await self._resolve_user_state_session_id(session_id)
        if target_type == TARGET_SESSION:
            if target_id:
                resolved_target_id = await self._resolve_user_state_session_id(target_id)
                if resolved_target_id != resolved_session_id:
                    raise ValueError("session target_id must match session_id")
            return {
                "target_type": TARGET_SESSION,
                "target_id": resolved_session_id,
                "session_id": resolved_session_id,
                "message_id": None,
            }
        if target_type == TARGET_MESSAGE:
            if target_id and message_id and target_id != message_id:
                raise ValueError("message target_id must match message_id")
            resolved_message_id = message_id or target_id
            if not resolved_message_id:
                raise ValueError("message target requires message_id or target_id")
            if not await self._user_state_message_exists(resolved_session_id, resolved_message_id):
                raise ValueError(f"message {resolved_message_id!r} is not in session {resolved_session_id!r}")
            return {
                "target_type": TARGET_MESSAGE,
                "target_id": resolved_message_id,
                "session_id": resolved_session_id,
                "message_id": resolved_message_id,
            }
        validate_target_kind(target_type)
        resolved_target = await resolve_insight_target(
            _active_archive_root(self.config),
            target_type=target_type,
            target_id=target_id,
            session_id=resolved_session_id,
            message_id=message_id,
        )
        # Strip the identity_key — the storage layer doesn't carry it,
        # the recall-pack/workspace resolver re-derives it.
        return {
            "target_type": resolved_target["target_type"],
            "target_id": resolved_target["target_id"],
            "session_id": resolved_target["session_id"],
            "message_id": resolved_target.get("message_id"),
        }

    async def _resolve_user_state_session_id(self, session_id: str) -> str:
        try:
            archive_resolved = await self._archive_resolve_session_id(session_id)
        except SessionNotFoundError:
            archive_resolved = None
        if archive_resolved is not None:
            return archive_resolved
        durable_resolved = _resolve_durable_user_state_session_id(
            _active_archive_root(self.config),
            session_id,
        )
        if durable_resolved is not None:
            return durable_resolved
        raise SessionNotFoundError(session_id)

    async def _user_state_message_exists(self, session_id: str, message_id: str) -> bool:
        return bool(await self._archive_message_exists(session_id, message_id))

    async def _archive_resolve_session_id(self, token: str) -> str | None:
        archive_db = _active_archive_root(self.config) / "index.db"
        if not archive_db.exists():
            return None
        try:
            return await run_archive_read(
                _active_archive_root(self.config),
                operation="archive.session.resolve",
                arguments={"token": token},
                work=lambda archive: archive.resolve_session_id(token),
                projection="session-id",
            )
        except KeyError:
            raise SessionNotFoundError(token) from None
        except ValueError:
            raise
        except Exception:
            return None

    async def _archive_message_exists(self, session_id: str, message_id: str) -> bool | None:
        archive_db = _active_archive_root(self.config) / "index.db"
        if not archive_db.exists():
            return None
        try:
            return await run_archive_read(
                _active_archive_root(self.config),
                operation="archive.message.exists",
                arguments={"session_id": session_id, "message_id": message_id},
                work=lambda archive: (
                    archive._conn.execute(
                        "SELECT 1 FROM messages WHERE session_id = ? AND message_id = ?",
                        (session_id, message_id),
                    ).fetchone()
                    is not None
                ),
                projection="message-existence",
            )
        except sqlite3.Error:
            return None

    async def add_mark(
        self,
        session_id: str,
        mark_type: str,
        *,
        target_type: str = TARGET_SESSION,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> bool:
        """Add a mark (star/pin/archive) to a session or message.

        Returns ``True`` if the mark was newly added, ``False`` if it already
        existed.

        Routed through ``OperationExecutor``/``MarkAddActuator`` (t46.9 phase
        2: the first MCP no-spec mutation family to gain executor routing).
        Raw target selectors travel to the daemon, where positional inputs are
        canonicalized under the pinned Source-bound seal before persistence.
        """
        from polylogue.core.user_state_targets import validate_mark_type

        mark_type = validate_mark_type(mark_type)
        target = self._user_state_mutation_input(
            session_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        receipt, _plan = await submit_facade_product(
            self.config,
            "add_mark",
            target_type=str(target["target_type"]),
            target_id=str(target["target_id"]),
            message_id=target["message_id"],
            mark_type=mark_type,
            owner_session_id=str(target["session_id"]) if target.get("session_id") else None,
        )
        return receipt.status == "applied"

    async def remove_mark(
        self,
        session_id: str,
        mark_type: str,
        *,
        target_type: str = TARGET_SESSION,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> bool:
        """Remove a mark from a session or message. Returns ``True`` if removed.

        Routed through ``OperationExecutor``/``MarkRemoveActuator`` (t46.9
        phase 2); see :meth:`add_mark` for the shared-contract rationale.
        """
        from polylogue.core.user_state_targets import validate_mark_type

        mark_type = validate_mark_type(mark_type)
        target = self._user_state_mutation_input(
            session_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        receipt, _plan = await submit_facade_product(
            self.config,
            "remove_mark",
            target_type=str(target["target_type"]),
            target_id=str(target["target_id"]),
            message_id=target["message_id"],
            mark_type=mark_type,
            owner_session_id=str(target["session_id"]) if target.get("session_id") else None,
        )
        return receipt.status == "applied"

    async def list_marks(
        self,
        *,
        mark_type: str | None = None,
        session_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List marks, optionally filtered by type, target, session, or message."""
        resolved_target_type = target_type
        resolved_target_id = target_id
        scope_session_id: str | None = None
        if message_id is not None:
            resolved_target_type = TARGET_MESSAGE
            resolved_target_id = message_id
        elif session_id is not None and target_id is None:
            try:
                scope_session_id = await self._resolve_user_state_session_id(session_id)
            except SessionNotFoundError:
                # A durable user assertion can outlive the rebuildable
                # session row. Keep the caller's canonical token.
                scope_session_id = session_id
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.marks.list",
            arguments={
                "mark_type": mark_type,
                "target_type": resolved_target_type,
                "target_id": resolved_target_id,
                "session_id": scope_session_id,
            },
            work=lambda archive: archive.list_marks(
                mark_type=mark_type,
                target_type=resolved_target_type,
                target_id=resolved_target_id,
                session_id=scope_session_id,
            ),
            projection="marks",
            stable_order="created_at,target_id",
        )

    async def save_annotation(
        self,
        annotation_id: str,
        session_id: str,
        note_text: str,
        *,
        target_type: str = TARGET_SESSION,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> bool:
        """Create or update an annotation. Returns ``True`` if newly created.

        Routed through ``OperationExecutor``/``AnnotationSaveActuator``
        (t46.9 phase 3); see :meth:`add_mark` for the shared-contract
        rationale. Raw target selectors are resolved by the daemon under its
        pinned Source-bound seal before persistence.
        """
        if not annotation_id.strip():
            raise ValueError("annotation_id must not be empty")
        if not note_text.strip():
            raise ValueError("note_text must not be empty")

        target = self._user_state_mutation_input(
            session_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        receipt, _plan = await submit_facade_product(
            self.config,
            "save_annotation",
            annotation_id=annotation_id,
            target_type=str(target["target_type"]),
            target_id=str(target["target_id"]),
            message_id=target["message_id"],
            note_text=note_text,
            owner_session_id=str(target["session_id"]) if target.get("session_id") else None,
        )
        return bool(receipt.domain_receipt.get("created"))

    async def get_annotation(self, annotation_id: str) -> dict[str, str] | None:
        """Get an annotation by ID."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.annotation.get",
            arguments={"annotation_id": annotation_id},
            work=lambda archive: archive.get_annotation(annotation_id),
            projection="annotation",
        )

    async def list_annotations(
        self,
        *,
        session_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List annotations, optionally filtered by target, session, or message."""
        resolved_target_type = target_type
        resolved_target_id = target_id
        scope_session_id: str | None = None
        if message_id is not None:
            resolved_target_type = "message"
            resolved_target_id = message_id
        elif session_id is not None and target_id is None:
            try:
                scope_session_id = await self._resolve_user_state_session_id(session_id)
            except SessionNotFoundError:
                # See list_marks: the user tier remains authoritative after
                # the rebuildable index row is gone.
                scope_session_id = session_id
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.annotations.list",
            arguments={
                "target_type": resolved_target_type,
                "target_id": resolved_target_id,
                "session_id": scope_session_id,
            },
            work=lambda archive: archive.list_annotations(
                target_type=resolved_target_type,
                target_id=resolved_target_id,
                session_id=scope_session_id,
            ),
            projection="annotations",
            stable_order="created_at,annotation_id",
        )

    async def delete_annotation(self, annotation_id: str) -> bool:
        """Delete an annotation. Returns ``True`` if deleted.

        Routed through ``OperationExecutor``/``AnnotationDeleteActuator``
        (t46.9 phase 3); see :meth:`add_mark` for the shared-contract
        rationale.
        """

        receipt, _plan = await submit_facade_product(self.config, "delete_annotation", annotation_id=annotation_id)
        return receipt.status == "applied"

    # ------------------------------------------------------------------
    # Saved views
    # ------------------------------------------------------------------

    async def save_view(self, view_id: str, name: str, query_json: str, *, watch: bool = False) -> bool:
        """Save a named query view. Returns ``True`` if newly created.

        Routed through ``OperationExecutor``/``SavedViewSaveActuator`` (t46.9
        phase 4); see :meth:`add_mark` for the shared-contract rationale.

        ``watch=True`` is the product's creation route for a standing query:
        the name is promoted into the durable watched-query substrate the
        daemon's standing-query convergence stage re-evaluates
        (polylogue-pm8cj). A selection with no evaluable predicate is refused
        at plan time.
        """

        receipt, _plan = await submit_facade_product(
            self.config, "save_view", view_id=view_id, name=name, query_json=query_json, watch=watch
        )
        return bool(receipt.domain_receipt.get("created"))

    async def get_view(self, view_id: str) -> dict[str, str] | None:
        """Get a saved view by ID."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.view.get",
            arguments={"view_id": view_id},
            work=lambda archive: archive.get_view(view_id),
            projection="saved-view",
        )

    async def list_views(self) -> list[dict[str, str]]:
        """List all saved views."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.views.list",
            arguments={},
            work=lambda archive: archive.list_views(),
            projection="saved-views",
            stable_order="name,view_id",
        )

    async def delete_view(self, view_id: str) -> bool:
        """Delete a saved view. Returns ``True`` if deleted.

        Routed through ``OperationExecutor``/``SavedViewDeleteActuator``
        (t46.9 phase 4); see :meth:`add_mark` for the shared-contract
        rationale.
        """

        receipt, _plan = await submit_facade_product(self.config, "delete_view", view_id=view_id)
        return receipt.status == "applied"

    # ------------------------------------------------------------------
    # Recall packs
    # ------------------------------------------------------------------

    async def _resolve_recall_pack_item(self, item: dict[str, object]) -> dict[str, object]:
        item_type = str(item.get("target_type") or item.get("type") or "session")
        if item_type == "session":
            session_id = str(item.get("session_id") or item.get("target_id") or item.get("id") or "")
            try:
                resolved_id = await self._resolve_user_state_session_id(session_id) if session_id else None
            except SessionNotFoundError:
                resolved_id = None
            if resolved_id is None:
                return {
                    "target_type": "session",
                    "target_id": session_id,
                    "session_id": session_id or None,
                    "status": "missing",
                    "disabled_reason": "session_not_found",
                }
            return {
                "target_type": "session",
                "target_id": resolved_id,
                "session_id": resolved_id,
                "status": "resolved",
                "identity_key": f"session:{resolved_id}",
            }

        if item_type == "message":
            session_id = str(item.get("session_id") or "")
            message_id = str(item.get("message_id") or item.get("target_id") or item.get("id") or "")
            try:
                target = await self._resolve_user_state_target(
                    session_id,
                    target_type="message",
                    message_id=message_id,
                )
            except (SessionNotFoundError, ValueError) as exc:
                return {
                    "target_type": "message",
                    "target_id": message_id,
                    "session_id": session_id or None,
                    "message_id": message_id or None,
                    "status": "missing",
                    "disabled_reason": str(exc) or "message_not_found",
                }
            session_target_id = str(target["session_id"])
            resolved_message_id = str(target["message_id"])
            return {
                "target_type": "message",
                "target_id": resolved_message_id,
                "session_id": session_target_id,
                "message_id": resolved_message_id,
                "status": "resolved",
                "identity_key": f"message:{session_target_id}:{resolved_message_id}",
            }

        if item_type == "annotation":
            annotation_id = str(item.get("annotation_id") or item.get("target_id") or item.get("id") or "")
            row = await self.get_annotation(annotation_id) if annotation_id else None
            if row is None:
                return {
                    "target_type": "annotation",
                    "target_id": annotation_id,
                    "annotation_id": annotation_id or None,
                    "status": "missing",
                    "disabled_reason": "annotation_not_found",
                }
            return {
                "target_type": "annotation",
                "target_id": row["annotation_id"],
                "annotation_id": row["annotation_id"],
                "session_id": row["session_id"],
                "message_id": row["message_id"] or None,
                "annotated_target_type": row["target_type"],
                "annotated_target_id": row["target_id"],
                "note_text": row["note_text"],
                "status": "resolved",
                "identity_key": f"annotation:{row['annotation_id']}",
            }

        if item_type == "mark":
            mark_type = str(item.get("mark_type") or "")
            mark_target_type = str(item.get("mark_target_type") or item.get("target_ref_type") or "session")
            mark_target_id = str(item.get("mark_target_id") or item.get("target_id") or item.get("id") or "")
            session_id = str(item.get("session_id") or "")
            mark_message_id: str | None = str(item.get("message_id") or "") or None
            if not mark_type:
                return {
                    "target_type": "mark",
                    "target_id": mark_target_id,
                    "session_id": session_id or None,
                    "message_id": mark_message_id,
                    "status": "missing",
                    "disabled_reason": "mark_type_missing",
                }
            rows = await self.list_marks(
                mark_type=mark_type,
                session_id=session_id or None,
                target_type=mark_target_type,
                target_id=mark_target_id or None,
                message_id=mark_message_id,
            )
            if not rows:
                return {
                    "target_type": "mark",
                    "target_id": f"{mark_target_type}:{mark_target_id}:{mark_type}",
                    "session_id": session_id or None,
                    "message_id": mark_message_id,
                    "mark_type": mark_type,
                    "mark_target_type": mark_target_type,
                    "mark_target_id": mark_target_id,
                    "status": "missing",
                    "disabled_reason": "mark_not_found",
                }
            row = rows[0]
            return {
                "target_type": "mark",
                "target_id": f"{row['target_type']}:{row['target_id']}:{row['mark_type']}",
                "session_id": row["session_id"],
                "message_id": row["message_id"] or None,
                "mark_type": row["mark_type"],
                "mark_target_type": row["target_type"],
                "mark_target_id": row["target_id"],
                "status": "resolved",
                "identity_key": f"mark:{row['target_type']}:{row['target_id']}:{row['mark_type']}",
            }

        from polylogue.core.user_state_targets import TARGET_KIND_NAMES

        if item_type in TARGET_KIND_NAMES:
            return await self._resolve_recall_pack_insight_item(item, item_type)

        return {
            "target_type": item_type,
            "target_id": str(item.get("target_id") or item.get("id") or ""),
            "status": "unsupported",
            "disabled_reason": "unsupported_target_type",
        }

    async def _resolve_recall_pack_insight_item(
        self,
        item: dict[str, object],
        item_type: str,
    ) -> dict[str, object]:
        """Resolve a recall-pack item for a non-session/message kind (#1113)."""

        session_id = str(item.get("session_id") or "")
        target_id = str(item.get("target_id") or item.get("id") or "")
        message_id_raw = item.get("message_id")
        message_id: str | None = str(message_id_raw) if message_id_raw else None

        # session targets default target_id to the session_id when omitted.
        if item_type == "session" and not target_id and session_id:
            target_id = session_id

        if not session_id:
            return {
                "target_type": item_type,
                "target_id": target_id,
                "session_id": None,
                "message_id": message_id,
                "status": "missing",
                "disabled_reason": "session_id_required",
            }
        try:
            resolved = await self._resolve_user_state_target(
                session_id,
                target_type=item_type,
                target_id=target_id or None,
                message_id=message_id,
            )
        except (SessionNotFoundError, ValueError) as exc:
            return {
                "target_type": item_type,
                "target_id": target_id,
                "session_id": session_id or None,
                "message_id": message_id,
                "status": "missing",
                "disabled_reason": str(exc) or f"{item_type}_not_found",
            }
        from polylogue.core.user_state_targets import identity_key

        resolved_target_id = str(resolved["target_id"])
        resolved_session_id = str(resolved["session_id"])
        resolved_message_id_raw = resolved.get("message_id")
        resolved_message_id: str | None = str(resolved_message_id_raw) if resolved_message_id_raw else None
        return {
            "target_type": item_type,
            "target_id": resolved_target_id,
            "session_id": resolved_session_id,
            "message_id": resolved_message_id,
            "status": "resolved",
            "identity_key": identity_key(
                item_type,
                session_id=resolved_session_id,
                target_id=resolved_target_id,
                message_id=resolved_message_id,
            ),
        }

    async def _build_recall_pack_payload(
        self,
        *,
        label: str,
        payload: dict[str, object],
    ) -> tuple[list[str], str]:
        explicit_items = payload.get("items")
        if not isinstance(explicit_items, list) or not all(isinstance(item, dict) for item in explicit_items):
            raise ValueError("recall pack payload must include an items list of objects")
        raw_items = list(explicit_items)

        items = [await self._resolve_recall_pack_item(item) for item in raw_items]
        resolved_session_ids: list[str] = []
        for item in items:
            session_id = item.get("session_id")
            if (
                item.get("status") == "resolved"
                and isinstance(session_id, str)
                and session_id not in resolved_session_ids
            ):
                resolved_session_ids.append(session_id)

        normalized_payload = {
            "schema_version": 1,
            "label": label,
            "summary": payload.get("summary") or payload.get("reason") or "",
            "items": items,
            "resolved_count": sum(1 for item in items if item.get("status") == "resolved"),
            "degraded_count": sum(1 for item in items if item.get("status") != "resolved"),
        }
        for key, value in payload.items():
            if key not in {"items", "summary", "reason"}:
                normalized_payload[key] = value
        import json

        return resolved_session_ids, json.dumps(normalized_payload, sort_keys=True, separators=(",", ":"))

    async def create_recall_pack(self, pack_id: str, label: str, payload_json: str) -> bool:
        """Save a recall pack. Returns ``True`` if newly created.

        Routed through ``OperationExecutor``/``RecallPackSaveActuator`` (t46.9
        phase 4); see :meth:`add_mark` for the shared-contract rationale.
        Item resolution (async, may consult insight-derived indexes) stays
        here and runs once before the actuator sees already-normalized
        ``session_ids_json``/``payload_json``.
        """
        import json

        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise ValueError("recall pack payload must be a JSON object")
        resolved_session_ids, normalized_payload_json = await self._build_recall_pack_payload(
            label=label,
            payload=payload,
        )
        session_ids_json = json.dumps(resolved_session_ids, sort_keys=True)

        receipt, _plan = await submit_facade_product(
            self.config,
            "create_recall_pack",
            pack_id=pack_id,
            label=label,
            session_ids_json=session_ids_json,
            payload_json=normalized_payload_json,
        )
        return bool(receipt.domain_receipt.get("created"))

    async def get_recall_pack(self, pack_id: str) -> dict[str, str] | None:
        """Get a recall pack by ID."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.recall_pack.get",
            arguments={"pack_id": pack_id},
            work=lambda archive: archive.get_recall_pack(pack_id),
            projection="recall-pack",
        )

    async def list_recall_packs(self) -> list[dict[str, str]]:
        """List all recall packs."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.recall_packs.list",
            arguments={},
            work=lambda archive: archive.list_recall_packs(),
            projection="recall-packs",
            stable_order="updated_at,pack_id",
        )

    async def delete_recall_pack(self, pack_id: str) -> bool:
        """Delete a recall pack. Returns ``True`` if deleted.

        Routed through ``OperationExecutor``/``RecallPackDeleteActuator``
        (t46.9 phase 4); see :meth:`add_mark` for the shared-contract
        rationale.
        """

        receipt, _plan = await submit_facade_product(self.config, "delete_recall_pack", pack_id=pack_id)
        return receipt.status == "applied"

    # ------------------------------------------------------------------
    # Reader workspaces
    # ------------------------------------------------------------------

    async def _build_workspace_targets(
        self, open_targets: Sequence[dict[str, object]]
    ) -> tuple[list[dict[str, object]], str]:
        import json

        items = [await self._resolve_recall_pack_item(item) for item in open_targets]
        return items, json.dumps(items, sort_keys=True, separators=(",", ":"))

    async def _build_workspace_active_target(self, active_target: dict[str, object]) -> str:
        import json

        if not active_target:
            return "{}"
        return json.dumps(await self._resolve_recall_pack_item(active_target), sort_keys=True, separators=(",", ":"))

    async def save_workspace(
        self,
        workspace_id: str,
        name: str,
        mode: str,
        open_targets_json: str,
        layout_json: str,
        active_target_json: str = "{}",
    ) -> bool:
        """Create or update a durable reader workspace.

        Routed through ``OperationExecutor``/``WorkspaceSaveActuator`` (t46.9
        phase 4); see :meth:`add_mark` for the shared-contract rationale.
        Validation and item resolution (async, may consult insight-derived
        indexes) stay here and run once before the actuator sees
        already-normalized JSON payloads.
        """
        import json

        workspace_id = workspace_id.strip()
        name = name.strip()
        mode = mode.strip()
        if not workspace_id:
            raise ValueError("workspace_id must not be empty")
        if not name:
            raise ValueError("name must not be empty")
        if mode not in {"tabs", "stack", "compare", "timeline"}:
            raise ValueError("mode must be one of: tabs, stack, compare, timeline")

        open_targets = json.loads(open_targets_json)
        if not isinstance(open_targets, list) or not all(isinstance(item, dict) for item in open_targets):
            raise ValueError("open_targets_json must encode a list of objects")
        _, normalized_targets_json = await self._build_workspace_targets(open_targets)

        layout = json.loads(layout_json)
        if not isinstance(layout, dict):
            raise ValueError("layout_json must encode an object")
        normalized_layout_json = json.dumps(layout, sort_keys=True, separators=(",", ":"))

        active_target = json.loads(active_target_json)
        if not isinstance(active_target, dict):
            raise ValueError("active_target_json must encode an object")
        normalized_active_json = await self._build_workspace_active_target(active_target)

        receipt, _plan = await submit_facade_product(
            self.config,
            "save_workspace",
            workspace_id=workspace_id,
            name=name,
            mode=mode,
            open_targets_json=normalized_targets_json,
            layout_json=normalized_layout_json,
            active_target_json=normalized_active_json,
        )
        return bool(receipt.domain_receipt.get("created"))

    async def get_workspace(self, workspace_id: str) -> dict[str, str] | None:
        """Get a durable reader workspace by ID."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.workspace.get",
            arguments={"workspace_id": workspace_id},
            work=lambda archive: archive.get_workspace(workspace_id),
            projection="workspace",
        )

    async def list_workspaces(self) -> list[dict[str, str]]:
        """List durable reader workspaces."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.workspaces.list",
            arguments={},
            work=lambda archive: archive.list_workspaces(),
            projection="workspaces",
            stable_order="updated_at,workspace_id",
        )

    async def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a durable reader workspace. Returns ``True`` if deleted.

        Routed through ``OperationExecutor``/``WorkspaceDeleteActuator``
        (t46.9 phase 4); see :meth:`add_mark` for the shared-contract
        rationale.
        """

        receipt, _plan = await submit_facade_product(self.config, "delete_workspace", workspace_id=workspace_id)
        return receipt.status == "applied"

    # ------------------------------------------------------------------
    # Learning corrections (#1131)
    #
    # User-recorded overrides that the insight materialization paths
    # consult after computing their base suggestion. Lives outside the
    # content-hash boundary by construction; see
    # :mod:`polylogue.analysis.feedback` and
    # :mod:`polylogue.storage.derived.feedback`.
    # ------------------------------------------------------------------

    async def record_correction(
        self,
        session_id: str,
        kind: str,
        payload: dict[str, str],
        *,
        note: str | None = None,
        author_ref: str | None = None,
        author_kind: str | None = None,
    ) -> LearningCorrection:
        """Record a typed user correction for a session.

        Resolves the session ID first (short IDs are accepted) so
        the durable row is keyed by the canonical ID. Raises
        :class:`SessionNotFoundError` when the target session does
        not exist and
        :class:`~polylogue.analysis.feedback.UnknownCorrectionKindError`
        when ``kind`` is not a recognized
        :class:`~polylogue.analysis.feedback.CorrectionKind`.

        Routed through ``OperationExecutor``/``CorrectionRecordActuator``
        (t46.9 phase 5); see :meth:`save_annotation` for the shared-contract
        rationale.
        """

        parse_correction_kind(kind)

        receipt, _plan = await submit_facade_product(
            self.config,
            "record_correction",
            session_id=session_id,
            kind=kind,
            payload={str(key): str(value) for key, value in payload.items()},
            note=note,
            author_ref=author_ref,
            author_kind=author_kind,
        )
        return LearningCorrection.model_validate(receipt.domain_receipt["correction"])

    async def list_corrections(
        self,
        *,
        session_id: str | None = None,
        kind: str | None = None,
    ) -> list[LearningCorrection]:
        """List stored corrections, optionally filtered by session/kind."""

        if kind is not None:
            parse_correction_kind(kind)
        try:
            return await run_archive_read(
                _active_archive_root(self.config),
                operation="user_state.corrections.list",
                arguments={"session_id": session_id, "kind": kind},
                work=lambda archive: archive.list_corrections(session_id=session_id, kind=kind),
                projection="corrections",
                stable_order="created_at,correction_id",
            )
        except KeyError as exc:
            raise SessionNotFoundError(str(session_id)) from exc

    async def delete_correction(self, session_id: str, kind: str) -> bool:
        """Delete one correction. Returns ``True`` when a row was removed.

        Routed through ``OperationExecutor``/``CorrectionDeleteActuator``
        (t46.9 phase 5); see :meth:`delete_annotation` for the
        shared-contract rationale.
        """

        parse_correction_kind(kind)

        receipt, _plan = await submit_facade_product(self.config, "delete_correction", session_id=session_id, kind=kind)
        return receipt.status == "applied"

    async def clear_corrections(self, session_id: str) -> int:
        """Delete every correction for a session. Returns the count.

        Routed through ``OperationExecutor``/``CorrectionsClearActuator``
        (t46.9 phase 5): the plan resolves the exact live set of
        correction kinds for the session, so a concurrent
        ``record_correction`` between preview and apply forces a replan
        instead of silently clearing a kind the caller never previewed.
        """

        receipt, _plan = await submit_facade_product(self.config, "clear_corrections", session_id=session_id)
        return int(receipt.affected_count)

    async def post_blackboard_note(
        self,
        *,
        kind: str,
        title: str,
        content: str,
        scope_repo: str | None = None,
        scope_session: str | None = None,
        scope_issue: int | None = None,
        scope_path: str | None = None,
        related_sessions: tuple[str, ...] = (),
        author_ref: str | None = None,
        author_kind: str = "user",
        evidence_refs: tuple[str, ...] = (),
        staleness: dict[str, object] | None = None,
        context_policy: dict[str, object] | None = None,
    ) -> BlackboardNote:
        """Post a note to the persistent agent blackboard (#1697).

        ``kind`` must be one of :data:`BLACKBOARD_KINDS`; raises ``ValueError``
        otherwise. The structured fields are encoded into the stored body and a
        fresh note id is allocated, so each call appends a distinct note. The
        optional assertion metadata fields are mirrored only into the unified
        assertion row (#1839/#1883), preserving the legacy blackboard row shape.

        Routed through ``OperationExecutor``/``BlackboardPostActuator`` (t46.9
        phase 6); see :meth:`add_mark` for the shared-contract rationale. The
        note id is minted here (not inside the actuator) so the plan hash the
        executor revalidates at EXECUTE time is stable across both PREPARE
        calls.
        """
        from polylogue.archive.blackboard import BLACKBOARD_KINDS

        if kind not in BLACKBOARD_KINDS:
            raise ValueError(f"kind must be one of {list(BLACKBOARD_KINDS)}, got {kind!r}")
        receipt, _plan = await submit_facade_product(
            self.config,
            "post_blackboard_note",
            kind=kind,
            title=title,
            content=content,
            scope_repo=scope_repo,
            scope_session=scope_session,
            scope_issue=scope_issue,
            scope_path=scope_path,
            related_sessions=related_sessions,
            author_ref=author_ref,
            author_kind=author_kind,
            evidence_refs=evidence_refs,
            staleness=staleness,
            context_policy=context_policy,
        )
        from polylogue.archive.blackboard import decode_blackboard_note

        domain_receipt = receipt.domain_receipt
        return decode_blackboard_note(
            note_id=str(domain_receipt["note_id"]),
            body=str(domain_receipt["body"]),
            target_type=cast("str | None", domain_receipt["target_type"]),
            target_id=cast("str | None", domain_receipt["target_id"]),
            created_at_ms=int(cast("int", domain_receipt["created_at_ms"])),
            updated_at_ms=int(cast("int", domain_receipt["updated_at_ms"])),
        )

    async def list_blackboard_notes(
        self,
        *,
        kind: str | None = None,
        scope_repo: str | None = None,
        unresolved: bool = False,
        limit: int = 20,
    ) -> list[BlackboardNote]:
        """Return the requested prefix of the canonical filtered blackboard page."""
        return list(
            (
                await self.read_blackboard_page(kind=kind, scope_repo=scope_repo, unresolved=unresolved, limit=limit)
            ).items
        )

    async def read_blackboard_page(
        self,
        *,
        kind: str | None = None,
        scope_repo: str | None = None,
        unresolved: bool = False,
        limit: int = 20,
        offset: int = 0,
    ) -> BlackboardPage:
        from polylogue.archive.blackboard import BlackboardPage, decode_blackboard_note

        if limit <= 0 or offset < 0:
            raise ValueError("blackboard page requires a positive limit and nonnegative offset")
        envelopes, total = await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.blackboard.list",
            arguments={"kind": kind, "scope_repo": scope_repo, "unresolved": unresolved},
            work=lambda archive: archive.read_blackboard_page(
                kind=kind, scope_repo=scope_repo, unresolved=unresolved, limit=limit, offset=offset
            ),
            page_size=limit,
            offset=offset,
            projection="blackboard-notes",
            stable_order="updated_at:desc,note_id",
        )
        return BlackboardPage(
            items=tuple(
                decode_blackboard_note(
                    note_id=envelope.note_id,
                    body=envelope.body,
                    target_type=envelope.target_type,
                    target_id=envelope.target_id,
                    created_at_ms=envelope.created_at_ms,
                    updated_at_ms=envelope.updated_at_ms,
                )
                for envelope in envelopes
            ),
            total=total,
            limit=limit,
            offset=offset,
        )

    async def get_setting(self, setting_key: str) -> ArchiveUserSettingEnvelope | None:
        """Read one durable ``user_settings`` row, or ``None`` when unset (polylogue-at44).

        This is the liveness slice: a closed, typed registry of setting
        keys (``subscription_tier`` today), not a free-form key-value store.
        The full scope x actor x override resolver belongs to the w8db epic.
        """

        return _archive_get_setting(self.config, setting_key)

    async def list_settings(self) -> list[ArchiveUserSettingEnvelope]:
        """List every stored ``user_settings`` row, ordered by key."""

        return _archive_list_settings(self.config)

    # ``set_setting`` is deliberately absent: ``user_settings`` is a durable
    # ``user.db`` tier and the daemon is its sole writer, so the write is the
    # declared ``mutation.user.setting.set`` operation rather than a facade
    # method that opens a writable store in whatever process holds the surface
    # (polylogue-gjwto / polylogue-r29bv). The reads above stay here.
