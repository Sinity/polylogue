"""Archive/query domain methods for the async Polylogue facade."""

from __future__ import annotations

import builtins
import json
import logging
import sqlite3
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import closing, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from polylogue.archive.actions.actions import Action
from polylogue.archive.attachment.models import Attachment
from polylogue.archive.blackboard import BlackboardNote
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import MessageRoleFilter, Role
from polylogue.archive.message.types import MessageType, validate_message_type_filter
from polylogue.archive.query.predicate import QueryFieldPredicate, QueryFieldRef
from polylogue.archive.query.spec import normalize_action_sequence, normalize_action_terms, parse_query_date
from polylogue.archive.query.transaction import archive_read_context, run_archive_read
from polylogue.archive.semantic.content_projection import ContentProjectionSpec, project_message_content
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.context.compiler import (
    DEFAULT_CONTEXT_IMAGE_MAX_CHARS_PER_MESSAGE,
    DEFAULT_CONTEXT_IMAGE_MAX_MESSAGES_PER_SESSION,
)
from polylogue.core.enums import AssertionKind, AssertionStatus, MaterialOrigin, Origin
from polylogue.core.errors import PolylogueError
from polylogue.core.json import JSONDocument
from polylogue.core.refs import EvidenceRef, ObjectRef, parse_delegation_edge_object_id, parse_public_ref
from polylogue.core.timestamps import parse_archive_datetime
from polylogue.core.types import SessionId
from polylogue.core.user_state_targets import TARGET_MESSAGE, TARGET_SESSION
from polylogue.insights.archive import (
    SessionProfileInsight,
    SessionProfileInsightQuery,
)
from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.insights.feedback import LearningCorrection, parse_correction_kind
from polylogue.paths import archive_file_set_index_available_for_paths, archive_file_set_root_for_paths
from polylogue.storage.insights.session.records import SessionProfileRecord
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.query_models import SessionRecordQuery
from polylogue.storage.search.models import SearchHit, SearchResult
from polylogue.storage.search.query_builders import session_web_url
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary, IndexStatus
from polylogue.storage.sqlite.archive_tiers.context_delivery_write import ArchiveContextDeliveryEnvelope
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveAttachmentRow,
    ArchiveMessageRow,
    ArchiveSessionEnvelope,
)
from polylogue.storage.sqlite.queries.message_query_reads import MessageTypeName
from polylogue.surfaces.chronicle import (
    ChronicleProjectionPayload,
    ChronicleSessionPayload,
    build_chronicle_projection_payload,
    build_chronicle_session_payload,
)
from polylogue.surfaces.temporal_evidence import (
    TemporalEvidenceEvent,
    TemporalEvidenceWindow,
    action_row_to_temporal_event,
    build_temporal_evidence_window,
    message_row_to_temporal_event,
    summary_to_temporal_event,
)

if TYPE_CHECKING:
    from polylogue.annotations.importer import AnnotationBatchImportRequest, AnnotationBatchImportResult
    from polylogue.annotations.join import AnnotationStructuralJoinResult
    from polylogue.annotations.schema import AnnotationSchemaRegistry
    from polylogue.api import Polylogue
    from polylogue.archive.filter.filters import SessionFilter
    from polylogue.archive.message.models import Message
    from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics
    from polylogue.archive.query.search_hits import SessionSearchHit
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.archive.session.neighbor_candidates import SessionNeighborCandidate
    from polylogue.archive.stats import ArchiveStats as StorageArchiveStats
    from polylogue.config import Config
    from polylogue.context.compiler import ContextImage, ContextOmission, ContextSpec
    from polylogue.context.hermes_delivery_correlation import HermesContextDeliveryCorrelation
    from polylogue.core.protocols import ProgressCallback
    from polylogue.insights.audit import InsightRigorAuditQuery, InsightRigorAuditReport
    from polylogue.insights.export_bundles import InsightExportBundleRequest, InsightExportBundleResult
    from polylogue.insights.pathology import PathologyReport
    from polylogue.insights.portfolio import PortfolioBundle
    from polylogue.insights.postmortem import PostmortemBundle
    from polylogue.insights.readiness import InsightReadinessQuery, InsightReadinessReport
    from polylogue.insights.resume import ResumeBrief, ResumeCandidate
    from polylogue.insights.transforms import SessionDigest
    from polylogue.operations import ArchiveStats
    from polylogue.readiness import ReadinessReport
    from polylogue.sources.parsers.hermes_lifecycle import HermesLifecycleReconciliation
    from polylogue.storage.insights.session.runtime import SessionInsightCounts
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.search.models import SearchResult
    from polylogue.storage.sqlite.archive_tiers.archive import (
        ArchiveSessionSearchHit,
        ArchiveSessionSummary,
        ArchiveStore,
    )
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        ArchiveAssertionBulkJudgmentEnvelope,
        ArchiveAssertionCandidateReviewEnvelope,
        ArchiveAssertionEnvelope,
    )
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveSessionEnvelope
    from polylogue.storage.usage import ProviderUsageReport
    from polylogue.surfaces.payloads import (
        ArchiveDebtListPayload,
        AssertionBulkJudgmentPayload,
        AssertionCandidateReviewListPayload,
        AssertionClaimPayload,
        AssertionJudgmentResultPayload,
        BulkTagMutationResult,
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

_BOUNDED_MESSAGES_FALLBACK_READ_VIEWS = frozenset({"raw", "context", "neighbors", "correlation", "chronicle"})

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

_CANDIDATE_CAPTURE_KIND_MAP: dict[str, AssertionKind] = {
    "note": AssertionKind.NOTE,
    "claim": AssertionKind.DECISION,
    "correction": AssertionKind.CORRECTION,
    "lesson": AssertionKind.LESSON,
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


logger = logging.getLogger(__name__)


class SessionNotFoundError(PolylogueError):
    """Raised when a requested session does not exist in the archive."""

    http_status_code = 404


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


def _resolution_action(label: str, command: str | None = None, href: str | None = None) -> Any:
    from polylogue.surfaces.payloads import RefResolutionActionPayload

    return RefResolutionActionPayload(label=label, command=command, href=href)


def _unresolved_ref_payload(
    ref: str, message: str, *, normalized_ref: str | None = None, kind: str | None = None
) -> Any:
    from polylogue.surfaces.payloads import PublicRefResolutionPayload

    return PublicRefResolutionPayload(
        ref=ref,
        normalized_ref=normalized_ref,
        kind=kind,
        resolved=False,
        caveats=(message,),
    )


def _oversized_annotation_batch_ref_payload(ref: str) -> Any | None:
    """Return a bounded unresolved descriptor for an oversized batch-like ref."""

    if not ref.startswith("annotation-batch"):
        return None
    from polylogue.surfaces.payloads import (
        AnnotationBatchRefDigestPayload,
        PublicRefResolutionPayload,
        model_json_document,
    )

    try:
        descriptor = AnnotationBatchRefDigestPayload.from_oversized_ref(ref)
    except (UnicodeEncodeError, ValueError):
        return None
    return PublicRefResolutionPayload(
        ref=f"annotation-batch:sha256-{descriptor.original_ref_sha256}",
        normalized_ref=None,
        kind="annotation-batch",
        resolved=False,
        payload_kind="annotation-batch-ref-digest",
        payload=model_json_document(descriptor),
        caveats=("oversized annotation batch reference omitted from the public response",),
    )


def _invalid_unicode_ref_payload(ref: str) -> Any | None:
    """Fail closed before an invalid Python string reaches parsing or SQLite."""

    from polylogue.surfaces.payloads import (
        InvalidUnicodeRefDigestPayload,
        PublicRefResolutionPayload,
        model_json_document,
    )

    try:
        descriptor = InvalidUnicodeRefDigestPayload.from_invalid_ref(ref)
    except ValueError:
        return None
    batch_like = ref.startswith("annotation-batch")
    stable_prefix = "annotation-batch:invalid-unicode" if batch_like else "invalid-unicode-ref"
    return PublicRefResolutionPayload(
        ref=f"{stable_prefix}:sha256-{descriptor.original_ref_surrogatepass_sha256}",
        normalized_ref=None,
        kind="annotation-batch" if batch_like else None,
        resolved=False,
        payload_kind="invalid-unicode-ref-digest",
        payload=model_json_document(descriptor),
        caveats=("invalid Unicode public reference omitted from the response",),
    )


#: ObjectRefKind values registered ahead of their backing storage tier
#: (polylogue-rxdo analysis-provenance epic). ``resolve_ref`` returns a typed
#: ``PendingObjectRefPayload`` (reason=substrate-pending) for these instead of
#: attempting a lookup against tables that do not exist yet.
_PENDING_OBJECT_REF_KINDS: frozenset[str] = frozenset({"query", "query-run", "result-set", "cohort", "analysis"})


def _pending_ref_payload(ref: str, normalized_ref: str, kind: str) -> Any:
    from polylogue.surfaces.payloads import PendingObjectRefPayload, PublicRefResolutionPayload, model_json_document

    return PublicRefResolutionPayload(
        ref=ref,
        normalized_ref=normalized_ref,
        kind=kind,
        resolved=False,
        payload_kind="pending",
        payload=model_json_document(PendingObjectRefPayload(kind=kind)),
        caveats=(f"{kind} substrate is not implemented yet (reason=substrate-pending)",),
    )


def _archive_action_sequence(values: Sequence[str]) -> tuple[str, ...]:
    return normalize_action_sequence("action_sequence", ",".join(values))


def _archive_index_available(config: Config) -> bool:
    return archive_file_set_index_available_for_paths(archive_root_path=config.archive_root, db_anchor=config.db_path)


def _active_archive_root(config: Config) -> Path:
    return archive_file_set_root_for_paths(archive_root_path=config.archive_root, db_anchor=config.db_path)


def _archive_context_message_window(
    messages: Sequence[Message],
    *,
    anchor_message_id: str | None,
    max_messages: int | None,
    max_chars_per_message: int | None,
    max_tokens: int | None = None,
) -> tuple[tuple[tuple[str, str], ...], int, int, int]:
    """Return normalized message rows plus omission/clipping counts for context."""

    rows: list[tuple[str, str, str]] = []
    clipped_messages = 0
    for message in messages:
        if not message.text:
            continue
        text = message.text
        if max_chars_per_message is not None and len(text) > max_chars_per_message:
            omitted_chars = len(text) - max_chars_per_message
            text = text[:max_chars_per_message].rstrip() + f"\n\n... {omitted_chars} chars omitted from this message."
            clipped_messages += 1
        rows.append(
            (
                str(message.id),
                str(getattr(message.role, "value", message.role)),
                text,
            )
        )
    if max_messages is None or len(rows) <= max_messages:
        window = rows
        omitted_before = 0
        omitted_after = 0
        if max_tokens is not None:
            window, budget_omitted_before, budget_clipped = _budget_context_message_window(window, max_tokens)
            omitted_before += budget_omitted_before
            clipped_messages += budget_clipped
        return (
            tuple((role, text) for _message_id, role, text in window),
            omitted_before,
            omitted_after,
            clipped_messages,
        )

    anchor_index = 0
    if anchor_message_id is not None:
        for index, (message_id, _role, _text) in enumerate(rows):
            if message_id == anchor_message_id:
                anchor_index = index
                break
    half_window = max_messages // 2
    start = max(0, anchor_index - half_window)
    start = min(start, max(0, len(rows) - max_messages))
    end = start + max_messages
    window = rows[start:end]
    omitted_before = start
    omitted_after = max(0, len(rows) - end)
    if max_tokens is not None:
        window, budget_omitted_before, budget_clipped = _budget_context_message_window(window, max_tokens)
        omitted_before += budget_omitted_before
        clipped_messages += budget_clipped
    return (
        tuple((role, text) for _message_id, role, text in window),
        omitted_before,
        omitted_after,
        clipped_messages,
    )


def _budget_context_message_window(
    rows: Sequence[tuple[str, str, str]],
    max_tokens: int,
) -> tuple[list[tuple[str, str, str]], int, int]:
    """Return a tail-biased message window that fits a small token budget."""

    if not rows:
        return [], 0, 0
    remaining = max(1, max_tokens - 48)
    selected: list[tuple[str, str, str]] = []
    clipped_messages = 0
    for message_id, role, text in reversed(rows):
        message_tokens = _context_message_token_estimate(role, text)
        if message_tokens <= remaining:
            selected.append((message_id, role, text))
            remaining -= message_tokens
            if remaining <= 0:
                break
            continue
        if not selected and remaining > 0:
            clipped_text = _clip_text_to_token_budget(text, remaining)
            if clipped_text:
                selected.append((message_id, role, clipped_text))
                clipped_messages += 1
            break
    if not selected:
        message_id, role, text = rows[-1]
        selected.append((message_id, role, _clip_text_to_token_budget(text, 1) or text[:1]))
        clipped_messages += 1
    selected.reverse()
    first_selected_id = selected[0][0]
    selected_start = next((index for index, row in enumerate(rows) if row[0] == first_selected_id), len(rows))
    return selected, selected_start, clipped_messages


def _context_message_token_estimate(role: str, text: str) -> int:
    return max(1, len(role.split()) + len(text.split()) + 1)


def _clip_text_to_token_budget(text: str, max_tokens: int) -> str:
    words = text.split()
    if not words:
        return ""
    if len(words) <= max_tokens:
        return text
    kept = max(1, max_tokens)
    omitted = len(words) - kept
    return " ".join(words[:kept]).rstrip() + f"\n\n... {omitted} words omitted from this message."


def _dedupe_object_refs(refs: Iterable[ObjectRef]) -> tuple[ObjectRef, ...]:
    deduped: list[ObjectRef] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        key = (ref.kind, ref.object_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return tuple(deduped)


def _dedupe_evidence_refs(refs: Iterable[EvidenceRef]) -> tuple[EvidenceRef, ...]:
    deduped: list[EvidenceRef] = []
    seen: set[tuple[str, str | None, int | None]] = set()
    for ref in refs:
        key = (ref.session_id, ref.message_id, ref.block_index)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return tuple(deduped)


def _archive_context_session_predicate(session_id: str) -> QueryFieldPredicate:
    """Build a bound exact-session predicate without reparsing public DSL text."""

    return QueryFieldPredicate(field="session.id", values=(session_id,), op="=").with_field_ref(
        QueryFieldRef(scope="session", name="id", source_name="session.id")
    )


def _archive_context_temporal_window(config: Config, summary: SessionSummary) -> TemporalEvidenceWindow:
    """Build a bounded temporal context window for one selected session."""

    session_id = str(summary.id)
    message_limit = 8
    action_limit = 4
    events: list[TemporalEvidenceEvent] = []
    if session_event := summary_to_temporal_event(summary):
        events.append(session_event)
    caveats: list[str] = []
    with archive_read_context(
        _active_archive_root(config),
        operation="archive.context.temporal_window",
        arguments={"session_id": session_id},
        page_size=message_limit,
        projection="temporal-window",
        stable_order="time,message_id",
    ) as archive:
        message_rows = archive.query_messages(
            _archive_context_session_predicate(session_id),
            limit=message_limit,
            sort="time",
            sort_direction="asc",
        )
        action_rows = archive.query_session_actions([session_id], limit=action_limit, sort_direction="asc")
    events.extend(event for row in message_rows if (event := message_row_to_temporal_event(row)) is not None)
    events.extend(event for row in action_rows if (event := action_row_to_temporal_event(row)) is not None)
    if len(message_rows) >= message_limit and (summary.message_count or 0) > message_limit:
        caveats.append("message_events_capped")
    if len(action_rows) >= action_limit:
        caveats.append("action_events_capped")
    return build_temporal_evidence_window(events, caveats=caveats)


async def _archive_context_chronicle_payload(
    config: Config,
    summary: SessionSummary,
    *,
    edge_limit: int = 8,
) -> ChronicleProjectionPayload:
    """Build a bounded chronicle projection for one selected session."""

    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

    archive_root = _active_archive_root(config)
    backend = SQLiteBackend(db_path=archive_root / "index.db")
    session_payloads: list[ChronicleSessionPayload] = []
    try:
        first_messages, last_messages, total = await backend.get_message_edge_windows(
            str(summary.id),
            message_role=(Role.USER, Role.ASSISTANT),
            message_type="message",
            material_origin=(MaterialOrigin.HUMAN_AUTHORED, MaterialOrigin.ASSISTANT_AUTHORED),
            edge_limit=edge_limit * 5,
        )
        session_payloads.append(
            build_chronicle_session_payload(
                summary,
                first_messages=first_messages,
                last_messages=last_messages,
                total_matching_messages=total,
                edge_limit=edge_limit,
            )
        )
    finally:
        await backend.close()
    return build_chronicle_projection_payload(session_payloads, edge_limit=edge_limit)


def _archive_query_kwargs(spec: SessionQuerySpec, *, default_limit: int | None) -> dict[str, object]:
    limit = spec.limit if spec.limit is not None else default_limit
    kwargs: dict[str, object] = {
        "offset": spec.offset,
        "origins": spec.origins,
        "excluded_origins": spec.excluded_origins,
        "tags": spec.tags,
        "excluded_tags": spec.excluded_tags,
        "repo_names": spec.repo_names,
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
        "min_messages": spec.min_messages,
        "max_messages": spec.max_messages,
        "min_words": spec.min_words,
        "max_words": spec.max_words,
        "since_ms": _archive_query_date_ms("since", spec.since),
        "until_ms": _archive_query_date_ms("until", spec.until),
        "since_session_id": spec.since_session_id,
        "boolean_predicate": spec.boolean_predicate,
    }
    if limit is not None:
        kwargs["limit"] = limit
    if spec.sort is not None:
        kwargs["sort"] = spec.sort
    if spec.reverse:
        kwargs["reverse"] = True
    if spec.sample is not None:
        kwargs["sample"] = spec.sample
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
    if limit is not None:
        query_kwargs["limit"] = limit
    if offset is not None:
        query_kwargs["offset"] = offset
    if query_text is not None:
        query_kwargs.pop("sample", None)
        return [archive.read_summary(hit.session_id) for hit in archive.search_summaries(query_text, **query_kwargs)]
    return cast(list[ArchiveSessionSummary], archive.list_summaries(**query_kwargs))


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


def _archive_count_sessions_for_spec(archive: Any, spec: SessionQuerySpec) -> int:
    query_kwargs = _archive_query_kwargs(spec, default_limit=None)
    for key in ("limit", "offset", "sort", "reverse", "sample"):
        query_kwargs.pop(key, None)
    query_text = _archive_text_query(spec)
    if query_text is not None:
        return int(archive.count_search_sessions(query_text, **query_kwargs))
    return int(archive.count_sessions(**query_kwargs))


def _archive_facet_buckets(
    archive: Any,
    spec: SessionQuerySpec | None,
    *,
    include_deferred: bool = True,
) -> Any:
    from polylogue.archive.query.facets import FacetBuckets

    if spec is None:
        summaries = cast(list[ArchiveSessionSummary], archive.list_summaries(limit=1_000_000))
    else:
        summaries = _archive_list_summaries_for_spec(archive, spec, default_limit=1_000_000)
    origins: dict[str, int] = {}
    tags: dict[str, int] = {}
    total_messages = 0
    session_ids: list[str] = []
    seen_session_ids: set[str] = set()
    for summary in summaries:
        if summary.session_id in seen_session_ids:
            continue
        seen_session_ids.add(summary.session_id)
        session_ids.append(summary.session_id)
        total_messages += summary.message_count
        origins[summary.origin] = origins.get(summary.origin, 0) + 1
        for tag in set(summary.tags):
            tags[tag] = tags.get(tag, 0) + 1
    sql_buckets = (
        _archive_aggregate_facet_families(
            archive._conn,
            session_ids=session_ids if spec is not None else None,
        )
        if include_deferred
        else {
            "repos": {},
            "role_counts": {},
            "material_origins": {},
            "message_types": {},
            "action_types": {},
            "has_flags": {},
            "omitted": {},
        }
    )
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
        total_sessions=len(session_ids),
        total_messages=total_messages,
    )


def _archive_aggregate_facet_families(
    conn: Any,
    *,
    session_ids: list[str] | None,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {
        "repos": {},
        "role_counts": {},
        "material_origins": {},
        "message_types": {},
        "action_types": {},
        "has_flags": {},
        "omitted": {},
    }
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
        return {str(row[0]): int(row[1] or 0) for row in rows if row[0]}

    def table_has_column(table: str, column: str) -> bool:
        try:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        except sqlite3.Error:
            return False
        return any(str(row[1]) == column for row in rows)

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
    if table_has_column("messages", "role"):
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
    if table_has_column("messages", "material_origin"):
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

    repo = _clean_repo_label(repo_name)
    if repo and not _is_noisy_repo_label(repo):
        return repo
    url_label = _repo_label_from_url(origin_url)
    if url_label and not _is_noisy_repo_label(url_label):
        return url_label
    root = _clean_repo_label(root_path)
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
            insights_ready = (
                insight_status.profile_rows_ready
                and insight_status.work_event_inference_rows_ready
                and insight_status.phase_rows_ready
                and insight_status.threads_ready
            )
            checks.append(
                ReadinessCheck(
                    "archive_session_insights",
                    VerifyStatus.OK if insights_ready else VerifyStatus.WARNING,
                    count=insight_status.profile_row_count,
                    summary=(
                        "session insight rows ready"
                        if insights_ready
                        else "session insight rows missing or stale; run rebuild_insights"
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
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = conn.execute("PRAGMA user_version").fetchone()
            version = int(row[0] or 0) if row is not None else 0
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return ReadinessCheck(name, VerifyStatus.ERROR, summary=str(exc))

    expected = ARCHIVE_VERSION_BY_TIER[tier]
    return ReadinessCheck(
        name,
        VerifyStatus.OK if version == expected else VerifyStatus.ERROR,
        summary=f"v{version}/{expected}: {path}",
    )


def _archive_list_assertion_claims(
    config: Config,
    *,
    kinds: Sequence[str | AssertionKind] | None = None,
    target_ref: str | None = None,
    scope_ref: str | None = None,
    statuses: Sequence[str | AssertionStatus] | None = ("active", "candidate"),
    context_inject: bool | None = None,
    limit: int | None = None,
) -> list[Any]:
    """Return assertion-backed lifecycle claims from ``user.db``."""

    from polylogue.storage.sqlite.archive_tiers.user_write import list_assertion_claims

    user_db = _active_archive_root(config) / "user.db"
    if not user_db.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            if kinds is None:
                return list_assertion_claims(
                    conn,
                    target_ref=target_ref,
                    scope_ref=scope_ref,
                    statuses=statuses,
                    context_inject=context_inject,
                    limit=limit,
                )
            return list_assertion_claims(
                conn,
                kinds=kinds,
                target_ref=target_ref,
                scope_ref=scope_ref,
                statuses=statuses,
                context_inject=context_inject,
                limit=limit,
            )
        finally:
            conn.close()
    except sqlite3.Error:
        return []


def _archive_get_context_delivery(
    config: Config,
    *,
    snapshot_ref: str,
    recipient_ref: str,
) -> ArchiveContextDeliveryEnvelope | None:
    """Read one delivery receipt only when it belongs to its recorded recipient."""

    from polylogue.storage.sqlite.archive_tiers.context_delivery_write import read_context_delivery

    user_db = _active_archive_root(config) / "user.db"
    if not user_db.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            receipt = read_context_delivery(conn, snapshot_ref)
            return receipt if receipt is not None and receipt.recipient_ref == recipient_ref else None
        finally:
            conn.close()
    except (sqlite3.Error, ValueError):
        return None


def _archive_correlate_hermes_context_deliveries(
    config: Config,
    *,
    hermes_session_native_id: str,
) -> tuple[HermesContextDeliveryCorrelation, ...]:
    """Correlate a Hermes session's drained ``context_injected`` events with their receipts.

    Read-only audit seam over two durable tiers (fs1.7 spool + fs1.11
    delivery ledger); see ``context.hermes_delivery_correlation`` for the
    join semantics. Returns an empty tuple, never raises, when either tier is
    unavailable -- consistent with the explicit-unavailable-state AC this
    correlation exists to satisfy. "Archive not yet initialized" (no
    source.db/user.db file at all) and "archive present but the read failed"
    (corrupt file, missing table, decode failure) both still return the same
    empty-tuple shape to the caller -- this facade method's contract predates
    this fix and changing its return type is a separate, larger decision --
    but the two cases are distinguished in the logs: only the second case
    logs a warning, so an operator/on-call scan for "hermes_context_deliveries
    read failed" is never confused with the ordinary "nothing ingested yet"
    path (review finding: these were previously collapsed into total silence).
    """

    from polylogue.context.hermes_delivery_correlation import correlate_hermes_context_deliveries

    archive_root = _active_archive_root(config)
    source_db = archive_root / "source.db"
    user_db = archive_root / "user.db"
    if not source_db.exists() or not user_db.exists():
        return ()
    try:
        source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        source_conn.row_factory = sqlite3.Row
        try:
            user_conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
            user_conn.row_factory = sqlite3.Row
            try:
                return correlate_hermes_context_deliveries(
                    source_conn,
                    user_conn,
                    hermes_session_native_id=hermes_session_native_id,
                )
            finally:
                user_conn.close()
        finally:
            source_conn.close()
    except (sqlite3.Error, ValueError):
        logger.warning(
            "hermes_context_deliveries read failed (archive present but unreadable): "
            "hermes_session_native_id=%s source_db=%s user_db=%s",
            hermes_session_native_id,
            source_db,
            user_db,
            exc_info=True,
        )
        return ()


def _archive_reconcile_hermes_session_lifecycle(
    config: Config,
    *,
    hermes_session_native_id: str,
) -> HermesLifecycleReconciliation | None:
    """Reconcile a Hermes session's drained lifecycle-event stream (fs1.7 AC).

    Read-only audit seam over two durable tiers (source.db lifecycle spool +
    index.db ingested snapshot); see
    ``context.hermes_lifecycle_reconciliation`` for the join semantics.
    Returns ``None``, never raises, when either tier is unavailable -- the
    caller distinguishes "not available yet" from "reconciled, zero events
    observed" (``total_events == 0``). "Archive not yet initialized" (no
    source.db/index.db file at all) and "archive present but the read
    failed" (corrupt file, missing table) both still return ``None`` to the
    caller -- this facade method's contract predates this fix and changing
    its return type is a separate, larger decision -- but the two cases are
    distinguished in the logs: only the second case logs a warning (mirrors
    ``_archive_correlate_hermes_context_deliveries``'s same review fix).
    """

    from polylogue.context.hermes_lifecycle_reconciliation import reconcile_hermes_session_lifecycle

    archive_root = _active_archive_root(config)
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        return None
    try:
        source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        source_conn.row_factory = sqlite3.Row
        try:
            index_conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
            index_conn.row_factory = sqlite3.Row
            try:
                return reconcile_hermes_session_lifecycle(
                    source_conn,
                    index_conn,
                    hermes_session_native_id=hermes_session_native_id,
                )
            finally:
                index_conn.close()
        finally:
            source_conn.close()
    except sqlite3.Error:
        logger.warning(
            "hermes_session_lifecycle reconciliation read failed (archive present but unreadable): "
            "hermes_session_native_id=%s source_db=%s index_db=%s",
            hermes_session_native_id,
            source_db,
            index_db,
            exc_info=True,
        )
        return None


def _archive_list_assertion_candidate_reviews(
    config: Config,
    *,
    target_ref: str | None = None,
    kinds: Sequence[str | AssertionKind] | None = None,
    statuses: Sequence[str | AssertionStatus] | None = None,
    limit: int | None = None,
) -> list[Any]:
    """Return candidate-review rows from ``user.db`` without active claims."""

    from polylogue.storage.sqlite.archive_tiers.user_write import list_assertion_candidate_reviews

    user_db = _active_archive_root(config) / "user.db"
    if not user_db.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            return list_assertion_candidate_reviews(
                conn,
                target_ref=target_ref,
                kinds=kinds,
                statuses=statuses,
                limit=limit,
            )
        finally:
            conn.close()
    except sqlite3.Error:
        return []


def _archive_judge_assertion_candidate(
    config: Config,
    *,
    candidate_ref: str,
    decision: str,
    reason: str | None = None,
    actor_ref: str = "user:local",
    inject: bool = False,
    replacement_kind: str | None = None,
    replacement_body_text: str | None = None,
    replacement_value: object | None = None,
) -> Any:
    """Write an assertion-candidate judgment to ``user.db``."""

    from polylogue.storage.sqlite.archive_tiers.user_write import judge_assertion_candidate

    user_db = _active_archive_root(config) / "user.db"
    if not user_db.exists():
        raise ValueError("assertion user tier is not initialized")
    try:
        conn = sqlite3.connect(user_db)
        conn.row_factory = sqlite3.Row
        try:
            result = judge_assertion_candidate(
                conn,
                candidate_ref=candidate_ref,
                decision=decision,
                reason=reason,
                actor_ref=actor_ref,
                inject=inject,
                replacement_kind=replacement_kind,
                replacement_body_text=replacement_body_text,
                replacement_value=replacement_value,
            )
            conn.commit()
            return result
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"failed to judge assertion candidate: {exc}") from exc


def _archive_capture_assertion_candidate(
    config: Config,
    *,
    body_text: str,
    kind: AssertionKind,
    refs: Sequence[str] = (),
    scope_refs: Sequence[str] = (),
    cwd: Path | None = None,
    author_ref: str = "user:local",
    author_kind: str = "user",
) -> Any:
    """Write one terminal-captured assertion through the user-tier gate."""

    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

    normalized_body = body_text.strip()
    if not normalized_body:
        raise ValueError("note text cannot be empty")

    assertion_id = f"assertion-terminal-note:{uuid.uuid4()}"
    resolved_refs: list[str] = []
    with ArchiveStore.open_existing(_active_archive_root(config), read_only=False) as archive:
        for ref in refs:
            if ref == "last":
                resolved_cwd = (cwd or Path.cwd()).resolve()
                repo_root = next(
                    (candidate for candidate in (resolved_cwd, *resolved_cwd.parents) if (candidate / ".git").exists()),
                    resolved_cwd,
                )
                summaries = archive.list_summaries(cwd_prefix=str(repo_root), limit=1)
                if not summaries:
                    raise ValueError("--ref last found no archived session for the current repository/cwd")
                session_ref = f"session:{summaries[0].session_id}"
                resolved_refs.append(session_ref)
                continue
            parsed = ObjectRef.parse(ref)
            if parsed.kind != "session":
                raise ValueError("--ref must be a session:<id> ref or 'last'")
            try:
                session_id = archive.resolve_session_id(parsed.object_id)
            except KeyError:
                raise ValueError(f"session ref not found: {parsed.object_id}") from None
            resolved_refs.append(f"session:{session_id}")

        normalized_scope_refs = [parse_public_ref(ref).format() for ref in scope_refs]
        target_ref = resolved_refs[0] if resolved_refs else f"assertion:{assertion_id}"
        user_db = archive.user_db_path

    try:
        conn = sqlite3.connect(user_db)
        conn.row_factory = sqlite3.Row
        try:
            envelope = upsert_assertion(
                conn,
                assertion_id=assertion_id,
                target_ref=target_ref,
                scope_ref=normalized_scope_refs[0] if normalized_scope_refs else None,
                kind=kind,
                key="terminal-note",
                value={
                    "capture_surface": "terminal",
                    "scope_refs": normalized_scope_refs,
                    "unanchored": not bool(resolved_refs),
                },
                body_text=normalized_body,
                author_ref=author_ref,
                author_kind=author_kind,
                evidence_refs=tuple(dict.fromkeys((*resolved_refs, *normalized_scope_refs))),
                status=AssertionStatus.CANDIDATE,
                context_policy={"inject": False, "promotion_required": True},
            )
            conn.commit()
            return envelope
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"failed to capture assertion candidate: {exc}") from exc


def _archive_judge_assertion_candidates(
    config: Config,
    *,
    items: Sequence[Any],
) -> Any:
    """Write an independently-recoverable bulk candidate judgment batch."""

    from polylogue.storage.sqlite.archive_tiers.user_write import judge_assertion_candidates

    user_db = _active_archive_root(config) / "user.db"
    if not user_db.exists():
        raise ValueError("assertion user tier is not initialized")
    try:
        conn = sqlite3.connect(user_db)
        conn.row_factory = sqlite3.Row
        try:
            result = judge_assertion_candidates(conn, items)
            conn.commit()
            return result
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"failed to judge assertion candidates: {exc}") from exc


def _archive_emit_pathology_assertions(
    config: Config,
    findings_by_session: Mapping[str, Sequence[Any]],
) -> int:
    """Upsert pathology findings as candidate assertions in ``user.db`` (#2383).

    Returns the number of candidate assertion rows written. Idempotent: the
    deterministic assertion id means re-emitting identical findings updates the
    same candidate rows rather than duplicating them.
    """

    from polylogue.storage.sqlite.archive_tiers.user_write import (
        upsert_pathology_findings_as_assertions,
    )

    if not findings_by_session:
        return 0
    user_db = _active_archive_root(config) / "user.db"
    if not user_db.exists():
        raise ValueError("assertion user tier is not initialized")
    emitted = 0
    try:
        conn = sqlite3.connect(user_db)
        conn.row_factory = sqlite3.Row
        try:
            for session_id, findings in findings_by_session.items():
                if not findings:
                    continue
                envelopes = upsert_pathology_findings_as_assertions(conn, session_id, list(findings))
                emitted += len(envelopes)
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"failed to emit pathology assertions: {exc}") from exc
    return emitted


def _archive_count_table_rows(conn: Any, table_name: str) -> int | None:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    if row is None:
        return None
    count_row = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(count_row[0] or 0) if count_row is not None else 0


def _maybe_parse_json_object(value: str | None) -> dict[str, object] | None:
    """Decode a stored JSON object column back into a mapping for domain blocks."""
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _archive_attachment_to_domain(attachment: ArchiveAttachmentRow) -> Attachment:
    return Attachment(
        id=attachment.attachment_id,
        name=attachment.display_name,
        mime_type=attachment.media_type,
        size_bytes=attachment.byte_count,
        path=None,
        source_url=attachment.source_url,
        caption=attachment.caption,
    )


def _archive_message_to_domain(message: ArchiveMessageRow, *, origin: Origin) -> Message:
    text = "\n\n".join(block.text for block in message.blocks if block.text) or None
    content_blocks: list[dict[str, object]] = [
        {
            key: value
            for key, value in {
                "id": block.block_id,
                "type": block.block_type,
                "text": block.text,
                "tool_name": block.tool_name,
                "tool_id": block.tool_id,
                "semantic_type": block.semantic_type,
                "tool_input": _maybe_parse_json_object(block.tool_input),
                "metadata": _maybe_parse_json_object(block.metadata),
                "tool_result_is_error": block.tool_result_is_error,
                "tool_result_exit_code": block.tool_result_exit_code,
            }.items()
            if value is not None
        }
        for block in message.blocks
    ]
    return Message(
        id=message.message_id,
        role=Role.normalize(message.role),
        text=text,
        timestamp=parse_archive_datetime(message.occurred_at),
        origin=origin,
        blocks=content_blocks,
        message_type=MessageType.normalize(message.message_type),
        material_origin=MaterialOrigin.normalize(message.material_origin),
        has_tool_use=message.has_tool_use,
        has_thinking=message.has_thinking,
        has_paste=message.has_paste,
        paste_boundary_state=message.paste_boundary_state,
        duration_ms=message.duration_ms,
        branch_index=message.variant_index,
        parent_id=message.parent_message_id,
        attachments=[_archive_attachment_to_domain(att) for att in message.attachments],
    )


def _archive_session_to_session(session: ArchiveSessionEnvelope) -> Session:
    origin = Origin.from_string(session.origin)
    messages = [_archive_message_to_domain(message, origin=origin) for message in session.messages]
    timestamps = [message.timestamp for message in messages if message.timestamp is not None]
    # Prefer the stored session timestamps (sessions.created_at_ms/updated_at_ms);
    # fall back to the message-timestamp envelope only when the session row has
    # none. The summary projection already uses the stored values, so this keeps
    # the full-read and summary-read session timelines consistent.
    stored_created = parse_archive_datetime(session.created_at)
    stored_updated = parse_archive_datetime(session.updated_at)
    return Session(
        id=SessionId(session.session_id),
        origin=origin,
        title=session.title,
        messages=MessageCollection(messages=messages),
        created_at=stored_created or (min(timestamps) if timestamps else None),
        updated_at=stored_updated or (max(timestamps) if timestamps else None),
        working_directories=tuple(session.working_directories),
        git_branch=session.git_branch,
        git_repository_url=session.git_repository_url,
        provider_project_ref=session.provider_project_ref,
        parent_id=SessionId(session.parent_session_id) if session.parent_session_id else None,
        branch_type=BranchType(session.branch_type) if session.branch_type else None,
        attachments=[_archive_attachment_to_domain(att) for att in session.orphan_attachments],
    )


def _archive_summary_to_domain(summary: ArchiveSessionSummary) -> SessionSummary:
    return SessionSummary(
        id=SessionId(summary.session_id),
        origin=Origin.from_string(summary.origin),
        title=summary.title,
        created_at=parse_archive_datetime(summary.created_at),
        updated_at=parse_archive_datetime(summary.updated_at),
        working_directories=tuple(summary.working_directories),
        git_branch=summary.git_branch,
        git_repository_url=summary.git_repository_url,
        provider_project_ref=summary.provider_project_ref,
        message_count=summary.message_count,
        tags_m2m=summary.tags,
    )


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
        SessionSummaryPayload,
        TargetRefPayload,
        reader_anchor,
        reader_message_actions,
    )

    return SessionSearchHitPayload(
        session=SessionSummaryPayload.from_summary(
            _archive_summary_to_domain(summary),
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


class _ArchiveInsightExportOperations:
    """Async operations adapter for registry-backed archive insight exports."""

    def __init__(self, archive: Any) -> None:
        self._archive = archive

    async def get_insight_readiness_report(self, query: object | None = None) -> InsightReadinessReport:
        from polylogue.insights.readiness import InsightReadinessQuery

        request = query if isinstance(query, InsightReadinessQuery) else None
        return cast("InsightReadinessReport", self._archive.insight_readiness_report(request))

    async def list_session_profile_insights(self, query: object) -> list[ArchiveInsightModel]:
        return list(
            self._archive.list_session_profile_insights(
                origin=str(origin) if (origin := getattr(query, "origin", None)) is not None else None,
                workflow_shape=getattr(query, "workflow_shape", None),
                terminal_state=getattr(query, "terminal_state", None),
                since_ms=_archive_query_date_ms("since", getattr(query, "since", None)),
                until_ms=_archive_query_date_ms("until", getattr(query, "until", None)),
                tier=str(getattr(query, "tier", "merged")),
                limit=getattr(query, "limit", None),
                offset=int(getattr(query, "offset", 0)),
            )
        )

    async def list_session_work_event_insights(self, query: object) -> list[ArchiveInsightModel]:
        return list(
            self._archive.list_session_work_event_insights(
                session_id=getattr(query, "session_id", None),
                origin=str(origin) if (origin := getattr(query, "origin", None)) is not None else None,
                heuristic_label=getattr(query, "heuristic_label", None),
                since_ms=_archive_query_date_ms("since", getattr(query, "since", None)),
                until_ms=_archive_query_date_ms("until", getattr(query, "until", None)),
                limit=getattr(query, "limit", None),
                offset=int(getattr(query, "offset", 0)),
            )
        )

    async def list_session_phase_insights(self, query: object) -> list[ArchiveInsightModel]:
        return list(
            self._archive.list_session_phase_insights(
                session_id=getattr(query, "session_id", None),
                origin=str(origin) if (origin := getattr(query, "origin", None)) is not None else None,
                kind=getattr(query, "kind", None),
                since_ms=_archive_query_date_ms("since", getattr(query, "since", None)),
                until_ms=_archive_query_date_ms("until", getattr(query, "until", None)),
                limit=getattr(query, "limit", None),
                offset=int(getattr(query, "offset", 0)),
            )
        )

    async def list_thread_insights(self, query: object) -> list[ArchiveInsightModel]:
        return list(
            self._archive.list_thread_insights(
                query=getattr(query, "query", None),
                since_ms=_archive_query_date_ms("since", getattr(query, "since", None)),
                until_ms=_archive_query_date_ms("until", getattr(query, "until", None)),
                limit=getattr(query, "limit", None),
                offset=int(getattr(query, "offset", 0)),
            )
        )

    async def list_session_tag_rollup_insights(self, query: object) -> list[ArchiveInsightModel]:
        return list(
            self._archive.list_session_tag_rollup_insights(
                origin=str(origin) if (origin := getattr(query, "origin", None)) is not None else None,
                query=getattr(query, "query", None),
                since_ms=_archive_query_date_ms("since", getattr(query, "since", None)),
                until_ms=_archive_query_date_ms("until", getattr(query, "until", None)),
                limit=getattr(query, "limit", None),
                offset=int(getattr(query, "offset", 0)),
            )
        )

    async def list_archive_coverage_insights(self, query: object) -> list[ArchiveInsightModel]:
        return list(
            self._archive.list_archive_coverage_insights(
                group_by=str(getattr(query, "group_by", "origin")),
                origin=str(origin) if (origin := getattr(query, "origin", None)) is not None else None,
                since_ms=_archive_query_date_ms("since", getattr(query, "since", None)),
                until_ms=_archive_query_date_ms("until", getattr(query, "until", None)),
                limit=getattr(query, "limit", None),
                offset=int(getattr(query, "offset", 0)),
            )
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
            return _archive_session_to_session(self._archive.read_session(resolved))
        except KeyError:
            return None

    async def list_summaries_by_query(self, query: SessionRecordQuery) -> builtins.list[SessionSummary]:
        origin, origins = self._origin_filters(
            origin=query.origin,
            origins=builtins.list(query.origins) if query.origins else [],
        )
        return [
            _archive_summary_to_domain(summary)
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
                summary = _archive_summary_to_domain(self._archive.read_summary(hit.session_id))
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

    Mirrors the ingest-time derivation (``pipeline/services/ingest_worker``):
    each message's content blocks are parsed into tool calls, then promoted
    to ``Action`` records. No storage round-trip — the domain
    session already carries the content blocks the actions are built from.
    """
    from polylogue.archive.actions.actions import build_actions, build_tool_calls_from_content_blocks

    events: builtins.list[Action] = []
    for message in session.messages:
        calls = build_tool_calls_from_content_blocks(
            origin=session.origin,
            content_blocks=message.blocks,
        )
        events.extend(build_actions(message, calls))
    return tuple(events)


def _rebuild_archive_session_insights(
    archive: Any,
    *,
    session_ids: Sequence[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SessionInsightCounts:
    """Rebuild durable session insights via the canonical materializer.

    This is a thin adapter over
    ``polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync``
    — the single rebuild stack shared with daemon convergence (#1743 P13). It
    resolves any session-id aliases against the archive, then delegates the
    whole rebuild (profiles, latency, work events, phases, threads +
    thread_sessions + 'thread' markers, tag rollups, provider-day aggregates)
    to the canonical path, which commits internally.
    """
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
    from polylogue.storage.insights.session.runtime import SessionInsightCounts

    resolved_ids = _archive_rebuild_session_ids(archive, session_ids) if session_ids is not None else None
    if session_ids is not None and not resolved_ids:
        return SessionInsightCounts()
    return rebuild_session_insights_sync(
        archive._conn,
        session_ids=resolved_ids,
        progress_callback=progress_callback,
    )


def _archive_rebuild_session_ids(archive: Any, session_ids: Sequence[str] | None) -> tuple[str, ...]:
    if session_ids is None:
        return tuple(summary.session_id for summary in archive.list_summaries(limit=1_000_000))
    resolved: list[str] = []
    for session_id in session_ids:
        with suppress(KeyError):
            resolved.append(archive.resolve_session_id(str(session_id)))
    return tuple(dict.fromkeys(resolved))


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


class PolylogueArchiveMixin:
    if TYPE_CHECKING:

        @property
        def config(self) -> Config: ...

        @property
        def repository(self) -> SessionRepository: ...

    async def import_annotation_batch(
        self,
        request: AnnotationBatchImportRequest,
        *,
        registry: AnnotationSchemaRegistry | None = None,
    ) -> AnnotationBatchImportResult:
        """Import bounded, provenance-stamped annotation candidates.

        This is the library binding for the shared annotation-import product
        operation used by the CLI and MCP surfaces. ``registry`` lets callers
        use a deliberately constructed schema registry without bypassing the
        facade.
        """
        from polylogue.annotations.importer import import_annotation_batch

        class _ActiveArchiveImportFacade:
            @property
            def archive_root(self) -> Path:
                return _active_archive_root(self._archive.config)

            def __init__(self, archive: PolylogueArchiveMixin) -> None:
                self._archive = archive

            async def resolve_ref(self, ref: str) -> PublicRefResolutionPayload:
                return await self._archive.resolve_ref(ref)

        import_facade = cast("Polylogue", _ActiveArchiveImportFacade(self))
        if registry is None:
            return await import_annotation_batch(import_facade, request)
        return await import_annotation_batch(import_facade, request, registry=registry)

    async def get_session(
        self,
        session_id: str,
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> Session | None:
        def read(archive: ArchiveStore) -> Session | None:
            try:
                resolved_id = archive.resolve_session_id(session_id)
            except KeyError:
                return None
            session = _archive_session_to_session(archive.read_session(resolved_id))
            if content_projection is None or not content_projection.filters_content():
                return session
            return session.with_content_projection(content_projection)

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.session.get",
            arguments={"session_id": session_id, "content_projection": content_projection},
            work=read,
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
        from polylogue.insights.transforms import compile_session_digest
        from polylogue.storage.query_models import SessionRecordQuery

        session = await self.get_session(session_id)
        if session is None:
            return None
        resolved_session_id = str(session.id)
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
        from polylogue.insights.postmortem import (
            PostmortemBundle as _PostmortemBundle,
        )
        from polylogue.insights.postmortem import (
            PostmortemScope,
            compile_postmortem_bundle,
        )

        cap = limit if limit is not None and limit > 0 else 200
        summaries = await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.postmortem.scope",
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
            logger.warning(
                "postmortem_bundle truncated: matched=%d cap=%d dropped=%d dropped_preview=%s",
                matched,
                cap,
                dropped,
                session_ids[cap : cap + 5],
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
        runs the deterministic detectors in :mod:`polylogue.insights.pathology`
        over the typed run projections. Returns the aggregate
        :class:`PathologyReport` (findings + per-kind distribution), the
        queryable distribution/summary view for #2383. The analysis cap defaults
        to 200 sessions.
        """
        from polylogue.insights.pathology import compile_pathology_report

        cap = limit if limit is not None and limit > 0 else 200
        summaries = await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.pathology.scope",
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
        analyzed_ids = session_ids[:cap]
        if matched > cap:
            logger.warning(
                "pathology_report truncated: matched=%d cap=%d dropped=%d dropped_preview=%s",
                matched,
                cap,
                matched - cap,
                session_ids[cap : cap + 5],
            )
        projections = []
        for sid in analyzed_ids:
            digest = await self._session_digest(sid)
            if digest is not None:
                projections.append(digest.run_projection)
        return compile_pathology_report(projections)

    async def materialize_pathology_assertions(
        self,
        spec: SessionQuerySpec | None = None,
        *,
        limit: int | None = None,
    ) -> int:
        """Emit pathology findings as candidate assertions queryable via #2006 (#2383).

        Runs the deterministic detectors over a matched session scope (the same
        path :meth:`pathology_report` uses) and upserts each finding as a private,
        non-injected ``AssertionKind.PATHOLOGY`` candidate in ``user.db``. The
        candidates are then queryable through the standard assertion-claims surface
        (``list_assertion_claims(kinds="pathology")``) and promotable through the
        existing accept/reject/defer lifecycle (Ref #2182) — nothing is
        auto-injected into agent context. Returns the number of candidate rows
        written. Idempotent by deterministic assertion id. The analysis cap
        defaults to 200 sessions.
        """
        from polylogue.insights.pathology import detect_session_pathologies

        cap = limit if limit is not None and limit > 0 else 200
        summaries = await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.pathology_assertions.scope",
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
        analyzed_ids = session_ids[:cap]
        if matched > cap:
            logger.warning(
                "materialize_pathology_assertions truncated: matched=%d cap=%d dropped=%d dropped_preview=%s",
                matched,
                cap,
                matched - cap,
                session_ids[cap : cap + 5],
            )
        findings_by_session: dict[str, list[Any]] = {}
        for sid in analyzed_ids:
            digest = await self._session_digest(sid)
            if digest is None:
                continue
            findings = detect_session_pathologies(digest.run_projection)
            if findings:
                findings_by_session[sid] = findings
        return _archive_emit_pathology_assertions(self.config, findings_by_session)

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
        from polylogue.insights.portfolio import (
            compile_portfolio_bundle,
        )
        from polylogue.insights.postmortem import PostmortemScope

        cap = limit if limit is not None and limit > 0 else 200
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
            logger.warning(
                "portfolio_bundle truncated: matched=%d cap=%d dropped=%d dropped_preview=%s",
                matched,
                cap,
                dropped,
                session_ids[cap : cap + 5],
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
        scope_ref: str | None = None,
        statuses: Sequence[str | AssertionStatus] | None = ("active", "candidate"),
        context_inject: bool | None = None,
        limit: int | None = None,
    ) -> list[ArchiveAssertionEnvelope]:
        """List assertion-backed lifecycle claims for read-surface consumers."""

        return cast(
            list["ArchiveAssertionEnvelope"],
            _archive_list_assertion_claims(
                self.config,
                kinds=kinds,
                target_ref=target_ref,
                scope_ref=scope_ref,
                statuses=statuses,
                context_inject=context_inject,
                limit=limit,
            ),
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

    async def correlate_hermes_context_deliveries(
        self, hermes_session_native_id: str
    ) -> tuple[HermesContextDeliveryCorrelation, ...]:
        """Correlate a Hermes session's ``context_injected`` events with delivery receipts.

        Read-only audit seam (fs1.11 x fs1.7): for every durable
        ``context_injected`` lifecycle event drained for this Hermes session,
        resolves the exact delivered context-image bytes, token budget, and
        rendered-token estimate from the existing delivery ledger. An event
        with no resolvable receipt (archive outage, or the write has not
        committed yet) is returned with ``available=False`` and an explicit
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

    async def list_assertion_claim_payloads(
        self,
        *,
        kinds: Sequence[str | AssertionKind] | None = None,
        target_ref: str | None = None,
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

        return await self.list_assertion_claim_payloads(
            kinds=kinds,
            target_ref=target_ref,
            statuses=(AssertionStatus.CANDIDATE,),
            limit=limit,
        )

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
        from polylogue.surfaces.payloads import AssertionCandidateReviewListPayload

        candidate_statuses = ASSERTION_CANDIDATE_REVIEW_STATUSES if statuses is None else statuses
        review_rows = cast(
            list["ArchiveAssertionCandidateReviewEnvelope"],
            _archive_list_assertion_candidate_reviews(
                self.config,
                target_ref=target_ref,
                kinds=kinds,
                statuses=candidate_statuses,
                limit=limit,
            ),
        )
        return AssertionCandidateReviewListPayload.from_envelopes(
            review_rows,
            limit=limit if limit is not None else len(review_rows),
            target_ref=target_ref,
            candidate_statuses=candidate_statuses,
        )

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

        from polylogue.surfaces.payloads import AssertionJudgmentResultPayload

        result = _archive_judge_assertion_candidate(
            self.config,
            candidate_ref=candidate_ref,
            decision=decision,
            reason=reason,
            actor_ref=actor_ref,
            inject=inject,
            replacement_kind=replacement_kind,
            replacement_body_text=replacement_body_text,
            replacement_value=replacement_value,
        )
        return AssertionJudgmentResultPayload.from_envelope(result)

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
    ) -> AssertionClaimPayload:
        """Capture a terminal assertion as a non-injected candidate for review."""

        from polylogue.surfaces.payloads import AssertionClaimPayload

        envelope = _archive_capture_assertion_candidate(
            self.config,
            body_text=body_text,
            kind=kind,
            refs=refs,
            scope_refs=scope_refs,
            cwd=cwd,
            author_ref=author_ref,
            author_kind=author_kind,
        )
        return AssertionClaimPayload.from_envelope(envelope)

    async def judge_assertion_candidates(
        self,
        *,
        items: Sequence[Any],
    ) -> AssertionBulkJudgmentPayload:
        """Apply a review batch with per-candidate partial-success outcomes."""

        from polylogue.surfaces.payloads import AssertionBulkJudgmentPayload

        result = _archive_judge_assertion_candidates(self.config, items=items)
        return AssertionBulkJudgmentPayload.from_envelope(cast("ArchiveAssertionBulkJudgmentEnvelope", result))

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

        from polylogue.annotations.join import (
            AnnotationStructuralJoinRequest,
            StructuralJoinArchive,
            join_typed_annotations,
        )

        request = AnnotationStructuralJoinRequest(
            schema_id=schema_id,
            schema_version=schema_version,
            statuses=tuple(AssertionStatus.from_string(status) for status in statuses),
            target_kind=target_kind,
            group_by=tuple(group_by),
            limit=limit,
            offset=offset,
        )
        return await join_typed_annotations(cast(StructuralJoinArchive, self), request)

    async def _compile_context_seed_query(
        self,
        spec: ContextSpec,
    ) -> tuple[list[str], dict[str, str], list[ContextOmission]]:
        """Resolve ContextSpec query/filter seed selection into session ids."""
        from polylogue.context.compiler import ContextOmission
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

    async def compile_context(self, spec: ContextSpec) -> ContextImage:
        """Compile a bounded context image from query/ref seeds and read views.

        This is the executable API boundary for ``ContextSpec``. It deliberately
        reuses existing query/read primitives and records unsupported or
        missing inputs as omissions instead of creating a parallel memory store.
        """
        from polylogue.context.compiler import (
            ContextImage,
            ContextOmission,
            ContextSegment,
            compile_assertion_context_segment,
            compile_chronicle_context_segment,
            compile_messages_context_segment,
            compile_query_unit_context_segment,
            compile_temporal_context_segment,
        )

        segments: list[ContextSegment] = []
        omitted: list[ContextOmission] = []
        requested_views = tuple(dict.fromkeys(spec.read_views))
        session_ids: list[str] = []
        seen_sessions: set[str] = set()
        message_anchor_by_session: dict[str, str] = {}

        for seed_ref in spec.seed_refs:
            if not seed_ref.startswith("session:"):
                omitted.append(
                    ContextOmission(
                        ref=seed_ref,
                        reason="unsupported",
                        detail="compile_context currently accepts session: refs as direct seeds",
                    )
                )
                continue
            session_id = seed_ref.removeprefix("session:")
            if session_id not in seen_sessions:
                seen_sessions.add(session_id)
                session_ids.append(session_id)

        query_session_ids, query_message_anchors, query_omissions = await self._compile_context_seed_query(spec)
        omitted.extend(query_omissions)
        for session_id in query_session_ids:
            if session_id not in seen_sessions:
                seen_sessions.add(session_id)
                session_ids.append(session_id)
        for session_id, message_id in query_message_anchors.items():
            message_anchor_by_session.setdefault(session_id, message_id)

        token_budget = spec.max_tokens
        token_total = 0

        def append_messages_segment(session_id: str, session: Session, view: str) -> bool:
            nonlocal token_total
            remaining_tokens = None if token_budget is None else max(1, token_budget - token_total)
            messages, omitted_before, omitted_after, clipped_messages = _archive_context_message_window(
                tuple(session.messages),
                anchor_message_id=message_anchor_by_session.get(session_id),
                max_messages=spec.max_messages_per_session,
                max_chars_per_message=spec.max_chars_per_message,
                max_tokens=remaining_tokens,
            )
            segment = compile_messages_context_segment(
                session_id=session_id,
                title=session.title,
                messages=messages,
                evidence_refs=(EvidenceRef(session_id=session_id),),
                omitted_before=omitted_before,
                omitted_after=omitted_after,
                clipped_messages=clipped_messages,
            )
            if token_budget is not None and token_total + segment.token_estimate > token_budget and not messages:
                omitted.append(
                    ContextOmission(
                        ref=f"session:{session_id}",
                        view=view,
                        reason="budget",
                        detail="segment exceeded the requested context token budget",
                    )
                )
                return False
            token_total += segment.token_estimate
            segments.append(segment)
            return True

        for expression in spec.unit_queries:
            try:
                envelope = await self.query_units(expression, limit=spec.unit_query_limit)
            except Exception as exc:
                omitted.append(
                    ContextOmission(
                        query=expression,
                        reason="unsupported",
                        detail=f"query-unit expression failed: {exc}",
                    )
                )
                continue
            segment = compile_query_unit_context_segment(envelope)
            if token_budget is not None and token_total + segment.token_estimate > token_budget:
                omitted.append(
                    ContextOmission(
                        query=expression,
                        view="query_unit",
                        reason="budget",
                        detail="query-unit segment exceeded the requested context token budget",
                    )
                )
                continue
            token_total += segment.token_estimate
            segments.append(segment)
        for session_id in session_ids:
            session = await self.get_session(session_id)
            summary = await self.get_session_summary(session_id)
            session_segment_start = len(segments)
            for view in requested_views:
                if view == "messages":
                    if session is None:
                        omitted.append(
                            ContextOmission(
                                ref=f"session:{session_id}",
                                view=view,
                                reason="not_found",
                                detail="session seed did not resolve to messages",
                            )
                        )
                        continue
                    append_messages_segment(session_id, session, view)
                    continue
                if view == "temporal":
                    if summary is None:
                        omitted.append(
                            ContextOmission(
                                ref=f"session:{session_id}",
                                view=view,
                                reason="not_found",
                                detail="session seed did not resolve to temporal evidence",
                            )
                        )
                        continue
                    window = _archive_context_temporal_window(self.config, summary)
                    segment = compile_temporal_context_segment(session_id=session_id, window=window)
                    if token_budget is not None and token_total + segment.token_estimate > token_budget:
                        omitted.append(
                            ContextOmission(
                                ref=f"session:{session_id}",
                                view=view,
                                reason="budget",
                                detail="segment exceeded the requested context token budget",
                            )
                        )
                        continue
                    token_total += segment.token_estimate
                    segments.append(segment)
                    continue
                if view == "chronicle":
                    if summary is None:
                        omitted.append(
                            ContextOmission(
                                ref=f"session:{session_id}",
                                view=view,
                                reason="not_found",
                                detail="session seed did not resolve to chronicle evidence",
                            )
                        )
                        continue
                    payload = await _archive_context_chronicle_payload(self.config, summary)
                    segment = compile_chronicle_context_segment(session_id=session_id, payload=payload)
                    if token_budget is not None and token_total + segment.token_estimate > token_budget:
                        omitted.append(
                            ContextOmission(
                                ref=f"session:{session_id}",
                                view=view,
                                reason="budget",
                                detail="segment exceeded the requested context token budget",
                            )
                        )
                        continue
                    token_total += segment.token_estimate
                    segments.append(segment)
                    continue
                omitted.append(
                    ContextOmission(
                        ref=f"session:{session_id}",
                        view=view,
                        reason="unsupported",
                        detail=(
                            "compile_context supports messages, temporal, chronicle read views "
                            "and explicit query-unit context"
                        ),
                    )
                )
            if (
                token_budget is not None
                and session is not None
                and len(segments) == session_segment_start
                and any(view in _BOUNDED_MESSAGES_FALLBACK_READ_VIEWS for view in requested_views)
            ):
                append_messages_segment(session_id, session, "messages")
            if spec.include_assertions:
                assertion_claims = await self.list_assertion_claim_payloads(
                    target_ref=f"session:{session_id}",
                    statuses=("active",),
                    context_inject=True,
                )
                for claim in assertion_claims:
                    segment = compile_assertion_context_segment(
                        assertion_id=claim.assertion_id,
                        kind=claim.kind,
                        body_text=claim.body_text,
                        target_ref=claim.target_ref,
                        evidence_ref_texts=claim.evidence_refs,
                    )
                    if token_budget is not None and token_total + segment.token_estimate > token_budget:
                        omitted.append(
                            ContextOmission(
                                ref=f"assertion:{claim.assertion_id}",
                                view="assertion",
                                reason="budget",
                                detail="assertion segment exceeded the requested context token budget",
                            )
                        )
                        continue
                    token_total += segment.token_estimate
                    segments.append(segment)

        object_refs = _dedupe_object_refs(ref for segment in segments for ref in segment.object_refs)
        evidence_refs = _dedupe_evidence_refs(ref for segment in segments for ref in segment.evidence_refs)
        assertion_refs = tuple(dict.fromkeys(ref for segment in segments for ref in segment.assertion_refs))
        caveats = tuple(dict.fromkeys(caveat for segment in segments for caveat in segment.caveats))

        return ContextImage(
            spec=spec,
            segments=tuple(segments),
            object_refs=object_refs,
            evidence_refs=evidence_refs,
            assertion_refs=assertion_refs,
            omitted=tuple(omitted),
            caveats=caveats,
            token_estimate=token_total,
        )

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
    ) -> ContextImage:
        """Compile a multi-session context image through ``compile_context``.

        This is a thin lens over the shared context engine, not a parallel
        assembler. Session selection runs through the query algebra (a seed ref
        or the context-image selection filters); compilation, token-budgeted
        accumulation, omission accounting, and assertion inclusion are all
        delegated to :meth:`compile_context`.
        """
        from polylogue.context.compiler import ContextSpec
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
    ) -> Any:
        """Build the shared typed context-preamble payload for one session."""
        from polylogue.context.preamble import build_context_preamble_payload

        return await build_context_preamble_payload(
            self,
            session_id=session_id,
            related_limit=related_limit,
            source_tool_calls={"context_preamble_payload": "polylogue-api"},
        )

    async def explain_query_expression(self, expression: str) -> JSONDocument:
        """Explain query DSL parsing, AST metadata, and lowering details."""
        from polylogue.archive.query.expression import explain_expression

        return cast(JSONDocument, explain_expression(expression).to_payload())

    async def query_units(
        self,
        expression: str,
        *,
        limit: int = 50,
        offset: int = 0,
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
    ) -> QueryUnitResultEnvelope:
        """Execute a terminal unit-source query."""
        from polylogue.archive.query.execution_control import (
            QueryExecutionContext,
            classify_unit_expression_workload,
            execute_archive_read,
        )
        from polylogue.archive.query.unit_results import query_unit_envelope, query_unit_request

        request = query_unit_request(
            expression=expression,
            limit=limit,
            offset=offset,
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
        ctx = QueryExecutionContext.create(
            query_text=expression,
            workload_class=classify_unit_expression_workload(expression),
            owner_ref="api.query_units",
        )
        return await execute_archive_read(
            _active_archive_root(self.config),
            lambda archive: query_unit_envelope(archive, request, execution_context=ctx),
            ctx=ctx,
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
        from polylogue.telemetry.otel_projection import project_query_unit_rows_to_otel

        rows: list[Any] = []
        for expression in expressions:
            envelope = await self.query_units(expression, limit=limit)
            rows.extend(envelope.items)
        return project_query_unit_rows_to_otel(
            source_ref,
            rows,
            include_message_text=include_message_text,
        )

    async def resolve_ref(self, ref: str) -> PublicRefResolutionPayload:
        """Resolve one public object/evidence ref into a bounded read payload."""
        from polylogue.surfaces.payloads import PublicRefResolutionPayload

        invalid_unicode_ref = _invalid_unicode_ref_payload(ref)
        if invalid_unicode_ref is not None:
            return cast(PublicRefResolutionPayload, invalid_unicode_ref)
        bounded_batch_ref = _oversized_annotation_batch_ref_payload(ref)
        if bounded_batch_ref is not None:
            # ``parse_public_ref`` deliberately falls back to EvidenceRef, whose
            # one-segment form accepts arbitrary colon-bearing session ids. Guard
            # batch-like malformed inputs before that fallback can misclassify
            # and reflect a multi-megabyte value through the session miss path.
            try:
                batch_candidate = ObjectRef.parse(ref)
            except ValueError:
                return cast(PublicRefResolutionPayload, bounded_batch_ref)
            if batch_candidate.kind != "annotation-batch":
                return cast(PublicRefResolutionPayload, bounded_batch_ref)
        try:
            parsed = parse_public_ref(ref)
        except ValueError as exc:
            if bounded_batch_ref is not None:
                return cast(PublicRefResolutionPayload, bounded_batch_ref)
            return cast(PublicRefResolutionPayload, _unresolved_ref_payload(ref, str(exc)))

        if isinstance(parsed, EvidenceRef):
            evidence_ref: EvidenceRef | None = parsed
            object_ref = parsed.to_object_ref()
        else:
            evidence_ref = None
            object_ref = parsed
        normalized_ref = parsed.format()
        archive_root = _active_archive_root(self.config)

        def read(archive: ArchiveStore) -> PublicRefResolutionPayload:
            if object_ref.kind == "session":
                return self._resolve_session_object_ref(archive, ref, normalized_ref, object_ref, evidence_ref)
            if object_ref.kind == "message":
                return self._resolve_message_object_ref(archive, ref, normalized_ref, object_ref, evidence_ref)
            if object_ref.kind == "block":
                return self._resolve_block_object_ref(archive, ref, normalized_ref, object_ref, evidence_ref)
            if object_ref.kind == "assertion":
                return self._resolve_assertion_object_ref(archive_root, ref, normalized_ref, object_ref)
            if object_ref.kind == "finding":
                return self._resolve_finding_object_ref(archive_root, ref, normalized_ref, object_ref)
            if object_ref.kind == "annotation-batch":
                return self._resolve_annotation_batch_object_ref(archive, ref, normalized_ref, object_ref)
            if object_ref.kind == "delegation":
                return self._resolve_delegation_object_ref(archive, ref, normalized_ref, object_ref)
            if object_ref.kind in {"run", "observed-event", "context-snapshot"}:
                return self._resolve_runtime_object_ref(archive, ref, normalized_ref, object_ref)
            if object_ref.kind in _PENDING_OBJECT_REF_KINDS:
                return cast(PublicRefResolutionPayload, _pending_ref_payload(ref, normalized_ref, object_ref.kind))
            return cast(
                PublicRefResolutionPayload,
                _unresolved_ref_payload(
                    ref,
                    f"unsupported public ref kind for resolution: {object_ref.kind}",
                    normalized_ref=normalized_ref,
                    kind=object_ref.kind,
                ),
            )

        return await run_archive_read(
            archive_root,
            operation="archive.resolve_ref",
            arguments={"ref": normalized_ref},
            work=read,
            projection="ref-resolution",
            stable_order="canonical",
        )

    def _resolve_session_object_ref(
        self,
        archive: Any,
        ref: str,
        normalized_ref: str,
        object_ref: ObjectRef,
        evidence_ref: EvidenceRef | None,
    ) -> PublicRefResolutionPayload:
        from polylogue.surfaces.payloads import PublicRefResolutionPayload, SessionSummaryPayload, model_json_document

        try:
            session_id = archive.resolve_session_id(object_ref.object_id)
        except KeyError:
            return cast(
                PublicRefResolutionPayload,
                _unresolved_ref_payload(ref, "session not found", normalized_ref=normalized_ref, kind="session"),
            )
        summaries = archive.list_summaries(session_id=session_id, limit=1)
        if not summaries:
            return cast(
                PublicRefResolutionPayload,
                _unresolved_ref_payload(ref, "session not found", normalized_ref=normalized_ref, kind="session"),
            )
        summary_payload = SessionSummaryPayload.from_summary(_archive_summary_to_domain(summaries[0]))
        return PublicRefResolutionPayload(
            ref=ref,
            normalized_ref=normalized_ref,
            kind="session",
            resolved=True,
            payload_kind="session-summary",
            payload=model_json_document(summary_payload),
            title=summary_payload.title,
            summary=f"{summary_payload.message_count} messages",
            object_refs=(f"session:{session_id}",),
            evidence_refs=() if evidence_ref is None else (evidence_ref.format(),),
            actions=(_resolution_action("read", f"polylogue find id:{session_id} then read --format json"),),
        )

    def _resolve_message_object_ref(
        self,
        archive: Any,
        ref: str,
        normalized_ref: str,
        object_ref: ObjectRef,
        evidence_ref: EvidenceRef | None,
    ) -> PublicRefResolutionPayload:
        from polylogue.surfaces.payloads import PublicRefResolutionPayload, SessionMessagePayload, model_json_document

        row = archive._conn.execute(
            """
            SELECT m.session_id, m.message_id
            FROM messages m
            WHERE m.message_id = ?
               OR ('message:' || m.session_id || ':' || m.message_id) = ?
            LIMIT 1
            """,
            (object_ref.object_id, normalized_ref),
        ).fetchone()
        if row is None:
            return cast(
                PublicRefResolutionPayload,
                _unresolved_ref_payload(ref, "message not found", normalized_ref=normalized_ref, kind="message"),
            )
        session_id = str(row["session_id"])
        message_id = str(row["message_id"])
        session = _archive_session_to_session(archive.read_session(session_id))
        message = next((item for item in session.messages if str(item.id) == message_id), None)
        if message is None:
            return cast(
                PublicRefResolutionPayload,
                _unresolved_ref_payload(ref, "message not found", normalized_ref=normalized_ref, kind="message"),
            )
        payload = SessionMessagePayload.from_message(message, session_id=session_id)
        return PublicRefResolutionPayload(
            ref=ref,
            normalized_ref=normalized_ref,
            kind="message",
            resolved=True,
            payload_kind="message",
            payload=model_json_document(payload),
            title=session.display_title,
            summary=(message.text or "")[:240],
            object_refs=(f"session:{session_id}", f"message:{message_id}"),
            evidence_refs=() if evidence_ref is None else (evidence_ref.format(),),
            actions=(_resolution_action("read session", f"polylogue find id:{session_id} then read --view messages"),),
        )

    def _resolve_block_object_ref(
        self,
        archive: Any,
        ref: str,
        normalized_ref: str,
        object_ref: ObjectRef,
        evidence_ref: EvidenceRef | None,
    ) -> PublicRefResolutionPayload:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveBlockQueryRow
        from polylogue.surfaces.payloads import BlockQueryRowPayload, PublicRefResolutionPayload, model_json_document

        block_index: int | None = None
        if object_ref.qualifiers:
            try:
                block_index = int(object_ref.qualifiers[0])
            except ValueError:
                return cast(
                    PublicRefResolutionPayload,
                    _unresolved_ref_payload(
                        ref,
                        "block ref qualifier must be an integer",
                        normalized_ref=normalized_ref,
                        kind="block",
                    ),
                )
        row = archive._conn.execute(
            """
            SELECT b.block_id, b.message_id, b.session_id, s.origin, s.title,
                   b.block_type, b.position, b.text, b.tool_name, b.semantic_type,
                   b.tool_command, b.tool_path
            FROM blocks b
            JOIN sessions s ON s.session_id = b.session_id
            WHERE b.block_id = ?
               OR (b.message_id = ? AND (? IS NOT NULL AND b.position = ?))
               OR ('block:' || b.block_id) = ?
            LIMIT 1
            """,
            (object_ref.object_id, object_ref.object_id, block_index, block_index, normalized_ref),
        ).fetchone()
        if row is None:
            return cast(
                PublicRefResolutionPayload,
                _unresolved_ref_payload(ref, "block not found", normalized_ref=normalized_ref, kind="block"),
            )
        payload = BlockQueryRowPayload.from_row(
            ArchiveBlockQueryRow(
                block_id=str(row["block_id"]),
                message_id=str(row["message_id"]),
                session_id=str(row["session_id"]),
                origin=str(row["origin"]),
                title=str(row["title"]) if row["title"] is not None else None,
                block_type=str(row["block_type"]),
                position=int(row["position"]),
                text=str(row["text"]) if row["text"] is not None else None,
                tool_name=str(row["tool_name"]) if row["tool_name"] is not None else None,
                semantic_type=str(row["semantic_type"]) if row["semantic_type"] is not None else None,
                tool_command=str(row["tool_command"]) if row["tool_command"] is not None else None,
                tool_path=str(row["tool_path"]) if row["tool_path"] is not None else None,
            )
        )
        return PublicRefResolutionPayload(
            ref=ref,
            normalized_ref=normalized_ref,
            kind="block",
            resolved=True,
            payload_kind="block",
            payload=model_json_document(payload),
            title=payload.title,
            summary=(payload.text or payload.tool_command or payload.tool_name or "")[:240],
            object_refs=(f"session:{payload.session_id}", f"message:{payload.message_id}", f"block:{payload.block_id}"),
            evidence_refs=() if evidence_ref is None else (evidence_ref.format(),),
            actions=(
                _resolution_action("read message", f"polylogue find id:{payload.session_id} then read --view messages"),
            ),
        )

    def _resolve_assertion_object_ref(
        self,
        archive_root: Path,
        ref: str,
        normalized_ref: str,
        object_ref: ObjectRef,
    ) -> PublicRefResolutionPayload:
        from polylogue.storage.sqlite.archive_tiers.user_write import read_assertion_envelope
        from polylogue.surfaces.payloads import AssertionClaimPayload, PublicRefResolutionPayload, model_json_document

        user_db = archive_root / "user.db"
        if not user_db.exists():
            return cast(
                PublicRefResolutionPayload,
                _unresolved_ref_payload(ref, "assertion not found", normalized_ref=normalized_ref, kind="assertion"),
            )
        with closing(sqlite3.connect(user_db)) as conn:
            conn.row_factory = sqlite3.Row
            envelope = read_assertion_envelope(conn, object_ref.object_id)
        if envelope is None:
            return cast(
                PublicRefResolutionPayload,
                _unresolved_ref_payload(ref, "assertion not found", normalized_ref=normalized_ref, kind="assertion"),
            )
        payload = AssertionClaimPayload.from_envelope(envelope)
        return PublicRefResolutionPayload(
            ref=ref,
            normalized_ref=normalized_ref,
            kind="assertion",
            resolved=True,
            payload_kind="assertion-claim",
            payload=model_json_document(payload),
            title=payload.key or payload.kind,
            summary=payload.body_text,
            object_refs=(normalized_ref, payload.target_ref),
            evidence_refs=payload.evidence_refs,
            actions=(_resolution_action("list assertion target", f"polylogue find {payload.target_ref} then read"),),
        )

    def _resolve_finding_object_ref(
        self,
        archive_root: Path,
        ref: str,
        normalized_ref: str,
        object_ref: ObjectRef,
    ) -> PublicRefResolutionPayload:
        from polylogue.storage.sqlite.finding_provenance import compute_finding_provenance
        from polylogue.surfaces.payloads import (
            FindingEvidenceRefState,
            FindingProvenancePayload,
            PublicRefResolutionPayload,
            model_json_document,
        )

        user_db = archive_root / "user.db"
        if not user_db.exists():
            return cast(
                PublicRefResolutionPayload,
                _unresolved_ref_payload(ref, "finding not found", normalized_ref=normalized_ref, kind="finding"),
            )
        with closing(sqlite3.connect(user_db)) as conn:
            conn.row_factory = sqlite3.Row
            provenance = compute_finding_provenance(conn, object_ref.object_id)
        if provenance is None:
            return cast(
                PublicRefResolutionPayload,
                _unresolved_ref_payload(ref, "finding not found", normalized_ref=normalized_ref, kind="finding"),
            )
        payload = FindingProvenancePayload(
            assertion_id=provenance.assertion_id,
            claim_key=provenance.claim_key,
            target_ref=provenance.target_ref,
            finding_kind=provenance.finding_kind,
            query_ref=provenance.query_ref,
            result_set_ref=provenance.result_set_ref,
            baseline_ref=provenance.baseline_ref,
            current_ref=provenance.current_ref,
            detector_ref=provenance.detector_ref,
            status=AssertionStatus.from_string(provenance.status),
            evidence=tuple(
                FindingEvidenceRefState(ref=item.ref, resolvable=item.resolvable, reason=item.reason)
                for item in provenance.evidence
            ),
            staleness_verdict=provenance.staleness_verdict,
            created_at_ms=provenance.created_at_ms,
            updated_at_ms=provenance.updated_at_ms,
        )
        caveats: tuple[str, ...] = ()
        if provenance.staleness_verdict != "current":
            caveats = (f"finding evidence staleness verdict: {provenance.staleness_verdict}",)
        object_refs = tuple(
            dict.fromkeys(
                ref_value
                for ref_value in (
                    normalized_ref,
                    provenance.target_ref,
                    provenance.query_ref,
                    provenance.result_set_ref,
                )
                if ref_value
            )
        )
        return PublicRefResolutionPayload(
            ref=ref,
            normalized_ref=normalized_ref,
            kind="finding",
            resolved=True,
            payload_kind="finding-provenance",
            payload=model_json_document(payload),
            title=provenance.claim_key or provenance.finding_kind or "finding",
            summary=f"{provenance.finding_kind or 'finding'} ({provenance.status})",
            object_refs=object_refs,
            evidence_refs=tuple(item.ref for item in provenance.evidence),
            caveats=caveats,
            actions=(_resolution_action("list target evidence", f"polylogue find {provenance.target_ref} then read"),),
        )

    def _resolve_annotation_batch_object_ref(
        self,
        archive: Any,
        ref: str,
        normalized_ref: str,
        object_ref: ObjectRef,
    ) -> PublicRefResolutionPayload:
        from polylogue.surfaces.payloads import (
            AnnotationBatchPayload,
            PublicRefResolutionPayload,
            RefResolutionActionPayload,
            model_json_document,
        )

        batch = archive.get_annotation_batch(object_ref.object_id)
        if batch is None:
            bounded = _oversized_annotation_batch_ref_payload(ref)
            if bounded is not None:
                return cast(PublicRefResolutionPayload, bounded)
            return cast(
                PublicRefResolutionPayload,
                _unresolved_ref_payload(
                    ref,
                    "annotation batch not found",
                    normalized_ref=normalized_ref,
                    kind="annotation-batch",
                ),
            )
        payload = AnnotationBatchPayload.from_batch(batch)
        scalar_ref_pairs = (
            (batch.batch_ref, payload.batch_ref),
            (batch.target_ref, payload.target_ref),
            (batch.source_result_ref, payload.source_result_ref),
            (batch.actor_ref, payload.actor_ref),
            (batch.model_ref, payload.model_ref),
            (batch.prompt_ref, payload.prompt_ref),
        )
        object_refs = tuple(dict.fromkeys(value for value, preview in scalar_ref_pairs if not preview.truncated))
        public_ref = normalized_ref
        public_normalized_ref: str | None = normalized_ref
        if payload.batch_ref.truncated:
            public_ref = f"annotation-batch:sha256-{payload.batch_ref.text_sha256}"
            public_normalized_ref = None
        actions: tuple[RefResolutionActionPayload, ...] = ()
        if not payload.target_ref.truncated:
            actions = (_resolution_action("read annotation target", f"polylogue find {batch.target_ref} then read"),)
        return PublicRefResolutionPayload(
            ref=public_ref,
            normalized_ref=public_normalized_ref,
            kind="annotation-batch",
            resolved=True,
            payload_kind="annotation-batch",
            payload=model_json_document(payload),
            title="annotation batch provenance",
            summary=(
                f"{payload.valid_count}/{payload.total_count} valid; "
                f"{payload.invalid_count} invalid; {payload.abstained_count} abstained"
            ),
            object_refs=object_refs,
            caveats=payload.truncation_caveats(),
            actions=actions,
        )

    def _resolve_delegation_object_ref(
        self,
        archive: Any,
        ref: str,
        normalized_ref: str,
        object_ref: ObjectRef,
    ) -> PublicRefResolutionPayload:
        """Resolve a ``delegation:`` ref against the polylogue-y964 `delegations`
        view. Two id shapes share one lookup: action-observed refs carry an
        ``instruction_tool_use_block_id`` verbatim; edge-only refs carry the
        deterministic ``edge:<parent>::<child>`` relation identity (no
        parent-side dispatch action exists to key off for edge_only/
        quarantined attempts). Missing, ambiguous, edge_only, and quarantined
        states are returned as typed payloads, never silently guessed."""

        from polylogue.surfaces.payloads import (
            DELEGATION_STATE_CAVEATS,
            DelegationCardPayload,
            PublicRefResolutionPayload,
            model_json_document,
        )

        edge_identity = parse_delegation_edge_object_id(object_ref.object_id)
        if edge_identity is not None:
            parent_session_id, child_session_id = edge_identity
            card = archive.get_delegation_card(parent_session_id=parent_session_id, child_session_id=child_session_id)
        else:
            card = archive.get_delegation_card(instruction_tool_use_block_id=object_ref.object_id)

        if card is None:
            return PublicRefResolutionPayload(
                ref=ref,
                normalized_ref=normalized_ref,
                kind="delegation",
                resolved=False,
                payload_kind="missing",
                caveats=("delegation attempt not found for this identity",),
            )

        payload = DelegationCardPayload.from_card(card)
        attempt = payload.attempt
        object_refs = [f"session:{attempt.parent_session_id}"]
        if attempt.child_session_id is not None:
            object_refs.append(f"session:{attempt.child_session_id}")
        if payload.run_ref is not None:
            object_refs.append(payload.run_ref)
        caveats: tuple[str, ...] = ()
        state_caveat = DELEGATION_STATE_CAVEATS.get(attempt.mapping_state)
        if state_caveat is not None:
            caveats = (state_caveat,)
        return PublicRefResolutionPayload(
            ref=ref,
            normalized_ref=normalized_ref,
            kind="delegation",
            resolved=True,
            payload_kind="delegation-card",
            payload=model_json_document(payload),
            title=f"delegation attempt ({attempt.mapping_state})",
            summary=(payload.instruction or "")[:240] or None,
            object_refs=tuple(object_refs),
            evidence_refs=payload.evidence_refs,
            caveats=caveats,
            actions=(
                _resolution_action("read parent session", f"polylogue find id:{attempt.parent_session_id} then read"),
            ),
        )

    def _resolve_runtime_object_ref(
        self,
        archive: Any,
        ref: str,
        normalized_ref: str,
        object_ref: ObjectRef,
    ) -> PublicRefResolutionPayload:
        from polylogue.surfaces.payloads import PublicRefResolutionPayload

        summary_offset = 0
        while True:
            summaries = archive.list_summaries(limit=200, offset=summary_offset)
            if not summaries:
                break
            summary_offset += len(summaries)
            for summary in summaries:
                resolved = self._resolve_runtime_object_ref_for_summary(
                    archive, ref, normalized_ref, object_ref, summary
                )
                if resolved is not None:
                    return resolved
            if len(summaries) < 200:
                break
        return cast(
            PublicRefResolutionPayload,
            _unresolved_ref_payload(
                ref, f"{object_ref.kind} not found", normalized_ref=normalized_ref, kind=object_ref.kind
            ),
        )

    def _resolve_runtime_object_ref_for_summary(
        self,
        archive: Any,
        ref: str,
        normalized_ref: str,
        object_ref: ObjectRef,
        summary: Any,
    ) -> PublicRefResolutionPayload | None:
        from polylogue.insights.transforms import compile_session_digest
        from polylogue.surfaces.payloads import (
            ContextSnapshotQueryRowPayload,
            ObservedEventQueryRowPayload,
            PublicRefResolutionPayload,
            RunQueryRowPayload,
            model_json_document,
        )

        session = _archive_session_to_session(archive.read_session(str(summary.session_id)))
        digest = compile_session_digest(session)
        if object_ref.kind == "run":
            for run in digest.run_projection.runs:
                if run.run_ref.format() != normalized_ref:
                    continue
                run_payload = RunQueryRowPayload(
                    run_ref=run.run_ref.format(),
                    session_id=str(summary.session_id),
                    origin=str(summary.origin),
                    title=summary.title,
                    native_session_id=run.native_session_id,
                    native_parent_session_id=run.native_parent_session_id,
                    parent_run_ref=run.parent_run_ref.format() if run.parent_run_ref else None,
                    agent_ref=run.agent_ref.format() if run.agent_ref else None,
                    lineage_refs=tuple(lineage.format() for lineage in run.lineage_refs),
                    provider_origin=run.provider_origin,
                    harness=run.harness,
                    role=run.role,
                    cwd=run.cwd,
                    git_branch=run.git_branch,
                    status=run.status,
                    confidence=run.confidence,
                    transcript_ref=run.transcript_ref.format() if run.transcript_ref else None,
                    evidence_refs=tuple(evidence.format() for evidence in run.evidence_refs),
                    context_snapshot_ref=run.context_snapshot_ref.format() if run.context_snapshot_ref else None,
                )
                return PublicRefResolutionPayload(
                    ref=ref,
                    normalized_ref=normalized_ref,
                    kind="run",
                    resolved=True,
                    payload_kind="run",
                    payload=model_json_document(run_payload),
                    title=summary.title,
                    summary=f"{run_payload.role} {run_payload.status}",
                    object_refs=(f"session:{summary.session_id}", normalized_ref),
                    evidence_refs=run_payload.evidence_refs,
                )
        if object_ref.kind == "observed-event":
            for event in digest.run_projection.events:
                if event.event_ref.format() != normalized_ref:
                    continue
                event_payload = ObservedEventQueryRowPayload(
                    event_ref=event.event_ref.format(),
                    session_id=str(summary.session_id),
                    origin=str(summary.origin),
                    title=summary.title,
                    kind=event.kind,
                    summary=event.summary,
                    delivery_state=event.delivery_state,
                    subject_ref=event.subject_ref.format() if event.subject_ref else None,
                    object_refs=tuple(item.format() for item in event.object_refs),
                    evidence_refs=tuple(item.format() for item in event.evidence_refs),
                )
                return PublicRefResolutionPayload(
                    ref=ref,
                    normalized_ref=normalized_ref,
                    kind="observed-event",
                    resolved=True,
                    payload_kind="observed-event",
                    payload=model_json_document(event_payload),
                    title=summary.title,
                    summary=event_payload.summary,
                    object_refs=(f"session:{summary.session_id}", normalized_ref, *event_payload.object_refs),
                    evidence_refs=event_payload.evidence_refs,
                )
        if object_ref.kind == "context-snapshot":
            for snapshot in digest.run_projection.context_snapshots:
                if snapshot.snapshot_ref.format() != normalized_ref:
                    continue
                snapshot_payload = ContextSnapshotQueryRowPayload(
                    snapshot_ref=snapshot.snapshot_ref.format(),
                    session_id=str(summary.session_id),
                    origin=str(summary.origin),
                    title=summary.title,
                    run_ref=snapshot.run_ref.format(),
                    boundary=snapshot.boundary,
                    inheritance_mode=snapshot.inheritance_mode,
                    segment_refs=tuple(item.format() for item in snapshot.segment_refs),
                    evidence_refs=tuple(item.format() for item in snapshot.evidence_refs),
                    metadata=dict(snapshot.metadata),
                )
                return PublicRefResolutionPayload(
                    ref=ref,
                    normalized_ref=normalized_ref,
                    kind="context-snapshot",
                    resolved=True,
                    payload_kind="context-snapshot",
                    payload=model_json_document(snapshot_payload),
                    title=summary.title,
                    summary=f"{snapshot_payload.boundary} ({snapshot_payload.inheritance_mode})",
                    object_refs=(
                        f"session:{summary.session_id}",
                        normalized_ref,
                        snapshot_payload.run_ref,
                        *snapshot_payload.segment_refs,
                    ),
                    evidence_refs=snapshot_payload.evidence_refs,
                )
        return None

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

    async def get_actions(self, session_id: str) -> tuple[Action, ...]:
        """Derive a session's actions from its content blocks.

        ``index.db`` exposes an ``actions`` view; these actions
        are derived on read from the session's tool-use/tool-result blocks —
        the same source the archive materializer hashed into durable rows.
        Returns an empty tuple when the session is absent.
        """
        session = await self.get_session(session_id)
        if session is None:
            return ()
        return _actions_for_session(session)

    async def get_actions_batch(
        self,
        session_ids: builtins.list[str],
    ) -> dict[str, tuple[Action, ...]]:
        """Batch counterpart of :meth:`get_actions`.

        Missing sessions are omitted from the result mapping, mirroring
        the archive repository batch reader.
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
                limit=50 if limit is None else limit,
            )
            sessions = [_archive_session_to_session(archive.read_session(summary.session_id)) for summary in summaries]
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
        limit: int | None = 50,
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
                _archive_summary_to_domain(summary)
                for summary in archive.list_summaries(
                    origin=origin,
                    limit=50 if limit is None else limit,
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
            with suppress(ValueError, ImportError):
                vector_provider = create_vector_provider(self.config, db_path=archive_root / "embeddings.db")
        sessions = await spec.list(self.config, vector_provider=vector_provider)
        if content_projection is None or not content_projection.filters_content():
            return sessions
        return [session.with_content_projection(content_projection) for session in sessions]

    async def search_session_hits(self, spec: SessionQuerySpec) -> builtins.list[SessionSearchHit]:
        """Return archive FTS/hybrid search-hit projections for a query spec.

        The hit projection carries match snippets and ranking metadata the
        :class:`SearchEnvelope` builder needs, distinct from the full
        session hydration of :meth:`list_sessions_for_spec`.
        """
        from polylogue.archive.query.search_hits import search_hits_for_plan

        vector_provider = None
        if spec.similar_text or spec.retrieval_lane == "hybrid":
            from polylogue.storage.search_providers import create_vector_provider

            archive_root = _active_archive_root(self.config)
            with suppress(ValueError, ImportError):
                vector_provider = create_vector_provider(self.config, db_path=archive_root / "embeddings.db")
        return await search_hits_for_plan(spec.to_plan(vector_provider=vector_provider), self.config)

    async def diagnose_query_miss(self, spec: SessionQuerySpec) -> QueryMissDiagnostics:
        """Best-effort explanation for an empty archive query result.

        The diagnostic is duck-typed over this facade: it reads whatever
        archive count/stats methods are available and degrades gracefully when
        a probe is absent.
        """
        from polylogue.archive.query.miss_diagnostics import diagnose_query_miss

        return await diagnose_query_miss(self, spec, config=self.config)

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
                hits=[
                    _archive_search_hit_to_domain(hit)
                    for hit in archive.search_summaries(
                        query,
                        limit=limit,
                        origin=source,
                        since_ms=_archive_query_date_ms("since", since),
                    )
                ]
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
        from polylogue.archive.query.expression import compile_expression_into
        from polylogue.archive.query.search_hits import session_search_hit_from_summary
        from polylogue.archive.query.spec import SessionQuerySpec
        from polylogue.surfaces.payloads import (
            InvalidSearchCursorError,
            SessionSearchHitPayload,
            build_search_envelope,
            decode_search_cursor,
            search_cursor_lane_matches_request,
        )

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
        decoded_cursor = decode_search_cursor(spec.cursor) if spec.cursor else None
        if decoded_cursor is not None and not search_cursor_lane_matches_request(
            decoded_cursor.lane,
            spec.retrieval_lane,
        ):
            raise InvalidSearchCursorError(
                f"cursor was minted for retrieval_lane={decoded_cursor.lane!r} "
                f"but this request is {spec.retrieval_lane!r}"
            )
        display_limit = spec.limit or limit
        display_offset = spec.offset
        fetch_offset = decoded_cursor.r if decoded_cursor is not None else display_offset
        fetch_limit = display_limit * 2 if decoded_cursor is not None else display_limit
        query_text = _archive_text_query(spec)

        def read(archive: ArchiveStore) -> tuple[tuple[SessionSearchHitPayload, ...], int]:
            if query_text is not None:
                hits = _archive_search_hits_for_spec(
                    archive,
                    spec,
                    query_text,
                    limit=fetch_limit,
                    offset=fetch_offset,
                )
                total = _archive_count_sessions_for_spec(archive, spec)
                hit_payloads = tuple(
                    _archive_search_hit_to_payload(hit, archive.read_summary(hit.session_id)) for hit in hits
                )
            elif query.strip():
                summaries = _archive_list_summaries_for_spec(
                    archive,
                    spec,
                    default_limit=display_limit,
                    limit=fetch_limit,
                    offset=fetch_offset,
                )
                total = _archive_count_sessions_for_spec(archive, spec)
                hit_payloads = tuple(
                    SessionSearchHitPayload.from_search_hit(
                        session_search_hit_from_summary(
                            _archive_summary_to_domain(summary),
                            rank=fetch_offset + index,
                            retrieval_lane=spec.retrieval_lane,
                            match_surface="session",
                            message_id=None,
                            snippet=None,
                        ),
                        message_count=summary.message_count,
                    )
                    for index, summary in enumerate(summaries, start=1)
                )
            else:
                hit_payloads = ()
                total = _archive_count_sessions_for_spec(archive, spec)
            return hit_payloads, total

        hit_payloads, total = await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.search_envelope",
            arguments={
                "query": query,
                "origin": origin,
                "since": since,
                "until": until,
                "retrieval_lane": retrieval_lane,
                "sort": sort,
                "cursor": cursor,
            },
            work=read,
            page_size=display_limit,
            offset=display_offset,
            projection="search-envelope",
            stable_order=spec.sort or "rank,session_id,message_id",
            workload_class="scan" if query_text is not None else "interactive",
        )
        return build_search_envelope(
            hit_payloads,
            total=total,
            limit=display_limit,
            offset=display_offset,
            query=query,
            retrieval_lane="dialogue" if query_text is not None else spec.retrieval_lane,
            sort=spec.sort,
            cursor=decoded_cursor,
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

    async def archive_list_sessions(
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
        limit: int = 50,
        offset: int = 0,
        sample: bool = False,
    ) -> list[ArchiveSessionSummary]:
        """List session summaries."""
        from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest

        arguments = {key: value for key, value in locals().items() if key not in {"self", "limit", "offset"}}
        arguments.update(
            {
                "has_types": tuple(has_types),
                "excluded_origins": tuple(excluded_origins),
                "tags": tuple(tags),
                "excluded_tags": tuple(excluded_tags),
                "repo_names": tuple(repo_names),
                "project_refs": tuple(project_refs),
                "tool_terms": tuple(tool_terms),
                "excluded_tool_terms": tuple(excluded_tool_terms),
                "action_terms": _archive_action_terms("action", action_terms),
                "excluded_action_terms": _archive_action_terms("exclude_action", excluded_action_terms),
                "action_sequence": _archive_action_sequence(action_sequence),
                "message_type": _archive_message_type(message_type),
                "since": _archive_query_date_ms("since", since),
                "until": _archive_query_date_ms("until", until),
            }
        )
        transaction = QueryTransaction(
            _active_archive_root(self.config),
            QueryTransactionRequest(
                operation="archive_list_sessions",
                arguments=arguments,
                page_size=limit,
                offset=max(0, offset),
            ),
        )
        return await transaction.run(
            lambda archive: list(
                archive.list_summaries(
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
                    limit=limit,
                    offset=offset,
                    sample=sample,
                )
            )
        )

    async def archive_search_sessions(
        self,
        query: str,
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
        limit: int = 20,
    ) -> list[ArchiveSessionSearchHit]:
        """Search session block text."""
        from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest

        arguments = {key: value for key, value in locals().items() if key not in {"self", "limit"}}
        arguments.update(
            {
                "has_types": tuple(has_types),
                "excluded_origins": tuple(excluded_origins),
                "tags": tuple(tags),
                "excluded_tags": tuple(excluded_tags),
                "repo_names": tuple(repo_names),
                "project_refs": tuple(project_refs),
                "tool_terms": tuple(tool_terms),
                "excluded_tool_terms": tuple(excluded_tool_terms),
                "action_terms": _archive_action_terms("action", action_terms),
                "excluded_action_terms": _archive_action_terms("exclude_action", excluded_action_terms),
                "action_sequence": _archive_action_sequence(action_sequence),
                "message_type": _archive_message_type(message_type),
                "since": _archive_query_date_ms("since", since),
                "until": _archive_query_date_ms("until", until),
            }
        )
        transaction = QueryTransaction(
            _active_archive_root(self.config),
            QueryTransactionRequest(
                operation="archive_search_sessions",
                arguments=arguments,
                page_size=limit,
            ),
        )
        return await transaction.run(
            lambda archive: list(
                archive.search_summaries(
                    query,
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
                    limit=limit,
                )
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
                "since": request.since,
                "until": request.until,
                "tier": request.tier,
            },
            work=lambda archive: archive.list_session_profile_insights(
                origin=request.origin,
                workflow_shape=request.workflow_shape,
                terminal_state=request.terminal_state,
                since_ms=_archive_query_date_ms("since", request.since),
                until_ms=_archive_query_date_ms("until", request.until),
                tier=request.tier,
                limit=request.limit,
                offset=request.offset,
                min_wallclock_seconds=request.min_wallclock_seconds,
                max_wallclock_seconds=request.max_wallclock_seconds,
                sort=request.sort,
            ),
            page_size=request.limit,
            offset=request.offset,
            projection="session-profile",
            stable_order="time,session_id",
        )

    def filter(self) -> SessionFilter:
        from polylogue.archive.filter.filters import SessionFilter
        from polylogue.storage.search_providers import create_vector_provider

        archive_root = _active_archive_root(self.config)
        vector_provider = None
        with suppress(ValueError, ImportError):
            vector_provider = create_vector_provider(self.config, db_path=archive_root / "embeddings.db")

        return SessionFilter(
            archive_root=archive_root,
            config=self.config,
            vector_provider=vector_provider,
        )

    async def provider_usage_report(
        self,
        *,
        origin: str | None = None,
        limit: int | None = 25,
        detail: str = "full",
    ) -> ProviderUsageReport:
        """Return provider usage accounting diagnostics for the active archive."""
        from polylogue.storage.usage import provider_usage_report_from_connection

        if detail not in {"headline", "full"}:
            raise ValueError("detail must be 'headline' or 'full'")
        usage_detail = cast(Literal["headline", "full"], detail)
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.provider_usage",
            arguments={"origin": origin, "limit": limit, "detail": detail},
            work=lambda archive: provider_usage_report_from_connection(
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

    async def stats(self) -> ArchiveStats:
        from polylogue.operations import ArchiveStats as PublicArchiveStats

        def read(archive: ArchiveStore) -> PublicArchiveStats:
            stats = archive.stats()
            word_row = archive._conn.execute("SELECT COALESCE(SUM(word_count), 0) FROM sessions").fetchone()
            recent = [
                _archive_session_to_session(archive.read_session(summary.session_id))
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
        from polylogue.archive.query.facets import (
            FacetBuckets as _FacetBuckets,
        )
        from polylogue.archive.query.facets import (
            compute_idf,
        )
        from polylogue.surfaces.payloads import (
            FacetBucketsPayload,
            FacetsResponse,
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
        global_buckets, scoped_buckets = await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.facets",
            arguments={"spec": spec, "include_deferred": include_deferred},
            work=lambda archive: (
                _archive_facet_buckets(archive, None, include_deferred=include_deferred),
                _archive_facet_buckets(archive, spec, include_deferred=include_deferred)
                if scoped_to_query
                else _archive_facet_buckets(archive, None, include_deferred=include_deferred),
            ),
            projection="facets",
            workload_class="scan",
        )
        idf_map = compute_idf(global_buckets) if include_idf else {}
        active = scoped_buckets if scoped_to_query else global_buckets
        complete_families = _FACET_COMPLETE_FAMILIES if include_deferred else _FACET_CORE_FAMILIES
        deferred_families = {} if include_deferred else dict.fromkeys(_FACET_DEFERRED_FAMILIES, "deferred_by_default")
        return FacetsResponse.model_validate(
            {
                "scoped_to_query": scoped_to_query,
                "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "stale": False,
                "stale_age_s": None,
                "budget_exceeded": False,
                "complete_families": complete_families,
                "deferred_families": deferred_families,
                "family_errors": {},
                "family_status": {
                    **{family: _family_status_payload(family, state="complete") for family in complete_families},
                    **{
                        family: _family_status_payload(family, state="deferred", reason=reason)
                        for family, reason in deferred_families.items()
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
                "idf": idf_map,
            }
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
        """Rebuild durable session-insight read models.

        When ``progress_callback`` is supplied, the full-rebuild DELETE phase
        emits a per-table heartbeat (#1607 parity) so a long rebuild shows
        forward motion instead of hanging silently.
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return _rebuild_archive_session_insights(
                archive,
                session_ids=session_ids,
                progress_callback=progress_callback,
            )

    async def resume_brief(
        self,
        session_id: str,
        *,
        related_limit: int = 6,
    ) -> ResumeBrief | None:
        """Build a compact handoff brief for an archived session."""
        from polylogue.insights.resume import ResumeOperations, build_resume_brief

        return await build_resume_brief(cast(ResumeOperations, self), session_id, related_limit=related_limit)

    async def find_resume_candidates(
        self, *, repo_path: str, cwd: str | None = None, recent_files: Sequence[str] = (), limit: int = 10
    ) -> tuple[ResumeCandidate, ...]:
        from polylogue.insights.resume import ResumeOperations, find_resume_candidates

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
    ) -> tuple[list[Message], int]:
        """Return paginated ``Message`` objects for a session.

        Raises ``SessionNotFoundError`` if the session does not exist.
        """
        if material_origin:
            session = await self.get_session(session_id, content_projection=content_projection)
            if session is None:
                raise SessionNotFoundError(session_id)
            messages = [
                message
                for message in session.messages
                if _archive_message_matches(
                    message,
                    message_role=message_role,
                    message_type=message_type,
                    material_origin=material_origin,
                )
            ]
            return messages[offset : offset + limit], len(messages)

        resolved_session_id = await self.repository.resolve_id(session_id) or session_id
        messages, total = await self.repository.get_messages_paginated(
            resolved_session_id,
            message_role=message_role,
            message_type=message_type,
            limit=limit,
            offset=offset,
        )
        if total == 0 and resolved_session_id == session_id and await self.repository.resolve_id(session_id) is None:
            raise SessionNotFoundError(session_id)
        if content_projection is not None and content_projection.filters_content():
            messages = project_message_content(messages, content_projection)
        return messages, total

    def iter_messages(
        self,
        session_id: str,
        *,
        message_roles: MessageRoleFilter = (),
        material_origin: tuple[MaterialOrigin, ...] = (),
        limit: int | None = None,
    ) -> AsyncIterator[Message]:
        async def _iter() -> AsyncIterator[Message]:
            if not material_origin:
                async for message in self.repository.iter_messages(
                    session_id,
                    message_roles=message_roles,
                    limit=limit,
                ):
                    yield message
                return

            session = await self.get_session(session_id)
            if session is None:
                return
            count = 0
            for message in session.messages:
                if message_roles and message.role not in message_roles:
                    continue
                if message.material_origin not in material_origin:
                    continue
                if limit is not None and count >= limit:
                    break
                count += 1
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
            work=lambda archive: _archive_list_summaries_for_spec(archive, spec, default_limit=50),
            page_size=limit,
            offset=offset,
            projection="session-summary",
            stable_order=sort or "date,session_id",
        )
        return [
            {
                "id": summary.session_id,
                "title": summary.title,
                "origin": summary.origin,
                "created_at": parse_archive_datetime(summary.created_at),
                "updated_at": parse_archive_datetime(summary.updated_at),
                "message_count": summary.message_count,
                "word_count": summary.word_count,
            }
            for summary in archive_summaries
        ]

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
        import asyncio

        from polylogue.insights.export_bundles import export_insight_bundle

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="insights.export_bundle",
            arguments={"request": request},
            work=lambda archive: asyncio.run(
                export_insight_bundle(_ArchiveInsightExportOperations(archive), self.config, request)
            ),
            page_size=1,
            projection="insight-export",
            workload_class="scan",
        )

    async def get_session_summary(self, session_id: str) -> SessionSummary | None:
        """Return a summary record for a single session, or ``None`` if not found."""

        def read(archive: ArchiveStore) -> SessionSummary | None:
            try:
                resolved_id = archive.resolve_session_id(session_id)
                return _archive_summary_to_domain(archive.read_summary(resolved_id))
            except KeyError:
                return None

        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.session_summary.get",
            arguments={"session_id": session_id},
            work=read,
            projection="session-summary",
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

    async def update_index(self, session_ids: list[str]) -> bool:
        """Repair the archive block-FTS index.

        The archive FTS index is trigger-maintained, so it stays in sync on every
        write and there is no per-session update primitive. This exposes the
        operator repair path: a full archive rebuild that reconciles the index
        with ``index.db`` blocks. ``session_ids`` is accepted for surface
        symmetry but the archive rebuild always reconciles the whole index.
        """
        del session_ids
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            archive.rebuild_index()
        return True

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
        import asyncio

        from polylogue.archive.session.neighbor_candidates import (
            NeighborDiscoveryRequest,
            discover_neighbor_candidates,
        )

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
            work=lambda archive: asyncio.run(discover_neighbor_candidates(_ArchiveNeighborRuntime(archive), request)),
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

        from polylogue.insights.session_commit import build_correlation_result, correlation_result_to_payload

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

        result = build_correlation_result(
            session_id=session_id,
            messages=messages,
            session_created_at=session.created_at,
            session_updated_at=session.updated_at,
            repo_path=repo,
            before_hours=since_hours,
            after_hours=since_hours,
            confidence_threshold=confidence_threshold,
        )
        return cast(JSONDocument, correlation_result_to_payload(result))

    async def get_session_tree(self, session_id: str) -> list[Session]:
        """Return the full session tree (parent + children) for a session."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="archive.session_tree",
            arguments={"session_id": session_id},
            work=lambda archive: [
                _archive_session_to_session(session) for session in archive.get_session_tree(session_id)
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

    async def delete_session(self, session_id: str) -> bool:
        """Permanently delete a session and all associated data.

        Returns ``True`` if something was deleted, ``False`` if the session
        was not found. Routes through :meth:`ArchiveMutationsMixin
        .delete_session_safe` so resolution and idempotency stay
        centralized (#862).
        """
        result = await self.delete_session_safe(session_id)
        return result.outcome == "deleted"

    async def delete_session_safe(self, session_id: str) -> DeleteSessionResult:
        """Typed delete that returns ``outcome="deleted"`` or ``"not_found"``."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import DeleteSessionResult

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                resolved = archive.resolve_session_id(session_id)
                deleted = archive.delete_sessions((resolved,))
            except KeyError:
                return DeleteSessionResult(
                    outcome="not_found",
                    session_id=session_id,
                    detail="session_not_found",
                )
        return DeleteSessionResult(
            outcome="deleted" if deleted else "not_found",
            session_id=resolved,
            detail=None if deleted else "session_not_found",
        )

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
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import TagMutationResult

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                resolved_v1 = archive.resolve_session_id(session_id)
                changed = archive.add_user_tags(
                    (resolved_v1,),
                    (tag,),
                    author_ref=author_ref,
                    author_kind=author_kind,
                )
            except KeyError:
                raise SessionNotFoundError(session_id) from None
        return TagMutationResult(
            outcome="added" if changed else "no_op",
            detail=None if changed else "already_present",
        )

    async def remove_tag(self, session_id: str, tag: str) -> TagMutationResult:
        """Remove a tag from a session.

        Returns a ``TagMutationResult`` with:
        - ``outcome="removed"`` if the tag was removed
        - ``outcome="not_present"`` if the tag was not present
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import TagMutationResult

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                resolved_v1 = archive.resolve_session_id(session_id)
                changed = archive.remove_user_tags((resolved_v1,), (tag,))
            except KeyError:
                raise SessionNotFoundError(session_id) from None
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
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import (
            MetadataKeyValidationError,
            MetadataMutationResult,
            validate_metadata_key,
        )

        validation_error = validate_metadata_key(key)
        if validation_error is not None:
            raise MetadataKeyValidationError(validation_error)

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                resolved = archive.resolve_session_id(session_id)
                changed = archive.set_user_metadata((resolved,), ((key, value),))
            except KeyError:
                raise SessionNotFoundError(session_id) from None
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
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import (
            MetadataKeyValidationError,
            MetadataMutationResult,
            validate_metadata_key,
        )

        validation_error = validate_metadata_key(key)
        if validation_error is not None:
            raise MetadataKeyValidationError(validation_error)

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                resolved = archive.resolve_session_id(session_id)
                changed = archive.delete_user_metadata(resolved, key)
            except KeyError:
                raise SessionNotFoundError(session_id) from None
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

        Validation (empty inputs and size limits) is enforced inside the
        :class:`ArchiveMutationsMixin` so every surface sees the same
        behavior.
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import BulkTagMutationResult

        if not session_ids:
            raise ValueError("bulk_tag_sessions requires at least one session_id")
        if not tags:
            raise ValueError("bulk_tag_sessions requires at least one tag")
        max_sessions = 100
        max_tags = 20
        if len(session_ids) > max_sessions:
            raise ValueError(f"bulk_tag_sessions supports at most {max_sessions} session_ids")
        if len(tags) > max_tags:
            raise ValueError(f"bulk_tag_sessions supports at most {max_tags} tags")
        affected_count = 0
        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            for session_id in session_ids:
                try:
                    resolved = archive.resolve_session_id(session_id)
                except KeyError:
                    continue
                if (
                    archive.add_user_tags(
                        (resolved,),
                        tuple(tags),
                        author_ref=author_ref,
                        author_kind=author_kind,
                    )
                    > 0
                ):
                    affected_count += 1
        return BulkTagMutationResult(
            session_count=len(session_ids),
            tag_count=len(tags),
            affected_count=affected_count,
            skipped_count=len(session_ids) - affected_count,
        )

    # ------------------------------------------------------------------
    # Marks
    # ------------------------------------------------------------------

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
        archive_resolved = await self._archive_resolve_session_id(session_id)
        if archive_resolved is None:
            raise SessionNotFoundError(session_id)
        return archive_resolved

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
        """
        from polylogue.core.user_state_targets import validate_mark_type

        mark_type = validate_mark_type(mark_type)
        target = await self._resolve_user_state_target(
            session_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.add_mark(str(target["target_type"]), str(target["target_id"]), mark_type)

    async def remove_mark(
        self,
        session_id: str,
        mark_type: str,
        *,
        target_type: str = TARGET_SESSION,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> bool:
        """Remove a mark from a session or message. Returns ``True`` if removed."""
        from polylogue.core.user_state_targets import validate_mark_type

        mark_type = validate_mark_type(mark_type)
        target = await self._resolve_user_state_target(
            session_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.remove_mark(str(target["target_type"]), str(target["target_id"]), mark_type)

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
        if message_id is not None:
            resolved_target_type = TARGET_MESSAGE
            resolved_target_id = message_id
        elif session_id is not None and target_id is None:
            try:
                resolved_target_id = await self._resolve_user_state_session_id(session_id)
                resolved_target_type = TARGET_SESSION
            except SessionNotFoundError:
                return []
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.marks.list",
            arguments={"mark_type": mark_type, "target_type": resolved_target_type, "target_id": resolved_target_id},
            work=lambda archive: archive.list_marks(
                mark_type=mark_type,
                target_type=resolved_target_type,
                target_id=resolved_target_id,
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
        """Create or update an annotation. Returns ``True`` if newly created."""
        if not annotation_id.strip():
            raise ValueError("annotation_id must not be empty")
        if not note_text.strip():
            raise ValueError("note_text must not be empty")
        target = await self._resolve_user_state_target(
            session_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.save_annotation(
                annotation_id,
                str(target["target_type"]),
                str(target["target_id"]),
                note_text,
            )

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
                return []
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
        """Delete an annotation. Returns ``True`` if deleted."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.delete_annotation(annotation_id)

    # ------------------------------------------------------------------
    # Saved views
    # ------------------------------------------------------------------

    async def save_view(self, view_id: str, name: str, query_json: str) -> bool:
        """Save a named query view. Returns ``True`` if newly created."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.save_view(view_id, name, query_json)

    async def get_view(self, view_id: str) -> dict[str, str] | None:
        """Get a saved view by ID."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.view.get",
            arguments={"view_id": view_id},
            work=lambda archive: archive.get_view(view_id),
            projection="saved-view",
        )

    async def get_view_by_name(self, name: str) -> dict[str, str] | None:
        """Get a saved view by name."""
        return await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.view_by_name.get",
            arguments={"name": name},
            work=lambda archive: archive.get_view_by_name(name),
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
        """Delete a saved view. Returns ``True`` if deleted."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.delete_view(view_id)

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
        """Save a recall pack. Returns ``True`` if newly created."""
        import json

        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise ValueError("recall pack payload must be a JSON object")
        resolved_session_ids, normalized_payload_json = await self._build_recall_pack_payload(
            label=label,
            payload=payload,
        )
        session_ids_json = json.dumps(resolved_session_ids, sort_keys=True)
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.save_recall_pack(pack_id, label, session_ids_json, normalized_payload_json)

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
        """Delete a recall pack. Returns ``True`` if deleted."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.delete_recall_pack(pack_id)

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
        """Create or update a durable reader workspace."""
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

        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.save_workspace(
                workspace_id=workspace_id,
                name=name,
                mode=mode,
                open_targets_json=normalized_targets_json,
                layout_json=normalized_layout_json,
                active_target_json=normalized_active_json,
            )

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
        """Delete a durable reader workspace. Returns ``True`` if deleted."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.delete_workspace(workspace_id)

    # ------------------------------------------------------------------
    # Learning corrections (#1131)
    #
    # User-recorded overrides that the insight materialization paths
    # consult after computing their base suggestion. Lives outside the
    # content-hash boundary by construction; see
    # :mod:`polylogue.insights.feedback` and
    # :mod:`polylogue.storage.insights.feedback`.
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
        :class:`~polylogue.insights.feedback.UnknownCorrectionKindError`
        when ``kind`` is not a recognized
        :class:`~polylogue.insights.feedback.CorrectionKind`.
        """

        normalized_payload = {str(key): str(value) for key, value in payload.items()}
        parse_correction_kind(kind)
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                return archive.record_correction(
                    session_id,
                    kind,
                    normalized_payload,
                    note=note,
                    author_ref=author_ref,
                    author_kind=author_kind,
                )
            except KeyError as exc:
                raise SessionNotFoundError(session_id) from exc

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
        """Delete one correction. Returns ``True`` when a row was removed."""

        parse_correction_kind(kind)
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                return archive.delete_correction(session_id, kind)
            except KeyError as exc:
                raise SessionNotFoundError(session_id) from exc

    async def clear_corrections(self, session_id: str) -> int:
        """Delete every correction for a session. Returns the count."""

        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                return archive.clear_corrections(session_id)
            except KeyError as exc:
                raise SessionNotFoundError(session_id) from exc

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
        """
        from polylogue.archive.blackboard import (
            BLACKBOARD_KINDS,
            build_blackboard_body,
            decode_blackboard_note,
        )
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        if kind not in BLACKBOARD_KINDS:
            raise ValueError(f"kind must be one of {list(BLACKBOARD_KINDS)}, got {kind!r}")
        body = build_blackboard_body(
            kind=kind,
            title=title,
            content=content,
            scope_repo=scope_repo,
            scope_issue=scope_issue,
            scope_path=scope_path,
            related_sessions=related_sessions,
        )
        note_id = str(uuid.uuid4())
        target_type = "session" if scope_session else None
        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            envelope = archive.post_blackboard_note(
                body,
                target_type=target_type,
                target_id=scope_session,
                note_id=note_id,
                author_ref=author_ref,
                author_kind=author_kind,
                evidence_refs=evidence_refs,
                staleness=staleness,
                context_policy=context_policy,
            )
        return decode_blackboard_note(
            note_id=envelope.note_id,
            body=envelope.body,
            target_type=envelope.target_type,
            target_id=envelope.target_id,
            created_at_ms=envelope.created_at_ms,
            updated_at_ms=envelope.updated_at_ms,
        )

    async def list_blackboard_notes(
        self,
        *,
        kind: str | None = None,
        scope_repo: str | None = None,
        unresolved: bool = False,
        limit: int = 20,
    ) -> list[BlackboardNote]:
        """List blackboard notes, newest first, with optional filters (#1697).

        ``unresolved`` narrows to open-work kinds (:data:`UNRESOLVED_KINDS`).
        Filtering runs on decoded notes, then the result is capped at ``limit``.
        """
        from polylogue.archive.blackboard import (
            UNRESOLVED_KINDS,
            decode_blackboard_note,
        )

        envelopes = await run_archive_read(
            _active_archive_root(self.config),
            operation="user_state.blackboard.list",
            arguments={"kind": kind, "scope_repo": scope_repo, "unresolved": unresolved},
            work=lambda archive: archive.list_blackboard_notes(),
            page_size=limit,
            projection="blackboard-notes",
            stable_order="updated_at:desc,note_id",
        )
        notes: list[BlackboardNote] = []
        for envelope in envelopes:
            note = decode_blackboard_note(
                note_id=envelope.note_id,
                body=envelope.body,
                target_type=envelope.target_type,
                target_id=envelope.target_id,
                created_at_ms=envelope.created_at_ms,
                updated_at_ms=envelope.updated_at_ms,
            )
            if kind is not None and note.kind != kind:
                continue
            if scope_repo is not None and note.scope_repo != scope_repo:
                continue
            if unresolved and note.kind not in UNRESOLVED_KINDS:
                continue
            notes.append(note)
            if limit > 0 and len(notes) >= limit:
                break
        return notes
