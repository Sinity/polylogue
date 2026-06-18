"""Terminal unit-query execution over the archive."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from polylogue.archive.query.archive_execution import _session_to_session
from polylogue.archive.query.expression import QueryUnitSource
from polylogue.archive.query.predicate import QueryBoolPredicate, QueryFieldPredicate, QueryNotPredicate, QueryPredicate
from polylogue.archive.query.spec import (
    normalize_action_sequence,
    normalize_action_terms,
    normalize_tool_terms,
    optional_int,
    optional_message_type,
    optional_text,
    parse_query_date,
    split_csv,
)
from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.run_projection import ContextSnapshot, ObservedEvent
from polylogue.insights.transforms import compile_recovery_digest
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary, ArchiveStore
from polylogue.surfaces.payloads import (
    ActionQueryRowPayload,
    AssertionQueryRowPayload,
    BlockQueryRowPayload,
    ContextSnapshotQueryRowPayload,
    MessageQueryRowPayload,
    ObservedEventQueryRowPayload,
    QueryUnitEnvelope,
    build_query_unit_envelope,
)

_SUMMARY_SCAN_BATCH_SIZE = 500


def _bool_param(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _epoch_ms(field: str, value: object) -> int | None:
    if isinstance(value, int):
        return value
    if value is None:
        return None
    parsed = parse_query_date(field, str(value))
    if parsed is None:
        return None
    return int(parsed.timestamp() * 1000)


def query_unit_session_filters(**params: object) -> dict[str, object]:
    """Normalize shared session filters for terminal query-unit rows.

    Terminal unit-source execution returns row-level
    results, but callers still need the same surrounding session scope as the
    normal session query surfaces.  This helper is the single cross-surface
    adapter into ``ArchiveStore.query_*``'s ``session_filters`` argument.
    """

    origin = optional_text(params.get("origin"))
    origins = split_csv(params.get("origins"))
    if not origins and origin is None:
        origins = split_csv(params.get("source"))
    excluded_origins = split_csv(params.get("excluded_origins") or params.get("exclude_origin"))
    tags = tuple(tag.lower() for tag in split_csv(params.get("tags") or params.get("tag")))
    excluded_tags = tuple(tag.lower() for tag in split_csv(params.get("excluded_tags") or params.get("exclude_tag")))
    repo_names = split_csv(params.get("repo_names") or params.get("repo"))
    has_types = split_csv(params.get("has_types") or params.get("has_type"))
    since_ms = params.get("since_ms")
    until_ms = params.get("until_ms")
    return {
        "origin": origin,
        "origins": origins,
        "excluded_origins": excluded_origins,
        "tags": tags,
        "excluded_tags": excluded_tags,
        "repo_names": repo_names,
        "has_types": has_types,
        "has_tool_use": _bool_param(params.get("has_tool_use") or params.get("filter_has_tool_use")),
        "has_thinking": _bool_param(params.get("has_thinking") or params.get("filter_has_thinking")),
        "has_paste": _bool_param(params.get("has_paste") or params.get("filter_has_paste")),
        "tool_terms": normalize_tool_terms(params.get("tool_terms") or params.get("tool")),
        "excluded_tool_terms": normalize_tool_terms(params.get("excluded_tool_terms") or params.get("exclude_tool")),
        "action_terms": normalize_action_terms("action", params.get("action_terms") or params.get("action")),
        "excluded_action_terms": normalize_action_terms(
            "exclude_action", params.get("excluded_action_terms") or params.get("exclude_action")
        ),
        "action_sequence": normalize_action_sequence(
            "action_sequence", params.get("action_sequence") or params.get("sequence")
        ),
        "action_text_terms": split_csv(params.get("action_text_terms") or params.get("action_text")),
        "referenced_paths": split_csv(params.get("referenced_paths") or params.get("referenced_path")),
        "cwd_prefix": optional_text(params.get("cwd_prefix")),
        "typed_only": _bool_param(params.get("typed_only")),
        "message_type": optional_message_type(params.get("message_type")),
        "title": optional_text(params.get("title")),
        "min_messages": optional_int(params.get("min_messages")),
        "max_messages": optional_int(params.get("max_messages")),
        "min_words": optional_int(params.get("min_words")),
        "max_words": optional_int(params.get("max_words")),
        "since_ms": int(since_ms) if isinstance(since_ms, int) else _epoch_ms("since", params.get("since")),
        "until_ms": int(until_ms) if isinstance(until_ms, int) else _epoch_ms("until", params.get("until")),
    }


def _object_ref_text(ref: ObjectRef | None) -> str | None:
    return ref.format() if ref is not None else None


def _evidence_ref_text(ref: EvidenceRef) -> str:
    return ref.format()


def _contains_value(haystacks: Iterable[str | None], needles: Sequence[str]) -> bool:
    if not needles:
        return True
    lowered_haystacks = [value.lower() for value in haystacks if value]
    return any(any(needle.lower() in haystack for haystack in lowered_haystacks) for needle in needles)


def _exact_or_contains(value: str | None, needles: Sequence[str], *, exact: bool = False) -> bool:
    if not needles:
        return True
    if value is None:
        return False
    lowered = value.lower()
    if exact:
        return any(lowered == needle.lower() for needle in needles)
    return any(needle.lower() in lowered for needle in needles)


def _numeric_field_matches(value: int | None, predicate: QueryFieldPredicate) -> bool:
    if value is None or not predicate.values:
        return False
    try:
        target = int(predicate.values[-1])
    except ValueError:
        return False
    if predicate.op == ">=":
        return value >= target
    if predicate.op == "<=":
        return value <= target
    return value == target


def _summary_timestamp_matches(summary: ArchiveSessionSummary, field: str, predicate: QueryFieldPredicate) -> bool:
    if not predicate.values:
        return False
    timestamp = _summary_updated_or_created_ms(summary)
    target = _epoch_ms(field, predicate.values[-1])
    if timestamp is None or target is None:
        return False
    if predicate.op == ">=":
        return timestamp >= target
    if predicate.op == "<=":
        return timestamp <= target
    return timestamp == target


def _summary_field_matches(summary: ArchiveSessionSummary, predicate: QueryFieldPredicate) -> bool:
    field = predicate.field.removeprefix("session.")
    values = predicate.values
    if field == "id":
        return _exact_or_contains(str(summary.session_id), values)
    if field == "origin":
        return _exact_or_contains(str(summary.origin), values, exact=True)
    if field == "repo":
        return _contains_value((summary.git_repository_url, *summary.working_directories), values)
    if field == "title":
        return _exact_or_contains(summary.title, values)
    if field == "tag":
        return _contains_value(summary.tags, values)
    if field == "cwd":
        return _contains_value(summary.working_directories, values)
    if field == "path":
        # Runtime-transform rows only carry session summaries here. Referenced
        # file paths are not in that projection, so fail closed instead of
        # treating path as cwd and broadening/missing results unpredictably.
        return False
    if field == "messages":
        return _numeric_field_matches(summary.message_count, predicate)
    if field == "words":
        return _numeric_field_matches(summary.word_count, predicate)
    if field == "date":
        return _summary_timestamp_matches(summary, field, predicate)
    if field == "since":
        return _summary_timestamp_matches(summary, field, QueryFieldPredicate(field=field, values=values, op=">="))
    if field == "until":
        return _summary_timestamp_matches(summary, field, QueryFieldPredicate(field=field, values=values, op="<="))
    return False


def _summary_updated_or_created_ms(summary: ArchiveSessionSummary) -> int | None:
    return _epoch_ms("updated_at", summary.updated_at) or _epoch_ms("created_at", summary.created_at)


def _observed_event_field_matches(
    event: ObservedEvent,
    summary: ArchiveSessionSummary,
    predicate: QueryFieldPredicate,
) -> bool:
    field = predicate.field
    if field.startswith("session."):
        return _summary_field_matches(summary, predicate)
    if field == "kind":
        return _exact_or_contains(event.kind, predicate.values, exact=True)
    if field == "delivery_state":
        return _exact_or_contains(event.delivery_state, predicate.values, exact=True)
    if field == "summary":
        return _contains_value((event.summary,), predicate.values)
    if field == "text":
        return _contains_value(
            (
                event.summary,
                _object_ref_text(event.subject_ref),
                *(_object_ref_text(ref) for ref in event.object_refs),
                *(_evidence_ref_text(ref) for ref in event.evidence_refs),
            ),
            predicate.values,
        )
    if field in {"subject", "subject_ref"}:
        return _contains_value((_object_ref_text(event.subject_ref),), predicate.values)
    if field in {"object", "object_ref"}:
        return _contains_value((_object_ref_text(ref) for ref in event.object_refs), predicate.values)
    if field == "evidence":
        return _contains_value((_evidence_ref_text(ref) for ref in event.evidence_refs), predicate.values)
    return False


def _observed_event_matches(
    event: ObservedEvent,
    summary: ArchiveSessionSummary,
    predicate: QueryPredicate,
) -> bool:
    if isinstance(predicate, QueryFieldPredicate):
        return _observed_event_field_matches(event, summary, predicate)
    if isinstance(predicate, QueryNotPredicate):
        return not _observed_event_matches(event, summary, predicate.child)
    if isinstance(predicate, QueryBoolPredicate):
        if predicate.op == "and":
            return all(_observed_event_matches(event, summary, child) for child in predicate.children)
        return any(_observed_event_matches(event, summary, child) for child in predicate.children)
    return False


def _observed_event_row(summary: ArchiveSessionSummary, event: ObservedEvent) -> ObservedEventQueryRowPayload:
    return ObservedEventQueryRowPayload(
        event_ref=event.event_ref.format(),
        session_id=str(summary.session_id),
        origin=str(summary.origin),
        title=summary.title,
        kind=event.kind,
        summary=event.summary,
        delivery_state=event.delivery_state,
        subject_ref=_object_ref_text(event.subject_ref),
        object_refs=tuple(ref.format() for ref in event.object_refs),
        evidence_refs=tuple(ref.format() for ref in event.evidence_refs),
    )


def _metadata_text(metadata: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(f"{key}:{value}" for key, value in sorted(metadata.items()))


def _context_snapshot_field_matches(
    snapshot: ContextSnapshot,
    summary: ArchiveSessionSummary,
    predicate: QueryFieldPredicate,
) -> bool:
    field = predicate.field
    if field.startswith("session."):
        return _summary_field_matches(summary, predicate)
    if field == "boundary":
        return _exact_or_contains(snapshot.boundary, predicate.values, exact=True)
    if field == "inheritance_mode":
        return _exact_or_contains(snapshot.inheritance_mode, predicate.values, exact=True)
    if field in {"run", "run_ref"}:
        return _contains_value((_object_ref_text(snapshot.run_ref),), predicate.values)
    if field in {"segment", "segment_ref"}:
        return _contains_value((_object_ref_text(ref) for ref in snapshot.segment_refs), predicate.values)
    if field == "evidence":
        return _contains_value((_evidence_ref_text(ref) for ref in snapshot.evidence_refs), predicate.values)
    if field == "metadata":
        return _contains_value(_metadata_text(snapshot.metadata), predicate.values)
    if field == "text":
        return _contains_value(
            (
                _object_ref_text(snapshot.snapshot_ref),
                _object_ref_text(snapshot.run_ref),
                snapshot.boundary,
                snapshot.inheritance_mode,
                *(_object_ref_text(ref) for ref in snapshot.segment_refs),
                *(_evidence_ref_text(ref) for ref in snapshot.evidence_refs),
                *_metadata_text(snapshot.metadata),
            ),
            predicate.values,
        )
    return False


def _context_snapshot_matches(
    snapshot: ContextSnapshot,
    summary: ArchiveSessionSummary,
    predicate: QueryPredicate,
) -> bool:
    if isinstance(predicate, QueryFieldPredicate):
        return _context_snapshot_field_matches(snapshot, summary, predicate)
    if isinstance(predicate, QueryNotPredicate):
        return not _context_snapshot_matches(snapshot, summary, predicate.child)
    if isinstance(predicate, QueryBoolPredicate):
        if predicate.op == "and":
            return all(_context_snapshot_matches(snapshot, summary, child) for child in predicate.children)
        return any(_context_snapshot_matches(snapshot, summary, child) for child in predicate.children)
    return False


def _context_snapshot_row(
    summary: ArchiveSessionSummary,
    snapshot: ContextSnapshot,
) -> ContextSnapshotQueryRowPayload:
    return ContextSnapshotQueryRowPayload(
        snapshot_ref=snapshot.snapshot_ref.format(),
        session_id=str(summary.session_id),
        origin=str(summary.origin),
        title=summary.title,
        run_ref=snapshot.run_ref.format(),
        boundary=snapshot.boundary,
        inheritance_mode=snapshot.inheritance_mode,
        segment_refs=tuple(ref.format() for ref in snapshot.segment_refs),
        evidence_refs=tuple(ref.format() for ref in snapshot.evidence_refs),
        metadata=dict(snapshot.metadata),
    )


def _iter_filtered_summaries(
    archive: ArchiveStore,
    session_filters: Mapping[str, object] | None,
) -> Iterable[ArchiveSessionSummary]:
    """Yield matching summaries without imposing an arbitrary archive cap."""

    summary_offset = 0
    filters = dict(session_filters or {})
    while True:
        summaries = cast(Any, archive.list_summaries)(
            limit=_SUMMARY_SCAN_BATCH_SIZE,
            offset=summary_offset,
            **filters,
        )
        if not summaries:
            return
        yield from summaries
        if len(summaries) < _SUMMARY_SCAN_BATCH_SIZE:
            return
        summary_offset += len(summaries)


def _query_context_snapshots(
    archive: ArchiveStore,
    source: QueryUnitSource,
    *,
    limit: int,
    offset: int,
    session_filters: Mapping[str, object] | None,
) -> tuple[ContextSnapshotQueryRowPayload, ...]:
    rows: list[ContextSnapshotQueryRowPayload] = []
    target_count = limit + 1
    skipped = 0
    for summary in _iter_filtered_summaries(archive, session_filters):
        session = _session_to_session(archive.read_session(str(summary.session_id)))
        digest = compile_recovery_digest(session)
        for snapshot in digest.run_projection.context_snapshots:
            if not _context_snapshot_matches(snapshot, summary, source.predicate):
                continue
            if skipped < offset:
                skipped += 1
                continue
            rows.append(_context_snapshot_row(summary, snapshot))
            if len(rows) >= target_count:
                return tuple(rows)
    return tuple(rows)


def _query_observed_events(
    archive: ArchiveStore,
    source: QueryUnitSource,
    *,
    limit: int,
    offset: int,
    session_filters: Mapping[str, object] | None,
) -> tuple[ObservedEventQueryRowPayload, ...]:
    rows: list[ObservedEventQueryRowPayload] = []
    target_count = limit + 1
    skipped = 0
    for summary in _iter_filtered_summaries(archive, session_filters):
        session = _session_to_session(archive.read_session(str(summary.session_id)))
        digest = compile_recovery_digest(session)
        for event in digest.run_projection.events:
            if not _observed_event_matches(event, summary, source.predicate):
                continue
            if skipped < offset:
                skipped += 1
                continue
            rows.append(_observed_event_row(summary, event))
            if len(rows) >= target_count:
                return tuple(rows)
    return tuple(rows)


def query_unit_rows(
    archive: ArchiveStore,
    source: QueryUnitSource,
    *,
    query: str,
    limit: int,
    offset: int = 0,
    session_filters: Mapping[str, object] | None = None,
) -> QueryUnitEnvelope:
    """Execute an explicit unit-source query."""

    fetch_limit = limit + 1
    if source.unit == "observed-event":
        event_rows = _query_observed_events(
            archive,
            source,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
        )
        return build_query_unit_envelope(
            event_rows[:limit],
            unit=source.unit,
            query=query,
            limit=limit,
            offset=offset,
            has_next=len(event_rows) > limit,
        )
    if source.unit == "context-snapshot":
        snapshot_rows = _query_context_snapshots(
            archive,
            source,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
        )
        return build_query_unit_envelope(
            snapshot_rows[:limit],
            unit=source.unit,
            query=query,
            limit=limit,
            offset=offset,
            has_next=len(snapshot_rows) > limit,
        )
    if source.unit == "message":
        message_rows = archive.query_messages(
            source.predicate,
            limit=fetch_limit,
            offset=offset,
            session_filters=session_filters,
        )
        return build_query_unit_envelope(
            tuple(MessageQueryRowPayload.from_row(row) for row in message_rows[:limit]),
            unit=source.unit,
            query=query,
            limit=limit,
            offset=offset,
            has_next=len(message_rows) > limit,
        )
    if source.unit == "action":
        action_rows = archive.query_actions(
            source.predicate,
            limit=fetch_limit,
            offset=offset,
            session_filters=session_filters,
        )
        return build_query_unit_envelope(
            tuple(ActionQueryRowPayload.from_row(row) for row in action_rows[:limit]),
            unit=source.unit,
            query=query,
            limit=limit,
            offset=offset,
            has_next=len(action_rows) > limit,
        )
    if source.unit == "assertion":
        assertion_rows = archive.query_assertions(
            source.predicate,
            limit=fetch_limit,
            offset=offset,
            session_filters=session_filters,
        )
        return build_query_unit_envelope(
            tuple(AssertionQueryRowPayload.from_row(row) for row in assertion_rows[:limit]),
            unit=source.unit,
            query=query,
            limit=limit,
            offset=offset,
            has_next=len(assertion_rows) > limit,
        )
    block_rows = archive.query_blocks(
        source.predicate,
        limit=fetch_limit,
        offset=offset,
        session_filters=session_filters,
    )
    return build_query_unit_envelope(
        tuple(BlockQueryRowPayload.from_row(row) for row in block_rows[:limit]),
        unit=source.unit,
        query=query,
        limit=limit,
        offset=offset,
        has_next=len(block_rows) > limit,
    )


__all__ = ["query_unit_rows", "query_unit_session_filters"]
