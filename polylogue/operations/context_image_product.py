"""Context-image product read over one supplied archive generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast, get_args

from polylogue.archive.context_models import (
    DEFAULT_CONTEXT_IMAGE_MAX_CHARS_PER_MESSAGE,
    DEFAULT_CONTEXT_IMAGE_MAX_MESSAGES_PER_SESSION,
    ContextImage,
    ContextOmission,
    ContextSegmentProfile,
    ContextSpec,
)
from polylogue.archive.hydration import archive_envelope_to_session, archive_summary_to_domain
from polylogue.archive.query.predicate import QueryFieldPredicate, QueryFieldRef
from polylogue.context.product_image import compile_context_image
from polylogue.core.async_bridge import complete_without_suspension
from polylogue.core.enums import Origin
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.chronicle import (
    ChronicleProjectionPayload,
    build_chronicle_projection_payload,
    build_chronicle_session_payload,
)
from polylogue.surfaces.payloads import AssertionClaimPayload
from polylogue.surfaces.projection_spec import projection_from_views
from polylogue.surfaces.temporal_evidence import (
    TemporalEvidenceEvent,
    TemporalEvidenceWindow,
    action_row_to_temporal_event,
    build_temporal_evidence_window,
    message_row_to_temporal_event,
    summary_to_temporal_event,
)

if TYPE_CHECKING:
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.archive.session.domain_models import SessionSummary

#: A context segment is a bounded excerpt of one session's temporal evidence,
#: not its full timeline; the caveats on the window name each cut it makes.
CONTEXT_TEMPORAL_MESSAGE_EVENTS = 8
CONTEXT_TEMPORAL_ACTION_EVENTS = 4
CONTEXT_CHRONICLE_EDGE_LIMIT = 8


def context_temporal_window(archive: ArchiveStore, summary: SessionSummary) -> TemporalEvidenceWindow:
    """The temporal context excerpt for one session, read from ``archive``."""
    session_id = str(summary.id)
    events: list[TemporalEvidenceEvent] = []
    if session_event := summary_to_temporal_event(summary):
        events.append(session_event)
    predicate = QueryFieldPredicate(field="session.id", values=(session_id,), op="=").with_field_ref(
        QueryFieldRef(scope="session", name="id", source_name="session.id")
    )
    message_rows = archive.query_messages(
        predicate, limit=CONTEXT_TEMPORAL_MESSAGE_EVENTS, sort="time", sort_direction="asc"
    )
    action_rows = archive.query_session_actions(
        [session_id], limit=CONTEXT_TEMPORAL_ACTION_EVENTS, sort_direction="asc"
    )
    events.extend(event for row in message_rows if (event := message_row_to_temporal_event(row)) is not None)
    events.extend(event for row in action_rows if (event := action_row_to_temporal_event(row)) is not None)
    caveats: list[str] = []
    if (
        len(message_rows) >= CONTEXT_TEMPORAL_MESSAGE_EVENTS
        and (summary.message_count or 0) > CONTEXT_TEMPORAL_MESSAGE_EVENTS
    ):
        caveats.append("message_events_capped")
    if len(action_rows) >= CONTEXT_TEMPORAL_ACTION_EVENTS:
        caveats.append("action_events_capped")
    return build_temporal_evidence_window(events, caveats=caveats)


def context_chronicle_payload(
    archive: ArchiveStore, summary: SessionSummary, *, edge_limit: int = CONTEXT_CHRONICLE_EDGE_LIMIT
) -> ChronicleProjectionPayload:
    """The chronicle context excerpt for one session, with lineage-composed edges."""
    from polylogue.operations.read_view_chronicle import chronicle_edges

    first, last, total = chronicle_edges(
        archive, str(summary.id), edge_limit, origin=Origin.from_string(summary.origin)
    )
    session = build_chronicle_session_payload(
        summary, first_messages=first, last_messages=last, total_matching_messages=total, edge_limit=edge_limit
    )
    return build_chronicle_projection_payload([session], edge_limit=edge_limit)


class PinnedContextImageSource:
    """Archive evidence for image composition, bound to a supplied reader."""

    def __init__(self, archive: ArchiveStore, *, observed_at_ms: int) -> None:
        self.archive = archive
        self.observed_at_ms = observed_at_ms

    async def _compile_context_seed_query(
        self, spec: ContextSpec
    ) -> tuple[list[str], dict[str, str], list[ContextOmission]]:
        from polylogue.api.archive import _archive_list_summaries_for_spec
        from polylogue.context.selection import clamp_context_image_limit, select_context_image_sessions

        has_filters = any(
            (spec.seed_project_path, spec.seed_project_repo, spec.seed_since, spec.seed_until, spec.seed_origin)
        )
        if has_filters or spec.seed_query == "":

            async def query_sessions(query_spec: SessionQuerySpec) -> list[object]:
                return [
                    archive_summary_to_domain(summary)
                    for summary in _archive_list_summaries_for_spec(
                        self.archive, query_spec, default_limit=spec.seed_query_limit
                    )
                ]

            selection = await select_context_image_sessions(
                query_sessions,
                clamp_context_image_limit,
                project_path=spec.seed_project_path,
                project_repo=spec.seed_project_repo,
                since=spec.seed_since,
                until=spec.seed_until,
                origin=spec.seed_origin,
                query=spec.seed_query or None,
                limit=spec.seed_query_limit,
            )
            if selection.sessions:
                return [str(summary.id) for summary in selection.sessions[: spec.seed_query_limit]], {}, []
            return (
                [],
                {},
                [
                    ContextOmission(
                        query=spec.seed_query or None, reason="not_found", detail="seed selection matched no sessions"
                    )
                ],
            )
        if spec.seed_query is None:
            return [], {}, []
        hits = self.archive.search_summaries(spec.seed_query, limit=spec.seed_query_limit)
        if not hits:
            return (
                [],
                {},
                [ContextOmission(query=spec.seed_query, reason="not_found", detail="seed query matched no sessions")],
            )
        anchors: dict[str, str] = {}
        for hit in hits:
            if hit.message_id:
                anchors.setdefault(hit.session_id, hit.message_id)
        return [hit.session_id for hit in hits], anchors, []

    async def query_units(self, expression: str, *, limit: int) -> object:
        raise ValueError("query-unit context is not supported by the context-image read operation")

    async def get_session(self, session_id: str) -> object | None:
        try:
            resolved = self.archive.resolve_session_id(session_id)
            summary = self.archive.read_summary(resolved)
            return archive_envelope_to_session(
                self.archive.read_session(resolved),
                display_label=summary.display_label,
                display_label_source=summary.display_label_source,
            )
        except KeyError:
            return None

    async def get_session_summary(self, session_id: str) -> object | None:
        try:
            return archive_summary_to_domain(self.archive.read_summary(self.archive.resolve_session_id(session_id)))
        except KeyError:
            return None

    def _context_temporal_window(self, summary: SessionSummary) -> TemporalEvidenceWindow:
        return context_temporal_window(self.archive, summary)

    async def _context_chronicle_payload(self, summary: SessionSummary) -> ChronicleProjectionPayload:
        return context_chronicle_payload(self.archive, summary)

    async def list_assertion_claim_payloads(
        self, *, target_ref: str, statuses: tuple[str, ...], context_inject: bool
    ) -> list[AssertionClaimPayload]:
        from polylogue.storage.sqlite.archive_tiers.user_write import (
            _ASSERTION_COLUMNS,
            ASSERTION_CLAIM_KINDS,
            _assertion_row_to_envelope,
        )

        self.archive.require_user_tier()
        connection = self.archive.index_connection
        if connection is None:
            raise ValueError("context image requires an index snapshot")
        kinds = tuple(str(kind.value) for kind in ASSERTION_CLAIM_KINDS)
        kind_slots = ", ".join("?" for _ in kinds)
        status_slots = ", ".join("?" for _ in statuses)
        rows = connection.execute(
            f"SELECT {_ASSERTION_COLUMNS} FROM user_tier.assertions "
            f"WHERE kind IN ({kind_slots}) AND target_ref = ? "
            f"AND COALESCE(status, 'active') IN ({status_slots}) "
            "AND (staleness_json IS NULL OR json_extract(staleness_json, '$.expires_at_ms') IS NULL "
            "OR json_extract(staleness_json, '$.expires_at_ms') > ?) "
            "ORDER BY updated_at_ms DESC, assertion_id",
            (*kinds, target_ref, *statuses, self.observed_at_ms),
        ).fetchall()
        claims = (_assertion_row_to_envelope(row) for row in rows)
        return [
            AssertionClaimPayload.from_envelope(claim)
            for claim in claims
            if bool(claim.context_policy.get("inject")) is context_inject
        ]


def context_image_from_pinned_reader(payload: Mapping[str, Any], *, archive: ArchiveStore) -> ContextImage:
    """Compile the public context-image lens without changing archive state."""
    seed_session_id = payload.get("seed_session_id")
    max_sessions = max(1, min(int(payload.get("max_sessions", 5)), 20))
    max_tokens = payload.get("max_tokens")
    query = payload.get("query")
    include_messages = bool(payload.get("include_messages", True))
    include_assertions = bool(payload.get("include_assertions", True))
    redact_paths = bool(payload.get("redact_paths", True))
    read_views = payload.get("read_views")
    views: tuple[str, ...] = (
        tuple(str(view) for view in read_views)
        if read_views is not None
        else (("messages",) if include_messages else ())
    )
    seed_session_ids = payload.get("seed_session_ids") or ()
    if seed_session_ids:
        seed_session_ids = tuple(seed_session_ids)[:max_sessions]
    seed_refs = (
        tuple(f"session:{session_id}" for session_id in seed_session_ids)
        if seed_session_ids
        else ((f"session:{seed_session_id}",) if seed_session_id is not None else ())
    )
    spec = ContextSpec(
        purpose=payload.get("purpose", "handoff"),
        seed_refs=seed_refs,
        seed_query=query if query is not None else ("" if not seed_refs else None),
        seed_query_limit=max_sessions,
        seed_project_path=payload.get("project_path"),
        seed_project_repo=payload.get("project_repo"),
        seed_since=payload.get("since"),
        seed_until=payload.get("until"),
        seed_origin=payload.get("origin"),
        read_views=views,
        max_tokens=max_tokens,
        max_messages_per_session=payload.get(
            "max_messages_per_session", DEFAULT_CONTEXT_IMAGE_MAX_MESSAGES_PER_SESSION
        ),
        max_chars_per_message=payload.get("max_chars_per_message", DEFAULT_CONTEXT_IMAGE_MAX_CHARS_PER_MESSAGE),
        include_assertions=include_assertions,
        redaction_policy="default" if redact_paths else "raw-opt-in",
        segment_profile=_segment_profile(payload.get("segment_profile", "default")),
    )
    if include_assertions:
        archive.require_user_tier()
    image = complete_without_suspension(
        compile_context_image(
            PinnedContextImageSource(archive, observed_at_ms=int(payload.get("observed_at_ms", 0))), spec
        )
    )
    projection = projection_from_views(
        ("context-image",),
        format="json",
        destination="stdout",
        layout="context-image",
        max_tokens=max_tokens,
        query=query,
        origin=payload.get("origin"),
        since=payload.get("since"),
        until=payload.get("until"),
        project_path=payload.get("project_path"),
        project_repo=payload.get("project_repo"),
        limit=max_sessions,
    )
    if seed_refs:
        selection = projection.selection.model_copy(update={"refs": seed_refs})
        projection = projection.model_copy(update={"selection": selection})
    return image.model_copy(update={"projection_spec": projection})


__all__ = ["context_image_from_pinned_reader"]


def _segment_profile(value: object) -> ContextSegmentProfile:
    """Refuse an undeclared segment profile instead of passing it through."""
    if value not in get_args(ContextSegmentProfile):
        raise ValueError(f"unsupported segment_profile: {value!r}")
    return cast("ContextSegmentProfile", value)
