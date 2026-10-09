"""Pinned archive projections for dialogue and temporal read views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

from polylogue.archive.hydration import (
    archive_envelope_to_session,
    archive_message_to_domain,
    archive_summary_to_domain,
)
from polylogue.archive.query.filter_kwargs import datetime_to_ms
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.core.enums import Origin
from polylogue.operations.query_lowering import cli_query_spec
from polylogue.operations.read_view_selection import select_read_view_summaries
from polylogue.operations.session_contracts import SessionRead
from polylogue.operations.transcript_window import read_transcript_window_sync
from polylogue.surfaces.temporal_evidence import (
    TemporalEvidenceEvent,
    action_row_to_temporal_event,
    build_temporal_evidence_window,
    message_row_to_temporal_event,
    summary_to_temporal_event,
)

if TYPE_CHECKING:
    from polylogue.archive.session.domain_models import SessionSummary
    from polylogue.core.protocols import VectorProvider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _request_params(payload: Mapping[str, object]) -> dict[str, object]:
    raw = payload.get("params", {})
    if not isinstance(raw, Mapping):
        raise ValueError("read view params must be an object")
    return {str(key): value for key, value in raw.items()}


def execute_dialogue_read(payload: Mapping[str, object], *, archive: ArchiveStore) -> dict[str, object]:
    """Return one composed transcript page, filtered to authored prose."""

    session_ref = payload.get("session_id")
    if not isinstance(session_ref, str) or not session_ref:
        raise ValueError("read.dialogue requires a session reference")
    offset = payload.get("offset", 0)
    limit = payload.get("limit", 100)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("read.dialogue offset must be a non-negative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 200:
        raise ValueError("read.dialogue limit must be between 1 and 200")
    continuation = payload.get("continuation")
    if continuation is not None and (not isinstance(continuation, str) or not continuation):
        raise ValueError("read.dialogue continuation must be a non-empty string")
    try:
        session_id = archive.resolve_session_id(session_ref)
    except KeyError:
        return {"view": "dialogue", "payload": {"session": None, "total_message_count": 0, "next_offset": None}}
    summary = archive.read_summary(session_id)
    request = SessionRead.model_validate(
        {"ref": session_ref, "continuation": continuation}
        if continuation
        else {"ref": session_ref, "limit": limit, "offset": offset}
    )
    latest_envelope = None

    def read(window_limit: int, window_offset: int) -> tuple[list[object], int, object]:
        nonlocal latest_envelope
        latest_envelope = archive.read_session_page(session_id, limit=window_limit, offset=window_offset)
        total = latest_envelope.total_message_count
        if total is None:
            raise ValueError("read.dialogue page omitted its composed message count")
        return (
            list(latest_envelope.messages),
            total,
            SimpleNamespace(
                complete=latest_envelope.lineage_complete,
                truncation_reason=latest_envelope.lineage_truncation_reason,
            ),
        )

    window = read_transcript_window_sync(
        archive,
        request,
        read=read,
        transaction_operation="read.dialogue",
        projection="dialogue-prose-v1",
    )
    assert latest_envelope is not None
    envelope = replace(latest_envelope, messages=tuple(window.rows))
    session = archive_envelope_to_session(
        envelope,
        display_label=summary.display_label,
        display_label_source=summary.display_label_source,
    ).with_content_projection(ContentProjectionSpec.prose_only())
    return {
        "view": "dialogue",
        "payload": {
            "session": session.model_dump(mode="json"),
            "total_message_count": window.total,
            "next_offset": window.next_offset,
            "continuation": window.continuation,
        },
    }


_EVIDENCE_PAGE_SIZE = 256


def _occurrence_events(
    archive: ArchiveStore, summaries: list[SessionSummary]
) -> tuple[list[TemporalEvidenceEvent], tuple[str, ...]]:
    """Page every composed occurrence and retain the read's named lineage gaps."""
    events: dict[str, TemporalEvidenceEvent] = {}
    gaps: list[str] = []
    for summary in summaries:
        offset = 0
        while True:
            archive.check_operation_read()
            page = archive.read_session_page(str(summary.id), limit=_EVIDENCE_PAGE_SIZE, offset=offset)
            if not page.lineage_complete:
                gaps.append(f"lineage_truncated:{page.lineage_truncation_reason or 'unknown'}")
            for message in page.messages:
                domain = archive_message_to_domain(message, origin=Origin.from_string(summary.origin))
                event = message_row_to_temporal_event(
                    SimpleNamespace(
                        message_id=message.message_id,
                        session_id=message.source_session_id or str(summary.id),
                        occurred_at_ms=datetime_to_ms(domain.timestamp),
                        role=message.role,
                        message_type=message.message_type,
                        position=message.position,
                        text=domain.text,
                    )
                )
                if event is not None:
                    events[event.event_id] = event
            message_ids = [message.message_id for message in page.messages]
            source_ids = list(dict.fromkeys(message.source_session_id or str(summary.id) for message in page.messages))
            action_offset = 0
            while message_ids:
                archive.check_operation_read()
                rows = archive.query_session_action_occurrences(
                    source_ids,
                    message_ids=message_ids,
                    limit=_EVIDENCE_PAGE_SIZE,
                    offset=action_offset,
                    sort_direction="asc",
                )
                for row in rows:
                    event = action_row_to_temporal_event(row)
                    if event is not None:
                        events[event.event_id] = event
                if len(rows) < _EVIDENCE_PAGE_SIZE:
                    break
                action_offset += len(rows)
            offset += len(page.messages)
            if not page.messages or (page.total_message_count is not None and offset >= page.total_message_count):
                break
    return list(events.values()), tuple(dict.fromkeys(gaps))


def execute_temporal_read(
    payload: Mapping[str, object], *, archive: ArchiveStore, vector_provider: VectorProvider | None = None
) -> dict[str, object]:
    """Build a temporal window from one pinned archive selection."""

    params = _request_params(payload)
    spec = cli_query_spec(params)
    if spec.limit is None:
        spec = replace(spec, limit=50)
    session_ref = payload.get("session_id")
    if isinstance(session_ref, str) and session_ref:
        try:
            summaries = [archive_summary_to_domain(archive.read_summary(archive.resolve_session_id(session_ref)))]
        except KeyError:
            summaries = []
    else:
        plan = spec.to_plan(vector_provider=vector_provider)
        summaries = select_read_view_summaries(plan, archive=archive, default_limit=50)
    session_events = [event for summary in summaries if (event := summary_to_temporal_event(summary)) is not None]
    occurrence_events, gaps = _occurrence_events(archive, summaries)
    window = build_temporal_evidence_window([*session_events, *occurrence_events], gaps=gaps)
    return {"view": "temporal", "payload": {"temporal_window": window.model_dump(mode="json")}}


__all__ = ["execute_dialogue_read", "execute_temporal_read"]
