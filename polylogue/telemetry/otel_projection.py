"""Outbound OTel-style projection over Polylogue evidence rows.

OpenTelemetry remains an export format here.  The archive truth is still the
session/message/action/run/context/assertion/ref substrate; this module only
maps those rows into a bounded observability-shaped payload for external tools.
"""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256

from polylogue.core.json import require_json_document
from polylogue.core.refs import normalize_public_ref_text
from polylogue.surfaces.payloads import (
    ActionQueryRowPayload,
    ContextSnapshotQueryRowPayload,
    MessageQueryRowPayload,
    ObservedEventQueryRowPayload,
    OtelLogRecordPayload,
    OtelProjectionPayload,
    OtelSpanPayload,
    QueryUnitRowPayload,
    RunQueryRowPayload,
)

_LOCAL_PATH_PREFIXES = ("/home/", "/Users/", "/realm/", "/var/", "/etc/")


def project_query_unit_rows_to_otel(
    source_ref: str,
    rows: Iterable[QueryUnitRowPayload],
    *,
    include_message_text: bool = False,
) -> OtelProjectionPayload:
    """Project terminal query-unit rows into an OTel-like JSON payload.

    ``include_message_text`` is intentionally opt-in. Tool outputs and absolute
    local paths are never copied into the projection; consumers can follow the
    Polylogue refs when they need the underlying evidence.
    """

    normalized_source_ref = normalize_public_ref_text(source_ref)
    trace_id = _trace_id(normalized_source_ref)
    spans: list[OtelSpanPayload] = []
    logs: list[OtelLogRecordPayload] = []
    refs: list[str] = [normalized_source_ref]
    run_span_ids: dict[str, str] = {}

    for row in rows:
        if isinstance(row, RunQueryRowPayload):
            span = _run_span(trace_id, row)
            spans.append(span)
            run_span_ids[row.run_ref] = span.span_id
            refs.extend([row.run_ref, *row.evidence_refs])
            if row.context_snapshot_ref:
                refs.append(row.context_snapshot_ref)
            continue
        if isinstance(row, ActionQueryRowPayload):
            span = _action_span(trace_id, row)
            spans.append(span)
            refs.extend(_action_refs(row))
            continue
        if isinstance(row, ObservedEventQueryRowPayload):
            logs.append(_observed_event_log(trace_id, row))
            refs.extend([row.event_ref, *row.object_refs, *row.evidence_refs])
            continue
        if isinstance(row, ContextSnapshotQueryRowPayload):
            logs.append(_context_snapshot_log(trace_id, row))
            refs.extend([row.snapshot_ref, row.run_ref, *row.segment_refs, *row.evidence_refs])
            continue
        if isinstance(row, MessageQueryRowPayload):
            logs.append(_message_log(trace_id, row, include_text=include_message_text))
            refs.append(f"message:{row.message_id}")

    caveats = (
        "OTel ids are projection ids; Polylogue refs remain canonical.",
        "Tool outputs and absolute local paths are omitted from outbound telemetry by default.",
    )
    return OtelProjectionPayload(
        source_ref=normalized_source_ref,
        trace_count=1 if spans or logs else 0,
        span_count=len(spans),
        log_count=len(logs),
        spans=tuple(spans),
        logs=tuple(logs),
        refs=tuple(dict.fromkeys(refs)),
        caveats=caveats,
    )


def _run_span(trace_id: str, row: RunQueryRowPayload) -> OtelSpanPayload:
    attributes: dict[str, object] = {
        "polylogue.run.ref": row.run_ref,
        "polylogue.session.id": row.session_id,
        "polylogue.origin": row.origin,
        "polylogue.run.role": row.role,
        "polylogue.run.status": row.status,
        "polylogue.run.harness": row.harness,
        "polylogue.run.provider_origin": row.provider_origin,
        "polylogue.run.confidence": row.confidence,
    }
    _set_optional(attributes, "polylogue.run.agent_ref", row.agent_ref)
    _set_optional(attributes, "polylogue.run.git_branch", row.git_branch)
    _set_optional(attributes, "polylogue.run.cwd.redacted", True if _is_local_path(row.cwd) else None)
    _set_optional(attributes, "polylogue.run.context_snapshot_ref", row.context_snapshot_ref)
    links = tuple(ref for ref in (*row.lineage_refs, *row.evidence_refs) if ref)
    if row.context_snapshot_ref:
        links = (*links, row.context_snapshot_ref)
    return OtelSpanPayload(
        trace_id=trace_id,
        span_id=_span_id(row.run_ref),
        parent_span_id=_span_id(row.parent_run_ref) if row.parent_run_ref else None,
        name=f"polylogue.run {row.role}",
        attributes=attributes,
        links=links,
    )


def _action_span(trace_id: str, row: ActionQueryRowPayload) -> OtelSpanPayload:
    attributes: dict[str, object] = {
        "polylogue.session.id": row.session_id,
        "polylogue.origin": row.origin,
        "polylogue.message.id": row.message_id,
        "polylogue.action.tool_use_block_id": row.tool_use_block_id,
        "polylogue.action.output_present": row.output_text is not None,
    }
    _set_optional(attributes, "polylogue.action.tool_result_block_id", row.tool_result_block_id)
    _set_optional(attributes, "polylogue.action.tool_name", row.tool_name)
    _set_optional(attributes, "polylogue.action.semantic_type", row.semantic_type)
    _set_optional(attributes, "polylogue.action.tool_command", row.tool_command)
    _set_optional(attributes, "polylogue.action.output_length", len(row.output_text) if row.output_text else None)
    if row.tool_path and _is_local_path(row.tool_path):
        attributes["polylogue.action.tool_path.redacted"] = True
    elif row.tool_path:
        attributes["polylogue.action.tool_path"] = row.tool_path
    return OtelSpanPayload(
        trace_id=trace_id,
        span_id=_span_id(f"action:{row.session_id}:{row.tool_use_block_id}"),
        name=f"polylogue.action {row.tool_name or row.semantic_type or 'tool'}",
        attributes=attributes,
        links=tuple(_action_refs(row)),
    )


def _observed_event_log(trace_id: str, row: ObservedEventQueryRowPayload) -> OtelLogRecordPayload:
    attributes: dict[str, object] = {
        "polylogue.observed_event.ref": row.event_ref,
        "polylogue.session.id": row.session_id,
        "polylogue.origin": row.origin,
        "polylogue.observed_event.kind": row.kind,
        "polylogue.observed_event.delivery_state": row.delivery_state,
    }
    _set_optional(attributes, "polylogue.observed_event.subject_ref", row.subject_ref)
    return OtelLogRecordPayload(
        trace_id=trace_id,
        body=row.summary,
        attributes=attributes,
        links=tuple(ref for ref in (row.event_ref, *row.object_refs, *row.evidence_refs) if ref),
    )


def _context_snapshot_log(trace_id: str, row: ContextSnapshotQueryRowPayload) -> OtelLogRecordPayload:
    metadata = require_json_document(dict(row.metadata), context="context snapshot metadata")
    attributes: dict[str, object] = {
        "polylogue.context_snapshot.ref": row.snapshot_ref,
        "polylogue.session.id": row.session_id,
        "polylogue.origin": row.origin,
        "polylogue.run.ref": row.run_ref,
        "polylogue.context_snapshot.boundary": row.boundary,
        "polylogue.context_snapshot.inheritance_mode": row.inheritance_mode,
        "polylogue.context_snapshot.metadata": metadata,
    }
    return OtelLogRecordPayload(
        trace_id=trace_id,
        body=f"context snapshot {row.boundary}",
        attributes=attributes,
        links=tuple(ref for ref in (row.snapshot_ref, row.run_ref, *row.segment_refs, *row.evidence_refs) if ref),
    )


def _message_log(trace_id: str, row: MessageQueryRowPayload, *, include_text: bool) -> OtelLogRecordPayload:
    attributes: dict[str, object] = {
        "polylogue.message.id": row.message_id,
        "polylogue.session.id": row.session_id,
        "polylogue.origin": row.origin,
        "polylogue.message.role": row.role,
        "polylogue.message.type": row.message_type,
        "polylogue.message.position": row.position,
        "polylogue.message.word_count": row.word_count,
        "polylogue.message.text_included": include_text,
    }
    body = row.text if include_text else f"message {row.role}"
    return OtelLogRecordPayload(
        trace_id=trace_id,
        body=body,
        attributes=attributes,
        links=(f"message:{row.message_id}",),
    )


def _action_refs(row: ActionQueryRowPayload) -> tuple[str, ...]:
    refs = [f"message:{row.message_id}", f"block:{row.tool_use_block_id}"]
    if row.tool_result_block_id:
        refs.append(f"block:{row.tool_result_block_id}")
    return tuple(refs)


def _trace_id(ref: str) -> str:
    return sha256(ref.encode("utf-8")).hexdigest()[:32]


def _span_id(ref: str) -> str:
    return sha256(ref.encode("utf-8")).hexdigest()[:16]


def _set_optional(attributes: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        attributes[key] = value


def _is_local_path(value: str | None) -> bool:
    return bool(value and value.startswith(_LOCAL_PATH_PREFIXES))


__all__ = ["project_query_unit_rows_to_otel"]
