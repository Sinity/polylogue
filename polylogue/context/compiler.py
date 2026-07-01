"""Successor-context compiler over query and archive evidence.

This module is intentionally thin.  It does not introduce a durable memory
store or a new handoff ontology; it composes selected archive refs, terminal
query-unit rows, and optional report transforms into one compiled context image
that CLI/API/MCP surfaces can share.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from contextlib import suppress
from typing import Literal, cast

from pydantic import Field, model_validator

from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.surfaces.chronicle import ChronicleProjectionPayload, render_chronicle_markdown
from polylogue.surfaces.temporal_evidence import TemporalEvidenceWindow

ContextPurpose = Literal["continue", "review", "handoff", "debug", "export"]
ContextSegmentKind = Literal["read_view", "query_unit", "assertion", "caveat"]
ContextOmissionReason = Literal["budget", "unsupported", "not_found", "policy", "redacted", "missing_evidence"]
DEFAULT_CONTEXT_IMAGE_MAX_MESSAGES_PER_SESSION = 24
DEFAULT_CONTEXT_IMAGE_MAX_CHARS_PER_MESSAGE = 1_800


class ContextSegment(ArchiveInsightModel):
    """One bounded section of a compiled context image."""

    segment_id: str
    kind: ContextSegmentKind
    title: str
    markdown: str | None = None
    payload_kind: str | None = None
    object_refs: tuple[ObjectRef, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    assertion_refs: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    token_estimate: int = 0
    lossiness: str | None = None


class ContextOmission(ArchiveInsightModel):
    """A requested context input that was omitted deliberately."""

    ref: str | None = None
    query: str | None = None
    view: str | None = None
    reason: ContextOmissionReason
    detail: str


class ContextSpec(ArchiveInsightModel):
    """Declarative request for a compiled context image.

    The spec is pure input. It does not imply delivery, durable recording, or
    automatic context injection.
    """

    purpose: ContextPurpose = "continue"
    seed_query: str | None = None
    seed_query_limit: int = Field(default=5, ge=1, le=50)
    seed_project_path: str | None = None
    seed_project_repo: str | None = None
    seed_since: str | None = None
    seed_until: str | None = None
    seed_origin: str | None = None
    seed_refs: tuple[str, ...] = ()
    read_views: tuple[str, ...] = ("messages",)
    unit_queries: tuple[str, ...] = ()
    unit_query_limit: int = Field(default=20, ge=1, le=200)
    max_tokens: int | None = Field(default=None, ge=1)
    max_messages_per_session: int | None = Field(default=None, ge=1, le=500)
    max_chars_per_message: int | None = Field(default=None, ge=1, le=20_000)
    include_assertions: bool = True
    include_candidates: bool = False
    redaction_policy: Literal["default", "raw-opt-in"] = "default"

    @model_validator(mode="after")
    def _requires_seed(self) -> ContextSpec:
        if self.seed_query is None and not self.seed_refs and not self.unit_queries and not _has_seed_filters(self):
            raise ValueError("ContextSpec requires seed_query, seed_refs, or unit_queries")
        return self


def _has_seed_filters(spec: ContextSpec) -> bool:
    return any(
        (
            spec.seed_project_path,
            spec.seed_project_repo,
            spec.seed_since,
            spec.seed_until,
            spec.seed_origin,
        )
    )


class ContextImage(ArchiveInsightModel):
    """Compiled, storage-free context payload over archive refs and views."""

    spec: ContextSpec
    selection_strategy: str = "context_spec_v1"
    redaction_policy: str = "default"
    segments: tuple[ContextSegment, ...]
    object_refs: tuple[ObjectRef, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    assertion_refs: tuple[str, ...] = ()
    omitted: tuple[ContextOmission, ...] = ()
    caveats: tuple[str, ...] = ()
    token_estimate: int = 0
    size_estimate: dict[str, int] = Field(default_factory=dict)


class ContextSnapshotRecord(ArchiveInsightModel):
    """Evidence record for context that was actually delivered."""

    snapshot_ref: str
    run_ref: str | None = None
    boundary: str
    inheritance_mode: str = "explicit"
    segment_refs: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)


def compile_messages_context_segment(
    *,
    session_id: str,
    title: str | None,
    messages: Sequence[tuple[str, str]],
    evidence_refs: Sequence[EvidenceRef],
    omitted_before: int = 0,
    omitted_after: int = 0,
    clipped_messages: int = 0,
) -> ContextSegment:
    """Compile a normalized message transcript into a context segment."""

    lines = [f"# Messages: {title or session_id}", ""]
    if omitted_before:
        lines.append(f"... {omitted_before} earlier messages omitted from this window.")
        lines.append("")
    for role, text in messages:
        lines.append(f"{role}: {text}")
        lines.append("")
    if omitted_after:
        lines.append(f"... {omitted_after} later messages omitted from this window.")
        lines.append("")
    markdown = "\n".join(lines).rstrip() + "\n"
    caveats: list[str] = []
    if omitted_before or omitted_after:
        caveats.append(f"message window omitted {omitted_before} earlier and {omitted_after} later messages")
    if clipped_messages:
        caveats.append(f"{clipped_messages} messages were clipped by character budget")
    return ContextSegment(
        segment_id=f"read-view:{session_id}:messages",
        kind="read_view",
        title="Messages",
        markdown=markdown,
        payload_kind="messages",
        object_refs=(ObjectRef(kind="session", object_id=session_id),),
        evidence_refs=tuple(evidence_refs),
        caveats=tuple(caveats),
        token_estimate=_estimate_tokens(markdown),
        lossiness="bounded_message_window" if caveats else "normalized_message_text",
    )


def compile_query_unit_context_segment(envelope: object) -> ContextSegment:
    """Compile a terminal query-unit envelope into a context segment."""

    payload = cast(dict[str, object], envelope.model_dump(mode="json", exclude_none=True))  # type: ignore[attr-defined]
    unit = str(payload.get("unit") or "unit")
    query = str(payload.get("query") or "")
    items = cast(list[dict[str, object]], payload.get("items") or [])
    object_refs, evidence_refs = _query_unit_refs(items)
    title = f"Query: {unit}"
    lines = [f"# {title}", "", f"- expression: `{query}`", f"- rows: {payload.get('total', len(items))}", ""]
    for index, item in enumerate(items[:20], start=1):
        lines.append(f"{index}. {_query_unit_item_summary(item)}")
    if len(items) > 20:
        lines.append(f"... {len(items) - 20} more rows omitted from this segment.")
    markdown = "\n".join(lines).rstrip() + "\n"
    return ContextSegment(
        segment_id=f"query-unit:{hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]}",
        kind="query_unit",
        title=title,
        markdown=markdown,
        payload_kind=f"query-unit:{unit}",
        object_refs=object_refs,
        evidence_refs=evidence_refs,
        token_estimate=_estimate_tokens(markdown),
        lossiness="bounded_query_unit_rows",
    )


def compile_temporal_context_segment(
    *,
    session_id: str,
    window: TemporalEvidenceWindow,
) -> ContextSegment:
    """Compile a temporal evidence window into a context segment."""

    lines = [
        "# Temporal Evidence",
        "",
        f"- Session: `{session_id}`",
        f"- Events: {window.event_count}",
        f"- Families: {', '.join(f'{key}={value}' for key, value in window.family_counts.items()) or 'none'}",
        f"- Kinds: {', '.join(f'{key}={value}' for key, value in window.kind_counts.items()) or 'none'}",
    ]
    if window.caveats:
        lines.append(f"- Caveats: {', '.join(window.caveats)}")
    lines.extend(["", "## Events"])
    if window.events:
        for event in window.events[:50]:
            label = event.label.replace("\n", " ").strip()
            lines.append(f"- {event.occurred_at.isoformat()} [{event.family}/{event.kind}] {label}")
        if len(window.events) > 50:
            lines.append(f"- ... {len(window.events) - 50} more events omitted from this segment.")
    else:
        lines.append("- none")
    markdown = "\n".join(lines).rstrip() + "\n"
    evidence_refs: list[EvidenceRef] = [EvidenceRef(session_id=session_id)]
    for event in window.events:
        for ref_text in event.evidence_refs:
            with suppress(ValueError):
                evidence_refs.append(EvidenceRef.parse(ref_text))
    caveats = list(window.caveats)
    if len(window.events) > 50:
        caveats.append("temporal_events_omitted_after_50")
    return ContextSegment(
        segment_id=f"read-view:{session_id}:temporal",
        kind="read_view",
        title="Temporal Evidence",
        markdown=markdown,
        payload_kind="temporal",
        object_refs=(ObjectRef(kind="session", object_id=session_id),),
        evidence_refs=tuple(dict.fromkeys(evidence_refs)),
        caveats=tuple(dict.fromkeys(caveats)),
        token_estimate=_estimate_tokens(markdown),
        lossiness="bounded_temporal_events",
    )


def compile_chronicle_context_segment(
    *,
    session_id: str,
    payload: ChronicleProjectionPayload,
) -> ContextSegment:
    """Compile a bounded chronicle projection into a context segment."""

    markdown = render_chronicle_markdown(payload)
    evidence_refs: list[EvidenceRef] = [EvidenceRef(session_id=session_id)]
    for session in payload.sessions:
        evidence_refs.append(EvidenceRef(session_id=session.session_id))
        for message in (*session.first_messages, *session.last_messages):
            evidence_refs.append(EvidenceRef(session_id=session.session_id, message_id=message.message_id))
    caveats = list(payload.caveats)
    for session in payload.sessions:
        caveats.extend(session.caveats)
    return ContextSegment(
        segment_id=f"read-view:{session_id}:chronicle",
        kind="read_view",
        title="Session Chronicle",
        markdown=markdown,
        payload_kind="chronicle",
        object_refs=(ObjectRef(kind="session", object_id=session_id),),
        evidence_refs=tuple(dict.fromkeys(evidence_refs)),
        caveats=tuple(dict.fromkeys(caveats)),
        token_estimate=_estimate_tokens(markdown),
        lossiness="bounded_first_last_projection",
    )


def compile_assertion_context_segment(
    *,
    assertion_id: str,
    kind: object,
    body_text: str | None,
    target_ref: str,
    evidence_ref_texts: Sequence[str] = (),
) -> ContextSegment:
    """Compile one injectable assertion claim into a context segment."""

    kind_text = str(getattr(kind, "value", kind))
    text = body_text or "(empty assertion)"
    markdown = f"# Assertion: {kind_text}\n\n- target: `{target_ref}`\n- {text}\n"
    object_refs = [ObjectRef(kind="assertion", object_id=assertion_id)]
    with suppress(ValueError):
        object_refs.append(ObjectRef.parse(target_ref))
    evidence_refs: list[EvidenceRef] = []
    for ref_text in evidence_ref_texts:
        try:
            evidence_refs.append(EvidenceRef.parse(ref_text))
        except ValueError:
            continue
    return ContextSegment(
        segment_id=f"assertion:{assertion_id}",
        kind="assertion",
        title=f"Assertion: {kind_text}",
        markdown=markdown,
        payload_kind="assertion",
        object_refs=tuple(dict.fromkeys(object_refs)),
        evidence_refs=tuple(dict.fromkeys(evidence_refs)),
        assertion_refs=(f"assertion:{assertion_id}",),
        token_estimate=_estimate_tokens(markdown),
        lossiness="assertion_claim_body",
    )


def context_snapshot_record_from_image(
    image: ContextImage,
    *,
    boundary: str,
    run_ref: str | None = None,
    inheritance_mode: str = "explicit",
) -> ContextSnapshotRecord:
    """Build a storage-free evidence record for delivered context.

    Compilation remains pure. Callers use this helper only at a delivery
    boundary, then persist or emit the returned record through the surface that
    actually performed the handoff.
    """
    if not boundary.strip():
        raise ValueError("ContextSnapshotRecord requires a delivery boundary")
    segment_refs = tuple(segment.segment_id for segment in image.segments)
    metadata: dict[str, str] = {
        "purpose": _metadata_value_to_text(image.spec.purpose),
        "read_views": _metadata_value_to_text(image.spec.read_views),
        "unit_queries": _metadata_value_to_text(image.spec.unit_queries),
        "unit_query_limit": _metadata_value_to_text(image.spec.unit_query_limit),
        "max_tokens": _metadata_value_to_text(image.spec.max_tokens),
        "max_messages_per_session": _metadata_value_to_text(image.spec.max_messages_per_session),
        "max_chars_per_message": _metadata_value_to_text(image.spec.max_chars_per_message),
        "token_estimate": _metadata_value_to_text(image.token_estimate),
        "include_assertions": _metadata_value_to_text(image.spec.include_assertions),
        "include_candidates": _metadata_value_to_text(image.spec.include_candidates),
        "redaction_policy": _metadata_value_to_text(image.spec.redaction_policy),
        "context_redaction_policy": _metadata_value_to_text(image.redaction_policy),
        "selection_strategy": _metadata_value_to_text(image.selection_strategy),
        "size_estimate": _metadata_value_to_text(image.size_estimate),
        "omitted_count": _metadata_value_to_text(len(image.omitted)),
        "assertion_refs": _metadata_value_to_text(image.assertion_refs),
        "caveats": _metadata_value_to_text(image.caveats),
    }
    fingerprint_payload = {
        "boundary": boundary,
        "run_ref": run_ref,
        "inheritance_mode": inheritance_mode,
        "segment_refs": segment_refs,
        "evidence_refs": tuple(ref.format() for ref in image.evidence_refs),
        "metadata": metadata,
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return ContextSnapshotRecord(
        snapshot_ref=f"context-snapshot:{fingerprint}",
        run_ref=run_ref,
        boundary=boundary,
        inheritance_mode=inheritance_mode,
        segment_refs=segment_refs,
        evidence_refs=image.evidence_refs,
        metadata=metadata,
    )


def _metadata_value_to_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def _query_unit_item_summary(item: dict[str, object]) -> str:
    for key in ("summary", "text", "title", "name", "command", "path", "target_ref", "run_ref", "message_id"):
        value = item.get(key)
        if value not in (None, ""):
            text = str(value).replace("\n", " ").strip()
            return text[:240] + ("..." if len(text) > 240 else "")
    return json.dumps(item, sort_keys=True, separators=(",", ":"))[:240]


def _query_unit_refs(items: Sequence[dict[str, object]]) -> tuple[tuple[ObjectRef, ...], tuple[EvidenceRef, ...]]:
    object_refs: list[ObjectRef] = []
    evidence_refs: list[EvidenceRef] = []
    seen_objects: set[str] = set()
    seen_evidence: set[str] = set()
    for item in items:
        for object_ref in _object_refs_from_query_unit_item(item):
            key = object_ref.format()
            if key not in seen_objects:
                seen_objects.add(key)
                object_refs.append(object_ref)
        for evidence_ref in _evidence_refs_from_query_unit_item(item):
            key = evidence_ref.format()
            if key not in seen_evidence:
                seen_evidence.add(key)
                evidence_refs.append(evidence_ref)
    return tuple(object_refs), tuple(evidence_refs)


def _object_refs_from_query_unit_item(item: dict[str, object]) -> tuple[ObjectRef, ...]:
    refs: list[ObjectRef] = []
    for key in (
        "run_ref",
        "parent_run_ref",
        "agent_ref",
        "context_snapshot_ref",
        "snapshot_ref",
        "event_ref",
        "subject_ref",
        "transcript_ref",
        "target_ref",
    ):
        refs.extend(_parse_public_object_refs(item.get(key)))
    for key in ("object_refs", "lineage_refs", "segment_refs"):
        refs.extend(_parse_public_object_refs(item.get(key)))
    session_id = item.get("session_id")
    if isinstance(session_id, str) and session_id:
        refs.append(ObjectRef(kind="session", object_id=session_id))
    message_id = item.get("message_id")
    if isinstance(message_id, str) and message_id:
        refs.append(ObjectRef(kind="message", object_id=message_id))
    block_index = item.get("block_index")
    if isinstance(message_id, str) and message_id and isinstance(block_index, int):
        refs.append(ObjectRef(kind="block", object_id=message_id, qualifiers=(str(block_index),)))
    assertion_id = item.get("assertion_id")
    if isinstance(assertion_id, str) and assertion_id:
        refs.append(ObjectRef(kind="assertion", object_id=assertion_id))
    path = item.get("path")
    if isinstance(path, str) and path:
        refs.append(ObjectRef(kind="file", object_id=path))
    return tuple(refs)


def _evidence_refs_from_query_unit_item(item: dict[str, object]) -> tuple[EvidenceRef, ...]:
    refs = list(_parse_evidence_refs(item.get("evidence_refs")))
    transcript_ref = item.get("transcript_ref")
    refs.extend(_parse_evidence_refs(transcript_ref))
    session_id = item.get("session_id")
    message_id = item.get("message_id")
    block_index = item.get("block_index")
    if isinstance(session_id, str) and session_id:
        if isinstance(message_id, str) and message_id:
            refs.append(EvidenceRef(session_id=session_id, message_id=message_id))
            if isinstance(block_index, int):
                refs.append(EvidenceRef(session_id=session_id, message_id=message_id, block_index=block_index))
        else:
            refs.append(EvidenceRef(session_id=session_id))
    return tuple(refs)


def _parse_public_object_refs(value: object) -> tuple[ObjectRef, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Sequence[object] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        return ()
    refs: list[ObjectRef] = []
    for raw in values:
        if not isinstance(raw, str) or not raw:
            continue
        try:
            refs.append(ObjectRef.parse(raw))
        except ValueError:
            continue
    return tuple(refs)


def _parse_evidence_refs(value: object) -> tuple[EvidenceRef, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Sequence[object] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        return ()
    refs: list[EvidenceRef] = []
    for raw in values:
        if not isinstance(raw, str) or not raw:
            continue
        try:
            refs.append(EvidenceRef.parse(raw))
        except ValueError:
            continue
    return tuple(refs)


__all__ = [
    "ContextImage",
    "ContextOmission",
    "ContextOmissionReason",
    "ContextPurpose",
    "ContextSegment",
    "ContextSegmentKind",
    "ContextSnapshotRecord",
    "ContextSpec",
    "DEFAULT_CONTEXT_IMAGE_MAX_CHARS_PER_MESSAGE",
    "DEFAULT_CONTEXT_IMAGE_MAX_MESSAGES_PER_SESSION",
    "compile_messages_context_segment",
    "compile_query_unit_context_segment",
    "context_snapshot_record_from_image",
]
