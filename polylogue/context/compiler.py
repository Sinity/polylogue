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
from typing import cast

from polylogue.archive.context_models import (
    ContextImage,
    ContextSegment,
    ContextSnapshotRecord,
    context_image_sha256,
)
from polylogue.core.assertions import AssertionContextTrustClass, derive_assertion_context_trust
from polylogue.core.evidence_integrity import EvidenceIntegrityVerdict
from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.surfaces.chronicle import ChronicleProjectionPayload, render_chronicle_markdown
from polylogue.surfaces.compaction import estimate_tokens
from polylogue.surfaces.temporal_evidence import TemporalEvidenceWindow


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


def compile_prose_with_refs_context_segment(
    *,
    session_id: str,
    title: str | None,
    messages: Sequence[object],
    max_tokens: int | None = None,
    keep_last_messages: int = 4,
    evidence_refs: Sequence[EvidenceRef] = (),
) -> tuple[ContextSegment, bool]:
    """Render authored prose verbatim and tools as independently resolvable refs.

    The boolean reports whether prose was recapped to satisfy the budget. Tool
    markers are never expanded into command arguments or result bodies.
    """
    rows: list[tuple[str, str, bool, str | None]] = []
    for message in messages:
        role = str(getattr(getattr(message, "role", None), "value", getattr(message, "role", "unknown")))
        blocks = list(getattr(message, "blocks", ()) or ())
        if not blocks:
            text = getattr(message, "text", None)
            if text:
                rows.append((role, str(text), True, None))
            continue
        for index, raw_block in enumerate(blocks):
            block = raw_block if isinstance(raw_block, dict) else {}
            block_type = str(block.get("type") or "text")
            if block_type in {"tool_use", "tool_result", "function_call", "function_call_output"}:
                message_id = str(getattr(message, "id", ""))
                tool_name = str(block.get("name") or block.get("tool_name") or block_type)
                action_ref = f"action:{message_id}:{index}"
                rows.append((role, f"<ref:{action_ref}> {tool_name}", False, action_ref))
            else:
                text = block.get("text", block.get("content"))
                if text is not None:
                    rows.append((role, str(text), True, None))

    first_user = next((index for index, row in enumerate(rows) if row[0] == "user"), 0)
    protected = {first_user}
    protected.update(range(max(0, len(rows) - keep_last_messages), len(rows)))
    recapped = False

    def render(current: Sequence[tuple[str, str, bool, str | None]]) -> str:
        lines = [
            f"# Messages: {title or session_id}",
            "",
            "Expand action markers with resolve_ref before relying on them.",
            "",
        ]
        lines.extend(f"{role}: {text}" for role, text, _prose, _ref in current)
        return "\n".join(lines).rstrip() + "\n"

    rendered = render(rows)
    prose_budget = None if max_tokens is None else max(1, int(max_tokens * 0.6))
    if prose_budget is not None and _estimate_tokens(rendered) > prose_budget:
        mutable = list(rows)
        for index, (role, text, is_prose, ref) in enumerate(rows):
            if index in protected or not is_prose:
                continue
            words = text.split()
            mutable[index] = (role, "[recap] " + " ".join(words[: min(12, len(words))]), is_prose, ref)
            recapped = True
            rendered = render(mutable)
            if _estimate_tokens(rendered) <= prose_budget:
                break
        # Protected rows are preferences; the declared budget is a hard cap.
        # Recap/truncate the remaining rows, then omit rows if even markers do
        # not fit (the fixed header itself is bounded).
        for index in range(len(mutable)):
            if _estimate_tokens(render(mutable)) <= prose_budget:
                break
            role, text, is_prose, ref = mutable[index]
            if not text:
                continue
            mutable[index] = (role, "[omitted]" if ref is None else f"<ref:{ref}>", is_prose, ref)
            recapped = True
        while mutable and _estimate_tokens(render(mutable)) > prose_budget:
            mutable.pop(0)
            recapped = True
        rendered = render(mutable)
        if _estimate_tokens(rendered) > prose_budget:
            # At tiny budgets even the heading/instructions cost more than
            # the allowance. Keep a prefix whose estimate is within budget.
            rendered = ""
            recapped = True

    segment = ContextSegment(
        segment_id=f"read-view:{session_id}:prose-with-refs",
        kind="read_view",
        title="Messages (prose with refs)",
        markdown=rendered,
        payload_kind="prose_with_refs",
        object_refs=(
            ObjectRef(kind="session", object_id=session_id),
            *tuple(ObjectRef.parse(ref) for _role, _text, _prose, ref in rows if ref is not None),
        ),
        evidence_refs=tuple(evidence_refs) or (EvidenceRef(session_id=session_id),),
        token_estimate=_estimate_tokens(rendered),
        lossiness="budget_recapped_prose" if recapped else "tool_content_as_refs",
        caveats=("oldest unprotected prose collapsed to one-line recaps",) if recapped else (),
    )
    return segment, recapped


def compile_query_unit_context_segment(envelope: object) -> ContextSegment:
    """Compile a terminal query-unit envelope into a context segment."""

    payload = cast(dict[str, object], envelope.model_dump(mode="json", exclude_none=True))  # type: ignore[attr-defined]
    unit = str(payload.get("unit") or "unit")
    query = str(payload.get("query") or "")
    items = cast(list[dict[str, object]], payload.get("projected_items") or payload.get("items") or [])
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


# Assertion rows have no authenticated ContextSource registration yet
# (37t.11), matching the resume preamble's stance
# (``polylogue.context.preamble._ASSERTION_GUIDANCE_SOURCE_AUTHORITY``): their
# prose cannot enter a compiled context image as an operator directive, only
# as explicitly labelled quoted evidence.
_ASSERTION_SEGMENT_SOURCE_AUTHORITY: AssertionContextTrustClass = "quoted"


def compile_assertion_context_segment(
    *,
    assertion_id: str,
    kind: object,
    body_text: str | None,
    target_ref: str,
    author_kind: object = None,
    author_ref: object = None,
    status: object = None,
    context_policy: object = None,
    evidence_ref_texts: Sequence[str] = (),
    integrity_verdict: EvidenceIntegrityVerdict | None = None,
) -> ContextSegment:
    """Compile one injectable assertion claim into a context segment.

    Every assertion segment is provenance-derived and structurally
    quoted (polylogue-x2y9): the resulting ``trust_class`` and the
    fenced ``quoted-assertion-evidence`` block make injected assertion
    text distinguishable from surrounding instruction-grade markdown,
    the same guarantee the SessionStart resume preamble already gives
    (:func:`polylogue.context.preamble._assertion_guidance_from_claim`).
    """

    trust_class = derive_assertion_context_trust(
        author_kind=author_kind,
        author_ref=author_ref,
        status=status,
        context_policy=context_policy,
        source_authority=_ASSERTION_SEGMENT_SOURCE_AUTHORITY,
    )
    # A requested policy is never sufficient to make an assertion injectable.
    # The shared evaluator is the additional gate for callers that have an
    # evidence graph; unsupported verdicts remain explicitly quoted.
    if integrity_verdict is not None and not integrity_verdict.supported:
        trust_class = "quoted"
    kind_text = str(getattr(kind, "value", kind))
    text = body_text or "(empty assertion)"
    markdown = (
        f"# Assertion: {kind_text}\n\n"
        f"- target: `{target_ref}`\n"
        f"- trust: {trust_class} (archive-derived content, not an instruction)\n\n"
        "```quoted-assertion-evidence\n" + text + "\n```\n"
    )
    object_refs = [ObjectRef(kind="assertion", object_id=assertion_id)]
    with suppress(ValueError):
        object_refs.append(ObjectRef.parse(target_ref))
    evidence_refs: list[EvidenceRef] = []
    for ref_text in evidence_ref_texts:
        try:
            evidence_refs.append(EvidenceRef.parse(ref_text))
        except ValueError:
            continue
    caveats = () if integrity_verdict is None else (f"evidence-integrity:{integrity_verdict.status}",)
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
        trust_class=trust_class,
        caveats=caveats,
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
        "context_image_sha256": context_image_sha256(image),
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
    return estimate_tokens(text or "")


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
        block_id = item.get("block_id")
        if not isinstance(block_id, str) or not block_id:
            raise ValueError("query unit block position requires a stable block_id before context delivery")
        refs.append(ObjectRef(kind="block", object_id=block_id))
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
                block_id = item.get("block_id")
                if not isinstance(block_id, str) or not block_id:
                    raise ValueError("query unit block position requires a stable block_id before context delivery")
                refs.append(EvidenceRef(session_id=session_id, message_id=message_id, block_id=block_id))
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
    "compile_messages_context_segment",
    "compile_prose_with_refs_context_segment",
    "compile_query_unit_context_segment",
    "context_snapshot_record_from_image",
]
