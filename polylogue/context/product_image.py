"""Context image composition over a caller-supplied archive evidence source.

The source owns the read snapshot. Composition returns the scheduler ledger in
its image; persistence is a separate writer concern.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from polylogue.archive.context_models import ContextImage, ContextSpec
from polylogue.context.scheduler import ContextItem, schedule_context
from polylogue.core.refs import EvidenceRef, ObjectRef

if TYPE_CHECKING:
    from polylogue.archive.message.models import Message
    from polylogue.archive.session.domain_models import Session


_BOUNDED_MESSAGES_FALLBACK_READ_VIEWS = frozenset({"raw", "context", "neighbors", "correlation", "chronicle"})


def _select_context_messages(
    messages: Sequence[Message],
    *,
    anchor_message_id: str | None,
    max_messages: int | None,
    max_chars_per_message: int | None,
) -> tuple[tuple[Message, ...], int, int, int]:
    """Shared identity-preserving message and character window for profiles."""
    rows: list[Message] = []
    clipped_messages = 0
    tool_types = {"tool_use", "tool_result", "function_call", "function_call_output"}
    for message in messages:
        if not message.text and not message.blocks:
            continue
        text = message.text
        prose_chars = sum(
            len(str(block.get("text", block.get("content")) or ""))
            for block in message.blocks
            if block.get("type") not in tool_types
        )
        if max_chars_per_message is not None and max(len(text or ""), prose_chars) > max_chars_per_message:
            clipped_messages += 1
            if text is not None and len(text) > max_chars_per_message:
                omitted_chars = len(text) - max_chars_per_message
                text = (
                    text[:max_chars_per_message].rstrip() + f"\n\n... {omitted_chars} chars omitted from this message."
                )
            remaining = max_chars_per_message
            blocks: list[dict[str, object]] = []
            for block in message.blocks:
                projected = dict(block)
                if block.get("type") not in tool_types:
                    key = "text" if "text" in block else "content"
                    value = block.get(key)
                    if value is not None:
                        prose = str(value)
                        projected[key] = prose[:remaining]
                        remaining = max(0, remaining - len(prose))
                blocks.append(projected)
            if prose_chars > max_chars_per_message:
                blocks.append(
                    {
                        "type": "text",
                        "text": f"\n\n... {prose_chars - max_chars_per_message} chars omitted from this message.",
                    }
                )
            message = message.copy_with_projected_content(text=text, blocks=blocks, attachments=message.attachments)
        rows.append(message)
    if max_messages is None or len(rows) <= max_messages:
        return tuple(rows), 0, 0, clipped_messages
    anchor_index = next((index for index, message in enumerate(rows) if message.id == anchor_message_id), 0)
    start = min(max(0, anchor_index - max_messages // 2), max(0, len(rows) - max_messages))
    end = start + max_messages
    return tuple(rows[start:end]), start, len(rows) - end, clipped_messages


def _archive_context_message_window(
    messages: Sequence[Message],
    *,
    anchor_message_id: str | None,
    max_messages: int | None,
    max_chars_per_message: int | None,
    max_tokens: int | None = None,
) -> tuple[tuple[tuple[str, str], ...], int, int, int]:
    selected, omitted_before, omitted_after, clipped_messages = _select_context_messages(
        messages,
        anchor_message_id=anchor_message_id,
        max_messages=max_messages,
        max_chars_per_message=max_chars_per_message,
    )
    rows = [(message.id, str(getattr(message.role, "value", message.role)), message.text or "") for message in selected]
    if max_tokens is not None:
        rows, budget_omitted, budget_clipped = _budget_context_message_window(rows, max_tokens)
        omitted_before += budget_omitted
        clipped_messages += budget_clipped
    return tuple((role, text) for _id, role, text in rows), omitted_before, omitted_after, clipped_messages


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
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for ref in refs:
        key = (ref.kind, ref.object_id, ref.qualifiers)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return tuple(deduped)


def _dedupe_evidence_refs(refs: Iterable[EvidenceRef]) -> tuple[EvidenceRef, ...]:
    deduped: list[EvidenceRef] = []
    seen: set[tuple[str, str | None, int | None, str | None]] = set()
    for ref in refs:
        key = (ref.session_id, ref.message_id, ref.block_index, ref.block_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return tuple(deduped)


async def compile_context_image(source: Any, spec: ContextSpec) -> ContextImage:
    from polylogue.archive.context_models import ContextImage, ContextOmission, ContextSegment
    from polylogue.context.compiler import (
        compile_assertion_context_segment,
        compile_chronicle_context_segment,
        compile_messages_context_segment,
        compile_prose_with_refs_context_segment,
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

    query_session_ids, query_message_anchors, query_omissions = await source._compile_context_seed_query(spec)
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
        if spec.segment_profile == "prose_with_refs":
            prose_messages, omitted_before, omitted_after, clipped_messages = _select_context_messages(
                tuple(session.messages),
                anchor_message_id=message_anchor_by_session.get(session_id),
                max_messages=spec.max_messages_per_session,
                max_chars_per_message=spec.max_chars_per_message,
            )
            segment, recapped = compile_prose_with_refs_context_segment(
                session_id=session_id,
                title=session.title,
                messages=prose_messages,
                omitted_before=omitted_before,
                omitted_after=omitted_after,
                clipped_messages=clipped_messages,
                max_tokens=remaining_tokens,
                evidence_refs=(EvidenceRef(session_id=session_id),),
            )
            if recapped:
                omitted.append(
                    ContextOmission(
                        ref=f"session:{session_id}",
                        view=view,
                        reason="budget",
                        detail="oldest unprotected prose collapsed to one-line recaps",
                    )
                )
            token_total += segment.token_estimate
            segments.append(segment)
            return True
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
            envelope = await source.query_units(expression, limit=spec.unit_query_limit)
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
        session = await source.get_session(session_id)
        summary = await source.get_session_summary(session_id)
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
                window = source._context_temporal_window(summary)
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
                payload = await source._context_chronicle_payload(summary)
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
            assertion_claims = await source.list_assertion_claim_payloads(
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
                    author_kind=claim.author_kind,
                    author_ref=claim.author_ref,
                    status=claim.status,
                    context_policy=claim.context_policy,
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

    # All compiled material crosses the same admission kernel used by live
    # sources. The compiler remains responsible for producing typed
    # archive segments; the scheduler alone decides what can be admitted
    # and receipts every candidate. Archive-derived segments are quoted
    # evidence by construction, never executable instructions.
    from polylogue.core.refs import ExecutionContextRef

    class _CompiledSegmentsSource:
        name = "archive-context"

        @staticmethod
        def _degrade(item: ContextItem) -> ContextItem:
            # The historical message compiler guarantees at least one
            # visible message plus its framing even at a one-token
            # request. Preserve that established shape while making the
            # scheduler record the bounded degraded admission.
            return replace(item, token_cost=1)

        def candidates(self, *, moment: str, target_session: str | None) -> Sequence[ContextItem]:
            del moment
            return tuple(
                ContextItem(
                    ref=segment.segment_id,
                    content=segment.markdown or "",
                    token_cost=segment.token_estimate,
                    ordinal_score=-index,
                    source=self.name,
                    trust_class="quoted",
                    material_class="evidence",
                    target_session=target_session,
                    degrade=self._degrade,
                )
                for index, segment in enumerate(segments)
            )

    execution_context = ExecutionContextRef.from_observation(
        {"purpose": spec.purpose, "redaction_policy": spec.redaction_policy, "read_views": spec.read_views},
        unknown_fields=("runtime",),
    )
    admission = schedule_context(
        (_CompiledSegmentsSource(),),
        moment=spec.purpose,
        target_session=session_ids[0] if session_ids else None,
        execution_context=execution_context,
        token_budget=spec.max_tokens if spec.max_tokens is not None else max(token_total, 1),
        now_ms=0,
    )
    admitted_ids = {item.ref for item in (*admission.quoted_evidence, *admission.executable_policy)}
    admitted_segments = tuple(segment for segment in segments if segment.segment_id in admitted_ids)
    omitted.extend(
        ContextOmission(
            ref=row.item_ref,
            view="scheduler",
            reason="budget",
            detail="segment rejected by the context scheduler budget",
        )
        for row in admission.ledger
        if row.decision == "dropped" and row.disclosure_verdict == "budget"
    )
    object_refs = _dedupe_object_refs(ref for segment in admitted_segments for ref in segment.object_refs)
    evidence_refs = _dedupe_evidence_refs(ref for segment in admitted_segments for ref in segment.evidence_refs)
    assertion_refs = tuple(dict.fromkeys(ref for segment in admitted_segments for ref in segment.assertion_refs))
    caveats = tuple(dict.fromkeys(caveat for segment in admitted_segments for caveat in segment.caveats))

    return ContextImage(
        spec=spec,
        segments=admitted_segments,
        object_refs=object_refs,
        evidence_refs=evidence_refs,
        assertion_refs=assertion_refs,
        omitted=tuple(omitted),
        caveats=caveats,
        # The estimate a caller budgets against must describe the payload
        # this image actually carries. ``admission.token_cost`` is the
        # scheduler's own charge, and a budget-degraded admission charges
        # the floor it applied rather than the segment it admitted: a
        # ``max_tokens=1`` image reported 1 token while returning 14
        # tokens of markdown (measured on the seeded two-session fixture).
        # Each segment's own ``token_estimate`` is honest, so the image's
        # is their sum. It can exceed ``spec.max_tokens`` -- that is the
        # minimum-viable-segment floor being visible instead of hidden.
        token_estimate=sum(segment.token_estimate for segment in admitted_segments),
        execution_context_ref=execution_context,
        ledger=admission.ledger,
        build_ref=admission.build_ref,
    )
