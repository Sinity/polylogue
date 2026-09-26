"""Pinned session detail and message products for the daemon web reader."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

from polylogue.archive.hydration import (
    archive_attachment_to_domain,
    archive_message_to_domain,
    archive_summary_to_domain,
)
from polylogue.archive.query.transaction import QueryContinuationInvalidError
from polylogue.operations.authority import authority_for_reader
from polylogue.operations.message_locator import locate_message_in_archive, window_offset_for_index
from polylogue.operations.transcript_window import read_transcript_window_sync, window_request
from polylogue.rendering.semantic_card_placement import (
    SemanticCardPlacement,
    semantic_card_placement_for_messages,
)
from polylogue.rendering.semantic_cards import lineage_descriptor_from_archive_envelope
from polylogue.surfaces.authority import serialize_authority
from polylogue.surfaces.outcome import decide_outcome, lineage_page_outcome
from polylogue.surfaces.payloads import (
    TargetRefPayload,
    message_topology_from_domain,
    reader_anchor,
    reader_message_actions,
    reader_session_actions,
)
from polylogue.surfaces.query_rows import session_row

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary, ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveMessageRow, ArchiveSessionEnvelope


@dataclass(frozen=True, slots=True)
class HttpSessionProjectionAdapters:
    """Pure wire helpers supplied by the HTTP adapter without reversing layers."""

    attachment: Callable[..., dict[str, object]]
    paste_spans: Callable[..., list[dict[str, object]]]


def _target(ref: TargetRefPayload) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=True)


def _actions(actions: Mapping[str, Any]) -> dict[str, object]:
    return {name: availability.model_dump(mode="json", exclude_none=True) for name, availability in actions.items()}


def _summary_payload(summary: ArchiveSessionSummary) -> dict[str, object]:
    domain = archive_summary_to_domain(summary)
    session_id = str(domain.id)
    row = session_row(domain)
    return {
        "id": session_id,
        "session_id": session_id,
        "title": domain.display_title,
        "origin": str(domain.origin),
        "target_ref": _target(TargetRefPayload.session(session_id)),
        "anchor": reader_anchor("session", session_id),
        "actions": _actions(reader_session_actions()),
        "date": summary.updated_at or summary.created_at,
        "created_at": summary.created_at,
        "updated_at": summary.updated_at,
        "message_count": domain.message_count,
        "word_count": summary.word_count,
        "terminal_state": row.outcome,
        "total_cost_usd": row.cost_usd,
        "relative_time": row.relative_time,
        "repo": domain.git_repository_url,
        "cwd_display": next(iter(domain.working_directories), None),
        "tags": list(domain.tags),
        "flags": None,
        "summary": None,
    }


def _semantic_placement(envelope: ArchiveSessionEnvelope) -> SemanticCardPlacement:
    return semantic_card_placement_for_messages(
        envelope.messages,
        session_id=envelope.session_id,
        provider_family=envelope.origin,
        lineage=lineage_descriptor_from_archive_envelope(envelope),
    )


def _message_attachments(
    session_id: str, message: ArchiveMessageRow, adapters: HttpSessionProjectionAdapters
) -> list[dict[str, object]]:
    return [
        adapters.attachment(
            archive_attachment_to_domain(attachment),
            session_id=session_id,
            message_id=str(message.message_id),
        )
        for attachment in message.attachments
    ]


def _message_payload(
    session_id: str,
    message: ArchiveMessageRow,
    adapters: HttpSessionProjectionAdapters,
    *,
    semantic_entries: Sequence[object] = (),
    semantic_cards: Sequence[object] = (),
    semantic_card_suppressed: bool = False,
) -> dict[str, object]:
    domain = archive_message_to_domain(message)
    message_id = str(domain.id)
    text = domain.text or ""
    has_paste = bool(domain.has_paste)
    return {
        "id": message_id,
        "identity_source": domain.identity_source,
        "role": str(domain.role),
        "text": text,
        "target_ref": _target(TargetRefPayload.message(session_id=session_id, message_id=message_id)),
        "anchor": reader_anchor("message", message_id),
        "actions": _actions(reader_message_actions()),
        "timestamp": message.occurred_at,
        "message_type": str(domain.message_type),
        "material_origin": str(domain.material_origin),
        "duration_ms": domain.duration_ms,
        **message_topology_from_domain(domain),
        "stop_reason": domain.stop_reason,
        "source_session_id": message.source_session_id,
        "inherited_prefix": message.source_session_id != session_id if message.source_session_id is not None else None,
        "word_count": message.word_count,
        "has_tool_use": bool(domain.has_tool_use),
        "has_thinking": bool(domain.has_thinking),
        "has_paste_evidence": has_paste,
        "paste_spans": adapters.paste_spans(text, has_paste=has_paste),
        "semantic_entries": list(semantic_entries),
        "semantic_cards": list(semantic_cards),
        "semantic_card_suppressed": semantic_card_suppressed,
        "attachments": _message_attachments(session_id, message, adapters),
    }


def execute_http_session_detail(
    payload: Mapping[str, object], *, archive: ArchiveStore, adapters: HttpSessionProjectionAdapters
) -> dict[str, object] | None:
    """Project summary or composed detail from the supplied archive reader."""

    session_ref = payload.get("session_id")
    if not isinstance(session_ref, str) or not session_ref:
        raise ValueError("session detail requires a session reference")
    shape = payload.get("shape", "full")
    if shape not in {"full", "summary"}:
        raise ValueError("session detail shape must be full or summary")
    try:
        session_id = archive.resolve_session_id(session_ref)
        summary = archive.read_summary(session_id)
    except KeyError:
        return None
    if shape == "summary":
        result = _summary_payload(summary)
        result.update(
            {
                "display_title": result["title"],
                "branch_type": summary.branch_type,
                "parent_id": summary.parent_id,
                "session_kind": summary.session_kind,
                "display_name": summary.display_name,
                "title_source": summary.title_source,
                "title_ref": summary.title_ref,
                "model": None,
                "total": result["message_count"],
            }
        )
        return result

    limit = payload.get("limit")
    offset = payload.get("offset", 0)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("session detail limit must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("session detail offset must be a non-negative integer")
    envelope = (
        archive.read_session_page(session_id, limit=limit, offset=offset)
        if limit is not None
        else archive.read_session(session_id)
    )
    placement = _semantic_placement(envelope)
    messages = [
        _message_payload(
            session_id,
            message,
            adapters,
            semantic_entries=placement.entries_for(str(message.message_id)),
            semantic_cards=placement.cards_for(str(message.message_id)),
            semantic_card_suppressed=placement.is_suppressed(str(message.message_id)),
        )
        for message in envelope.messages
    ]
    attachments = [
        attachment for message in messages for attachment in cast(list[dict[str, object]], message["attachments"])
    ]
    attachments.extend(
        adapters.attachment(
            archive_attachment_to_domain(attachment),
            session_id=session_id,
            message_id=attachment.message_id,
        )
        for attachment in envelope.orphan_attachments
    )
    total = envelope.total_message_count if envelope.total_message_count is not None else len(messages)
    return {
        "id": session_id,
        "session_id": session_id,
        "title": envelope.title,
        "display_title": envelope.title or session_id,
        "origin": envelope.origin,
        "target_ref": _target(TargetRefPayload.session(session_id)),
        "anchor": reader_anchor("session", session_id),
        "actions": _actions(reader_session_actions()),
        "created_at": summary.created_at,
        "updated_at": summary.updated_at,
        "message_count": total,
        "word_count": summary.word_count,
        "messages": messages,
        "attachments": attachments,
        "semantic_entries": list(placement.session_entries),
        "tags": list(summary.tags),
        "branch_type": envelope.branch_type,
        "parent_id": envelope.parent_session_id,
        "repo": envelope.git_repository_url,
        "cwd_display": next(iter(envelope.working_directories), None),
        "model": None,
        "flags": None,
        "summary": None,
        "total": total,
    }


def execute_http_session_messages(
    payload: Mapping[str, object],
    *,
    archive: ArchiveStore,
    adapters: HttpSessionProjectionAdapters,
    server_identity: Literal["daemon", "direct"] = "daemon",
) -> dict[str, object]:
    """Project one snapshot-bound composed message window for the web reader."""

    started_at = monotonic()
    session_ref = payload.get("session_id")
    if not isinstance(session_ref, str) or not session_ref:
        raise ValueError("session messages require a session reference")
    limit = payload.get("limit", 50)
    offset = payload.get("offset", 0)
    continuation = payload.get("continuation")
    around = payload.get("around")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("session messages limit must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("session messages offset must be a non-negative integer")
    if around and continuation:
        raise QueryContinuationInvalidError("around and continuation name two different windows")
    authority = serialize_authority(
        authority_for_reader(archive, server_identity=server_identity, started_at=started_at)
    )
    try:
        session_id = archive.resolve_session_id(session_ref)
    except KeyError:
        return {
            "messages": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "next_offset": None,
            "continuation": None,
            "lineage_complete": True,
            "lineage_truncation_reason": None,
            "outcome": decide_outcome(matched=0, error="session_not_found").to_dict(),
            "authority": authority,
        }
    if isinstance(around, str) and around:
        offset = window_offset_for_index(locate_message_in_archive(archive, session_id, around).index, limit)

    latest_envelope: ArchiveSessionEnvelope | None = None

    def read(page_limit: int, page_offset: int) -> tuple[list[object], int, object]:
        nonlocal latest_envelope
        latest_envelope = archive.read_session_page(session_id, limit=page_limit, offset=page_offset)
        rows: list[object] = list(latest_envelope.messages)
        total = latest_envelope.total_message_count if latest_envelope.total_message_count is not None else len(rows)
        return (
            rows,
            total,
            SimpleNamespace(
                complete=latest_envelope.lineage_complete,
                truncation_reason=latest_envelope.lineage_truncation_reason,
            ),
        )

    window = read_transcript_window_sync(
        archive,
        window_request(session_id, limit=limit, offset=offset, continuation=cast(str | None, continuation)),
        read=read,
    )
    assert latest_envelope is not None
    placement = _semantic_placement(latest_envelope)
    return {
        "session_id": latest_envelope.session_id,
        "messages": [
            _message_payload(
                latest_envelope.session_id,
                cast("ArchiveMessageRow", message),
                adapters,
                semantic_entries=placement.entries_for(str(cast("ArchiveMessageRow", message).message_id)),
                semantic_cards=placement.cards_for(str(cast("ArchiveMessageRow", message).message_id)),
                semantic_card_suppressed=placement.is_suppressed(str(cast("ArchiveMessageRow", message).message_id)),
            )
            for message in window.rows
        ],
        "semantic_entries": list(placement.session_entries),
        "total": window.total,
        "limit": window.limit,
        "offset": window.offset,
        "next_offset": window.next_offset,
        "continuation": window.continuation,
        "lineage_complete": latest_envelope.lineage_complete,
        "lineage_truncation_reason": latest_envelope.lineage_truncation_reason,
        "outcome": lineage_page_outcome(
            matched=window.total,
            complete=latest_envelope.lineage_complete,
            truncation_reason=latest_envelope.lineage_truncation_reason,
        ).to_dict(),
        "authority": authority,
    }


__all__ = ["HttpSessionProjectionAdapters", "execute_http_session_detail", "execute_http_session_messages"]
