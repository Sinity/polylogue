"""Snapshot-bound windows over per-session evidence, shared by every surface.

Each relation has its own continuation projection and stable order. Transcript
window arithmetic remains shared: offsets and totals count completed rows,
not physical fragments. File-edit and web-content rows may span responses;
the token additionally carries their field/byte cursor, without changing the
logical result identity. Source-tier material pages bind their own epoch.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from polylogue.archive.query.transaction import (
    QueryContinuation,
    QueryContinuationInvalidError,
    QueryContinuationStaleError,
    QueryTransactionRequest,
)
from polylogue.operations.evidence_payloads import DEFAULT_EVIDENCE_PAGE_BYTES, EvidencePayloadPage
from polylogue.operations.transcript_window import bind_snapshot, window_result

EvidenceReader = Callable[[int, int], tuple[list[dict[str, object]], int]]
EvidencePayloadReader = Callable[[int, int, Mapping[str, object] | None, int], EvidencePayloadPage]


@dataclass(frozen=True, slots=True)
class EvidenceWindowFamily:
    kind: str
    projection: str
    stable_order: str


SESSION_EVENTS_WINDOW = EvidenceWindowFamily("events", "session-events-v1", "position")
RAW_ARTIFACTS_WINDOW = EvidenceWindowFamily("raw", "session-artifacts-v1", "acquired_at_ms desc,raw_id")
FILE_EDITS_WINDOW = EvidenceWindowFamily("file-edits", "session-file-edits-v2", "message_id,tool_use_block_id")
WEB_CONTENT_WINDOW = EvidenceWindowFamily("web-content", "session-web-content-v2", "message_id,block_id,position")
SESSION_MATERIALS_WINDOW = EvidenceWindowFamily("materials", "session-materials-v1", "created_at_ms,material_id")

EVIDENCE_WINDOW_FAMILIES: dict[str, EvidenceWindowFamily] = {
    family.kind: family
    for family in (
        SESSION_EVENTS_WINDOW,
        RAW_ARTIFACTS_WINDOW,
        FILE_EDITS_WINDOW,
        WEB_CONTENT_WINDOW,
        SESSION_MATERIALS_WINDOW,
    )
}
_SOURCE_EPOCH_ARGUMENT = "source_epoch"


def _window_arguments(family: EvidenceWindowFamily, ref: str) -> dict[str, object]:
    return {"ref": ref, "kind": family.kind}


def frame_evidence_window(
    family: EvidenceWindowFamily,
    *,
    ref: str,
    limit: int,
    offset: int,
    continuation: str | None,
    source_epoch: str | None = None,
) -> QueryTransactionRequest:
    """Resolve a request, rejecting foreign families, selections and snapshots."""
    if continuation is None:
        arguments = _window_arguments(family, ref)
        if source_epoch is not None:
            arguments[_SOURCE_EPOCH_ARGUMENT] = source_epoch
        return QueryTransactionRequest(
            operation="session.read",
            arguments=arguments,
            page_size=limit,
            offset=offset,
            projection=family.projection,
            stable_order=family.stable_order,
        )
    decoded = QueryContinuation.decode(continuation)
    transaction = decoded.request
    if transaction.operation != "session.read" or decoded.result_ref != transaction.result_ref:
        raise QueryContinuationInvalidError("continuation belongs to another operation")
    if transaction.projection != family.projection or transaction.stable_order != family.stable_order:
        raise QueryContinuationInvalidError(
            f"continuation belongs to the {transaction.projection!r} read family/order, "
            f"not the {family.projection!r} window this request asks for"
        )
    arguments = dict(transaction.arguments)
    issued_source_epoch = arguments.pop(_SOURCE_EPOCH_ARGUMENT, None)
    if arguments != _window_arguments(family, ref):
        raise QueryContinuationInvalidError(f"continuation belongs to another {family.kind} read")
    if issued_source_epoch != source_epoch:
        raise QueryContinuationStaleError(issued_epoch=str(issued_source_epoch), current_epoch=str(source_epoch))
    return transaction


def _payload_budget(framed: QueryTransactionRequest, max_bytes: int) -> int:
    """Reserve the actual token/envelope cost before reading any payload bytes."""
    # SQLite counts and field lengths fit signed 64-bit integers. Reserve the
    # longest next coordinates rather than guessing how long this token is.
    largest = 2**63 - 1
    token = QueryContinuation(
        framed.next(offset=largest), framed.result_ref, cursor={"field": largest, "byte": largest}
    ).encode()
    envelope = {
        "relation": framed.arguments["kind"],
        "rows": [],
        "total": largest,
        "returned": largest,
        "limit": framed.page_size,
        "offset": framed.offset,
        "next_offset": largest,
        "continuation": token,
        "complete": False,
        "row_fragment": None,
    }
    budget = max_bytes - len(json.dumps(envelope, ensure_ascii=True).encode("utf-8")) - 256
    if budget < 512:
        raise ValueError("evidence byte budget is too small for its continuation envelope")
    return budget


def read_evidence_window(
    archive: Any,
    family: EvidenceWindowFamily,
    *,
    ref: str,
    limit: int,
    offset: int,
    continuation: str | None,
    read: EvidenceReader,
    source_epoch: Callable[[], str] | None = None,
    read_payload: EvidencePayloadReader | None = None,
    max_bytes: int = DEFAULT_EVIDENCE_PAGE_BYTES,
) -> Mapping[str, object]:
    """Read one advancing page inside the caller's already-pinned snapshot.

    ``returned`` counts completed rows. A partial row is never placed in
    ``rows`` or reported complete: ``row_fragment`` names its byte coverage,
    and its continuation advances within the same row until all fields have
    arrived. Small rows keep their ordinary row projection.
    """
    issued_source_epoch = source_epoch() if source_epoch is not None else None
    transaction = frame_evidence_window(
        family, ref=ref, limit=limit, offset=offset, continuation=continuation, source_epoch=issued_source_epoch
    )
    framed = bind_snapshot(archive, transaction)
    cursor = QueryContinuation.decode(continuation).cursor if continuation is not None else None
    if read_payload is None:
        if cursor is not None:
            raise QueryContinuationInvalidError("this evidence family cannot resume field fragments")
        rows, total = read(framed.page_size, framed.offset)
        page = EvidencePayloadPage(rows=rows, total=total)
    else:
        page = read_payload(framed.page_size, framed.offset, cursor, _payload_budget(framed, max_bytes))
    bind_snapshot(archive, framed)
    if source_epoch is not None and (current := source_epoch()) != issued_source_epoch:
        raise QueryContinuationStaleError(issued_epoch=str(issued_source_epoch), current_epoch=current)
    # The shared arithmetic sees only completed records. The final fragment
    # completes one record; unfinished fragments do not advance the row offset.
    completed = page.rows if page.fragment is None else ([page.fragment] if page.completed_rows else [])
    window = window_result(completed, page.total, framed)
    next_token = window.continuation
    if page.cursor is not None:
        assert window.next_offset is not None
        next_token = QueryContinuation(
            framed.next(offset=window.next_offset), framed.result_ref, cursor=page.cursor
        ).encode()
    return {
        "relation": family.kind,
        "rows": page.rows,
        "total": window.total,
        "returned": page.completed_rows,
        "limit": window.limit,
        "offset": window.offset,
        "next_offset": window.next_offset,
        "continuation": next_token,
        "complete": window.complete,
        "row_fragment": page.fragment,
    }


__all__ = [
    "EVIDENCE_WINDOW_FAMILIES",
    "FILE_EDITS_WINDOW",
    "RAW_ARTIFACTS_WINDOW",
    "SESSION_EVENTS_WINDOW",
    "SESSION_MATERIALS_WINDOW",
    "WEB_CONTENT_WINDOW",
    "EvidenceReader",
    "EvidenceWindowFamily",
    "frame_evidence_window",
    "read_evidence_window",
]
