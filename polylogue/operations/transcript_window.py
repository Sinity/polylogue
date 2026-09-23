"""One execution route for the transcript window (polylogue-ijbwq).

The request is "give me messages ``[offset, offset + limit)`` of this session".
Before this module it had two owners that were not interchangeable
(``docs/design/transcript-window-responsibility.md`` recorded the split): the
``session.read`` declared operation, which alone carried snapshot-bound
continuation with epoch validation, and ``Polylogue.get_messages_paginated``,
which the Python API, MCP and HTTP reached directly and which resumed by
re-asking for an offset — so a write landing between two pages silently
shifted the second one on three of the four public surfaces.

This module owns the parts that must not differ by surface:

* **window arithmetic** — ``[offset, offset + limit)``, ``next_offset``,
  ``complete``;
* **snapshot binding** — the archive epoch is read before *and* after the
  storage read, so a continuation is only minted for a window that was
  composed against one snapshot;
* **epoch validation** — resuming a continuation issued against an older
  snapshot raises ``QueryContinuationStaleError`` rather than paging into
  shifted rows;
* **continuation vocabulary** — one opaque ``QueryContinuation`` token, so a
  token minted on one surface is readable on every other.

What deliberately stays with each surface is the **row projection**: the CLI,
MCP and Python API answer with domain ``Message`` rows, while the HTTP
web-reader answers with composed ``ArchiveMessageRow`` rows that additionally
carry ``source_session_id``/``inherited_prefix`` (lineage-prefix provenance the
domain model does not represent) and the stored per-message ``word_count``.
That is a difference of what is rendered, not of how the window is decided, so
the reader is a declared parameter of this one route rather than a second
route. Collapsing it would drop real fields from the web reader; see the
``Reader`` protocol below.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from polylogue.archive.query.transaction import (
    QueryContinuation,
    QueryContinuationInvalidError,
    QueryTransaction,
    QueryTransactionRequest,
    archive_snapshot_epoch,
    validate_continuation_epoch,
)
from polylogue.operations.session_contracts import SessionRead

RowT = TypeVar("RowT")

#: Projection token stamped into every transcript-window transaction. It is the
#: session-owner projection the ``sessions.read`` operation already minted, so
#: a continuation issued by that operation and one issued by the MCP ``read``
#: tool are the same token, not two dialects.
TRANSCRIPT_WINDOW_PROJECTION = "session-owner-v1"

#: Stable order the window is composed in. Transcript position is the only
#: order the storage layer guarantees for a composed lineage transcript.
TRANSCRIPT_WINDOW_ORDER = "position"


@dataclass(frozen=True, slots=True)
class TranscriptWindow(Generic[RowT]):
    """One delivered transcript window plus the binding that produced it."""

    rows: list[RowT]
    total: int
    limit: int
    offset: int
    next_offset: int | None
    continuation: str | None
    lineage_complete: bool
    lineage_truncation_reason: str | None
    transaction: QueryTransactionRequest

    @property
    def complete(self) -> bool:
        """Whether this window is the last one for the bound snapshot."""

        return self.next_offset is None

    @property
    def gaps(self) -> list[str]:
        """Named coverage gaps, so a truncated lineage cannot read as ``ok``."""

        return [] if self.lineage_complete else [str(self.lineage_truncation_reason)]


#: A storage read for one window: ``(limit, offset)`` in, rows plus the
#: session total plus the lineage-completeness signal out. Two production
#: readers exist and they answer with different row types on purpose (see the
#: module docstring); neither decides the window.
Reader = Callable[[int, int], Awaitable[tuple[list[Any], int, Any]]]
SyncReader = Callable[[int, int], tuple[list[Any], int, Any]]


def _window_arguments(request: SessionRead) -> dict[str, object]:
    """Project the request onto the transaction arguments a resume must match.

    ``limit``/``offset`` are the window *coordinates*, carried by the
    transaction itself; everything else is the request identity, so a
    continuation minted for one filter set cannot resume another.
    """

    return request.model_dump(mode="json", exclude={"continuation", "limit", "offset"})


def frame_request(request: SessionRead) -> tuple[SessionRead, QueryTransactionRequest]:
    """Resolve a window request, honouring a continuation over its coordinates.

    A continuation carries the whole request it was minted from. Supplying a
    conflicting field alongside it is a caller error, not a silently ignored
    argument: the two disagree about which window the caller wants.
    """

    continuation = request.continuation
    if not continuation:
        return request, QueryTransactionRequest(
            operation=request.operation,
            arguments=_window_arguments(request),
            page_size=request.limit,
            offset=request.offset,
            projection=TRANSCRIPT_WINDOW_PROJECTION,
            stable_order=TRANSCRIPT_WINDOW_ORDER,
        )

    decoded = QueryContinuation.decode(continuation)
    transaction = decoded.request
    if (
        transaction.operation != request.operation
        or transaction.projection != TRANSCRIPT_WINDOW_PROJECTION
        or decoded.result_ref != transaction.result_ref
    ):
        raise QueryContinuationInvalidError("continuation belongs to another session operation")
    original = SessionRead.model_validate(
        {**dict(transaction.arguments), "limit": transaction.page_size, "offset": transaction.offset}
    )
    reference = original.model_dump(mode="json")
    supplied = request.model_dump(mode="json", exclude_unset=True, exclude={"continuation", "operation"})
    for name, value in supplied.items():
        if value != reference[name]:
            raise QueryContinuationInvalidError(f"continuation conflicts with {name}")
    return original, transaction


def bind_snapshot(archive: Any, transaction: QueryTransactionRequest) -> QueryTransactionRequest:
    """Stamp (or revalidate) the archive epoch this window is composed against.

    A transaction that already carries an epoch is being resumed, so the epoch
    is *validated* — a write landing since it was issued makes the resume stale
    rather than silently shifting the rows.
    """

    epoch = (
        validate_continuation_epoch(transaction, archive=archive)
        if transaction.archive_epoch
        else archive_snapshot_epoch(archive)
    )
    return transaction.with_archive_epoch(epoch)


def window_result(
    rows: list[RowT],
    total: int,
    transaction: QueryTransactionRequest,
    *,
    lineage_complete: bool = True,
    lineage_truncation_reason: object = None,
) -> TranscriptWindow[RowT]:
    """Decide the window coordinates and mint the continuation, once.

    Every surface's ``next_offset``/``complete``/``continuation`` comes from
    here, so an off-by-one or a missing token is a single defect rather than
    four independent ones.
    """

    next_offset = transaction.offset + len(rows) if transaction.offset + len(rows) < total else None
    continuation = (
        QueryContinuation(transaction.next(offset=next_offset), transaction.result_ref).encode()
        if next_offset is not None
        else None
    )
    return TranscriptWindow(
        rows=rows,
        total=total,
        limit=transaction.page_size,
        offset=transaction.offset,
        next_offset=next_offset,
        continuation=continuation,
        lineage_complete=lineage_complete,
        lineage_truncation_reason=(str(lineage_truncation_reason) if lineage_truncation_reason is not None else None),
        transaction=transaction,
    )


def _completeness(completeness: Any) -> tuple[bool, object]:
    complete = bool(getattr(completeness, "complete", True))
    return complete, getattr(completeness, "truncation_reason", None)


async def read_transcript_window(
    archive_root: Path,
    request: SessionRead,
    *,
    read: Reader,
) -> TranscriptWindow[Any]:
    """Answer one transcript window through the single bound execution route.

    The epoch is bound before the storage read and revalidated after it: a
    continuation is only minted when the window that was actually composed and
    the snapshot it was framed against are the same one. Every public surface
    reaches this function; none of them decides a window itself.
    """

    request, transaction = frame_request(request)

    async def bind(current: QueryTransactionRequest) -> QueryTransactionRequest:
        return await QueryTransaction(archive_root, current).run(lambda archive: bind_snapshot(archive, current))

    framed = await bind(transaction)
    rows, total, completeness = await read(framed.page_size, framed.offset)
    await bind(framed)
    complete, reason = _completeness(completeness)
    return window_result(
        list(rows),
        total,
        framed,
        lineage_complete=complete,
        lineage_truncation_reason=reason,
    )


def read_transcript_window_sync(
    archive: Any,
    request: SessionRead,
    *,
    read: SyncReader,
) -> TranscriptWindow[Any]:
    """Answer one transcript window against an already-pinned archive reader.

    The HTTP web reader and the ``session.read`` declared operation both run
    inside a reader the caller already opened (``archive_read_context`` and the
    operation kernel's pinned snapshot). Opening a second transaction from
    inside one would read a different snapshot than the one their payload is
    composed from, so they bind against the reader they hold — the same
    framing, validation and continuation vocabulary as the async route.
    """

    request, transaction = frame_request(request)
    framed = bind_snapshot(archive, transaction)
    rows, total, completeness = read(framed.page_size, framed.offset)
    bind_snapshot(archive, framed)
    complete, reason = _completeness(completeness)
    return window_result(
        list(rows),
        total,
        framed,
        lineage_complete=complete,
        lineage_truncation_reason=reason,
    )


async def message_transcript_window(api: Any, request: SessionRead) -> TranscriptWindow[Any]:
    """Answer a transcript window as domain ``Message`` rows.

    This is the binding the Python API, the CLI ``read --view messages`` verb,
    the MCP ``read`` tool and the HTTP database branch all reach. The storage
    read is the repository's bounded query. The public facade's
    ``get_messages_paginated`` method is a compatibility projection over this
    route; calling it here would re-enter the route and leave two execution
    owners.
    """

    session_id = request.ref.removeprefix("session:")

    async def read(limit: int, offset: int) -> tuple[list[Any], int, Any]:
        if request.material_origin:
            from polylogue.archive.message.types import MessageType

            session = await api.get_session(session_id)
            if session is None:
                raise ValueError(f"session not found: {session_id}")
            messages = [
                message
                for message in session.messages
                if (not request.message_role or message.role in request.message_role)
                and (
                    request.message_type is None or message.message_type == MessageType.normalize(request.message_type)
                )
                and (not request.material_origin or message.material_origin in request.material_origin)
            ]
            completeness = await api.repository.get_lineage_completeness(session_id)
            return list(messages[offset : offset + limit]), len(messages), completeness

        resolved_session_id = await api.repository.resolve_id(session_id) or session_id
        messages, total, completeness = await api.repository.get_messages_paginated(
            resolved_session_id,
            message_role=tuple(request.message_role),
            message_type=request.message_type,
            limit=limit,
            offset=offset,
        )
        if total == 0 and resolved_session_id == session_id and await api.repository.resolve_id(session_id) is None:
            raise ValueError(f"session not found: {session_id}")
        return list(messages), total, completeness

    return await read_transcript_window(Path(api.archive_root), request, read=read)


def window_request(
    ref: str,
    *,
    limit: int = 50,
    offset: int = 0,
    continuation: str | None = None,
    filters: Mapping[str, object] | None = None,
) -> SessionRead:
    """Build the one request model every surface lowers its window onto."""

    payload: dict[str, object] = {"ref": ref, "limit": limit, "offset": offset, **dict(filters or {})}
    if continuation is not None:
        payload["continuation"] = continuation
        payload.pop("limit", None)
        payload.pop("offset", None)
    return SessionRead.model_validate(payload)


__all__ = [
    "TRANSCRIPT_WINDOW_ORDER",
    "TRANSCRIPT_WINDOW_PROJECTION",
    "Reader",
    "SyncReader",
    "TranscriptWindow",
    "bind_snapshot",
    "message_transcript_window",
    "frame_request",
    "read_transcript_window",
    "read_transcript_window_sync",
    "window_request",
    "window_result",
]
