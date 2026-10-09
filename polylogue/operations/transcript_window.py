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
* **continuation binding** — the same ``QueryContinuation`` encoding and epoch
  checks across routes. Declared projections identify the dialect; replay
  across dialects is refused with a typed error naming both.

What deliberately stays with each surface is the **row projection**: the CLI,
MCP and Python API answer with domain ``Message`` rows, while the HTTP
web-reader answers with composed ``ArchiveMessageRow`` rows that additionally
carry ``source_session_id``/``inherited_prefix`` (lineage-prefix provenance the
domain model does not represent) and the stored per-message ``word_count``.
That is a difference of what is rendered, not of how the window is decided, so
the reader is a declared parameter of this one route rather than a second
route. The legacy ``session.read`` transcript envelope also uses these window
mechanics but retains its ``session-read-v1`` continuation dialect because its
projection arguments differ from the typed ``sessions.read`` owner contract.
The token's decoded projection identifies its dialect. Replaying one dialect
against the other is refused with a typed continuation error naming both
projections. Collapsing the projections would drop real fields from the web
reader; see the ``Reader`` protocol below.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import ValidationError

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
    session_id: str | None = None

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
Reader = Callable[[str, int, int], Awaitable[tuple[list[Any], int, Any]]]
SyncReader = Callable[[int, int], tuple[list[Any], int, Any]]


def _window_arguments(request: SessionRead) -> dict[str, object]:
    """Project the request onto the transaction arguments a resume must match.

    ``limit``/``offset`` are the window *coordinates*, carried by the
    transaction itself; everything else is the request identity, so a
    continuation minted for one filter set cannot resume another.
    """

    return request.model_dump(mode="json", exclude={"continuation", "limit", "offset"})


def frame_request(
    request: SessionRead,
    *,
    transaction_operation: str | None = None,
    projection: str = TRANSCRIPT_WINDOW_PROJECTION,
    extra_arguments: Mapping[str, object] | None = None,
) -> tuple[SessionRead, QueryTransactionRequest]:
    """Resolve a window request, honouring a continuation over its coordinates.

    A continuation carries the whole request it was minted from. Supplying a
    conflicting field alongside it is a caller error, not a silently ignored
    argument: the two disagree about which window the caller wants.
    """

    operation = transaction_operation or request.operation
    arguments = {**_window_arguments(request), **dict(extra_arguments or {})}
    continuation = request.continuation
    if not continuation:
        return request, QueryTransactionRequest(
            operation=operation,
            arguments=arguments,
            page_size=request.limit,
            offset=request.offset,
            projection=projection,
            stable_order=TRANSCRIPT_WINDOW_ORDER,
        )

    decoded = QueryContinuation.decode(continuation)
    transaction = decoded.request
    if transaction.operation != operation or transaction.projection != projection:
        actual = f"{transaction.operation}/{transaction.projection}"
        expected = f"{operation}/{projection}"
        raise QueryContinuationInvalidError(f"continuation dialect mismatch: expected {expected}, got {actual}")
    if decoded.result_ref != transaction.result_ref:
        raise QueryContinuationInvalidError("continuation result identity does not match its bound request")
    original_arguments = {
        key: value for key, value in transaction.arguments.items() if key not in (extra_arguments or {})
    }
    try:
        original = SessionRead.model_validate(
            {**original_arguments, "limit": transaction.page_size, "offset": transaction.offset}
        )
    except ValidationError as exc:
        raise QueryContinuationInvalidError("continuation arguments are not a transcript window request") from exc
    reference = {**_window_arguments(original), **dict(extra_arguments or {})}
    if dict(transaction.arguments) != reference:
        raise QueryContinuationInvalidError("continuation arguments do not match the requested transcript window")
    supplied = request.model_dump(mode="json", exclude_unset=True, exclude={"continuation", "operation", "limit"})
    for name, value in supplied.items():
        if value != original.model_dump(mode="json")[name]:
            raise QueryContinuationInvalidError(f"continuation conflicts with {name}")
    if "limit" not in request.model_fields_set:
        return original, transaction
    if request.limit > transaction.page_size:
        raise QueryContinuationInvalidError("continuation cannot widen its bound window")
    # Only the window narrows; the selection and offset bound into the token
    # stay, and the new request's defaults must not replace them.
    return original.model_copy(update={"limit": request.limit}), replace(transaction, page_size=request.limit)


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
    index_path: Path | None = None,
) -> TranscriptWindow[Any]:
    """Answer one transcript window through the single bound execution route.

    The epoch is bound before the storage read and revalidated after it: a
    continuation is only minted when the window that was actually composed and
    the snapshot it was framed against are the same one. Every public surface
    reaches this function; none of them decides a window itself.
    """

    request, transaction = frame_request(request)

    def initial_frame(archive: Any) -> tuple[QueryTransactionRequest, str]:
        from polylogue.core.errors import SessionNotFoundError

        framed = bind_snapshot(archive, transaction)
        ref = request.ref.removeprefix("session:")
        try:
            session_id = archive.resolve_session_id(ref)
        except KeyError as exc:
            raise SessionNotFoundError(ref) from exc
        return framed, session_id

    framed, session_id = await QueryTransaction(archive_root, transaction).run(initial_frame, index_path=index_path)
    rows, total, completeness = await read(session_id, framed.page_size, framed.offset)
    await QueryTransaction(archive_root, framed).run(
        lambda archive: bind_snapshot(archive, framed), index_path=index_path
    )
    complete, reason = _completeness(completeness)
    return replace(
        window_result(list(rows), total, framed, lineage_complete=complete, lineage_truncation_reason=reason),
        session_id=session_id,
    )


def read_transcript_window_sync(
    archive: Any,
    request: SessionRead,
    *,
    read: SyncReader,
    transaction_operation: str | None = None,
    projection: str = TRANSCRIPT_WINDOW_PROJECTION,
    extra_arguments: Mapping[str, object] | None = None,
) -> TranscriptWindow[Any]:
    """Answer one transcript window against an already-pinned archive reader.

    The HTTP web reader and the legacy ``session.read`` executor run inside a
    reader the caller already opened (``archive_read_context`` and the
    operation kernel's pinned snapshot). Opening a second transaction from
    inside one would read a different snapshot than the one their payload is
    composed from, so they bind against the reader they hold. Callers may
    declare a distinct transaction dialect while sharing these mechanics.
    """

    request, transaction = frame_request(
        request,
        transaction_operation=transaction_operation,
        projection=projection,
        extra_arguments=extra_arguments,
    )
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


async def message_transcript_window(
    api: Any, request: SessionRead, *, content_projection: Any = None, around: str | None = None
) -> TranscriptWindow[Any]:
    """Answer a transcript window as domain ``Message`` rows.

    This is the binding the Python API, the CLI ``read --view messages`` verb,
    the MCP ``read`` tool and the HTTP database branch all reach. The storage
    read is the repository's bounded query. The public facade's
    ``get_messages_paginated`` method is a compatibility projection over this
    route; calling it here would re-enter the route and leave two execution
    owners.
    """

    from polylogue.archive.message.types import MessageType

    # The reader filters by the selection the window is framed with. A resumed
    # request that states only its continuation carries default filters, so
    # reading those would serve unfiltered rows at a filtered token's offset.
    # The repository's explicit index path is authoritative for compatibility
    # facades constructed with a split configured root and active database.
    active_db = Path(api.repository.backend.db_path)
    active_root = active_db.parent
    if active_root.name == ".index-generations":
        active_root = active_root.parent
    elif active_root.parent.name == ".index-generations":
        active_root = active_root.parent.parent
    if around is not None:
        if request.continuation is not None or request.offset:
            raise ValueError("around and an explicit window coordinate name two different windows")
        if request.message_role or request.message_type is not None or request.material_origin or content_projection:
            raise ValueError("around cannot be combined with transcript filters")
        _request, transaction = frame_request(request)

        def anchored_window(archive: Any) -> TranscriptWindow[Any]:
            from types import SimpleNamespace

            from polylogue.archive.hydration import archive_message_to_domain
            from polylogue.core.enums import Origin
            from polylogue.core.errors import SessionNotFoundError
            from polylogue.operations.message_locator import window_offset_around

            ref = request.ref.removeprefix("session:")
            try:
                session_id = archive.resolve_session_id(ref)
            except KeyError as exc:
                raise SessionNotFoundError(ref) from exc
            offset = window_offset_around(archive, session_id, around, request.limit)
            anchored = request.model_copy(update={"offset": offset})

            def read_page(limit: int, offset: int) -> tuple[list[Any], int, Any]:
                envelope = archive.read_session_page(session_id, limit=limit, offset=offset)
                if envelope.total_message_count is None:
                    raise ValueError("bounded transcript page lacks its declared total")
                return (
                    [
                        archive_message_to_domain(row, origin=Origin(envelope.origin), include_null_block_fields=True)
                        for row in envelope.messages
                    ],
                    envelope.total_message_count,
                    SimpleNamespace(
                        complete=envelope.lineage_complete,
                        truncation_reason=envelope.lineage_truncation_reason,
                    ),
                )

            return replace(read_transcript_window_sync(archive, anchored, read=read_page), session_id=session_id)

        return await QueryTransaction(active_root, transaction).run(anchored_window, index_path=active_db)

    submitted = request
    request, _transaction = frame_request(submitted)

    async def storage_page(resolved_session_id: str, limit: int, offset: int) -> tuple[list[Any], int, Any]:
        # The one bounded storage read of the transcript window; every branch
        # below pages through it.
        messages, total, completeness = await api.repository.get_messages_paginated(
            resolved_session_id,
            message_role=tuple(request.message_role),
            message_type=request.message_type,
            limit=limit,
            offset=offset,
        )
        return list(messages), total, completeness

    async def read(resolved_session_id: str, limit: int, offset: int) -> tuple[list[Any], int, Any]:
        projecting = content_projection is not None and content_projection.filters_content()
        if projecting or request.material_origin:
            # Filters the SQL page cannot apply (a content projection, a
            # material-origin filter) stream raw pages through the filter so a
            # bounded window never hydrates the whole transcript; only the
            # requested window and one raw page are held at a time.
            from polylogue.archive.semantic.content_projection import ContentProjectionStream

            stream = ContentProjectionStream(content_projection) if projecting else None
            window: list[Any] = []
            total = 0
            raw_offset = 0
            page_size = 500
            completeness = None
            while True:
                raw, raw_total, completeness = await storage_page(resolved_session_id, page_size, raw_offset)
                for message in stream.project_page(raw) if stream is not None else raw:
                    if request.material_origin and message.material_origin not in request.material_origin:
                        continue
                    if request.message_role and message.role not in request.message_role:
                        continue
                    if request.message_type is not None and message.message_type != MessageType.normalize(
                        request.message_type
                    ):
                        continue
                    if offset <= total < offset + limit:
                        window.append(message)
                    total += 1
                raw_offset += len(raw)
                if not raw or raw_offset >= raw_total:
                    break
            return window, total, completeness
        messages, total, completeness = await storage_page(resolved_session_id, limit, offset)
        return messages, total, completeness

    return await read_transcript_window(active_root, submitted, read=read, index_path=active_db)


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
