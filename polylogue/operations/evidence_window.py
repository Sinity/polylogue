"""The windowed-evidence continuation family (polylogue-r3cuz).

``operations/transcript_window.py`` owns the **message** window vocabulary:
one projection token (``session-owner-v1``) shared by every surface that pages
a transcript, so a token minted by the Python API resumes on the CLI.  That
sharing is the point of the family, and it is also why a *different* kind of
row must not mint into it: a token that resumes "artifacts 50..100" while
carrying the message family's projection is readable by a reader that will
compose messages with it.

So this module declares a second family -- per relation, not per surface.  A
family names three things and nothing else:

* the ``session.read`` kind it pages,
* its own projection token, which is what makes a foreign continuation
  refusable **by name** instead of silently resumable, and
* the stable order the relation is windowed in, because a continuation is only
  meaningful against an order the reader guarantees.

What it deliberately does *not* own is the window arithmetic or the snapshot
binding.  ``next_offset``/``complete``/token minting and the
before-and-after epoch check come from ``transcript_window``'s
``window_result``/``bind_snapshot``, which take a transaction and know nothing
about messages: an off-by-one or a missing epoch revalidation stays a single
defect across both families rather than two independent ones.  The separation
that matters is the *vocabulary*, and :func:`frame_evidence_window` is where
it is enforced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from polylogue.archive.query.transaction import (
    QueryContinuation,
    QueryContinuationInvalidError,
    QueryTransactionRequest,
)
from polylogue.operations.transcript_window import bind_snapshot, window_result

#: One storage read for one evidence page: ``(limit, offset)`` in, the page's
#: rows plus the **relation's own** total out.  A reader that returned the
#: windowed count as the total would make every clipped page look whole, which
#: is exactly what the contract refuses -- so the total is the reader's
#: responsibility and ``EvidenceWindowBody`` checks the arithmetic.
EvidenceReader = Callable[[int, int], tuple[list[dict[str, object]], int]]


@dataclass(frozen=True, slots=True)
class EvidenceWindowFamily:
    """One per-session relation that is paged rather than answered whole."""

    #: The ``session.read`` kind this family serves.
    kind: str
    #: The transaction projection stamped into every token this family mints.
    #: Distinct per relation, so a page of one relation can never resume the
    #: other -- nor the message window.
    projection: str
    #: The order the reader guarantees, recorded in the token.
    stable_order: str


#: ``Session.session_events`` -- provider evidence riding the session timeline.
#: Ordered by the substrate's own ``position``, which is the order the
#: repository hydrates a session's events in.
SESSION_EVENTS_WINDOW = EvidenceWindowFamily(
    kind="events",
    projection="session-events-v1",
    stable_order="position",
)

#: Source-tier acquisition rows for one session's ``raw_id``.  Ordered exactly
#: as ``ArchiveStore.raw_artifacts_for_session`` windows them.
RAW_ARTIFACTS_WINDOW = EvidenceWindowFamily(
    kind="raw",
    projection="session-artifacts-v1",
    stable_order="acquired_at_ms desc,raw_id",
)

EVIDENCE_WINDOW_FAMILIES: dict[str, EvidenceWindowFamily] = {
    family.kind: family for family in (SESSION_EVENTS_WINDOW, RAW_ARTIFACTS_WINDOW)
}


def _window_arguments(family: EvidenceWindowFamily, ref: str) -> dict[str, object]:
    """The request identity a resume must match, minus the coordinates.

    ``limit``/``offset`` live on the transaction itself, so they are not part
    of the identity; the reference and the relation are, which is what keeps a
    token minted for one session's events from resuming another session's.
    """

    return {"ref": ref, "kind": family.kind}


def frame_evidence_window(
    family: EvidenceWindowFamily,
    *,
    ref: str,
    limit: int,
    offset: int,
    continuation: str | None,
) -> QueryTransactionRequest:
    """Resolve one evidence page request, honouring a continuation over coordinates.

    A continuation from another family is refused **by name**.  That refusal is
    the reason this module exists: the message window and each evidence
    relation all page ``session.read``, so an operation-name check alone would
    accept a transcript token here and compose artifacts against a window the
    caller minted for messages.
    """

    if continuation is None:
        return QueryTransactionRequest(
            operation="session.read",
            arguments=_window_arguments(family, ref),
            page_size=limit,
            offset=offset,
            projection=family.projection,
            stable_order=family.stable_order,
        )

    decoded = QueryContinuation.decode(continuation)
    transaction = decoded.request
    if transaction.operation != "session.read" or decoded.result_ref != transaction.result_ref:
        raise QueryContinuationInvalidError("continuation belongs to another operation")
    if transaction.projection != family.projection:
        raise QueryContinuationInvalidError(
            f"continuation belongs to the {transaction.projection!r} read family, "
            f"not the {family.projection!r} window this request asks for"
        )
    if dict(transaction.arguments) != _window_arguments(family, ref):
        raise QueryContinuationInvalidError(f"continuation belongs to another {family.kind} read")
    return transaction


def read_evidence_window(
    archive: Any,
    family: EvidenceWindowFamily,
    *,
    ref: str,
    limit: int,
    offset: int,
    continuation: str | None,
    read: EvidenceReader,
) -> Mapping[str, object]:
    """Answer one evidence page against an already-pinned archive reader.

    The returned mapping is exactly the declared ``EvidenceWindowBody``: the
    caller validates it into the contract rather than assembling a second
    shape of its own, so the reported bound and the rows are decided in one
    place.

    Like the message route, the epoch is bound before the storage read and
    revalidated after it, so a continuation is only minted for a page that was
    composed against one snapshot.
    """

    transaction = frame_evidence_window(
        family,
        ref=ref,
        limit=limit,
        offset=offset,
        continuation=continuation,
    )
    framed = bind_snapshot(archive, transaction)
    rows, total = read(framed.page_size, framed.offset)
    bind_snapshot(archive, framed)
    # ``window_result`` is the shared arithmetic, not the message vocabulary:
    # it decides ``next_offset``/``complete`` and mints the token from the
    # transaction it is handed -- which carries *this* family's projection.
    window = window_result(list(rows), total, framed)
    return {
        "relation": family.kind,
        "rows": list(window.rows),
        "total": window.total,
        "returned": len(window.rows),
        "limit": window.limit,
        "offset": window.offset,
        "next_offset": window.next_offset,
        "continuation": window.continuation,
        "complete": window.complete,
    }


__all__ = [
    "EVIDENCE_WINDOW_FAMILIES",
    "RAW_ARTIFACTS_WINDOW",
    "SESSION_EVENTS_WINDOW",
    "EvidenceReader",
    "EvidenceWindowFamily",
    "frame_evidence_window",
    "read_evidence_window",
]
