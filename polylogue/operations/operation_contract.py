"""Shared scheduling primitives for typed Operation surfaces.

This module used to define a generic ``OperationRequest``/``OperationAck``
subclassing framework meant to serve every future mutating operation (ingest,
reindex, rebuild-insights, gc, replay, schema-upgrade). Only ``IMPORT`` ever
landed on it (:mod:`polylogue.operations.import_operations`); the other nine
:class:`polylogue.operations.specs.OperationKind` members have zero production
call sites and nothing dispatches on ``.kind``. Collapsed (polylogue-a7xr.14)
to concrete ``ImportRequest``/``ImportAck`` models in
:mod:`polylogue.operations.import_operations` — reintroduce a shared base
here if and when a second operation actually lands (check first: it may
already be imminent).

What remains here is genuinely reusable, not speculative:

* :class:`OperationStatus` — the scheduling status enum, now defined in
  :mod:`polylogue.operations.operation_status` (polylogue-8s70) and re-exported
  here for backward compatibility. Every member has a real consumer
  (:mod:`polylogue.readiness.capability` maps ``RUNNING``/``COMPLETED``/
  ``FAILED`` to capability readiness states) -- that consumer is on the hot
  path of every ``polylogue status`` invocation and imports the enum from its
  new zero-dependency home directly, to avoid paying for
  :class:`OperationFollowUp`'s pydantic base.
* :class:`OperationFollowUp` — the polling-hint shape returned by every
  accepted/pending operation. A plain data shape, not a polymorphic base —
  keeping it costs nothing and any second operation would want the same
  fields.

The wire envelope (``ImportAck``'s JSON shape on the daemon HTTP
``POST /api/ingest`` response) is unchanged by this collapse — only the
abstract layer above it was removed.
"""

from __future__ import annotations

from polylogue.core.json import JSONDocument, json_document
from polylogue.operations.operation_status import OperationStatus
from polylogue.surfaces.payloads import SurfacePayloadModel


class OperationFollowUp(SurfacePayloadModel):
    """How a client should look up the operation after the ack.

    Every accepted operation populates this so clients (CLI watchers, MCP
    tool callers, browser dashboards) do not have to know per-operation
    polling conventions.

    Fields
    ------
    status_endpoint:
        Surface-relative identifier for the status surface — e.g.
        ``"daemon.import_status"`` or the daemon HTTP path
        ``"/api/operations/{id}"``. Surfaces are free to choose their
        identifier convention; the contract is that the same string is
        accepted by their lookup tool.
    status_token:
        Optional opaque token the surface may require alongside the
        operation id (e.g. a cursor, a request-scoped auth handle).
    poll_after_ms:
        Suggested delay before the first poll. Surfaces use this to throttle
        clients that would otherwise busy-poll.
    """

    status_endpoint: str
    status_token: str | None = None
    poll_after_ms: int = 0

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json"))


__all__ = [
    "OperationFollowUp",
    "OperationStatus",
]
