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

* :class:`OperationStatus` — the scheduling status enum. Every member has a
  real consumer (:mod:`polylogue.readiness.capability` maps
  ``RUNNING``/``COMPLETED``/``FAILED`` to capability readiness states).
* :class:`OperationFollowUp` — the polling-hint shape returned by every
  accepted/pending operation. A plain data shape, not a polymorphic base —
  keeping it costs nothing and any second operation would want the same
  fields.

The wire envelope (``ImportAck``'s JSON shape on the daemon HTTP
``POST /api/ingest`` response) is unchanged by this collapse — only the
abstract layer above it was removed.
"""

from __future__ import annotations

from enum import Enum

from polylogue.core.json import JSONDocument, json_document
from polylogue.surfaces.payloads import SurfacePayloadModel


class OperationStatus(str, Enum):
    """Scheduling status carried by every Operation ack.

    Terminal vs in-flight semantics are intentionally simple:

    * ``ACCEPTED`` — the operation passed validation and was admitted into
      the registry. A follow-up hint is populated so clients can poll for
      completion.
    * ``REJECTED`` — the request failed validation, lacked permissions,
      collided with an existing idempotency key, or otherwise could not be
      admitted. ``error`` is populated; ``follow_up`` is omitted.
    * ``PENDING`` — admitted but not yet started. Distinguishes the brief
      window where an ack has been issued before the worker picks up the
      work; useful for queued operations.
    * ``RUNNING`` / ``COMPLETED`` / ``FAILED`` are re-emitted by status
      surfaces after the work has begun or finished (see
      :mod:`polylogue.readiness.capability`). The initial scheduling
      response is always one of ``ACCEPTED``, ``REJECTED``, or ``PENDING``.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


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
