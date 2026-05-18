"""Typed Operation contract — the reusable foundation across mutating actions.

This module defines the **generic Operation pattern** that Polylogue uses to
schedule, accept, and report on mutating work across surfaces (CLI, MCP,
daemon HTTP, in-process API). It is the strategic foundation called out in
#1247: the first instance lands for ingest in :mod:`polylogue.operations.import_operations`,
and subsequent slices (#845 convergence work, #996 maintenance cluster,
#999 health) consume the same contract instead of inventing parallel ones.

Design goals
------------

* **One shape, many operations.** Every mutating action — ingest, reindex,
  rebuild-insights, gc, replay, schema-upgrade — is a typed Operation with the
  same scheduling envelope: ``operation_id``, accepted/rejected ``status``,
  ``follow_up`` lookup hints for clients that poll, ``submitted_at``
  timestamp, and an optional human ``message``.
* **Request and Result are typed siblings.** A subclass of
  :class:`OperationRequest` declares its input fields. The accepted/rejected
  ack carries a :class:`OperationFollowUp` so the client knows exactly which
  endpoint, polling token, or status query to use next.
* **Pydantic with ``extra='forbid'`` and ``frozen=True``.** Adding a field is
  a deliberate API change, not accidental drift. Surfaces cannot stash
  extra keys that the parity tests will not see.
* **JSON byte-stable serialization.** :meth:`OperationAck.to_dict` and
  :meth:`OperationFollowUp.to_dict` reduce tuples and enums to JSON
  primitives so every surface (CLI ``--json``, MCP tool result, daemon HTTP
  body) emits identical bytes for identical inputs.
* **Per-operation registry on the side, not here.** This module owns the
  *contract*; the registry that maps operation names to handlers is the
  consumer's responsibility (daemon HTTP for now). Keeping the contract
  free of the registry keeps it import-cheap and reusable from contexts
  that should not pull in the daemon.

Reuse plan for the broader Operations pattern
---------------------------------------------

Each future operation defines a request and an accepted ack by subclassing
:class:`OperationRequest` and either using :class:`OperationAck` directly or
extending it. The shared :class:`OperationFollowUp` shape encodes the
``status_endpoint`` / ``status_token`` / ``poll_after_ms`` triple that
every client needs to track in-flight work.

Concrete examples staged behind this foundation:

* **Import** (this PR): :class:`polylogue.operations.import_operations.ImportRequest`
  declares ``source_path``, ``source_name``, optional ``staged_path``, and an
  ``idempotency_key``; the ack returns the operation id plus a follow-up
  hint pointing at the daemon status endpoint.
* **Maintenance** (#996): ``MaintenanceRequest`` will declare the
  maintenance ``kind`` and ``scope``; the existing
  :class:`polylogue.maintenance.envelope.MaintenanceOperationEnvelope`
  becomes the structural superset of the ack for terminal results.
* **Convergence** (#845): ``ConvergenceRequest`` will declare the
  convergence ``stage`` and ``cursor``; the ack returns a follow-up that
  routes the caller to the convergence-status surface.
* **Schema upgrade** (post-#1247): ``SchemaUpgradeRequest`` will declare
  the target ``schema_version`` and ``dry_run`` flag; rejected acks carry
  the structured incompatibility reason as ``error``.

Each instance keeps its own request/result subclass so type-checkers can
see the explicit input fields, while clients can rely on the shared
:class:`OperationStatus` / :class:`OperationFollowUp` envelope shape.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

from pydantic import ConfigDict

from polylogue.core.json import JSONDocument, json_document
from polylogue.operations.specs import OperationKind
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
    * ``RUNNING`` / ``COMPLETED`` / ``FAILED`` are reserved for status
      surfaces that re-emit an ack after the work has begun or finished.
      The initial scheduling response is always one of ``ACCEPTED``,
      ``REJECTED``, or ``PENDING``.
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


class OperationRequest(SurfacePayloadModel):
    """Base class for typed operation requests.

    Subclasses declare the per-operation input fields. The base carries
    metadata that is shared across every Operation — the operation
    ``kind`` (matched to :class:`polylogue.operations.specs.OperationKind`)
    and an optional caller-supplied ``idempotency_key`` so retries do not
    schedule duplicate work.

    Subclasses must set the class variable ``operation_kind`` to declare
    which :class:`OperationKind` they target. The base ``kind`` field is
    pinned to that value via :meth:`__init_subclass__` so callers do not
    have to repeat it at every call site.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Subclasses pin the operation kind via this class variable so the
    #: request shape carries the kind without requiring callers to pass it.
    operation_kind: ClassVar[OperationKind]

    idempotency_key: str | None = None

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json"))


class OperationAck(SurfacePayloadModel):
    """Scheduling acknowledgement returned by every Operation surface.

    A surface accepts a typed :class:`OperationRequest`, validates it, and
    returns an ``OperationAck``. The ack is the **single shape every
    client parses** — the per-operation request/result subclass carries
    domain detail, but the scheduling envelope is universal so clients can
    write one polling helper instead of N.

    Fields
    ------
    operation_id:
        Stable identifier the surface assigned. Required even for rejected
        operations so clients can correlate logs.
    kind:
        Operation kind from :class:`OperationKind`. Lets clients route the
        ack to the right per-kind handler without re-parsing the request.
    status:
        :class:`OperationStatus` — accepted, rejected, or pending. Other
        values are reserved for status-surface re-emissions.
    message:
        Human-readable summary. Always safe to log.
    error:
        Populated when ``status == REJECTED`` (or ``FAILED`` for terminal
        re-emissions). ``None`` when the operation was admitted.
    follow_up:
        Populated for accepted/pending operations. Tells the client which
        status surface to poll and how often.
    """

    operation_id: str
    kind: OperationKind
    status: OperationStatus
    message: str = ""
    error: str | None = None
    follow_up: OperationFollowUp | None = None

    @classmethod
    def accepted(
        cls,
        *,
        operation_id: str,
        kind: OperationKind,
        follow_up: OperationFollowUp,
        message: str = "",
    ) -> OperationAck:
        return cls(
            operation_id=operation_id,
            kind=kind,
            status=OperationStatus.ACCEPTED,
            message=message,
            follow_up=follow_up,
        )

    @classmethod
    def rejected(
        cls,
        *,
        operation_id: str,
        kind: OperationKind,
        error: str,
        message: str = "",
    ) -> OperationAck:
        return cls(
            operation_id=operation_id,
            kind=kind,
            status=OperationStatus.REJECTED,
            error=error,
            message=message,
        )

    @classmethod
    def pending(
        cls,
        *,
        operation_id: str,
        kind: OperationKind,
        follow_up: OperationFollowUp,
        message: str = "",
    ) -> OperationAck:
        return cls(
            operation_id=operation_id,
            kind=kind,
            status=OperationStatus.PENDING,
            message=message,
            follow_up=follow_up,
        )

    def is_accepted(self) -> bool:
        return self.status in (OperationStatus.ACCEPTED, OperationStatus.PENDING)

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json", exclude_none=False))


def _require_operation_kind(cls: type[OperationRequest]) -> OperationKind:
    """Read the ``operation_kind`` class variable from a request subclass.

    Raises ``TypeError`` if a subclass forgets to declare its kind — the
    contract requires every operation to pick its kind explicitly.
    """
    kind: Any = getattr(cls, "operation_kind", None)
    if not isinstance(kind, OperationKind):
        raise TypeError(f"{cls.__name__} must declare 'operation_kind: ClassVar[OperationKind]'")
    return kind


__all__ = [
    "OperationAck",
    "OperationFollowUp",
    "OperationRequest",
    "OperationStatus",
    "_require_operation_kind",
]
