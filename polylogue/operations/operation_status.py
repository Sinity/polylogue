"""``OperationStatus``: the scheduling status enum shared by every Operation ack.

Split out of :mod:`polylogue.operations.operation_contract` (polylogue-8s70):
that module's other export, ``OperationFollowUp``, subclasses
``polylogue.surfaces.payloads.SurfacePayloadModel`` (a pydantic base), which
pulls in a wide swath of the archive/semantic pricing and payload machinery
(observed ~280ms of import cost). ``OperationStatus`` is a plain ``str, Enum``
with zero runtime dependency on any of that -- but importing it from the same
module used to force the whole chain onto callers who only want the status
enum, notably :mod:`polylogue.readiness.capability`, which is on the hot path
of every ``polylogue status``/``polylogue agents status`` invocation.
"""

from __future__ import annotations

from enum import Enum


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


__all__ = ["OperationStatus"]
