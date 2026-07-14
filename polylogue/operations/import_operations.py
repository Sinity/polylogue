"""Typed import operation request/result models.

These are the **only** concrete instance of what used to be a generic
Operation subclassing pattern (see the collapse note in
:mod:`polylogue.operations.operation_contract`, polylogue-a7xr.14): ``kind``
is a fixed literal rather than a subclass-declared class variable, since
there is exactly one operation to pin it to.

Why this lives next to ``import_contracts.py``
-----------------------------------------------

:mod:`polylogue.operations.import_contracts` already owns a dataclass
:class:`ImportOperation` that captures the **terminal result envelope**
returned by daemon ingest, CLI ingest, and the live watcher. That
envelope is widely consumed across surfaces (``polylogue/api/contracts/*``,
``polylogue/cli/commands/ingest.py``, ``polylogue/daemon/http.py``) and
is not affected by this module.

The models defined here are the **typed scheduling contract** — the
request that surfaces accept, and the ack they return at admission time
— with explicit Pydantic fields and the shared follow-up envelope. The
two layers compose:

* :class:`ImportRequest` carries the input the operator (or the CLI/MCP/
  daemon adapter) wants to import.
* :class:`ImportAck` is the immediate response: operation id, accepted/
  rejected status, follow-up hint for polling the daemon status surface.
* :class:`ImportOperation` (the existing dataclass) is the terminal
  result envelope re-emitted by the status surface once the work has
  completed or failed.

Fields
------

* ``source_path`` — the originating filesystem path or URI the caller
  asked to ingest.
* ``staged_path`` — optional path the surface copied the input to (e.g.
  daemon HTTP stages uploads into ``archive_root/inbox``). ``None`` when
  no staging happened.
* ``source_name`` — caller-supplied label that identifies the logical
  source (e.g. ``"claude-code-export-2026-01"``). Distinct from
  ``source_path`` so the same source can be re-uploaded under a stable
  name.
* ``operation_id`` — assigned by the surface and echoed in the ack.
* Follow-up status lookup hints live on
  :class:`polylogue.operations.operation_contract.OperationFollowUp`.

Wire stability: the daemon HTTP ``POST /api/ingest`` response consumes
``ImportAck.to_dict()`` directly — its JSON field names/shape are the
production wire contract and are unchanged by the polylogue-a7xr.14 collapse
(only the abstract ``OperationRequest``/``OperationAck`` base classes above
this module were removed).
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import ConfigDict

from polylogue.core.json import JSONDocument, json_document
from polylogue.operations.operation_contract import OperationFollowUp, OperationStatus
from polylogue.operations.specs import OperationKind
from polylogue.surfaces.payloads import SurfacePayloadModel


class ImportRequest(SurfacePayloadModel):
    """Typed input for one import operation.

    Carries the four scheduling fields (source_path, staged_path,
    source_name, idempotency_key). ``operation_kind`` is a fixed
    ``ClassVar`` (not a pydantic field — it does not appear in
    :meth:`to_dict`'s wire payload, matching the shape from before the
    polylogue-a7xr.14 collapse) since this model only ever carries
    :attr:`OperationKind.IMPORT`.

    Surfaces validate the request with Pydantic at the boundary — invalid
    types or unknown extra keys raise :class:`pydantic.ValidationError`
    before any work is scheduled.

    Fields
    ------
    source_path:
        Path or URI the caller wants to ingest. Required. The contract
        does not constrain the value to an existing filesystem path so
        URI-shaped sources (https, drive://) remain compatible.
    source_name:
        Caller-supplied label that names the logical source. Required so
        the surface can record per-source progress without parsing the
        path.
    staged_path:
        Optional path the surface should treat as the actual byte source.
        Surfaces that copy/upload before processing (daemon HTTP ingest,
        ``polylogue ingest --copy``) populate this; in-place surfaces
        leave it ``None``.
    idempotency_key:
        Optional caller-supplied key so retries do not schedule duplicate
        work.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_kind: ClassVar[OperationKind] = OperationKind.IMPORT

    source_path: str
    source_name: str
    staged_path: str | None = None
    idempotency_key: str | None = None

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json"))


class ImportAck(SurfacePayloadModel):
    """Scheduling acknowledgement for an import operation.

    The single shape every import client parses — operation id,
    accepted/rejected/pending ``status``, and (for accepted/pending) a
    :class:`OperationFollowUp` telling the client where and how often to
    poll. ``kind`` is fixed to :attr:`OperationKind.IMPORT`.

    Fields
    ------
    operation_id:
        Stable identifier the surface assigned. Required even for rejected
        operations so clients can correlate logs.
    status:
        :class:`OperationStatus` — accepted, rejected, or pending. Other
        values are reserved for status-surface re-emissions.
    message:
        Human-readable summary. Always safe to log.
    error:
        Populated when ``status == REJECTED``. ``None`` when the
        operation was admitted.
    follow_up:
        Populated for accepted/pending operations. Tells the client which
        status surface to poll and how often.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str
    kind: Literal[OperationKind.IMPORT] = OperationKind.IMPORT
    status: OperationStatus
    message: str = ""
    error: str | None = None
    follow_up: OperationFollowUp | None = None

    @classmethod
    def accept_import(
        cls,
        *,
        operation_id: str,
        follow_up: OperationFollowUp,
        message: str = "",
    ) -> ImportAck:
        return cls(
            operation_id=operation_id,
            status=OperationStatus.ACCEPTED,
            message=message,
            follow_up=follow_up,
        )

    @classmethod
    def reject_import(
        cls,
        *,
        operation_id: str,
        error: str,
        message: str = "",
    ) -> ImportAck:
        return cls(
            operation_id=operation_id,
            status=OperationStatus.REJECTED,
            error=error,
            message=message,
        )

    @classmethod
    def pending_import(
        cls,
        *,
        operation_id: str,
        follow_up: OperationFollowUp,
        message: str = "",
    ) -> ImportAck:
        return cls(
            operation_id=operation_id,
            status=OperationStatus.PENDING,
            message=message,
            follow_up=follow_up,
        )

    def is_accepted(self) -> bool:
        return self.status in (OperationStatus.ACCEPTED, OperationStatus.PENDING)

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json", exclude_none=False))


__all__ = [
    "ImportAck",
    "ImportRequest",
]
