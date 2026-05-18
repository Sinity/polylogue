"""Typed import operation request/result models — first instance of #1247.

This module is the **first concrete instance** of the generic Operation
pattern defined in :mod:`polylogue.operations.operation_contract`. It
specifies what an import operation looks like on the wire: which fields
the request carries, what the ack returns, and how clients look up the
operation after scheduling.

Why this lives next to ``import_contracts.py``
-----------------------------------------------

:mod:`polylogue.operations.import_contracts` already owns a dataclass
:class:`ImportOperation` that captures the **terminal result envelope**
returned by daemon ingest, CLI ingest, and the live watcher. That
envelope is widely consumed across surfaces (``polylogue/api/contracts/*``,
``polylogue/cli/commands/ingest.py``, ``polylogue/daemon/http.py``) and
is not being replaced in this slice.

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

Future slices (#1248, #1249, #1250) wire surfaces to accept
:class:`ImportRequest` and emit :class:`ImportAck` so the existing
adapters can drop their per-adapter input/output shapes.

Fields aligned with the #1247 acceptance criteria
-------------------------------------------------

The issue calls out four scheduling fields explicitly. They land here as
explicit Pydantic attributes so type-checkers and surface tests both see
them:

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

The fifth scheduling concern — *follow-up status lookup hints* — lives
on :class:`OperationFollowUp` and is shared across every Operation, not
just import.
"""

from __future__ import annotations

from typing import ClassVar

from polylogue.operations.operation_contract import (
    OperationAck,
    OperationFollowUp,
    OperationRequest,
    OperationStatus,
)
from polylogue.operations.specs import OperationKind


class ImportRequest(OperationRequest):
    """Typed input for one import operation.

    Carries the four scheduling fields called out in #1247 (source_path,
    staged_path, source_name, idempotency_key) plus the explicit operation
    ``kind`` pinned to :attr:`OperationKind.IMPORT`.

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
    """

    operation_kind: ClassVar[OperationKind] = OperationKind.IMPORT

    source_path: str
    source_name: str
    staged_path: str | None = None


class ImportAck(OperationAck):
    """Scheduling acknowledgement specific to import operations.

    Inherits every field from :class:`OperationAck`. The subclass exists
    so per-operation factories can pin :attr:`kind` to
    :attr:`OperationKind.IMPORT` without callers having to pass it on
    every call site, and so future import-specific ack fields (for
    example a per-source retry policy hint) land here instead of on the
    shared base.

    The factory helpers below ensure the kind is set correctly.
    """

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
            kind=OperationKind.IMPORT,
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
            kind=OperationKind.IMPORT,
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
            kind=OperationKind.IMPORT,
            status=OperationStatus.PENDING,
            message=message,
            follow_up=follow_up,
        )


__all__ = [
    "ImportAck",
    "ImportRequest",
]
