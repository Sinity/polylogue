"""Import-light operation dispatch for CLI adapters.

The CLI owns syntax and delivery only.  This module is the small seam between
those concerns and an operation transport: both daemon and direct execution
return the same typed result, while authority metadata records which executor
served it.  It intentionally has no archive, storage, or daemon-server
imports.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from polylogue.operations.daemon_protocol import (
    MAX_OPERATION_RESULT_BYTES,
    DaemonAuthority,
    DaemonOperationSpec,
    OperationStatus,
    daemon_operation_spec,
)


class OperationKernelError(RuntimeError):
    """Base class for typed operation dispatch failures."""


class OperationUnavailableError(OperationKernelError):
    """The daemon is absent and the operation cannot execute directly."""


class OperationEnvelopeError(OperationKernelError):
    """The selected transport did not return the declared operation envelope."""


class OperationFailedError(OperationKernelError):
    """The selected executor returned a typed operation error."""

    def __init__(self, code: str, detail: object = None, data: Mapping[str, object] | None = None) -> None:
        self.code = code
        self.detail = detail
        self.data: Mapping[str, object] = data or {}
        super().__init__(f"{code}: {detail}" if detail else code)


class OperationIndeterminateError(OperationKernelError):
    """A write request reached the daemon and no receipt came back.

    Distinct from :class:`OperationUnavailableError`: the daemon may have
    applied the write, so retrying is not safe.
    """


class OperationCancelledError(OperationKernelError):
    """The operation reached a cancelled terminal state."""

    def __init__(self, operation: str, detail: object = None) -> None:
        self.operation = operation
        self.detail = detail
        super().__init__(f"{operation} was cancelled" if detail is None else f"{operation} was cancelled: {detail}")


def _result_size(value: object) -> int:
    """Return the bounded wire size of a JSON-compatible operation result."""
    try:
        return len(json.dumps(value, separators=(",", ":"), default=str).encode())
    except (TypeError, ValueError, OverflowError) as exc:
        raise OperationEnvelopeError("operation result is not JSON serializable") from exc


@dataclass(frozen=True, slots=True)
class OperationRequest:
    """A lowered operation request; no surface-specific query vocabulary."""

    operation: str
    payload: Mapping[str, object]

    @property
    def spec(self) -> DaemonOperationSpec:
        spec = daemon_operation_spec(self.operation)
        if spec is None:
            raise OperationKernelError(f"operation is not declared: {self.operation}")
        return spec


@dataclass(frozen=True, slots=True)
class OperationResult:
    """Typed result shared by daemon and direct execution paths."""

    operation: str
    value: object
    authority: Mapping[str, object]
    envelope: Mapping[str, object] | None = None


OperationCall = Callable[[OperationRequest], Mapping[str, Any] | None]
DirectCall = Callable[[OperationRequest], object]


class OperationKernel:
    """Dispatch one declared operation without changing its semantics.

    ``daemon_call`` returns a protocol envelope or ``None`` when the daemon is
    unavailable.  Only read operations may then use ``direct_call``.  A
    daemon response containing an error is final: falling through to a local
    executor would turn a typed server result into an unsafe semantic retry.
    """

    def __init__(self, daemon_call: OperationCall, direct_call: DirectCall | None = None) -> None:
        self._daemon_call = daemon_call
        self._direct_call = direct_call

    def execute(self, request: OperationRequest) -> OperationResult:
        spec = request.spec
        try:
            envelope = self._daemon_call(request)
        except (TimeoutError, ConnectionError, OSError):
            envelope = None
        except Exception as exc:
            # The stdlib daemon client uses typed transport errors. Keep those
            # distinctions visible to callers while preserving direct fallback
            # for ordinary daemon absence.
            name = type(exc).__name__
            if name == "DaemonMutationIndeterminateError":
                raise OperationIndeterminateError(str(exc)) from exc
            if name == "DaemonOperationProtocolError" and "size" in str(exc):
                raise OperationFailedError("result_too_large", str(exc)) from exc
            raise OperationFailedError("daemon_transport_error", str(exc)) from exc
        if envelope is not None:
            if envelope.get("operation") not in (None, request.operation):
                raise OperationEnvelopeError("daemon returned a different operation")
            error = envelope.get("error")
            if isinstance(error, Mapping):
                code = error.get("code")
                data = error.get("data")
                raise OperationFailedError(
                    str(code or "operation_failed"),
                    error.get("detail"),
                    data if isinstance(data, Mapping) else None,
                )
            if error is not None:
                raise OperationEnvelopeError("daemon returned a malformed error envelope")
            outcome = envelope.get("outcome", OperationStatus.COMPLETED.value)
            if outcome == OperationStatus.INTERRUPTED.value:
                raise OperationCancelledError(request.operation, envelope.get("detail"))
            if outcome not in {OperationStatus.COMPLETED.value, OperationStatus.ACCEPTED.value}:
                raise OperationFailedError(str(outcome), envelope.get("detail"))
            if "result" not in envelope:
                raise OperationEnvelopeError("daemon response omitted the operation result")
            value = envelope.get("result")
            if _result_size(value) > MAX_OPERATION_RESULT_BYTES:
                raise OperationFailedError("result_too_large", "daemon operation result exceeds the bounded size")
            generation = envelope.get("generation")
            if isinstance(generation, Mapping) and generation.get("state") in {"stale", "mismatch"}:
                raise OperationFailedError("stale_generation", generation.get("reason"))
            authority = envelope.get("authority")
            if not isinstance(authority, Mapping):
                authority = {"mode": "daemon", "class": spec.authority.value}
            else:
                authority = {"mode": "daemon", "class": spec.authority.value, **authority}
            return OperationResult(
                request.operation,
                value,
                authority,
                envelope,
            )

        if not spec.direct_allowed or spec.authority is not DaemonAuthority.READ or self._direct_call is None:
            raise OperationUnavailableError(f"daemon is unavailable for operation: {request.operation}")
        value = self._direct_call(request)
        if _result_size(value) > MAX_OPERATION_RESULT_BYTES:
            raise OperationFailedError("result_too_large", "direct operation result exceeds the bounded size")
        return OperationResult(
            request.operation,
            value,
            {"mode": "direct", "class": spec.authority.value, "fallback": spec.fallback.value},
        )


__all__ = [
    "OperationCancelledError",
    "OperationFailedError",
    "OperationIndeterminateError",
    "OperationEnvelopeError",
    "OperationKernel",
    "OperationKernelError",
    "OperationRequest",
    "OperationResult",
    "OperationUnavailableError",
]
