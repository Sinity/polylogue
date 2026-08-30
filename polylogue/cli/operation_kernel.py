"""Import-light operation dispatch for CLI adapters.

The CLI owns syntax and delivery only.  This module is the small seam between
those concerns and an operation transport: both daemon and direct execution
return the same typed result, while authority metadata records which executor
served it.  It intentionally has no archive, storage, or daemon-server
imports.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from polylogue.operations.daemon_protocol import (
    DaemonAuthority,
    DaemonOperationSpec,
    daemon_operation_spec,
)


class OperationKernelError(RuntimeError):
    """Base class for typed operation dispatch failures."""


class OperationUnavailableError(OperationKernelError):
    """The daemon is absent and the operation cannot execute directly."""


class OperationFailedError(OperationKernelError):
    """The selected executor returned a typed operation error."""

    def __init__(self, code: str, detail: object = None) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


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


def execute_candidate_semantic(request: object) -> object:
    """Run the declared candidate semantic operation through the CLI seam."""

    from polylogue.operations.candidate_proof import run_candidate_semantic

    return run_candidate_semantic(request)  # type: ignore[arg-type]


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
        envelope = self._daemon_call(request)
        if envelope is not None:
            error = envelope.get("error")
            if isinstance(error, Mapping):
                code = error.get("code")
                raise OperationFailedError(str(code or "operation_failed"), error.get("detail"))
            value = envelope.get("result")
            authority = envelope.get("authority")
            return OperationResult(
                request.operation,
                value,
                authority if isinstance(authority, Mapping) else {"mode": "daemon", "class": spec.authority.value},
                envelope,
            )

        if not spec.direct_allowed or spec.authority is not DaemonAuthority.READ or self._direct_call is None:
            raise OperationUnavailableError(f"daemon is unavailable for operation: {request.operation}")
        return OperationResult(
            request.operation,
            self._direct_call(request),
            {"mode": "direct", "class": spec.authority.value, "fallback": spec.fallback.value},
        )


__all__ = [
    "OperationFailedError",
    "OperationKernel",
    "OperationKernelError",
    "OperationRequest",
    "OperationResult",
    "OperationUnavailableError",
    "execute_candidate_semantic",
]
