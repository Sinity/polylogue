"""Typed daemon-client failures, importable without the client itself.

These are the exceptions the daemon transport raises.  Callers that must
distinguish an indeterminate mutation from an ordinary transport failure need
the *classes*, not the transport: keeping them here lets an import-light
dispatcher (``cli/operation_kernel.py``) branch on ``isinstance`` instead of
comparing ``type(exc).__name__``, which silently stopped matching whenever a
class was renamed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DaemonMutationIndeterminateError",
    "DaemonOperationProtocolError",
    "DaemonOperationRejected",
    "DaemonOperationRejectedError",
    "DaemonResponseError",
]


class DaemonResponseError(RuntimeError):
    """A daemon response with a typed non-success HTTP envelope."""

    def __init__(
        self,
        *,
        status: int,
        code: str | None,
        detail: str | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.code = code
        self.detail = detail or code or f"daemon returned HTTP {status}"
        self.payload = payload or {}
        self.completed_chunks = self.payload.get("completed_chunks")
        self.affected_count = self.payload.get("affected_count")
        super().__init__(self.detail)


class DaemonMutationIndeterminateError(RuntimeError):
    """A confirmed mutation may have reached the daemon without a receipt."""

    def __init__(self, *, method: str, path: str, request_id: str | None = None) -> None:
        self.method = method
        self.path = path
        self.request_id = request_id
        super().__init__(f"daemon outcome is indeterminate after {method} {path}")


class DaemonOperationProtocolError(RuntimeError):
    """A daemon operation response was not a v1 typed envelope."""


class DaemonOperationRejectedError(RuntimeError):
    """The daemon refused an operation before durable acceptance."""

    def __init__(self, outcome: str, detail: str | None = None) -> None:
        self.outcome = outcome
        self.detail = detail or outcome
        super().__init__(self.detail)


DaemonOperationRejected = DaemonOperationRejectedError
