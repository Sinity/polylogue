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
from typing import Any, cast

from polylogue.operations.daemon_errors import (
    DaemonMutationIndeterminateError,
    DaemonOperationProtocolError,
)
from polylogue.operations.daemon_protocol import (
    DAEMON_OPERATION_PROTOCOL,
    MAX_OPERATION_RESULT_BYTES,
    DaemonOperationSpec,
    OperationStatus,
    daemon_operation_spec,
)


class OperationKernelError(RuntimeError):
    """Base class for typed operation dispatch failures."""


class OperationUnavailableError(OperationKernelError):
    """The daemon is absent and the operation cannot execute directly."""

    code = "daemon_required"

    def __init__(self, detail: object = None, *, operation: str | None = None) -> None:
        self.detail = detail
        #: The declared operation that had no executor, carried as a field so
        #: the machine envelope can name it without a client parsing it back
        #: out of the message (polylogue-re6s3 AC4).
        self.operation = operation
        super().__init__(str(detail) if detail is not None else self.code)


class OperationEnvelopeError(OperationKernelError):
    """The selected transport did not return the declared operation envelope."""


class OperationFailedError(OperationKernelError):
    """The selected executor returned a typed operation error."""

    def __init__(
        self,
        code: str,
        detail: object = None,
        data: Mapping[str, object] | None = None,
        *,
        request_id: str | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.data: Mapping[str, object] = data or {}
        #: The transport's call id when it reported one. A daemon-side failure
        #: is correlated with the daemon log by this id, so the adapter that
        #: renders the refusal can name it (polylogue-jtrtj).
        self.request_id = request_id
        super().__init__(f"{code}: {detail}" if detail else code)


class OperationIndeterminateError(OperationKernelError):
    """A write request reached the daemon and no receipt came back.

    Distinct from :class:`OperationUnavailableError`: the daemon may have
    applied the write, so retrying is not safe.

    ``request_id`` is the transport's call id for the unresolved write, and it
    is the only recovery authority a caller has: the daemon's durable
    lifecycle, ``DaemonClient.await_operation`` and ``DaemonClient.cancel`` are
    all keyed on it. Dropping it turns "inspect daemon audit state before
    retrying" into a lookup with no key, which is how the sibling REST defect
    in polylogue-8r4zq reads on this side of the socket.
    """

    def __init__(self, detail: object = None, *, request_id: str | None = None) -> None:
        self.detail = detail
        self.request_id = request_id or None
        message = str(detail) if detail is not None else "daemon outcome is indeterminate"
        super().__init__(f"{message} (request {self.request_id})" if self.request_id else message)


class OperationCancelledError(OperationKernelError):
    """The operation reached a cancelled terminal state.

    ``code`` carries the executor's own name for the abort when it supplied one
    (``QueryCancelledError`` for a deadline/cancellation abort) so a surface can
    render the typed refusal rather than re-deriving one from the class.
    ``request_id`` carries the transport's call id so the operator-facing
    refusal can name the call to correlate in the daemon log.
    """

    def __init__(
        self,
        operation: str,
        detail: object = None,
        code: str | None = None,
        *,
        request_id: str | None = None,
    ) -> None:
        self.operation = operation
        self.detail = detail
        self.code = code or "operation_cancelled"
        self.request_id = request_id
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


class OperationKernel:
    """Dispatch one declared operation without changing its semantics.

    ``daemon_call`` returns a protocol envelope or ``None`` when the daemon is
    unavailable. The canonical client owns any permitted direct read. A
    daemon response containing an error is final: falling through to a local
    executor would turn a typed server result into an unsafe semantic retry.
    """

    def __init__(self, daemon_call: OperationCall) -> None:
        self._daemon_call = daemon_call

    def execute(self, request: OperationRequest) -> OperationResult:
        spec = request.spec
        try:
            envelope = self._daemon_call(request)
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise OperationFailedError("daemon_transport_error", str(exc)) from exc
        except DaemonMutationIndeterminateError as exc:
            # A confirmed mutation may already have landed; never re-issue it.
            # The transport knows which call is unresolved, so carry that id
            # instead of flattening the exception to its message: it is what
            # settles the write.
            raise OperationIndeterminateError(str(exc), request_id=exc.request_id) from exc
        except DaemonOperationProtocolError as exc:
            if "size" in str(exc):
                raise OperationFailedError("result_too_large", str(exc)) from exc
            raise OperationFailedError("daemon_transport_error", str(exc)) from exc
        except Exception as exc:
            raise OperationFailedError("daemon_transport_error", str(exc)) from exc
        if envelope is not None:
            raw_call_id = envelope.get("request_id")
            call_id = str(raw_call_id) if raw_call_id else None
            if envelope.get("operation") not in (None, request.operation):
                raise OperationEnvelopeError("daemon returned a different operation")
            outcome = envelope.get("outcome", OperationStatus.COMPLETED.value)
            # An explanatory error cannot demote accepted, unresolved effects
            # to an ordinary failure that callers may safely retry.
            if outcome in {"indeterminate", "disconnected-after-acceptance", "restarted"}:
                raise OperationIndeterminateError(
                    f"{request.operation} requires receipt recovery",
                    request_id=call_id,
                )
            error = envelope.get("error")
            if outcome in {"cancelled", OperationStatus.INTERRUPTED.value}:
                # The abort's own error body names the deadline, the call id and
                # the executor's typed code; preferring the (usually absent)
                # result body dropped all three and left "<op> was cancelled".
                aborted = error if isinstance(error, Mapping) else {}
                raise OperationCancelledError(
                    request.operation,
                    aborted.get("detail") or envelope.get("result") or envelope.get("detail"),
                    code=str(aborted.get("code")) if aborted.get("code") else None,
                    request_id=call_id,
                )
            if isinstance(error, Mapping):
                code = error.get("code")
                data = error.get("data")
                raise OperationFailedError(
                    str(code or "operation_failed"),
                    error.get("detail"),
                    data if isinstance(data, Mapping) else None,
                    request_id=call_id,
                )
            if error is not None:
                raise OperationEnvelopeError("daemon returned a malformed error envelope")
            if outcome not in {OperationStatus.COMPLETED.value, OperationStatus.ACCEPTED.value}:
                result = envelope.get("result")
                raise OperationFailedError(
                    str(outcome),
                    envelope.get("detail"),
                    result if isinstance(result, Mapping) else None,
                    request_id=call_id,
                )
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

        raise OperationUnavailableError(
            f"daemon is unavailable for operation: {request.operation}", operation=request.operation
        )


def _direct_execution_context(config: Any, *, archive_root: Any = None, read_control: Any = None) -> Any:
    """Build the local read authority — only once a direct read is certain.

    ``DaemonReadDependencies`` resolves a vector binding, which opens the
    embeddings tier's configuration and can touch the filesystem.  A warm
    daemon answers without ever needing it, so constructing this eagerly made
    every served-over-UDS query pay for an executor it did not use.
    """
    from time import time

    from polylogue.operations.daemon_reads import DaemonReadDependencies, vector_binding_from_config
    from polylogue.operations.operation_context import OperationContext

    context = OperationContext.direct_read(
        archive_root if archive_root is not None else config.archive_root,
        read_dependencies=DaemonReadDependencies(
            vector_binding=vector_binding_from_config(config),
            status_now_ms=int(time() * 1000),
            status_config=config,
        ),
    )
    if read_control is not None:
        from dataclasses import replace as _replace

        context = _replace(context, read_control=read_control)
    return context


def _execute_directly(
    config: Any,
    operation: str,
    payload: dict[str, object],
    *,
    archive_root: Any = None,
    read_control: Any = None,
) -> Mapping[str, Any]:
    """Run one declared read in-process and return its envelope.

    A handler's typed *refusal of the request* — "semantic retrieval is
    unavailable", "sample does not combine with search terms" — is a result of
    the operation, not a transport fault, so it is framed as the declared error
    envelope here.  Letting it propagate as a bare exception would escape the
    kernel's error map and reach the operator as a traceback with no exit code.

    Deliberately narrow.  Faults that describe the *archive* rather than the
    request — a missing tier, a schema skew, a raw ``sqlite3`` error — keep
    propagating, because callers distinguish them: ``ops status`` reads a
    missing index as "no archive yet, daemon not running" and would otherwise
    report the far less useful "daemon unreachable".
    """
    import uuid

    from polylogue.core.errors import EmbeddingRetrievalNotReadyError
    from polylogue.operations.daemon_execution import execute_operation
    from polylogue.operations.daemon_protocol import DaemonOperationRequest

    request = DaemonOperationRequest.from_dict(
        {
            "protocol": DAEMON_OPERATION_PROTOCOL,
            "operation": operation,
            "payload": payload,
            "request_id": uuid.uuid4().hex,
            "archive_root": str(archive_root if archive_root is not None else config.archive_root),
        }
    )
    context = _direct_execution_context(config, archive_root=archive_root, read_control=read_control)
    try:
        return cast("Mapping[str, Any]", execute_operation(request, context).to_dict())
    except EmbeddingRetrievalNotReadyError as exc:
        return {
            "operation": operation,
            "outcome": "failed",
            "error": {"code": getattr(exc, "code", None) or "embedding_retrieval_not_ready", "detail": str(exc)},
        }
    except ValueError as exc:
        # Declared read handlers state a refused request as ``ValueError``.
        # A refusal that carries its own declared code keeps it: a stale or
        # foreign continuation is refused by the same token on every surface,
        # and flattening every refusal to ``invalid_request`` dropped exactly
        # the token a caller branches on.
        return {
            "operation": operation,
            "outcome": "failed",
            "error": {"code": str(getattr(exc, "code", "") or "invalid_request"), "detail": str(exc)},
        }


def dispatch(
    config: Any,
    request: OperationRequest,
    *,
    daemon_disabled: bool = False,
    daemon_only: bool = False,
    archive_root: Any = None,
    deadline_ms: int | None = None,
    read_control: Any = None,
) -> OperationResult:
    """Execute one declared read over the daemon, or directly when it is absent.

    The transport choice is the only thing decided here: the same declared
    handler answers either way, so a CLI route never branches on whether a
    daemon is running.

    The file set read is resolved from the configuration, not assumed to be
    ``config.archive_root``: a ``--db`` pin at a non-active generation names the
    file set that index belongs to, and reading the active generation instead
    would silently answer from different rows than the operator pinned.
    ``archive_root`` overrides that resolution for a caller that already knows
    the root.  ``deadline_ms`` overrides the operation's declared deadline for the socket
    call and ``read_control`` carries the caller's cancellation/deadline state
    into a direct read; all three are passed through rather than reinterpreted.

    ``daemon_only`` refuses the local fallback instead of taking it, raising
    :class:`OperationUnavailableError` when no socket answers. It exists for
    callers whose latency budget the direct reader cannot meet -- shell
    completion runs on a keystroke, and executing the read locally means
    building the execution graph and opening the archive for a TAB press. Such
    a caller wants the typed "no daemon" refusal, not a correct answer seconds
    later.
    """
    spec = request.spec
    operation = request.operation
    payload = dict(request.payload)
    if archive_root is not None:
        root = archive_root
    else:
        # Imported here, and from the addressing module rather than
        # ``operation_context``: resolving *where* to read must not pull in the
        # machinery for actually reading. A daemon-only dispatch may never open
        # an archive at all.
        from polylogue.operations.archive_root import operation_archive_root

        root = operation_archive_root(config)

    if daemon_disabled:
        if daemon_only or not spec.direct_allowed:
            raise OperationUnavailableError(f"daemon is unavailable for operation: {operation}", operation=operation)
        envelope = _execute_directly(config, operation, payload, archive_root=root, read_control=read_control)
        return OperationKernel(lambda _request: envelope).execute(request)

    from polylogue.daemon.api_auth import resolve_api_auth_token
    from polylogue.daemon.socket_path import daemon_socket_path
    from polylogue.daemon_client import DaemonClient

    client = DaemonClient(
        daemon_socket_path(root),
        timeout_s=(deadline_ms / 1000 if deadline_ms is not None else spec.deadline_s),
        auth_token=lambda: resolve_api_auth_token(
            getattr(config, "api_auth_token", None),
            allow_no_auth=getattr(config, "api_allow_no_auth", False),
        ),
    )

    def _ask_daemon(call_request: OperationRequest) -> Mapping[str, Any] | None:
        return client.operation(call_request.operation, dict(call_request.payload), archive_root=str(root))

    try:
        return OperationKernel(_ask_daemon).execute(request)
    except OperationUnavailableError:
        # No socket answered.  Fall through to the local reader.
        if daemon_only or not spec.direct_allowed:
            raise

    # Deliberately executed OUTSIDE the kernel.  The kernel's catch-all maps any
    # exception from its call into ``daemon_transport_error``, which is the right
    # reading of a failure while talking to a daemon and the wrong reading of one
    # raised by the local reader: callers distinguish "no archive here" from
    # "could not reach the daemon", and ``ops status`` renders those as two
    # different diagnoses. Only now is a direct read certain, so only now is its
    # context built.
    envelope = _execute_directly(config, operation, payload, archive_root=root, read_control=read_control)
    return OperationKernel(lambda _request: envelope).execute(request)


def configured_read_operation(
    config: Any,
    operation: str,
    payload: dict[str, object],
    *,
    daemon_disabled: bool = False,
) -> OperationResult:
    """Name-based adapter for read callers that have not adopted Seam A yet."""
    return dispatch(
        config,
        OperationRequest(operation, payload),
        daemon_disabled=daemon_disabled,
    )


def configured_mutation_operation(config: Any, operation: str, payload: dict[str, object]) -> dict[str, object]:
    """Execute a declared mutation through the resident daemon only.

    A missing socket becomes the typed ``daemon_required`` result; transport
    failures after connection remain indeterminate and are never retried
    through a local writer.
    """
    from polylogue.daemon.api_auth import resolve_api_auth_token
    from polylogue.daemon.socket_path import daemon_socket_path
    from polylogue.daemon_client import DaemonClient
    from polylogue.operations.archive_root import operation_archive_root
    from polylogue.operations.daemon_protocol import MUTATION_OPERATION_NAMES

    if operation not in MUTATION_OPERATION_NAMES:
        raise OperationKernelError(f"operation is not a declared mutation: {operation}")
    # The file set is resolved exactly as ``configured_read_operation``
    # resolves it. A ``--db`` pin at a non-active generation names the file set
    # the operator chose; selection and confirmation already ran against that
    # resolved root, so addressing the mutation at ``config.archive_root``
    # instead sent a confirmed delete to whichever daemon happened to serve the
    # configured root -- a different archive.
    root = operation_archive_root(config)
    client = DaemonClient(
        daemon_socket_path(root),
        auth_token=lambda: resolve_api_auth_token(
            getattr(config, "api_auth_token", None),
            allow_no_auth=getattr(config, "api_allow_no_auth", False),
        ),
    )
    result = OperationKernel(
        lambda request: client.operation_to_completion(
            request.operation,
            dict(request.payload),
            archive_root=str(root),
        )
    ).execute(OperationRequest(operation, payload))
    if not isinstance(result.value, Mapping):
        raise OperationEnvelopeError(f"{operation} returned a non-object result")
    return {str(key): value for key, value in result.value.items()}


def configured_accepted_operation(config: Any, operation: str, payload: dict[str, object]) -> dict[str, object]:
    """Submit a declared operation and take its durable acceptance reference.

    The sibling :func:`configured_mutation_operation` waits for the effect to
    land.  This one returns as soon as the daemon has *durably admitted* the
    work, which is the contract of a CLI verb that schedules rather than
    applies: ``polylogue import`` prints "Scheduled" and returns, and its
    ``--wait`` is a separate opt-in.  Waiting here would silently turn that
    into a blocking command.

    The declared operation envelope is returned whole, because acceptance is
    carried by ``accepted_reference`` — the durable handle — and not by the
    result body.

    An operation whose spec does not declare ``accepted_reference`` is
    refused: "submit and take the durable reference" is this function's whole
    contract, and there is nothing to take when the operation declares no
    durable reference, so returning early would just discard the outcome.
    """
    from polylogue.daemon.api_auth import resolve_api_auth_token
    from polylogue.daemon.socket_path import daemon_socket_path
    from polylogue.daemon_client import DaemonClient
    from polylogue.operations.archive_root import operation_archive_root
    from polylogue.operations.daemon_protocol import MUTATION_OPERATION_NAMES

    if operation not in MUTATION_OPERATION_NAMES:
        raise OperationKernelError(f"operation is not a declared mutation: {operation}")
    spec = daemon_operation_spec(operation)
    if spec is None or not spec.accepted_reference:
        raise OperationKernelError(f"operation does not declare a durable acceptance reference: {operation}")
    # Same resolution as the read and mutation routes: a scheduled write must
    # be admitted by the daemon that owns the pinned file set.
    root = operation_archive_root(config)
    client = DaemonClient(
        daemon_socket_path(root),
        timeout_s=spec.deadline_s,
        auth_token=lambda: resolve_api_auth_token(
            getattr(config, "api_auth_token", None),
            allow_no_auth=getattr(config, "api_allow_no_auth", False),
        ),
    )
    result = OperationKernel(
        lambda request: client.operation(
            request.operation,
            dict(request.payload),
            archive_root=str(root),
        )
    ).execute(OperationRequest(operation, payload))
    if result.envelope is None:
        raise OperationEnvelopeError(f"{operation} returned no operation envelope")
    return {str(key): value for key, value in result.envelope.items()}


__all__ = [
    "OperationCancelledError",
    "OperationEnvelopeError",
    "OperationFailedError",
    "OperationIndeterminateError",
    "OperationKernel",
    "OperationKernelError",
    "OperationRequest",
    "OperationResult",
    "OperationUnavailableError",
    "configured_accepted_operation",
    "configured_mutation_operation",
    "configured_read_operation",
    "dispatch",
]
