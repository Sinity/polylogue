"""Minimal stdlib UDS client for daemon-owned maintenance and read routes."""

from __future__ import annotations

import errno
import http.client
import json
import socket
import uuid
from contextlib import suppress
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.operations.operation_context import OperationContext

from polylogue.operations.daemon_protocol import (
    DAEMON_OPERATION_PROTOCOL,
    MAX_OPERATION_RESULT_BYTES,
    AcceptedOperationReference,
    DaemonAuthority,
    DaemonOperationOutcome,
    DaemonOperationRequest,
    daemon_operation_spec,
    validate_operation_result,
)


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


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout: float | None) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path
        self.connected = False

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.socket_path))
        self.connected = True


class DaemonClient:
    """Transport adapter for the daemon's existing AF_UNIX HTTP routes."""

    def __init__(self, socket_path: Path, *, timeout_s: float | None = 0.1, auth_token: str | None = None) -> None:
        self.socket_path = socket_path
        self.timeout_s = timeout_s
        self.auth_token = auth_token
        self.last_elapsed_ms: int | None = None
        self.last_status: int | None = None

    def request_json(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        *,
        raise_for_status: bool = False,
        accepted_statuses: frozenset[int] = frozenset({200}),
    ) -> dict[str, Any] | None:
        response = self._request_json_response(method, path, body, mutation=False)
        if response is None:
            return None
        status, payload = response
        if status not in accepted_statuses:
            if raise_for_status:
                self._raise_response_error(status, payload)
            return None
        return payload

    @staticmethod
    def _raise_response_error(status: int, payload: dict[str, Any] | None) -> None:
        envelope = payload if isinstance(payload, dict) else {}
        code = envelope.get("error")
        detail = envelope.get("detail")
        raise DaemonResponseError(
            status=status,
            code=code if isinstance(code, str) else None,
            detail=detail if isinstance(detail, str) else None,
            payload=envelope,
        )

    def _request_json_response(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        *,
        mutation: bool = False,
        timeout_s: float | None = None,
    ) -> tuple[int, dict[str, Any] | None] | None:
        """Return the response status with its decoded JSON object, if any."""

        connection = _UnixHTTPConnection(self.socket_path, self.timeout_s if timeout_s is None else timeout_s)
        raw = json.dumps(body, separators=(",", ":")).encode() if body is not None else None
        started_at = perf_counter()
        try:
            headers = {"Host": "127.0.0.1", "Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            connection.request(method, path, body=raw, headers=headers)
            response = connection.getresponse()
            declared_length = response.getheader("Content-Length")
            if declared_length is not None and int(declared_length) > MAX_OPERATION_RESULT_BYTES:
                raise DaemonOperationProtocolError("daemon response exceeds the bounded result size")
            response_body = response.read(MAX_OPERATION_RESULT_BYTES + 1)
            if len(response_body) > MAX_OPERATION_RESULT_BYTES:
                raise DaemonOperationProtocolError("daemon response exceeds the bounded result size")
            try:
                decoded = json.loads(response_body.decode())
            except (UnicodeDecodeError, ValueError):
                decoded = None
            self.last_elapsed_ms = round((perf_counter() - started_at) * 1000)
            self.last_status = response.status
            return response.status, decoded if isinstance(decoded, dict) else None
        except KeyboardInterrupt as exc:
            if mutation and connection.connected:
                raise DaemonMutationIndeterminateError(
                    method=method, path=path, request_id=str((body or {}).get("request_id", ""))
                ) from exc
            raise
        except (OSError, TimeoutError, ValueError, http.client.HTTPException, DaemonOperationProtocolError) as exc:
            if mutation and connection.connected:
                raise DaemonMutationIndeterminateError(
                    method=method, path=path, request_id=str((body or {}).get("request_id", ""))
                ) from exc
            if (
                not connection.connected
                and isinstance(exc, OSError)
                and exc.errno in {errno.ENOENT, errno.ECONNREFUSED}
            ):
                return None
            raise DaemonOperationProtocolError("daemon transport failed; direct fallback is not permitted") from exc
        finally:
            connection.close()

    def operation(
        self,
        operation: str,
        payload: dict[str, object] | None = None,
        *,
        archive_root: str | None = None,
        index_schema_version: int | None = None,
        daemon_version: str | None = None,
        expected_archive_identity: str | None = None,
        expected_generation_id: str | None = None,
        request_id: str | None = None,
        deadline_ms: int | None = None,
        cancellation_token: str | None = None,
    ) -> dict[str, Any] | None:
        """Issue one archive-scoped operation request; no health probe is needed."""

        spec = daemon_operation_spec(operation)
        if spec is None:
            raise DaemonOperationProtocolError(f"operation is not declared: {operation}")
        request = DaemonOperationRequest(
            operation=operation,
            payload=payload or {},
            archive_root=archive_root,
            index_schema_version=index_schema_version,
            daemon_version=daemon_version,
            expected_archive_identity=expected_archive_identity,
            expected_generation_id=expected_generation_id,
            request_id=request_id or uuid.uuid4().hex,
            deadline_ms=deadline_ms or max(1, round(spec.deadline_s * 1000)),
            cancellation_token=cancellation_token,
        )
        request = DaemonOperationRequest.from_dict(request.to_dict())
        # A write never gives up the way a read does: once the request is on
        # the socket, an offline retry would make the actuator outcome
        # ambiguous, so the transport reports indeterminacy instead of absence.
        writes = spec.authority is not DaemonAuthority.READ
        # The server owns the execution deadline. Allow its bounded response
        # to arrive afterward without mutating a client shared by other calls.
        deadline_ms = request.deadline_ms
        if writes and deadline_ms is None:
            raise DaemonOperationProtocolError("write operation request has no execution deadline")
        raw = self._request_json_response(
            "POST",
            "/api/operation",
            request.to_dict(),
            mutation=writes,
            timeout_s=(deadline_ms / 1000 + 1.0) if writes and deadline_ms is not None else None,
        )
        if raw is None:
            return None
        status, response = raw
        if (
            status in {401, 503}
            and isinstance(response, dict)
            and response.get("protocol") == DAEMON_OPERATION_PROTOCOL
            and response.get("outcome") == "rejected"
            and isinstance(response.get("error"), dict)
            and (status, response["error"].get("code"))
            in {
                (401, "unauthorized"),
                (401, "peer_authentication_unavailable"),
                (503, "connection_backpressure"),
            }
        ):
            # The bounded ingress rejects these requests before dispatch. This
            # is a known refusal, not absence or a possibly committed mutation.
            raise DaemonOperationRejectedError(str(response["error"]["code"]))
        try:
            return self._validate_operation_response(request, status, response)
        except DaemonOperationProtocolError as exc:
            if writes:
                raise DaemonMutationIndeterminateError(
                    method="POST",
                    path="/api/operation",
                    request_id=request.request_id,
                ) from exc
            raise

    @staticmethod
    def _validate_operation_response(
        request: DaemonOperationRequest,
        status: int,
        response: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if status not in {200, 202, 400, 404, 408, 409, 413, 429, 503} or response is None:
            raise DaemonOperationProtocolError(f"daemon returned an incompatible operation response (HTTP {status})")
        if response.get("protocol") != DAEMON_OPERATION_PROTOCOL:
            raise DaemonOperationProtocolError("daemon returned an invalid operation protocol envelope")
        if response.get("operation") != request.operation or response.get("request_id") != request.request_id:
            raise DaemonOperationProtocolError("daemon returned a different operation or request id")
        archive = response.get("archive")
        if request.archive_root is not None and (
            not isinstance(archive, dict)
            or Path(str(archive.get("root", ""))).resolve() != Path(request.archive_root).resolve()
        ):
            raise DaemonOperationProtocolError("daemon returned a different archive identity")
        if response.get("outcome") not in {outcome.value for outcome in DaemonOperationOutcome}:
            raise DaemonOperationProtocolError("daemon returned an unknown lifecycle outcome")
        for key in (
            "archive",
            "generation",
            "readiness",
            "authority",
            "served_by",
            "timing",
            "schema_versions",
            "authority_snapshot",
        ):
            if not isinstance(response.get(key), dict):
                raise DaemonOperationProtocolError(f"daemon omitted typed {key} authority")
        timing = response["timing"]
        if any(type(timing.get(key)) is not int or timing[key] < 0 for key in ("elapsed_ms", "queue_ms")):
            raise DaemonOperationProtocolError("daemon returned invalid timing evidence")
        if any(type(value) is not int or value < 0 for value in response["schema_versions"].values()):
            raise DaemonOperationProtocolError("daemon returned invalid observed schema versions")
        reference = response.get("accepted_reference")
        if reference is not None and (
            not isinstance(reference, dict)
            or reference.get("request_id") != request.request_id
            or reference.get("archive_identity") != response["archive"].get("archive_identity")
        ):
            raise DaemonOperationProtocolError("daemon returned a mismatched durable request reference")
        try:
            if reference is not None:
                AcceptedOperationReference.model_validate(reference)
            if response.get("outcome") == "completed" and response.get("error") is None:
                validate_operation_result(request.operation, response.get("result"))
        except (ValueError, RuntimeError) as exc:
            raise DaemonOperationProtocolError(str(exc)) from exc
        return response

    def cancel(self, request_id: str, *, archive_root: str | None = None) -> dict[str, Any] | None:
        """Issue the declared control operation; completion is one exchange."""
        return self.operation(
            "operation.cancel",
            {"request_id": request_id},
            archive_root=archive_root,
        )

    def operation_with_read_fallback(
        self,
        operation: str,
        payload: dict[str, object] | None = None,
        *,
        context: OperationContext,
    ) -> dict[str, Any]:
        """Use the pinned reader only when the socket is absent.

        This fallback is deliberately named and scoped as a read fallback.
        ``operation()`` is the only route for writes; keeping the old generic
        name made it too easy for a new CLI adapter to mistake this for an
        offline mutation escape hatch.
        """
        from polylogue.operations.daemon_execution import execute_operation

        spec = daemon_operation_spec(operation)
        if spec is None:
            raise DaemonOperationProtocolError(f"operation is not declared: {operation}")
        response = self.operation(operation, payload, archive_root=str(context.archive_root))
        if response is not None:
            return response
        if not spec.direct_allowed or context.serving_identity != "direct" or context.runtime is not None:
            raise DaemonOperationRejected("daemon-required", "daemon is required for this operation")
        request = DaemonOperationRequest.from_dict(
            {
                "protocol": DAEMON_OPERATION_PROTOCOL,
                "operation": operation,
                "payload": payload or {},
                "archive_root": str(context.archive_root),
                "request_id": uuid.uuid4().hex,
            }
        )
        return execute_operation(request, context).to_dict()

    def await_operation(
        self,
        request_id: str,
        *,
        archive_root: str,
        after_sequence: int = 0,
        timeout_ms: int = 30_000,
    ) -> dict[str, Any] | None:
        """Wait on the durable event sequence using the same bounded POST endpoint."""
        return self.operation(
            "operation.await",
            {"request_id": request_id, "after_sequence": after_sequence, "timeout_ms": timeout_ms},
            archive_root=archive_root,
        )

    def operation_to_completion(
        self,
        operation: str,
        payload: dict[str, object],
        *,
        archive_root: str,
        request_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Follow accepted work with event-driven waits, never mutation retries."""
        spec = daemon_operation_spec(operation)
        if spec is None:
            raise DaemonOperationProtocolError(f"operation is not declared: {operation}")
        deadline = perf_counter() + spec.deadline_s
        try:
            envelope = self.operation(operation, payload, archive_root=archive_root, request_id=request_id)
        except DaemonMutationIndeterminateError as exc:
            if isinstance(exc.__cause__, KeyboardInterrupt) and exc.request_id:
                # The request identity exists before its first byte is sent.
                # Interrupting the initial POST must signal the same accepted
                # work, while retaining indeterminate truth until a receipt.
                with suppress(
                    DaemonMutationIndeterminateError, DaemonOperationProtocolError, DaemonOperationRejectedError
                ):
                    self.cancel(exc.request_id, archive_root=archive_root)
            raise
        if envelope is None or envelope.get("outcome") not in {"accepted", "running"}:
            return envelope
        target = str(envelope["request_id"])
        state = envelope.get("result")
        sequence = int(state.get("sequence", 0)) if isinstance(state, dict) else 0
        while perf_counter() < deadline:
            timeout_ms = max(1, min(30_000, int((deadline - perf_counter()) * 1000)))
            try:
                waited = self.await_operation(
                    target, archive_root=archive_root, after_sequence=sequence, timeout_ms=timeout_ms
                )
            except DaemonMutationIndeterminateError as exc:
                if isinstance(exc.__cause__, KeyboardInterrupt):
                    with suppress(
                        DaemonMutationIndeterminateError, DaemonOperationProtocolError, DaemonOperationRejectedError
                    ):
                        self.cancel(target, archive_root=archive_root)
                raise
            except KeyboardInterrupt:
                with suppress(
                    DaemonMutationIndeterminateError, DaemonOperationProtocolError, DaemonOperationRejectedError
                ):
                    self.cancel(target, archive_root=archive_root)
                raise
            if waited is None:
                raise DaemonMutationIndeterminateError(method="POST", path="/api/operation", request_id=target)
            state = waited.get("result")
            if not isinstance(state, dict) or "sequence" not in state or "outcome" not in state:
                raise DaemonOperationProtocolError("operation await omitted its durable lifecycle")
            reference = state.get("reference")
            accepted = envelope.get("accepted_reference")
            if (
                not isinstance(reference, dict)
                or not isinstance(accepted, dict)
                or any(
                    reference.get(key) != accepted.get(key)
                    for key in (
                        "request_id",
                        "archive_identity",
                        "principal_ref",
                        "fingerprint",
                        "operation_name",
                    )
                )
            ):
                raise DaemonOperationProtocolError("operation await returned a different durable request")
            sequence = int(state["sequence"])
            if state["outcome"] not in {"accepted", "running"}:
                result = state.get("result", state)
                if state["outcome"] == "completed":
                    try:
                        validate_operation_result(operation, result)
                    except RuntimeError as exc:
                        raise DaemonOperationProtocolError(str(exc)) from exc
                # Receipt recovery observed source/audit authority, not the
                # original executing reader. Keep that actual provenance and
                # its matching timing instead of synthesizing an all-tier pin.
                return {
                    **envelope,
                    **{
                        key: waited[key]
                        for key in (
                            "archive",
                            "generation",
                            "readiness",
                            "served_by",
                            "timing",
                            "schema_versions",
                            "authority_snapshot",
                            "degraded_components",
                        )
                    },
                    "outcome": state["outcome"],
                    "result": result,
                    "accepted_reference": reference,
                    "progress": {"state": state["outcome"]},
                }
        return {**envelope, "outcome": "indeterminate", "result": state}


__all__ = [
    "DaemonClient",
    "DaemonMutationIndeterminateError",
    "DaemonOperationProtocolError",
    "DaemonOperationRejected",
    "DaemonOperationRejectedError",
    "DaemonResponseError",
]
