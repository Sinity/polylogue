"""Transport-only stdlib UDS client for the declared daemon operation protocol.

The public surface is deliberately the operation vocabulary and nothing else:
:meth:`DaemonClient.operation`, the control verbs built on it, and the one
named read fallback.  There is no way to ask this client for an arbitrary
daemon HTTP path, because the CLI rewrite's contract is that the warm path
speaks the archive-scoped operation protocol and never a browser route or a
separate liveness probe.  A generic ``request_json(method, path, ...)`` used
to sit here with no production caller, which made "the CLI issues no health
preflight" a claim about discipline rather than about the code.
"""

from __future__ import annotations

import errno
import http.client
import json
import os
import socket
import struct
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.operations.operation_context import OperationContext

from polylogue.operations.daemon_errors import (
    DaemonMutationIndeterminateError,
    DaemonOperationProtocolError,
    DaemonOperationRejected,
    DaemonOperationRejectedError,
    DaemonSocketOwnershipError,
)
from polylogue.operations.daemon_protocol import (
    DAEMON_OPERATION_OUTCOMES,
    DAEMON_OPERATION_PROTOCOL,
    MAX_OPERATION_RESULT_BYTES,
    AcceptedOperationReference,
    DaemonAuthority,
    DaemonOperationRequest,
    daemon_operation_spec,
    validate_operation_result,
)


def _reject_foreign_peer(sock: socket.socket, socket_path: Path) -> None:
    """Refuse a socket served by another user before any credential is sent.

    The daemon's socket path is predictable and its fallback locations sit
    under a shared temporary directory, so another local user can bind a
    replacement at the same name. This client sends ``Authorization: Bearer
    <machine token>`` on its first request, with no prior exchange that could
    establish who is listening -- so whoever answers receives the bearer. The
    kernel's ``SO_PEERCRED`` is the one answer that cannot be spoofed: it is
    filled in by the kernel at ``connect`` time, not by the peer, and it is
    read before a single header goes out.

    This mirrors ``polylogue.daemon.uds._peer_principal``, which is the same
    check in the other direction.
    """

    peercred = getattr(socket, "SO_PEERCRED", None)
    if peercred is None:  # pragma: no cover - non-Linux platforms
        return
    try:
        credentials = sock.getsockopt(socket.SOL_SOCKET, peercred, struct.calcsize("3i"))
        _pid, uid, _gid = struct.unpack("3i", credentials)
    except (OSError, struct.error) as exc:
        raise DaemonSocketOwnershipError(
            f"peer credentials are unavailable for {socket_path}; refusing to send the machine bearer"
        ) from exc
    if uid != os.geteuid():
        raise DaemonSocketOwnershipError(
            f"{socket_path} is served by uid {uid}, not {os.geteuid()}; refusing to send the machine bearer"
        )


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout: float | None) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path
        self.connected = False

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.socket_path))
        try:
            _reject_foreign_peer(self.sock, self.socket_path)
        except DaemonSocketOwnershipError:
            self.sock.close()
            self.sock = None
            raise
        self.connected = True


class DaemonClient:
    """Transport adapter for the daemon's existing AF_UNIX HTTP routes."""

    def __init__(
        self,
        socket_path: Path,
        *,
        timeout_s: float | None = 0.1,
        auth_token: str | None | Callable[[], str | None] = None,
    ) -> None:
        """Bind the transport; ``auth_token`` may be a thunk resolved after connect.

        The bearer token is a *credential for a daemon that answered*, not a
        precondition for having a transport object. Resolving it eagerly made
        every CLI read mint and persist a token into the archive root even
        when no daemon was listening -- a pure read writing to the thing it
        reads, and an outright failure on a read-only archive root. Passing a
        callable defers that cost to the moment a connection is established.
        """
        self.socket_path = socket_path
        self.timeout_s = timeout_s
        self._auth_token = auth_token
        self.last_elapsed_ms: int | None = None
        self.last_status: int | None = None

    @property
    def auth_token(self) -> str | None:
        """Resolve the configured token, memoizing a thunk's first answer."""
        if callable(self._auth_token):
            self._auth_token = self._auth_token()
        return self._auth_token

    @auth_token.setter
    def auth_token(self, value: str | None | Callable[[], str | None]) -> None:
        """Allow callers to override the credential, thunk or resolved alike."""
        self._auth_token = value

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
            # Connect before resolving credentials: an absent socket must cost
            # nothing, least of all a write into the archive root.
            connection.connect()
            headers = {"Host": "127.0.0.1", "Content-Type": "application/json"}
            token = self.auth_token
            if token:
                headers["Authorization"] = f"Bearer {token}"
            connection.request(method, path, body=raw, headers=headers)
            with connection.getresponse() as response:
                lengths = response.headers.get_all("Content-Length", [])
                if (
                    response.getheader("Transfer-Encoding") is not None
                    or len(lengths) != 1
                    or not lengths[0].isascii()
                    or not lengths[0].isdecimal()
                ):
                    raise DaemonOperationProtocolError("daemon response has invalid HTTP framing")
                declared_length = int(lengths[0])
                if declared_length > MAX_OPERATION_RESULT_BYTES:
                    raise DaemonOperationProtocolError("daemon response exceeds the bounded result size")
                response_body = response.read(MAX_OPERATION_RESULT_BYTES + 1)
                if len(response_body) != declared_length:
                    raise DaemonOperationProtocolError("daemon response body is incomplete")
                try:
                    decoded = json.loads(response_body.decode())
                except (UnicodeDecodeError, ValueError):
                    decoded = None
                self.last_elapsed_ms = round((perf_counter() - started_at) * 1000)
                self.last_status = response.status
                return response.status, decoded if isinstance(decoded, dict) else None
        except DaemonSocketOwnershipError:
            # A hard refusal, never a transport hiccup: the bearer was withheld
            # and no fallback may paper over an impostor on the socket path.
            raise
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
            isinstance(response, dict)
            and response.get("protocol") == DAEMON_OPERATION_PROTOCOL
            and response.get("outcome") == "rejected"
            and response.get("pre_dispatch") is True
            and isinstance(response.get("error"), dict)
        ):
            # The bounded ingress refused before dispatch and said so on the
            # envelope. A marked refusal is a known refusal for every code,
            # not absence and never a possibly committed mutation; whitelisting
            # codes one at a time failed open into the most expensive outcome.
            raise DaemonOperationRejectedError(str(response["error"].get("code") or "rejected"))
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
        if response.get("outcome") not in DAEMON_OPERATION_OUTCOMES:
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
        snapshot = response["authority_snapshot"]
        degraded = response.get("degraded_components")
        if not isinstance(degraded, list) or any(not isinstance(item, str) or not item for item in degraded):
            raise DaemonOperationProtocolError("daemon returned invalid degradation evidence")
        spec = daemon_operation_spec(request.operation)
        assert spec is not None
        if any(
            (
                snapshot.get("archive_identity") != response["archive"].get("archive_identity"),
                snapshot.get("generation") != response["generation"].get("id"),
                snapshot.get("schema_versions") != response["schema_versions"],
                response["archive"].get("tier_schema_versions") != response["schema_versions"],
                response["generation"].get("tier_schema_versions") != response["schema_versions"],
                response["archive"].get("index_schema_version") != response["schema_versions"].get("index"),
                snapshot.get("served_by") != response["served_by"].get("identity"),
                snapshot.get("served_by") != response["authority"].get("mode"),
                snapshot.get("elapsed_ms") != timing["elapsed_ms"],
                snapshot.get("queue_ms") != timing["queue_ms"],
                snapshot.get("degraded_components") != degraded,
                response["readiness"].get("degraded_components") != degraded,
                response["authority"].get("class") != spec.authority.value,
                response["authority"].get("fallback") != spec.fallback.value,
                response["authority"].get("writes") != "daemon-owned",
                response["archive"].get("daemon_version") != response["served_by"].get("daemon_version"),
                response["readiness"].get("ready") is not (not degraded),
                response["readiness"].get("state") != ("degraded" if degraded else "ready"),
            )
        ):
            raise DaemonOperationProtocolError("daemon returned incoherent operation authority")
        if snapshot.get("served_by") != "daemon":
            raise DaemonOperationProtocolError("machine response did not come from resident authority")
        if any(
            not isinstance(snapshot.get(key), str) or not snapshot[key] for key in ("archive_identity", "generation")
        ):
            raise DaemonOperationProtocolError("daemon omitted archive or generation identity")
        # A typed refusal may explain a stale precondition using current
        # authority. Successful execution must actually satisfy that binding.
        if response["outcome"] not in {"rejected", "failed", "cancelled", "timed-out"}:
            if request.expected_archive_identity not in (None, snapshot["archive_identity"]):
                raise DaemonOperationProtocolError("daemon served a stale archive identity")
            if response.get("accepted_reference") is None and request.expected_generation_id not in (
                None,
                snapshot["generation"],
            ):
                raise DaemonOperationProtocolError("daemon served a stale generation")
        reference = response.get("accepted_reference")
        if reference is not None and (
            not isinstance(reference, dict)
            or reference.get("request_id") != request.request_id
            or reference.get("archive_identity") != response["archive"].get("archive_identity")
            or reference.get("operation_name") != request.operation
            or reference.get("fingerprint") != request.fingerprint
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
        # One receipt read is always owed.  The submit above can consume the
        # whole completion budget on its own -- its own socket timeout is the
        # operation deadline plus a second -- and a plain ``while`` then
        # reports indeterminate without ever asking the durable lifecycle that
        # already holds the receipt.  "We never looked" is not an honest
        # indeterminate for a write the daemon durably accepted, and the
        # recovery it forces on the operator is the read skipped here.
        consulted = False
        while not consulted or perf_counter() < deadline:
            consulted = True
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
    "DaemonSocketOwnershipError",
]
