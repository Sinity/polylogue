"""The recovery id of an unresolved write survives the real UDS mutation route."""

from __future__ import annotations

import queue
from pathlib import Path

import pytest

from polylogue.cli.operation_kernel import (
    OperationIndeterminateError,
    OperationKernel,
    OperationRequest,
)
from tests.infra.daemon_operations import running_daemon_operations


def test_an_unresolved_uds_mutation_reaches_the_cli_seam_with_its_recovery_id(tmp_path: Path) -> None:
    """A real mutation that loses its receipt still names the request to settle.

    This executes the production mutation route: a real archive, a real
    ``DaemonAPIUnixHTTPServer`` on a real AF_UNIX socket, and the real
    ``DaemonClient``. The handler dies mid-request, so the client sees a
    connected socket and no receipt -- exactly the state where re-issuing the
    write is unsafe and the operator is told to inspect daemon audit state.
    That instruction is only actionable with the request id, and the daemon's
    durable lifecycle (``operation.await``/``operation.cancel``) is keyed on it.

    Anti-vacuity: ``OperationKernel.execute`` previously raised
    ``OperationIndeterminateError(str(exc))``, and
    ``DaemonMutationIndeterminateError.__str__`` carries only the method and
    path. Restoring that line leaves this write unsettleable and turns both
    assertions red -- the attribute does not even exist.
    """

    sink: queue.SimpleQueue[str] = queue.SimpleQueue()

    class _DyingRuntime:
        """Stand-in for a runtime that fails after the request is on the wire."""

        def call(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            raise RuntimeError("runtime died mid-mutation")

    with running_daemon_operations(tmp_path / "archive", server_error_sink=sink) as stack:
        object.__setattr__(stack.server, "operation_runtime", _DyingRuntime())

        def submit(request: OperationRequest) -> dict[str, object] | None:
            return stack.client.operation(
                request.operation,
                dict(request.payload),
                archive_root=str(stack.archive_root),
                request_id="unsettled-uds-write",
            )

        with pytest.raises(OperationIndeterminateError) as raised:
            OperationKernel(submit).execute(
                OperationRequest(
                    "mutation.session.tag",
                    {"session_ids": ["codex-session:recovery"], "tags": ["recovery"]},
                )
            )

    assert raised.value.request_id == "unsettled-uds-write"
    assert "unsettled-uds-write" in str(raised.value)
    assert not sink.empty()
