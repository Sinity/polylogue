"""``operation.cancel`` on a queued, pre-acceptance mutation.

The scheduler completes a queued task's future synchronously when its
cancellation handle fires, and the runtime's ``settled`` callback removes the
exchange from ``_exchanges`` in the same call. A cancel request that just fired
that handle therefore finds neither a live exchange nor a durable machine
record on its next look, and answering ``operation_reference_unknown`` would
report "no such operation" for the operation it just cancelled.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from polylogue.operations.daemon_protocol import DAEMON_OPERATION_SPECS, DaemonOperationRequest
from polylogue.operations.mutation_transaction import MutationPrincipal
from tests.infra.daemon_operations import running_daemon_operations


def _principal() -> MutationPrincipal:
    return MutationPrincipal(
        actor_ref=f"daemon:unix:uid:{os.getuid()}",
        capabilities=frozenset(spec.capability for spec in DAEMON_OPERATION_SPECS),
        surface="cli",
        role_label="daemon-unix-peer",
    )


def test_cancel_of_a_queued_mutation_reports_cancelled(tmp_path: Path) -> None:
    """The cancel request must report the cancellation it caused.

    Anti-vacuity: the target is genuinely queued behind a saturated kernel --
    its exchange exists and its future is not done when the cancel is issued --
    so the scheduler's synchronous pre-start completion is the path under test.
    A target that had already started would take the durable-record branch and
    prove nothing about it.
    """
    entered = threading.Semaphore(0)
    release = threading.Event()

    def block_worker() -> None:
        entered.release()
        assert release.wait(timeout=60)

    target_id = "queued-cancel-target"
    principal = _principal()

    with running_daemon_operations(tmp_path / "archive") as stack:
        blockers = [stack.execution_kernel.submit(block_worker) for _ in range(2)]
        assert all(entered.acquire(timeout=5) for _ in blockers)

        target = DaemonOperationRequest(
            "mutation.session.delete.preview",
            {"session_ids": ["codex:absent"]},
            request_id=target_id,
            archive_root=str(stack.archive_root),
        )
        target_envelopes: list[dict[str, object]] = []

        def call_target() -> None:
            target_envelopes.append(stack.runtime.call(target, principal))

        caller = threading.Thread(target=call_target, name="queued-cancel-target", daemon=True)
        caller.start()
        try:
            with stack.runtime._condition:
                assert stack.runtime._condition.wait_for(lambda: target_id in stack.runtime._exchanges, timeout=10)
                exchange = stack.runtime._exchanges[target_id]
                assert stack.runtime._condition.wait_for(lambda: exchange.future is not None, timeout=10)
            assert exchange.future is not None
            assert not exchange.future.done(), "the target must still be queued or this test is vacuous"
            assert not exchange.acceptance_started

            cancel = DaemonOperationRequest(
                "operation.cancel",
                {"request_id": target_id},
                request_id="queued-cancel-request",
                archive_root=str(stack.archive_root),
            )
            cancel_envelope = stack.runtime.call(cancel, principal)
            caller.join(timeout=10)
            assert not caller.is_alive()
        finally:
            release.set()
            for blocker in blockers:
                blocker.future.result(timeout=20)

    assert target_envelopes and target_envelopes[0]["outcome"] == "cancelled"
    assert cancel_envelope["outcome"] != "rejected", cancel_envelope.get("error")
    result = cancel_envelope.get("result")
    assert isinstance(result, dict)
    assert result.get("outcome") == "cancelled"


def test_cancel_of_an_unknown_reference_is_refused(tmp_path: Path) -> None:
    """Opposite direction: an id this daemon never saw is still unknown.

    Anti-vacuity for the fix above: answering ``cancelled`` unconditionally
    would satisfy the queued-cancel test and silently turn every unknown
    reference into a claimed cancellation.
    """
    with running_daemon_operations(tmp_path / "archive") as stack:
        cancel = DaemonOperationRequest(
            "operation.cancel",
            {"request_id": "never-existed"},
            request_id="unknown-cancel-request",
            archive_root=str(stack.archive_root),
        )
        envelope = stack.runtime.call(cancel, _principal())

    assert envelope["outcome"] == "rejected"
    error = envelope.get("error")
    assert isinstance(error, dict)
    assert error["code"] == "operation_reference_unknown"
