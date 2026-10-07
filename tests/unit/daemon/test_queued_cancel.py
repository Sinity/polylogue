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
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.operations.daemon_protocol import (
    DAEMON_OPERATION_SPECS,
    DaemonOperationEnvelope,
    DaemonOperationRequest,
)
from polylogue.operations.mutation_transaction import MutationPrincipal
from tests.infra.daemon_operations import DaemonOperationStack, running_daemon_operations


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


def test_cancel_fences_a_live_redrive_despite_a_resent_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resent request's own live exchange must not skip the durable fence.

    Anti-vacuity (#5717): a client resending the original ingest request
    while its startup re-drive still runs creates its own ``_exchanges``
    entry. Gating the durable fence and the re-drive cancellation flag on
    ``exchange is None`` alone treats that resend as pre-acceptance and
    skips both, so a live re-drive of this exact target observes neither
    signal and can still materialize and finalize after the client believes
    it cancelled. The fix keys the fence on ``exchange.acceptance_started``
    instead: this test proves the fence step is *entered* (the bridge is
    asked to run it) for an accepted, resent exchange, and stays skipped
    for a genuinely pre-acceptance one -- the opposite-direction pin.
    """
    from concurrent.futures import Future

    from polylogue.daemon.operation_runtime import _Exchange
    from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge
    from polylogue.operations.operation_context_types import OperationContext

    fenced_actors: list[str] = []
    original_run_sync = DaemonWriteThreadBridge.run_sync_with_timeout

    def recording_run_sync(
        self: DaemonWriteThreadBridge,
        actor: str,
        timeout: float,
        function: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        if actor == "operation.cancel":
            fenced_actors.append(actor)
            return None
        return original_run_sync(self, actor, timeout, function, *args, **kwargs)

    monkeypatch.setattr(DaemonWriteThreadBridge, "run_sync_with_timeout", recording_run_sync)
    from polylogue.operations.audit import AuditRepository

    original_lookup = AuditRepository.machine_request_for_principal

    def durable_lookup(self: AuditRepository, identity: str, request_id: str, actor: str) -> dict[str, object] | None:
        # Only the accepted request has a durable record; the resent exchange
        # itself never crosses its own acceptance path.
        if request_id == "accepted-resend":
            return {"artifact_kind": "source-generation", "request_id": request_id}
        if request_id == "pre-acceptance-resend":
            return None
        return original_lookup(self, identity, request_id, actor)

    monkeypatch.setattr(AuditRepository, "machine_request_for_principal", durable_lookup)
    from polylogue.daemon import operation_runtime
    from polylogue.operations.machine_lifecycle import machine_request_state as original_state

    def durable_state(audit: AuditRepository, record: dict[str, object], **window: int) -> dict[str, object]:
        # The control read pages the request's parts (``parts_offset`` /
        # ``parts_limit``); the stand-in forwards that window unchanged.
        if record.get("request_id") == "accepted-resend":
            return {"outcome": "cancelled", "sequence": 1, "effect": "indeterminate"}
        return original_state(audit, record, **window)

    monkeypatch.setattr(operation_runtime, "machine_request_state", durable_state)

    def _cancel_with_resent_exchange(stack: DaemonOperationStack, *, acceptance_started: bool, request_id: str) -> None:
        principal = _principal()
        request = DaemonOperationRequest(
            "mutation.session.delete.preview",
            {"session_ids": ["codex:absent"]},
            request_id=request_id,
            archive_root=str(stack.archive_root),
        )
        context = OperationContext(archive_root=stack.archive_root, principal=principal, serving_identity="daemon")
        future: Future[DaemonOperationEnvelope] = Future()
        exchange = _Exchange(
            request=request,
            context=context,
            deadline=1e18,
            deadline_unix_ms=0,
            future=future,
            acceptance_started=acceptance_started,
        )
        with stack.runtime._condition:
            stack.runtime._exchanges[request_id] = exchange

        cancel = DaemonOperationRequest(
            "operation.cancel",
            {"request_id": request_id},
            request_id=f"{request_id}-cancel",
            archive_root=str(stack.archive_root),
        )
        try:
            stack.runtime.call(cancel, principal)
        finally:
            # The stand-in exchange has no worker to settle its future; leaving
            # it registered after a failed call would hang runtime shutdown
            # and hide that failure behind the test timeout.
            with stack.runtime._condition:
                stack.runtime._exchanges.pop(request_id, None)
            future.cancel()

    with running_daemon_operations(tmp_path / "archive") as stack:
        # Opposite-direction pin first: a genuinely pre-acceptance resend
        # must still skip the fence, exactly as before this fix.
        _cancel_with_resent_exchange(stack, acceptance_started=False, request_id="pre-acceptance-resend")
        assert fenced_actors == []

        # The defect's exact shape: a resend of an accepted request, whose
        # own exchange never started acceptance (Codex P1, #5717).
        _cancel_with_resent_exchange(stack, acceptance_started=False, request_id="accepted-resend")

    assert fenced_actors == ["operation.cancel"]
