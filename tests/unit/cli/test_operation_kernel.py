"""Laws for the import-light CLI operation seam."""

from __future__ import annotations

import pytest

from polylogue.cli.operation_kernel import (
    OperationEnvelopeError,
    OperationFailedError,
    OperationKernel,
    OperationRequest,
    OperationUnavailableError,
)


def test_daemon_envelope_is_validated_before_renderer_handoff() -> None:
    request = OperationRequest("cli.query", {})
    with pytest.raises(OperationEnvelopeError, match="omitted"):
        OperationKernel(lambda _request: {"operation": "cli.query"}).execute(request)


def test_failed_outcome_is_typed_and_never_directly_retried() -> None:
    called = False

    def direct(_request: OperationRequest) -> object:
        nonlocal called
        called = True
        return {"items": []}

    with pytest.raises(OperationFailedError, match="cancelled"):
        OperationKernel(lambda _request: {"outcome": "cancelled", "result": None}, direct).execute(
            OperationRequest("cli.query", {})
        )
    assert called is False


def test_daemon_and_direct_reads_share_the_result_contract() -> None:
    request = OperationRequest("cli.query", {"params": {"query": ("needle",)}})
    daemon = OperationKernel(lambda _request: {"result": {"items": [1]}, "authority": {"mode": "daemon"}})
    direct = OperationKernel(lambda _request: None, lambda _request: {"items": [1]})

    daemon_result = daemon.execute(request)
    direct_result = direct.execute(request)

    assert daemon_result.value == direct_result.value
    assert daemon_result.authority["mode"] == "daemon"
    assert direct_result.authority["mode"] == "direct"


def test_typed_daemon_error_does_not_fall_through_to_direct_execution() -> None:
    called = False

    def direct(_request: OperationRequest) -> object:
        nonlocal called
        called = True
        return {"unsafe": True}

    with pytest.raises(OperationFailedError, match="bad_query"):
        OperationKernel(
            lambda _request: {"error": {"code": "bad_query", "detail": "invalid"}},
            direct,
        ).execute(OperationRequest("cli.query", {}))
    assert called is False


def test_non_read_operation_cannot_use_direct_fallback() -> None:
    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_SPECS, DaemonAuthority, DaemonFallback

    original = tuple(DAEMON_OPERATION_SPECS)
    try:
        # The production registry currently contains reads only; this law
        # exercises the guard against a future mutating declaration.
        from polylogue.operations import daemon_protocol

        daemon_protocol.DAEMON_OPERATION_SPECS = original + (
            daemon_protocol.DaemonOperationSpec("test.write", DaemonAuthority.WRITE, DaemonFallback.DIRECT_READ),
        )
        with pytest.raises(OperationUnavailableError):
            OperationKernel(lambda _request: None, lambda _request: {"written": True}).execute(
                OperationRequest("test.write", {})
            )
    finally:
        daemon_protocol.DAEMON_OPERATION_SPECS = original
