"""Laws for the import-light CLI operation seam."""

from __future__ import annotations

import pytest

from polylogue.cli.operation_kernel import (
    OperationCancelledError,
    OperationEnvelopeError,
    OperationFailedError,
    OperationIndeterminateError,
    OperationKernel,
    OperationRequest,
    OperationUnavailableError,
)


def test_daemon_envelope_is_validated_before_renderer_handoff() -> None:
    request = OperationRequest("cli.query", {})
    with pytest.raises(OperationEnvelopeError, match="omitted"):
        OperationKernel(lambda _request: {"operation": "cli.query"}).execute(request)


def test_failed_outcome_is_typed_and_never_directly_retried() -> None:
    with pytest.raises(OperationCancelledError):
        OperationKernel(lambda _request: {"outcome": "cancelled", "result": None}).execute(
            OperationRequest("cli.query", {})
        )


def test_daemon_and_direct_reads_share_the_result_contract() -> None:
    request = OperationRequest("cli.query", {"params": {"query": ("needle",)}})
    daemon = OperationKernel(lambda _request: {"result": {"items": [1]}, "authority": {"mode": "daemon"}})
    direct = OperationKernel(lambda _request: {"result": {"items": [1]}, "authority": {"mode": "direct"}})

    daemon_result = daemon.execute(request)
    direct_result = direct.execute(request)

    assert daemon_result.value == direct_result.value
    assert daemon_result.authority["mode"] == "daemon"
    assert direct_result.authority["mode"] == "direct"


def test_typed_daemon_error_does_not_fall_through_to_direct_execution() -> None:
    with pytest.raises(OperationFailedError, match="bad_query"):
        OperationKernel(
            lambda _request: {"error": {"code": "bad_query", "detail": "invalid"}},
        ).execute(OperationRequest("cli.query", {}))


def test_non_read_operation_cannot_use_direct_fallback() -> None:
    with pytest.raises(OperationUnavailableError) as exc_info:
        OperationKernel(lambda _request: None).execute(OperationRequest("mutation.session.tag", {}))
    assert exc_info.value.code == "daemon_required"


@pytest.mark.parametrize(
    ("envelope", "code"),
    [
        ({"outcome": "timeout", "result": None}, "timeout"),
        ({"generation": {"state": "stale"}, "result": {}}, "stale_generation"),
    ],
)
def test_terminal_and_stale_daemon_states_are_typed(envelope: dict[str, object], code: str) -> None:
    with pytest.raises(OperationFailedError) as exc_info:
        OperationKernel(lambda _request: envelope).execute(OperationRequest("cli.query", {}))
    assert exc_info.value.code == code


def test_oversized_result_is_rejected_before_rendering() -> None:
    from polylogue.operations.daemon_protocol import MAX_OPERATION_RESULT_BYTES

    with pytest.raises(OperationFailedError) as exc_info:
        OperationKernel(lambda _request: {"result": "x" * (MAX_OPERATION_RESULT_BYTES + 1)}).execute(
            OperationRequest("cli.query", {})
        )
    assert exc_info.value.code == "result_too_large"


def test_timeout_is_not_evidence_of_daemon_absence() -> None:
    with pytest.raises(OperationFailedError, match="daemon_transport_error"):
        OperationKernel(
            lambda _request: (_ for _ in ()).throw(TimeoutError("deadline")),
        ).execute(OperationRequest("cli.query", {}))


def test_error_detail_cannot_demote_indeterminate_effects_to_retryable_failure() -> None:
    with pytest.raises(OperationIndeterminateError, match="accepted-request"):
        OperationKernel(
            lambda _request: {
                "outcome": "indeterminate",
                "request_id": "accepted-request",
                "error": {"code": "after_commit_failure", "detail": "receipt reconciliation required"},
            }
        ).execute(OperationRequest("mutation.session.tag", {}))
