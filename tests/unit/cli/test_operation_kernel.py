"""Laws for the import-light CLI operation seam."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
from polylogue.operations.daemon_errors import (
    DaemonMutationIndeterminateError,
    DaemonOperationProtocolError,
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


@pytest.mark.parametrize(
    "operation",
    [
        "mutation.session.excision",
        "mutation.session.lifecycle-request",
        "mutation.identity-reset",
        "mutation.raw-authority-blocker.resolve",
        "maintenance.reset",
        "maintenance.blob-gc.recover",
        "mutation.session.tag",
    ],
)
def test_mutation_operations_cannot_use_direct_fallback(operation: str) -> None:
    with pytest.raises(OperationUnavailableError) as exc_info:
        OperationKernel(lambda _request: None).execute(OperationRequest(operation, {}))
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


def _raising(exc: Exception) -> OperationKernel:
    def call(_request: OperationRequest) -> dict[str, object]:
        raise exc

    return OperationKernel(call)


def test_a_raised_indeterminate_mutation_is_typed_by_class_not_by_name() -> None:
    """A transport exception is classified by its type, never by its spelling.

    Anti-vacuity: the kernel used to compare ``type(exc).__name__`` against the
    literal ``"DaemonMutationIndeterminateError"``.  ``_RenamedError`` below is that
    exact class under a different name, so a name-matching kernel demotes it to
    a retryable ``daemon_transport_error`` and this test goes red; only an
    ``isinstance`` branch keeps an accepted-but-unreceipted mutation typed.
    """

    class _RenamedError(DaemonMutationIndeterminateError):
        pass

    with pytest.raises(OperationIndeterminateError, match="/api/operation"):
        _raising(_RenamedError(method="POST", path="/api/operation")).execute(
            OperationRequest("mutation.session.tag", {})
        )


def test_a_raised_size_protocol_error_is_reported_as_an_oversized_result() -> None:
    """The size refusal survives the move off name matching.

    Anti-vacuity: dropping the ``DaemonOperationProtocolError`` branch leaves
    the generic ``daemon_transport_error`` code, which this assertion rejects.
    """

    with pytest.raises(OperationFailedError) as exc_info:
        _raising(DaemonOperationProtocolError("result exceeds the declared size limit")).execute(
            OperationRequest("cli.query", {})
        )
    assert exc_info.value.code == "result_too_large"


@pytest.mark.parametrize(
    ("envelope", "code", "detail"),
    [
        (
            {
                "outcome": "timed-out",
                "error": {"code": "QueryTimeoutError", "detail": "archive read exceeded deadline (call call-1)"},
            },
            "QueryTimeoutError",
            "archive read exceeded deadline (call call-1)",
        ),
        (
            {
                "outcome": "cancelled",
                "error": {"code": "QueryCancelledError", "detail": "archive read cancelled (call call-1)"},
            },
            "QueryCancelledError",
            "archive read cancelled (call call-1)",
        ),
    ],
)
def test_deadline_and_cancellation_aborts_keep_their_code_and_call_id(
    envelope: dict[str, object], code: str, detail: str
) -> None:
    """A deadline or cancellation abort reaches the surface typed and with its call id.

    The executor raises ``QueryTimeoutError``/``QueryCancelledError`` and the
    daemon renders each as a ``timed-out``/``cancelled`` envelope whose error
    body carries the code and the call id. The kernel must hand both on: a
    surface cannot name the deadline, the call id or a remedy it never
    received, and the raw sqlite ``interrupted`` text is not one of them.

    Anti-vacuity: the cancelled branch previously preferred the envelope's
    (absent) ``result`` body over its ``error`` body, which dropped the code and
    the call id and left only "<operation> was cancelled" -- that turns both
    assertions red.
    """
    with pytest.raises((OperationFailedError, OperationCancelledError)) as raised:
        OperationKernel(lambda _request: envelope).execute(OperationRequest("cli.query", {}))
    abort = raised.value
    assert isinstance(abort, OperationFailedError | OperationCancelledError)
    assert abort.code == code
    assert str(abort.detail) == detail
    assert "interrupted" not in str(abort.detail)


def test_envelope_indeterminate_outcome_keeps_its_recovery_request_id() -> None:
    """An unresolved write reaches the surface with the id that can settle it.

    ``operation_to_completion`` returns ``outcome="indeterminate"`` when its
    receipt wait runs out of deadline. The only way a caller can then decide
    whether the write landed is to re-ask the daemon about *that request*, so
    the kernel must publish the id as state, not bury it in prose.

    Anti-vacuity: interpolating the id into the message and leaving
    ``request_id`` unset -- the shape this branch shipped with -- fails the
    attribute assertion below while still passing a ``match=`` on the text.
    """
    with pytest.raises(OperationIndeterminateError) as raised:
        OperationKernel(
            lambda _request: {
                "outcome": "indeterminate",
                "request_id": "unsettled-write",
                "result": {"sequence": 3},
            }
        ).execute(OperationRequest("mutation.session.tag", {}))
    assert raised.value.request_id == "unsettled-write"
    assert "unsettled-write" in str(raised.value)


def test_a_raised_indeterminate_mutation_keeps_its_recovery_request_id() -> None:
    """The transport's call id survives the hop into the kernel's typed error.

    Anti-vacuity: the branch used to raise ``OperationIndeterminateError(str(exc))``.
    ``DaemonMutationIndeterminateError.__str__`` names only the method and path,
    so flattening it to text loses the id outright and both assertions go red.
    """
    with pytest.raises(OperationIndeterminateError) as raised:
        _raising(
            DaemonMutationIndeterminateError(method="POST", path="/api/operation", request_id="in-flight-write")
        ).execute(OperationRequest("mutation.session.tag", {}))
    assert raised.value.request_id == "in-flight-write"
    assert "in-flight-write" in str(raised.value)


def test_an_empty_transport_request_id_is_not_reported_as_recovery_authority() -> None:
    """``""`` is absence, not a key. Reporting it would send a caller to no row.

    ``DaemonClient._request_json_response`` builds the id with
    ``str((body or {}).get("request_id", ""))``, so a body without one yields an
    empty string rather than ``None``.

    Anti-vacuity: assigning ``exc.request_id`` straight through makes
    ``request_id`` the empty string and puts "(request )" in the message.
    """
    with pytest.raises(OperationIndeterminateError) as raised:
        _raising(DaemonMutationIndeterminateError(method="POST", path="/api/operation", request_id="")).execute(
            OperationRequest("mutation.session.tag", {})
        )
    assert raised.value.request_id is None
    assert "(request" not in str(raised.value)


def test_mutation_and_read_address_the_same_resolved_file_set(tmp_path: Path) -> None:
    """A `--db` split-root pin decides where a declared mutation is sent.

    `configured_read_operation` resolves the file set through
    `operation_archive_root`, so selection and confirmation run against the
    pinned root. Addressing the mutation at `config.archive_root` instead
    delivered a confirmed `delete` to whichever daemon served the *configured*
    root -- a different archive.

    Anti-vacuity: restoring `daemon_socket_path(config.archive_root)` makes the
    recorded socket root the configured root while the read route still
    resolves the pinned one, so both assertions below fail.
    """
    from polylogue.cli import operation_kernel
    from polylogue.operations.archive_root import operation_archive_root

    configured = tmp_path / "configured"
    pinned = tmp_path / "pinned"
    configured.mkdir()
    pinned.mkdir()
    config = SimpleNamespace(
        archive_root=configured,
        db_path=pinned / "index.db",
        api_auth_token=None,
        api_allow_no_auth=True,
    )
    assert operation_archive_root(config) == pinned

    sockets: list[Path] = []
    declared_roots: list[str] = []

    class _Client:
        def __init__(self, socket_path: Path, **_kwargs: object) -> None:
            sockets.append(Path(socket_path))

        def operation_to_completion(self, operation: str, payload: dict[str, object], *, archive_root: str) -> dict:
            declared_roots.append(archive_root)
            return {"result": {"status": "ok"}, "authority": {"mode": "daemon"}}

    with patch("polylogue.daemon_client.DaemonClient", _Client):
        operation_kernel.configured_mutation_operation(
            config, "mutation.session.delete.preview", {"session_ids": ["a"]}
        )

    from polylogue.daemon.socket_path import daemon_socket_path

    assert declared_roots == [str(pinned)]
    assert sockets == [Path(daemon_socket_path(pinned))]
    assert sockets != [Path(daemon_socket_path(configured))]


def test_configured_root_without_a_pin_still_addresses_itself(tmp_path: Path) -> None:
    """The opposite direction: no split-root pin keeps the configured root."""
    from polylogue.cli import operation_kernel

    configured = tmp_path / "configured"
    configured.mkdir()
    config = SimpleNamespace(
        archive_root=configured,
        db_path=configured / "index.db",
        api_auth_token=None,
        api_allow_no_auth=True,
    )

    declared_roots: list[str] = []

    class _Client:
        def __init__(self, socket_path: Path, **_kwargs: object) -> None:
            pass

        def operation_to_completion(self, operation: str, payload: dict[str, object], *, archive_root: str) -> dict:
            declared_roots.append(archive_root)
            return {"result": {"status": "ok"}, "authority": {"mode": "daemon"}}

    with patch("polylogue.daemon_client.DaemonClient", _Client):
        operation_kernel.configured_mutation_operation(
            config, "mutation.session.delete.preview", {"session_ids": ["a"]}
        )

    assert declared_roots == [str(configured)]
