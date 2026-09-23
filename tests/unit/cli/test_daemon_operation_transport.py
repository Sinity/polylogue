"""CLI reads travel as a declared operation and its parameters, not as a URL.

Both root-query read routes -- the session page and the query-unit page --
lower to a declared operation (Seam A, :mod:`polylogue.cli.lowering`) and are
dispatched by :func:`polylogue.cli.operation_kernel.dispatch`.  The parameters
ride on the operation envelope, so there is no query string to build and none
to parse back.

Anti-vacuity: :func:`test_unit_route_sends_parameters_as_values` asserts the
parameter mapping the daemon client receives.  Reintroducing a query-string hop
makes ``params`` a URL fragment or a re-parsed ``dict[str, list[str]]`` rather
than the values the caller passed, and the assertion fails.
:func:`test_a_disabled_daemon_never_opens_a_client` goes red if ``dispatch``
constructs its transport before deciding that the daemon is disabled.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from polylogue.cli.operation_kernel import OperationRequest, dispatch


@pytest.fixture
def _config(tmp_path: Any) -> Any:
    config = MagicMock()
    config.archive_root = tmp_path
    config.db_path = tmp_path / "index.db"
    config.api_auth_token = None
    config.api_allow_no_auth = True
    return config


def _captured_operation(
    monkeypatch: pytest.MonkeyPatch, config: Any, request: OperationRequest
) -> tuple[str, dict[str, object]]:
    """Dispatch ``request`` and return the (operation, payload) the client saw."""
    seen: dict[str, Any] = {}

    class _Client:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def operation(self, operation: str, payload: dict[str, object], **_kwargs: object) -> dict[str, object]:
            seen["operation"] = operation
            seen["payload"] = payload
            return {"operation": operation, "outcome": "completed", "result": {"items": [], "total": 0}}

    monkeypatch.setattr("polylogue.daemon_client.DaemonClient", _Client)
    dispatch(config, request)
    return seen["operation"], seen["payload"]


def test_session_route_lowers_to_the_cli_query_operation(monkeypatch: pytest.MonkeyPatch, _config: Any) -> None:
    """The session page names its operation; the params are the ones lowered."""
    from polylogue.cli.lowering import lower_cli_query
    from polylogue.cli.root_request import RootModeRequest

    request = lower_cli_query(
        RootModeRequest(params={"repo": "polylogue"}, query_terms=()),
        limit=50,
        offset=0,
    )
    operation, payload = _captured_operation(monkeypatch, _config, request)

    assert operation == "cli.query"
    assert payload["params"] == {"repo": "polylogue", "query": [], "limit": 50, "offset": 0}


def test_unit_route_sends_parameters_as_values(monkeypatch: pytest.MonkeyPatch, _config: Any) -> None:
    """Query-unit parameters reach the operation unencoded, tuples included."""
    params: dict[str, object] = {
        "expression": "messages where text:timeout",
        "limit": 10,
        "offset": 0,
        "origin": ("codex-session", "claude-code-session"),
    }
    operation, payload = _captured_operation(monkeypatch, _config, OperationRequest("query.units", {"params": params}))

    assert operation == "query.units"
    assert payload["params"] == params


def test_a_disabled_daemon_never_opens_a_client(monkeypatch: pytest.MonkeyPatch, _config: Any) -> None:
    """``--no-daemon`` is a typed refusal and never constructs transport."""
    from polylogue.cli.operation_kernel import OperationUnavailableError

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a disabled daemon must not construct a client")

    monkeypatch.setattr("polylogue.daemon_client.DaemonClient", _explode)

    with pytest.raises(OperationUnavailableError, match="polylogued run"):
        dispatch(_config, OperationRequest("cli.query", {"params": {}}), daemon_disabled=True)
