"""CLI daemon reads travel as an operation and its parameters, not as a URL.

Both daemon-backed read routes -- the session page and the query-unit page --
lower to a declared operation on the UDS protocol. The parameters ride on the
operation envelope, so there is no query string to build and none to parse back:
the daemon coerces each value to the ``list[str]`` its handler reads.

Anti-vacuity: :func:`test_unit_route_sends_parameters_as_values` asserts the
parameter mapping the kernel receives. Reintroducing a query-string hop makes
``params`` a URL fragment or a re-parsed ``dict[str, list[str]]`` rather than
the values the caller passed, and the assertion fails.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from polylogue.cli import archive_query


@pytest.fixture
def _config(tmp_path: Any) -> Any:
    config = MagicMock()
    config.archive_root = tmp_path
    config.db_path = tmp_path / "index.db"
    config.api_auth_token = None
    config.api_allow_no_auth = True
    return config


def _captured_operation(monkeypatch: pytest.MonkeyPatch, config: Any, call: Any) -> tuple[str, dict[str, object]]:
    """Run ``call`` and return the (operation, payload) the daemon client saw."""
    seen: dict[str, Any] = {}

    class _Client:
        last_elapsed_ms = None

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def operation_with_direct_fallback(
            self, operation: str, payload: dict[str, object], **_kwargs: object
        ) -> dict[str, object]:
            seen["operation"] = operation
            seen["payload"] = payload
            return {"operation": operation, "outcome": "completed", "result": {"items": [], "total": 0}}

    monkeypatch.setattr("polylogue.daemon_client.DaemonClient", _Client)
    monkeypatch.setattr(archive_query, "_daemon_disabled", lambda **_kwargs: False)
    call(config)
    return seen["operation"], seen["payload"]


def test_session_route_lowers_to_the_cli_query_operation(monkeypatch: pytest.MonkeyPatch, _config: Any) -> None:
    """The session page names its operation; the params are the ones passed."""
    operation, payload = _captured_operation(
        monkeypatch,
        _config,
        lambda config: archive_query._fetch_daemon_sessions_payload(
            config, {"limit": 50, "offset": 0, "repo": "polylogue"}
        ),
    )

    assert operation == "cli.query"
    assert payload["params"] == {"limit": 50, "offset": 0, "repo": "polylogue"}


def test_unit_route_sends_parameters_as_values(monkeypatch: pytest.MonkeyPatch, _config: Any) -> None:
    """Query-unit parameters reach the operation unencoded, tuples included."""
    params: dict[str, object] = {
        "expression": "messages where text:timeout",
        "limit": 10,
        "offset": 0,
        "origin": ("codex-session", "claude-code-session"),
    }
    operation, payload = _captured_operation(
        monkeypatch,
        _config,
        lambda config: archive_query._fetch_daemon_payload(config, "query.units", params),
    )

    assert operation == "query.units"
    assert payload["params"] == params


def test_a_disabled_daemon_never_opens_a_client(monkeypatch: pytest.MonkeyPatch, _config: Any) -> None:
    """``--no-daemon`` executes the canonical read without constructing transport."""
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(_config.archive_root)

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a disabled daemon must not construct a client")

    monkeypatch.setattr("polylogue.daemon_client.DaemonClient", _explode)

    result = archive_query._fetch_daemon_payload(_config, "cli.query", {}, disabled=True)
    assert result is not None
    assert result["items"] == []
    assert result["total"] == 0
