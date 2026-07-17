"""First-party web credential lifecycle and HTTP composition tests."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.cookies import SimpleCookie
from unittest.mock import MagicMock

import pytest

from polylogue.daemon import web_shell, web_shell_realtime
from polylogue.daemon.web_auth import (
    WEB_CREDENTIAL_COOKIE,
    WebCredentialBootstrapPayload,
    WebCredentialDecision,
    WebCredentialRegistry,
    WebCredentialScope,
    credential_cookie,
    read_web_credential_cookie,
    same_origin_from_headers,
)
from tests.unit.daemon.test_daemon_http_security import (
    _capture_responses,
    _make_handler,
    _MockServer,
)


def _cookie_pair(set_cookie: str) -> tuple[str, str]:
    parsed = SimpleCookie()
    parsed.load(set_cookie)
    morsel = parsed[WEB_CREDENTIAL_COOKIE]
    return morsel.key, morsel.value


def _decision(
    registry: WebCredentialRegistry,
    token: str | None,
    *,
    scope: WebCredentialScope = "read",
    origin: str = "http://127.0.0.1:8766",
    host: str = "127.0.0.1:8766",
) -> WebCredentialDecision:
    return registry.validate(
        token,
        required_scope=scope,
        host_header=host,
        origin_header=origin,
        referer_header="",
        fetch_site="same-origin",
    )


def test_registry_tracks_only_digest_and_exposes_no_secret_in_payload_or_repr() -> None:
    registry = WebCredentialRegistry(ttl_s=30)
    issued = registry.issue("http://127.0.0.1:8766")

    assert issued.token not in repr(issued)
    public_payload = issued.public_payload().model_dump(mode="json")
    assert issued.token not in json.dumps(public_payload, sort_keys=True)
    assert public_payload["state"] == "ready"
    assert _decision(registry, issued.token).allowed is True


def test_registry_reports_missing_expired_revoked_wrong_origin_and_scope() -> None:
    now = [100.0]
    registry = WebCredentialRegistry(ttl_s=5, clock=lambda: now[0])
    issued = registry.issue("http://127.0.0.1:8766", scopes=("read",))

    assert _decision(registry, None).state == "web_credential_missing"
    assert _decision(registry, issued.token, scope="user_state").state == ("web_credential_insufficient_scope")
    assert _decision(registry, issued.token, origin="http://127.0.0.1:9999").state == ("web_credential_wrong_origin")
    now[0] = 106.0
    assert _decision(registry, issued.token).state == "web_credential_expired"

    replacement = registry.issue("http://127.0.0.1:8766", previous_token=issued.token)
    assert _decision(registry, issued.token).state == "web_credential_revoked"
    assert registry.revoke(replacement.token).state == "web_credential_revoked"
    assert _decision(registry, replacement.token).state == "web_credential_revoked"
    assert _decision(registry, "not-ascii-\N{LATIN SMALL LETTER E WITH ACUTE}").state == "web_credential_invalid"


def test_registry_bounds_rotation_records_and_preserves_recent_lifecycle_state() -> None:
    registry = WebCredentialRegistry(ttl_s=30, max_records=6, max_records_per_origin=4)
    origin = "http://127.0.0.1:8766"
    issued = registry.issue(origin)
    previous = issued

    for _ in range(12):
        previous = issued
        issued = registry.issue(origin, previous_token=previous.token)

    assert registry.retained_record_count == 4
    assert _decision(registry, previous.token).state == "web_credential_revoked"
    assert _decision(registry, issued.token).allowed is True


def test_bootstrap_rotates_http_only_cookie_and_authenticates_read_route() -> None:
    server = _MockServer()
    origin = "http://127.0.0.1:8766"
    bootstrap = _make_handler(
        "POST",
        "/api/web-auth/session",
        origin=origin,
        host="127.0.0.1:8766",
        web_client=True,
        server=server,
    )
    send_error, _ = _capture_responses(bootstrap)
    send_cookie_json = MagicMock()
    bootstrap._send_json_with_cookie = send_cookie_json  # type: ignore[method-assign]
    bootstrap.do_POST()

    send_error.assert_not_called()
    status, payload = send_cookie_json.call_args.args
    set_cookie = send_cookie_json.call_args.kwargs["set_cookie"]
    assert status == HTTPStatus.CREATED
    assert payload["credential"]["state"] == "ready"
    assert WebCredentialBootstrapPayload.model_validate(payload).credential.scopes == (
        "events",
        "read",
        "user_state",
    )
    cookie_name, token = _cookie_pair(set_cookie)
    assert cookie_name == WEB_CREDENTIAL_COOKIE
    assert token not in json.dumps(payload)
    assert "HttpOnly" in set_cookie
    assert "SameSite=Strict" in set_cookie
    assert "Path=/" in set_cookie

    reader = _make_handler(
        "GET",
        "/api/status",
        host="127.0.0.1:8766",
        cookie=f"{cookie_name}={token}",
        fetch_site="same-origin",
        web_client=True,
        server=server,
    )
    reader._handle_status = MagicMock()  # type: ignore[method-assign]
    reader.do_GET()
    reader._handle_status.assert_called_once_with({})


@pytest.mark.parametrize(
    ("path", "handler_name"),
    [
        ("/api/reset", "_handle_reset"),
        ("/api/ingest", "_handle_ingest"),
        ("/api/maintenance/plan", "_handle_maintenance_plan"),
        ("/api/maintenance/run", "_handle_maintenance_run"),
    ],
)
def test_web_credential_cannot_execute_archive_control_routes(path: str, handler_name: str) -> None:
    server = _MockServer()
    origin = "http://127.0.0.1:8766"
    issued = server.web_credentials.issue(origin)
    handler = _make_handler(
        "POST",
        path,
        origin=origin,
        host="127.0.0.1:8766",
        cookie=f"{WEB_CREDENTIAL_COOKIE}={issued.token}",
        fetch_site="same-origin",
        web_client=True,
        server=server,
    )
    route_handler = MagicMock()
    setattr(handler, handler_name, route_handler)
    send_error, _ = _capture_responses(handler)

    handler.do_POST()

    send_error.assert_called_once_with(HTTPStatus.UNAUTHORIZED, "unauthorized")
    route_handler.assert_not_called()


def test_bootstrap_rejects_wrong_origin_without_minting_cookie() -> None:
    handler = _make_handler(
        "POST",
        "/api/web-auth/session",
        origin="http://127.0.0.1:9999",
        host="127.0.0.1:8766",
        web_client=True,
    )
    send_error, send_json = _capture_responses(handler)
    handler.do_POST()

    send_error.assert_called_once()
    assert send_error.call_args.args[:2] == (HTTPStatus.FORBIDDEN, "web_credential_wrong_origin")
    assert send_error.call_args.kwargs["extra_headers"]["X-Polylogue-Web-Credential-State"] == (
        "web_credential_wrong_origin"
    )
    send_json.assert_not_called()


def test_bootstrap_host_rejection_uses_typed_wrong_origin_contract() -> None:
    handler = _make_handler(
        "POST",
        "/api/web-auth/session",
        origin="https://evil.example.com",
        host="evil.example.com",
        web_client=True,
    )
    send_error, _ = _capture_responses(handler)

    handler.do_POST()

    assert send_error.call_args.args[:2] == (HTTPStatus.FORBIDDEN, "web_credential_wrong_origin")
    assert send_error.call_args.kwargs["extra_headers"]["X-Polylogue-Web-Credential-State"] == (
        "web_credential_wrong_origin"
    )


def test_non_loopback_bootstrap_rejection_uses_typed_wrong_origin_contract() -> None:
    server = _MockServer()
    server.api_host = "archive.internal"
    handler = _make_handler(
        "POST",
        "/api/web-auth/session",
        origin="http://archive.internal:8766",
        host="archive.internal:8766",
        web_client=True,
        server=server,
    )
    send_error, _ = _capture_responses(handler)

    handler.do_POST()

    assert send_error.call_args.args[:2] == (HTTPStatus.FORBIDDEN, "web_credential_wrong_origin")
    assert send_error.call_args.kwargs["extra_headers"]["X-Polylogue-Web-Credential-State"] == (
        "web_credential_wrong_origin"
    )


def test_non_ascii_cookie_is_total_across_bootstrap_read_and_revoke() -> None:
    origin = "http://127.0.0.1:8766"
    malformed_cookie = f'{WEB_CREDENTIAL_COOKIE}="\N{LATIN SMALL LETTER E WITH ACUTE}"'
    server = _MockServer()

    bootstrap = _make_handler(
        "POST",
        "/api/web-auth/session",
        origin=origin,
        host="127.0.0.1:8766",
        cookie=malformed_cookie,
        web_client=True,
        server=server,
    )
    bootstrap_error, _ = _capture_responses(bootstrap)
    bootstrap_cookie_json = MagicMock()
    bootstrap._send_json_with_cookie = bootstrap_cookie_json  # type: ignore[method-assign]
    bootstrap.do_POST()
    bootstrap_error.assert_not_called()
    assert bootstrap_cookie_json.call_args.args[0] == HTTPStatus.CREATED

    reader = _make_handler(
        "GET",
        "/api/sessions",
        host="127.0.0.1:8766",
        cookie=malformed_cookie,
        fetch_site="same-origin",
        web_client=True,
        server=server,
    )
    reader_error, _ = _capture_responses(reader)
    reader.do_GET()
    assert reader_error.call_args.args[:2] == (HTTPStatus.UNAUTHORIZED, "web_credential_invalid")

    revoke = _make_handler(
        "DELETE",
        "/api/web-auth/session",
        origin=origin,
        host="127.0.0.1:8766",
        cookie=malformed_cookie,
        fetch_site="same-origin",
        web_client=True,
        server=server,
    )
    revoke_error, _ = _capture_responses(revoke)
    revoke.do_DELETE()
    assert revoke_error.call_args.args[:2] == (HTTPStatus.UNAUTHORIZED, "web_credential_invalid")


def test_browser_missing_credential_has_explicit_recoverable_state() -> None:
    handler = _make_handler(
        "GET",
        "/api/sessions",
        host="127.0.0.1:8766",
        fetch_site="same-origin",
        web_client=True,
    )
    send_error, _ = _capture_responses(handler)
    handler.do_GET()

    assert send_error.call_args.args[:2] == (HTTPStatus.UNAUTHORIZED, "web_credential_missing")
    assert send_error.call_args.kwargs["extra_headers"]["X-Polylogue-Web-Credential-State"] == (
        "web_credential_missing"
    )


def test_cookie_and_origin_helpers_reject_authority_confusion() -> None:
    assert same_origin_from_headers("http://127.0.0.1:8766", "127.0.0.1:8766") == ("http://127.0.0.1:8766")
    assert same_origin_from_headers("http://127.0.0.1:9999", "127.0.0.1:8766") is None
    assert same_origin_from_headers("http://localhost:8766", "127.0.0.1:8766") is None

    header = credential_cookie("opaque-value", ttl_s=30)
    assert read_web_credential_cookie(header) == "opaque-value"
    assert read_web_credential_cookie("not a valid cookie; =") is None


def test_current_shell_consumes_shared_cookie_contract_without_url_secret() -> None:
    html = web_shell.WEB_SHELL_HTML
    realtime = web_shell_realtime.REALTIME_JS

    assert "fetch('/api/web-auth/session'" in html
    assert "credentials: 'same-origin'" in html
    assert "X-Polylogue-Web-Client" in html
    assert "WEB_AUTH_FAILURES" in html
    assert "new EventSource(url, {withCredentials: true})" in realtime
    assert "bootstrapWebCredential()" in realtime
    assert "access_token" not in realtime
    assert "access_token" not in html
