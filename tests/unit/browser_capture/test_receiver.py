from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.client import HTTPConnection, HTTPResponse
from pathlib import Path
from threading import Thread
from typing import cast

import pytest
from click.testing import CliRunner

from polylogue.browser_capture.models import (
    BrowserCaptureAcceptedPayload,
    BrowserCaptureArchiveStatePayload,
    BrowserCaptureEnvelope,
    BrowserCaptureErrorPayload,
    BrowserCaptureReceiverStatusPayload,
)
from polylogue.browser_capture.receiver import (
    capture_artifact_path,
    capture_artifact_ref,
    existing_capture_state,
    write_capture_envelope,
)
from polylogue.browser_capture.route_contracts import (
    BROWSER_CAPTURE_ROUTE_CONTRACTS,
    browser_capture_route_contract_for,
)
from polylogue.browser_capture.server import make_server
from polylogue.daemon.cli import main as daemon_cli

_EXTENSION_ORIGIN = "chrome-extension://polylogue-test"
_CHATGPT_ORIGIN = "https://chatgpt.com"


def _payload(provider: str = "chatgpt", session_id: str = "conv-123") -> dict[str, object]:
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {
            "source_url": "https://chatgpt.com/c/conv-123",
            "page_title": "ChatGPT - Work plan",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-dom-v1",
        },
        "session": {
            "provider": provider,
            "provider_session_id": session_id,
            "title": "Work plan",
            "turns": [{"provider_turn_id": "u1", "role": "user", "text": "Draft"}],
        },
    }


@contextmanager
def _running_receiver(
    tmp_path: Path,
    *,
    auth_token: str | None = None,
    extra_origins: tuple[str, ...] = (),
) -> Iterator[tuple[str, int]]:
    server = make_server(
        "127.0.0.1",
        0,
        spool_path=tmp_path,
        auth_token=auth_token,
        extra_origins=extra_origins,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(
    host: str, port: int, method: str, path: str, *, body: object | str | None = None, origin: str
) -> HTTPResponse:
    conn = HTTPConnection(host, port)
    headers = {"Origin": origin}
    payload: str | None
    if isinstance(body, str):
        payload = body
        headers["Content-Type"] = "application/json"
    elif body is not None:
        payload = json.dumps(body)
        headers["Content-Type"] = "application/json"
    else:
        payload = None
    conn.request(method, path, body=payload, headers=headers)
    return conn.getresponse()


def test_capture_artifact_path_is_deterministic(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload(session_id="c/with spaces"))

    first = capture_artifact_path(envelope, tmp_path)
    second = capture_artifact_path(envelope, tmp_path)
    artifact_ref = capture_artifact_ref(envelope, tmp_path)

    assert first == second
    assert first.parent.name == "chatgpt"
    assert "with-spaces" in first.name
    assert artifact_ref == first.relative_to(tmp_path).as_posix()
    assert Path(artifact_ref).is_absolute() is False


def test_write_capture_envelope_replaces_same_artifact(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload())

    first = write_capture_envelope(envelope, spool_path=tmp_path)
    second = write_capture_envelope(envelope, spool_path=tmp_path)

    assert first.path == second.path
    assert first.replaced is False
    assert second.replaced is True
    assert json.loads(first.path.read_text(encoding="utf-8"))["session"]["provider_session_id"] == "conv-123"


def test_existing_capture_state_reports_written_artifact(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload())
    write_capture_envelope(envelope, spool_path=tmp_path)

    state = existing_capture_state("chatgpt", "conv-123", spool_path=tmp_path)
    typed = BrowserCaptureArchiveStatePayload.model_validate(state)

    assert typed.captured is True
    assert typed.provider == "chatgpt"
    assert typed.artifact_ref == capture_artifact_ref(envelope, tmp_path)
    assert Path(typed.artifact_ref).is_absolute() is False


def test_browser_capture_status_daemon_cli_json(cli_workspace: dict[str, Path]) -> None:
    runner = CliRunner()

    result = runner.invoke(daemon_cli, ["browser-capture", "status", "--format", "json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    typed = BrowserCaptureReceiverStatusPayload.model_validate(payload)
    assert payload["ok"] is True
    assert typed.receiver == "polylogue-browser-capture"
    assert typed.spool_ready is True
    assert typed.auth_required is False
    assert typed.allowed_origins == ["chrome-extension://*"]
    assert "spool_path" not in payload


def test_browser_capture_route_contracts_cover_receiver_boundary() -> None:
    concrete_routes = {(contract.method, contract.pattern) for contract in BROWSER_CAPTURE_ROUTE_CONTRACTS}

    assert ("GET", "/v1/status") in concrete_routes
    assert ("GET", "/v1/archive-state") in concrete_routes
    assert ("POST", "/v1/browser-captures") in concrete_routes
    assert browser_capture_route_contract_for("POST", "/v1/browser-captures") is not None


def test_receiver_rejects_web_origins_by_default(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        response = _request(host, port, "GET", "/v1/status", origin=_CHATGPT_ORIGIN)
        error = BrowserCaptureErrorPayload.model_validate(json.loads(response.read()))

    assert response.status == HTTPStatus.FORBIDDEN
    assert response.getheader("X-Request-ID")
    assert error.error == "origin_not_allowed"


def test_receiver_rejects_extra_web_origin_without_token(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="web origins require"):
        make_server("127.0.0.1", 0, spool_path=tmp_path, extra_origins=(_CHATGPT_ORIGIN,))


def test_receiver_accepts_extension_capture_and_reports_typed_dto(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "POST",
            "/v1/browser-captures",
            body=_payload(),
            origin=_EXTENSION_ORIGIN,
        )
        body = BrowserCaptureAcceptedPayload.model_validate(json.loads(response.read()))

    assert response.status == HTTPStatus.ACCEPTED
    assert response.getheader("X-Request-ID")
    assert body.ok is True
    assert body.receiver == "polylogue-browser-capture"
    assert body.source == "browser-extension"
    assert body.schema_version == 1
    assert body.capture_id == "chatgpt:conv-123"
    assert body.artifact_ref == capture_artifact_ref(BrowserCaptureEnvelope.model_validate(_payload()), tmp_path)
    assert Path(body.artifact_ref).is_absolute() is False
    assert (tmp_path / body.artifact_ref).exists()


def test_receiver_echoes_safe_request_id_header(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "GET",
            "/v1/status",
            headers={
                "Origin": _EXTENSION_ORIGIN,
                "X-Request-ID": "dev-loop/request 123",
            },
        )
        response = conn.getresponse()
        response.read()
        conn.close()

    assert response.status == HTTPStatus.OK
    assert response.getheader("X-Request-ID") == "dev-looprequest123"


@pytest.mark.parametrize(
    ("raw_body", "expected_error"),
    [
        ("{not-json", "invalid_json"),
        ({"polylogue_capture_kind": "browser_llm_session", "schema_version": 1}, "invalid_payload"),
        (
            {
                **_payload(),
                "session": {"provider": "chatgpt", "provider_session_id": "conv-123", "turns": []},
            },
            "invalid_payload",
        ),
    ],
    ids=["invalid-json", "missing-session", "empty-turns"],
)
def test_receiver_rejects_malformed_capture_payloads(
    tmp_path: Path,
    raw_body: object | str,
    expected_error: str,
) -> None:
    with _running_receiver(tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "POST",
            "/v1/browser-captures",
            body=raw_body,
            origin=_EXTENSION_ORIGIN,
        )
        error = BrowserCaptureErrorPayload.model_validate(json.loads(response.read()))

    assert response.status == HTTPStatus.BAD_REQUEST
    assert error.error == expected_error
    assert list(tmp_path.rglob("*.json")) == []


def test_receiver_auth_allows_cors_preflight_without_bearer_token(tmp_path: Path) -> None:
    with _running_receiver(tmp_path, auth_token="secret", extra_origins=(_CHATGPT_ORIGIN,)) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "OPTIONS",
            "/v1/browser-captures",
            headers={
                "Origin": _CHATGPT_ORIGIN,
                "Access-Control-Request-Headers": "authorization, content-type",
            },
        )
        response = conn.getresponse()
        response.read()
        conn.close()

    assert response.status == HTTPStatus.NO_CONTENT


def test_receiver_allows_extra_web_origin_only_with_token(tmp_path: Path) -> None:
    with _running_receiver(tmp_path, auth_token="secret", extra_origins=(_CHATGPT_ORIGIN,)) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "POST",
            "/v1/browser-captures",
            body=json.dumps(_payload()),
            headers={
                "Content-Type": "application/json",
                "Origin": _CHATGPT_ORIGIN,
                "Authorization": "Bearer secret",
            },
        )
        response = conn.getresponse()
        body = BrowserCaptureAcceptedPayload.model_validate(json.loads(response.read()))
        conn.close()

    assert response.status == HTTPStatus.ACCEPTED
    assert body.ok is True
    assert Path(body.artifact_ref).is_absolute() is False
