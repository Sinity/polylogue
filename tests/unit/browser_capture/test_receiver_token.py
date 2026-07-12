"""Auto-minted browser-capture receiver bearer token (polylogue-gnie).

The receiver used to default to no auth at all: any local process (not just
a browser page) could read spool/archive-lifecycle state and post forged
captures. These tests cover the auto-mint/load/rotate primitives and prove,
via a real HTTP round trip against the threaded receiver server, that the
resolved default token actually gates both GET and POST routes and that the
same value ``browser-capture token show`` prints is what pairs a client.
"""

from __future__ import annotations

import json
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import cast

from polylogue.browser_capture.receiver import (
    load_or_mint_receiver_token,
    resolve_receiver_auth_token,
)
from polylogue.browser_capture.server import make_server

_EXTENSION_ORIGIN = "chrome-extension://polylogue-test"


def _capture_payload(session_id: str = "conv-123") -> dict[str, object]:
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {
            "source_url": "https://chatgpt.com/c/conv-123",
            "page_title": "ChatGPT - Work plan",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-dom-v1",
            "extension_instance_id": "test-extension-instance",
        },
        "session": {
            "provider": "chatgpt",
            "provider_session_id": session_id,
            "title": "Work plan",
            "turns": [{"provider_turn_id": "u1", "role": "user", "text": "Draft"}],
        },
    }


@contextmanager
def _running_receiver(spool_path: Path, *, auth_token: str | None) -> Iterator[tuple[str, int]]:
    server = make_server("127.0.0.1", 0, spool_path=spool_path, auth_token=auth_token)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_load_or_mint_receiver_token_creates_0600_file_and_persists(tmp_path: Path) -> None:
    token_path = tmp_path / "receiver-token"

    first = load_or_mint_receiver_token(token_path)
    second = load_or_mint_receiver_token(token_path)

    assert first == second
    assert token_path.read_text(encoding="utf-8").strip() == first
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_load_or_mint_receiver_token_rotate_changes_the_value(tmp_path: Path) -> None:
    token_path = tmp_path / "receiver-token"

    original = load_or_mint_receiver_token(token_path)
    rotated = load_or_mint_receiver_token(token_path, rotate=True)
    reloaded = load_or_mint_receiver_token(token_path)

    assert rotated != original
    assert reloaded == rotated


def test_resolve_receiver_auth_token_prefers_explicit_token(tmp_path: Path) -> None:
    token_path = tmp_path / "receiver-token"

    resolved = resolve_receiver_auth_token("explicit-secret", token_path=token_path)

    assert resolved == "explicit-secret"
    assert not token_path.exists()


def test_resolve_receiver_auth_token_allow_no_auth_returns_none(tmp_path: Path) -> None:
    token_path = tmp_path / "receiver-token"

    resolved = resolve_receiver_auth_token(None, allow_no_auth=True, token_path=token_path)

    assert resolved is None
    assert not token_path.exists()


def test_resolve_receiver_auth_token_default_mints_and_matches_persisted_file(tmp_path: Path) -> None:
    token_path = tmp_path / "receiver-token"

    resolved = resolve_receiver_auth_token(None, token_path=token_path)

    assert resolved == token_path.read_text(encoding="utf-8").strip()
    # A second resolution (e.g. daemon restart) must pair with the same
    # already-configured extension -- the token must not silently rotate.
    assert resolve_receiver_auth_token(None, token_path=token_path) == resolved


def test_default_resolved_token_gates_get_and_post_and_pairs_end_to_end(tmp_path: Path) -> None:
    """Reproduces the operator pairing flow end-to-end over real HTTP:

    daemon resolves/mints a token -> receiver requires it on every route
    (not just POST) -> a client presenting the value `token show` would
    print succeeds; anything else (or nothing) is refused.
    """
    token_path = tmp_path / "receiver-token"
    spool = tmp_path / "spool"
    resolved_token = resolve_receiver_auth_token(None, token_path=token_path)

    with _running_receiver(spool, auth_token=resolved_token) as (host, port):
        unauthenticated_status = HTTPConnection(host, port)
        unauthenticated_status.request("GET", "/v1/status", headers={"Origin": _EXTENSION_ORIGIN})
        unauthenticated_status_response = unauthenticated_status.getresponse()
        unauthenticated_status_response.read()
        unauthenticated_status.close()

        authenticated_status = HTTPConnection(host, port)
        authenticated_status.request(
            "GET",
            "/v1/status",
            headers={"Origin": _EXTENSION_ORIGIN, "Authorization": f"Bearer {resolved_token}"},
        )
        authenticated_status_response = authenticated_status.getresponse()
        authenticated_status_response.read()
        authenticated_status.close()

        unauthenticated_post = HTTPConnection(host, port)
        unauthenticated_post.request(
            "POST",
            "/v1/browser-captures",
            body=json.dumps(_capture_payload()),
            headers={"Origin": _EXTENSION_ORIGIN, "Content-Type": "application/json"},
        )
        unauthenticated_post_response = unauthenticated_post.getresponse()
        unauthenticated_post_response.read()
        unauthenticated_post.close()

        authenticated_post = HTTPConnection(host, port)
        authenticated_post.request(
            "POST",
            "/v1/browser-captures",
            body=json.dumps(_capture_payload()),
            headers={
                "Origin": _EXTENSION_ORIGIN,
                "Content-Type": "application/json",
                "Authorization": f"Bearer {resolved_token}",
            },
        )
        authenticated_post_response = authenticated_post.getresponse()
        authenticated_post_response.read()
        authenticated_post.close()

    assert unauthenticated_status_response.status == HTTPStatus.UNAUTHORIZED
    assert authenticated_status_response.status == HTTPStatus.OK
    assert unauthenticated_post_response.status == HTTPStatus.UNAUTHORIZED
    assert authenticated_post_response.status == HTTPStatus.ACCEPTED


def test_allow_no_auth_receiver_serves_unauthenticated_by_explicit_opt_out(tmp_path: Path) -> None:
    token_path = tmp_path / "receiver-token"
    spool = tmp_path / "spool"
    resolved_token = resolve_receiver_auth_token(None, allow_no_auth=True, token_path=token_path)
    assert resolved_token is None

    with _running_receiver(spool, auth_token=resolved_token) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request("GET", "/v1/status", headers={"Origin": _EXTENSION_ORIGIN})
        response = conn.getresponse()
        response.read()
        conn.close()

    assert response.status == HTTPStatus.OK
