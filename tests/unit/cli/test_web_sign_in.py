"""The CLI signs the browser in with a one-time ticket in the URL fragment (polylogue-n3xdn)."""

from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import URLError

import pytest

from polylogue.cli.shared import web_sign_in


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def test_opened_url_carries_a_ticket_fragment_minted_with_the_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: open the bare URL and the browser lands on the sign-in page."""
    seen: list[Any] = []

    def fake_urlopen(request: Any, *, timeout: float) -> _Response:
        del timeout
        seen.append(request)
        return _Response(json.dumps({"ok": True, "ticket": "t/1", "expires_at": "2026-01-01T00:00:00Z"}).encode())

    monkeypatch.setattr(web_sign_in, "_api_token", lambda _env: "owner-token")
    monkeypatch.setattr(web_sign_in, "urlopen", fake_urlopen)

    url = web_sign_in.signed_in_web_url(object(), "http://127.0.0.1:8766/", "http://127.0.0.1:8766/s/abc")  # type: ignore[arg-type]

    assert url == "http://127.0.0.1:8766/s/abc#polylogue-ticket=t%2F1"
    assert seen[0].full_url == "http://127.0.0.1:8766/api/web-auth/ticket"
    assert seen[0].get_header("Authorization") == "Bearer owner-token"


def test_unreachable_daemon_falls_back_to_the_plain_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(_request: Any, *, timeout: float) -> _Response:
        raise URLError("connection refused")

    monkeypatch.setattr(web_sign_in, "_api_token", lambda _env: "owner-token")
    monkeypatch.setattr(web_sign_in, "urlopen", refuse)

    assert web_sign_in.signed_in_web_url(object(), "http://127.0.0.1:8766", "http://127.0.0.1:8766/") == (  # type: ignore[arg-type]
        "http://127.0.0.1:8766/"
    )


def test_no_token_deployment_opens_the_plain_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_sign_in, "_api_token", lambda _env: None)
    assert web_sign_in.signed_in_web_url(object(), "http://x", "http://x/") == "http://x/"  # type: ignore[arg-type]
