"""The CLI signs the browser in with a one-time ticket in the URL fragment (polylogue-n3xdn)."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
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
    """Anti-vacuity: open the bare URL and the browser lands on the sign-in page; append the
    fragment to the archive URL and an already-credentialed browser never consumes it."""
    seen: list[Any] = []

    def fake_urlopen(request: Any, **kwargs: object) -> _Response:
        # A deadline would turn a slow ticket mint into an unsigned URL.
        assert kwargs.get("timeout") is None
        seen.append(request)
        return _Response(json.dumps({"ok": True, "ticket": "t/1", "expires_at": "2026-01-01T00:00:00Z"}).encode())

    monkeypatch.setattr(web_sign_in, "_api_token", lambda _env: "owner-token")
    monkeypatch.setattr(web_sign_in, "_open_loopback", fake_urlopen)

    url = web_sign_in.signed_in_web_url(object(), "http://127.0.0.1:8766/", "http://127.0.0.1:8766/s/abc")  # type: ignore[arg-type]

    # The ticket never reaches the browser process's argv: only a local,
    # owner-only redirect file's path is returned, and the ticketed URL lives
    # solely in that file's content.
    assert url.startswith("file://")
    from urllib.parse import unquote, urlsplit

    redirect_path = unquote(urlsplit(url).path)
    assert oct(os.stat(redirect_path).st_mode & 0o777) == oct(0o600)
    content = Path(redirect_path).read_text(encoding="utf-8")
    assert "http://127.0.0.1:8766/web-auth/sign-in?next=/s/abc#polylogue-ticket=t%2F1" in content
    assert seen[0].full_url == "http://127.0.0.1:8766/api/web-auth/ticket"
    assert seen[0].get_header("Authorization") == "Bearer owner-token"


def test_unreachable_daemon_falls_back_to_the_plain_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(_request: Any, **_kwargs: object) -> _Response:
        raise URLError("connection refused")

    monkeypatch.setattr(web_sign_in, "_api_token", lambda _env: "owner-token")
    monkeypatch.setattr(web_sign_in, "_open_loopback", refuse)

    assert web_sign_in.signed_in_web_url(object(), "http://127.0.0.1:8766", "http://127.0.0.1:8766/") == (  # type: ignore[arg-type]
        "http://127.0.0.1:8766/"
    )


def test_no_token_deployment_opens_the_plain_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_sign_in, "_api_token", lambda _env: None)
    assert web_sign_in.signed_in_web_url(object(), "http://x", "http://x/") == "http://x/"  # type: ignore[arg-type]


def test_the_redirect_url_encodes_uri_significant_path_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity (Codex P2, #5704): concatenate ``file://`` and the path and a
    ``#`` in TMPDIR turns the rest of the path into a fragment."""
    from urllib.parse import unquote, urlsplit

    tmp = tmp_path / "tmp#private"
    tmp.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(tmp))
    url = web_sign_in._local_redirect_url("http://127.0.0.1:8766/")

    parts = urlsplit(url)
    assert parts.fragment == ""
    assert Path(unquote(parts.path)).exists()


def test_the_ticket_exchange_bypasses_environment_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P1, #5704): open with the default opener and a set
    ``HTTP_PROXY`` receives the owner bearer."""
    import urllib.request

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    opened: list[tuple[object, ...]] = []

    class Opener:
        def __init__(self, handlers: tuple[object, ...]) -> None:
            self.handlers = handlers

        def open(self, request: object) -> object:
            opened.append(self.handlers)
            raise URLError("stop")

    monkeypatch.setattr(web_sign_in, "build_opener", lambda *handlers: Opener(handlers))
    with pytest.raises(URLError):
        web_sign_in._open_loopback(urllib.request.Request("http://127.0.0.1:8766/api/web-auth/ticket"))

    (handlers,) = opened
    proxies = [handler for handler in handlers if isinstance(handler, urllib.request.ProxyHandler)]
    assert proxies and getattr(proxies[0], "proxies", None) == {}
