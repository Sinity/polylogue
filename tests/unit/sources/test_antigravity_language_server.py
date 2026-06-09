"""Unit tests for the Antigravity language-server export adapter.

The adapter spawns the local Antigravity language server binary and talks JSON
over a private loopback HTTP port. Tests here mock either the binary
discovery or the HTTP transport so they run without requiring the real
language server to be installed.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest

from polylogue.sources.parsers import antigravity
from polylogue.sources.parsers.antigravity import (
    AntigravityExportError,
    AntigravityLanguageServerClient,
    AntigravitySessionSummary,
    discover_language_server,
    iter_language_server_exports,
)
from polylogue.types import Provider


class _FakeHTTPResponse:
    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)

    def read(self) -> bytes:
        return self._buffer.read()

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


@pytest.fixture
def fake_client(tmp_path: Path) -> AntigravityLanguageServerClient:
    """A client whose start() is a no-op so tests can drive ._post directly."""

    client = AntigravityLanguageServerClient(tmp_path)

    # Make start()/close() inert and lock the port so _post emits a stable URL.
    def _noop_start() -> None:
        return None

    def _noop_close() -> None:
        return None

    client.start = _noop_start  # type: ignore[method-assign]
    client.close = _noop_close  # type: ignore[method-assign]
    return client


def test_summary_from_payload_requires_cascade_id() -> None:
    assert AntigravitySessionSummary.from_payload({}) is None
    assert AntigravitySessionSummary.from_payload({"cascadeId": ""}) is None
    summary = AntigravitySessionSummary.from_payload(
        {
            "cascadeId": "cascade-1",
            "title": "Title",
            "workspaceName": "ws",
            "snippet": "snip",
            "lastModifiedTime": "2026-03-05T04:21:34Z",
        }
    )
    assert summary is not None
    assert summary.cascade_id == "cascade-1"
    assert summary.workspace_name == "ws"


def test_markdown_export_payload_round_trips() -> None:
    summary = AntigravitySessionSummary(
        cascade_id="c1",
        title="t",
        workspace_name="w",
        snippet="s",
        last_modified_time="2026-03-05T04:21:34Z",
    )
    payload = antigravity.markdown_export_payload(summary, "### User Input\n\nhi\n")
    assert payload["source"] == "antigravity_language_server"
    assert payload["cascadeId"] == "c1"
    assert payload["title"] == "t"
    assert payload["workspaceName"] == "w"
    assert payload["snippet"] == "s"
    assert payload["lastModifiedTime"] == "2026-03-05T04:21:34Z"
    assert antigravity.looks_like_markdown_export(payload) is True


def test_looks_like_markdown_export_rejects_foreign_payloads() -> None:
    assert antigravity.looks_like_markdown_export({"cascadeId": "x", "markdown": "y"}) is False
    assert antigravity.looks_like_markdown_export({"source": "antigravity_language_server", "cascadeId": "x"}) is False


def test_search_sessions_returns_summaries(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(endpoint: str, payload: Any) -> dict[str, Any]:
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {
            "results": [
                {
                    "cascadeId": "cascade-1",
                    "title": "First",
                    "workspaceName": "ws",
                    "snippet": "snip",
                    "lastModifiedTime": "2026-03-05T04:21:34Z",
                },
                {"title": "missing-id"},  # rejected, no cascadeId
                "not-an-object",  # rejected, not a dict
                {"cascadeId": "cascade-2"},
            ]
        }

    monkeypatch.setattr(fake_client, "_post", fake_post)

    summaries = fake_client.search_sessions(limit=5, query="anything")

    assert captured["endpoint"].endswith("/SearchConversations")
    assert captured["payload"] == {"query": "anything", "limit": 5}
    assert [s.cascade_id for s in summaries] == ["cascade-1", "cascade-2"]
    assert summaries[0].workspace_name == "ws"


def test_search_sessions_handles_non_list_results(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fake_client, "_post", lambda *_a, **_k: {"results": "boom"})
    assert fake_client.search_sessions() == []


def test_export_markdown_returns_string(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fake_client,
        "_post",
        lambda endpoint, payload: {"markdown": "### User Input\n\nhello\n"},
    )
    assert "User Input" in fake_client.export_markdown("cascade-1")


def test_export_markdown_rejects_missing_or_empty_markdown(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fake_client, "_post", lambda *_a, **_k: {})
    with pytest.raises(AntigravityExportError):
        fake_client.export_markdown("cascade-1")

    monkeypatch.setattr(fake_client, "_post", lambda *_a, **_k: {"markdown": ""})
    with pytest.raises(AntigravityExportError):
        fake_client.export_markdown("cascade-1")

    monkeypatch.setattr(fake_client, "_post", lambda *_a, **_k: {"markdown": 42})
    with pytest.raises(AntigravityExportError):
        fake_client.export_markdown("cascade-1")


def test_post_wraps_url_errors(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_url_error(*_a: object, **_k: object) -> _FakeHTTPResponse:
        raise URLError("connection refused")

    monkeypatch.setattr("polylogue.sources.parsers.antigravity.urlopen", raise_url_error)

    with pytest.raises(AntigravityExportError) as exc_info:
        fake_client._post("/endpoint", {"q": "x"})
    assert "connection refused" in str(exc_info.value)


def test_post_rejects_non_object_responses(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(*_a: object, **_k: object) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(b"[1, 2, 3]")

    monkeypatch.setattr("polylogue.sources.parsers.antigravity.urlopen", fake_urlopen)

    with pytest.raises(AntigravityExportError) as exc_info:
        fake_client._post("/endpoint", {})
    assert "non-object JSON" in str(exc_info.value)


def test_post_returns_decoded_object(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(*_a: object, **_k: object) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(b'{"ok": true, "n": 1}')

    monkeypatch.setattr("polylogue.sources.parsers.antigravity.urlopen", fake_urlopen)

    result = fake_client._post("/endpoint", {"q": "x"})
    assert result == {"ok": True, "n": 1}


def test_discover_language_server_prefers_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "language_server_linux_x64"
    target.write_text("#!/bin/sh\nexit 0\n")
    target.chmod(0o755)

    monkeypatch.setenv("POLYLOGUE_ANTIGRAVITY_LANGUAGE_SERVER", str(target))
    # Even if PATH would shadow it, the env var wins.
    monkeypatch.setattr(
        "polylogue.sources.parsers.antigravity.shutil.which",
        lambda _name: "/should/not/be/used",
    )

    found = discover_language_server()
    assert found == target


def test_discover_language_server_falls_back_to_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POLYLOGUE_ANTIGRAVITY_LANGUAGE_SERVER", raising=False)
    monkeypatch.setattr(
        "polylogue.sources.parsers.antigravity.shutil.which",
        lambda name: "/usr/local/bin/language_server_linux_x64" if name == "language_server_linux_x64" else None,
    )
    monkeypatch.setattr("polylogue.sources.parsers.antigravity.glob", lambda _pattern: [])
    found = discover_language_server()
    assert found == Path("/usr/local/bin/language_server_linux_x64")


def test_discover_language_server_returns_none_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POLYLOGUE_ANTIGRAVITY_LANGUAGE_SERVER", raising=False)
    monkeypatch.setattr("polylogue.sources.parsers.antigravity.shutil.which", lambda _name: None)
    monkeypatch.setattr("polylogue.sources.parsers.antigravity.glob", lambda _pattern: [])
    assert discover_language_server() is None


def test_client_start_raises_when_binary_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "polylogue.sources.parsers.antigravity.discover_language_server",
        lambda: None,
    )
    client = AntigravityLanguageServerClient(tmp_path)
    with pytest.raises(AntigravityExportError) as exc_info:
        client.start()
    assert (
        "not be found" in str(exc_info.value)
        or "not be found" in str(exc_info.value).lower()
        or "was not found" in str(exc_info.value)
    )


class _FakeClientForExports:
    """Drop-in fake of AntigravityLanguageServerClient for driver tests."""

    def __init__(
        self,
        summaries: list[AntigravitySessionSummary],
        markdown: dict[str, str],
    ) -> None:
        self._summaries = summaries
        self._markdown = markdown
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def search_sessions(self, *, limit: int = 10000, query: str = "") -> list[AntigravitySessionSummary]:
        return list(self._summaries)

    def export_markdown(self, cascade_id: str) -> str:
        return self._markdown[cascade_id]


def test_iter_language_server_exports_yields_parsed_sessions(
    tmp_path: Path,
) -> None:
    summaries = [
        AntigravitySessionSummary(cascade_id="cascade-1", title="One", workspace_name="ws"),
        AntigravitySessionSummary(cascade_id="cascade-2", title="Two"),
    ]
    markdown = {
        "cascade-1": "### User Input\n\nhello\n\n### Planner Response\n\nhi\n",
        "cascade-2": "### User Input\n\nfoo\n\n### Planner Response\n\nbar\n",
    }
    fake = _FakeClientForExports(summaries, markdown)

    sessions = list(iter_language_server_exports(tmp_path, client=fake))  # type: ignore[arg-type]

    assert [c.provider_session_id for c in sessions] == [
        "cascade-1",
        "cascade-2",
    ]
    assert all(c.source_name is Provider.ANTIGRAVITY for c in sessions)
    assert sessions[0].messages[0].text == "hello"
    assert sessions[0].messages[1].text == "hi"
    # Externally supplied client must not be started or closed by the driver.
    assert fake.started is False
    assert fake.closed is False


def test_iter_language_server_exports_manages_owned_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summaries = [
        AntigravitySessionSummary(cascade_id="cascade-1", title="One"),
    ]
    markdown = {"cascade-1": "### User Input\n\nhello\n\n### Planner Response\n\nhi\n"}
    fake = _FakeClientForExports(summaries, markdown)

    monkeypatch.setattr(
        "polylogue.sources.parsers.antigravity.AntigravityLanguageServerClient",
        lambda root: fake,
    )

    sessions = list(iter_language_server_exports(tmp_path))

    assert [c.provider_session_id for c in sessions] == ["cascade-1"]
    assert fake.started is True
    assert fake.closed is True


def test_iter_language_server_exports_closes_client_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _ExplodingClient(_FakeClientForExports):
        def export_markdown(self, cascade_id: str) -> str:
            raise AntigravityExportError("boom")

    fake = _ExplodingClient(
        [AntigravitySessionSummary(cascade_id="cascade-1", title="One")],
        {},
    )
    monkeypatch.setattr(
        "polylogue.sources.parsers.antigravity.AntigravityLanguageServerClient",
        lambda root: fake,
    )

    def _consume() -> Iterator[Any]:
        yield from iter_language_server_exports(tmp_path)

    with pytest.raises(AntigravityExportError):
        list(_consume())

    assert fake.started is True
    assert fake.closed is True
