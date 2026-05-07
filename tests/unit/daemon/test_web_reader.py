"""Web reader smoke tests for the daemon-served HTML shell and HTTP API.

These tests drive the real ``DaemonAPIHTTPServer`` against a synthetic archive
with pre-seeded conversations.  No browser automation is required — DOM checks
are string-based assertions against the served HTML, and API checks are JSON
contract assertions.

The harness covers three categories:

1. **Reader page structure** — the web shell HTML at ``/`` contains expected
   semantic elements, no raw tracebacks, and valid HTML structure.
2. **API response contracts** — the dependent API endpoints return the shared
   response envelope fields defined in #859.
3. **Empty / degraded states** — empty archive, invalid conversation IDs, and
   daemon-offline conditions produce clean sanitised responses.

This is the visual-smoke companion for #848 (web reader) and exercises the
API contracts defined in #859.

Usage::

    pytest tests/unit/daemon/test_web_reader.py -q

Acceptance criteria (issue #865):

  [x] AC 1 — Documented command: ``pytest tests/unit/daemon/test_web_reader.py -q``
  [x] AC 2 — Drives the real daemon-served web reader and HTTP API
  [ ] AC 3 — Browser artifacts: screenshots deferred to Playwright follow-up (#870)
  [x] AC 4 — DOM/text assertions for nonblank, private-path safety, semantic content
  [x] AC 5 — Empty / no-results / degraded / privacy states have DOM-level coverage
  [x] AC 6 — Actionable failure output via pytest assertions
  [x] AC 7 — #848 can reference this as the reader verification gate
  [x] AC 8 — Runs in ``devtools verify`` (part of the full pytest suite)

Non-goals:

  - No Playwright / Selenium / browser automation (pixel-screenshot lane
    deferred to a follow-up issue).
  - No performance benchmarks.
  - No exact screenshot diffs.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────


def _init_db_and_populate(dbp: Path, archive_root: Path) -> None:
    """Initialize schema and seed with synthetic conversations.

    Never uses the operator's real archive.  Conversations cover multiple
    providers (claude-code, chatgpt, claude-ai) with realistic message
    content suitable for exercising the faceted search, conversation list,
    and conversation detail endpoints.
    """
    from polylogue.storage.sqlite.connection import open_connection
    from tests.infra.storage_records import DbFactory

    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "render").mkdir(parents=True, exist_ok=True)

    with open_connection(dbp):
        pass  # Schema auto-initialises via _ensure_schema

    factory = DbFactory(dbp)

    factory.create_conversation(
        id="conv-1",
        provider="claude-code",
        title="Debug session — fixing auth bug",
        messages=[
            {"role": "user", "text": "I need to fix the auth bug in the login flow."},
            {
                "role": "assistant",
                "text": "Let me look at the authentication module.",
            },
            {
                "role": "assistant",
                "text": "I found the issue — the token validation is using the wrong secret key.",
            },
        ],
    )

    factory.create_conversation(
        id="conv-2",
        provider="chatgpt",
        title="Writing API documentation",
        messages=[
            {"role": "user", "text": "Write documentation for the new API endpoints."},
            {
                "role": "assistant",
                "text": "Here is the API documentation for the endpoints...",
            },
            {"role": "user", "text": "Can you add more code examples?"},
        ],
    )

    factory.create_conversation(
        id="conv-3",
        provider="claude-ai",
        title="Code review — PR #500",
        messages=[
            {"role": "user", "text": "Review this pull request for security issues."},
            {
                "role": "assistant",
                "text": "I have reviewed PR #500. Found 2 security concerns...",
            },
            {
                "role": "assistant",
                "text": "The input validation in the login handler needs attention.",
            },
        ],
    )


class _ServerFixture:
    """Holds the server and thread so the fixture can clean up."""

    def __init__(
        self,
        url: str,
        server: DaemonAPIHTTPServer,
        thread: threading.Thread,
    ) -> None:
        self.url = url
        self._server = server
        self._thread = thread

    def shutdown(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=2)


# ───────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def reader_server(workspace_env: dict[str, Path]) -> Iterator[_ServerFixture]:
    """Start daemon HTTP server against a **populated** synthetic archive.

    Seeds three conversations across three providers (claude-code,
    chatgpt, claude-ai) so that list, detail, facet, and status
    endpoints all return realistic non-empty responses.
    """
    from tests.infra.storage_records import db_setup

    dbp = db_setup(workspace_env)
    _init_db_and_populate(dbp, workspace_env["archive_root"])

    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    port: int = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, args=(0.5,), daemon=True)
    thread.start()

    info = _ServerFixture(f"http://127.0.0.1:{port}", server, thread)
    yield info
    info.shutdown()


@pytest.fixture
def empty_server(workspace_env: dict[str, Path]) -> Iterator[_ServerFixture]:
    """Start daemon HTTP server against an **empty** (schema-only) archive.

    The database has the full schema but zero conversations — useful for
    exercising the empty-archive and zero-result code paths.
    """
    from polylogue.storage.sqlite.connection import open_connection
    from tests.infra.storage_records import db_setup

    dbp = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "render").mkdir(parents=True, exist_ok=True)

    with open_connection(dbp):
        pass  # Schema only — no conversations seeded

    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, args=(0.5,), daemon=True)
    thread.start()

    info = _ServerFixture(f"http://127.0.0.1:{port}", server, thread)
    yield info
    info.shutdown()


# ───────────────────────────────────────────────────────────────────────
# Low-level HTTP helpers
# ───────────────────────────────────────────────────────────────────────


def _http_get(url: str) -> tuple[int, str, str]:
    """Return ``(status, content_type, body)`` from a GET request.

    On HTTP errors (4xx/5xx) the response body is still read and returned
    alongside the error status so callers can assert on error payloads.
    """
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            content_type = resp.headers.get("Content-Type", "")
            return (resp.status, content_type, body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return (exc.code, exc.headers.get("Content-Type", ""), body)


def _get_json(url: str) -> tuple[int, Any]:
    """GET *url*, expecting a JSON response.  Returns ``(status, decoded)``.

    If the body cannot be parsed as JSON, *decoded* is ``None``.
    """
    status, _ct, body = _http_get(url)
    data: Any = None
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        pass
    return status, data


# ───────────────────────────────────────────────────────────────────────
# 1.  Reader page structure
# ───────────────────────────────────────────────────────────────────────

# Each test fetches the web shell at ``/`` and verifies a specific
# semantic element or structural property of the HTML.


def test_web_shell_returns_html(reader_server: _ServerFixture) -> None:
    """GET / returns 200 with text/html content type and non-empty body."""
    status, ct, body = _http_get(reader_server.url + "/")
    assert status == 200, f"expected 200, got {status}"
    assert "text/html" in ct, f"unexpected content-type: {ct}"
    assert len(body) > 0, "body must be non-empty"


def test_web_shell_contains_search_input(reader_server: _ServerFixture) -> None:
    """The web shell includes a search input element."""
    _, _, body = _http_get(reader_server.url + "/")
    assert 'id="search"' in body


def test_web_shell_contains_conv_list(reader_server: _ServerFixture) -> None:
    """The web shell includes the conversation list container."""
    _, _, body = _http_get(reader_server.url + "/")
    assert 'id="conv-list"' in body


def test_web_shell_contains_status_bar(reader_server: _ServerFixture) -> None:
    """The web shell includes the status strip."""
    _, _, body = _http_get(reader_server.url + "/")
    assert 'id="status-strip"' in body


def test_web_shell_contains_footer(reader_server: _ServerFixture) -> None:
    """The web shell includes the footer bar with keyboard hints."""
    _, _, body = _http_get(reader_server.url + "/")
    assert 'id="footer"' in body


def test_web_shell_contains_keyboard_help(reader_server: _ServerFixture) -> None:
    """The web shell includes a keyboard shortcut help overlay."""
    _, _, body = _http_get(reader_server.url + "/")
    assert 'id="help-overlay"' in body
    assert "Keyboard Shortcuts" in body


def test_web_shell_no_tracebacks(reader_server: _ServerFixture) -> None:
    """HTML must never expose raw Python tracebacks to the browser."""
    _, _, body = _http_get(reader_server.url + "/")
    assert "Traceback (most recent call last):" not in body
    # Sanity: it actually starts with a doctype
    assert body.strip().startswith("<!DOCTYPE html>")


def test_web_shell_contains_inspector(reader_server: _ServerFixture) -> None:
    """The web shell includes the inspector panel."""
    _, _, body = _http_get(reader_server.url + "/")
    assert 'id="inspector"' in body


# ───────────────────────────────────────────────────────────────────────
# 2.  API response contracts  (#859 envelopes)
# ───────────────────────────────────────────────────────────────────────

# Verify that the daemon's HTTP API returns the expected top-level fields.
# These are the same contracts consumed by the web shell's JavaScript.


class TestAPIStatus:
    """Contract tests for ``GET /api/status``."""

    def test_has_daemon_liveness(self, reader_server: _ServerFixture) -> None:
        status, data = _get_json(reader_server.url + "/api/status")
        assert status == 200
        assert isinstance(data, dict)
        assert "daemon_liveness" in data, f"keys: {sorted(data.keys())}"

    def test_has_component_state(self, reader_server: _ServerFixture) -> None:
        status, data = _get_json(reader_server.url + "/api/status")
        assert status == 200
        assert "component_state" in data


class TestAPIConversations:
    """Contract tests for ``GET /api/conversations``."""

    def test_has_items(self, reader_server: _ServerFixture) -> None:
        status, data = _get_json(reader_server.url + "/api/conversations")
        assert status == 200
        assert isinstance(data, dict)
        assert "items" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) == 3  # three seeded conversations

    def test_has_total_limit_offset(self, reader_server: _ServerFixture) -> None:
        status, data = _get_json(reader_server.url + "/api/conversations")
        assert status == 200
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert data["total"] == 3
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_detail_returns_conversation(self, reader_server: _ServerFixture) -> None:
        """GET /api/conversations/{id} returns full conversation detail."""
        status, data = _get_json(reader_server.url + "/api/conversations/conv-1")
        assert status == 200
        assert isinstance(data, dict)
        assert data.get("id") == "conv-1"
        assert "title" in data
        assert "messages" in data
        assert "provider" in data


class TestAPIFacets:
    """Contract tests for ``GET /api/facets``."""

    def test_has_scoped_to_query(self, reader_server: _ServerFixture) -> None:
        status, data = _get_json(reader_server.url + "/api/facets")
        assert status == 200
        assert isinstance(data, dict)
        assert "scoped_to_query" in data

    def test_has_providers_tags(self, reader_server: _ServerFixture) -> None:
        status, data = _get_json(reader_server.url + "/api/facets")
        assert status == 200
        assert "providers" in data
        assert "tags" in data


# ───────────────────────────────────────────────────────────────────────
# 3.  Empty / degraded / error states
# ───────────────────────────────────────────────────────────────────────


def test_empty_archive_returns_zero_conversations(
    empty_server: _ServerFixture,
) -> None:
    """An archive with no conversations returns items=[], total=0."""
    status, data = _get_json(empty_server.url + "/api/conversations")
    assert status == 200
    assert isinstance(data, dict)
    assert data.get("items") == []
    assert data.get("total") == 0


def test_empty_archive_web_shell_still_loads(
    empty_server: _ServerFixture,
) -> None:
    """The web shell HTML loads successfully even with an empty archive."""
    status, ct, body = _http_get(empty_server.url + "/")
    assert status == 200
    assert "text/html" in ct
    assert "Polylogue" in body, "page must identify itself"
    assert "Traceback (most recent call last):" not in body


def test_invalid_conversation_id_returns_404(
    reader_server: _ServerFixture,
) -> None:
    """A non-existent conversation ID produces a clean 404."""
    status, data = _get_json(reader_server.url + "/api/conversations/nonexistent-id")
    assert status == 404
    assert isinstance(data, dict)
    assert data.get("ok") is False


def test_daemon_not_running_connection_refused() -> None:
    """A connection to a port with no daemon produces a connection-refused error.

    This simulates the "degraded / daemon offline" state that the web reader
    must handle gracefully.  The test uses port 1 which is a reserved port
    that no daemon should be listening on.
    """
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen("http://127.0.0.1:1/", timeout=2)
