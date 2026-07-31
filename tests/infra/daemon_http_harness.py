"""Shared HTTP-transport mocks for ``DaemonAPIHandler`` tests (polylogue-myhg).

CodeRabbit flagged (PR #2559) that the ``_MockServer``/``_MockHeaders``/
``_make_handler`` trio was hand-duplicated near-identically across three
daemon test files, and drift already happened once: the ``host=`` parameter
was added to two of the three copies during a fast-follow, not all three.
This module is the single place that logic lives now.

**What this mocks, and why that is still a real test:** ``DaemonAPIHandler``
is a ``socketserver.BaseRequestHandler`` subclass that normally gets
constructed by ``ThreadingHTTPServer`` from a live TCP connection. Standing
up a real socket per test is unnecessary I/O for exercising handler logic,
so these helpers replace exactly the transport it sits on:

- ``MockDaemonServer`` stands in for the listening ``DaemonAPIHTTPServer``
  (its thread pool / admission semaphore / configured auth token), not for
  the handler being tested.
- ``MockHeaders`` stands in for the parsed ``http.client.HTTPMessage`` a
  real socket read would produce.
- ``handler.rfile``/``handler.wfile`` are ``BytesIO`` standing in for the
  socket's read/write ends.

Everything downstream of the transport — ``do_GET``/``do_POST`` dispatch,
``_check_auth``, cross-origin checks, route handlers, JSON/SSE
serialization — is the real, unmocked ``DaemonAPIHandler`` implementation
built via ``DaemonAPIHandler.__new__``. Do not add production-logic
short-circuits here: the point of this harness is to exercise the real
handler against a fake network, not to replace the handler.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from io import BytesIO
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

from polylogue.daemon.web_auth import WebCredentialRegistry

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


class MockDaemonServer:
    """Stand-in for ``DaemonAPIHTTPServer``: the listening socket, thread
    pool, and admission semaphore the real server owns — not the
    request-handling logic under test. ``archive_query_executor``/
    ``archive_query_admission`` are generously sized shared class
    attributes; no test in this harness exercises their throttling."""

    api_host = "127.0.0.1"
    archive_query_executor = ThreadPoolExecutor(max_workers=1)
    archive_query_admission = threading.BoundedSemaphore(64)

    def __init__(self, *, auth_token: str = "", web_credentials: WebCredentialRegistry | None = None) -> None:
        self.auth_token = auth_token
        self.web_credentials = web_credentials if web_credentials is not None else WebCredentialRegistry()


class MockHeaders:
    """Stand-in for ``http.client.HTTPMessage`` (the parsed header block a
    real socket read would produce)."""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def make_daemon_handler(
    method: str,
    path: str,
    *,
    body: bytes = b"",
    auth_header: str = "",
    origin: str = "",
    host: str = "",
    cookie: str = "",
    referer: str = "",
    fetch_site: str = "",
    web_client: bool = False,
    extra_headers: dict[str, str] | None = None,
    server: object | None = None,
) -> DaemonAPIHandler:
    """Build a real ``DaemonAPIHandler`` with only its transport replaced.

    The handler is a genuine ``DaemonAPIHandler`` instance — ``do_GET``/
    ``do_POST``/``_check_auth``/route dispatch all run the production
    implementation against it. Callers invoke ``handler.do_GET()`` (etc.)
    and assert against ``handler.wfile`` or a patched ``_send_json``/
    ``_send_error`` (see ``capture_json_response``/``capture_responses``).

    ``host`` defaults to unset, matching every pre-existing caller of the
    predecessor per-file helpers — ``_check_host_admission_logic`` treats
    an absent Host header as permissive, so omitting it is equivalent to a
    non-browser client and does not gate auth/origin coverage. Pass ``host``
    explicitly to test the Host gate itself.

    Pass ``server=`` to control the auth token the request is checked
    against (default: no token configured, i.e. auth open — the local-dev
    default). ``extra_headers`` is a low-level escape hatch for header
    shapes the named kwargs don't cover; named kwargs win on conflict since
    they are applied first and ``extra_headers`` is merged in last.
    """
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast("DaemonAPIHTTPServer", server or MockDaemonServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.protocol_version = "HTTP/1.1"

    headers: dict[str, str] = {"Content-Length": str(len(body))}
    if auth_header:
        headers["Authorization"] = auth_header
    if origin:
        headers["Origin"] = origin
    if host:
        headers["Host"] = host
    if cookie:
        headers["Cookie"] = cookie
    if referer:
        headers["Referer"] = referer
    if fetch_site:
        headers["Sec-Fetch-Site"] = fetch_site
    if web_client:
        headers["X-Polylogue-Web-Client"] = "1"
    if extra_headers:
        headers.update(extra_headers)

    handler.headers = cast("Message[str, str]", MockHeaders(headers))
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    return handler


def capture_json_response(handler: DaemonAPIHandler) -> MagicMock:
    """Patch ``handler._send_json`` so its call args can be asserted
    instead of parsing raw bytes off ``handler.wfile``."""
    send_json = MagicMock()
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_json


def capture_responses(handler: DaemonAPIHandler) -> tuple[MagicMock, MagicMock]:
    """Patch both ``_send_error`` and ``_send_json`` so a single call site
    can assert whichever response path a route actually took."""
    send_error = MagicMock()
    send_json = MagicMock()
    handler._send_error = send_error  # type: ignore[method-assign]
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_error, send_json
