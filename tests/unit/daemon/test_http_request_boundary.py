"""The daemon HTTP request boundary answers an escaped exception instead of dropping the connection."""

from __future__ import annotations

import http.client
import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus

import pytest

from polylogue.archive.query.transaction import QueryArchiveEpochUnreadableError
from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer
from polylogue.logging import DEBUG, capture, set_level
from polylogue.surfaces.payloads import QueryFailurePayload


@contextmanager
def _running_server(escaped: list[object] | None = None) -> Iterator[int]:
    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    if escaped is not None:
        # socketserver reports an exception that escaped the handler here.
        server.handle_error = lambda request, client_address: escaped.append(client_address)  # type: ignore[method-assign]
    server.auth_token = ""
    server.api_host = "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, name="http-request-boundary", daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_escaped_read_error_is_answered_with_a_500_error_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    """An undecorated read route that raises answers 500 ``outcome: error`` JSON.

    Anti-vacuity: removing the ``except Exception`` boundary in ``do_GET``
    lets the error reach socketserver, which drops the connection, so
    ``getresponse()`` raises ``RemoteDisconnected`` and this test fails.
    """

    def _raising_route(self: DaemonAPIHandler, params: dict[str, list[str]]) -> None:
        raise QueryArchiveEpochUnreadableError("user.db is absent")

    monkeypatch.setattr(DaemonAPIHandler, "_serve_webui_pastes", _raising_route)
    with _running_server() as port:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            connection.request("GET", "/p")
            response = connection.getresponse()
            body = response.read()
        finally:
            connection.close()

    assert response.status == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.getheader("Content-Type") == "application/json"
    payload = json.loads(body)
    assert payload["ok"] is False
    assert payload["error"] == "internal_error"
    assert payload["outcome"]["state"] == "error"
    # The answer stays inside the declared shared error contract.
    assert QueryFailurePayload.model_validate(payload).outcome.state == "error"


def test_error_after_response_started_closes_without_a_second_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """A route failing mid-response is not answered with a second status line.

    Anti-vacuity: dropping the ``_response_started`` check makes the boundary
    append a 500 envelope after the 200 body, so the raw bytes on the socket
    carry two status lines and this goes red.
    """

    def _half_written_route(self: DaemonAPIHandler, params: dict[str, list[str]]) -> None:
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")
        raise RuntimeError("failed after the answer began")

    monkeypatch.setattr(DaemonAPIHandler, "_serve_webui_pastes", _half_written_route)
    with _running_server() as port:
        # ``http.client`` would silently reconnect, so read the raw stream until
        # the server closes it and count the status lines it sent.
        with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
            sock.sendall(b"GET /p HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            received = b""
            while chunk := sock.recv(65536):
                received += chunk

    assert received.startswith(b"HTTP/1.")
    assert b" 200 " in received.split(b"\r\n", 1)[0]
    assert received.count(b"HTTP/1.") == 1, received
    assert received.endswith(b"{}")


def test_disconnect_while_answering_an_escaped_error_is_logged_not_escaped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client gone before the boundary's error answer is written is a debug event.

    Anti-vacuity: without the disconnect guard around the boundary answer, the
    ``BrokenPipeError`` raised while writing it escapes to socketserver, which
    calls ``handle_error`` and this test records it.
    """

    def _raising_route(self: DaemonAPIHandler, params: dict[str, list[str]]) -> None:
        raise RuntimeError("route failed")

    def _gone(self: DaemonAPIHandler, *args: object, **kwargs: object) -> None:
        raise BrokenPipeError("client went away")

    monkeypatch.setattr(DaemonAPIHandler, "_serve_webui_pastes", _raising_route)
    monkeypatch.setattr(DaemonAPIHandler, "_send_json", _gone)
    escaped: list[object] = []
    previous_level = set_level(DEBUG)
    try:
        with capture() as records, _running_server(escaped) as port:
            with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
                sock.sendall(b"GET /p HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
                while sock.recv(65536):
                    pass
    finally:
        set_level(previous_level)

    assert escaped == []
    disconnects = [record for record in records if record.get("event") == "daemon.http.client_disconnected"]
    assert len(disconnects) == 1
    assert disconnects[0]["error_type"] == "BrokenPipeError"
