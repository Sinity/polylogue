"""CaptureJob HTTP bodies stay streamed across the registry boundary."""

from __future__ import annotations

import json
import socket
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from types import TracebackType
from typing import Any, cast

import pytest

from polylogue.browser_capture import server as server_module
from polylogue.browser_capture.native_preparation import json_chunks
from polylogue.browser_capture.server import BrowserCaptureHandler, make_server
from polylogue.core.staged_body import BODY_READ_CHUNK_BYTES, BodyStorageExhaustedError, StagedBody, stage_body
from polylogue.schemas.observation_spill import StreamedJSONDocument

TOKEN = "capture-http-stream-test-token"


@contextmanager
def receiver(tmp_path: Path) -> Iterator[tuple[str, int]]:
    server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=TOKEN)
    server.daemon_threads = False
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "127.0.0.1", server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
        "X-Polylogue-Client-Protocol": "2",
    }


def test_capture_job_http_streams_arbitrary_nested_request_and_response_collections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = [f"event-{index:06d}-" + "x" * 48 for index in range(30_000)]
    body = {
        "provider": "chatgpt",
        "scope": {"kind": "account", "key": "opaque-scope"},
        "intent": "native_fetch",
        "refs": {"arbitrary": values},
        "payload": {"records": values},
        "unowned_extension_field": {"ignored": True},
    }
    content = json.dumps(body, separators=(",", ":")).encode()
    assert len(content) > 2 * BODY_READ_CHUNK_BYTES
    read_sizes: list[int] = []
    response_writes: list[int] = []
    original_stage = stage_body

    def observed_stage(read: Callable[[int], bytes], length: int, *, spool_root: Path) -> StagedBody:
        def observed_read(size: int) -> bytes:
            read_sizes.append(size)
            return read(size)

        return original_stage(observed_read, length, spool_root=spool_root)

    original_setup = BrowserCaptureHandler.setup

    class RecordingWriter:
        def __init__(self, stream: Any) -> None:
            self._stream = stream

        def write(self, value: bytes) -> int:
            response_writes.append(len(value))
            return cast(int, self._stream.write(value))

        def flush(self) -> None:
            self._stream.flush()

    def observed_setup(handler: BrowserCaptureHandler) -> None:
        original_setup(handler)
        handler.wfile = cast(Any, RecordingWriter(handler.wfile))

    class Registry:
        @contextmanager
        def result_scope(self) -> Iterator[None]:
            yield

        def event(self, job_id: str, request: dict[str, object]) -> dict[str, object]:
            assert job_id == "job"
            assert "unowned_extension_field" not in request
            nested_refs = request["refs"]
            nested_payload = request["payload"]
            assert isinstance(nested_refs, dict)
            assert isinstance(nested_payload, dict)
            nested_refs_values = nested_refs["arbitrary"]
            nested_payload_values = nested_payload["records"]
            assert isinstance(nested_refs_values, list)
            assert isinstance(nested_payload_values, list)
            assert nested_refs_values[-1] == values[-1]
            assert nested_payload_values[0] == values[0]
            return {"refs": nested_refs, "payload": nested_payload}

    monkeypatch.setattr("polylogue.browser_capture.server.stage_body", observed_stage)
    monkeypatch.setattr(BrowserCaptureHandler, "setup", observed_setup)
    monkeypatch.setattr(server_module, "registry_for_receiver", lambda *_args, **_kwargs: Registry())
    with receiver(tmp_path) as (host, port):
        connection = HTTPConnection(host, port)
        connection.request("POST", "/v1/capture-jobs/job/events", content, headers=_headers())
        response = connection.getresponse()
        actual = json.loads(response.read())
        assert response.status == 200
        assert response.getheader("Content-Length") is not None
        connection.close()

    assert actual == {"refs": body["refs"], "payload": body["payload"]}
    assert len(read_sizes) >= 3
    assert max(read_sizes) <= BODY_READ_CHUNK_BYTES
    assert response_writes
    assert max(response_writes) <= 64 * 1024
    assert not list((tmp_path / ".staging").glob(".capture-*.tmp"))


def test_get_capture_job_event_encodes_lazy_response_inside_registry_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Registry:
        scope_active = False

        @contextmanager
        def result_scope(self) -> Iterator[None]:
            self.scope_active = True
            try:
                yield
            finally:
                self.scope_active = False

        def events(self, _job_id: str, _query: dict[str, object]) -> dict[str, object]:
            registry = self

            class ScopedPayload(dict[str, object]):
                def items(self) -> Any:
                    assert registry.scope_active
                    return super().items()

            return ScopedPayload({"events": [{"refs": {"opaque": "kept"}}]})

    monkeypatch.setattr(server_module, "registry_for_receiver", lambda *_args, **_kwargs: Registry())
    with receiver(tmp_path) as (host, port):
        connection = HTTPConnection(host, port)
        connection.request(
            "GET",
            "/v1/capture-jobs/job/events?provider=chatgpt&scope=%7B%22kind%22%3A%22account%22%2C%22key%22%3A%22scope%22%7D&client_protocol=2",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()

    assert response.status == 200
    assert payload == {"events": [{"refs": {"opaque": "kept"}}]}


def test_late_json_scope_value_error_preserves_single_response_and_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_exit = StreamedJSONDocument.__exit__
    close_flags: list[bool] = []
    original_finish = BrowserCaptureHandler._finish_observed_request

    def fail_after_close(
        document: StreamedJSONDocument,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        original_exit(document, kind, value, traceback)
        if kind is None:
            raise ValueError("late streamed document cleanup failure")

    class Registry:
        @contextmanager
        def result_scope(self) -> Iterator[None]:
            yield

        def event(self, _job_id: str, _body: dict[str, object]) -> dict[str, object]:
            return {"event": "accepted"}

    def observed_finish(handler: BrowserCaptureHandler, method: str, started_at: float) -> None:
        close_flags.append(handler.close_connection)
        original_finish(handler, method, started_at)

    monkeypatch.setattr(StreamedJSONDocument, "__exit__", fail_after_close)
    monkeypatch.setattr(BrowserCaptureHandler, "_finish_observed_request", observed_finish)
    monkeypatch.setattr(server_module, "registry_for_receiver", lambda *_args, **_kwargs: Registry())
    body = json.dumps({"request_id": "late-error", "refs": {}, "payload": {}}).encode()

    with receiver(tmp_path) as (host, port):
        with socket.create_connection((host, port)) as connection:
            request_bytes = (
                f"POST /v1/capture-jobs/job/events HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Authorization: Bearer {TOKEN}\r\n"
                "Content-Type: application/json\r\n"
                "X-Polylogue-Client-Protocol: 2\r\n"
                "Connection: close\r\n"
                f"Content-Length: {len(body)}\r\n\r\n"
            ).encode("ascii") + body
            connection.sendall(request_bytes)
            response_parts: list[bytes] = []
            while chunk := connection.recv(64 * 1024):
                response_parts.append(chunk)
        wire = b"".join(response_parts)

    status_lines = [line for line in wire.split(b"\r\n") if line.startswith(b"HTTP/")]
    assert len(status_lines) == 1, repr(wire[:2048])
    headers, response_body = wire.split(b"\r\n\r\n", maxsplit=1)
    assert headers.split(b"\r\n", maxsplit=1)[0].endswith(b" 200 OK")
    declared_length = next(
        int(line.split(b":", maxsplit=1)[1])
        for line in headers.split(b"\r\n")[1:]
        if line.lower().startswith(b"content-length:")
    )
    assert declared_length == len(response_body)
    assert json.loads(response_body) == {"event": "accepted"}
    assert close_flags == [True]
    assert not list((tmp_path / ".staging").glob(".capture-*.tmp"))


def test_response_spool_exhaustion_sends_one_fixed_507_without_staging_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    def exhausted(*_args: object, **_kwargs: object) -> StagedBody:
        nonlocal attempts
        attempts += 1
        raise BodyStorageExhaustedError(1, 0)

    monkeypatch.setattr(server_module, "stage_body_chunks", exhausted)
    with receiver(tmp_path) as (host, port):
        connection = HTTPConnection(host, port)
        connection.request("GET", "/v1/no-such", headers={"Authorization": f"Bearer {TOKEN}"})
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()

    assert attempts == 1
    assert response.status == 507
    assert payload == {"error": {"code": "spool_storage_exhausted", "details": {}}}
    assert not list((tmp_path / ".staging").glob(".capture-*.tmp"))


def test_native_preparation_json_chunks_preserve_spilled_provider_metadata(tmp_path: Path) -> None:
    metadata = {
        "capture_fidelity": "native_full",
        "collection": [{"ordinal": index, "labels": [f"label-{index}", "café"]} for index in range(1200)],
    }
    body_path = tmp_path / "native-metadata.json"
    body_path.write_text(json.dumps({"provider_meta": metadata}), encoding="utf-8")

    with StreamedJSONDocument(body_path) as root:
        assert isinstance(root, dict)
        spilled = root["provider_meta"]
        assert isinstance(spilled, dict)
        streamed = b"".join(json_chunks(spilled))
        streamed_sorted = b"".join(json_chunks(spilled, sort_keys=True))

    expected = json.dumps(metadata, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    expected_sorted = json.dumps(metadata, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
    assert streamed == expected
    assert streamed_sorted == expected_sorted


@pytest.mark.parametrize(
    ("content", "declared_extra", "expected_error"),
    [(b'{"refs":', 0, "invalid_json"), (b'{"refs":{}', 16, "incomplete_body")],
)
def test_capture_job_http_discards_staged_body_after_parse_or_disconnect_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
    declared_extra: int,
    expected_error: str,
) -> None:
    class Registry:
        @contextmanager
        def result_scope(self) -> Iterator[None]:
            yield

    monkeypatch.setattr(server_module, "registry_for_receiver", lambda *_args, **_kwargs: Registry())
    with receiver(tmp_path) as (host, port):
        connection = HTTPConnection(host, port)
        connection.putrequest("POST", "/v1/capture-jobs")
        for key, value in _headers().items():
            connection.putheader(key, value)
        connection.putheader("Content-Length", str(len(content) + declared_extra))
        connection.endheaders()
        connection.send(content)
        if declared_extra:
            assert connection.sock is not None
            connection.sock.shutdown(socket.SHUT_WR)
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()

    assert response.status == 400
    assert payload["error"] == expected_error
    assert not list((tmp_path / ".staging").glob(".capture-*.tmp"))


def test_capture_job_lazy_sqlite_full_preserves_storage_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.schemas.observation_spill import SpilledObject

    original_enter = StreamedJSONDocument.__enter__
    failure = sqlite3.OperationalError("neutral SQLite physical value storage exhausted")
    failure.sqlite_errorcode = sqlite3.SQLITE_FULL
    failure.sqlite_errorname = "SQLITE_FULL"

    class FailedConnection:
        def execute(self, *args: Any, **kwargs: Any) -> Any:
            raise failure

    def fail_lazy_read(document: StreamedJSONDocument) -> object:
        root = original_enter(document)
        assert isinstance(root, SpilledObject)
        root._connection = cast(sqlite3.Connection, FailedConnection())
        return root

    monkeypatch.setattr(StreamedJSONDocument, "__enter__", fail_lazy_read)
    with receiver(tmp_path) as (host, port):
        connection = HTTPConnection(host, port)
        try:
            connection.request("POST", "/v1/capture-jobs", body=b'{"provider":"chatgpt"}', headers=_headers())
            response = connection.getresponse()
            assert response.status == 507
            assert json.loads(response.read()) == {"error": {"code": "spool_storage_exhausted", "details": {}}}
        finally:
            connection.close()
