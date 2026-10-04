"""Finite framing and cleanup at the operation response owner."""

from __future__ import annotations

import io
import json
from typing import BinaryIO

import pytest

from polylogue.operations.read_result_transport import TRANSFER_BYTES, decode_json_response, staged_json_response


def test_large_utf8_response_roundtrips_without_aggregate_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import polylogue.operations.read_result_transport as transport

    class IncrementalOnly(json.JSONEncoder):
        def encode(self, value: object) -> str:
            raise AssertionError("whole-envelope encoder used")

    monkeypatch.setattr(transport, "json", SimpleNamespace(JSONEncoder=IncrementalOnly))
    value = {"text": '[λ]\\"\n' * (2 * 1024 * 1024), "integer": 2**100, "decimal": 1.25, "none": None}
    with staged_json_response(value) as staged:
        size = staged.seek(0, 2)
        assert size > 8 * 1024 * 1024
        staged.seek(0)
        assert decode_json_response(staged, size) == value
    assert staged.closed


@pytest.mark.parametrize("payload", [{"bad": float("nan")}, {"bad": object()}])
def test_encoding_failure_publishes_no_staged_response(payload: object) -> None:
    reached = False
    with pytest.raises((TypeError, ValueError)):
        with staged_json_response(payload):
            reached = True
    assert not reached


@pytest.mark.parametrize("raw", [b'{"x":', b"{} trailing", b'{"x":"\xff"}', b"", b"{} {}"])
def test_malformed_or_truncated_json_never_returns_a_value(raw: bytes) -> None:
    with pytest.raises((ValueError, UnicodeDecodeError)):
        decode_json_response(io.BytesIO(raw), len(raw))


def test_short_framed_body_refuses() -> None:
    with pytest.raises(EOFError, match="incomplete"):
        decode_json_response(io.BytesIO(b"{}"), 3)


def test_decoder_uses_bounded_reads() -> None:
    class Observed(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and 0 <= size <= TRANSFER_BYTES
            return super().read(size)

    value = {"items": list(range(20000))}
    raw = json.dumps(value).encode()
    assert decode_json_response(Observed(raw), len(raw)) == value


def test_delivery_cancel_closes_staged_owner() -> None:
    staged: BinaryIO | None = None
    with pytest.raises(KeyboardInterrupt):
        with staged_json_response({"text": "complete"}) as staged:
            raise KeyboardInterrupt
    assert staged is not None and staged.closed


@pytest.mark.parametrize("outcome,status", [("completed", 200), ("accepted", 202), ("failed", 409)])
def test_http_operation_response_preserves_framing_and_status(
    outcome: str, status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.daemon.http import DaemonAPIHandler

    handler = object.__new__(DaemonAPIHandler)
    received: list[int] = []
    headers: dict[str, str] = {}
    monkeypatch.setattr(handler, "send_response", received.append)
    monkeypatch.setattr(handler, "send_header", lambda key, value: headers.__setitem__(key, value))
    monkeypatch.setattr(handler, "_send_request_id_header", lambda: headers.__setitem__("X-Request-ID", "original"))
    monkeypatch.setattr(handler, "end_headers", lambda: None)
    handler.wfile = io.BytesIO()
    payload: dict[str, object] = {"outcome": outcome, "result": {"text": "λ" * 100000}}
    handler._send_daemon_operation(payload)
    raw = handler.wfile.getvalue()
    assert received == [status]
    assert int(headers["Content-Length"]) == len(raw)
    assert headers["X-Request-ID"] == "original"
    assert raw.endswith(b"\n")
    assert json.loads(raw) == payload


def test_http_encoding_failure_does_not_publish_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.daemon.http import DaemonAPIHandler

    handler = object.__new__(DaemonAPIHandler)
    headers: list[int] = []
    monkeypatch.setattr(handler, "send_response", headers.append)
    with pytest.raises(TypeError):
        handler._send_daemon_operation({"outcome": "completed", "result": object()})
    assert headers == []
