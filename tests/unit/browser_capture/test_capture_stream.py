"""Streamed capture parsing keeps the stdlib decoder's values without retaining large scalars."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json

import pytest

from polylogue.browser_capture import capture_stream
from polylogue.core.json import dumps_bytes


def test_numbers_parse_as_the_stdlib_decoder_reads_them() -> None:
    """A wide integer stays exact and a fraction becomes the stdlib float.

    Anti-vacuity: ijson's float mode reads the integer token through a native
    double, so it either overflows or comes back as an inexact float.
    """
    raw = b"[184467440737095516161234, 1.5, 1e2, -7]"
    numbers = [value for event, value in capture_stream._json_events(io.BytesIO(raw)) if event == "number"]
    assert numbers == json.loads(raw)
    assert [type(value) for value in numbers] == [int, float, float, int]


def test_raw_payload_shape_keeps_only_the_scalars_detection_reads() -> None:
    """Native-payload detection sees container kinds and the bridge marker, nothing else.

    Anti-vacuity: retaining every root scalar keeps the ``padding`` string in
    the shape beside its digest.
    """
    payload = {"padding": "x" * 4096, "polylogue_bridge_projection": "compact", "mapping": {"a": 1}, "items": [1]}
    events = capture_stream._json_events(io.BytesIO(json.dumps(payload).encode()))
    event, value = next(events)
    fold = capture_stream._read_raw_payload(events, event, value)
    assert fold.shape == {"padding": None, "polylogue_bridge_projection": "compact", "mapping": {}, "items": []}


def test_a_string_digest_is_its_whole_encoding_hashed_piecewise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hashing a string in chunks gives the digest of its one-shot JSON encoding.

    Anti-vacuity: a chunk boundary that re-quoted or re-escaped its piece would
    hash different bytes than ``dumps_bytes`` of the whole string.
    """
    monkeypatch.setattr(capture_stream, "_SCALAR_DIGEST_CHUNK_CHARS", 3)
    value = 'a"b\\c\n\t日本語\U0001f600\x01end' * 5
    assert capture_stream._scalar_digest(value) == hashlib.sha256(b"s" + dumps_bytes(value)).digest()
    assert capture_stream._scalar_digest("") == hashlib.sha256(b"s" + dumps_bytes("")).digest()


@pytest.mark.parametrize(
    "carrier",
    [
        base64.b64encode(bytes(range(256)) * 3).decode(),
        "data:image/png;base64," + base64.b64encode(b"png bytes!").decode(),
        base64.b64encode(b"ab").decode(),
        "QQ==QQ==",
        "not base64!",
        "QUJD" + "Q",
        "data:text/plain,QUJD",
    ],
)
def test_a_carrier_digest_matches_a_one_shot_decode(monkeypatch: pytest.MonkeyPatch, carrier: str) -> None:
    """Chunked decoding accepts, refuses and hashes exactly as one ``b64decode`` call.

    Anti-vacuity: decoding without the 4-aligned step, or letting mid-carrier
    padding through a chunk boundary, changes the digest or the verdict.
    """
    monkeypatch.setattr(capture_stream, "_CARRIER_DECODE_CHUNK_CHARS", 8)
    data = carrier.split(";base64,", 1)[1] if carrier.startswith("data:") and ";base64," in carrier else carrier
    try:
        expected: bytes | None = hashlib.sha256(base64.b64decode(data, validate=True)).digest()
    except (ValueError, binascii.Error):
        expected = None
    assert capture_stream.carrier_digest(carrier) == expected
