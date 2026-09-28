"""Streamed capture parsing keeps the stdlib decoder's values without retaining large scalars."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
from pathlib import Path

import pytest

from polylogue.browser_capture import capture_decode, capture_stream
from polylogue.core.json import dumps_bytes


def test_numbers_parse_as_the_stdlib_decoder_reads_them() -> None:
    """A wide integer stays exact and a fraction becomes the stdlib float.

    Anti-vacuity: ijson's float mode reads the integer token through a native
    double, so it either overflows or comes back as an inexact float.
    """
    raw = b"[184467440737095516161234, 1.5, 1e2, -7]"
    numbers = [value for event, value in capture_decode._json_events(io.BytesIO(raw)) if event == "number"]
    assert numbers == json.loads(raw)
    assert [type(value) for value in numbers] == [int, float, float, int]


def test_raw_payload_shape_keeps_only_the_scalars_detection_reads() -> None:
    """Native-payload detection sees container kinds and the bridge marker, nothing else.

    Anti-vacuity: retaining every root scalar keeps the ``padding`` string in
    the shape beside its digest.
    """
    payload = {"padding": "x" * 4096, "polylogue_bridge_projection": "compact", "mapping": {"a": 1}, "items": [1]}
    events = capture_decode._json_events(io.BytesIO(json.dumps(payload).encode()))
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
    monkeypatch.setattr(capture_decode, "_CARRIER_DECODE_CHUNK_CHARS", 8)
    data = carrier.split(";base64,", 1)[1] if carrier.startswith("data:") and ";base64," in carrier else carrier
    try:
        expected: bytes | None = hashlib.sha256(base64.b64decode(data, validate=True)).digest()
    except (ValueError, binascii.Error):
        expected = None
    assert capture_stream.carrier_digest(carrier) == expected


def _capture_with_carriers(carriers: dict[str, bytes]) -> dict[str, object]:
    ids = iter(carriers)
    turn_attachment = next(ids)
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {
            "source_url": "https://chatgpt.com/c/conv-spill",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-dom-v1",
            "extension_instance_id": "test-extension-instance",
        },
        "session": {
            "provider": "chatgpt",
            "provider_session_id": "conv-spill",
            "title": "Spill",
            "turns": [
                {
                    "provider_turn_id": "u1",
                    "role": "user",
                    "text": "Here is a file",
                    "attachments": [
                        {
                            "provider_attachment_id": turn_attachment,
                            "name": f"{turn_attachment}.bin",
                            "content_base64": base64.b64encode(carriers[turn_attachment]).decode(),
                        }
                    ],
                },
                {"provider_turn_id": "a1", "role": "assistant", "text": "Received"},
            ],
            "attachments": [
                {
                    "provider_attachment_id": attachment_id,
                    "name": f"{attachment_id}.bin",
                    "inline_base64": "data:application/octet-stream;base64,"
                    + base64.b64encode(carriers[attachment_id]).decode(),
                }
                for attachment_id in ids
            ],
        },
    }


def test_ingest_worker_decodes_a_capture_as_a_stream_and_spills_its_carriers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker never reads a capture whole, and its carriers land in the blob store.

    Anti-vacuity: the ordinary decode calls ``read_bytes`` on the raw blob
    (refused here), and a parse that kept the carriers inline yields
    ``inline_bytes`` instead of a blob reference. Identity parity: each
    attachment's identity payload equals the one a whole-document parse
    derives, so the streamed route cannot mint different attachment ids.
    """
    from polylogue.core.enums import Provider
    from polylogue.pipeline.ids import _attachment_hash_payload
    from polylogue.pipeline.services.ingest_worker import ingest_record
    from polylogue.sources.parsers import browser_capture as browser_capture_parser
    from polylogue.storage.blob_store import BlobStore, reset_blob_store
    from polylogue.storage.runtime import RawSessionRecord

    carriers = {"att-turn": bytes(range(256)) * 9, "att-session": b"session attachment bytes"}
    raw = json.dumps(_capture_with_carriers(carriers)).encode()
    blob_root = tmp_path / "blobs"
    store = BlobStore(blob_root)
    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: blob_root)
    reset_blob_store()
    raw_id, blob_size = store.write_from_bytes(raw)
    raw_blob = store.blob_path(raw_id)
    original_read_bytes = Path.read_bytes

    def refuse_whole_read(self: Path) -> bytes:
        if self == raw_blob:
            raise AssertionError("the capture was read whole")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", refuse_whole_read)
    record = RawSessionRecord(
        raw_id=raw_id,
        source_name="browser-capture",
        source_path=str(tmp_path / "spool" / "chatgpt" / "conv-spill-0123456789ab.json"),
        payload_provider=Provider.CHATGPT,
        source_index=None,
        blob_size=blob_size,
        acquired_at="2026-01-01T00:00:00+00:00",
        file_mtime=None,
    )
    try:
        result = ingest_record(record, str(tmp_path / "archive"), "advisory", blob_root_str=str(blob_root))
    finally:
        reset_blob_store()

    assert result.error is None, result.error
    (payload,) = result.sessions
    attachments = {a.provider_attachment_id: a for a in payload.parsed_session.attachments}
    assert set(attachments) == set(carriers)
    expected = {
        a.provider_attachment_id: a for a in browser_capture_parser.parse(json.loads(raw), "fallback").attachments
    }
    for attachment_id, content in carriers.items():
        attachment = attachments[attachment_id]
        assert attachment.inline_bytes is None
        assert attachment.precomputed_blob == (hashlib.sha256(content).hexdigest(), len(content))
        assert original_read_bytes(store.blob_path(attachment.precomputed_blob[0])) == content
        assert _attachment_hash_payload(attachment) == _attachment_hash_payload(expected[attachment_id])
        assert attachment.size_bytes == expected[attachment_id].size_bytes
        assert attachment.upload_origin == expected[attachment_id].upload_origin
