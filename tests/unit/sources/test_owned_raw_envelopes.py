"""Raw envelopes retain exact archival decoding through the caller's lifetime."""

from __future__ import annotations

import contextlib
import io
import json
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.raw_payload.decode import _decode_jsonl_payload, build_raw_payload_envelope
from polylogue.core.json import JSONValue, decode_provider_utf8, dumps_bytes, is_json_value
from polylogue.schemas import observation_spill
from polylogue.schemas.observation_spill import _ScalarTokenStore
from polylogue.sources import decoder_json
from polylogue.sources.decoder_json import DecodedRecordSequence
from polylogue.storage.sqlite.connection_profile import scratch_connection_context


def _oracle(raw: bytes | str) -> tuple[list[object], int]:
    records: list[object] = []
    malformed = 0
    first = True
    stream = io.BytesIO(raw) if isinstance(raw, bytes) else io.StringIO(raw)
    for line in stream:
        try:
            text = decode_provider_utf8(line) if isinstance(line, bytes) else line
        except UnicodeError:
            malformed += 1
            continue
        if first:
            text = text.lstrip("\ufeff")
            first = False
        text = text.strip()
        if not text:
            continue
        try:
            records.append(json.loads(text))
        except ValueError:
            malformed += 1
    return records, malformed


@pytest.mark.parametrize(
    "raw",
    [
        b'\xff\n\xef\xbb\xbf\xef\xbb\xbf{"a":1}\n\xef\xbb\xbf{"a":2}\n',
        b'\n\xef\xbb\xbf\xef\xbb\xbf{"a":1}\n{"a":2}\n',
        '\u2003{"a":1}\u2003\n{"a":2}\n',
        '{"a":"\ud800\udc00"}\n{"a":"\ud800\\udc00"}\n',
        b'{"a":"\xed\xa0\x80\\udc00"}\n{"a":"\\ud800\xed\xb0\x80"}\n',
        b'{"a":NaN}\n{"a":Infinity}\n{"a":-Infinity}\n{"a":}\n',
        b'{"a":1,"a":2}\nfalse\n[1,2]\n',
    ],
)
def test_archive_disk_and_memory_policies_match(raw: bytes | str, monkeypatch: pytest.MonkeyPatch) -> None:
    expected, failures = _oracle(raw)
    for threshold in (1, 64 * 1024):
        monkeypatch.setattr(decoder_json, "_JSONL_MEMORY_BYTES", threshold)
        tape, malformed, detail = _decode_jsonl_payload(raw)
        with contextlib.closing(tape):
            assert is_json_value(tape)
            encoder = json.JSONEncoder()
            assert "".join(encoder.iterencode(tape)) == "".join(encoder.iterencode(expected))
            assert malformed == failures
            assert (detail is None) == (failures == 0)


def test_raw_tape_is_read_only_and_core_serializer_uses_records() -> None:
    tape = DecodedRecordSequence([{"a": 1}, {"a": 2}])
    with contextlib.closing(tape):
        assert tape == [{"a": 1}, {"a": 2}]
        assert tape != [] and bool(tape) and {"a": 1} in tape
        assert dumps_bytes(tape) == b'[{"a":1},{"a":2}]'
        for method, args in [
            ("append", ({"a": 3},)),
            ("clear", ()),
            ("extend", ([],)),
            ("insert", (0, {})),
            ("pop", ()),
            ("remove", ({},)),
            ("reverse", ()),
            ("sort", ()),
            ("__setitem__", (0, {})),
            ("__delitem__", (0,)),
            ("__iadd__", ([],)),
            ("__imul__", (2,)),
        ]:
            with pytest.raises(TypeError):
                getattr(tape, method)(*args)
    with pytest.raises(RuntimeError):
        len(tape)


def test_envelope_owner_closes_large_record_after_taxonomy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_scratch = scratch_connection_context

    @contextlib.contextmanager
    def small_cells(**kwargs: Any) -> Generator[sqlite3.Connection, None, None]:
        with original_scratch(**kwargs) as connection:
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 32768)
            yield connection

    monkeypatch.setattr(observation_spill, "scratch_connection_context", small_cells)
    original_read = _ScalarTokenStore.read

    def selected_only(self: _ScalarTokenStore, kind: str, ordinal: int) -> JSONValue:
        row = self.connection.execute(
            "SELECT decoded_bytes FROM json_scalar_tokens WHERE kind=? AND token=?", (kind, ordinal)
        ).fetchone()
        assert row is None or row[0] < 4 * 1024 * 1024, "unknown archival scalar was materialized"
        return original_read(self, kind, ordinal)

    monkeypatch.setattr(_ScalarTokenStore, "read", selected_only)
    path = tmp_path / "neutral.jsonl"
    with path.open("wb") as output:
        output.write(b'{"type":"session_meta","payload":{"id":"neutral"},"unknown":"')
        for _ in range(4096):
            output.write(b"x" * 1024)
        output.write(
            b'"}\n{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"neutral"}]}}\n'
        )
    with build_raw_payload_envelope(path, source_path=str(path), fallback_provider="codex") as envelope:
        assert envelope.wire_format == "jsonl" and envelope.artifact.schema_eligible
        assert isinstance(envelope.payload, DecodedRecordSequence)
        record = envelope.payload[0]
        assert isinstance(record, dict) and record["payload"] == {"id": "neutral"}
        tape = envelope.payload
    with pytest.raises(RuntimeError):
        tape[0]
