"""Complete Boolean-bearing records must not become deferred EOF frontiers."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from polylogue.core.json_envelope import _PrefixStringReader
from polylogue.sources.live.batch_support import (
    jsonl_complete_prefix,
    jsonl_complete_prefix_path,
    jsonl_frontier_of_handle,
    jsonl_parse_prefix_size,
)


@pytest.mark.parametrize("tail", [b'{"flag":true}', b'{"flag":false}', b'{"flags":[true,{"done":false}]}'])
@pytest.mark.parametrize("ending", [b"", b"\n", b"\n \r\n"])
def test_complete_boolean_tail_reaches_physical_eof(tmp_path: Path, tail: bytes, ending: bytes) -> None:
    payload = b'{"first":1}\n' + tail + ending
    path = tmp_path / "session.jsonl"
    path.write_bytes(payload)
    boundary = jsonl_complete_prefix(payload)
    file_boundary = jsonl_complete_prefix_path(path)
    with path.open("rb") as handle:
        opened_boundary = jsonl_frontier_of_handle(handle, len(payload))
    for observed in (boundary, file_boundary, opened_boundary):
        assert observed.prefix_size == len(payload)
        assert not observed.incomplete_tail
        assert not observed.malformed_record
        assert jsonl_parse_prefix_size(observed, len(payload)) is None


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 19])
def test_boolean_literals_and_number_continuations_survive_lexical_chunks(chunk_size: int) -> None:
    payload = b'{"flags":[true,false,{"nested":true}],"numbers":[-1,1e-3,0.2,4E+5]}'
    reader = _PrefixStringReader(io.BytesIO(), syntax_only=True)
    transported = b"".join(reader.feed(payload[at : at + chunk_size]) for at in range(0, len(payload), chunk_size))
    transported += reader.finish()
    decoded = json.loads(transported)
    assert decoded["flags"] == [True, False, {"nested": True}]
    assert len(decoded["numbers"]) == 4


@pytest.mark.parametrize("number", [b"01", b"+1", b"-", b"1e", b"1e+", b"1e--2", b"1.2.3", b"-.1"])
@pytest.mark.parametrize("ending", [b"", b"\n"])
def test_boolean_fix_keeps_invalid_numeric_tails_unaccepted(tmp_path: Path, number: bytes, ending: bytes) -> None:
    prefix = b'{"first":1}\n'
    payload = prefix + b'{"flag":false,"number":' + number + b"}" + ending
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(payload)
    boundary = jsonl_complete_prefix(payload)
    file_boundary = jsonl_complete_prefix_path(path)
    for observed in (boundary, file_boundary):
        assert observed.prefix_size == len(prefix)
        assert observed.incomplete_tail
        assert observed.malformed_record == bool(ending)
        # Finished malformed records stay visible to the strict parser;
        # an unfinished tail exposes only the already-complete prefix.
        assert jsonl_parse_prefix_size(observed, len(payload)) == (None if ending else len(prefix))
