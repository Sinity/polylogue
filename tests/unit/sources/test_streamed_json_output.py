"""Streamed JSON output preserves the actual encoders and private publication."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.json import JSONValue, dumps_bytes
from polylogue.schemas import observation_spill
from polylogue.schemas.observation_spill import SpilledKey, StreamedJSONDocument, _load_node, _ScalarTokenStore
from polylogue.sources import streamed_json_output
from polylogue.sources.streamed_json_output import write_streamed_json
from polylogue.storage.sqlite.connection_profile import scratch_connection_context


@pytest.mark.parametrize("member_format", [False, True])
@pytest.mark.parametrize(
    "raw",
    [
        '{"z": [1e2, 1.00, -0, -0.0, 1e30, 5e-324, true, false, null], "a": {"empty": "", "escapes": "é\\n\\t\\\\\\""}}',
        '{"duplicate": 1, "duplicate": 2, "emoji": "\\ud83d\\ude00"}',
        '{"surrogate\\ud800": "lone\\udc00"}',
        '{"nonfinite": [NaN, Infinity, -Infinity]}',
    ],
)
def test_borrowed_tree_bytes_and_errors_match_original_encoder(tmp_path: Path, raw: str, member_format: bool) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(raw.encode("utf-8"))
    value = json.loads(raw)
    target = tmp_path / "output.json"
    try:
        expected = json.dumps(value, ensure_ascii=True).encode() if member_format else dumps_bytes(value)
    except (ValueError, UnicodeError, TypeError) as error:
        # Nonfinite and surrogate provider input is accepted by the same tape
        # policy as bundle records, before the output codec decides its result.
        owner = StreamedJSONDocument(None)
        with owner:
            node = owner.append_document(source, allow_nonfinite=True, provider_utf8=True)
            document = _load_node(owner.connection, node)
            with pytest.raises(type(error)):
                write_streamed_json(document, target, member_format=member_format)
        assert not target.exists()
        with pytest.raises(type(error)):
            write_streamed_json(value, target, member_format=member_format)
    else:
        owner = StreamedJSONDocument(None)
        with owner:
            node = owner.append_document(source, allow_nonfinite=True, provider_utf8=True)
            document = _load_node(owner.connection, node)
            write_streamed_json(document, target, member_format=member_format)
        assert target.read_bytes() == expected
        write_streamed_json(value, target, member_format=member_format)
        assert target.read_bytes() == expected
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("member_format", [False, True])
def test_large_key_and_scalar_encode_without_selected_reads_under_cell_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member_format: bool
) -> None:
    original = scratch_connection_context

    @contextlib.contextmanager
    def small_cells(**kwargs: Any) -> Generator[sqlite3.Connection, None, None]:
        with original(**kwargs) as connection:
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 32768)
            yield connection

    monkeypatch.setattr(observation_spill, "scratch_connection_context", small_cells)
    value = {('é\\"\n' * 40000): ['x\\"\t😀' * 40000, {"nested": "complete"}]}
    source = tmp_path / "large.json"
    source.write_text(json.dumps(value, ensure_ascii=True))
    expected = json.dumps(value, ensure_ascii=True).encode() if member_format else dumps_bytes(value)
    original_read = _ScalarTokenStore.read

    def selected(self: _ScalarTokenStore, kind: str, ordinal: int) -> JSONValue:
        size = self.connection.execute(
            "SELECT decoded_bytes FROM json_scalar_tokens WHERE kind=? AND token=?", (kind, ordinal)
        ).fetchone()[0]
        assert size < 32768, "output reconstructed a large scalar"
        return original_read(self, kind, ordinal)

    def no_key_read(self: SpilledKey) -> str:
        raise AssertionError("output reconstructed a key")

    monkeypatch.setattr(_ScalarTokenStore, "read", selected)
    monkeypatch.setattr(SpilledKey, "read", no_key_read)
    target = tmp_path / "output.json"
    with StreamedJSONDocument(source) as document:
        write_streamed_json(document, target, member_format=member_format)
    assert target.read_bytes() == expected


def test_cancelled_output_retains_existing_destination_and_removes_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "output.json"
    target.write_bytes(b"previous complete output")
    count = 0

    def cancel() -> None:
        nonlocal count
        count += 1
        if count == 4:
            raise InterruptedError("neutral cancellation")

    monkeypatch.setattr(streamed_json_output, "check_compute_cancelled", cancel)
    with pytest.raises(InterruptedError):
        write_streamed_json({"large": "x" * 100000}, target)
    assert target.read_bytes() == b"previous complete output"
    assert list(tmp_path.iterdir()) == [target]


def test_prepared_jsonl_bundle_uses_complete_streamed_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.core.enums import Provider
    from polylogue.pipeline.ids import session_content_hash
    from polylogue.sources import decoder_json, prepared_jsonl
    from polylogue.sources.dispatch import parse_payload

    value = {
        "uuid": "neutral-session",
        "name": "Neutral session",
        "chat_messages": [{"uuid": "neutral-message", "sender": "human", "text": "Complete neutral prompt"}],
        "unused": 'é\\"\n' * 40000,
    }
    source = tmp_path / "sessions.jsonl"
    source.write_text(json.dumps(value, ensure_ascii=True) + "\n")
    (expected,) = parse_payload(Provider.CLAUDE_AI, [value], "fallback")
    expected_hash = session_content_hash(expected)
    monkeypatch.setattr(decoder_json, "_JSONL_MEMORY_BYTES", 1)
    original = write_streamed_json
    observed = []

    def member(record: object, destination: Path, *, member_format: bool = False) -> None:
        assert member_format
        original(record, destination, member_format=member_format)
        observed.append(destination.read_bytes())

    monkeypatch.setattr(prepared_jsonl, "write_streamed_json", member)
    artifact = prepared_jsonl.prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_AI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    (actual,) = artifact.iter_sessions()
    assert actual.content_hash == expected_hash
    assert observed == [json.dumps(value, ensure_ascii=True).encode()]
    artifact.close()
