"""The record strategy changes storage without changing decoder values."""

from __future__ import annotations

import contextlib
import io
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.schemas import observation_spill
from polylogue.schemas.observation_spill import _ScalarTokenStore
from polylogue.sources import decoder_json
from polylogue.sources.decoder_json import DecodedRecordSequence, JsonlDecodeError, _iter_jsonl_stream
from polylogue.sources.decoders import logger


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2,"flag":true,"none":null}',
        b'{"x":NaN,"positive":Infinity,"negative":-Infinity}',
        b'{"x":"\xed\xa0\x80\\udc00"}',
        b'{"x":"\\ud800\xed\xb0\x80"}',
        b'{"x":"\xed\xa0\x80\xed\xb0\x80"}',
        b'\xef\xbb\xbf\xef\xbb\xbf{"x":"cleaned BOM"}',
        b'{"x":"cleaned\x00value"}',
        '{"x":"é"}'.encode("utf-16"),
        '{"x":"é"}'.encode("utf-32"),
        b' \v\f{"x":1}\v\f ',
        b'{"broken":}',
        b'{"x":1}{"x":2}',
        b'{"x":"bad\xff"}',
    ],
)
def test_disk_and_memory_strategies_preserve_legacy_decode_values(monkeypatch: pytest.MonkeyPatch, raw: bytes) -> None:
    expected = list(_iter_jsonl_stream(logger, io.BytesIO(raw), "neutral.jsonl"))
    for threshold in (1, 64 * 1024):
        monkeypatch.setattr(decoder_json, "_JSONL_MEMORY_BYTES", threshold)
        with contextlib.closing(DecodedRecordSequence.from_jsonl(io.BytesIO(raw), "neutral.jsonl")) as records:
            # Streaming encoding uses the Mapping interface; one-shot C
            # encoding deliberately reads dict internals directly.
            encoder = json.JSONEncoder()
            assert "".join(encoder.iterencode(list(records))) == "".join(encoder.iterencode(expected))


def test_large_jsonl_tape_owns_unknown_scalar_until_final_parser_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_scratch = observation_spill.scratch_connection_context

    @contextlib.contextmanager
    def small_cells(**kwargs):
        with original_scratch(**kwargs) as connection:
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 32768)
            yield connection

    monkeypatch.setattr(observation_spill, "scratch_connection_context", small_cells)
    original_read = _ScalarTokenStore.read

    def selected(self, kind, ordinal):
        row = self.connection.execute(
            "SELECT decoded_bytes FROM json_scalar_tokens WHERE kind=? AND token=?", (kind, ordinal)
        ).fetchone()
        assert row is None or row[0] < 4 * 1024 * 1024
        return original_read(self, kind, ordinal)

    monkeypatch.setattr(_ScalarTokenStore, "read", selected)
    source = tmp_path / "records.jsonl"
    with source.open("wb") as output:
        for index in range(2):
            output.write(b'{"selected":' + str(index).encode() + b',"unknown":"')
            for _ in range(4096):
                output.write(b"x" * 1024)
            output.write(b'"}\n')
    with (
        source.open("rb") as handle,
        contextlib.closing(DecodedRecordSequence.from_jsonl(handle, "neutral.jsonl")) as tape,
    ):
        # A parser may exhaust its input and only then finish its session.
        retained = list(tape)
        assert [record["selected"] for record in retained] == [0, 1]
        assert tape[1]["selected"] == 1
        assert [record["selected"] for record in tape] == [0, 1]
    with pytest.raises(RuntimeError, match="closed"):
        len(tape)


def test_record_retry_failure_preserves_physical_error_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(decoder_json, "_JSONL_MEMORY_BYTES", 1)
    raw = b'\n{"selected":1}\n\n{"broken":}\n{"selected":2}\n'
    with contextlib.closing(DecodedRecordSequence.from_jsonl(io.BytesIO(raw), "neutral.jsonl")) as tape:
        assert [record["selected"] for record in tape] == [1, 2]
    with pytest.raises(JsonlDecodeError) as failure:
        DecodedRecordSequence.from_jsonl(io.BytesIO(raw), "neutral.jsonl", fail_on_decode_error=True)
    assert failure.value.line_number == 4


@pytest.mark.parametrize("provider", ["hermes", "claude-code", "codex"])
@pytest.mark.parametrize("stream", [False, True])
def test_production_preparation_keeps_unselected_scalar_spilled_and_preserves_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str, stream: bool
) -> None:
    from polylogue.sources.prepared_jsonl import prepare_jsonl_blob

    original_scratch = observation_spill.scratch_connection_context

    @contextlib.contextmanager
    def small_cells(**kwargs):
        with original_scratch(**kwargs) as connection:
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 32768)
            yield connection

    monkeypatch.setattr(observation_spill, "scratch_connection_context", small_cells)
    original_read = _ScalarTokenStore.read

    def selected(self, kind, ordinal):
        row = self.connection.execute(
            "SELECT decoded_bytes FROM json_scalar_tokens WHERE kind=? AND token=?", (kind, ordinal)
        ).fetchone()
        if row is not None and row[0] >= 4 * 1024 * 1024:
            import traceback

            raise AssertionError(
                "unused scalar was selected: "
                + " / ".join(
                    f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
                    for frame in traceback.extract_stack()[-9:-1]
                )
            )
        return original_read(self, kind, ordinal)

    monkeypatch.setattr(_ScalarTokenStore, "read", selected)
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "origin-capability" / f"{provider}-session.jsonl"
    source = tmp_path / "input.jsonl"
    outcomes = []
    for size in (1024, 4 * 1024 * 1024):
        with source.open("wb") as output:
            for line in fixture.read_bytes().splitlines():
                output.write(line[:-1] + b',"neutral_unused":"')
                for _ in range(size // 1024):
                    output.write(b"x" * 1024)
                output.write(b'"}\n')
        artifact = prepare_jsonl_blob(
            str(source),
            str(source),
            provider,
            "neutral",
            is_stream=stream,
            shard_directory=str(tmp_path / f"prepared-{size}"),
            strict_jsonl_records=True,
        )
        try:
            assert artifact.error is None, artifact.error
            sessions = artifact.session_sequence()
            assert len(sessions) > 0
            outcomes.append([(session.provider_session_id, session.content_hash) for session in sessions])
        finally:
            artifact.discard()
    assert outcomes[0] == outcomes[1]


def test_codex_replay_spool_borrows_lazy_nodes_without_reading_unknown_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources.parsers import codex

    monkeypatch.setattr(codex, "_CODEX_REPLAY_MEMORY_BUDGET_BYTES", 0)
    test_production_preparation_keeps_unselected_scalar_spilled_and_preserves_hash(tmp_path, monkeypatch, "codex", True)


@pytest.mark.parametrize("content", [[1], [None], ["wrong"], [[{}]], [{"text": "neutral"}], [{"type": 1}]])
def test_codex_recognition_preserves_content_union_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content
) -> None:
    from polylogue.sources.parsers.codex import _validate_record

    payload = {"type": "message", "role": "user", "content": content}
    expected = _validate_record(payload, index=1) is not None
    monkeypatch.setattr(decoder_json, "_JSONL_MEMORY_BYTES", 1)
    raw = json.dumps(payload).encode()
    with contextlib.closing(DecodedRecordSequence.from_jsonl(io.BytesIO(raw), "neutral.jsonl")) as records:
        assert (_validate_record(records[0], index=1) is not None) == expected


def test_codex_direct_content_unknown_scalar_keeps_original_nested_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources.parsers.codex import _validate_record

    original_read = _ScalarTokenStore.read

    def selected(self, kind, ordinal):
        row = self.connection.execute(
            "SELECT decoded_bytes FROM json_scalar_tokens WHERE kind=? AND token=?", (kind, ordinal)
        ).fetchone()
        assert row is None or row[0] < 4 * 1024 * 1024
        return original_read(self, kind, ordinal)

    monkeypatch.setattr(_ScalarTokenStore, "read", selected)
    path = tmp_path / "direct.jsonl"
    with path.open("wb") as output:
        output.write(b'{"type":"message","role":"user","content":[{"type":"input_text","text":"neutral","unknown":"')
        for _ in range(4096):
            output.write(b"x" * 1024)
        output.write(b'"}]}')
    with (
        path.open("rb") as source,
        contextlib.closing(DecodedRecordSequence.from_jsonl(source, "direct.jsonl")) as tape,
    ):
        assert _validate_record(tape[0], index=1) is not None
        assert tape[0]["content"][0]["text"] == "neutral"


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}',
        b'{"x":"\xed\xa0\x80\\udc00"}',
        b'{"x":"\\ud800\xed\xb0\x80"}',
        b'\xef\xbb\xbf\xef\xbb\xbf{"x":1}',
        b"\xef\xbb\xbf   ",
        b'{"x":NaN}',
        b'\v{"x":1}\f',
        b'{"x":"bad\xff"}',
        b'{"x":1}{"x":2}',
        '{"x":"é"}'.encode("utf-16"),
    ],
)
def test_recognition_strategies_preserve_strict_decoder_contract(monkeypatch: pytest.MonkeyPatch, raw: bytes) -> None:
    from polylogue.core.json import JSONDecodeError, decode_provider_utf8, loads
    from polylogue.sources.detection_projection import iter_decoded_jsonl_records

    def refuse(_value: str):
        raise ValueError("non-finite JSON constant")

    cleaned = raw.strip(b" \t\r\n")
    if cleaned.startswith(b"\xef\xbb\xbf"):
        cleaned = cleaned[3:].lstrip(b" \t\r\n")
    expected = []
    refused_expected = False
    if cleaned:
        try:
            expected = [loads(cleaned)]
        except JSONDecodeError:
            try:
                expected = [json.loads(decode_provider_utf8(cleaned), parse_constant=refuse)]
            except (UnicodeError, ValueError):
                refused_expected = True
    for threshold in (1, 64 * 1024):
        monkeypatch.setattr(decoder_json, "_JSONL_MEMORY_BYTES", threshold)
        failures = []
        encoder = json.JSONEncoder()
        actual = [
            "".join(encoder.iterencode(record))
            for record in iter_decoded_jsonl_records(io.BytesIO(raw), on_decode_failure=failures.append)
        ]
        assert bool(failures) == refused_expected
        assert actual == ["".join(encoder.iterencode(record)) for record in expected]


def test_recognition_exposes_valid_prefix_before_late_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import ijson

    from polylogue.sources.detection_projection import iter_decoded_jsonl_records

    monkeypatch.setattr(decoder_json, "_JSONL_MEMORY_BYTES", 1)
    records = iter_decoded_jsonl_records(io.BytesIO(b'{"selected":1}\n{"broken":}\n'))
    try:
        assert next(records)["selected"] == 1
        with pytest.raises(ijson.JSONError, match="line 2"):
            next(records)
    finally:
        records.close()
