"""Exact scalar transport keeps unused values outside the Python tree."""

from __future__ import annotations

import contextlib
import gc
import json
import math
import sqlite3
import tracemalloc
from pathlib import Path

import pytest
from ijson import JSONError

from polylogue.core.enums import Provider, ValidationMode
from polylogue.schemas import observation_spill
from polylogue.schemas.observation_spill import StreamedJSONDocument, _ScalarTokenStore
from polylogue.schemas.retained_validation import _SampleValidationReducer, validate_retained_document
from polylogue.schemas.runtime_registry import SchemaRegistry


def _small_sqlite_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    original = observation_spill.scratch_connection_context

    @contextlib.contextmanager
    def scratch(**kwargs: object):
        with original(**kwargs) as connection:
            # A real cell limit, restricted to this controlled scratch owner.
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 32768)
            yield connection

    monkeypatch.setattr(observation_spill, "scratch_connection_context", scratch)


def _record_stream(path: Path, fixture: Path, size: int) -> None:
    with path.open("wb") as output:
        for line in fixture.read_bytes().splitlines():
            output.write(line.rstrip()[:-1] + b',"neutral_unused":"')
            for _ in range(size // 1024):
                output.write(b"x" * 1024)
            output.write(b'"}\n')


@pytest.mark.parametrize("provider", ["hermes", "claude-code", "codex"])
def test_actual_package_resolution_keeps_unused_four_mib_scalar_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    _small_sqlite_cells(monkeypatch)
    source = Path(__file__).resolve().parents[2] / "fixtures" / "origin-capability" / f"{provider}-session.jsonl"
    target = tmp_path / "input.jsonl"
    original_read = _ScalarTokenStore.read

    def selected_only(self: _ScalarTokenStore, kind: str, ordinal: int):
        row = self.connection.execute(
            "SELECT decoded_bytes FROM json_scalar_tokens WHERE kind=? AND token=?", (kind, ordinal)
        ).fetchone()
        assert row is None or row[0] < 4 * 1024 * 1024, "unused scalar was materialized"
        return original_read(self, kind, ordinal)

    monkeypatch.setattr(_ScalarTokenStore, "read", selected_only)
    verdicts = []
    for size in (1024, 4 * 1024 * 1024):
        _record_stream(target, source, size)
        verdicts.append(
            validate_retained_document(
                provider,
                target,
                mode=ValidationMode.ADVISORY,
                raw_id="neutral",
                revision_sha256="neutral",
                evidence_id="neutral",
                jsonl=True,
                registry=SchemaRegistry(storage_root=tmp_path / "registry"),
            )
        )
    assert verdicts[0] == verdicts[1]
    assert verdicts[1].sample_count > 0
    assert verdicts[1].invalid_count == 0


def test_selected_scalar_values_match_stdlib_at_chunk_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_sqlite_cells(monkeypatch)
    selected = ["é", "\ud800", "\ud83d\ude00", 'slash\\quote"line\n', "é" * 100000, 1, -0.0, 1.5, None, True]
    path = tmp_path / "selected.json"
    text = json.dumps({"values": selected}, ensure_ascii=True)
    path.write_text(text)
    with StreamedJSONDocument(path) as document:
        assert list(document["values"]) == json.loads(text)["values"]
    # Direct provider UTF-8 surrogate code units also survive the exact sink.
    path.write_bytes(b'{"value":"' + "\ud800".encode("utf-8", "surrogatepass") + b'"}')
    with StreamedJSONDocument(path) as document:
        assert document["value"] == "\ud800"


def test_nonfinite_number_dialect_and_long_float_match_stdlib(tmp_path: Path) -> None:
    path = tmp_path / "numbers.json"
    text = "[NaN,Infinity,-Infinity,-0.0,1.0,0." + "0" * 100000 + "1]"
    path.write_text(text)
    expected = json.loads(text)
    with StreamedJSONDocument(path) as document:
        assert math.isnan(document[0])
        assert list(document[1:]) == expected[1:]


@pytest.mark.parametrize("suffix", [b"\\q", b"\x00", b"\\u123", b"\xc3"])
def test_invalid_string_suffix_is_validated_after_transport_chunks(tmp_path: Path, suffix: bytes) -> None:
    path = tmp_path / "invalid.json"
    with path.open("wb") as output:
        output.write(b'{"unknown":"' + b"x" * 100000 + suffix + b'"}')
    with pytest.raises((ValueError, UnicodeDecodeError, JSONError)):
        with StreamedJSONDocument(path):
            pass


def test_boolean_additional_and_drift_do_not_request_unknown_value(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    path.write_text('{"known":1,"unknown":"' + "x" * 100000 + '"}')
    with StreamedJSONDocument(path) as document:
        for additional, invalid in ((True, 0), (False, 1)):
            schema = {
                "type": "object",
                "properties": {"known": {"type": "integer"}},
                "additionalProperties": additional,
            }
            reducer = _SampleValidationReducer(schema, Provider.HERMES, None, document._connection, source_path=None)
            reducer.observe(document)
            assert reducer.invalid_count == invalid
            assert reducer.drift_count == 1


def test_unused_scalar_decode_memory_does_not_follow_token_length(tmp_path: Path) -> None:
    peaks = []
    for size in (1024 * 1024, 4 * 1024 * 1024):
        path = tmp_path / "memory.json"
        with path.open("wb") as output:
            output.write(b'{"unknown":"')
            for _ in range(size // 1024):
                output.write(b"x" * 1024)
            output.write(b'"}')
        gc.collect()
        tracemalloc.start()
        try:
            with StreamedJSONDocument(path) as document:
                assert document.structure_value("unknown") == ""
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    assert peaks[1] < 2 * peaks[0]


def test_codex_package_recognition_does_not_copy_unknown_payload_scalar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_sqlite_cells(monkeypatch)
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "origin-capability" / "codex-session.jsonl"
    target = tmp_path / "nested.jsonl"
    with target.open("wb") as output:
        for line in fixture.read_bytes().splitlines():
            small = json.loads(line)
            small["payload"]["neutral_unused"] = "neutral_scalar_placeholder"
            before, after = json.dumps(small).encode().split(b'"neutral_scalar_placeholder"')
            output.write(before + b'"')
            for _ in range(4096):
                output.write(b"x" * 1024)
            output.write(b'"' + after + b"\n")
    original = _ScalarTokenStore.read

    def selected_only(self: _ScalarTokenStore, kind: str, ordinal: int):
        row = self.connection.execute(
            "SELECT decoded_bytes FROM json_scalar_tokens WHERE kind=? AND token=?", (kind, ordinal)
        ).fetchone()
        assert row is None or row[0] < 4 * 1024 * 1024, "recognition copied an unknown payload member"
        return original(self, kind, ordinal)

    monkeypatch.setattr(_ScalarTokenStore, "read", selected_only)
    verdict = validate_retained_document(
        "codex",
        target,
        mode=ValidationMode.ADVISORY,
        raw_id="neutral",
        revision_sha256="neutral",
        evidence_id="neutral",
        jsonl=True,
        registry=SchemaRegistry(storage_root=tmp_path / "registry"),
    )
    assert verdict.sample_count == 3
    assert verdict.invalid_count == 0


def test_selected_scalar_read_cancellation_closes_the_chunk_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cancel.json"
    path.write_text('{"selected":"' + "x" * 100000 + '"}')
    owner = StreamedJSONDocument(path)
    with owner as document:

        def cancelled() -> None:
            raise RuntimeError("neutral scalar cancellation")

        with monkeypatch.context() as patch:
            patch.setattr(observation_spill, "check_compute_cancelled", cancelled)
            with pytest.raises(RuntimeError, match="neutral scalar cancellation"):
                document["selected"]
        assert owner.connection.execute("SELECT 1").fetchone()[0] == 1
        assert len(document["selected"]) == 100000


def test_selected_scalar_transport_preserves_lowering_and_content_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.pipeline.ids import session_content_hash
    from polylogue.sources.dispatch import detect_provider_evidence, parse_payload

    _small_sqlite_cells(monkeypatch)
    path = tmp_path / "hash.json"
    selected = {
        "session_id": "neutral",
        "platform": "hermes",
        "messages": [{"id": "neutral-message", "role": "user", "content": "cafe\u0301" * 20000}],
    }
    with path.open("wb") as output:
        output.write(json.dumps(selected).encode()[:-1] + b',"neutral_unused":"')
        for _ in range(4096):
            output.write(b"x" * 1024)
        output.write(b'"}')
    eager = json.loads(path.read_bytes())
    expected = parse_payload(Provider.HERMES, eager, "neutral-fallback")
    with StreamedJSONDocument(path) as document:
        assert detect_provider_evidence(document) == detect_provider_evidence(eager)
        actual = parse_payload(Provider.HERMES, document, "neutral-fallback")
        assert [session_content_hash(session) for session in actual] == [
            session_content_hash(session) for session in expected
        ]
        assert actual[0].messages[0].text == expected[0].messages[0].text
