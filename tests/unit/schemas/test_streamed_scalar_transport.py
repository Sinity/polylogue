"""Exact scalar transport keeps unused values outside the Python tree."""

from __future__ import annotations

import contextlib
import gc
import json
import math
import sqlite3
import tracemalloc
from collections.abc import Generator
from pathlib import Path
from typing import Any, BinaryIO

import pytest
from ijson import JSONError

from polylogue.core.enums import Provider, ValidationMode
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas import observation_spill
from polylogue.schemas.observation_spill import SpilledArray, SpilledObject, StreamedJSONDocument, _ScalarTokenStore
from polylogue.schemas.retained_validation import _SampleValidationReducer, validate_retained_document
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.storage.sqlite.connection_profile import scratch_connection_context


def _small_sqlite_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    original = scratch_connection_context

    @contextlib.contextmanager
    def scratch(**kwargs: Any) -> Generator[sqlite3.Connection, None, None]:
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

    def selected_only(self: _ScalarTokenStore, kind: str, ordinal: int) -> JSONValue:
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
                signature_directory=(target).parent,
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
        assert isinstance(document, SpilledObject)
        values = document["values"]
        assert isinstance(values, list)
        assert list(values) == json.loads(text)["values"]
    # Direct provider UTF-8 surrogate code units also survive the exact sink.
    path.write_bytes(b'{"value":"' + "\ud800".encode("utf-8", "surrogatepass") + b'"}')
    with StreamedJSONDocument(path) as document:
        assert isinstance(document, SpilledObject)
        assert document["value"] == "\ud800"


def test_nonfinite_number_dialect_and_long_float_match_stdlib(tmp_path: Path) -> None:
    path = tmp_path / "numbers.json"
    text = "[NaN,Infinity,-Infinity,-0.0,1.0,0." + "0" * 100000 + "1]"
    path.write_text(text)
    expected = json.loads(text)
    with StreamedJSONDocument(path) as document:
        assert isinstance(document, SpilledArray)
        first = document[0]
        assert isinstance(first, float)
        assert math.isnan(first)
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
        assert isinstance(document, SpilledObject)
        for additional, invalid in ((True, 0), (False, 1)):
            schema = {
                "type": "object",
                "properties": {"known": {"type": "integer"}},
                "additionalProperties": additional,
            }
            reducer = _SampleValidationReducer(
                schema, Provider.HERMES, None, document._connection, source_path=None, signature_directory=tmp_path
            )
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
                assert isinstance(document, SpilledObject)
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

    def selected_only(self: _ScalarTokenStore, kind: str, ordinal: int) -> JSONValue:
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
        signature_directory=(target).parent,
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
        assert isinstance(document, SpilledObject)

        def cancelled() -> None:
            raise RuntimeError("neutral scalar cancellation")

        with monkeypatch.context() as patch:
            patch.setattr(observation_spill, "check_compute_cancelled", cancelled)
            with pytest.raises(RuntimeError, match="neutral scalar cancellation"):
                document["selected"]
        assert owner.connection.execute("SELECT 1").fetchone()[0] == 1
        selected = document["selected"]
        assert isinstance(selected, str)
        assert len(selected) == 100000


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
        assert isinstance(document, SpilledObject)
        assert detect_provider_evidence(document) == detect_provider_evidence(eager)
        actual = parse_payload(Provider.HERMES, document, "neutral-fallback")
        assert [session_content_hash(session) for session in actual] == [
            session_content_hash(session) for session in expected
        ]
        assert actual[0].messages[0].text == expected[0].messages[0].text


@pytest.mark.parametrize(
    "spelling",
    [
        b"\xed\xa0\x80\\udc00",
        b"\\ud800\xed\xb0\x80",
        b"\xed\xa0\x80\xed\xb0\x80",
        b"\\ud800\\udc00",
    ],
)
def test_mixed_surrogate_spellings_match_provider_decoder_at_every_byte_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: bytes
) -> None:
    from polylogue.core.json import decode_provider_utf8

    raw = b'{"value":"' + spelling + b'"}'
    expected = json.loads(decode_provider_utf8(raw))["value"]
    path = tmp_path / "mixed.json"
    path.write_bytes(raw)
    original = observation_spill._ExactJSONText

    class SplitInput:
        def __init__(self, source: BinaryIO, split: int) -> None:
            self.source = source
            self.split = split
            self.first = True

        def read(self, size: int = -1) -> bytes:
            if self.first:
                self.first = False
                return self.source.read(self.split)
            return self.source.read(size)

    for split in range(1, len(raw)):
        monkeypatch.setattr(
            observation_spill,
            "_ExactJSONText",
            lambda source, encoding, split=split, **kwargs: original(SplitInput(source, split), encoding, **kwargs),
        )
        with StreamedJSONDocument(path) as document:
            assert isinstance(document, SpilledObject)
            assert document["value"] == expected


def test_shared_record_tree_rolls_back_failed_attempt_and_keeps_scalar_ordinals(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    owner = StreamedJSONDocument(None)
    with owner as records:
        assert isinstance(records, SpilledArray)
        path.write_text('{"same":"first"}')
        owner.append_document(path)
        node_count = owner.connection.execute("SELECT COUNT(*) FROM json_nodes").fetchone()[0]
        chunk_count = owner.connection.execute("SELECT COUNT(*) FROM json_scalar_chunks").fetchone()[0]
        path.write_text('{"same":"incomplete')
        with pytest.raises(JSONError):
            owner.append_document(path)
        assert len(records) == 1
        assert owner.connection.execute("SELECT COUNT(*) FROM json_nodes").fetchone()[0] == node_count
        assert owner.connection.execute("SELECT COUNT(*) FROM json_scalar_chunks").fetchone()[0] == chunk_count
        path.write_text('{"same":"last","same":"replacement"}')
        owner.append_document(path)
        retained = list(records)
        assert all(isinstance(record, dict) for record in retained)
        assert [record.get("same") for record in retained if isinstance(record, dict)] == ["first", "replacement"]


def test_key_tokens_preserve_exact_identity_order_without_whole_key_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Digest acceleration cannot replace duplicate equality or original-key ordering."""
    from polylogue.schemas.observation_spill import SpilledKey

    _small_sqlite_cells(monkeypatch)
    path = tmp_path / "giant-keys.json"
    key_chars = 128 * 1024
    with path.open("wb") as output:
        output.write(b'{"selected":7,"')
        for _ in range(key_chars // 1024):
            output.write(b"k" * 1024)
        output.write(b'":1,"')
        for _ in range(key_chars // 1024):
            output.write(b"\\u006b" * 1024)
        output.write(b'":2,"')
        for _ in range(key_chars // 1024):
            output.write(b"k" * 1024)
        output.write(b'z":3,"a":4}')
    original_read = SpilledKey.read

    def selected_names_only(self: SpilledKey) -> str:
        row = self.connection.execute("SELECT short_chars FROM json_key_meta WHERE token=?", (self.token,)).fetchone()
        assert row is not None and row[0] <= 128, "giant key was materialized"
        return original_read(self)

    monkeypatch.setattr(SpilledKey, "read", selected_names_only)
    with StreamedJSONDocument(path) as document:
        assert isinstance(document, SpilledObject)
        assert len(document) == 4 and document["selected"] == 7 and "missing" not in document
        entries = list(document.key_entries())
        assert entries[0][0].small_name == "selected"
        assert entries[1][0].small_name is None and entries[2][0].small_name is None
        # Last duplicate wins without moving its first insertion position.
        from polylogue.schemas.observation_spill import _load_node

        assert _load_node(document._connection, entries[1][1]) == 2
        assert entries[1][0].compare(entries[2][0]) < 0
        sorted_entries = list(document.key_entries(sorted_keys=True))
        assert sorted_entries[0][0].small_name == "a"
        assert sorted_entries[-1][0].small_name == "selected"
        assert sorted_entries[1][0].token == entries[1][0].token
        assert sum(map(len, entries[1][0].iter_utf8_chunks())) == key_chars


def test_structural_consumers_keep_giant_original_key_order_without_reading_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.schemas.generation.dynamic_keys import observed_structure_schema
    from polylogue.schemas.shape_fingerprint import fingerprint_parts

    _small_sqlite_cells(monkeypatch)
    giant = "é" * 70000
    value = {giant + "z": {"selected": 1}, "a": [True], giant: "text"}
    path = tmp_path / "keys.json"
    path.write_text(json.dumps(value))
    expected_schema = observed_structure_schema(value)
    expected_fingerprint = "".join(fingerprint_parts(value))
    original = observation_spill.SpilledKey.read

    def selected_only(key: observation_spill.SpilledKey) -> str:
        assert key.small_name is not None, "structural consumer reconstructed a content-sized key"
        return original(key)

    monkeypatch.setattr(observation_spill.SpilledKey, "read", selected_only)
    with StreamedJSONDocument(path) as document:
        assert observed_structure_schema(document) == expected_schema
        assert "".join(fingerprint_parts(document)) == expected_fingerprint


@pytest.mark.parametrize("ending", ["'", '"', "'\"", "\ud800\n\t\x00"])
def test_selected_giant_profile_key_preserves_exact_tokens_identity_and_package_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    from polylogue.schemas.observation_identity import fingerprint_hash, profile_cluster_id
    from polylogue.schemas.observation_spill import profile_token_chunks, profile_token_repr_chunks, profile_token_text
    from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage

    _small_sqlite_cells(monkeypatch)
    key = "b" * 70000 + ending
    payload = {"uuid": "neutral", "chat_messages": [], key: True}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload))
    registry = SchemaRegistry(storage_root=tmp_path / "registry")
    decoded = registry._observed_payloads("claude-ai", payload, source_path="neutral.json")
    expected = tuple(profile_token_text(token) for token in decoded[0].profile_tokens)
    element = SchemaElementManifest(
        element_kind="session_document",
        schema_file="session.schema.json.gz",
        sample_count=1,
        artifact_count=1,
        profile_tokens=["field:" + key],
    )
    package = SchemaVersionPackage(
        provider="claude-ai",
        version="neutral",
        anchor_kind="session_document",
        default_element_kind="session_document",
        first_seen="2026-01-01T00:00:00Z",
        last_seen="2026-01-01T00:00:00Z",
        bundle_scope_count=0,
        sample_count=1,
        elements=[element],
    )
    catalog = SchemaPackageCatalog(
        provider="claude-ai",
        packages=[package],
        default_version="neutral",
        latest_version="neutral",
        recommended_version="neutral",
    )
    monkeypatch.setattr(registry, "load_package_catalog", lambda _provider: catalog)
    original = observation_spill.SpilledKey.read

    def selected_only(token: observation_spill.SpilledKey) -> str:
        assert token.small_name is not None, "profile key was reconstructed"
        return original(token)

    monkeypatch.setattr(observation_spill.SpilledKey, "read", selected_only)
    with registry.observe_stream("claude-ai", path, source_path="neutral.json") as (observations, _cohort):
        tokens = observations[0].profile_tokens
        assert (
            tuple(b"".join(profile_token_chunks(token)).decode("utf-8", "surrogatepass") for token in tokens)
            == expected
        )
        for token, literal in zip(tokens, expected, strict=True):
            assert b"".join(profile_token_repr_chunks(token)) == repr(literal).encode("utf-8")
        assert profile_cluster_id("session_document", tokens) == fingerprint_hash(
            ("session_document", tuple(sorted(expected)))
        )
        result = registry.resolve_observation("claude-ai", observations, source_path="neutral.json")
        assert result is not None and result.reason == "profile_family"
        assert result == registry.resolve_observation("claude-ai", decoded, source_path="neutral.json")


def test_record_profile_field_union_retains_selected_giant_names_on_existing_token_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.schemas.observation_identity import fingerprint_hash, profile_cluster_id
    from polylogue.schemas.observation_runtime import _record_profile_tokens
    from polylogue.schemas.observation_spill import profile_token_chunks, profile_token_text

    _small_sqlite_cells(monkeypatch)
    key = "b" * 70000
    records: list[JSONDocument] = [{"type": "neutral", key: True}, {"type": "neutral", key: 1}]
    expected = tuple(profile_token_text(token) for token in _record_profile_tokens(records, record_type_key="type"))
    path = tmp_path / "records.json"
    path.write_text(json.dumps(records))
    original = observation_spill.SpilledKey.read

    def selected_only(token: observation_spill.SpilledKey) -> str:
        assert token.small_name is not None, "record profile key was reconstructed"
        return original(token)

    monkeypatch.setattr(observation_spill.SpilledKey, "read", selected_only)
    with StreamedJSONDocument(path) as document:
        assert isinstance(document, list)
        samples: list[JSONDocument] = []
        for item in document:
            assert isinstance(item, dict)
            samples.append(item)
        tokens = _record_profile_tokens(samples, record_type_key="type")
        assert (
            tuple(b"".join(profile_token_chunks(token)).decode("utf-8", "surrogatepass") for token in tokens)
            == expected
        )
        assert profile_cluster_id("session_record_stream", tokens) == fingerprint_hash(
            ("session_record_stream", tuple(sorted(expected)))
        )


def test_giant_drift_paths_sort_compare_and_survive_both_scratch_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.schemas.packages import SchemaResolution
    from polylogue.schemas.retained_validation import _stronger_drift
    from polylogue.schemas.validator import _iter_drift_paths

    _small_sqlite_cells(monkeypatch)
    first = "é" * 70000
    later = first + "z"
    payload = {later: True, "rows": [{first: "x" * 70000}], first: 1}
    schema = {
        "type": "object",
        "properties": {"rows": {"type": "array", "items": {"type": "object", "additionalProperties": True}}},
        "additionalProperties": True,
    }
    expected = ",".join(sorted(_iter_drift_paths(payload, schema, "")))
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload))
    original = observation_spill.SpilledKey.read
    original_scalar = _ScalarTokenStore.read

    def selected_only(token: observation_spill.SpilledKey) -> str:
        assert token.small_name is not None, "drift key was reconstructed"
        return original(token)

    def selected_scalar(store: _ScalarTokenStore, kind: str, token: int) -> JSONValue:
        row = store.connection.execute(
            "SELECT decoded_bytes FROM json_scalar_tokens WHERE kind=? AND token=?", (kind, token)
        ).fetchone()
        assert row is None or row[0] < 70000, "unused drift value was reconstructed"
        return original_scalar(store, kind, token)

    monkeypatch.setattr(observation_spill.SpilledKey, "read", selected_only)
    monkeypatch.setattr(_ScalarTokenStore, "read", selected_scalar)
    resolution = SchemaResolution(
        provider="hermes",
        package_version="neutral",
        element_kind="session_document",
        exact_structure_id=None,
        bundle_scope=None,
        reason="exact_structure",
    )
    with StreamedJSONDocument(path) as document:
        assert isinstance(document, SpilledObject)
        with scratch_connection_context(prefix="neutral-drift-", filename="paths.sqlite") as connection:
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 32768)
            reducer = _SampleValidationReducer(
                schema, Provider.HERMES, resolution, connection, source_path=None, signature_directory=tmp_path
            )
            reducer.observe(document)
            assert reducer.invalid_count == 0 and reducer.drift_count == 3
            observation = reducer.strongest
            assert observation is not None
            assert observation.unseen_key_signature.byte_count > 32768
            second_path = tmp_path / "second-drift.json"
            second_key = "\x00" * 70000
            second_path.write_text(json.dumps({second_key: True}))
            with StreamedJSONDocument(second_path) as second_document:
                assert isinstance(second_document, SpilledObject)
                reducer.observe(second_document)
            winner = reducer.strongest
            assert winner is not None and winner is not observation
            assert connection.execute("SELECT COUNT(*) FROM retained_drift_chunks").fetchone()[0] == 0
    # The signature carriers own preparation files, not either closing SQL scratch.
    assert b"".join(observation.unseen_key_signature.iter_utf8_chunks()).decode("utf-8", "surrogatepass") == expected
    assert b"".join(winner.unseen_key_signature.iter_utf8_chunks()).decode("utf-8", "surrogatepass") == second_key
    assert _stronger_drift(observation, winner) is winner
    assert _stronger_drift(winner, observation) is winner
