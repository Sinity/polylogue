"""Complete streamed schema evidence agrees with decoded registry resolution."""

from __future__ import annotations

import gc
import json
import tracemalloc
from pathlib import Path

import pytest
from ijson.common import JSONError

from polylogue.core.json import JSONDocument
from polylogue.schemas.generation.dynamic_keys import (
    legacy_structure_schema_digest,
    observed_structure_schema,
    structure_schema_digest,
)
from polylogue.schemas.observation_identity import bundle_scope_identity, fingerprint_hash, schema_cluster_id
from polylogue.schemas.observation_spill import StreamedJSONDocument
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.schemas.shape_fingerprint import _structure_fingerprint


@pytest.mark.parametrize("witness_kind", ["canonical", "legacy", "cluster", "bundle", "profile"])
def test_streamed_resolution_preserves_witness_aliases_and_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, witness_kind: str
) -> None:
    payload: JSONDocument = {
        "uuid": "neutral-session",
        "chat_messages": [{"sender": "human", "content": [{"type": "text", "text": "neutral"}]}],
        "metadata": {"z": 1, "a": None, "variants": [{"left": True}, {"right": []}]},
        "trackedFileBackups": {"one": {"z": 1, "a": "value"}, "two": {"later": False}},
    }
    schema = observed_structure_schema(payload)
    canonical = structure_schema_digest(schema)
    legacy = legacy_structure_schema_digest(schema)
    assert canonical != legacy
    cluster = schema_cluster_id(payload, "session_document")
    witness = {"canonical": canonical, "legacy": legacy, "cluster": cluster}.get(witness_kind)
    packages = []
    for version in ("v1", "v2"):
        element = SchemaElementManifest(
            element_kind="session_document",
            schema_file="session.schema.json.gz",
            sample_count=1,
            artifact_count=1,
            exact_structure_ids=[witness] if witness else [],
            bundle_scope_identities=[bundle_scope_identity("neutral")] if witness_kind == "bundle" else [],
            profile_tokens=["field:uuid"] if witness_kind == "profile" else [],
        )
        packages.append(
            SchemaVersionPackage(
                provider="claude-ai",
                version=version,
                anchor_kind="session_document",
                default_element_kind="session_document",
                first_seen="2026-01-01T00:00:00Z",
                last_seen="2026-01-01T00:00:00Z",
                bundle_scope_count=1,
                sample_count=1,
                elements=[element],
            )
        )
    catalog = SchemaPackageCatalog(
        provider="claude-ai",
        packages=packages,
        default_version="v2",
        latest_version="v2",
        recommended_version="v2",
    )
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    monkeypatch.setattr(registry, "load_package_catalog", lambda _provider: catalog)
    path = tmp_path / "neutral.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    observations, cohort = registry.observe_stream("claude-ai", path, source_path="neutral.json")
    assert list(observations) == registry._observed_payloads("claude-ai", payload, source_path="neutral.json")
    assert cohort == fingerprint_hash(("session_document", _structure_fingerprint(payload)))
    spill = StreamedJSONDocument(path)
    with spill as lazy_payload:
        shared_context = registry.observe_payload(
            "claude-ai",
            lazy_payload,
            source_path="neutral.json",
            schema_store=spill.store_schema,
        )
        assert spill.connection.execute("SELECT COUNT(*) FROM json_nodes").fetchone()[0] > 0
    assert shared_context == (observations, cohort)
    with pytest.raises(RuntimeError, match="schema spill is closed"):
        _ = spill.connection
    decoded = registry.resolve_payload("claude-ai", payload, source_path="neutral.json")
    streamed = registry.resolve_observation("claude-ai", observations, source_path="neutral.json")
    assert streamed == decoded
    assert streamed is not None
    assert streamed.package_version == "v2"
    assert streamed.reason == {"bundle": "bundle_scope", "profile": "profile_family"}.get(
        witness_kind, "exact_structure"
    )
    if witness:
        assert streamed.exact_structure_id == witness


@pytest.mark.parametrize(
    "text",
    [
        '{"uuid":"s","chat_messages":[],"extra":{"old":true},"extra":{"new":null}}',
        '[{}, {"uuid":"s","chat_messages":[],"field":{"漢字":"\\ud800"}}]',
        '{"uuid":"s","chat_messages":[],"mapping":{"000000000000000000000001":{"z":1},"000000000000000000000002":{"a":2}}}',
    ],
)
def test_streamed_observation_preserves_duplicate_keys_and_document_shapes(tmp_path: Path, text: str) -> None:
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    path = tmp_path / "neutral.json"
    path.write_text(text, encoding="utf-8")
    payload = json.loads(text)
    observations, cohort = registry.observe_stream("claude-ai", path, source_path="neutral.json")
    assert list(observations) == registry._observed_payloads("claude-ai", payload, source_path="neutral.json")
    assert cohort == fingerprint_hash(("session_document", _structure_fingerprint(payload)))


def test_jsonl_stream_exposes_all_roots_as_one_lazy_sequence(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"n":1}\n["two"]\nnull\n', encoding="utf-8")
    spill = StreamedJSONDocument(path, jsonl=True)
    with spill as payload:
        assert isinstance(payload, list)
        assert list(payload) == [{"n": 1}, ["two"], None]
        assert len(payload) == 3

    path.write_text('{"n":1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(JSONError):
        with StreamedJSONDocument(path, jsonl=True):
            pass


def test_streamed_dynamic_keys_and_growing_nested_variants_do_not_accumulate_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    element = SchemaElementManifest(
        element_kind="session_document",
        schema_file="neutral.schema.json.gz",
        sample_count=1,
        artifact_count=1,
        exact_structure_ids=["0" * 64],
    )
    package = SchemaVersionPackage(
        provider="claude-ai",
        version="v1",
        anchor_kind="session_document",
        default_element_kind="session_document",
        first_seen="2026-01-01T00:00:00Z",
        last_seen="2026-01-01T00:00:00Z",
        bundle_scope_count=0,
        sample_count=1,
        elements=[element],
    )
    monkeypatch.setattr(
        registry,
        "load_package_catalog",
        lambda _provider: SchemaPackageCatalog(
            provider="claude-ai",
            packages=[package],
            default_version="v1",
            latest_version="v1",
            recommended_version="v1",
        ),
    )
    peaks = []
    for count in (64, 2048):
        path = tmp_path / f"neutral-{count}.json"
        with path.open("w", encoding="utf-8") as output:
            output.write('{"uuid":"neutral","chat_messages":[],"dynamic":{')
            for index in range(count):
                if index:
                    output.write(",")
                # The final schema grows through distinct nested finite
                # properties, even after the outer dynamic IDs collapse.
                output.write(json.dumps(f"{index:032x}"))
                output.write(":" + json.dumps({f"group{index % 64}": {f"variant{index // 64}": {"value": True}}}))
            output.write("}}")
        gc.collect()
        tracemalloc.start()
        try:
            observations, _cohort = registry.observe_stream("claude-ai", path)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert observations[0].source_witnesses
    # The observer retains indexed disk nodes, not a decoded payload or the
    # growing recursive schema. This margin includes transient SQLite/Python
    # cursor and decoder allocations; a 32x shape increase stays bounded.
    assert peaks[1] < peaks[0] + 2_000_000, peaks


@pytest.mark.parametrize("view", ["keys", "sorted_keys", "items", "array"])
def test_abandoned_lazy_view_propagates_its_cursor_close_failure(tmp_path: Path, view: str) -> None:
    import sqlite3
    import sys
    from collections.abc import Generator
    from typing import Any, cast

    from polylogue.schemas.observation_spill import SpilledArray, SpilledObject, StreamedJSONReadError

    path = tmp_path / "neutral.json"
    path.write_text('{"neutral":[1,2]}')
    observed: list[object] = []
    previous_hook = sys.unraisablehook
    sys.unraisablehook = observed.append
    try:
        with StreamedJSONDocument(path) as document:
            assert isinstance(document, SpilledObject)
            actual = document._connection
            failed_cursors: list[sqlite3.Cursor] = []

            class FailedCloseCursor:
                def __init__(self, cursor: sqlite3.Cursor) -> None:
                    self.cursor = cursor

                def __iter__(self) -> FailedCloseCursor:
                    return self

                def __next__(self) -> Any:
                    return next(self.cursor)

                def close(self) -> None:
                    raise sqlite3.OperationalError("neutral lazy cursor close failure")

            class Connection:
                def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> Any:
                    cursor = actual.execute(sql, parameters)
                    if "ORDER BY" in sql and "json_scalar_chunks" not in sql:
                        failed_cursors.append(cursor)
                        return FailedCloseCursor(cursor)
                    return cursor

            connection = cast(sqlite3.Connection, Connection())
            mapping = SpilledObject(connection, document._node_id)
            if view == "array":
                array = document["neutral"]
                assert isinstance(array, SpilledArray)
                iterator = iter(SpilledArray(connection, array._node_id))
            elif view == "items":
                iterator = iter(mapping.items())
            elif view == "sorted_keys":
                iterator = mapping.sorted_keys()
            else:
                iterator = iter(mapping)
            try:
                next(iterator)
                with pytest.raises(StreamedJSONReadError, match="streamed_json_read_failed"):
                    cast(Generator[object, None, None], iterator).close()
                assert not observed
            finally:
                for cursor in failed_cursors:
                    cursor.close()
    finally:
        sys.unraisablehook = previous_hook
