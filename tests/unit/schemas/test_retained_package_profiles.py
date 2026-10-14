"""Explicit retained units use the original journal/package/profile pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.archive.raw_payload import extract_record_samples_from_raw_content
from polylogue.core.enums import Provider
from polylogue.schemas.generation.provider_bundle import _build_provider_bundle
from polylogue.schemas.generation.workflow import persist_generated_provider_bundle
from polylogue.schemas.observation import extract_schema_units_from_payload, resolve_provider_config
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.selection import select_synthetic_schema


def test_declared_retained_stream_builds_loadable_measured_package_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "retained.jsonl"
    records = [
        {
            "type": "session_meta",
            "payload": {"id": "11111111-1111-4111-8111-111111111111", "timestamp": "2025-01-01T00:00:00Z"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "neutral input"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "neutral output"}],
            },
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in records))
    config = resolve_provider_config(Provider.CODEX)
    samples = extract_record_samples_from_raw_content(source, max_samples=None, record_type_key=config.record_type_key)
    units = extract_schema_units_from_payload(
        samples,
        source_name=Provider.CODEX,
        source_path=source,
        raw_id="neutral-retained-revision",
        observed_at="2025-01-01T00:00:00Z",
        config=config,
        full_corpus=True,
        compact_values=False,
    )
    assert len(units) == 1
    assert not isinstance(units[0].schema_samples, list)
    bundle = _build_provider_bundle(
        "codex",
        db_path=tmp_path / "never-created.db",
        max_samples=None,
        privacy_config=None,
        full_corpus=True,
        observed_units=iter(units),
    )
    assert bundle.result.success, bundle.result.error
    assert bundle.result.sample_count == len(records)
    assert bundle.catalog is not None
    assert bundle.package_workload_profiles
    assert not (tmp_path / "never-created.db").exists()
    registry_path = tmp_path / "packages"
    persist_generated_provider_bundle(registry_path, "codex", bundle)
    selection = select_synthetic_schema("codex", registry_factory=lambda: SchemaRegistry(storage_root=registry_path))
    assert selection.workload_profile is not None
    corpus = SyntheticCorpus.from_selection(selection)

    def refuse_discarded_path(*args: object, **kwargs: object) -> int:
        raise AssertionError("the original header/count joint already owns this record scope")

    monkeypatch.setattr(corpus, "_profile_message_count", refuse_discarded_path)
    try:
        generated = corpus.generate(count=1, messages_per_session=range(3, 4), seed=21)
        decoded = [json.loads(line) for line in generated[0].splitlines()]
        assert len(decoded) == len(records)
        assert decoded[0]["type"] == "session_meta"
        assert decoded[1]["payload"]["role"] == "user"
        assert decoded[2]["payload"]["role"] == "assistant"
    finally:
        corpus._relation_solver.close()


def test_public_source_wrapper_profiles_only_current_stream_and_keeps_historical_structure(tmp_path: Path) -> None:
    from polylogue.schemas.generation.workflow import build_provider_bundle_from_sources
    from polylogue.schemas.source_inference import SchemaSourceInput

    root = tmp_path / "originals"
    root.mkdir()
    native_id = "11111111-1111-4111-8111-111111111111"
    old = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2025-01-01T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "historical_field": True,
                "content": [{"type": "input_text", "text": "neutral historical"}],
            },
        },
    ]
    current = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2025-01-02T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "neutral input"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "neutral output"}],
            },
        },
    ]
    for name, records in (("old.jsonl", old), ("current.jsonl", current)):
        (root / name).write_text("".join(json.dumps(row) + "\n" for row in records))
    bundle = build_provider_bundle_from_sources(
        "codex",
        source_inputs=(SchemaSourceInput("codex", root),),
        cache_path=tmp_path / "private-cache.sqlite3",
        max_workers=1,
        privacy_config=None,
        prior_catalog=None,
    )
    assert bundle.result.success
    assert bundle.result.sample_count == 3
    assert "historical_field" in json.dumps(bundle.result.schema)
    assert bundle.catalog is not None
    package = bundle.catalog.packages[0]
    assert package.workload_profile_file == "workload-profile.json.gz"
    profile = bundle.package_workload_profiles[package.version]
    elements = profile["elements"]
    assert isinstance(elements, dict)
    stream = elements["session_record_stream"]
    assert isinstance(stream, dict)
    assert stream["artifact_count"] == 1
    assert stream["sample_count"] == 3
    registry_path = tmp_path / "registry"
    persist_generated_provider_bundle(registry_path, "codex", bundle)
    selection = select_synthetic_schema("codex", registry_factory=lambda: SchemaRegistry(storage_root=registry_path))
    corpus = SyntheticCorpus.from_selection(selection)
    try:
        sessions = corpus.iter_sessions(count=1, seed=23)
        try:
            generated = next(sessions)
            rows = [json.loads(line) for line in b"".join(generated.chunks).splitlines()]
            assert len(rows) == 3
            assert rows[0]["type"] == "session_meta"
            assert [row["payload"]["role"] for row in rows[1:]] == ["user", "assistant"]
        finally:
            sessions.close()
    finally:
        corpus._relation_solver.close()


def test_selected_source_profile_refuses_input_changed_during_original_observation(tmp_path: Path) -> None:
    import pytest

    from polylogue.schemas.source_inference import SchemaSourceInput, SourceInferenceError, infer_sources

    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"type": "user", "sessionId": "neutral", "message": {"role": "user", "content": "neutral"}}) + "\n"
    )
    reached = []

    def alter_original(unit: object) -> None:
        reached.append(unit)
        source.write_text(source.read_text() + "\n")

    with pytest.raises(SourceInferenceError):
        infer_sources(
            (SchemaSourceInput("claude-code", source),),
            cache_path=tmp_path / "cache.sqlite3",
            max_workers=1,
            observation_recorder=alter_original,
        )
    assert reached


def test_measured_codex_scope_keeps_the_original_header_and_message_wire_family(qualified_schema_registry) -> None:
    from polylogue.sources.dispatch import parse_payload
    from tests.infra.qualified_schema_registry import neutral_source_expectations

    expected = neutral_source_expectations(providers=("codex",))["codex"]
    expected_counts = {item.message_count for item in expected}
    selection = select_synthetic_schema("codex", registry_factory=lambda: qualified_schema_registry)
    corpus = SyntheticCorpus.from_selection(selection)
    forms = set()
    try:
        for raw in corpus.generate(count=12, seed=42):
            records = [json.loads(line) for line in raw.splitlines()]
            envelope = records[0].get("type") == "session_meta"
            forms.add("envelope" if envelope else "flat")
            if envelope:
                assert all(
                    record["type"] == "response_item" and isinstance(record["payload"], dict) for record in records[1:]
                )
            else:
                assert all(record["type"] == "message" and "payload" not in record for record in records[1:])
            sessions = parse_payload("codex", records, "neutral-family")
            assert len(sessions) == 1
            assert len(sessions[0].messages) in expected_counts
        assert forms == {"flat", "envelope"}
    finally:
        corpus._relation_solver.close()


def test_measured_claude_blocks_keep_their_original_body_field_owners(
    qualified_schema_registry: SchemaRegistry,
) -> None:
    from polylogue.core.enums import BlockType
    from polylogue.sources.dispatch import parse_payload

    selection = select_synthetic_schema("claude-code", registry_factory=lambda: qualified_schema_registry)
    corpus = SyntheticCorpus.from_selection(selection)
    observed_tools = 0
    try:
        for ordinal, raw in enumerate(corpus.generate(count=12, seed=7)):
            records = [json.loads(line) for line in raw.splitlines()]
            for record in records:
                content = record["message"]["content"]
                if not isinstance(content, list):
                    continue
                for block in content:
                    kind = block["type"]
                    fields = set(block) & {"text", "thinking", "content", "input"}
                    assert fields == {"text" if kind == "text" else "input" if kind == "tool_use" else "content"}
            sessions = parse_payload("claude-code", records, f"neutral-{ordinal}")
            assert len(sessions) == 1
            assert len(sessions[0].messages) in {2, 4}
            calls = [
                block.tool_id
                for message in sessions[0].messages
                for block in message.blocks
                if block.type is BlockType.TOOL_USE
            ]
            results = [
                block.tool_id
                for message in sessions[0].messages
                for block in message.blocks
                if block.type is BlockType.TOOL_RESULT
            ]
            assert calls == results
            observed_tools += len(calls)
    finally:
        corpus._relation_solver.close()
    assert observed_tools > 0
