"""Source-backed packages publish and resolve observed structure witnesses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.archive.raw_payload import ReplayableRecordSamples
from polylogue.schemas import runtime_registry, source_inference
from polylogue.schemas.generation.dynamic_keys import observed_structure_schema, structure_schema_digest
from polylogue.schemas.generation.models import _ProviderBundle
from polylogue.schemas.generation.workflow import (
    build_provider_bundle_from_sources,
    persist_generated_provider_bundle,
)
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.runtime_registry import _ObservedPayload
from polylogue.schemas.source_inference import SchemaSourceInput


def _gemini_cli_sources(root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    document: dict[str, object] = {
        "sessionId": "document-session",
        "projectHash": "synthetic-project",
        "kind": "main",
        "startTime": "2026-01-01T00:00:00Z",
        "lastUpdated": "2026-01-01T00:01:00Z",
        "messages": [{"id": "document-turn", "type": "user", "content": "original document body"}],
    }
    records: list[dict[str, object]] = [
        {key: value for key, value in document.items() if key not in {"messages", "sessionId"}}
        | {"sessionId": "checkpoint-session"},
        {
            "id": "checkpoint-turn",
            "timestamp": "2026-01-01T00:00:01Z",
            "type": "user",
            "content": "original record body",
        },
    ]
    (root / "document.json").write_text(json.dumps(document), encoding="utf-8")
    (root / "checkpoint.jsonl").write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return document, records


def _with_novel_content(
    document: dict[str, object], records: list[dict[str, object]]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    novel_document = json.loads(json.dumps(document))
    messages = novel_document["messages"]
    assert isinstance(messages, list) and isinstance(messages[0], dict)
    messages[0]["content"] = "novel document content"
    novel_records = json.loads(json.dumps(records))
    novel_records[1]["content"] = "novel record content"
    return novel_document, novel_records


def _build_bundle(root: Path, cache: Path) -> _ProviderBundle:
    return build_provider_bundle_from_sources(
        "gemini-cli",
        source_inputs=(SchemaSourceInput("gemini-cli", root),),
        cache_path=cache,
        max_workers=1,
        privacy_config=None,
        prior_catalog=None,
    )


def test_source_witnesses_survive_cold_and_warm_package_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached source evidence must resolve novel document and stream content by structure."""
    sources = tmp_path / "sources"
    sources.mkdir()
    document, records = _gemini_cli_sources(sources)
    cache = tmp_path / "source-cache.sqlite3"

    cold = _build_bundle(sources, cache)
    assert cold.catalog is not None
    assert cold.package_schemas
    package = cold.catalog.packages[0]
    for element in package.elements:
        schema = cold.package_schemas[package.version][element.element_kind]
        assert element.exact_structure_ids
        assert schema["x-polylogue-exact-structure-ids"] == element.exact_structure_ids
        assert all(len(witness) == 64 and int(witness, 16) >= 0 for witness in element.exact_structure_ids)

    output = tmp_path / "published"
    persist_generated_provider_bundle(output, "gemini-cli", cold)
    fresh_registry = SchemaRegistry(storage_root=output)
    novel_document, novel_records = _with_novel_content(document, records)

    document_resolution = fresh_registry.resolve_payload("gemini-cli", novel_document, source_path="document.json")
    stream_resolution = fresh_registry.resolve_payload("gemini-cli", novel_records, source_path="checkpoint.jsonl")
    assert document_resolution is not None
    assert stream_resolution is not None
    assert document_resolution.reason == stream_resolution.reason == "exact_structure"
    assert document_resolution.element_kind == "session_document"
    assert stream_resolution.element_kind == "session_record_stream"
    assert document_resolution.exact_structure_id == structure_schema_digest(observed_structure_schema(novel_document))
    stream_witnesses = {structure_schema_digest(observed_structure_schema(record)) for record in novel_records}
    stream_element = next(element for element in package.elements if element.element_kind == "session_record_stream")
    assert stream_resolution.exact_structure_id in stream_witnesses
    assert stream_resolution.exact_structure_id in stream_element.exact_structure_ids

    replayable = ReplayableRecordSamples("\n".join(json.dumps(record) for record in novel_records).encode())
    replay_resolution = fresh_registry.resolve_payload("gemini-cli", replayable, source_path="checkpoint.jsonl")
    assert replay_resolution == stream_resolution

    malformed_resolution = fresh_registry.resolve_payload(
        "gemini-cli", [*novel_records, {"unexpected": True}], source_path="checkpoint.jsonl"
    )
    assert malformed_resolution is None or malformed_resolution.reason != "exact_structure"

    monkeypatch.setattr(
        source_inference,
        "_collect_candidate",
        lambda *_args, **_kwargs: pytest.fail("warm source inference collected instead of reusing cache evidence"),
    )
    warm = _build_bundle(sources, cache)
    assert warm.catalog is not None
    assert warm.package_schemas == cold.package_schemas
    assert [element.exact_structure_ids for element in warm.catalog.packages[0].elements] == [
        element.exact_structure_ids for element in package.elements
    ]


def test_legacy_structure_ids_do_not_trigger_source_shape_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 16-character observation route remains available without source witnesses."""
    element = SchemaElementManifest(
        element_kind="session_document",
        schema_file="session_document.schema.json.gz",
        sample_count=1,
        artifact_count=1,
        exact_structure_ids=["0123456789abcdef"],
    )
    package = SchemaVersionPackage(
        provider="chatgpt",
        version="v1",
        anchor_kind="session_document",
        default_element_kind="session_document",
        first_seen="2026-01-01T00:00:00Z",
        last_seen="2026-01-01T00:00:00Z",
        bundle_scope_count=0,
        sample_count=1,
        elements=[element],
    )
    catalog = SchemaPackageCatalog(
        provider="chatgpt",
        packages=[package],
        latest_version="v1",
        default_version="v1",
        recommended_version="v1",
    )
    registry = SchemaRegistry()
    monkeypatch.setattr(registry, "load_package_catalog", lambda _provider: catalog)
    monkeypatch.setattr(
        registry,
        "_observed_payloads",
        lambda _provider, _payload, source_path=None: [
            _ObservedPayload(
                artifact_kind="session_document",
                bundle_scope=None,
                exact_structure_id="0123456789abcdef",
                profile_tokens=(),
                schema_samples=({"unneeded": "shape"},),
            )
        ],
    )
    monkeypatch.setattr(
        runtime_registry,
        "_structure_witnesses",
        lambda _samples: pytest.fail("legacy catalogs must not compute source witnesses"),
    )

    resolution = registry.resolve_payload("chatgpt", {"unneeded": "shape"})

    assert resolution is not None
    assert resolution.reason == "exact_structure"
    assert resolution.exact_structure_id == "0123456789abcdef"
