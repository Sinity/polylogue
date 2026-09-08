"""Source-backed packages publish and resolve observed structure witnesses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.core.json import JSONDocument, json_document
from polylogue.schemas import runtime_registry, source_inference
from polylogue.schemas.generation.dynamic_keys import (
    canonicalize_structure_schema,
    legacy_structure_schema_digest,
    observed_structure_schema,
    retain_exact_structure_witnesses,
    structure_schema_digest,
)
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


def _package(provider: str, version: str, witnesses: list[str], *, family: str = "family") -> SchemaVersionPackage:
    element = SchemaElementManifest(
        element_kind="session_record_stream",
        schema_file="session_record_stream.schema.json.gz",
        sample_count=len(witnesses),
        artifact_count=1,
        exact_structure_ids=witnesses,
    )
    return SchemaVersionPackage(
        provider=provider,
        version=version,
        anchor_kind=element.element_kind,
        default_element_kind=element.element_kind,
        first_seen="2026-01-01T00:00:00Z",
        last_seen="2026-01-01T00:00:00Z",
        bundle_scope_count=0,
        sample_count=len(witnesses),
        anchor_profile_family_id=family,
        elements=[element],
    )


def _catalog(provider: str, packages: list[SchemaVersionPackage]) -> SchemaPackageCatalog:
    return SchemaPackageCatalog(
        provider=provider,
        packages=packages,
        latest_version=packages[-1].version,
        default_version=packages[-1].version,
        recommended_version=packages[-1].version,
    )


def _source_schema(witnesses: list[str]) -> JSONDocument:
    return json_document({"type": "object", "x-polylogue-exact-structure-ids": witnesses})


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


def test_canonical_structure_digest_recurses_only_through_required_arrays() -> None:
    """Nested object and dynamic-map key order cannot split source families."""
    first = {
        "items": [{"beta": {"y": 1, "x": 2}, "alpha": "value"}],
        "dynamic": {"550e8400-e29b-41d4-a716-446655440000": {"second": True, "first": "value"}},
        "ordered": ["first", "second"],
    }
    reordered = {
        "ordered": ["first", "second"],
        "dynamic": {"550e8400-e29b-41d4-a716-446655440000": {"first": "value", "second": True}},
        "items": [{"alpha": "value", "beta": {"x": 2, "y": 1}}],
    }
    first_schema = observed_structure_schema(first)
    reordered_schema = observed_structure_schema(reordered)
    assert structure_schema_digest(first_schema) == structure_schema_digest(reordered_schema)
    assert canonicalize_structure_schema(first_schema) == canonicalize_structure_schema(reordered_schema)
    assert first_schema["properties"] != reordered_schema["properties"]
    assert structure_schema_digest(observed_structure_schema({"value": 1})) != structure_schema_digest(
        observed_structure_schema({"value": "1"})
    )
    assert structure_schema_digest(observed_structure_schema({"value": 1})) != structure_schema_digest(
        observed_structure_schema({})
    )
    assert structure_schema_digest(observed_structure_schema({"value": 1})) != structure_schema_digest(
        observed_structure_schema({"value": 1, "additional": True})
    )


def test_source_exact_matching_covers_every_sample_and_shipped_digest_alias() -> None:
    """A shared header alone cannot select a package for an entire stream."""
    provider = "gemini-cli"
    shared_header = {"type": "header", "metadata": {"z": 1, "a": 2}}
    old_body = {"type": "message", "body": {"old": True}}
    new_body = {"type": "message", "body": {"new": True}}
    old_witnesses = [structure_schema_digest(observed_structure_schema(sample)) for sample in (shared_header, old_body)]
    new_witnesses = [structure_schema_digest(observed_structure_schema(sample)) for sample in (shared_header, new_body)]
    old = _package(provider, "v1", old_witnesses, family="old")
    new = _package(provider, "v2", new_witnesses, family="new")
    registry = SchemaRegistry()

    old_resolution = registry._resolve_observation(
        [old, new],
        _ObservedPayload("session_record_stream", None, None, (), (shared_header, old_body)),
        package_rank={"v1": 0, "v2": 1},
        observation_index=0,
    )
    assert old_resolution is not None
    assert old_resolution.reason == "exact_structure"
    assert old_resolution.resolved.package.version == "v1"

    uncovered = registry._resolve_observation(
        [old, new],
        _ObservedPayload("session_record_stream", None, None, (), (shared_header, old_body, {"type": "extra"})),
        package_rank={"v1": 0, "v2": 1},
        observation_index=0,
    )
    empty = registry._resolve_observation(
        [old, new],
        _ObservedPayload("session_record_stream", None, None, (), ()),
        package_rank={"v1": 0, "v2": 1},
        observation_index=0,
    )
    assert uncovered is None
    assert empty is None

    shipped_sample = {"outer": {"z": 1, "a": 2}}
    old_digest = legacy_structure_schema_digest(observed_structure_schema(shipped_sample))
    assert old_digest != structure_schema_digest(observed_structure_schema(shipped_sample))
    legacy_64 = _package(provider, "v3", [old_digest], family="legacy")
    legacy_resolution = registry._resolve_observation(
        [legacy_64],
        _ObservedPayload("session_record_stream", None, None, (), (shipped_sample,)),
        package_rank={"v3": 0},
        observation_index=0,
    )
    assert legacy_resolution is not None
    assert legacy_resolution.reason == "exact_structure"
    assert legacy_resolution.exact_structure_id == old_digest


def test_same_family_publication_retains_and_reports_source_witnesses(tmp_path: Path) -> None:
    """Refreshes retain old exact matches in both persisted public representations."""
    provider = "synthetic-source-witnesses"
    registry = SchemaRegistry(storage_root=tmp_path)
    old = [f"{1:064x}"]
    current = [f"{2:064x}"]
    first = _package(provider, "v1", old)
    registry.replace_provider_packages(
        provider, _catalog(provider, [first]), {"v1": {"session_record_stream": _source_schema(old)}}
    )
    refreshed = _package(provider, "v1", current)
    registry.replace_provider_packages(
        provider, _catalog(provider, [refreshed]), {"v1": {"session_record_stream": _source_schema(current)}}
    )

    restarted = SchemaRegistry(storage_root=tmp_path)
    package = restarted.load_package_catalog(provider)
    assert package is not None
    element = package.packages[0].element("session_record_stream")
    assert element is not None
    assert element.exact_structure_ids == [*old, *current]
    schema = restarted.get_element_schema(provider, version="v1", element_kind="session_record_stream")
    assert schema is not None
    assert schema["x-polylogue-exact-structure-ids"] == element.exact_structure_ids
    assert schema["x-polylogue-omitted-current-structure-witness-count"] == 0

    saturated = [f"{index:064x}" for index in range(1_024)]
    additions = [f"{index:064x}" for index in (2_000, 1_999, 2_001)]
    retention = retain_exact_structure_witnesses(saturated, additions)
    assert retention.exact_structure_ids == tuple(saturated)
    assert retention.omitted_current_witness_count == 3
    saturated_provider = "synthetic-saturated-source-witnesses"
    saturated_registry = SchemaRegistry(storage_root=tmp_path / "saturated")
    saturated_registry.replace_provider_packages(
        saturated_provider,
        _catalog(saturated_provider, [_package(saturated_provider, "v1", saturated)]),
        {"v1": {"session_record_stream": _source_schema(saturated)}},
    )
    saturated_registry.replace_provider_packages(
        saturated_provider,
        _catalog(saturated_provider, [_package(saturated_provider, "v1", additions)]),
        {"v1": {"session_record_stream": _source_schema(additions)}},
    )
    saturated_catalog = SchemaRegistry(storage_root=tmp_path / "saturated").load_package_catalog(saturated_provider)
    assert saturated_catalog is not None
    saturated_element = saturated_catalog.packages[0].element("session_record_stream")
    assert saturated_element is not None
    assert saturated_element.exact_structure_ids == saturated
    assert saturated_element.omitted_current_structure_witness_count == 3
