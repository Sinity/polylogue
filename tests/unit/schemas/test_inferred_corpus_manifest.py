from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.json import JSONValue
from polylogue.scenarios import CorpusSpec
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry
from polylogue.schemas.synthetic.models import SchemaRecord
from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS, PROVIDER_WIRE_ROUTES
from tests.infra.inferred_corpus import (
    CorpusManifestKey,
    InferredCorpusManifest,
    InferredCorpusManifestEntry,
    assert_inferred_corpus_convergence_handoff_complete,
    assert_inferred_corpus_manifest_complete,
    build_inferred_corpus_convergence_handoff,
    compile_inferred_corpus_manifest,
    read_inferred_corpus_manifest,
    write_inferred_corpus_manifest,
)


def _registry() -> SchemaRegistry:
    return SchemaRegistry(storage_root=SCHEMA_DIR)


def _catalog_keys(registry: SchemaRegistry) -> set[CorpusManifestKey]:
    keys: set[CorpusManifestKey] = set()
    for provider in registry.list_providers():
        catalog = registry.load_package_catalog(provider)
        assert catalog is not None
        for package in catalog.packages:
            for element in package.elements:
                keys.add(CorpusManifestKey(provider, package.version, element.element_kind))
    return keys


def _manifest_with_real_persisted_schema() -> InferredCorpusManifest:
    """Build a serialization fixture from a schema in the live persisted registry."""
    registry = _registry()
    manifest = compile_inferred_corpus_manifest(registry=registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )
    schema = registry.get_element_schema(provider, version=package_version, element_kind=element_kind)
    assert isinstance(schema, dict)
    spec = CorpusSpec.for_provider(
        provider,
        package_version=package_version,
        element_kind=element_kind,
        count=1,
        messages_min=1,
        messages_max=1,
        seed=42,
        session_native_ids=("manifest-test",),
    )
    supported = InferredCorpusManifestEntry(
        key=target.key,
        spec=spec,
        generator_schema=schema,
    )
    return InferredCorpusManifest(entries=tuple(supported if entry is target else entry for entry in manifest.entries))


class _RegistryProxy:
    def __init__(self, base: SchemaRegistry) -> None:
        self.base = base
        self.catalog_overrides: dict[str, object] = {}
        self.schema_overrides: dict[tuple[str, str, str], object] = {}

    def list_providers(self) -> list[str]:
        return self.base.list_providers()

    def load_package_catalog(self, provider: str) -> object:
        return self.catalog_overrides.get(provider, self.base.load_package_catalog(provider))

    def get_element_schema(self, provider: str, *, version: str = "default", element_kind: str | None = None) -> object:
        assert element_kind is not None
        package = self.base.get_package(provider, version=version)
        assert package is not None
        key = (provider, package.version, element_kind)
        return self.schema_overrides.get(
            key,
            self.base.get_element_schema(provider, version=package.version, element_kind=element_kind),
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self.base, name)


def test_manifest_covers_every_persisted_package_version_element() -> None:
    registry = _registry()

    manifest = compile_inferred_corpus_manifest(registry=registry)

    assert {
        CorpusManifestKey(entry.key.provider, entry.key.package_version, entry.key.element_kind)
        for entry in manifest.entries
    } == _catalog_keys(registry)
    assert len(manifest.entries) > len(registry.list_providers())
    assert manifest.receipt_state == "catalog_only"
    assert manifest.manifest_id.startswith("manifest:sha256:")
    assert len(manifest.payload_sha256) == 64


def test_persisted_manifest_round_trip_validates_identity_and_integrity(tmp_path: Path) -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    path = tmp_path / "inferred-manifest.json"

    write_inferred_corpus_manifest(manifest, path)

    assert read_inferred_corpus_manifest(path) == manifest


@pytest.mark.parametrize("field", ["manifest_id", "payload_sha256"])
def test_persisted_manifest_rejects_tampered_hash_fields(tmp_path: Path, field: str) -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    path = tmp_path / "inferred-manifest.json"
    write_inferred_corpus_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="(identity|integrity) mismatch"):
        read_inferred_corpus_manifest(path)


def test_persisted_manifest_rejects_unknown_schema_version(tmp_path: Path) -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    path = tmp_path / "inferred-manifest.json"
    write_inferred_corpus_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        read_inferred_corpus_manifest(path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"manifest_id": 1, "manifest_id": 2}', "duplicate"),
        ('{"manifest_id": NaN}', "non-finite"),
    ],
)
def test_persisted_manifest_rejects_noncanonical_json(tmp_path: Path, payload: str, message: str) -> None:
    path = tmp_path / "noncanonical.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_inferred_corpus_manifest(path)


def test_persisted_manifest_rejects_unhashed_extra_fields(tmp_path: Path) -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    path = tmp_path / "inferred-manifest.json"
    write_inferred_corpus_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields changed"):
        read_inferred_corpus_manifest(path)


def test_persisted_manifest_rejects_spec_identity_tampering(tmp_path: Path) -> None:
    manifest = _manifest_with_real_persisted_schema()
    path = tmp_path / "inferred-manifest.json"
    write_inferred_corpus_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    supported = next(entry for entry in payload["entries"] if entry["supported"] is True)
    supported["spec"]["provider"] = "wrong-provider"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        read_inferred_corpus_manifest(path)


def test_persisted_selection_preserves_workload_profile(tmp_path: Path) -> None:
    base = _manifest_with_real_persisted_schema()
    supported = next(entry for entry in base.entries if entry.spec is not None)
    profile: SchemaRecord = {"elements": {supported.key.element_kind: {"structural_variants": []}}}
    profiled = InferredCorpusManifest(
        entries=tuple(
            replace(entry, workload_profile=profile) if entry is supported else entry for entry in base.entries
        )
    )
    path = tmp_path / "profiled-manifest.json"
    write_inferred_corpus_manifest(profiled, path)

    persisted = read_inferred_corpus_manifest(path)
    handoff = build_inferred_corpus_convergence_handoff(path)

    persisted_supported = next(entry for entry in persisted.entries if entry.spec is not None)
    assert persisted_supported.workload_profile == profile
    assert handoff.selections[0].workload_profile == profile


@pytest.mark.parametrize("extra_field", ["workload_profile", "spec"])
def test_persisted_manifest_rejects_noncanonical_optional_entry_fields(tmp_path: Path, extra_field: str) -> None:
    manifest = _manifest_with_real_persisted_schema()
    path = tmp_path / "noncanonical-manifest.json"
    write_inferred_corpus_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if extra_field == "workload_profile":
        supported = next(entry for entry in payload["entries"] if entry["supported"] is True)
        supported[extra_field] = None
    else:
        unsupported = next(entry for entry in payload["entries"] if entry["supported"] is False)
        unsupported[extra_field] = None
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields changed|workload_profile"):
        read_inferred_corpus_manifest(path)


def test_manifest_is_independent_of_default_version_selection() -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    for provider in registry.list_providers():
        catalog = registry.load_package_catalog(provider)
        assert catalog is not None
        if len(catalog.packages) > 1:
            proxy.catalog_overrides[provider] = replace(
                catalog,
                default_version=catalog.packages[0].version,
            )

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]

    assert {
        CorpusManifestKey(entry.key.provider, entry.key.package_version, entry.key.element_kind)
        for entry in manifest.entries
    } == _catalog_keys(registry)
    assert {entry.key.package_version for entry in manifest.entries} >= {"v1", "v2"}


def test_completeness_guard_detects_a_removed_enumerated_entry() -> None:
    registry = _registry()
    manifest = compile_inferred_corpus_manifest(registry=registry)
    reduced = InferredCorpusManifest(entries=manifest.entries[:-1])

    with pytest.raises(AssertionError, match="coverage mismatch"):
        assert_inferred_corpus_manifest_complete(reduced, registry)


def test_missing_element_schema_becomes_explicit_unsupported_record() -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    target_provider, _, _ = _first_wired_catalog_entry(registry)
    catalog = registry.load_package_catalog(target_provider)
    assert catalog is not None
    target_package = catalog.packages[0]
    target_element = target_package.elements[0]
    proxy.schema_overrides[(target_provider, target_package.version, target_element.element_kind)] = None

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (
            entry.key.provider,
            entry.key.package_version,
            entry.key.element_kind,
        )
        == (target_provider, target_package.version, target_element.element_kind)
    )

    assert target.spec is None
    assert target.unsupported is not None
    assert target.unsupported.reason == "missing_schema"


def test_catalog_element_marked_unsupported_is_retained_as_a_typed_record() -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, _, _ = _first_wired_catalog_entry(registry)
    catalog = registry.load_package_catalog(provider)
    assert catalog is not None
    package = catalog.packages[0]
    element = package.elements[0]
    unsupported_element = replace(element, supported=False)
    proxy.catalog_overrides[provider] = replace(
        catalog,
        packages=[
            replace(
                package,
                elements=[unsupported_element, *package.elements[1:]],
            ),
            *catalog.packages[1:],
        ],
    )

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (
            entry.key.provider,
            entry.key.package_version,
            entry.key.element_kind,
        )
        == (provider, package.version, element.element_kind)
    )

    assert target.spec is None
    assert target.unsupported is not None
    assert target.unsupported.reason == "unsupported_element"


def test_removed_wire_format_is_explicit_and_does_not_drop_provider_entries() -> None:
    registry = _registry()
    provider = next(name for name in registry.list_providers() if name in PROVIDER_WIRE_FORMATS)
    formats = dict(PROVIDER_WIRE_FORMATS)
    formats.pop(provider)

    manifest = compile_inferred_corpus_manifest(registry=registry, wire_formats=formats)

    provider_entries = [entry for entry in manifest.entries if entry.key.provider == provider]
    assert provider_entries
    assert all(entry.spec is None for entry in provider_entries)
    assert {entry.unsupported.reason for entry in provider_entries if entry.unsupported is not None} == {
        "provider_without_wire_format"
    }


def test_unsupported_json_schema_construct_is_keyed_and_receiptable() -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider = next(name for name in registry.list_providers() if name in PROVIDER_WIRE_FORMATS)
    catalog = registry.load_package_catalog(provider)
    assert catalog is not None
    package = catalog.packages[0]
    element = package.elements[0]
    schema = registry.get_element_schema(provider, version=package.version, element_kind=element.element_kind)
    assert isinstance(schema, dict)
    mutated = dict(schema)
    raw_properties = mutated.get("properties")
    assert isinstance(raw_properties, dict)
    properties = dict(raw_properties)
    properties["manifest_unsupported"] = {"enum": ["future"]}
    mutated["properties"] = properties
    proxy.schema_overrides[(provider, package.version, element.element_kind)] = mutated
    receipt = {"receipt_id": "tnqqt-package-receipt-placeholder", "status": "pending"}

    manifest = compile_inferred_corpus_manifest(registry=proxy, package_receipt=receipt)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (
            entry.key.provider,
            entry.key.package_version,
            entry.key.element_kind,
        )
        == (provider, package.version, element.element_kind)
    )

    assert target.spec is None
    assert target.unsupported is not None
    assert target.unsupported.reason == "unsupported_json_schema_construct"
    assert {"enum", "required"} <= set(target.unsupported.details)
    assert ("type", "supported") in {(item.construct, item.state) for item in target.key.construct_support}
    assert manifest.package_receipt == receipt
    assert manifest.receipt_state == "package_receipt_attached"
    assert manifest.to_payload()["package_receipt"] == receipt


def test_every_actual_schema_keyword_is_keyed_and_unhandled_constraints_fail_closed() -> None:
    registry = _registry()
    manifest = compile_inferred_corpus_manifest(registry=registry)

    observed_constructs: set[str] = set()
    for entry in manifest.entries:
        observed_constructs.update(item.construct for item in entry.key.construct_support)

    assert {"$id", "$schema", "maxLength", "minLength", "required"} <= observed_constructs
    browser_entry = next(entry for entry in manifest.entries if entry.key.provider == "browser-capture")
    construct_states = {item.construct: item.state for item in browser_entry.key.construct_support}
    assert construct_states["minLength"] == "unsupported"
    assert construct_states["maxLength"] == "unsupported"
    assert construct_states["required"] == "unsupported"


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("pattern", "^[A-Z]+$"),
        ("format", "uuid"),
        ("minimum", 1),
        ("maxItems", 1),
    ],
)
def test_unhandled_standard_constraints_are_typed_unsupported_records(keyword: str, value: object) -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider = next(name for name in registry.list_providers() if name in PROVIDER_WIRE_FORMATS)
    catalog = registry.load_package_catalog(provider)
    assert catalog is not None
    package = catalog.packages[0]
    element = package.elements[0]
    schema = registry.get_element_schema(provider, version=package.version, element_kind=element.element_kind)
    assert isinstance(schema, dict)
    mutated = dict(schema)
    raw_properties = mutated.get("properties")
    assert isinstance(raw_properties, dict)
    properties: dict[str, JSONValue] = dict(raw_properties)
    properties["unhandled_constraint"] = {"type": "string", keyword: cast(JSONValue, value)}
    mutated["properties"] = properties
    proxy.schema_overrides[(provider, package.version, element.element_kind)] = mutated

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package.version, element.element_kind)
    )

    assert target.spec is None
    assert target.unsupported is not None
    assert target.unsupported.reason == "unsupported_json_schema_construct"
    assert keyword in target.unsupported.details
    assert (keyword, "unsupported") in {(item.construct, item.state) for item in target.key.construct_support}


@pytest.mark.parametrize(
    ("annotation", "schema_type", "value"),
    [
        ("x-polylogue-range", "string", [1, 2]),
        ("x-polylogue-array-lengths", "string", [1, 2]),
        ("x-polylogue-multiline", "integer", True),
        (
            "x-polylogue-foreign-keys",
            "string",
            [{"source": "$.id", "target": "$.parent"}],
        ),
    ],
)
def test_annotation_wrong_node_shape_or_path_fails_closed(
    annotation: str,
    schema_type: str,
    value: JSONValue,
) -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    schema = registry.get_element_schema(provider, version=package_version, element_kind=element_kind)
    assert isinstance(schema, dict)
    mutated: dict[str, JSONValue] = dict(schema)
    raw_properties = mutated.get("properties")
    assert isinstance(raw_properties, dict)
    properties: dict[str, JSONValue] = dict(raw_properties)
    properties["annotation_probe"] = {"type": schema_type, annotation: value}
    mutated["properties"] = properties
    proxy.schema_overrides[(provider, package_version, element_kind)] = mutated

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.unsupported is not None
    assert annotation in target.unsupported.details


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        ("x-polylogue-range", [3, 2]),
        (
            "x-polylogue-time-deltas",
            [{"field_a": "$.a", "field_b": "$.b", "min_delta": 5, "max_delta": 2, "avg_delta": 3}],
        ),
        ("x-polylogue-string-lengths", [{"path": "$.text", "min": 10, "max": 2, "avg": 4, "stddev": -1}]),
        (
            "x-polylogue-observed-distribution",
            {
                "numeric": {"histogram": [[1, 2]], "log_base": 1.1},
                "array_length": {"histogram": [[1, 0]], "log_base": 1.1},
            },
        ),
        (
            "x-polylogue-observed-distribution",
            {"numeric": {"histogram": [[1, 2]], "log_base": 1.1, "stddev": -1}},
        ),
        (
            "x-polylogue-observed-distribution",
            {"numeric": {"histogram": [[1, 2]], "log_base": 1.1, "min": 0, "max": 10, "p50": 100}},
        ),
        (
            "x-polylogue-observed-distribution",
            {"numeric": {"histogram": [[10**1000, 1]], "log_base": 1.1}},
        ),
        (
            "x-polylogue-observed-distribution",
            {"numeric": {"histogram": [[711, 1]], "log_base": 2.718281828459045}},
        ),
    ],
)
def test_invalid_numeric_relation_or_partial_distribution_fails_closed(
    annotation: str,
    value: JSONValue,
) -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    schema = registry.get_element_schema(provider, version=package_version, element_kind=element_kind)
    assert isinstance(schema, dict)
    mutated: dict[str, JSONValue] = dict(schema)
    if annotation in {"x-polylogue-time-deltas", "x-polylogue-string-lengths"}:
        mutated[annotation] = value
    else:
        raw_properties = mutated.get("properties")
        assert isinstance(raw_properties, dict)
        properties: dict[str, JSONValue] = dict(raw_properties)
        properties["annotation_probe"] = {
            "type": "number",
            annotation: value,
        }
        mutated["properties"] = properties
    proxy.schema_overrides[(provider, package_version, element_kind)] = mutated

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.unsupported is not None
    assert annotation in target.unsupported.details


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        (
            "x-polylogue-string-lengths",
            [{"path": "not-a-generated-path", "min": 1, "max": 4, "avg": 2, "stddev": 1}],
        ),
        (
            "x-polylogue-foreign-keys",
            [{"source": "$.missing", "target": "$.missing_id"}],
        ),
        (
            "x-polylogue-time-deltas",
            [{"field_a": "$.missing_a", "field_b": "$.missing_b", "min_delta": 1, "max_delta": 2, "avg_delta": 1.5}],
        ),
    ],
)
def test_relation_annotation_paths_must_resolve_in_schema(
    annotation: str,
    value: JSONValue,
) -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    proxy.schema_overrides[(provider, package_version, element_kind)] = _schema_with_root_annotation(
        registry,
        provider=provider,
        package_version=package_version,
        element_kind=element_kind,
        annotation=annotation,
        value=value,
    )

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.unsupported is not None
    assert annotation in target.unsupported.details


def test_time_delta_paths_must_have_compatible_types() -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    schema = registry.get_element_schema(provider, version=package_version, element_kind=element_kind)
    assert isinstance(schema, dict)
    mutated: dict[str, JSONValue] = dict(schema)
    raw_properties = mutated.get("properties")
    assert isinstance(raw_properties, dict)
    properties: dict[str, JSONValue] = dict(raw_properties)
    properties["time_delta_text"] = {"type": "string"}
    properties["time_delta_number"] = {"type": "integer"}
    mutated["properties"] = properties
    mutated["x-polylogue-time-deltas"] = [
        {
            "field_a": "$.time_delta_text",
            "field_b": "$.time_delta_number",
            "min_delta": 1,
            "max_delta": 2,
            "avg_delta": 1.5,
        }
    ]
    proxy.schema_overrides[(provider, package_version, element_kind)] = mutated

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.unsupported is not None
    assert "x-polylogue-time-deltas" in target.unsupported.details


def test_convergence_handoff_rejects_an_omitted_supported_spec() -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    handoff = build_inferred_corpus_convergence_handoff(manifest)
    assert handoff.specs == ()
    assert_inferred_corpus_convergence_handoff_complete(manifest, handoff)


def _first_wired_catalog_entry(
    registry: SchemaRegistry,
    *,
    annotation: str | None = None,
) -> tuple[str, str, str]:
    for provider in registry.list_providers():
        route = PROVIDER_WIRE_ROUTES.get(provider)
        if route is None or route.status != "supported":
            continue
        catalog = registry.load_package_catalog(provider)
        assert catalog is not None
        for package in catalog.packages:
            for element in package.elements:
                if annotation is not None:
                    proxy = _RegistryProxy(registry)
                    proxy.schema_overrides[(provider, package.version, element.element_kind)] = _schema_with_annotation(
                        registry,
                        provider=provider,
                        package_version=package.version,
                        element_kind=element.element_kind,
                        annotation=annotation,
                    )
                    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
                    candidate = next(
                        entry
                        for entry in manifest.entries
                        if (
                            entry.key.provider,
                            entry.key.package_version,
                            entry.key.element_kind,
                        )
                        == (provider, package.version, element.element_kind)
                    )
                    if (annotation, "supported") not in {
                        (item.construct, item.state) for item in candidate.key.construct_support
                    }:
                        continue
                return provider, package.version, element.element_kind
    raise AssertionError("expected a persisted provider with a wire format")


def test_persisted_package_route_is_nonempty_and_uses_persisted_selection() -> None:
    manifest = _manifest_with_real_persisted_schema()

    assert manifest.supported_specs
    entry = next(entry for entry in manifest.entries if entry.spec is not None)
    assert entry.key.provider in PROVIDER_WIRE_FORMATS
    assert entry.generator_schema is not None
    handoff = build_inferred_corpus_convergence_handoff(manifest)
    assert handoff.selections
    assert handoff.selections[0].schema == entry.generator_schema


def _schema_with_annotation(
    registry: SchemaRegistry,
    *,
    provider: str,
    package_version: str,
    element_kind: str,
    annotation: str,
) -> dict[str, JSONValue]:
    schema = registry.get_element_schema(provider, version=package_version, element_kind=element_kind)
    assert isinstance(schema, dict)
    mutated: dict[str, JSONValue] = dict(schema)
    raw_properties = mutated.get("properties")
    assert isinstance(raw_properties, dict)
    properties: dict[str, JSONValue] = dict(raw_properties)
    properties["annotation_probe"] = {"type": "string", annotation: "uuid"}
    mutated["properties"] = properties
    return mutated


def _schema_with_root_annotation(
    registry: SchemaRegistry,
    *,
    provider: str,
    package_version: str,
    element_kind: str,
    annotation: str,
    value: JSONValue,
) -> dict[str, JSONValue]:
    schema = registry.get_element_schema(provider, version=package_version, element_kind=element_kind)
    assert isinstance(schema, dict)
    mutated: dict[str, JSONValue] = dict(schema)
    mutated[annotation] = value
    return mutated


def test_known_generator_annotation_remains_supported_and_is_keyed() -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(
        registry,
        annotation="x-polylogue-format",
    )
    proxy.schema_overrides[(provider, package_version, element_kind)] = _schema_with_annotation(
        registry,
        provider=provider,
        package_version=package_version,
        element_kind=element_kind,
        annotation="x-polylogue-format",
    )

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.spec is None
    assert ("x-polylogue-format", "supported") in {
        (item.construct, item.state) for item in target.key.construct_support
    }


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        ("x-polylogue-format", "markdown"),
        ("x-polylogue-semantic-role", "identifier"),
        ("x-polylogue-foreign-keys", [{"source": "", "target": "$.id"}]),
        ("x-polylogue-time-deltas", [{"field_a": "$.a", "field_b": "$.b", "min_delta": "bad"}]),
    ],
)
def test_unenforced_annotation_values_are_typed_unsupported_records(annotation: str, value: JSONValue) -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    proxy.schema_overrides[(provider, package_version, element_kind)] = _schema_with_root_annotation(
        registry,
        provider=provider,
        package_version=package_version,
        element_kind=element_kind,
        annotation=annotation,
        value=value,
    )

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.spec is None
    assert target.unsupported is not None
    assert annotation in target.unsupported.details
    assert (annotation, "unsupported") in {(item.construct, item.state) for item in target.key.construct_support}


@pytest.mark.parametrize(
    "annotation",
    [
        "x-polylogue-score",
        "x-polylogue-evidence",
        "x-polylogue-generated-at",
        "x-polylogue-artifact-kind",
        "x-polylogue-package-version",
        "x-polylogue-unknown-constraint",
        "x-third-party-constraint",
    ],
)
def test_unenforced_x_annotation_becomes_a_typed_unsupported_record(annotation: str) -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    proxy.schema_overrides[(provider, package_version, element_kind)] = _schema_with_annotation(
        registry,
        provider=provider,
        package_version=package_version,
        element_kind=element_kind,
        annotation=annotation,
    )

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.spec is None
    assert target.unsupported is not None
    assert target.unsupported.reason == "unsupported_json_schema_construct"
    assert annotation in target.unsupported.details
    assert (annotation, "unsupported") in {(item.construct, item.state) for item in target.key.construct_support}


def test_live_catalog_provenance_annotations_are_censused_as_unsupported() -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())

    score_entries = [
        entry
        for entry in manifest.entries
        if any(item.construct == "x-polylogue-score" for item in entry.key.construct_support)
    ]
    assert score_entries
    assert all(
        ("x-polylogue-score", "unsupported") in {(item.construct, item.state) for item in entry.key.construct_support}
        for entry in score_entries
    )
    assert not manifest.supported_specs
