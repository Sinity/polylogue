from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from polylogue.core.json import JSONValue
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry
from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS
from tests.infra.inferred_corpus import (
    CorpusManifestKey,
    InferredCorpusConvergenceHandoff,
    InferredCorpusManifest,
    assert_inferred_corpus_convergence_handoff_complete,
    assert_inferred_corpus_manifest_complete,
    build_inferred_corpus_convergence_handoff,
    compile_inferred_corpus_manifest,
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
    target_provider = registry.list_providers()[0]
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
    provider = registry.list_providers()[0]
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
    assert target.unsupported.details == ("enum", "required")
    assert target.key.construct_support[-1].construct == "type"
    assert target.key.construct_support[-1].state == "supported"
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


def test_convergence_handoff_rejects_an_omitted_supported_spec() -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    handoff = build_inferred_corpus_convergence_handoff(manifest)
    assert handoff.specs

    incomplete = InferredCorpusConvergenceHandoff(
        manifest_id=handoff.manifest_id,
        specs=handoff.specs[1:],
    )

    with pytest.raises(AssertionError, match="omitted or substituted"):
        assert_inferred_corpus_convergence_handoff_complete(manifest, incomplete)
