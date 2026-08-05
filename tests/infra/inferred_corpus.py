"""Runtime compiler for the persisted schema-package corpus manifest.

This is deliberately a test-infrastructure manifest, not an inference receipt.
The persisted registry catalog tells us which package elements exist.  A later
inference/package-receipt lane can be attached through ``package_receipt``
without changing the catalog census or claiming that catalog presence proves
inference completion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

from polylogue.core.json import JSONDocument
from polylogue.scenarios import CorpusSpec
from polylogue.schemas.operator.registry import RuntimeSchemaRegistryLike
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.models import SchemaRecord, SyntheticSchemaSelection
from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS, WireFormat

ConstructSupportState: TypeAlias = Literal["supported", "unsupported"]
UnsupportedCorpusReason: TypeAlias = Literal[
    "provider_without_wire_format",
    "unsupported_element",
    "missing_schema",
    "unsupported_json_schema_construct",
]
PackageReceipt: TypeAlias = JSONDocument

# These are the schema constructs the current recursive synthetic runtime can
# consume as structure.  ``additionalProperties`` is intentionally included:
# the generator emits declared properties and does not need to materialize
# arbitrary unknown keys for the current corpus contract.
_SUPPORTED_SCHEMA_CONSTRUCTS = frozenset(
    {
        "$anchor",
        "$comment",
        "$id",
        "$schema",
        "additionalProperties",
        "anyOf",
        "default",
        "deprecated",
        "description",
        "examples",
        "items",
        "oneOf",
        "properties",
        "readOnly",
        "title",
        "type",
        "writeOnly",
    }
)

# The synthetic runtime can select structural branches and emit declared
# properties, but it does not validate generated values against JSON Schema.
# Every assertion keyword outside the explicit structural subset therefore
# fails closed. This keeps a manifest spec from implying conformance to a
# pattern, format, range, collection bound, or other constraint it cannot
# prove.
_STANDARD_SCHEMA_KEYWORDS = frozenset(
    {
        "$anchor",
        "$comment",
        "$defs",
        "$dynamicAnchor",
        "$dynamicRef",
        "$id",
        "$recursiveAnchor",
        "$recursiveRef",
        "$ref",
        "$schema",
        "$vocabulary",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "contains",
        "contentEncoding",
        "contentMediaType",
        "contentSchema",
        "default",
        "definitions",
        "dependentRequired",
        "dependentSchemas",
        "dependencies",
        "deprecated",
        "description",
        "else",
        "enum",
        "examples",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "formatAssertion",
        "if",
        "items",
        "maxContains",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minContains",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "multipleOf",
        "not",
        "oneOf",
        "pattern",
        "patternProperties",
        "prefixItems",
        "properties",
        "propertyNames",
        "readOnly",
        "required",
        "then",
        "title",
        "type",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
        "writeOnly",
    }
)
_SCHEMA_MAPPING_KEYWORDS = frozenset(
    {
        "$defs",
        "additionalProperties",
        "contentSchema",
        "contains",
        "dependentSchemas",
        "dependencies",
        "else",
        "if",
        "items",
        "not",
        "patternProperties",
        "properties",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SCHEMA_ARRAY_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})

# These annotations are read by the production synthetic runtime, semantic
# value generator, or relation solver. Their support verdict means the
# corresponding generator path consumes the annotation, rather than merely
# tolerating its namespace.
_SUPPORTED_SYNTHETIC_ANNOTATIONS = frozenset(
    {
        "x-polylogue-array-lengths",
        "x-polylogue-foreign-keys",
        "x-polylogue-format",
        "x-polylogue-frequency",
        "x-polylogue-multiline",
        "x-polylogue-mutually-exclusive",
        "x-polylogue-observed-distribution",
        "x-polylogue-range",
        "x-polylogue-semantic-role",
        "x-polylogue-string-lengths",
        "x-polylogue-time-deltas",
        "x-polylogue-values",
    }
)

# These package annotations are consumed by registry/profile loading or retain
# provenance only. They do not constrain an emitted instance, so recognizing
# them cannot claim synthetic schema conformance. Shape-affecting annotations
# without a synthetic consumer, such as ``x-polylogue-ref`` and dynamic-key
# markers, are deliberately absent and fail closed below.
_SUPPORTED_CATALOG_METADATA_ANNOTATIONS = frozenset(
    {
        "x-polylogue-anchor-profile-family-id",
        "x-polylogue-artifact-kind",
        "x-polylogue-element-bundle-scope-count",
        "x-polylogue-element-first-seen",
        "x-polylogue-element-kind",
        "x-polylogue-element-last-seen",
        "x-polylogue-evidence",
        "x-polylogue-evidence-confidence",
        "x-polylogue-exact-structure-ids",
        "x-polylogue-generated-at",
        "x-polylogue-generator",
        "x-polylogue-observed-artifact-count",
        "x-polylogue-package-profile-family-ids",
        "x-polylogue-package-version",
        "x-polylogue-profile-family-ids",
        "x-polylogue-profile-tokens",
        "x-polylogue-promoted-at",
        "x-polylogue-registered-at",
        "x-polylogue-sample-count",
        "x-polylogue-sample-granularity",
        "x-polylogue-score",
        "x-polylogue-version",
    }
)


@dataclass(frozen=True, order=True)
class ConstructSupport:
    """Support verdict for one JSON Schema construct found in an element."""

    construct: str
    state: ConstructSupportState

    def to_payload(self) -> dict[str, str]:
        return {"construct": self.construct, "state": self.state}


@dataclass(frozen=True, order=True)
class CorpusManifestKey:
    """Stable identity for one provider/package/element support decision."""

    provider: str
    package_version: str
    element_kind: str
    construct_support: tuple[ConstructSupport, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "package_version": self.package_version,
            "element_kind": self.element_kind,
            "construct_support": [item.to_payload() for item in self.construct_support],
        }


@dataclass(frozen=True)
class UnsupportedCorpusRecord:
    """Typed refusal retained instead of silently dropping a catalog entry."""

    reason: UnsupportedCorpusReason
    details: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {"reason": self.reason, "details": list(self.details)}


@dataclass(frozen=True)
class InferredCorpusManifestEntry:
    """One complete catalog census row, either executable or explicitly refused."""

    key: CorpusManifestKey
    spec: CorpusSpec | None = None
    unsupported: UnsupportedCorpusRecord | None = None

    def __post_init__(self) -> None:
        if (self.spec is None) == (self.unsupported is None):
            raise ValueError("manifest entry requires exactly one of spec or unsupported")
        if self.spec is not None and (
            self.spec.provider,
            self.spec.package_version,
            self.spec.element_kind,
        ) != (
            self.key.provider,
            self.key.package_version,
            self.key.element_kind,
        ):
            raise ValueError("CorpusSpec identity does not match manifest key")

    @property
    def supported(self) -> bool:
        return self.spec is not None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"key": self.key.to_payload(), "supported": self.supported}
        if self.spec is not None:
            payload["spec"] = self.spec.to_payload()
        if self.unsupported is not None:
            payload["unsupported"] = self.unsupported.to_payload()
        return payload


@dataclass(frozen=True)
class InferredCorpusManifest:
    """Deterministic runtime census with an optional future package receipt."""

    entries: tuple[InferredCorpusManifestEntry, ...]
    package_receipt: PackageReceipt | None = None

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.entries, key=lambda entry: entry.key))
        if ordered != self.entries:
            raise ValueError("inferred corpus manifest entries must be sorted by manifest key")
        keys = [entry.key for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("inferred corpus manifest contains duplicate keys")

    @property
    def supported_specs(self) -> tuple[CorpusSpec, ...]:
        return tuple(entry.spec for entry in self.entries if entry.spec is not None)

    @property
    def unsupported_records(self) -> tuple[UnsupportedCorpusRecord, ...]:
        return tuple(entry.unsupported for entry in self.entries if entry.unsupported is not None)

    @property
    def receipt_state(self) -> Literal["catalog_only", "package_receipt_attached"]:
        return "package_receipt_attached" if self.package_receipt is not None else "catalog_only"

    @property
    def manifest_id(self) -> str:
        encoded = json.dumps(self._payload_without_id(), sort_keys=True, separators=(",", ":"))
        return f"manifest:sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"

    def _payload_without_id(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "receipt_state": self.receipt_state,
            "package_receipt": self.package_receipt,
            "entries": [entry.to_payload() for entry in self.entries],
        }

    def to_payload(self) -> dict[str, object]:
        return {"manifest_id": self.manifest_id, **self._payload_without_id()}


@dataclass(frozen=True)
class InferredCorpusConvergenceHandoff:
    """Exact executable manifest subset admitted to the convergence loop."""

    manifest_id: str
    specs: tuple[CorpusSpec, ...]


def _schema_constructs(schema: object) -> tuple[ConstructSupport, ...]:
    """Census every schema-scope keyword without treating property names as keywords."""

    found: dict[str, ConstructSupportState] = {}

    def visit(node: object) -> None:
        if not isinstance(node, Mapping):
            return
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            if key.startswith("x-"):
                found[key] = (
                    "supported"
                    if key in _SUPPORTED_SYNTHETIC_ANNOTATIONS | _SUPPORTED_CATALOG_METADATA_ANNOTATIONS
                    else "unsupported"
                )
                continue
            if key in _SUPPORTED_SCHEMA_CONSTRUCTS:
                found[key] = "supported"
            elif key in _STANDARD_SCHEMA_KEYWORDS:
                found[key] = "unsupported"
            else:
                found[key] = "unsupported"

            if key in _SCHEMA_MAPPING_KEYWORDS and isinstance(value, Mapping):
                if key in {"$defs", "dependentSchemas", "dependencies", "patternProperties", "properties"}:
                    for child in value.values():
                        visit(child)
                else:
                    visit(value)
            elif key in _SCHEMA_ARRAY_KEYWORDS and isinstance(value, Sequence) and not isinstance(value, str):
                for child in value:
                    visit(child)

    visit(schema)
    return tuple(ConstructSupport(construct, found[construct]) for construct in sorted(found))


def build_inferred_corpus_convergence_handoff(
    manifest: InferredCorpusManifest,
) -> InferredCorpusConvergenceHandoff:
    """Bind every supported manifest spec to one convergence-property invocation."""

    handoff = InferredCorpusConvergenceHandoff(manifest_id=manifest.manifest_id, specs=manifest.supported_specs)
    assert_inferred_corpus_convergence_handoff_complete(manifest, handoff)
    return handoff


def assert_inferred_corpus_convergence_handoff_complete(
    manifest: InferredCorpusManifest,
    handoff: InferredCorpusConvergenceHandoff,
) -> None:
    """Reject a stale, omitted, or substituted convergence handoff."""

    if handoff.manifest_id != manifest.manifest_id:
        raise AssertionError(
            "inferred corpus convergence handoff belongs to a different manifest: "
            f"expected={manifest.manifest_id!r}, actual={handoff.manifest_id!r}"
        )
    if handoff.specs != manifest.supported_specs:
        raise AssertionError(
            "inferred corpus convergence handoff omitted or substituted supported specs: "
            f"expected={manifest.supported_specs!r}, actual={handoff.specs!r}"
        )


def _stable_seed(key: CorpusManifestKey) -> int:
    material = "\x1f".join((key.provider, key.package_version, key.element_kind)).encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def _catalog_entries(
    registry: RuntimeSchemaRegistryLike,
) -> tuple[tuple[str, SchemaPackageCatalog, SchemaVersionPackage, SchemaElementManifest], ...]:
    result: list[tuple[str, SchemaPackageCatalog, SchemaVersionPackage, SchemaElementManifest]] = []
    for provider in sorted(set(registry.list_providers())):
        catalog = registry.load_package_catalog(provider)
        if catalog is None:
            raise RuntimeError(f"registry provider {provider!r} has no persisted package catalog")
        for package in sorted(catalog.packages, key=lambda item: item.version):
            for element in sorted(package.elements, key=lambda item: item.element_kind):
                result.append((provider, catalog, package, element))
    return tuple(result)


def _unsupported_reason(
    *,
    element: SchemaElementManifest,
    schema: SchemaRecord | None,
    wire_format: WireFormat | None,
    construct_support: tuple[ConstructSupport, ...],
) -> UnsupportedCorpusRecord | None:
    if wire_format is None:
        return UnsupportedCorpusRecord("provider_without_wire_format")
    if not element.supported:
        return UnsupportedCorpusRecord("unsupported_element")
    if schema is None or element.schema_file is None:
        return UnsupportedCorpusRecord("missing_schema")
    unsupported_constructs = tuple(item.construct for item in construct_support if item.state == "unsupported")
    if unsupported_constructs:
        return UnsupportedCorpusRecord("unsupported_json_schema_construct", unsupported_constructs)
    return None


def _compile_entry(
    *,
    provider: str,
    package: SchemaVersionPackage,
    element: SchemaElementManifest,
    registry: RuntimeSchemaRegistryLike,
    wire_formats: Mapping[str, WireFormat],
) -> InferredCorpusManifestEntry:
    key_without_constructs = CorpusManifestKey(provider, package.version, element.element_kind)
    schema = registry.get_element_schema(
        provider,
        version=package.version,
        element_kind=element.element_kind,
    )
    construct_support = _schema_constructs(schema)
    key = replace(key_without_constructs, construct_support=construct_support)
    wire_format = wire_formats.get(provider)
    unsupported = _unsupported_reason(
        element=element,
        schema=schema if isinstance(schema, dict) else None,
        wire_format=wire_format,
        construct_support=construct_support,
    )
    if unsupported is not None:
        return InferredCorpusManifestEntry(key=key, unsupported=unsupported)

    if not isinstance(schema, dict) or wire_format is None:
        raise AssertionError("supported manifest entry lost schema or wire format")
    profile_loader = getattr(registry, "get_workload_profile", None)
    workload_profile = profile_loader(provider, package.version) if callable(profile_loader) else None
    spec = CorpusSpec.for_provider(
        provider,
        package_version=package.version,
        element_kind=element.element_kind,
        count=1,
        messages_min=4,
        messages_max=4,
        seed=_stable_seed(key),
        origin="inferred.schema-package-manifest",
        tags=("inferred", "schema", "synthetic", "manifest"),
    )
    # Construct the production generator against the exact package/version/
    # element.  No generation is performed here, so compiling the receipt does
    # not turn an unverified catalog into an inference claim.
    SyntheticCorpus.from_selection(
        SyntheticSchemaSelection(
            provider=provider,
            package_version=package.version,
            element_kind=element.element_kind,
            schema=schema,
            wire_format=wire_format,
            workload_profile=workload_profile if isinstance(workload_profile, dict) else None,
        )
    )
    return InferredCorpusManifestEntry(key=key, spec=spec)


def assert_inferred_corpus_manifest_complete(
    manifest: InferredCorpusManifest,
    registry: RuntimeSchemaRegistryLike,
) -> None:
    """Fail loudly when a manifest omits any currently persisted catalog entry."""

    expected = {
        CorpusManifestKey(provider, package.version, element.element_kind)
        for provider, _catalog, package, element in _catalog_entries(registry)
    }
    actual = {
        CorpusManifestKey(entry.key.provider, entry.key.package_version, entry.key.element_kind)
        for entry in manifest.entries
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise AssertionError(
            f"inferred corpus manifest coverage mismatch: missing={missing!r}, unexpected={unexpected!r}"
        )


def compile_inferred_corpus_manifest(
    *,
    registry: RuntimeSchemaRegistryLike,
    package_receipt: PackageReceipt | None = None,
    wire_formats: Mapping[str, WireFormat] | None = None,
) -> InferredCorpusManifest:
    """Compile every persisted package/version/element into a typed manifest."""

    formats = PROVIDER_WIRE_FORMATS if wire_formats is None else wire_formats
    entries = tuple(
        _compile_entry(
            provider=provider,
            package=package,
            element=element,
            registry=registry,
            wire_formats=formats,
        )
        for provider, catalog, package, element in _catalog_entries(registry)
    )
    manifest = InferredCorpusManifest(
        entries=tuple(sorted(entries, key=lambda entry: entry.key)), package_receipt=package_receipt
    )
    assert_inferred_corpus_manifest_complete(manifest, registry)
    return manifest


__all__ = [
    "ConstructSupport",
    "CorpusManifestKey",
    "InferredCorpusConvergenceHandoff",
    "InferredCorpusManifest",
    "InferredCorpusManifestEntry",
    "PackageReceipt",
    "UnsupportedCorpusRecord",
    "assert_inferred_corpus_convergence_handoff_complete",
    "assert_inferred_corpus_manifest_complete",
    "build_inferred_corpus_convergence_handoff",
    "compile_inferred_corpus_manifest",
]
