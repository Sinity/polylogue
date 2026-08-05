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
from pathlib import Path
from typing import Literal, TypeAlias, cast

from polylogue.core.json import JSONDocument
from polylogue.core.sources import origin_from_provider
from polylogue.scenarios import CorpusSpec
from polylogue.schemas.operator.receipt import (
    SchemaInferenceReceipt,
    SchemaReceiptRegistry,
    package_hashes_for_registry,
)
from polylogue.schemas.operator.registry import RuntimeSchemaRegistryLike
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.classification import ConstructSupport, classify_schema_constructs
from polylogue.schemas.synthetic.models import SchemaRecord, SyntheticSchemaSelection
from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS, WireFormat

INFERRED_CORPUS_MANIFEST_SCHEMA_VERSION = 1
UnsupportedCorpusReason: TypeAlias = Literal[
    "provider_without_wire_format",
    "unsupported_element",
    "missing_schema",
    "unsupported_json_schema_construct",
]
PackageReceipt: TypeAlias = JSONDocument


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
    generator_schema: SchemaRecord | None = None
    workload_profile: SchemaRecord | None = None

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
        if self.spec is not None and not isinstance(self.generator_schema, dict):
            raise ValueError("supported manifest entry requires a generator schema")
        if self.unsupported is not None and self.generator_schema is not None:
            raise ValueError("unsupported manifest entry must not carry a generator schema")
        if self.unsupported is not None and self.workload_profile is not None:
            raise ValueError("unsupported manifest entry must not carry a workload profile")

    @property
    def supported(self) -> bool:
        return self.spec is not None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"key": self.key.to_payload(), "supported": self.supported}
        if self.spec is not None:
            payload["spec"] = self.spec.to_payload()
        if self.unsupported is not None:
            payload["unsupported"] = self.unsupported.to_payload()
        if self.generator_schema is not None:
            payload["generator_schema"] = self.generator_schema
        if self.workload_profile is not None:
            payload["workload_profile"] = self.workload_profile
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
        encoded = _canonical_json(self._payload_without_id())
        return f"manifest:sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"

    @property
    def payload_sha256(self) -> str:
        encoded = _canonical_json({"manifest_id": self.manifest_id, **self._payload_without_id()})
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _payload_without_id(self) -> dict[str, object]:
        return {
            "schema_version": INFERRED_CORPUS_MANIFEST_SCHEMA_VERSION,
            "receipt_state": self.receipt_state,
            "package_receipt": self.package_receipt,
            "entries": [entry.to_payload() for entry in self.entries],
        }

    def to_payload(self) -> dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            **self._payload_without_id(),
            "payload_sha256": self.payload_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> InferredCorpusManifest:
        expected_fields = {
            "manifest_id",
            "schema_version",
            "receipt_state",
            "package_receipt",
            "entries",
            "payload_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError(f"inferred corpus manifest fields changed: {sorted(set(payload) ^ expected_fields)}")
        schema_version = payload.get("schema_version")
        if schema_version != INFERRED_CORPUS_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "unsupported inferred corpus manifest schema_version: "
                f"expected={INFERRED_CORPUS_MANIFEST_SCHEMA_VERSION}, actual={schema_version!r}"
            )
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("inferred corpus manifest entries must be a list")
        entries = tuple(_manifest_entry_from_payload(item) for item in raw_entries)
        receipt_state = payload.get("receipt_state")
        package_receipt = payload.get("package_receipt")
        if receipt_state not in {"catalog_only", "package_receipt_attached"}:
            raise ValueError(f"invalid inferred corpus manifest receipt_state: {receipt_state!r}")
        if receipt_state == "catalog_only" and package_receipt is not None:
            raise ValueError("catalog_only manifest must not carry a package receipt")
        if receipt_state == "package_receipt_attached" and not isinstance(package_receipt, dict):
            raise ValueError("package_receipt_attached manifest requires a JSON object receipt")
        manifest = cls(
            entries=entries,
            package_receipt=package_receipt if isinstance(package_receipt, dict) else None,
        )
        expected_manifest_id = manifest.manifest_id
        if payload.get("manifest_id") != expected_manifest_id:
            raise ValueError(
                "inferred corpus manifest identity mismatch: "
                f"expected={expected_manifest_id!r}, actual={payload.get('manifest_id')!r}"
            )
        expected_payload_sha256 = manifest.payload_sha256
        if payload.get("payload_sha256") != expected_payload_sha256:
            raise ValueError(
                "inferred corpus manifest payload integrity mismatch: "
                f"expected={expected_payload_sha256!r}, actual={payload.get('payload_sha256')!r}"
            )
        return manifest


def _require_inference_handoff(manifest: InferredCorpusManifest) -> SchemaInferenceReceipt:
    if manifest.receipt_state == "catalog_only" or manifest.package_receipt is None:
        raise ValueError("campaign mode requires a persisted schema-inference handoff, not catalog-only data")
    return SchemaInferenceReceipt.from_payload(manifest.package_receipt)


@dataclass(frozen=True)
class InferredCorpusConvergenceHandoff:
    """Exact executable manifest subset admitted to the convergence loop."""

    manifest_id: str
    specs: tuple[CorpusSpec, ...]
    selections: tuple[SyntheticSchemaSelection, ...]


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _manifest_entry_from_payload(payload: object) -> InferredCorpusManifestEntry:
    if not isinstance(payload, Mapping):
        raise ValueError("inferred corpus manifest entry must be a JSON object")
    entry_fields = {"key", "supported", "spec", "unsupported", "generator_schema", "workload_profile"}
    if set(payload) - entry_fields:
        raise ValueError(f"inferred corpus manifest entry fields changed: {sorted(set(payload) - entry_fields)}")
    raw_key = payload.get("key")
    if not isinstance(raw_key, Mapping):
        raise ValueError("inferred corpus manifest entry key must be a JSON object")
    if set(raw_key) != {"provider", "package_version", "element_kind", "construct_support"}:
        raise ValueError("inferred corpus manifest key fields changed")
    provider = raw_key.get("provider")
    package_version = raw_key.get("package_version")
    element_kind = raw_key.get("element_kind")
    raw_constructs = raw_key.get("construct_support", [])
    if not isinstance(raw_constructs, list) or not all(isinstance(item, Mapping) for item in raw_constructs):
        raise ValueError("manifest construct_support must be a list of objects")
    constructs: list[ConstructSupport] = []
    for raw_construct in cast(list[Mapping[str, object]], raw_constructs):
        construct = raw_construct.get("construct")
        state = raw_construct.get("state")
        if not isinstance(construct, str) or not construct or state not in {"supported", "unsupported"}:
            raise ValueError("manifest construct_support contains an invalid row")
        constructs.append(ConstructSupport(construct=construct, state=state))
    if (
        not isinstance(provider, str)
        or not isinstance(package_version, str)
        or not isinstance(element_kind, str)
        or not element_kind
        or tuple(sorted(constructs)) != tuple(constructs)
        or len({item.construct for item in constructs}) != len(constructs)
    ):
        raise ValueError("manifest entry key has invalid identity or construct ordering")
    key = CorpusManifestKey(provider, package_version, element_kind, tuple(constructs))
    supported = payload.get("supported")
    raw_spec = payload.get("spec")
    raw_unsupported = payload.get("unsupported")
    raw_schema = payload.get("generator_schema")
    raw_workload_profile = payload.get("workload_profile")
    if supported is True:
        expected_entry_fields = {"key", "supported", "spec", "generator_schema"}
        if "workload_profile" in payload:
            expected_entry_fields.add("workload_profile")
        if set(payload) != expected_entry_fields:
            raise ValueError("supported manifest entry fields changed")
        if not isinstance(raw_spec, Mapping) or not isinstance(raw_schema, Mapping):
            raise ValueError("supported manifest entry requires spec and generator_schema")
        if "workload_profile" in payload and not isinstance(raw_workload_profile, Mapping):
            raise ValueError("manifest workload_profile must be a JSON object when present")
        spec_fields = {
            "origin",
            "path_targets",
            "artifact_targets",
            "conceptual_path_targets",
            "conceptual_artifact_targets",
            "operation_targets",
            "maintenance_targets",
            "tags",
            "docs_role",
            "caption",
            "narrative_order",
            "audience",
            "demonstrates",
            "privacy_level",
            "media",
            "visual_style",
            "provider",
            "package_version",
            "count",
            "messages_min",
            "messages_max",
            "style",
            "element_kind",
            "seed",
            "session_native_ids",
            "profile",
        }
        if set(raw_spec) - spec_fields:
            raise ValueError("inferred corpus spec fields changed")
        spec = CorpusSpec.from_payload(cast(dict[str, object], raw_spec))
        return InferredCorpusManifestEntry(
            key=key,
            spec=spec,
            generator_schema=cast(SchemaRecord, dict(raw_schema)),
            workload_profile=(
                cast(SchemaRecord, dict(raw_workload_profile)) if isinstance(raw_workload_profile, Mapping) else None
            ),
        )
    if supported is not False:
        raise ValueError("manifest entry supported must be boolean")
    if set(payload) != {"key", "supported", "unsupported"}:
        raise ValueError("unsupported manifest entry fields changed")
    if not isinstance(raw_unsupported, Mapping):
        raise ValueError("unsupported manifest entry requires unsupported metadata only")
    reason = raw_unsupported.get("reason")
    details = raw_unsupported.get("details", [])
    if set(raw_unsupported) != {"reason", "details"}:
        raise ValueError("manifest unsupported record fields changed")
    valid_reasons = {
        "provider_without_wire_format",
        "unsupported_element",
        "missing_schema",
        "unsupported_json_schema_construct",
    }
    if (
        reason not in valid_reasons
        or not isinstance(details, list)
        or not all(isinstance(item, str) for item in details)
    ):
        raise ValueError("manifest unsupported record is invalid")
    return InferredCorpusManifestEntry(
        key=key,
        unsupported=UnsupportedCorpusRecord(
            reason=cast(UnsupportedCorpusReason, reason),
            details=tuple(cast(str, item) for item in details),
        ),
    )


def write_inferred_corpus_manifest(manifest: InferredCorpusManifest, path: Path) -> None:
    """Persist one canonical manifest payload with an independently checked hash."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_payload(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_inferred_corpus_manifest(
    path: Path, *, campaign_mode: bool = False, registry: RuntimeSchemaRegistryLike | None = None
) -> InferredCorpusManifest:
    """Read and validate a persisted manifest before exposing executable rows."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"unable to read inferred corpus manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("inferred corpus manifest root must be a JSON object")
    manifest = InferredCorpusManifest.from_payload(payload)
    if campaign_mode:
        _require_inference_handoff(manifest)
        if registry is None:
            raise ValueError("campaign mode requires a live schema registry")
        providers = tuple(sorted({entry.key.provider for entry in manifest.entries}))
        _validate_inference_handoff(manifest, registry, providers=providers)
    return manifest


def _schema_constructs(schema: object) -> tuple[ConstructSupport, ...]:
    """Use the production classifier for campaign admission decisions."""

    return classify_schema_constructs(schema)


def build_inferred_corpus_convergence_handoff(
    manifest: InferredCorpusManifest | Path,
    *,
    campaign_mode: bool = False,
    registry: RuntimeSchemaRegistryLike | None = None,
) -> InferredCorpusConvergenceHandoff:
    """Bind every supported row from memory or persisted disk to convergence."""

    persisted_manifest = (
        read_inferred_corpus_manifest(manifest, campaign_mode=campaign_mode, registry=registry)
        if isinstance(manifest, Path)
        else manifest
    )
    if campaign_mode:
        _require_inference_handoff(persisted_manifest)
        if registry is None:
            raise ValueError("campaign mode requires a live schema registry")
        providers = tuple(sorted({entry.key.provider for entry in persisted_manifest.entries}))
        _validate_inference_handoff(persisted_manifest, registry, providers=providers)
    selections = tuple(_selection_for_entry(entry) for entry in persisted_manifest.entries if entry.spec is not None)
    handoff = InferredCorpusConvergenceHandoff(
        manifest_id=persisted_manifest.manifest_id,
        specs=persisted_manifest.supported_specs,
        selections=selections,
    )
    assert_inferred_corpus_convergence_handoff_complete(persisted_manifest, handoff)
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
    expected_selections = tuple(_selection_for_entry(entry) for entry in manifest.entries if entry.spec is not None)
    if handoff.selections != expected_selections:
        raise AssertionError(
            "inferred corpus convergence handoff omitted or substituted generator selections: "
            f"expected={expected_selections!r}, actual={handoff.selections!r}"
        )


def _selection_for_entry(entry: InferredCorpusManifestEntry) -> SyntheticSchemaSelection:
    if entry.spec is None or entry.generator_schema is None:
        raise ValueError("unsupported inferred corpus entry cannot produce a generator selection")
    wire_format = PROVIDER_WIRE_FORMATS.get(entry.key.provider)
    if wire_format is None:
        raise ValueError(f"inferred corpus entry has no production wire format: {entry.key.provider!r}")
    return SyntheticSchemaSelection(
        provider=entry.key.provider,
        package_version=entry.key.package_version,
        element_kind=entry.key.element_kind,
        schema=entry.generator_schema,
        wire_format=wire_format,
        workload_profile=entry.workload_profile,
    )


def _stable_seed(key: CorpusManifestKey) -> int:
    material = "\x1f".join((key.provider, key.package_version, key.element_kind)).encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def _catalog_entries(
    registry: RuntimeSchemaRegistryLike,
    providers: Sequence[str] | None = None,
) -> tuple[tuple[str, SchemaPackageCatalog, SchemaVersionPackage, SchemaElementManifest], ...]:
    result: list[tuple[str, SchemaPackageCatalog, SchemaVersionPackage, SchemaElementManifest]] = []
    provider_names = providers if providers is not None else registry.list_providers()
    for provider in sorted(set(provider_names)):
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
    selection = SyntheticSchemaSelection(
        provider=provider,
        package_version=package.version,
        element_kind=element.element_kind,
        schema=schema,
        wire_format=wire_format,
        workload_profile=workload_profile if isinstance(workload_profile, dict) else None,
    )
    witness = SyntheticCorpus.from_selection(selection).generate(
        count=1,
        messages_per_session=range(spec.messages_min, spec.messages_min + 1),
        seed=spec.seed,
        style=spec.style,
        session_native_ids=spec.session_native_ids[:1],
    )
    if len(witness) != 1 or not witness[0]:
        raise ValueError("persisted schema selection did not produce a real synthetic corpus witness")
    return InferredCorpusManifestEntry(
        key=key,
        spec=spec,
        generator_schema=schema,
        workload_profile=workload_profile if isinstance(workload_profile, dict) else None,
    )


def assert_inferred_corpus_manifest_complete(
    manifest: InferredCorpusManifest,
    registry: RuntimeSchemaRegistryLike,
    *,
    providers: Sequence[str] | None = None,
) -> None:
    """Fail loudly when a manifest omits any currently persisted catalog entry."""

    expected = {
        CorpusManifestKey(provider, package.version, element.element_kind)
        for provider, _catalog, package, element in _catalog_entries(registry, providers)
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
    providers: Sequence[str] | None = None,
    campaign_mode: bool = False,
) -> InferredCorpusManifest:
    """Compile every persisted package/version/element into a typed manifest."""

    formats = PROVIDER_WIRE_FORMATS if wire_formats is None else wire_formats
    if campaign_mode and package_receipt is None:
        raise ValueError("campaign mode requires a persisted schema-inference handoff")
    entries = tuple(
        _compile_entry(
            provider=provider,
            package=package,
            element=element,
            registry=registry,
            wire_formats=formats,
        )
        for provider, catalog, package, element in _catalog_entries(registry, providers)
    )
    manifest = InferredCorpusManifest(
        entries=tuple(sorted(entries, key=lambda entry: entry.key)), package_receipt=package_receipt
    )
    assert_inferred_corpus_manifest_complete(manifest, registry, providers=providers)
    if campaign_mode:
        _validate_inference_handoff(manifest, registry, providers=providers)
    return manifest


def _validate_inference_handoff(
    manifest: InferredCorpusManifest,
    registry: RuntimeSchemaRegistryLike,
    *,
    providers: Sequence[str] | None,
) -> None:
    receipt = _require_inference_handoff(manifest)
    if not manifest.supported_specs:
        raise ValueError("campaign mode has no executable synthetic corpus selection")
    expected_packages = package_hashes_for_registry(cast(SchemaReceiptRegistry, registry), providers)
    if receipt.packages != expected_packages:
        raise ValueError("schema-inference handoff package/version/element hashes do not match the registry")

    catalog_entries = _catalog_entries(registry, providers)
    expected_coverage = {(provider, origin_from_provider(provider).value) for provider, *_rest in catalog_entries}
    actual_coverage = {(item.provider, item.origin) for item in receipt.coverage_decisions}
    if actual_coverage != expected_coverage:
        raise ValueError(
            "schema-inference handoff does not contain complete origin/provider coverage: "
            f"missing={sorted(expected_coverage - actual_coverage)!r}, "
            f"unexpected={sorted(actual_coverage - expected_coverage)!r}"
        )
    entries_by_provider: dict[str, list[InferredCorpusManifestEntry]] = {}
    for entry in manifest.entries:
        entries_by_provider.setdefault(entry.key.provider, []).append(entry)
    for coverage in receipt.coverage_decisions:
        provider_entries = entries_by_provider.get(coverage.provider, [])
        if any(entry.spec is not None for entry in provider_entries):
            expected_decision = "committed"
        elif provider_entries and all(
            entry.unsupported is not None and entry.unsupported.reason == "unsupported_json_schema_construct"
            for entry in provider_entries
        ):
            expected_decision = "nonrepresentable"
        else:
            expected_decision = "unsupported"
        if coverage.decision != expected_decision:
            raise ValueError("schema-inference handoff coverage decision changed")

    for provider, _catalog, package, element in catalog_entries:
        live_entry = next(
            (
                candidate
                for candidate in manifest.entries
                if (candidate.key.provider, candidate.key.package_version, candidate.key.element_kind)
                == (provider, package.version, element.element_kind)
            ),
            None,
        )
        if live_entry is None:
            raise ValueError("schema-inference manifest is missing a live registry entry")
        live_schema = registry.get_element_schema(provider, version=package.version, element_kind=element.element_kind)
        live_constructs = _schema_constructs(live_schema)
        if live_entry.key.construct_support != live_constructs:
            raise ValueError("schema-inference manifest classifier output changed")
        live_unsupported = _unsupported_reason(
            element=element,
            schema=live_schema if isinstance(live_schema, dict) else None,
            wire_format=PROVIDER_WIRE_FORMATS.get(provider),
            construct_support=live_constructs,
        )
        if (live_entry.unsupported is None) != (live_unsupported is None):
            raise ValueError("schema-inference manifest executable support changed")
        if live_unsupported is not None:
            if live_entry.unsupported != live_unsupported:
                raise ValueError("schema-inference manifest unsupported decision changed")
            continue
        if live_entry.generator_schema != live_schema:
            raise ValueError("schema-inference manifest generator schema changed")
        selection = _selection_for_entry(live_entry)
        spec = live_entry.spec
        if spec is None:
            raise ValueError("schema-inference manifest executable entry has no corpus spec")
        witness = SyntheticCorpus.from_selection(selection).generate(
            count=1,
            messages_per_session=range(spec.messages_min, spec.messages_min + 1),
            seed=spec.seed,
            style=spec.style,
            session_native_ids=spec.session_native_ids[:1],
        )
        if len(witness) != 1 or not witness[0]:
            raise ValueError("schema-inference manifest selection produced no executable witness")

    expected_unsupported = {
        (
            entry.key.provider,
            entry.key.package_version,
            entry.key.element_kind,
            "nonrepresentable" if entry.unsupported.reason == "unsupported_json_schema_construct" else "unsupported",
            entry.unsupported.reason,
            entry.unsupported.details,
        )
        for entry in manifest.entries
        if entry.unsupported is not None
    }
    actual_unsupported = {
        (
            item.provider,
            item.package_version,
            item.element_kind,
            item.decision,
            item.reason,
            item.details,
        )
        for item in receipt.unsupported_decisions
    }
    if actual_unsupported != expected_unsupported:
        raise ValueError(
            "schema-inference handoff unsupported/nonrepresentable decisions changed: "
            f"expected={sorted(expected_unsupported)!r}, actual={sorted(actual_unsupported)!r}"
        )


__all__ = [
    "ConstructSupport",
    "CorpusManifestKey",
    "INFERRED_CORPUS_MANIFEST_SCHEMA_VERSION",
    "InferredCorpusConvergenceHandoff",
    "InferredCorpusManifest",
    "InferredCorpusManifestEntry",
    "PackageReceipt",
    "UnsupportedCorpusRecord",
    "assert_inferred_corpus_convergence_handoff_complete",
    "assert_inferred_corpus_manifest_complete",
    "build_inferred_corpus_convergence_handoff",
    "compile_inferred_corpus_manifest",
    "read_inferred_corpus_manifest",
    "write_inferred_corpus_manifest",
]
