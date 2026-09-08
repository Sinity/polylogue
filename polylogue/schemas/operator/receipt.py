"""Content-addressed handoff receipts for committed inferred schemas."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast

from polylogue.core.durable_fs import atomic_replace
from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONDocument
from polylogue.core.sources import origin_from_provider
from polylogue.schemas.operator.registry import RuntimeSchemaRegistryLike
from polylogue.schemas.package_publication import provider_tree_lock
from polylogue.schemas.packages import SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.runtime_registry import canonical_schema_provider
from polylogue.schemas.synthetic.classification import classify_schema_constructs
from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS

SCHEMA_INFERENCE_HANDOFF_SCHEMA = "polylogue.schema-inference-handoff.v1"
SCHEMA_INFERENCE_HANDOFF_FILENAME = "schema-inference-handoff.json"

CoverageDecision: TypeAlias = Literal["committed", "unsupported", "nonrepresentable"]
_COVERAGE_DECISIONS = frozenset({"committed", "unsupported", "nonrepresentable"})


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, order=True, slots=True)
class SchemaInferenceCoverageDecision:
    """The explicit decision for one origin/provider pair in a handoff."""

    origin: str
    provider: str
    decision: CoverageDecision
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.origin or not self.provider or self.decision not in _COVERAGE_DECISIONS:
            raise ValueError("schema inference coverage decisions require identity and a valid decision")
        if self.decision != "committed" and not self.reason:
            raise ValueError("unsupported or nonrepresentable coverage decisions require a reason")

    def to_payload(self) -> JSONDocument:
        return {
            "origin": self.origin,
            "provider": self.provider,
            "decision": self.decision,
            "reason": self.reason,
        }


@dataclass(frozen=True, order=True, slots=True)
class SchemaInferenceUnsupportedDecision:
    """A typed refusal for an element that cannot enter the inferred corpus."""

    provider: str
    package_version: str
    element_kind: str
    decision: Literal["unsupported", "nonrepresentable"]
    reason: str
    details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.provider or not self.package_version or not self.element_kind:
            raise ValueError("unsupported schema decisions require package identity")
        if self.decision not in {"unsupported", "nonrepresentable"} or not self.reason:
            raise ValueError("unsupported schema decisions require a supported decision and reason")
        if tuple(sorted(set(self.details))) != self.details:
            raise ValueError("unsupported schema decision details must be sorted and unique")

    def to_payload(self) -> JSONDocument:
        return {
            "provider": self.provider,
            "package_version": self.package_version,
            "element_kind": self.element_kind,
            "decision": self.decision,
            "reason": self.reason,
            "details": list(self.details),
        }


@dataclass(frozen=True, order=True, slots=True)
class SchemaElementContentHash:
    element_kind: str
    content_hash: str

    def __post_init__(self) -> None:
        if not self.element_kind:
            raise ValueError("schema element hash requires an element kind")
        _require_digest(self.content_hash, field="element content_hash")

    def to_payload(self) -> JSONDocument:
        return {"element_kind": self.element_kind, "content_hash": self.content_hash}


@dataclass(frozen=True, order=True, slots=True)
class SchemaInferenceInputManifest:
    provider: str
    digest: str | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("input manifest requires a provider")
        if self.digest is None:
            if not self.unavailable_reason:
                raise ValueError("unavailable input manifest requires a reason")
        else:
            _require_digest(self.digest, field="input manifest digest")
            if self.unavailable_reason is not None:
                raise ValueError("bound input manifest cannot have an unavailable reason")

    def to_payload(self) -> JSONDocument:
        return {"provider": self.provider, "digest": self.digest, "unavailable_reason": self.unavailable_reason}


@dataclass(frozen=True, order=True, slots=True)
class SchemaPackageContentHash:
    """Exact persisted hashes for one package version and its elements."""

    provider: str
    package_version: str
    package_hash: str
    version_hash: str
    element_hashes: tuple[SchemaElementContentHash, ...]

    def __post_init__(self) -> None:
        if not self.provider or not self.package_version:
            raise ValueError("schema package hashes require provider and version")
        _require_digest(self.package_hash, field="package_hash")
        _require_digest(self.version_hash, field="version_hash")
        if tuple(sorted(self.element_hashes)) != self.element_hashes:
            raise ValueError("schema element hashes must be sorted")
        if len({item.element_kind for item in self.element_hashes}) != len(self.element_hashes):
            raise ValueError("schema package element hashes must be unique")

    def to_payload(self) -> JSONDocument:
        return {
            "provider": self.provider,
            "package_version": self.package_version,
            "package_hash": self.package_hash,
            "version_hash": self.version_hash,
            "element_hashes": [item.to_payload() for item in self.element_hashes],
        }


@dataclass(frozen=True, slots=True)
class SchemaInferenceReceipt:
    """Immutable aggregate handoff from generation input to inferred corpus."""

    input_manifests: tuple[SchemaInferenceInputManifest, ...]
    coverage_decisions: tuple[SchemaInferenceCoverageDecision, ...]
    packages: tuple[SchemaPackageContentHash, ...]
    unsupported_decisions: tuple[SchemaInferenceUnsupportedDecision, ...] = ()

    def __post_init__(self) -> None:
        if tuple(sorted(self.input_manifests)) != self.input_manifests:
            raise ValueError("input manifests must be sorted")
        if tuple(sorted(self.coverage_decisions)) != self.coverage_decisions:
            raise ValueError("coverage decisions must be sorted")
        if tuple(sorted(self.packages)) != self.packages:
            raise ValueError("package hashes must be sorted")
        if tuple(sorted(self.unsupported_decisions)) != self.unsupported_decisions:
            raise ValueError("unsupported decisions must be sorted")
        coverage_keys = [(item.origin, item.provider) for item in self.coverage_decisions]
        if len(coverage_keys) != len(set(coverage_keys)):
            raise ValueError("coverage decisions must have unique origin/provider pairs")
        package_keys = [(item.provider, item.package_version) for item in self.packages]
        if len(package_keys) != len(set(package_keys)):
            raise ValueError("package hashes must have unique provider/version pairs")
        unsupported_keys = [
            (item.provider, item.package_version, item.element_kind) for item in self.unsupported_decisions
        ]
        if len(unsupported_keys) != len(set(unsupported_keys)):
            raise ValueError("unsupported decisions must have unique package elements")

    @property
    def receipt_digest(self) -> str:
        return hash_payload(self._payload_without_digest())

    def _payload_without_digest(self) -> JSONDocument:
        return {
            "schema": SCHEMA_INFERENCE_HANDOFF_SCHEMA,
            "input_manifests": [item.to_payload() for item in self.input_manifests],
            "coverage_decisions": [item.to_payload() for item in self.coverage_decisions],
            "packages": [item.to_payload() for item in self.packages],
            "unsupported_decisions": [item.to_payload() for item in self.unsupported_decisions],
        }

    def to_payload(self) -> JSONDocument:
        return {**self._payload_without_digest(), "receipt_digest": self.receipt_digest}

    def merged_with(self, other: SchemaInferenceReceipt) -> SchemaInferenceReceipt:
        providers = {item.provider for item in other.coverage_decisions}
        coverage = tuple(
            sorted(
                [item for item in self.coverage_decisions if item.provider not in providers]
                + list(other.coverage_decisions)
            )
        )
        package_providers = {item.provider for item in other.packages}
        packages = tuple(
            sorted([item for item in self.packages if item.provider not in package_providers] + list(other.packages))
        )
        unsupported = tuple(
            sorted(
                [item for item in self.unsupported_decisions if item.provider not in package_providers]
                + list(other.unsupported_decisions)
            )
        )
        manifests = tuple(
            sorted(
                [item for item in self.input_manifests if item.provider not in providers] + list(other.input_manifests)
            )
        )
        return SchemaInferenceReceipt(manifests, coverage, packages, unsupported)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> SchemaInferenceReceipt:
        expected = {
            "schema",
            "input_manifests",
            "coverage_decisions",
            "packages",
            "unsupported_decisions",
            "receipt_digest",
        }
        if set(payload) != expected or payload.get("schema") != SCHEMA_INFERENCE_HANDOFF_SCHEMA:
            raise ValueError("schema inference handoff fields or schema changed")
        receipt = cls(
            input_manifests=tuple(
                _input_manifest_from_payload(item) for item in _list_of_mappings(payload, "input_manifests")
            ),
            coverage_decisions=tuple(
                _coverage_from_payload(item) for item in _list_of_mappings(payload, "coverage_decisions")
            ),
            packages=tuple(_package_from_payload(item) for item in _list_of_mappings(payload, "packages")),
            unsupported_decisions=tuple(
                _unsupported_from_payload(item) for item in _list_of_mappings(payload, "unsupported_decisions")
            ),
        )
        if payload.get("receipt_digest") != receipt.receipt_digest:
            raise ValueError("schema inference handoff receipt digest mismatch")
        return receipt


def _list_of_mappings(payload: Mapping[str, object], field: str) -> list[Mapping[str, object]]:
    value = payload.get(field)
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"schema inference handoff {field} must be a list of objects")
    return [cast(Mapping[str, object], item) for item in value]


def _coverage_from_payload(payload: Mapping[str, object]) -> SchemaInferenceCoverageDecision:
    if set(payload) != {"origin", "provider", "decision", "reason"}:
        raise ValueError("coverage decision fields changed")
    reason = payload.get("reason")
    if not isinstance(reason, str):
        reason = None
    return SchemaInferenceCoverageDecision(
        origin=str(payload.get("origin")),
        provider=str(payload.get("provider")),
        decision=cast(CoverageDecision, payload.get("decision")),
        reason=reason,
    )


def _unsupported_from_payload(payload: Mapping[str, object]) -> SchemaInferenceUnsupportedDecision:
    if set(payload) != {"provider", "package_version", "element_kind", "decision", "reason", "details"}:
        raise ValueError("unsupported decision fields changed")
    details = payload.get("details")
    if not isinstance(details, list) or not all(isinstance(item, str) for item in details):
        raise ValueError("unsupported decision details must be a list of strings")
    return SchemaInferenceUnsupportedDecision(
        provider=str(payload.get("provider")),
        package_version=str(payload.get("package_version")),
        element_kind=str(payload.get("element_kind")),
        decision=cast(Literal["unsupported", "nonrepresentable"], payload.get("decision")),
        reason=str(payload.get("reason")),
        details=tuple(details),
    )


def _element_from_payload(payload: Mapping[str, object]) -> SchemaElementContentHash:
    if set(payload) != {"element_kind", "content_hash"}:
        raise ValueError("element hash fields changed")
    return SchemaElementContentHash(
        str(payload.get("element_kind")), _require_digest(payload.get("content_hash"), field="content_hash")
    )


def _input_manifest_from_payload(payload: Mapping[str, object]) -> SchemaInferenceInputManifest:
    if set(payload) != {"provider", "digest", "unavailable_reason"}:
        raise ValueError("input manifest fields changed")
    digest = payload.get("digest")
    reason = payload.get("unavailable_reason")
    if digest is not None and not isinstance(digest, str):
        raise ValueError("input manifest digest must be a string or null")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("input manifest unavailable_reason must be a string or null")
    return SchemaInferenceInputManifest(str(payload.get("provider")), digest, reason)


def _package_from_payload(payload: Mapping[str, object]) -> SchemaPackageContentHash:
    if set(payload) != {"provider", "package_version", "package_hash", "version_hash", "element_hashes"}:
        raise ValueError("package hash fields changed")
    return SchemaPackageContentHash(
        provider=str(payload.get("provider")),
        package_version=str(payload.get("package_version")),
        package_hash=_require_digest(payload.get("package_hash"), field="package_hash"),
        version_hash=_require_digest(payload.get("version_hash"), field="version_hash"),
        element_hashes=tuple(_element_from_payload(item) for item in _list_of_mappings(payload, "element_hashes")),
    )


class SchemaReceiptRegistry(RuntimeSchemaRegistryLike, Protocol):
    """Committed-registry operations required to hash persisted files."""

    def read_committed_file(self, provider: str, relative_path: str) -> bytes | None: ...


def _package_hashes_for_package(
    registry: SchemaReceiptRegistry, provider: str, package: SchemaVersionPackage
) -> SchemaPackageContentHash:
    provider_token = str(canonical_schema_provider(provider))
    version_root = f"versions/{package.version}"

    def committed_hash(relative_path: str) -> str:
        content = registry.read_committed_file(provider_token, relative_path)
        if content is None:
            raise ValueError(f"committed schema artifact is missing: {provider_token}/{relative_path}")
        return hashlib.sha256(content).hexdigest()

    element_hashes: list[SchemaElementContentHash] = []
    package_hash = committed_hash(f"{version_root}/package.json")
    version_files: list[dict[str, str]] = [{"path": "package.json", "hash": package_hash}]
    for element in sorted(package.elements, key=lambda item: item.element_kind):
        if element.schema_file is None:
            continue
        content_hash = committed_hash(f"{version_root}/elements/{element.schema_file}")
        element_hashes.append(SchemaElementContentHash(element.element_kind, content_hash))
        version_files.append({"path": f"elements/{element.schema_file}", "hash": content_hash})
    if package.workload_profile_file is not None:
        content_hash = committed_hash(f"{version_root}/{package.workload_profile_file}")
        version_files.append({"path": package.workload_profile_file, "hash": content_hash})
    return SchemaPackageContentHash(
        provider=provider_token,
        package_version=package.version,
        package_hash=package_hash,
        version_hash=hash_payload(version_files),
        element_hashes=tuple(element_hashes),
    )


def package_hashes_for_registry(
    registry: SchemaReceiptRegistry, providers: Sequence[str] | None = None
) -> tuple[SchemaPackageContentHash, ...]:
    provider_names = tuple(providers) if providers is not None else tuple(registry.list_providers())
    records: list[SchemaPackageContentHash] = []
    for provider in sorted(set(provider_names)):
        catalog = registry.load_package_catalog(provider)
        if not isinstance(catalog, SchemaPackageCatalog):
            raise ValueError(f"registry provider {provider!r} has no persisted package catalog")
        records.extend(_package_hashes_for_package(registry, provider, package) for package in catalog.packages)
    return tuple(sorted(records))


def _unsupported_for_package(
    registry: SchemaReceiptRegistry, provider: str, package: SchemaVersionPackage
) -> tuple[SchemaInferenceUnsupportedDecision, ...]:
    decisions: list[SchemaInferenceUnsupportedDecision] = []
    for element in package.elements:
        if not element.supported or element.schema_file is None:
            decisions.append(
                SchemaInferenceUnsupportedDecision(
                    provider,
                    package.version,
                    element.element_kind,
                    "unsupported",
                    "unsupported_element" if not element.supported else "missing_schema",
                )
            )
            continue
        if provider not in PROVIDER_WIRE_FORMATS:
            decisions.append(
                SchemaInferenceUnsupportedDecision(
                    provider,
                    package.version,
                    element.element_kind,
                    "unsupported",
                    "provider_without_wire_format",
                )
            )
            continue
        schema = registry.get_element_schema(provider, version=package.version, element_kind=element.element_kind)
        if not isinstance(schema, Mapping):
            decisions.append(
                SchemaInferenceUnsupportedDecision(
                    provider, package.version, element.element_kind, "unsupported", "missing_schema"
                )
            )
            continue
        unsupported = tuple(
            item.construct for item in classify_schema_constructs(schema) if item.state == "unsupported"
        )
        if unsupported:
            decisions.append(
                SchemaInferenceUnsupportedDecision(
                    provider,
                    package.version,
                    element.element_kind,
                    "nonrepresentable",
                    "unsupported_json_schema_construct",
                    unsupported,
                )
            )
    return tuple(sorted(decisions))


def build_schema_inference_receipt(
    registry: SchemaReceiptRegistry, *, provider: str, input_manifest_digest: str | None = None
) -> SchemaInferenceReceipt:
    provider_token = str(canonical_schema_provider(provider))
    catalog = registry.load_package_catalog(provider_token)
    if not isinstance(catalog, SchemaPackageCatalog) or not catalog.packages:
        raise ValueError(f"schema commit produced no persisted packages for {provider_token}")
    packages = tuple(
        sorted(_package_hashes_for_package(registry, provider_token, package) for package in catalog.packages)
    )
    unsupported = tuple(
        sorted(
            item for package in catalog.packages for item in _unsupported_for_package(registry, provider_token, package)
        )
    )
    unsupported_keys = {(item.package_version, item.element_kind) for item in unsupported}
    representable = {
        (package.version, element.element_kind)
        for package in catalog.packages
        for element in package.elements
        if (package.version, element.element_kind) not in unsupported_keys
    }
    if representable:
        coverage_decision: CoverageDecision = "committed"
        coverage_reason = "persisted package/version/element hashes recorded"
    else:
        coverage_decision = (
            "nonrepresentable"
            if unsupported and all(item.decision == "nonrepresentable" for item in unsupported)
            else "unsupported"
        )
        coverage_reason = "provider has no executable persisted schema element"
    origin = origin_from_provider(provider_token).value
    coverage = SchemaInferenceCoverageDecision(
        origin=origin,
        provider=provider_token,
        decision=coverage_decision,
        reason=coverage_reason,
    )
    manifest = SchemaInferenceInputManifest(
        provider_token,
        input_manifest_digest,
        None if input_manifest_digest is not None else "input_manifest_unavailable",
    )
    return SchemaInferenceReceipt((manifest,), (coverage,), packages, unsupported)


def load_schema_inference_receipt(path: Path) -> SchemaInferenceReceipt:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"unable to read schema inference handoff {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("schema inference handoff root must be an object")
    return SchemaInferenceReceipt.from_payload(cast(Mapping[str, object], payload))


def write_schema_inference_receipt(
    receipt: SchemaInferenceReceipt, path: Path, *, merge: bool = False
) -> SchemaInferenceReceipt:
    """Publish a receipt, optionally merging provider updates under the publication lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with provider_tree_lock(path.parent, exclusive=True):
        if merge and path.exists():
            receipt = load_schema_inference_receipt(path).merged_with(receipt)
        atomic_replace(path, (json.dumps(receipt.to_payload(), indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return receipt


__all__ = [
    "SCHEMA_INFERENCE_HANDOFF_FILENAME",
    "SCHEMA_INFERENCE_HANDOFF_SCHEMA",
    "SchemaElementContentHash",
    "SchemaInferenceCoverageDecision",
    "SchemaInferenceInputManifest",
    "SchemaInferenceReceipt",
    "SchemaInferenceUnsupportedDecision",
    "SchemaPackageContentHash",
    "SchemaReceiptRegistry",
    "build_schema_inference_receipt",
    "load_schema_inference_receipt",
    "package_hashes_for_registry",
    "write_schema_inference_receipt",
]
