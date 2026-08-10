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
from typing import Literal, TypeAlias, cast, get_args

from polylogue.core.json import JSONDocument
from polylogue.core.sources import origin_from_provider
from polylogue.maintenance.schema_inference_gate import validate_schema_inference_gate_receipt
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
from polylogue.schemas.synthetic.wire_formats import (
    CATALOG_ELEMENT_UNSUPPORTED_REASON,
    PROVIDER_WIRE_FORMATS,
    ConstructCoverage,
    WireFormat,
    WireParserWitness,
    WireSupportEntry,
    WireSupportReceipt,
    build_wire_support_receipt,
    validate_wire_support_entry_keys,
    wire_support_entry_key,
    wire_support_key,
)

INFERRED_CORPUS_MANIFEST_SCHEMA_VERSION = 3
UnsupportedCorpusReason: TypeAlias = Literal[
    "provider_without_wire_format",
    "wire_support_selection_unwitnessed",
    "wire_support_receipt_incomplete",
    "unsupported_wire_route",
    "unsupported_element",
    "missing_schema",
    "unsupported_json_schema_construct",
]
PackageReceipt: TypeAlias = JSONDocument
_WIRE_AUTHORITY_ONLY_REASONS = frozenset(
    {
        "wire_support_selection_unwitnessed",
        "wire_support_receipt_incomplete",
        "unsupported_wire_route",
    }
)


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
    wire_support_receipt: JSONDocument | None = None

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
            "wire_support_receipt": self.wire_support_receipt,
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
            "wire_support_receipt",
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
        wire_support_receipt = payload.get("wire_support_receipt")
        if receipt_state not in {"catalog_only", "package_receipt_attached"}:
            raise ValueError(f"invalid inferred corpus manifest receipt_state: {receipt_state!r}")
        if receipt_state == "catalog_only" and package_receipt is not None:
            raise ValueError("catalog_only manifest must not carry a package receipt")
        if receipt_state == "package_receipt_attached" and not isinstance(package_receipt, dict):
            raise ValueError("package_receipt_attached manifest requires a JSON object receipt")
        if wire_support_receipt is not None and not isinstance(wire_support_receipt, dict):
            raise ValueError("wire_support_receipt must be a JSON object when present")
        manifest = cls(
            entries=entries,
            package_receipt=package_receipt if isinstance(package_receipt, dict) else None,
            wire_support_receipt=wire_support_receipt if isinstance(wire_support_receipt, dict) else None,
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
        _wire_support_entry_index(manifest)
        return manifest


def _require_inference_handoff(manifest: InferredCorpusManifest) -> SchemaInferenceReceipt:
    if manifest.receipt_state == "catalog_only" or manifest.package_receipt is None:
        raise ValueError("campaign mode requires a persisted schema-inference handoff, not catalog-only data")
    return SchemaInferenceReceipt.from_payload(manifest.package_receipt)


def _validate_authoritative_gate_binding(
    receipt: SchemaInferenceReceipt,
    *,
    gate_receipt_path: Path | None,
    archive_root: Path | None,
) -> None:
    if gate_receipt_path is None or archive_root is None:
        raise ValueError("campaign mode requires an authoritative gate receipt path and archive root")
    try:
        payload = json.loads(gate_receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"unable to read authoritative schema-inference gate receipt: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("authoritative schema-inference gate receipt must be a JSON object")
    gate_digest = validate_schema_inference_gate_receipt(payload, archive_root=archive_root)
    if receipt.gate_receipt_digest != gate_digest:
        raise ValueError("schema-inference handoff gate receipt digest does not match the authoritative PASS receipt")


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
    valid_reasons = set(get_args(UnsupportedCorpusReason))
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
    path: Path,
    *,
    campaign_mode: bool = False,
    registry: RuntimeSchemaRegistryLike | None = None,
    gate_receipt_path: Path | None = None,
    archive_root: Path | None = None,
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
        _validate_inference_handoff(
            manifest,
            registry,
            providers=providers,
            gate_receipt_path=gate_receipt_path,
            archive_root=archive_root,
        )
    return manifest


def _schema_constructs(schema: object) -> tuple[ConstructSupport, ...]:
    """Use the production classifier for campaign admission decisions."""

    return classify_schema_constructs(schema)


def build_inferred_corpus_convergence_handoff(
    manifest: InferredCorpusManifest | Path,
    *,
    campaign_mode: bool = False,
    registry: RuntimeSchemaRegistryLike | None = None,
    gate_receipt_path: Path | None = None,
    archive_root: Path | None = None,
) -> InferredCorpusConvergenceHandoff:
    """Bind every supported row from memory or persisted disk to convergence."""

    read_validated_campaign_manifest = isinstance(manifest, Path) and campaign_mode
    persisted_manifest = (
        read_inferred_corpus_manifest(
            manifest,
            campaign_mode=campaign_mode,
            registry=registry,
            gate_receipt_path=gate_receipt_path,
            archive_root=archive_root,
        )
        if isinstance(manifest, Path)
        else manifest
    )
    if campaign_mode and not read_validated_campaign_manifest:
        _require_inference_handoff(persisted_manifest)
        if registry is None:
            raise ValueError("campaign mode requires a live schema registry")
        providers = tuple(sorted({entry.key.provider for entry in persisted_manifest.entries}))
        _validate_inference_handoff(
            persisted_manifest,
            registry,
            providers=providers,
            gate_receipt_path=gate_receipt_path,
            archive_root=archive_root,
        )
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


def _wire_support_entry_index(
    manifest: InferredCorpusManifest,
) -> dict[tuple[str, str | None, str | None], WireSupportEntry]:
    """Recover the exact wire decision bound into a persisted manifest.

    The manifest stores the canonical receipt payload so a campaign read can
    replay the same unsupported-route decision without consulting a fresh
    support probe.  Treat malformed authority as a hard validation failure;
    falling back to an unbound decision would make the persisted claim weaker.
    """

    receipt = manifest.wire_support_receipt
    if receipt is None:
        return {}
    raw_entries = receipt.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("wire_support_receipt entries must be a list")
    entries: list[WireSupportEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("wire_support_receipt entries must be objects")
        entries.append(_wire_support_entry_from_payload(raw_entry))
    validate_wire_support_entry_keys(entries, boundary="persisted wire support receipt")
    return {wire_support_entry_key(entry): entry for entry in entries}


def _wire_support_entry_from_payload(payload: Mapping[str, object]) -> WireSupportEntry:
    def required_string(field: str) -> str:
        value = payload.get(field)
        if not isinstance(value, str):
            raise ValueError(f"wire_support_receipt entry {field} must be a string")
        return value

    def optional_string(field: str) -> str | None:
        value = payload.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"wire_support_receipt entry {field} must be a string or null")
        return value

    def required_int(value: object, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"wire_support_receipt entry {field} must be an integer")
        return value

    def string_tuple(value: object, field: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"wire_support_receipt entry {field} must be a list of strings")
        return tuple(value)

    status = payload.get("status")
    if status not in {"supported", "unsupported"}:
        raise ValueError("wire_support_receipt entry status is invalid")
    schema_valid = payload.get("schema_valid")
    if schema_valid is not None and not isinstance(schema_valid, bool):
        raise ValueError("wire_support_receipt entry schema_valid must be boolean or null")
    raw_coverage = payload.get("construct_coverage")
    coverage: ConstructCoverage | None = None
    if raw_coverage is not None:
        if not isinstance(raw_coverage, Mapping):
            raise ValueError("wire_support_receipt construct_coverage must be an object or null")
        raw_reasons = raw_coverage.get("nonrepresentable_reasons")
        if not isinstance(raw_reasons, list) or not all(isinstance(item, Mapping) for item in raw_reasons):
            raise ValueError("wire_support_receipt nonrepresentable_reasons must be a list of objects")
        reasons: list[tuple[str, str]] = []
        for item in raw_reasons:
            keyword = item.get("keyword")
            reason = item.get("reason")
            if not isinstance(keyword, str) or not isinstance(reason, str):
                raise ValueError("wire_support_receipt nonrepresentable reasons require strings")
            reasons.append((keyword, reason))
        coverage = ConstructCoverage(
            schema_keywords=string_tuple(raw_coverage.get("schema_keywords"), "schema_keywords"),
            exercised_keywords=string_tuple(raw_coverage.get("exercised_keywords"), "exercised_keywords"),
            missing_keywords=string_tuple(raw_coverage.get("missing_keywords"), "missing_keywords"),
            nonrepresentable_keywords=string_tuple(
                raw_coverage.get("nonrepresentable_keywords"), "nonrepresentable_keywords"
            ),
            nonrepresentable_reasons=tuple(reasons),
        )
    raw_witnesses = payload.get("parser_witnesses")
    if not isinstance(raw_witnesses, list) or not all(isinstance(item, Mapping) for item in raw_witnesses):
        raise ValueError("wire_support_receipt parser_witnesses must be a list of objects")
    witnesses: list[WireParserWitness] = []
    for raw_witness in raw_witnesses:
        artifact_kind = raw_witness.get("artifact_kind")
        if artifact_kind not in {"baseline", "coverage"}:
            raise ValueError("wire_support_receipt parser witness artifact_kind is invalid")
        validation_error = raw_witness.get("validation_error")
        if validation_error is not None and not isinstance(validation_error, str):
            raise ValueError("wire_support_receipt parser witness validation_error must be string or null")
        witnesses.append(
            WireParserWitness(
                index=required_int(raw_witness.get("index"), "parser_witness.index"),
                exercised_keywords=string_tuple(
                    raw_witness.get("exercised_keywords"), "parser_witness.exercised_keywords"
                ),
                parsed_session_count=required_int(
                    raw_witness.get("parsed_session_count"), "parser_witness.parsed_session_count"
                ),
                parsed_message_count=required_int(
                    raw_witness.get("parsed_message_count"), "parser_witness.parsed_message_count"
                ),
                validation_error=validation_error,
                artifact_kind=cast(Literal["baseline", "coverage"], artifact_kind),
                artifact_evidence=string_tuple(
                    raw_witness.get("artifact_evidence"), "parser_witness.artifact_evidence"
                ),
            )
        )
    return WireSupportEntry(
        provider=required_string("provider"),
        status=status,
        reason=optional_string("reason"),
        package_version=optional_string("package_version"),
        element_kind=optional_string("element_kind"),
        schema_valid=schema_valid,
        parsed_session_count=required_int(payload.get("parsed_session_count"), "parsed_session_count"),
        parsed_message_count=required_int(payload.get("parsed_message_count"), "parsed_message_count"),
        construct_coverage=coverage,
        validation_error=optional_string("validation_error"),
        parser_witnesses=tuple(witnesses),
    )


def _schema_unsupported_reason(
    *,
    element: SchemaElementManifest,
    schema: SchemaRecord | None,
    wire_format: WireFormat | None,
    construct_support: tuple[ConstructSupport, ...],
) -> UnsupportedCorpusRecord | None:
    if not element.supported:
        return UnsupportedCorpusRecord("unsupported_element")
    if schema is None or element.schema_file is None:
        return UnsupportedCorpusRecord("missing_schema")
    if wire_format is None:
        return UnsupportedCorpusRecord("provider_without_wire_format")
    unsupported_constructs = tuple(item.construct for item in construct_support if item.state == "unsupported")
    if unsupported_constructs:
        return UnsupportedCorpusRecord("unsupported_json_schema_construct", unsupported_constructs)
    return None


def _unsupported_reason(
    *,
    element: SchemaElementManifest,
    schema: SchemaRecord | None,
    wire_format: WireFormat | None,
    construct_support: tuple[ConstructSupport, ...],
    support_entry: WireSupportEntry | None,
    support_receipt_bound: bool,
) -> UnsupportedCorpusRecord | None:
    if support_entry is not None:
        if support_entry.status == "unsupported":
            reason: UnsupportedCorpusReason = (
                "unsupported_element"
                if support_entry.reason == CATALOG_ELEMENT_UNSUPPORTED_REASON
                else "unsupported_wire_route"
            )
            return UnsupportedCorpusRecord(
                reason,
                (support_entry.reason or "route is explicitly unsupported",),
            )
        if not support_entry.healthy:
            details = tuple(
                detail
                for detail in (
                    support_entry.validation_error,
                    *(witness.validation_error for witness in support_entry.parser_witnesses),
                )
                if detail
            )
            return UnsupportedCorpusRecord("wire_support_receipt_incomplete", details)
    elif support_receipt_bound:
        return UnsupportedCorpusRecord(
            "wire_support_selection_unwitnessed",
            (f"no exact parser witness for {element.schema_file!r}",),
        )
    return _schema_unsupported_reason(
        element=element,
        schema=schema,
        wire_format=wire_format,
        construct_support=construct_support,
    )


def _compile_entry(
    *,
    provider: str,
    package: SchemaVersionPackage,
    element: SchemaElementManifest,
    registry: RuntimeSchemaRegistryLike,
    wire_formats: Mapping[str, WireFormat],
    support_entry: WireSupportEntry | None,
    support_receipt_bound: bool,
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
        support_entry=support_entry,
        support_receipt_bound=support_receipt_bound,
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
    wire_support_receipt: WireSupportReceipt | None = None,
    providers: Sequence[str] | None = None,
    campaign_mode: bool = False,
    gate_receipt_path: Path | None = None,
    archive_root: Path | None = None,
) -> InferredCorpusManifest:
    """Compile every persisted package/version/element into a typed manifest."""

    formats = PROVIDER_WIRE_FORMATS if wire_formats is None else wire_formats
    support_entries: dict[tuple[str, str | None, str | None], WireSupportEntry] = {}
    if wire_support_receipt is not None:
        validate_wire_support_entry_keys(wire_support_receipt.entries, boundary="manifest wire support receipt")
        support_entries = {wire_support_entry_key(entry): entry for entry in wire_support_receipt.entries}
    if campaign_mode and package_receipt is None:
        raise ValueError("campaign mode requires a persisted schema-inference handoff")
    entries = tuple(
        _compile_entry(
            provider=provider,
            package=package,
            element=element,
            registry=registry,
            wire_formats=formats,
            support_entry=support_entries.get(wire_support_key(provider, package.version, element.element_kind)),
            support_receipt_bound=wire_support_receipt is not None,
        )
        for provider, catalog, package, element in _catalog_entries(registry, providers)
    )
    manifest = InferredCorpusManifest(
        entries=tuple(sorted(entries, key=lambda entry: entry.key)),
        package_receipt=package_receipt,
        wire_support_receipt=cast(JSONDocument, wire_support_receipt.to_dict())
        if wire_support_receipt is not None
        else None,
    )
    assert_inferred_corpus_manifest_complete(manifest, registry, providers=providers)
    if campaign_mode:
        _validate_inference_handoff(
            manifest,
            registry,
            providers=providers,
            gate_receipt_path=gate_receipt_path,
            archive_root=archive_root,
        )
    return manifest


def _validate_inference_handoff(
    manifest: InferredCorpusManifest,
    registry: RuntimeSchemaRegistryLike,
    *,
    providers: Sequence[str] | None,
    gate_receipt_path: Path | None,
    archive_root: Path | None,
) -> None:
    receipt = _require_inference_handoff(manifest)
    _validate_authoritative_gate_binding(
        receipt,
        gate_receipt_path=gate_receipt_path,
        archive_root=archive_root,
    )
    _validate_current_wire_support_route(manifest, registry)
    if not manifest.supported_specs:
        raise ValueError("campaign mode has no executable synthetic corpus selection")
    expected_packages = package_hashes_for_registry(cast(SchemaReceiptRegistry, registry), providers)
    if receipt.packages != expected_packages:
        raise ValueError("schema-inference handoff package/version/element hashes do not match the registry")

    catalog_entries = _catalog_entries(registry, providers)
    support_entries = _wire_support_entry_index(manifest)
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
        # The package receipt records schema inference authority.  Wire-route
        # refusals are independently bound by the serialized WireSupportReceipt
        # and must not rewrite a committed schema package decision.
        schema_blocking_reasons = tuple(
            entry.unsupported.reason
            for entry in provider_entries
            if entry.unsupported is not None and entry.unsupported.reason not in _WIRE_AUTHORITY_ONLY_REASONS
        )
        if any(entry.spec is not None for entry in provider_entries):
            expected_decision = "committed"
        elif not schema_blocking_reasons:
            expected_decision = "committed" if coverage.provider in PROVIDER_WIRE_FORMATS else "unsupported"
        elif all(reason == "unsupported_json_schema_construct" for reason in schema_blocking_reasons):
            expected_decision = "nonrepresentable"
        else:
            expected_decision = "unsupported"
        if coverage.decision != expected_decision:
            raise ValueError("schema-inference handoff coverage decision changed")

    expected_unsupported: set[tuple[str, str, str, str, str, tuple[str, ...]]] = set()
    entries_by_wire_key = {
        wire_support_key(entry.key.provider, entry.key.package_version, entry.key.element_kind): entry
        for entry in manifest.entries
    }
    for provider, _catalog, package, element in catalog_entries:
        live_entry = entries_by_wire_key.get(wire_support_key(provider, package.version, element.element_kind))
        if live_entry is None:
            raise ValueError("schema-inference manifest is missing a live registry entry")
        live_schema = registry.get_element_schema(provider, version=package.version, element_kind=element.element_kind)
        live_constructs = _schema_constructs(live_schema)
        schema_unsupported = _schema_unsupported_reason(
            element=element,
            schema=live_schema if isinstance(live_schema, dict) else None,
            wire_format=PROVIDER_WIRE_FORMATS.get(provider),
            construct_support=live_constructs,
        )
        if schema_unsupported is not None:
            expected_unsupported.add(
                (
                    provider,
                    package.version,
                    element.element_kind,
                    "nonrepresentable"
                    if schema_unsupported.reason == "unsupported_json_schema_construct"
                    else "unsupported",
                    schema_unsupported.reason,
                    schema_unsupported.details,
                )
            )
        if live_entry.key.construct_support != live_constructs:
            raise ValueError("schema-inference manifest classifier output changed")
        live_unsupported = _unsupported_reason(
            element=element,
            schema=live_schema if isinstance(live_schema, dict) else None,
            wire_format=PROVIDER_WIRE_FORMATS.get(provider),
            construct_support=live_constructs,
            support_entry=support_entries.get(wire_support_key(provider, package.version, element.element_kind)),
            support_receipt_bound=manifest.wire_support_receipt is not None,
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


def _validate_current_wire_support_route(
    manifest: InferredCorpusManifest,
    registry: RuntimeSchemaRegistryLike,
) -> None:
    """Re-run the exact persisted wire witnesses through current production code."""

    persisted = manifest.wire_support_receipt
    if persisted is None:
        return
    witness_seed = persisted.get("witness_seed")
    if isinstance(witness_seed, bool) or not isinstance(witness_seed, int):
        raise ValueError("wire_support_receipt witness_seed must be an integer")
    raw_providers = persisted.get("catalog_providers")
    if not isinstance(raw_providers, list) or not all(isinstance(provider, str) for provider in raw_providers):
        raise ValueError("wire_support_receipt catalog_providers must be a list of strings")
    current = build_wire_support_receipt(
        registry=registry,
        seed=witness_seed,
        providers=tuple(cast(str, provider) for provider in raw_providers),
    )
    rebuilt = current.to_dict()
    if rebuilt != persisted:
        changed_fields = sorted(key for key in set(rebuilt) | set(persisted) if rebuilt.get(key) != persisted.get(key))
        raise ValueError(
            "schema-inference wire-support receipt changed under the current parser or wire-normalizer route: "
            f"changed_fields={changed_fields!r}"
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
