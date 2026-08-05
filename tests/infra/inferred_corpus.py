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
import math
import re
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
from polylogue.schemas.synthetic.models import SchemaRecord, SyntheticSchemaSelection
from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS, WireFormat

ConstructSupportState: TypeAlias = Literal["supported", "unsupported"]
INFERRED_CORPUS_MANIFEST_SCHEMA_VERSION = 1
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
_SUPPORTED_FORMAT_VALUES = frozenset(
    {"uuid4", "uuid", "hex-id", "iso8601", "unix-epoch", "unix-epoch-str", "url", "email", "mime-type", "base64"}
)
_SUPPORTED_SEMANTIC_ROLE_VALUES = frozenset({"message_role", "message_body", "message_timestamp", "session_title"})

_PERSISTED_SCHEMA_METADATA_ANNOTATIONS = frozenset(
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
        "x-polylogue-high-cardinality-keys",
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


def read_inferred_corpus_manifest(path: Path, *, campaign_mode: bool = False) -> InferredCorpusManifest:
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
    return manifest


def _is_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _number(value: object) -> int | float | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if _is_number(value) else None
    if isinstance(value, float) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def _valid_pair(value: object, *, integral: bool = False, nonnegative: bool = False) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    first = _number(value[0])
    second = _number(value[1])
    if first is None or second is None:
        return False
    if integral and (not isinstance(first, int) or not isinstance(second, int)):
        return False
    if nonnegative and (first < 0 or second < 0):
        return False
    return first <= second


def _histogram_bucket_is_safe(index: int, log_base: int | float) -> bool:
    try:
        sampled_value = math.expm1((abs(float(index)) - 0.5) * math.log(float(log_base)))
    except (OverflowError, ValueError):
        return False
    return math.isfinite(sampled_value)


def _valid_observed_distribution(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    for distribution in value.values():
        if not isinstance(distribution, Mapping):
            return False
        histogram = distribution.get("histogram")
        log_base = _number(distribution.get("log_base"))
        if not isinstance(histogram, list) or not histogram or log_base is None or log_base <= 1:
            return False
        log_base_float = float(log_base)
        if not all(
            isinstance(bucket, list)
            and len(bucket) == 2
            and isinstance(bucket[0], int)
            and not isinstance(bucket[0], bool)
            and isinstance(bucket[1], int)
            and not isinstance(bucket[1], bool)
            and _number(bucket[0]) is not None
            and _number(bucket[1]) is not None
            and bucket[1] > 0
            and _histogram_bucket_is_safe(bucket[0], log_base_float)
            for bucket in histogram
        ):
            return False
        for key in ("min", "max", "mean", "p0", "p50", "p90", "p95", "p99", "p100", "stddev"):
            if key in distribution and not _is_number(distribution[key]):
                return False
        minimum = _number(distribution.get("min"))
        maximum = _number(distribution.get("max"))
        if minimum is not None and maximum is not None and minimum > maximum:
            return False
        stddev = _number(distribution.get("stddev"))
        if stddev is not None and stddev < 0:
            return False
        ordered_stats = tuple(_number(distribution.get(key)) for key in ("p0", "p50", "p90", "p95", "p99", "p100"))
        present_stats = tuple(value for value in ordered_stats if value is not None)
        if any(left > right for left, right in zip(present_stats, present_stats[1:], strict=False)):
            return False
        if any(
            (minimum is not None and value < minimum) or (maximum is not None and value > maximum)
            for value in present_stats
        ):
            return False
        mean = _number(distribution.get("mean"))
        if mean is not None:
            if minimum is not None and mean < minimum:
                return False
            if maximum is not None and mean > maximum:
                return False
            p0 = _number(distribution.get("p0"))
            p100 = _number(distribution.get("p100"))
            if p0 is not None and mean < p0:
                return False
            if p100 is not None and mean > p100:
                return False
    return True


def _relation_records(value: object, required: tuple[str, ...]) -> list[dict[str, object]] | None:
    if not isinstance(value, list) or not value:
        return None
    records: list[dict[str, object]] = []
    for record in value:
        if not isinstance(record, dict):
            return None
        if not all(isinstance(record.get(field), str) and record[field].strip() for field in required):
            return None
        records.append(record)
    return records


_SCHEMA_PATH_SEGMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def _schema_nodes_at_path(schema: SchemaRecord, path: object) -> tuple[SchemaRecord, ...]:
    """Resolve the JSONPath subset emitted by the relation solver."""

    if not isinstance(path, str) or not path.startswith("$"):
        return ()
    if path == "$":
        return (schema,)
    if not path.startswith("$."):
        return ()
    nodes: tuple[SchemaRecord, ...] = (schema,)
    for raw_segment in path[2:].split("."):
        if raw_segment.endswith("[*]"):
            segment = raw_segment[:-3]
            wants_items = True
        else:
            segment = raw_segment
            wants_items = False
        if not _SCHEMA_PATH_SEGMENT.fullmatch(segment):
            return ()
        next_nodes: list[SchemaRecord] = []
        for node in nodes:
            variants = [node]
            for branch_key in ("anyOf", "oneOf"):
                branches = node.get(branch_key)
                if isinstance(branches, list):
                    variants.extend(branch for branch in branches if isinstance(branch, dict))
            for variant in variants:
                properties = variant.get("properties")
                child = properties.get(segment) if isinstance(properties, dict) else None
                if not isinstance(child, dict):
                    continue
                if wants_items:
                    child_variants = [child]
                    for child_branch_key in ("anyOf", "oneOf"):
                        child_branches = child.get(child_branch_key)
                        if isinstance(child_branches, list):
                            child_variants.extend(branch for branch in child_branches if isinstance(branch, dict))
                    for child_variant in child_variants:
                        items = child_variant.get("items")
                        if isinstance(items, dict):
                            next_nodes.append(items)
                else:
                    next_nodes.append(child)
        nodes = tuple(next_nodes)
        if not nodes:
            return ()
    return nodes


def _schema_nodes_have_type(nodes: tuple[SchemaRecord, ...], allowed: set[str]) -> bool:
    return bool(nodes) and all(bool(types := _schema_types(node)) and types <= allowed for node in nodes)


def _schema_type_family(nodes: tuple[SchemaRecord, ...]) -> str | None:
    types = {_schema_type(node) for node in nodes}
    if types and types <= {"string"}:
        return "string"
    if types and types <= {"number", "integer"}:
        return "numeric"
    return None


def _relation_annotation_paths_are_enforced(
    key: str,
    value: object,
    root_schema: SchemaRecord,
) -> bool:
    records = _relation_records(
        value,
        {
            "x-polylogue-foreign-keys": ("source", "target"),
            "x-polylogue-time-deltas": ("field_a", "field_b"),
            "x-polylogue-mutually-exclusive": ("parent",),
            "x-polylogue-string-lengths": ("path",),
        }[key],
    )
    if records is None:
        return False
    if key == "x-polylogue-foreign-keys":
        return False
    if key == "x-polylogue-time-deltas":
        # The solver parses these records, but no production generation path
        # calls get_time_delta yet. Keep them fail-closed until it does.
        return False
    if key == "x-polylogue-mutually-exclusive":
        for record in records:
            fields = record.get("fields")
            if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
                return False
            if not all(
                _schema_nodes_have_type(
                    _schema_nodes_at_path(root_schema, f"{record['parent']}.{field}"),
                    {"string", "number", "integer", "boolean", "array", "object", "null"},
                )
                for field in fields
            ):
                return False
        return True
    if key == "x-polylogue-string-lengths":
        return all(
            bool(
                _schema_nodes_at_path(root_schema, record["path"])
                and any("string" in _schema_types(node) for node in _schema_nodes_at_path(root_schema, record["path"]))
            )
            for record in records
        )
    return False


def _synthetic_annotation_is_enforced(key: str, value: object) -> bool:
    """Return whether the production generator or solver enforces this payload."""

    if key not in _SUPPORTED_SYNTHETIC_ANNOTATIONS:
        return False
    if key == "x-polylogue-format":
        return isinstance(value, str) and value in _SUPPORTED_FORMAT_VALUES
    if key == "x-polylogue-semantic-role":
        return isinstance(value, str) and value in _SUPPORTED_SEMANTIC_ROLE_VALUES
    if key == "x-polylogue-frequency":
        frequency = _number(value)
        return frequency is not None and 0 <= frequency <= 1
    if key == "x-polylogue-multiline":
        return isinstance(value, bool)
    if key == "x-polylogue-values":
        return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item for item in value)
    if key == "x-polylogue-range":
        return _valid_pair(value)
    if key == "x-polylogue-array-lengths":
        return _valid_pair(value, integral=True, nonnegative=True)
    if key == "x-polylogue-observed-distribution":
        return _valid_observed_distribution(value)
    if key == "x-polylogue-foreign-keys":
        return _relation_records(value, ("source", "target")) is not None
    if key == "x-polylogue-time-deltas":
        # The solver parses these records, but no production generation path
        # calls get_time_delta yet. Keep them fail-closed until it does.
        return False
    if key == "x-polylogue-mutually-exclusive":
        records = _relation_records(value, ("parent",))
        return records is not None and all(
            isinstance(fields := record.get("fields"), list)
            and len(fields) >= 2
            and all(isinstance(field, str) and field for field in fields)
            for record in records
        )
    if key == "x-polylogue-string-lengths":
        records = _relation_records(value, ("path",))
        if records is None:
            return False
        string_records: list[tuple[int | float | None, int | float | None, int | float | None, int | float | None]] = [
            (
                _number(record.get("min")),
                _number(record.get("max")),
                _number(record.get("avg")),
                _number(record.get("stddev")),
            )
            for record in records
        ]
        for minimum, maximum, average, stddev in string_records:
            if (
                not _valid_pair([minimum, maximum], integral=True, nonnegative=True)
                or average is None
                or stddev is None
                or minimum is None
                or maximum is None
                or not minimum <= average <= maximum
                or stddev < 0
            ):
                return False
        return True
    raise AssertionError(f"missing annotation validator for {key}")


def _schema_type(node: Mapping[str, object]) -> str | None:
    schema_type = node.get("type")
    if isinstance(schema_type, str):
        return schema_type
    if isinstance(schema_type, list):
        types = [item for item in schema_type if isinstance(item, str) and item != "null"]
        return types[0] if len(types) == 1 else None
    return None


def _schema_types(node: Mapping[str, object]) -> set[str]:
    schema_type = node.get("type")
    if isinstance(schema_type, str):
        return {schema_type}
    if isinstance(schema_type, list):
        return {item for item in schema_type if isinstance(item, str)}
    types: set[str] = set()
    for keyword in ("anyOf", "oneOf"):
        variants = node.get(keyword)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, Mapping):
                    types.update(_schema_types(variant))
    return types


def _annotation_is_enforced_at_node(
    key: str,
    value: object,
    node: Mapping[str, object],
    path: str,
    root_schema: SchemaRecord,
) -> bool:
    schema_type = _schema_type(node)
    schema_types = _schema_types(node)
    union_path = ".anyOf[" in path or ".oneOf[" in path
    if key in {
        "x-polylogue-foreign-keys",
        "x-polylogue-time-deltas",
        "x-polylogue-mutually-exclusive",
        "x-polylogue-string-lengths",
    }:
        return (
            path == "$"
            and _synthetic_annotation_is_enforced(key, value)
            and _relation_annotation_paths_are_enforced(key, value, root_schema)
        )
    if key == "x-polylogue-frequency":
        return path != "$" and _synthetic_annotation_is_enforced(key, value)
    if key in {"x-polylogue-array-lengths", "x-polylogue-observed-distribution"}:
        if key == "x-polylogue-array-lengths":
            return _synthetic_annotation_is_enforced(key, value) and (
                schema_type == "array" or "array" in schema_types or union_path
            )
        if schema_type not in {"array", "number", "integer"} or not isinstance(value, Mapping):
            return False
        expected_distribution = "array_length" if schema_type == "array" else "numeric"
        return expected_distribution in value and _valid_observed_distribution(value)
    if key == "x-polylogue-range":
        return schema_type in {"number", "integer"} and _synthetic_annotation_is_enforced(key, value)
    if key in {"x-polylogue-format", "x-polylogue-values", "x-polylogue-multiline"}:
        if key == "x-polylogue-format" and schema_type in {"number", "integer"}:
            return value == "unix-epoch"
        return _synthetic_annotation_is_enforced(key, value) and (
            schema_type == "string" or "string" in schema_types or union_path
        )
    if key == "x-polylogue-semantic-role":
        role = value if isinstance(value, str) else None
        if role == "message_timestamp":
            return bool(schema_types & {"string", "number", "integer"}) and _synthetic_annotation_is_enforced(
                key, value
            )
        if role == "message_container":
            return schema_type == "object"
        return schema_type == "string" and _synthetic_annotation_is_enforced(key, value)
    return key in _PERSISTED_SCHEMA_METADATA_ANNOTATIONS


def _schema_constructs(schema: object) -> tuple[ConstructSupport, ...]:
    """Census schema keywords and annotations at the paths production consumes."""

    found: dict[str, ConstructSupportState] = {}

    def record_support(key: str, state: ConstructSupportState) -> None:
        if found.get(key) == "unsupported":
            return
        found[key] = state

    def visit(node: object, path: str = "$") -> None:
        if not isinstance(node, Mapping):
            return
        node_record = node
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            if key.startswith("x-"):
                if key in _PERSISTED_SCHEMA_METADATA_ANNOTATIONS:
                    continue
                record_support(
                    key,
                    "supported"
                    if _annotation_is_enforced_at_node(key, value, node_record, path, cast(SchemaRecord, schema))
                    else "unsupported",
                )
                continue
            if key in _SUPPORTED_SCHEMA_CONSTRUCTS:
                record_support(key, "supported")
            elif key in _STANDARD_SCHEMA_KEYWORDS:
                record_support(key, "unsupported")
            else:
                record_support(key, "unsupported")

            if key in _SCHEMA_MAPPING_KEYWORDS and isinstance(value, Mapping):
                if key in {"$defs", "dependentSchemas", "dependencies", "patternProperties", "properties"}:
                    for child_name, child in value.items():
                        child_path = f"{path}.{child_name}" if key == "properties" else f"{path}.{key}.{child_name}"
                        visit(child, child_path)
                else:
                    visit(value, f"{path}.{key}")
            elif key in _SCHEMA_ARRAY_KEYWORDS and isinstance(value, Sequence) and not isinstance(value, str):
                for index, child in enumerate(value):
                    visit(child, f"{path}.{key}[{index}]")
            elif key == "items":
                if isinstance(value, Sequence) and not isinstance(value, str):
                    for index, child in enumerate(value):
                        visit(child, f"{path}.items[{index}]")
                else:
                    visit(value, f"{path}.items")
            elif key == "additionalProperties" and isinstance(value, Mapping):
                visit(value, f"{path}.additionalProperties")

    visit(schema)
    return tuple(ConstructSupport(construct, found[construct]) for construct in sorted(found))


def build_inferred_corpus_convergence_handoff(
    manifest: InferredCorpusManifest | Path,
    *,
    campaign_mode: bool = False,
) -> InferredCorpusConvergenceHandoff:
    """Bind every supported row from memory or persisted disk to convergence."""

    persisted_manifest = (
        read_inferred_corpus_manifest(manifest, campaign_mode=campaign_mode) if isinstance(manifest, Path) else manifest
    )
    if campaign_mode:
        _require_inference_handoff(persisted_manifest)
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
    expected_packages = package_hashes_for_registry(cast(SchemaReceiptRegistry, registry), providers)
    if receipt.packages != expected_packages:
        raise ValueError("schema-inference handoff package/version/element hashes do not match the registry")

    expected_coverage = {
        (provider, origin_from_provider(provider).value)
        for provider, _catalog, _package, _element in _catalog_entries(registry, providers)
    }
    actual_coverage = {(item.provider, item.origin) for item in receipt.coverage_decisions}
    if actual_coverage != expected_coverage:
        raise ValueError(
            "schema-inference handoff does not contain complete origin/provider coverage: "
            f"missing={sorted(expected_coverage - actual_coverage)!r}, "
            f"unexpected={sorted(actual_coverage - expected_coverage)!r}"
        )
    if any(item.decision != "committed" for item in receipt.coverage_decisions):
        raise ValueError("schema-inference handoff contains a non-committed coverage decision")

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
